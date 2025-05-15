import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from tqdm import tqdm
from pathlib import Path
from functools import partial
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.nn import TransformerEncoderLayer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from utils.collate_fn import skip_broken_collate_fn

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class TrainConfig:
    num_workers: int = 8
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-2
    clip_grad: float = 1.0
    output_dir:str = "output"
    resume: str = ""


class AppState(Stateful):
    def __init__(self, model, optimizer=None, epoch=1, step=1, loss=float('inf')):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.step = step
        self.loss = loss

    def state_dict(self):
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]
        self.loss = state_dict["loss"]


class ModelTrainer:
    def __init__(
            self,
            rank,
            world_size,
            device,
            model,
            optimizer,
            config:TrainConfig
        ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        
        self.model = model.to(self.device)
        self.optimizer = optimizer
        
        self.config = config

        self.best_loss = float('inf')
        self.global_step = 1
        self.start_epoch = 1

        self.scaler = torch.GradScaler(self.device.type)
        self.writer = SummaryWriter(log_dir=Path(self.config.output_dir) / 'logs') if self.rank == 0 else None
    
    def setup(self):
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.rank)

    def cleanup(self):
        dist.destroy_process_group()

    def _fsdp_wrap_model(self, model):
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=int(1e6),
            force_leaf_modules={TransformerEncoderLayer}
        )
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
        model = FSDP(model.to(self.device), auto_wrap_policy=auto_wrap_policy, mixed_precision=mp_policy)
        return model

    def _create_scheduler(self, T_0=10, T_mult=2, eta_min=1e-6):
        return CosineAnnealingWarmRestarts(self.optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

    def _create_dataloader(self, dataset):
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
        )
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=skip_broken_collate_fn,
            pin_memory=True,
            drop_last=True
        )
        return dataloader, sampler

    def _save_checkpoint(self, model, optimizer, epoch, step, loss, path):
        checkpoint_path = Path(path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        state_dict = { "app": AppState(model, optimizer, epoch, step, loss) }
        dcp.save(state_dict, checkpoint_id=checkpoint_path)

    def _load_checkpoint(self, model, optimizer):
        app_state = AppState(model, optimizer)
        dcp.load(
            state_dict={"app": app_state},
            checkpoint_id=self.config.resume,
        )

        self.start_epoch = app_state.epoch + 1
        self.global_step = app_state.step
        self.best_loss = app_state.loss

        if self.rank == 0:
            print(f"[INFO] Resumed from {self.config.resume} | epoch: {self.start_epoch} | step: {self.global_step} | loss: {self.best_loss}")

    def train(self):
        raise NotImplementedError("Subclasses must implement method: train")


class GeneratorTrainer(ModelTrainer):
    def train(
            self,
            dataset
        ):
        if self.config.resume:
            self._load_checkpoint(self.model.unet, self.optimizer)

        self.model.unet = self._fsdp_wrap_model(self.model.unet)

        dataloader, sampler = self._create_dataloader(dataset)
        scheduler = self._create_scheduler()

        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self.model.train()
            total_loss = 0.0
            sampler.set_epoch(epoch)

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=self.rank != 0)
            for idx, batch in enumerate(dataloader):
                if batch is None:
                    continue

                images = batch["image"].to(self.device, non_blocking=True)
                raw_text = raw_text = [t['raw_text'] for t in batch["text"]]

                with torch.autocast(self.device.type):
                    loss = self.model.train_step(images, raw_text)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.unet.parameters(), self.config.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                scheduler.step(epoch + idx / len(dataloader))
                
                total_loss += loss.item()
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})

                if self.global_step % 100 == 0 and self.writer is not None:
                    self.writer.add_scalar("Loss/generator_step", loss.item(), self.global_step)
                self.global_step += 1

            avg_loss = total_loss / len(dataloader)
            self._save_checkpoint(
                self.model.unet, self.optimizer, epoch, self.global_step, self.best_loss, f'generator_{epoch}'
            )

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_loss = avg_loss
                self._save_checkpoint(
                    self.model.unet, self.optimizer, epoch, self.global_step, self.best_loss, "generator_best"
                )

            self.model.eval()
            with torch.no_grad():
                sample_text = ["1girl blue_archive shiroko_(blue_archive)"]
                gen = self.model(sample_text)

                if self.rank == 0 and self.writer is not None:
                    grid = make_grid(gen, nrow=1)
                    self.writer.add_image(f"Generated/Epoch_{epoch}", grid, epoch)
