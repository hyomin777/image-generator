import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from tqdm import tqdm
from pathlib import Path
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import TransformerEncoderLayer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig, FullOptimStateDictConfig, StateDictType
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import Dataset, DataLoader
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
    data_dir:str = ""
    resume: str = ""


class ModelTrainer:
    def __init__(
            self,
            rank,
            world_size,
            model_cls: nn.Module,
            dataset_cls: Dataset,
            config:TrainConfig
        ):
        self.config = config

        self.rank = rank
        self.world_size = world_size
        self.setup()

        self.device = torch.device("cuda", rank)

        self.model_cls = model_cls
        self.model = self._create_model().to(self.device)

        self.dataset_cls = dataset_cls
        self.dataset = self._create_dataset()

        self.best_loss = float('inf')
        self.step = 1
        self.start_epoch = 1

        self.scaler = torch.GradScaler(self.device.type)
        self.writer = SummaryWriter(log_dir=Path(self.config.output_dir) / 'logs') if self.rank == 0 else None

    def setup(self):
        if not torch.distributed.is_initialized():
            dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.rank)

    def cleanup(self):
        dist.destroy_process_group()

    def _fsdp_wrap_model(self, model):
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=int(5e5),
            force_leaf_modules={TransformerEncoderLayer}
        )
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
        model = FSDP(model.to(self.device), auto_wrap_policy=auto_wrap_policy, mixed_precision=mp_policy)
        return model

    def _create_model(self):
        raise NotImplementedError

    def _create_dataset(self):
        raise NotImplementedError

    def _create_optimizer(self, parameters):
        self.optimizer = torch.optim.AdamW(parameters, lr=self.config.lr, weight_decay=self.config.weight_decay)

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

    def _save_weights(self, model, path):
        with FSDP.state_dict_type(
            model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()

        if self.rank == 0:
            weights_path = Path(self.config.output_dir) / 'weights'
            weights_path.mkdir(parents=True, exist_ok=True)
            torch.save(model_state, os.path.join(weights_path, path))

    def _save_checkpoint(self, model, optimizer, epoch, path):
        with FSDP.state_dict_type(
            model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()

        if self.rank == 0:
            checkpoint = {
                "model": model_state,
                "optimizer": optimizer_state,
                "step": self.step,
                "loss": self.best_loss,
                "epoch": epoch,
            }

            checkpoint_path = Path(self.config.output_dir) / 'checkpoints'
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, os.path.join(checkpoint_path, path))
            print(f"[INFO] Saved checkpoint to {os.path.join(checkpoint_path, path)}")

    def _load_checkpoint(self, model, optimizer):
        if not os.path.exists(self.config.resume):
            if self.rank == 0:
                print(f"[ERROR] Checkpoint file {self.config.resume} not found!")
            return False

        try:
            map_location = {"cuda:0": f"cuda:{self.rank}"}
            checkpoint = torch.load(self.config.resume, map_location=map_location)

            with FSDP.state_dict_type(
                model,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])

            self.start_epoch = checkpoint["epoch"]
            self.step = checkpoint["step"] + 1
            self.best_loss = checkpoint["loss"]

            if self.rank == 0:
                print(f"[INFO] Resumed from {self.config.resume} | epoch: {self.start_epoch} | step: {self.step} | loss: {self.best_loss}")
            return True
        except Exception as e:
            if self.rank == 0:
                print(f"[ERROR] Failed to load checkpoint: {e}")
            return False

    def train(self):
        raise NotImplementedError


class GeneratorTrainer(ModelTrainer):
    def _create_model(self):
        return self.model_cls(self.device)

    def _create_dataset(self):
        return self.dataset_cls(self.config.data_dir)

    def train(self):
        self.model.unet = self._fsdp_wrap_model(self.model.unet)
        self._create_optimizer(self.model.unet.parameters())

        dataloader, sampler = self._create_dataloader(self.dataset)
        scheduler = self._create_scheduler()

        if self.config.resume:
            file_ext = Path(self.config.resume).suffix
            if not file_ext:
                self.config.resume += ".pth"
            elif file_ext not in ['.pth', '.pt']:
                if self.rank == 0:
                    print(f"[WARNING] Unexpected checkpoint extension: {file_ext}. Expected .pth or .pt")

        resume_success = False
        if self.config.resume:
            resume_success = self._load_checkpoint(self.model.unet, self.optimizer)
            if not resume_success and self.rank == 0:
                print(f"[WARNING] Could not load checkpoint {self.config.resume}, starting from scratch")

        torch.autograd.set_detect_anomaly(True)
        if self.rank == 0:
            print(f"[INFO] Starting training from epoch {self.start_epoch} to {self.config.epochs}")

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self.model.train()
            total_loss = 0.0
            sampler.set_epoch(epoch)

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=self.rank != 0)
            for idx, batch in enumerate(dataloader):
                if batch is None:
                    if self.rank == 0:
                        print(f"[WARNING] Skipping empty batch at epoch {epoch}, batch {idx}")
                    continue

                try:
                    images = batch["image"].to(self.device, non_blocking=True)
                    raw_text = [t['raw_text'] for t in batch["text"]]

                    with torch.autocast(self.device.type):
                        loss = self.model.train_step(images, raw_text)

                    # Check for NaN loss
                    is_nan = torch.tensor(float(torch.isnan(loss)), device=self.device)
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(is_nan, op=torch.distributed.ReduceOp.MAX)

                    if is_nan > 0:
                        print(f"NaN detected in loss at epoch {epoch}, batch {idx}")
                        continue

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.unet.parameters(), self.config.clip_grad)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    scheduler.step(epoch + idx / len(dataloader))

                    total_loss += loss.item()
                    progress_bar.update(1)
                    progress_bar.set_postfix({"loss": loss.item()})

                    if self.step % 10000 == 0:
                        self._save_checkpoint(
                            self.model.unet,
                            self.optimizer,
                            epoch,
                            f'generator_{epoch}_{self.step}.pth'
                        )

                    self.step += 1

                except Exception as e:
                    if self.rank == 0:
                        print(f"[ERROR] Error in training loop at epoch {epoch}, batch {idx}: {e}")
                    continue

            avg_loss = total_loss / max(1, len(dataloader))

            self._save_checkpoint(
                self.model.unet, self.optimizer, epoch+1, f'generator_{epoch}.pth'
            )

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self._save_weights(self.model.unet, "unet.pth")
                print(f"[INFO] New best loss: {self.best_loss}, saved model weights")

            # Generate sample images
            self.model.eval()
            with torch.no_grad():
                try:
                    sample_text = ["1girl blue_archive shiroko_(blue_archive)"]
                    gen = self.model(sample_text)

                    if self.writer is not None:
                        grid = make_grid(gen, nrow=1)
                        self.writer.add_image(f"Generated/Epoch_{epoch}", grid, epoch)
                        print(f"[INFO] Generated sample image for epoch {epoch}")
                except Exception as e:
                    print(f"[ERROR] Failed to generate sample image: {e}")
