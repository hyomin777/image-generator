import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
from pathlib import Path
from functools import partial

import torch
import torch.distributed as dist
from torch.nn import TransformerEncoderLayer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig, FullOptimStateDictConfig, StateDictType
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset import LMDBImageDataset
from image_generator import ImageGenerator
from utils.collate_fn import skip_broken_collate_fn
from transform import normalize

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def fsdp_main(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

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

    model = ImageGenerator(device, args.text_encoder_path, args.tokenizer_path)
    model.vae.to(device)
    model.text_encoder.to(device)
    model.unet = FSDP(model.unet.to(device), auto_wrap_policy=auto_wrap_policy, mixed_precision=mp_policy)

    # Dataloader
    dataset = LMDBImageDataset(args.data_dir)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=skip_broken_collate_fn)

    # Optimizer & Scaler & Scheduler
    optimizer = torch.optim.AdamW(model.unet.parameters(), lr=args.lr)
    scaler = torch.GradScaler(device.type)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # TensorBoard
    writer = SummaryWriter(log_dir=Path(args.output_dir) / 'logs') if rank == 0 else None

    global_step = 1
    best_loss = 1
    start_epoch = 1

    if args.resume and os.path.exists(args.resume):
        map_location = {"cuda:0": f"cuda:{rank}"}
        checkpoint = torch.load(args.resume, map_location=map_location)

        with FSDP.state_dict_type(
            model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            prev_unet_state = checkpoint["unet"]
            prev_optim_state = checkpoint["optimizer"]

            model.unet.load_state_dict(prev_unet_state)
            optimizer.load_state_dict(prev_optim_state)

        global_step = checkpoint["global_step"]
        best_loss = checkpoint["best_loss"]
        start_epoch = checkpoint["epoch"] + 1

        if rank == 0:
            print(f"[INFO] Resumed from {args.resume} | epoch: {start_epoch} | step: {global_step} | loss: {best_loss}")


    print(f"Rank={rank}, UNet param count: {sum(p.numel() for p in model.unet.parameters())}")
    print(f"Rank={rank}, TextEncoder param count: {sum(p.numel() for p in model.text_encoder.parameters())}")
    print(f"Rank={rank}, VAE param count: {sum(p.numel() for p in model.vae.parameters())}")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        sampler.set_epoch(epoch)

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=rank != 0)

        for idx, batch in enumerate(dataloader):
            if batch is None:
                continue

            images = batch["image"].to(device, non_blocking=True)
            images = normalize(images)
            raw_text = raw_text = [t['raw_text'] for t in batch["text"]]

            with torch.autocast(device.type):
                loss = model.train_step(images, raw_text)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step(epoch + idx / len(dataloader))

            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

            if global_step % 100 == 0 and writer is not None:
                writer.add_scalar("Loss/generator_step", loss.item(), global_step)

            global_step += 1

        avg_loss = total_loss / len(dataloader)
        with FSDP.state_dict_type(
            model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            unet_state = model.unet.state_dict()
            optimizer_state = optimizer.state_dict()

        if rank == 0:
            if writer is not None:
                writer.add_scalar("Loss/generator_epoch", avg_loss, epoch)

            if avg_loss < best_loss:
                best_loss = avg_loss
                weights_path = Path(args.output_dir) / 'weights'
                weights_path.mkdir(parents=True, exist_ok=True)
                torch.save(unet_state, os.path.join(weights_path, "unet.pth"))
                print(f'[Epoch {epoch}] Unet weights saved with loss {avg_loss:.4f}', flush=True)

            checkpoint = {
                "unet": unet_state,
                "optimizer": optimizer_state,
                "global_step": global_step,
                "best_loss": best_loss,
                "epoch": epoch,
            }
            checkpoint_path = Path(args.output_dir) / 'checkpoints'
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, os.path.join(checkpoint_path, f"generator_checkpoint_{epoch}.pt"))

            print(f"[Epoch {epoch}] LR: {scheduler.get_last_lr()[0]:.8f}")

        model.eval()
        with torch.no_grad():
            sample_text = ["1girl black_shirt blue_archive shiroko_(blue_archive) animal ears wolf girl sweater"]
            gen = model(sample_text)

            if rank == 0:
                grid = make_grid(gen, nrow=1)
                writer.add_image(f"Generated/Epoch_{epoch}", grid, epoch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default='tokenizer/tokenizer.json')
    parser.add_argument("--text_encoder_path", type=str, default='output/weights/text_encoder.pth')
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=60)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=5.0)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    fsdp_main(local_rank, world_size, args)

