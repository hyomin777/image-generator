import os

import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import CLIPModel

from tqdm import tqdm
from dataset import ImageDataset
from tag_encoder import TagEncoder
from loss import cosine_contrastive_loss
from utils.ddp import setup, cleanup


def train_tag_encoder(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # image encoder
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip.eval()

    # tag encoder
    tag_encoder = TagEncoder().to(device)
    tag_encoder = DDP(tag_encoder, device_ids=[rank], find_unused_parameters=True)

    optimizer = optim.AdamW(tag_encoder.parameters(), lr=args.lr)

    # dataset and DDP-compatible loader
    dataset = ImageDataset(
        Path(args.data_dir)
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # train loop
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, desc=f"[GPU {rank}] Epoch {epoch}", disable=(rank != 0))

        for batch in dataloader:
            images = batch["image"].to(device)
            tags = batch["text"]

            with torch.no_grad():
                image_embeds = clip.get_image_features(pixel_values=images)

            tag_embeds = tag_encoder(tags)
            print(f"[rank {rank}] tag_embeds shape: {tag_embeds.shape}")
            loss = cosine_contrastive_loss(tag_embeds, image_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item()})

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = Path(args.output_dir) / 'tag_encoder.pt'
        torch.save(tag_encoder.module.state_dict(), save_path)

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="image dataset directory")
    parser.add_argument("--output_dir", type=str, default="output", help="model save directory")
    parser.add_argument("--epochs", type=int, default=10, help="epochs for training")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="data loader worker count")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(train_tag_encoder, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
