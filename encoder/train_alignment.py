import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from dataset import ImageDataset
from image_encoder import load_image_encoder
from text_encoder import TextEncoder
from tokenizer.tokenizer import load_tokenizer
from loss import cosine_contrastive_loss
from utils.ddp import setup, cleanup
from utils.gpu_manager import get_gpu_temp, wait_for_cooldown
from utils.collate_fn import skip_broken_collate_fn
from utils.save_model import save_checkpoint, save_weights, load_checkpoint


def train_alignment(rank, world_size, args):
    setup()
    device = torch.device(f'cuda:{args.local_rank}')

    tokenizer = load_tokenizer(args.tokenizer_path)

    image_encoder = load_image_encoder(device)
    image_encoder.load_state_dict(torch.load(args.image_encoder_path))
    text_encoder  = TextEncoder(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = optim.AdamW(text_encoder.parameters(), lr=args.lr)

    start_epoch = 1
    best_loss = float('inf')
    if args.resume:
        start_epoch, best_loss = load_checkpoint(text_encoder, optimizer, Path(args.output_dir), 'text_encoder')

    dataset = ImageDataset(Path(args.data_dir))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=skip_broken_collate_fn
    )

    image_encoder = DDP(image_encoder, device_ids=[args.local_rank])
    text_encoder = DDP(text_encoder, device_ids=[args.local_rank])

    image_encoder.eval()
    for epoch in range(start_epoch, args.epochs + 1):
        text_encoder.train()
        total_loss = 0
        sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, desc=f"[GPU {rank}] Epoch {epoch}", disable=(rank != 0))

        for batch in dataloader:
            if batch is None:
                progress_bar.update(1)
                continue
            if get_gpu_temp(rank) >= 80:
                wait_for_cooldown(rank)

            images = batch["image"].to(device)
            raw_texts = [t["raw_text"] for t in batch["text"]]

            tokenized_raw = tokenizer(
                raw_texts,
                padding=True,
                truncation=True,
                max_length=128, 
                return_tensors='pt'
            )
            input_ids_raw = tokenized_raw.input_ids.to(device)
            attention_mask_raw = tokenized_raw.attention_mask.to(device)

            with torch.no_grad():
                image_embeds = image_encoder.module.get_image_features(pixel_values=images)
            raw_text_embeds = text_encoder(input_ids=input_ids_raw, attention_mask=attention_mask_raw)

            loss = cosine_contrastive_loss(raw_text_embeds, image_embeds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")
            save_weights(text_encoder.module, avg_loss, f'text_encoder_align', Path(args.output_dir))
            if epoch % 10 == 0:
                save_checkpoint(epoch, text_encoder.module, image_encoder.module, optimizer, avg_loss, Path(args.output_dir))

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/mnt/usb/images')
    parser.add_argument("--tokenizer_path", type=str, default='tokenizer/tokenizer.json')
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--local_rank", type=int, default=os.environ.get("LOCAL_RANK", 0))
    args = parser.parse_args()

    rank = args.local_rank
    world_size = torch.cuda.device_count()
    train_alignment(rank, world_size, args)


if __name__ == '__main__':
    main()
