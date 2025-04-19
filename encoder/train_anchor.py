import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

from dataset import RefinedImageDataset
from encoder.text_encoder import TextEncoder
from encoder.image_encoder import load_image_encoder
from loss import cosine_contrastive_loss
from tokenizer.tokenizer import load_tokenizer
from utils.ddp import setup, cleanup
from utils.collate_fn import skip_broken_collate_fn
from utils.gpu_manager import get_gpu_temp, wait_for_cooldown
from utils.logging import log_text_image_embeddings
from utils.save_model import save_checkpoint, load_checkpoint, save_weights


def train_anchor(rank, world_size, args):
    setup()
    device = torch.device(f'cuda:{args.local_rank}')

    writer = None
    if rank == 0:
        log_dir = os.path.join(args.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    tokenizer = load_tokenizer(tokenizer_file=args.tokenizer_path)

    image_encoder = load_image_encoder(device)
    text_encoder = TextEncoder(vocab_size=tokenizer.vocab_size).to(device)

    params_to_optimize = list(text_encoder.parameters()) + [p for p in image_encoder.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=args.lr)

    start_epoch = 1
    best_loss = float('inf')
    if args.resume:
        _, _ = load_checkpoint(text_encoder, optimizer, Path(args.output_dir), 'text_encoder')
        start_epoch, best_loss = load_checkpoint(image_encoder, optimizer, Path(args.output_dir), 'image_encoder')

    dataset = RefinedImageDataset(Path(args.data_dir))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=skip_broken_collate_fn
    )

    image_encoder = DDP(image_encoder, device_ids=[args.local_rank], find_unused_parameters=False)
    text_encoder = DDP(text_encoder, device_ids=[args.local_rank], find_unused_parameters=False)

    image_encoder.train()
    text_encoder.train()
    global_step = (start_epoch - 1) * len(dataloader)

    for epoch in range(start_epoch, args.epochs + 1):
        total_loss = 0.0
        sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, desc=f"[GPU {rank}] Epoch {epoch}", disable=(rank != 0))

        for _, batch in enumerate(dataloader):
            if batch is None:
                progress_bar.update(1)
                continue
            if get_gpu_temp(rank) >= 80:
                wait_for_cooldown(rank)

            images = batch["image"].to(device)
            raw_text = [t['raw_text'] for t in batch["text"]]

            tokenized_raw = tokenizer(
                raw_text,
                padding=True,
                truncation=True,
                max_length=64,
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

            if rank == 0 and writer is not None:
                writer.add_scalar('Loss/train_anchor', loss.item(), global_step)
                global_step += 1

        avg_loss = total_loss / len(dataloader)
        if rank == 0 and avg_loss < best_loss:
            best_loss = avg_loss
            save_weights(text_encoder.module, best_loss, 'text_encoder', Path(args.output_dir))
            save_weights(image_encoder.module, best_loss, 'image_encoder', Path(args.output_dir))
            print(f'[Epoch {epoch}] encoder saved with loss {avg_loss:.4f}', flush=True)

        if rank == 0 and epoch % 10 == 0:
            save_checkpoint(epoch, text_encoder.module, optimizer, best_loss, Path(args.output_dir), 'text_encoder')
            save_checkpoint(epoch, image_encoder.module, optimizer, best_loss, Path(args.output_dir), 'image_encoder')
            sample_batch = None
            for sample in dataloader:
                if sample is not None:
                    sample_batch = sample
                    break

            if sample_batch:
                images = sample_batch["image"].to(device)
                raw_texts = [t["raw_text"] for t in sample_batch["text"]]
                log_text_image_embeddings(
                    writer,
                    tag=f"Embeddings/Epoch_{epoch}",
                    images=images,
                    raw_texts=raw_texts,
                    image_model=image_encoder,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    device=device
                )

    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--local_rank", type=int, default=os.environ.get("LOCAL_RANK", 0))
    args = parser.parse_args()

    rank = args.local_rank
    world_size = torch.cuda.device_count()

    train_anchor(rank, world_size, args)


if __name__ == '__main__':
    main()
