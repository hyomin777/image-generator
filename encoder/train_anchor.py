import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm

from dataset import RefinedImageDataset
from loss import cosine_contrastive_loss
from utils.ddp import setup, cleanup
from utils.gpu_manager import get_gpu_temp, wait_for_cooldown
from utils.logging import log_text_image_embeddings
from utils.save_model import save_checkpoint, load_checkpoint, save_weights
from train import setup_encoder_train_components, initialize_encoders, wrap_encoders, summary_writer


def train_anchor(rank, world_size, args):
    setup()
    device = torch.device(f'cuda:{args.local_rank}')

    tokenizer, dataloader = setup_encoder_train_components(args, rank, world_size, RefinedImageDataset)
    image_encoder, text_encoder = initialize_encoders(tokenizer, device, lora=True)
    optimizer = optim.AdamW(
        list(text_encoder.parameters()) + [p for p in image_encoder.parameters() if p.requires_grad],
        lr=args.lr
    )
    writer = None
    if rank == 0:
        writer = summary_writer(args)

    if args.resume:
        _, _ = load_checkpoint(text_encoder, optimizer, Path(args.output_dir), 'text_encoder')
        start_epoch, best_loss = load_checkpoint(image_encoder, optimizer, Path(args.output_dir), 'image_encoder')
    else:
        start_epoch, best_loss = 1, float('inf')

    image_encoder, text_encoder = wrap_encoders(args, image_encoder, text_encoder)

    global_step = (start_epoch - 1) * len(dataloader)
    for epoch in range(start_epoch, args.epochs + 1):
        image_encoder.train()
        text_encoder.train()
        total_loss = 0.0
        dataloader.sampler.set_epoch(epoch)
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
                max_length=128,
                return_tensors='pt'
            )
            input_ids_raw = tokenized_raw.input_ids.to(device)
            attention_mask_raw = tokenized_raw.attention_mask.to(device)

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
                global_step += 1
                if global_step % 100 == 0:
                    writer.add_scalar('Loss/anchor_step', loss.item(), global_step)

        avg_loss = total_loss / len(dataloader)
        if rank == 0 and writer is not None:
            writer.add_scalar('Loss/anchor_epoch', avg_loss, epoch)

        if rank == 0 and avg_loss < best_loss:
            best_loss = avg_loss
            save_weights(text_encoder.module, 'text_encoder', Path(args.output_dir))
            save_weights(image_encoder.module, 'image_encoder', Path(args.output_dir))
            print(f'[Epoch {epoch}] encoder saved with loss {avg_loss:.4f}', flush=True)

        if rank == 0 and epoch % 10 == 0:
            save_checkpoint(epoch, text_encoder.module, optimizer, best_loss, Path(args.output_dir), 'text_encoder')
            save_checkpoint(epoch, image_encoder.module, optimizer, best_loss, Path(args.output_dir), 'image_encoder')
            
            sample_batch = next((s for s in dataloader if s is not None), None)
            if sample_batch:
                images = sample_batch["image"].to(device)
                raw_texts = [t["raw_text"] for t in sample_batch["text"]]
                log_text_image_embeddings(
                    writer,
                    tag=f"Embeddings/Epoch_{epoch}",
                    images=images,
                    raw_texts=raw_texts,
                    image_encoder=image_encoder,
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
    parser.add_argument("--batch_size", type=int, default=32)
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
