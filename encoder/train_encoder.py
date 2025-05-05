import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from encoder.image_encoder import ImageEncoder
from encoder.text_encoder import TextEncoder
from dataset import LMDBImageDataset
from tokenizer.tokenizer import load_tokenizer
from loss import cosine_contrastive_loss
from setup_training import wrap_model
from utils.collate_fn import skip_broken_collate_fn
from transform import normalize
from utils.tensorboard_logging import log_text_image_embeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_anchor(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    dataset = LMDBImageDataset(args.data_dir)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=skip_broken_collate_fn)

    tokenizer = load_tokenizer(args.tokenizer_path)
    image_encoder = ImageEncoder().to(device)
    text_encoder = TextEncoder(vocab_size=tokenizer.vocab_size).to(device)

    params_to_optimize = list(text_encoder.parameters()) + [p for p in image_encoder.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=args.lr)

    scaler = torch.GradScaler(device.type)
    writer = SummaryWriter(log_dir=Path(args.output_dir) / 'logs') if rank == 0 else None

    global_step = 1
    best_loss = float('inf')
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        checkpoint = torch.load(args.resume, map_location=map_location)

        text_encoder.load_state_dict(checkpoint['text_encoder'])
        image_encoder.load_state_dict(checkpoint['image_encoder'])
        global_step = checkpoint["global_step"]
        best_loss = checkpoint["best_loss"]
        start_epoch = checkpoint["epoch"] + 1

        if rank == 0:
            print(f"[INFO] Resumed from {args.resume} | epoch: {start_epoch} | step: {global_step} | loss: {best_loss}")

    image_encoder = wrap_model(rank, image_encoder)
    text_encoder = wrap_model(rank, text_encoder)

    for epoch in range(start_epoch, args.epochs + 1):
        image_encoder.train()
        text_encoder.train()
        total_loss = 0.0
        dataloader.sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, desc=f"[GPU {rank}] Epoch {epoch}", disable=(rank != 0))

        for batch in dataloader:
            if batch is None:
                progress_bar.update(1)
                continue

            images = batch["image"].to(device, non_blocking=True)
            images = normalize(images)
            raw_text = [t['raw_text'] for t in batch["text"]]

            tokenized_raw = tokenizer(
                raw_text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            input_ids_raw = tokenized_raw.input_ids.to(device, non_blocking=True)
            attention_mask_raw = tokenized_raw.attention_mask.to(device, non_blocking=True)

            with torch.autocast(device.type):
                image_embeds = image_encoder(pixel_values=images)
                raw_text_embeds = text_encoder(input_ids=input_ids_raw, attention_mask=attention_mask_raw)
                loss = cosine_contrastive_loss(raw_text_embeds, image_embeds)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

            if rank == 0 and global_step % 100 == 0:
                if writer is not None:
                    writer.add_scalar('Loss/anchor_step', loss.item(), global_step)
            global_step += 1

        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            if writer is not None:
                writer.add_scalar('Loss/anchor_epoch', avg_loss, epoch)

            text_encoder_state = text_encoder.module.state_dict() if hasattr(text_encoder, "module") else text_encoder.state_dict()
            image_encoder_state = image_encoder.module.state_dict() if hasattr(image_encoder, "module") else image_encoder.state_dict()

            if avg_loss < best_loss:
                best_loss = avg_loss
                weights_path = Path(args.output_dir) / 'weights'
                weights_path.mkdir(parents=True, exist_ok=True)
                torch.save(text_encoder_state, weights_path / 'text_encoder.pth')
                torch.save(image_encoder_state, weights_path / 'image_encoder.pth')
                print(f'[Epoch {epoch}] Encoder weights saved with loss {avg_loss:.4f}', flush=True)

            checkpoint = {
                'text_encoder': text_encoder_state,
                'image_encoder': image_encoder_state,
                'optimizer': optimizer.state_dict(),
                'global_step': global_step,
                'best_loss': best_loss,
                'epoch': epoch
            }
            checkpoint_path = Path(args.output_dir) / 'checkpoints'
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path / f'encoder_checkpoint_{epoch}.pt')
            print(f'[Epoch {epoch}] Encoder checkpoint saved', flush=True)

            if writer is not None:
                sample_batch = next(
                    (s for s in dataloader if s is not None), None)
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
                        device=device,
                        global_step=epoch
                    )

    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/usb/lmdb")
    parser.add_argument("--tokenizer_path", type=str, default='tokenizer/tokenizer.json')
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train_anchor(local_rank, world_size, args)
