import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torchvision.utils import make_grid
from tqdm import tqdm

from image_generator import ImageGenerator
from dataset import RefinedImageDataset, LMDBImageDataset
from utils.gpu_manager import get_gpu_temp, wait_for_cooldown
from utils.save_model import save_checkpoint, load_checkpoint, save_weights
from setup_training import setup, cleanup, summary_writer, setup_train_dataloader, wrap_model


def train_generator(args):
    setup()
    device = torch.device(f'cuda:{args.local_rank}')

    # image generator
    image_generator = ImageGenerator(device, args.tokenizer_path)

    # dataset
    dataloader = setup_train_dataloader(args, LMDBImageDataset)

    # optimizer
    params_to_optimize = [p for p in image_generator.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=args.lr)

    # Mixed Precision scaler
    scaler = torch.GradScaler(device.type)

    # tensorboard writer
    writer = None
    if args.local_rank == 0:
        writer = summary_writer(args)

    if args.resume_epoch:
        _, _, image_generator.unet = load_checkpoint(image_generator.unet, device, optimizer, Path(args.output_dir), f'generator_unet_{args.resume_epoch}')
        _, best_loss, image_generator.text_encoder = load_checkpoint(image_generator.text_encoder, device, optimizer, Path(args.output_dir), f'generator_text_encoder_{args.resume_epoch}')
    else:
        best_loss = 1

    start_epoch = 2

    image_generator = wrap_model(args.local_rank, image_generator)

    # train loop
    global_step = (start_epoch - 1) * len(dataloader)
    for epoch in range(start_epoch, args.epochs + 1):
        image_generator.train()
        total_loss = 0.0
        dataloader.sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, desc=f"[GPU {args.local_rank}] Epoch {epoch}", disable=(args.local_rank != 0))

        num_skipped_batches = 0
        last_loss_value = None
        for batch in dataloader:
            if batch is None:
                progress_bar.update(1)
                continue

            images = batch["image"].to(device)
            raw_text = [t['raw_text'] for t in batch["text"]]

            optimizer.zero_grad()

            with torch.autocast(device.type):
                loss = image_generator.module.train_step(images, raw_text)

            if loss is None:
                num_skipped_batches += 1
                progress_bar.update(1)
                continue

            last_loss_value = loss.item()

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(image_generator.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

            if args.local_rank == 0 and writer is not None:
                global_step += 1
                if global_step % 200 == 0:
                    writer.add_scalar('Loss/generator_step', loss.item(), global_step)

        avg_loss = total_loss / len(dataloader)
        if args.local_rank == 0 and avg_loss < best_loss:
            best_loss = avg_loss
            save_weights(image_generator.module.unet, 'unet', Path(args.output_dir))
            save_weights(image_generator.module.text_encoder, 'text_encoder', Path(args.output_dir))
            print(f'[Epoch {epoch}] Generator saved with loss {avg_loss:.4f}', flush=True)

        if args.local_rank == 0:
            print(f"[Epoch {epoch}] Skipped {num_skipped_batches} batches due to NaN or INF")
            save_checkpoint(
                epoch,
                image_generator.module.unet,
                optimizer, best_loss, Path(args.output_dir), f'generator_unet_{epoch}')
            save_checkpoint(
                epoch,
                image_generator.module.text_encoder,
                optimizer, best_loss, Path(args.output_dir), f'generator_text_encoder_{epoch}')
            writer.add_scalar('Loss/generator_epoch', avg_loss, epoch)

            image_generator.eval()
            with torch.no_grad():
                sample_text = ['1girl black_shirt black_skirt blue_archive black_halo shiroko_(blue_archive)']
                generated_image = image_generator.module(sample_text)
                grid = make_grid(generated_image, nrow=1)
                writer.add_image(f'Generated/Epoch_{epoch}', grid, epoch)
    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/hhd/dataset")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--text_encoder_path", type=str, default="output/weights/text_encoder.pth")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--local_rank", type=int, default=os.environ.get("LOCAL_RANK", 0))
    parser.add_argument("--resume_epoch", type=int)
    args = parser.parse_args()

    train_generator(args)


if __name__ == '__main__':
    main()
