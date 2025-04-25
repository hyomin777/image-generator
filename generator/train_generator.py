import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
from pathlib import Path

import torch
import torch.optim as optim

from tqdm import tqdm

from image_generator import ImageGenerator, load_image_generator
from dataset import RefinedImageDataset
from utils.ddp import setup, cleanup
from utils.gpu_manager import get_gpu_temp, wait_for_cooldown
from utils.save_model import save_checkpoint, load_checkpoint, save_weights
from train import summary_writer, setup_train_dataloader, wrap_model 


def train_generator(args):
    setup()
    device = torch.device(f'cuda:{args.local_rank}')

    # image generator
    image_generator: ImageGenerator = load_image_generator(device, args.tokenizer_path)

    # dataset
    dataloader = setup_train_dataloader(args, RefinedImageDataset)

    # optimizer
    params_to_optimize = [param for param in image_generator.parameters() if param.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=args.lr)

    # tensorboard writer
    writer = None
    if args.local_rank == 0:
        writer = summary_writer(args)
    
    # load checkpoint
    if args.resume:
        start_epoch, best_loss = load_checkpoint(image_generator, optimizer, Path(args.output_dir), 'image_generator')
    else:
        start_epoch, best_loss = 1, float('inf')

    image_generator = wrap_model(args.local_rank, image_generator)

    # train loop
    global_step = (start_epoch - 1) * len(dataloader)
    for epoch in range(start_epoch, args.epochs + 1):
        image_generator.train()
        total_loss = 0.0
        dataloader.sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, desc=f"[GPU {args.local_rank}] Epoch {epoch}", disable=(args.local_rank != 0))

        for batch in dataloader:
            if batch is None:
                progress_bar.update(1)
                continue
            if get_gpu_temp(args.local_rank) >= 80:
                wait_for_cooldown(args.local_rank)

            images = batch["image"].to(device)
            raw_text = [t['raw_text'] for t in batch["text"]]

            loss = image_generator.train_step(images, raw_text)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

            if args.local_rank == 0 and writer is not None:
                global_step += 1
                if global_step % 100 == 0:
                    writer.add_scalar('Loss/generatpr_step', loss.item(), global_step)

        avg_loss = total_loss / len(dataloader)
        if args.local_rank == 0 and avg_loss < best_loss:
            best_loss = avg_loss
            save_weights(image_generator.module, 'image_generator', Path(args.output_dir))
            print(f'[Epoch {epoch}] Generator saved with loss {avg_loss:.4f}', flush=True)
       
        if args.local_rank == 0 and epoch % 5 == 0:
            save_checkpoint(epoch, image_generator.module, optimizer, best_loss, Path(args.output_dir), 'image_generator')

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/usb/refined_images")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=os.environ.get("LOCAL_RANK", 0))
    parser.add_argument("--resume", action='store_true')
    args = parser.parse_args()

    train_generator(args)


if __name__ == '__main__':
    main()