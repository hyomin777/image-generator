import os

import argparse
from pathlib import Path 

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from diffusers.optimization import get_scheduler

from tqdm import tqdm
from dataset import ImageDataset
from model import TextToImageModel


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="image dataset directory")
    parser.add_argument("--output_dir", type=str,
                        default="output", help="model save directory")
    parser.add_argument("--epochs", type=int, default=10, help="epochs for training")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float,
                        default=1e-4, help="learning rate")
    parser.add_argument("--min_clip_score", type=float,
                        default=0.2, help="minimum clip score")
    parser.add_argument("--min_image_size", type=int,
                        default=256, help="minimum image size")
    parser.add_argument("--num_workers", type=int,
                        default=4, help="data loader worker count")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextToImageModel()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    model = model.to(device)

    dataset = ImageDataset(
        Path(args.data_dir),
        min_clip_score=args.min_clip_score,
        min_image_size=args.min_image_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * args.epochs
    )

    writer = SummaryWriter(log_dir=args.output_dir + "/logs")
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            images = batch["image"].to(device, non_blocking=True)
            texts = batch["text"]

            loss = model.module.train_step(images, texts, model.module.scheduler)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            global_step = epoch * len(dataloader) + step
            writer.add_scalar("train/loss", loss.item(), global_step)

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

        save_path = os.path.join(args.output_dir, f"model_epoch_{epoch}.pt")
        if isinstance(model, DataParallel):
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)
        print(f"model saved to {save_path}")
    
    validation_prompts = [
        "BlueArchive",
        "kasuga_tsubaki, blue_archive"
    ]

    model.eval()
    with torch.no_grad():
        for i, prompt in enumerate(validation_prompts):
            if isinstance(model, DataParallel):
                generated_image = model.module(prompt)  # (1, 3, 512, 512)
            else:
                generated_image = model(prompt)
            save_path = os.path.join(args.output_dir, f"val_epoch_{epoch}_img_{i}.png")
            save_image(generated_image, save_path)


if __name__ == "__main__":
    train()

