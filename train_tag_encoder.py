import os

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from transformers import CLIPProcessor, CLIPModel

from tqdm import tqdm

from dataset import ImageDataset
from tag_encoder import TagEncoder
from loss import cosine_contrastive_loss


def train_tag_encoder():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="image dataset directory")
    parser.add_argument("--output_dir", type=str,
                        default="output", help="model save directory")
    parser.add_argument("--epochs", type=int, default=10,
                        help="epochs for training")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="data loader worker count")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # image encoder
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip.eval()

    # tag encoder
    tag_encoder = TagEncoder()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        tag_encoder = DataParallel(tag_encoder)
    tag_encoder = tag_encoder.to(device)
    optimizer = optim.AdamW(tag_encoder.parameters(), lr=1e-5)

    # dataset & dataloader
    dataset = ImageDataset(
        Path(args.data_dir)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # train loop
    for epoch in range(args.epochs):
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")
        for batch in dataloader:
            images = batch["image"].to(device)
            tags = batch["text"]

            with torch.no_grad():
                image_embeds = clip.get_image_features(
                    pixel_values=images).to(device)

            tag_embeds = tag_encoder(tags).to(device)
            loss = cosine_contrastive_loss(tag_embeds, image_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

    if isinstance(tag_encoder, DataParallel):
        model_to_save = tag_encoder.module
    else:
        model_to_save = tag_encoder

    # save model
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"tag_encoder_epoch_{epoch}.pt")
    torch.save(model_to_save.state_dict(), save_path)


if __name__ == '__main__':
    train_tag_encoder()
