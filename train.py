import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import clip
from model import TextToImageModel
from diffusers.optimization import get_scheduler
from tqdm import tqdm
import argparse
from torch.nn.parallel import DataParallel


class WebImageDataset(Dataset):
    def __init__(self, data_dir, min_clip_score=0.2, min_image_size=256):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(
            data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cpu")

        self.filtered_files = []
        for img_file in tqdm(self.image_files, desc="Filtering images"):
            img_path = os.path.join(data_dir, img_file)
            try:
                image = Image.open(img_path).convert('RGB')
                if min(image.size) < min_image_size:
                    continue

                text = os.path.splitext(img_file)[0].replace('_', ' ')
                image_input = self.preprocess(image).unsqueeze(0)
                text_input = clip.tokenize([text])

                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input)
                    text_features = self.clip_model.encode_text(text_input)
                    similarity = torch.cosine_similarity(
                        image_features, text_features)

                if similarity.item() >= min_clip_score:
                    self.filtered_files.append(img_file)

            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue

        print(
            f"Filtered {len(self.filtered_files)} images from {len(self.image_files)} total images")

    def __len__(self):
        return len(self.filtered_files)

    def __getitem__(self, idx):
        img_name = self.filtered_files[idx]
        img_path = os.path.join(self.data_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        image = self.preprocess(image)

        text = os.path.splitext(img_name)[0].replace('_', ' ')

        return {"image": image, "text": text}


def main():
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
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)
    model = model.to(device)

    dataset = WebImageDataset(
        args.data_dir,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * args.epochs
    )

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            images = batch["image"].to(device, non_blocking=True)
            texts = batch["text"]

            with torch.set_grad_enabled(True):
                outputs = model(texts)
                loss = nn.MSELoss()(outputs, images)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

        save_path = os.path.join(args.output_dir, f"model_epoch_{epoch}.pt")
        if isinstance(model, DataParallel):
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)
        print(f"model saved to {save_path}")


if __name__ == "__main__":
    main()
