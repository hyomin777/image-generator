import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import CLIPModel, PreTrainedTokenizerFast

from tqdm import tqdm
from dataset import ImageDataset
from tag_encoder import TagEncoder
from loss import cosine_contrastive_loss
from utils.ddp import setup, cleanup
from utils.gpu_manager import get_gpu_temp, wait_for_cooldown
from utils.collate_fn import skip_broken_collate_fn


def train_tag_encoder(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # load tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
    tokenizer.pad_token = '[PAD]'
    tokenizer.cls_token = '[CLS]'
    tokenizer.sep_token = '[SEP]'

    # image encoder
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip.train()

    image_resume_path = Path(args.output_dir) / 'image_encoder.pt'
    if image_resume_path.exists():
        print(f'[rank: {rank}] Loading weights from {image_resume_path}', flush=True)
        clip.load_state_dict(torch.load(image_resume_path, map_location=device))

    for param in clip.parameters():
        param.requires_grad = False

    for param in clip.vision_model.encoder.layers[-2:].parameters():
        param.requires_grad = True

    clip = DDP(clip, device_ids=[rank], find_unused_parameters=False)

    # tag encoder
    vocab_size = tokenizer.vocab_size
    tag_encoder = TagEncoder(vocab_size=vocab_size).to(device)
    tag_resume_path = Path(args.output_dir) / 'tag_encoder.pt'

    if tag_resume_path.exists():
        print(f'[rank: {rank}] Loading weights from {tag_resume_path}', flush=True)
        tag_encoder.load_state_dict(torch.load(tag_resume_path, map_location=device))

    tag_encoder = DDP(tag_encoder, device_ids=[rank], find_unused_parameters=False)

    # optimizer
    params_to_optimize = list(tag_encoder.parameters()) + list(p for p in clip.parameters() if p.requires_grad)
    optimizer = optim.AdamW(params_to_optimize, lr=args.lr)


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
        pin_memory=True,
        collate_fn=skip_broken_collate_fn
    )

    # train loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        total_loss = 0.0
        sampler.set_epoch(epoch)
        progress_bar = tqdm(
            dataloader,
            desc=f"[GPU {rank}] Epoch {epoch}",
            disable=(rank != 0)
        )

        for batch in dataloader:
            if batch is None:
                continue
            temp = get_gpu_temp(rank)
            if temp >= 80:
                wait_for_cooldown(rank)

            images = batch["image"].to(device)
            raw_texts = batch["raw_text"]
            translated_texts = batch['translated_text']

            # tokenize
            tokenized_raw = tokenizer(raw_texts, padding=True, truncation=True, max_length=32, return_tensors='pt')
            tokenized_translated = tokenizer(translated_texts, padding=True, truncation=True, max_length=32, return_tensors='pt')

            input_ids_raw = tokenized_raw.input_ids.to(device)
            attention_mask_raw = tokenized_raw.attention_mask.to(device)

            input_ids_translated = tokenized_translated.input_ids.to(device)
            attention_mask_translated = tokenized_translated.attention_mask.to(device)

            with torch.no_grad():
                image_embeds = clip.module.get_image_features(pixel_values=images)

            # get embeddings
            raw_text_embeds = tag_encoder(input_ids=input_ids_raw, attention_mask=attention_mask_raw)
            translated_text_embeds = tag_encoder(input_ids=input_ids_translated, attention_mask=attention_mask_translated)

            # compute contrastive loss
            loss_image_raw = cosine_contrastive_loss(raw_text_embeds, image_embeds)
            loss_image_translated = cosine_contrastive_loss(translated_text_embeds, image_embeds)
            loss_raw_translated = cosine_contrastive_loss(raw_text_embeds, translated_text_embeds)

            loss = ((loss_image_raw + loss_image_translated) / 2) + loss_raw_translated

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        if rank == 0 and avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(args.output_dir, exist_ok=True)

            tag_encoder_path = Path(args.output_dir) / 'tag_encoder.pt'
            torch.save(tag_encoder.module.state_dict(), tag_encoder_path)

            image_encoder_path = Path(args.output_dir) / 'image_encoder.pt'
            torch.save(clip.module.state_dict(), image_encoder_path)
            print(f'[Epoch {epoch}] encoder saved with loss {avg_loss:.4f}', flush=True)

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="image dataset directory")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="path to tokenizer.json")
    parser.add_argument("--output_dir", type=str, default="output", help="model save directory")
    parser.add_argument("--epochs", type=int, default=100, help="epochs for training")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="data loader worker count")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(train_tag_encoder, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
