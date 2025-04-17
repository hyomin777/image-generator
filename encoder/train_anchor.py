import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import PreTrainedTokenizerFast, CLIPModel
from tqdm import tqdm

from dataset import RefinedImageDataset
from tag_encoder import TagEncoder
from lora import LoRALinear
from loss import cosine_contrastive_loss
from utils.ddp import setup, cleanup
from utils.gpu_manager import get_gpu_temp, wait_for_cooldown
from utils.collate_fn import skip_broken_collate_fn


def save_checkpoint(epoch, text_encoder, image_encoder, optimizer, best_loss, output_dir:Path):
    checkpoint = {
        'epoch': epoch,
        'text_encoder_state_dict': text_encoder.state_dict(),
        'image_encoder_state_dict': image_encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss
    }
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f'checkpoint.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved at epoch {epoch}')


def load_checkpoint(text_encoder, image_encoder, optimizer, output_dir:Path):
    checkpoint_path = output_dir / 'checkpoints/checkpoint.pth'

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        image_encoder.load_state_dict(checkpoint['image_encoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f'Checkpoint loaded from epoch {checkpoint["epoch"]}')
        return start_epoch, best_loss
    return 1, 0.0


def save_weights(model, loss, save_name, output_dir:Path):
    output_dir = output_dir / 'weights'
    output_dir.mkdir(exist_ok=True)
    weigths_path = output_dir / f'{save_name}.pth'
    torch.save(model.state_dict(), weigths_path)
    print(f'Model saved with accuracy: {loss:.2f}%')


def train_anchor(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # tensorboard writer
    writer = None
    if rank == 0:
        log_dir = os.path.join(args.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    # load tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
    tokenizer.pad_token = '[PAD]'
    tokenizer.cls_token = '[CLS]'
    tokenizer.sep_token = '[SEP]'

    # image encoder
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip.eval()

    for param in clip.parameters():
        param.requires_grad = False

    for i in [-2, -1]:
        block = clip.vision_model.encoder.layers[i]

        sa = block.self_attn
        sa.q_proj = LoRALinear(sa.q_proj, r=4, alpha=1.0).to(device)
        sa.v_proj = LoRALinear(sa.v_proj, r=4, alpha=1.0).to(device)

    # tag encoder
    vocab_size = tokenizer.vocab_size
    tag_encoder = TagEncoder(vocab_size=vocab_size).to(device)

    # optimizer
    params_to_optimize = list(tag_encoder.parameters()) + list(p for p in clip.parameters() if p.requires_grad)
    optimizer = optim.AdamW(params_to_optimize, lr=args.lr)

    # resume
    start_epoch = 1
    best_loss = float('inf')
    if args.resume:
        start_epoch, best_loss = load_checkpoint(tag_encoder, clip, optimizer)

    # dataset and DDP-compatible loader
    dataset = RefinedImageDataset(
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
    clip = DDP(clip, device_ids=[rank], find_unused_parameters=False)
    tag_encoder = DDP(tag_encoder, device_ids=[rank], find_unused_parameters=False)

    # train loop
    clip.train()
    tag_encoder.train()
    for epoch in range(start_epoch, args.epochs+1):
        total_loss = 0.0
        sampler.set_epoch(epoch)
        progress_bar = tqdm(
            dataloader,
            desc=f"[GPU {rank}] Epoch {epoch}",
            disable=(rank != 0)
        )

        for step, batch in enumerate(dataloader):
            if batch is None:
                continue
            temp = get_gpu_temp(rank)
            if temp >= 80:
                wait_for_cooldown(rank)

            images = batch["image"].to(device)
            raw_text = [t['raw_text'] for t in batch["text"]]

            # tokenize
            tokenized_raw = tokenizer(raw_text, padding=True, truncation=True, max_length=32, return_tensors='pt')
            input_ids_raw = tokenized_raw.input_ids.to(device)
            attention_mask_raw = tokenized_raw.attention_mask.to(device)

            with torch.no_grad():
                image_embeds = clip.module.get_image_features(pixel_values=images)

            # get embeddings
            raw_text_embeds = tag_encoder(input_ids=input_ids_raw, attention_mask=attention_mask_raw)

            # compute contrastive loss
            loss = cosine_contrastive_loss(raw_text_embeds, image_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

            if rank == 0 and writer is not None:
                global_step = epoch * len(dataloader) + step
                writer.add_scalar('Loss/train_anchor', loss.item(), global_step)

        avg_loss = total_loss / len(dataloader)
        if rank == 0 and avg_loss < best_loss:
            best_loss = avg_loss
            save_weights(tag_encoder.module, best_loss, 'text_encoder', args.output_dir)
            save_weights(clip.moudle, best_loss, 'image_encoder', args.output_dir)
            print(f'[Epoch {epoch}] encoder saved with loss {avg_loss:.4f}', flush=True)

        if rank == 0 and epoch % 10 == 0:
            save_checkpoint(epoch, tag_encoder.module, clip.module, optimizer, best_loss)

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
    mp.spawn(train_anchor, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
