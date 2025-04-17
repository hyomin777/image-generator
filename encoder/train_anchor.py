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


@torch.no_grad()
def log_text_image_embeddings(writer, tag, images, raw_texts, image_model, text_encoder, tokenizer, device):
    if writer is None:
        return

    images = images[:100]
    raw_texts = raw_texts[:100]

    tokenized = tokenizer(
        raw_texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors='pt'
    )
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    image_model_eval = image_model.module if hasattr(image_model, "module") else image_model
    image_embeds = image_model_eval.get_image_features(pixel_values=images)

    text_encoder_eval = text_encoder.module if hasattr(text_encoder, "module") else text_encoder
    text_embeds = text_encoder_eval(input_ids=input_ids, attention_mask=attention_mask)

    all_embeds = torch.cat([image_embeds, text_embeds], dim=0)
    all_labels = [f"IMG: {t}" for t in raw_texts] + [f"TXT: {t}" for t in raw_texts]
    dummy_imgs = torch.zeros_like(images.cpu())
    all_imgs = torch.cat([images.cpu(), dummy_imgs], dim=0)

    writer.add_embedding(
        all_embeds,
        metadata=all_labels,
        label_img=all_imgs,
        tag=tag
    )


def save_checkpoint(epoch, text_encoder, image_encoder, optimizer, best_loss, output_dir: Path):
    checkpoint = {
        'epoch': epoch,
        'text_encoder_state_dict': text_encoder.state_dict(),
        'image_encoder_state_dict': image_encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss
    }
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / 'checkpoint.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved at epoch {epoch}')


def load_checkpoint(text_encoder, image_encoder, optimizer, output_dir: Path):
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
    return 1, float('inf')


def save_weights(model, loss, save_name, output_dir: Path):
    output_dir = output_dir / 'weights'
    output_dir.mkdir(exist_ok=True, parents=True)
    weights_path = output_dir / f'{save_name}.pth'
    torch.save(model.state_dict(), weights_path)
    print(f'Model saved: {save_name} with loss {loss:.4f}')


def train_anchor(rank, world_size, args):
    setup()
    device = torch.device(f'cuda:{args.local_rank}')

    writer = None
    if rank == 0:
        log_dir = os.path.join(args.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
    tokenizer.pad_token = '[PAD]'
    tokenizer.cls_token = '[CLS]'
    tokenizer.sep_token = '[SEP]'

    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip.eval()
    for param in clip.parameters():
        param.requires_grad = False
    for i in [-2, -1]:
        block = clip.vision_model.encoder.layers[i]
        sa = block.self_attn
        sa.q_proj = LoRALinear(sa.q_proj, r=4, alpha=1.0).to(device)
        sa.v_proj = LoRALinear(sa.v_proj, r=4, alpha=1.0).to(device)

    vocab_size = tokenizer.vocab_size
    tag_encoder = TagEncoder(vocab_size=vocab_size).to(device)

    params_to_optimize = list(tag_encoder.parameters()) + [p for p in clip.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=args.lr)

    start_epoch = 1
    best_loss = float('inf')
    if args.resume:
        start_epoch, best_loss = load_checkpoint(tag_encoder, clip, optimizer, Path(args.output_dir))

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

    clip = DDP(clip, device_ids=[args.local_rank], find_unused_parameters=False)
    tag_encoder = DDP(tag_encoder, device_ids=[args.local_rank], find_unused_parameters=False)

    clip.train()
    tag_encoder.train()
    for epoch in range(start_epoch, args.epochs + 1):
        total_loss = 0.0
        sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, desc=f"[GPU {rank}] Epoch {epoch}", disable=(rank != 0))

        for step, batch in enumerate(dataloader):
            if batch is None:
                progress_bar.update(1)
                continue
            if get_gpu_temp(rank) >= 80:
                wait_for_cooldown(rank)

            images = batch["image"].to(device)
            raw_text = [t['raw_text'] for t in batch["text"]]

            tokenized_raw = tokenizer(raw_text, padding=True, truncation=True, max_length=32, return_tensors='pt')
            input_ids_raw = tokenized_raw.input_ids.to(device)
            attention_mask_raw = tokenized_raw.attention_mask.to(device)

            with torch.no_grad():
                image_embeds = clip.module.get_image_features(pixel_values=images)

            raw_text_embeds = tag_encoder(input_ids=input_ids_raw, attention_mask=attention_mask_raw)

            loss = cosine_contrastive_loss(raw_text_embeds, image_embeds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

            if rank == 0 and writer is not None:
                global_step = epoch * len(dataloader) + step
                writer.add_scalar('Loss/train_anchor', loss.item(), global_step)

        avg_loss = total_loss / len(dataloader)
        if rank == 0 and avg_loss < best_loss:
            best_loss = avg_loss
            save_weights(tag_encoder.module, best_loss, 'text_encoder', Path(args.output_dir))
            save_weights(clip.module, best_loss, 'image_encoder', Path(args.output_dir))
            print(f'[Epoch {epoch}] encoder saved with loss {avg_loss:.4f}', flush=True)

        if rank == 0 and epoch % 10 == 0:
            save_checkpoint(epoch, tag_encoder.module, clip.module, optimizer, best_loss, Path(args.output_dir))
            sample_batch = next(iter(dataloader))
            images = sample_batch["image"].to(device)
            raw_texts = [t["raw_text"] for t in sample_batch["text"]]
            log_text_image_embeddings(
                writer,
                tag=f"Embeddings/Epoch_{epoch}",
                images=images,
                raw_texts=raw_texts,
                image_model=clip,
                text_encoder=tag_encoder,
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
