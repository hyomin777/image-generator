import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
from pathlib import Path

import torch
from torchvision.utils import make_grid
from tqdm import tqdm
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from image_generator import ImageGenerator
from dataset import LMDBImageDataset
from utils.temp_manager import wait_for_cooldown
from setup_training import summary_writer, setup_train_dataloader


def save_fsdp_weights(model, name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{name}.pth"

    with FSDP.state_dict_type(
        model,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ):
        state_dict = model.state_dict()

    if torch.distributed.get_rank() == 0:
        torch.save(state_dict, save_path)
        print(f"[FSDP SAVE] {save_path}")


def train_generator(args):
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_type = "FULL_STATE_DICT",
        state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=True),
        auto_wrap_policy = lambda m, *_: False,
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    device = accelerator.device

    # Model
    image_generator = ImageGenerator(device, args.tokenizer_path)
    image_generator.unet = FSDP(image_generator.unet)
    image_generator.text_encoder = FSDP(image_generator.text_encoder)

    # Dataset & Dataloader
    dataloader = setup_train_dataloader(args, LMDBImageDataset, accelerator)

    unet_fsdp, text_encoder_fsdp, dataloader = accelerator.prepare(
        image_generator.unet, image_generator.text_encoder, dataloader
    )
    image_generator.vae.to(device, dtype=torch.float32)
    image_generator.text_encoder = text_encoder_fsdp
    image_generator.unet = unet_fsdp

    # Optimizer
    params_to_optimize = [p for p in image_generator.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr)

    # TensorBoard
    if accelerator.is_main_process:
        writer = summary_writer(args)
    else:
        writer = None

    # Resume
    if args.resume:
        accelerator.load_state(args.resume)


    # Train Loop
    global_step = 1
    best_loss = 1

    for epoch in range(1, args.epochs + 1):
        image_generator.train()
        total_loss = 0.0

        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        progress_bar = tqdm(dataloader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            if batch is None:
                continue

            if global_step % 50 == 0:
                wait_for_cooldown(gpu_id=accelerator.local_process_index)

            images = batch["image"]
            raw_text = [t['raw_text'] for t in batch["text"]]

            with accelerator.autocast():
                loss = image_generator.train_step(images, raw_text)

            if loss is None:
                continue

            optimizer.zero_grad()
            accelerator.backward(loss)

            accelerator.unscale_gradients()
            torch.nn.utils.clip_grad_norm_(image_generator.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

            if accelerator.is_main_process and global_step % 100 == 0:
                writer.add_scalar('Loss/generator_step', loss.item(), global_step)
            global_step += 1

        avg_loss = total_loss / len(dataloader)
        if accelerator.is_main_process:
            accelerator.save_state(os.path.join(args.output_dir, f"generator_{epoch}"))
            writer.add_scalar('Loss/generator_epoch', avg_loss, epoch)
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_fsdp_weights(image_generator.unet, 'unet', Path(args.output_dir))
                save_fsdp_weights(image_generator.text_encoder, 'text_encoder', Path(args.output_dir))

            image_generator.eval()
            with torch.no_grad():
                sample_text = ["1girl black_shirt black_skirt blue_archive black_halo shiroko_(blue_archive)"]
                generated_image = image_generator(sample_text)
                grid = make_grid(generated_image, nrow=1)
                writer.add_image(f'Generated/Epoch_{epoch}', grid, epoch)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/hhd/dataset")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train_generator(args)
