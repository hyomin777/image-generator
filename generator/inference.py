import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


import argparse
from pathlib import Path
from functools import partial
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig, FullOptimStateDictConfig, StateDictType
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from image_generator import ImageGenerator
from torch.nn import TransformerEncoderLayer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def fsdp_main(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=int(1e6),
        force_leaf_modules={TransformerEncoderLayer}
    )

    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    model = ImageGenerator(device, args.tokenizer_path)
    model.vae.to(device)
    model.text_encoder.to(device)
    model.unet = FSDP(model.unet.to(device), auto_wrap_policy=auto_wrap_policy, mixed_precision=mp_policy)

    # TensorBoard
    writer = SummaryWriter(log_dir=Path(args.output_dir) / 'logs') if rank == 0 else None

    if args.resume and os.path.exists(args.resume):
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        checkpoint = torch.load(args.resume, map_location=map_location)

        with FSDP.state_dict_type(model, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                                   optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),):
            model.unet.load_state_dict(checkpoint["unet"])
            model.text_encoder.load_state_dict(checkpoint["text_encoder"])

        if rank == 0:
            print(f"[INFO] Resumed from {args.resume}")


    print(f"Rank={rank}, UNet param count: {sum(p.numel() for p in model.unet.parameters())}")
    print(f"Rank={rank}, TextEncoder param count: {sum(p.numel() for p in model.text_encoder.parameters())}")
    print(f"Rank={rank}, VAE param count: {sum(p.numel() for p in model.vae.parameters())}")
    model.eval()
    with torch.no_grad():
        sample_text = ["1girl black_shirt blue_archive shiroko_(blue_archive)"]
        gen = model(sample_text)

        if rank == 0:
            grid = make_grid(gen, nrow=1)
            writer.add_image(f"Generated/Epoch_{args.epoch}", grid, args.epoch)
            print("Image Generated Successfully")
    dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    fsdp_main(local_rank, world_size, args)
