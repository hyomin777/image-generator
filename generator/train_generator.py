import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
from dataset import LMDBImageDataset
from image_generator import ImageGenerator
from trainer import GeneratorTrainer, TrainConfig


def train_generator(rank, world_size, args):
    config = TrainConfig(
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        resume=args.resume
    )
    trainer = GeneratorTrainer(rank, world_size, ImageGenerator, LMDBImageDataset, config)
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train_generator(local_rank, world_size, args)
