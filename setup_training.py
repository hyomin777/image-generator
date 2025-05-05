import os
from pathlib import Path

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from utils.collate_fn import skip_broken_collate_fn


def setup():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def setup_train_dataloader(args, dataset_cls):
    dataset = dataset_cls(Path(args.data_dir))
    sampler = DistributedSampler(dataset, num_replicas=args.num_workers, rank=args.local_rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=skip_broken_collate_fn
    )
    return dataloader


def wrap_model(rank, model):
    model = DDP(model, device_ids=[rank])
    return model


def summary_writer(args):
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer
