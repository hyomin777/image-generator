import torch


def skip_broken_collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None

    images = [b["image"] for b in batch]
    text = [b["text"] for b in batch]
    images = torch.stack(images, dim=0)

    return {"image": images, "text": text}
