import torch
import torch.nn.functional as F


def cosine_contrastive_loss(text_embeds, image_embeds, temperature=0.07):
    text_embeds = F.normalize(text_embeds, dim=-1)
    image_embeds = F.normalize(image_embeds, dim=-1)
    logits = text_embeds @ image_embeds.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return F.cross_entropy(logits, labels)