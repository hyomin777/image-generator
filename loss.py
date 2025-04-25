import torch
import torch.nn.functional as F


def cosine_contrastive_loss(text_embeds, image_embeds, temperature=0.07):
    # normalize embeddings
    text_embeds = F.normalize(text_embeds, dim=-1)
    image_embeds = F.normalize(image_embeds, dim=-1)

    # similarity matrix
    logits_per_text = text_embeds @ image_embeds.T / temperature  # (B, B)
    logits_per_image = logits_per_text.T  # (B, B)

    # ground truth labels (diagonal is positive)
    batch_size = text_embeds.size(0)
    labels = torch.arange(batch_size, device=text_embeds.device)

    # cross-entropy losses
    loss_t2i = F.cross_entropy(logits_per_text, labels)  # text → image
    loss_i2t = F.cross_entropy(logits_per_image, labels)  # image → text

    return (loss_t2i + loss_i2t) / 2

