import torch
from torch.utils.tensorboard import SummaryWriter


@torch.no_grad()
def log_text_image_embeddings(writer: SummaryWriter, tag, images, raw_texts, image_encoder, text_encoder, tokenizer, device):
    if writer is None:
        return

    images = images[:100]
    raw_texts = raw_texts[:100]

    tokenized = tokenizer(
        raw_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    image_encoder = image_encoder.module if hasattr(image_encoder, "module") else image_encoder
    image_encoder.eval()
    image_embeds = image_encoder.get_image_features(pixel_values=images)

    text_encoder = text_encoder.module if hasattr(text_encoder, "module") else text_encoder
    text_encoder.eval()
    text_embeds = text_encoder(input_ids=input_ids, attention_mask=attention_mask)

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