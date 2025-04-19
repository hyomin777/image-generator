import torch


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
        max_length=64,
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