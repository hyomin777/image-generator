import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from encoder.text_encoder import TextEncoder
from transformers import CLIPModel, PreTrainedTokenizerFast
from dataset import ImageDataset
from pathlib import Path


def evaluate_similarity_matrix(text_encoder, image_encoder, tokenizer, dataset, device, num_samples=32, save_path='output/similarity_matrix.png'):
    text_encoder.eval()
    image_encoder.eval()

    images, texts = [], []
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        images.append(sample['image'])
        texts.append(sample['text'])

    images = torch.stack(images).to(device)
    with torch.no_grad():
        image_embeds = image_encoder.get_image_features(pixel_values=images)
        tokenized = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        text_embeds = text_encoder(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask)

    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    similarity = text_embeds @ image_embeds.T

    plt.figure(figsize=(8, 6))
    plt.imshow(similarity.cpu().numpy(), cmap='viridis')
    plt.colorbar(label='Cosine Similarity')
    plt.title('Text ↔ Image Similarity Matrix')
    plt.xlabel('Image Index')
    plt.ylabel('Text Index')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[+] Similarity matrix saved to {save_path}")


def retrieve_images_by_text(text_encoder, image_encoder, tokenizer, dataset, text_query, device, top_k=5):
    text_encoder.eval()
    image_encoder.eval()

    all_images = torch.stack([sample['image'] for sample in dataset]).to(device)
    with torch.no_grad():
        image_embeds = image_encoder.get_image_features(pixel_values=all_images)
        tokenized = tokenizer([text_query], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        text_embed = text_encoder(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask)

    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embed = F.normalize(text_embed, dim=-1)

    sims = (text_embed @ image_embeds.T).squeeze(0)
    top_indices = sims.topk(top_k).indices.tolist()
    print(f"[Query: '{text_query}'] Top {top_k} matching image indices: {top_indices}")
    return top_indices


def retrieve_tags_by_image(text_encoder, image_encoder, tokenizer, dataset, image_index, device, top_k=5):
    text_encoder.eval()
    image_encoder.eval()

    texts = [sample['text'] for sample in dataset]
    with torch.no_grad():
        image_embed = image_encoder.get_image_features(pixel_values=dataset[image_index]['image'].unsqueeze(0).to(device))
        tokenized = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        text_embeds = text_encoder(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask)

    image_embed = F.normalize(image_embed, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    sims = (image_embed @ text_embeds.T).squeeze(0)
    top_indices = sims.topk(top_k).indices.tolist()
    print(f"[Image {image_index}] Top {top_k} matching text tags:")
    for i in top_indices:
        print(f"  - {texts[i]}")
    return [texts[i] for i in top_indices]


def plot_positive_negative_distribution(text_encoder, image_encoder, tokenizer, dataset, device, num_samples=128, save_path='output/pos_neg_hist.png'):
    text_encoder.eval()
    image_encoder.eval()

    pos_sims = []
    neg_sims = []

    for i in range(min(num_samples, len(dataset) - 1)):
        img = dataset[i]['image'].unsqueeze(0).to(device)
        text = dataset[i]['text']
        neg_text = dataset[i + 1]['text']

        with torch.no_grad():
            img_embed = F.normalize(image_encoder.get_image_features(pixel_values=img), dim=-1)
            pos_tok = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            neg_tok = tokenizer([neg_text], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            text_embed = F.normalize(text_encoder(input_ids=pos_tok.input_ids, attention_mask=pos_tok.attention_mask), dim=-1)
            neg_embed = F.normalize(text_encoder(input_ids=neg_tok.input_ids, attention_mask=neg_tok.attention_mask), dim=-1)

        pos_sims.append((img_embed @ text_embed.T).item())
        neg_sims.append((img_embed @ neg_embed.T).item())

    plt.hist(pos_sims, bins=20, alpha=0.6, label="Positive")
    plt.hist(neg_sims, bins=20, alpha=0.6, label="Negative")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Positive vs Negative Similarity Distribution")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'[+] Pos/Neg similarity histogram saved to {save_path}')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/tokenizer.json")
    tokenizer.pad_token = "[PAD]"
    tokenizer.cls_token = "[CLS]"
    tokenizer.sep_token = "[SEP]"

    text_encoder = TextEncoder(vocab_size=tokenizer.vocab_size).to(device)
    text_encoder.load_state_dict(torch.load("output/text_encoder.pt", map_location=device))
    text_encoder.eval()

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.load_state_dict(torch.load("output/image_encoder.pt", map_location=device))
    clip_model.eval()

    dataset = ImageDataset(Path("/mnt/usb/image"))

    evaluate_similarity_matrix(text_encoder, clip_model, tokenizer, dataset, device)
    plot_positive_negative_distribution(text_encoder, clip_model, tokenizer, dataset, device)
    retrieve_tags_by_image(text_encoder, clip_model, tokenizer, dataset, image_index=0, device=device, top_k=5)

