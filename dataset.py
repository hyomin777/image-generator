import os
import json
from pathlib import Path

import clip
import torch
from torch.utils.data import Dataset
from torch.distributed import is_initialized, get_rank

from tqdm import tqdm
from PIL import Image

from utils.translator import translate, save_cache, load_cache, get_cache


class ImageDataset(Dataset):
    def __init__(self, data_dir: Path, min_clip_score=0.2, min_image_size=256):
        self.data_dir = data_dir
        self.metadata_dir = data_dir / 'metadata'
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cpu")

        self.filtered_files = []
        self.image_to_tags = {}

        load_cache()
        for img_file in tqdm(self.image_files, desc="Filtering images"):
            try:
                img_path = data_dir / img_file
                metadata_path = self.metadata_dir / (Path(img_file).stem + '.json') 

                if not metadata_path.exists():
                    continue

                image = Image.open(img_path).convert('RGB')
                if min(image.size) < min_image_size:
                    continue

                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                raw_tags = metadata.get('tags', [])
                raw_tags = list(set(raw_tags))

                if not raw_tags:
                    continue

                if len(raw_tags) > 10:
                    raw_tags = raw_tags[:10]

                tags = [translate(tag) for tag in raw_tags]
                text = ' '.join(tags)
                image_input = self.preprocess(image).unsqueeze(0)
                text_input = clip.tokenize([text], truncate=True)

#                with torch.no_grad():
#                    image_features = self.clip_model.encode_image(image_input)
#                   text_features = self.clip_model.encode_text(text_input)
#                    similarity = torch.cosine_similarity(image_features, text_features)

#                if similarity.item() >= min_clip_score:
                self.filtered_files.append(img_file)
                self.image_to_tags[img_file] = text

            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue

        print(f"Filtered {len(self.filtered_files)} images from {len(self.image_files)} total images")

        if not is_initialized() or get_rank() == 0:
            print(f'[cache] saving {len(get_cache())} entries to cache.json')
            save_cache()


    def __len__(self):
        return len(self.filtered_files)

    def __getitem__(self, idx):
        img_name = self.filtered_files[idx]
        img_path = self.data_dir / img_name

        image = Image.open(img_path).convert('RGB')
        image = self.preprocess(image)

        text = self.image_to_tags[img_name]

        return {"image": image, "text": text}
