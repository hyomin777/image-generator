import os
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.translator import load_cache, get_cache


class ImageDataset(Dataset):
    def __init__(self, data_dir: Path, max_tags=10):
        self.data_dir = data_dir
        self.metadata_dir = data_dir / 'metadata'
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.filtered_files = []
        self.image_to_tags = {}

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        load_cache()
        cache = get_cache()

        for img_file in self.image_files:
            try:
                img_path = data_dir / img_file
                metadata_path = self.metadata_dir / (Path(img_file).stem + '.json')

                if not metadata_path.exists():
                    continue

                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                raw_tags = list(set(metadata.get('tags', [])))[:max_tags]
                if not raw_tags:
                    continue

                raw_text = ' '.join(raw_tags)
                trainlated_text = ' '.join([cache.get(f"{tag}|en", tag) for tag in raw_tags])

                self.filtered_files.append(img_file)
                self.image_to_tags[img_file] = {'raw_text': raw_text, 'translated_text', translated_text}

            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue

        print(f"Filtered {len(self.filtered_files)} images from {len(self.image_files)} total images")

    def __len__(self):
        return len(self.filtered_files)

    def __getitem__(self, idx):
        img_name = self.filtered_files[idx]
        img_path = self.data_dir / img_name

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[WARNING] Failed to load {img_name}: {e}")
            return None

        image = self.transform(image)
        texts = self.image_to_tags[img_name]

        return {"image": image, "raw_text": texts['raw_text'], 'translated_text': texts['translated_text']}

