import os
import io
import lmdb
import pickle
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.translator import load_cache, translate

Image.MAX_IMAGE_PIXELS = None


class BaseImageDataset(Dataset):
    def __init__(self, data_dir: Path, is_train=True):
        self.data_dir = data_dir
        self.metadata_dir = data_dir / 'metadata'
        self.image_paths = []
        self.image_to_tags = {}

        all_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in all_files:
            img_path = data_dir / img_file
            metadata_path = self.metadata_dir / (Path(img_file).stem + '.json')

            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception:
                continue
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                tags = metadata.get('tags', [])
                if not tags:
                    continue
            except Exception:
                continue

            self.image_paths.append(img_file)
        print(f"Loaded {len(self.image_paths)} clean images out of {len(all_files)} files.")

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomApply([
                    transforms.RandomResizedCrop(
                        size=(224, 224),
                        scale=(0.8, 1.0),
                        ratio=(0.8, 1.2)
                    )
                ], p=0.8),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.05,
                        contrast=0.05,
                        saturation=0.05,
                        hue=0.02
                    )
                ], p=0.5),
                transforms.RandomApply([
                    transforms.RandomRotation(15)
                ], p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
             ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])

        self._map_tag_to_image()

    def _map_tag_to_image(self):
        raise NotImplementedError("Subclasses must implement _map_tag_to_image")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = self.data_dir / img_name

        with Image.open(img_path) as img:
            image = img.convert('RGB').copy()

        image = self.transform(image)
        text = self.image_to_tags[img_name]
        return {"image": image, "text": text}


class RefinedImageDataset(BaseImageDataset):
    def _map_tag_to_image(self):
        for img_file in self.image_paths:
            try:
                metadata_path = self.metadata_dir / (Path(img_file).stem + '.json')

                if not metadata_path.exists():
                    continue

                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                raw_tags = list(set(metadata.get('tags', [])))
                if not raw_tags:
                    continue

                raw_text = ' '.join(raw_tags)
                self.image_to_tags[img_file] = {'raw_text': raw_text}

            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue

        print(f"Filtered {len(self.image_paths)} images total")


class ImageDataset(BaseImageDataset):
    def __init__(self, data_dir: Path,is_train=True):
        super().__init__(data_dir, is_train)
        load_cache()

    def _map_tag_to_image(self):
        for img_file in self.image_paths:
            try:
                metadata_path = self.metadata_dir / (Path(img_file).stem + '.json')

                if not metadata_path.exists():
                    continue

                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                raw_title = metadata.get('title', '')
                raw_tags = list(set(metadata.get('tags', [])))
                if not raw_title and not raw_tags:
                    continue

                if raw_title:
                    raw_text = raw_title + ' ' + ' '.join(raw_tags)
                else:
                    raw_text = ' '.join(raw_tags)

                translated_title = translate(raw_title)
                translated_tags = [translate(tag) for tag in raw_tags]
                translated_text = (translated_title + ' ' if translated_title else '') + ' '.join(translated_tags)
                self.image_to_tags[img_file] = {'raw_text': raw_text, 'translated_text': translated_text}

            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue

        print(f"Filtered {len(self.image_paths)} images total")


class LMDBImageDataset(Dataset):
    def __init__(self, lmdb_dir: Path, is_train=True):
        self.lmdb_dir = lmdb_dir

        self.env = lmdb.open(
            str(self.lmdb_dir),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=32
        )

        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] // 2

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomApply([
                    transforms.RandomResizedCrop(
                        size=(224, 224),
                        scale=(0.8, 1.0),
                        ratio=(0.8, 1.2)
                    )
                ], p=0.8),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.05,
                        contrast=0.05,
                        saturation=0.05,
                        hue=0.02
                    )
                ], p=0.5),
                transforms.RandomApply([
                    transforms.RandomRotation(15)
                ], p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            img_bytes = txn.get(f'image-{idx:08d}'.encode())
            meta_bytes = txn.get(f'meta-{idx:08d}'.encode())

        try:
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:        #, 아예 에러를 터뜨리지 말고 넘어가게
            return self.__getitem__((idx + 1) % len(self))  # 다음 인덱스로 넘어가기

        meta = pickle.loads(meta_bytes)
        raw_text = meta['raw_text']

        image = self.transform(image)

        return {"image": image, "text":{'raw_text': raw_text}}
