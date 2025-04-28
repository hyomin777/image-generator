import os
import lmdb
import json
import cv2
import tqdm
import pickle
from pathlib import Path


def encode_image(image_path):
    # Load image file and Encode to jpeg
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    _, img_encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return img_encoded.tobytes()

def encode_metadata(metadata_path):
    # Load metadata(json) and serialize
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    raw_tags = list(set(metadata.get('tags', [])))
    if not raw_tags:
        return None

    raw_text = ' '.join(raw_tags)
    metadata = {
        'raw_text': raw_text
    }
    return pickle.dumps(metadata)

def make_lmdb(image_dir, output_path, write_frequency=5000):
    image_dir = Path(image_dir)
    metadata_dir = image_dir / 'metadata'
    files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    db = lmdb.open(str(output_path), map_size=600 * 1024 * 1024 * 1024)  # 600GB
    txn = db.begin(write=True)

    for idx, img_file in enumerate(tqdm.tqdm(files)):
        img_path = image_dir / img_file
        metadata_path = metadata_dir / (Path(img_file).stem + '.json')

        try:
            img_encoded = encode_image(img_path)
        except Exception as e:
            print(f"Skipping broken image {img_file}: {e}")
            continue

        if not metadata_path.exists():
            print(f"Skipping {img_file}: metadata missing")
            continue

        try:
            meta_encoded = encode_metadata(metadata_path)
            if meta_encoded is None:
                continue
        except Exception as e:
            print(f"Skipping broken metadata {metadata_path}: {e}")
            continue

        key_img = f"image-{idx:08d}".encode('ascii')
        key_meta = f"meta-{idx:08d}".encode('ascii')

        txn.put(key_img, img_encoded)
        txn.put(key_meta, meta_encoded)

        if idx % write_frequency == 0:
            txn.commit()
            print(f'commited at idx: {idx}')
            txn = db.begin(write=True)

    txn.commit()
    db.close()
    print(f"Finished LMDB dataset with {len(files)} images.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default='/mnt/usb/refined_images')
    parser.add_argument("--output_path", type=str, default='/mnt/hhd/dataset')
    args = parser.parse_args()

    make_lmdb(args.image_dir, args.output_path)
