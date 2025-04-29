import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
import argparse
from pathlib import Path


def process_metadata(metadata_path):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    raw_tags = list(set(metadata.get('tags', [])))
    if not raw_tags:
        return None

    cleaned_tags = []
    for tag in raw_tags:
        if not tag:
            continue
        cleaned_tags.append(tag)

    return list(dict.fromkeys(cleaned_tags))


def generate_tags_file(
    data_dir: str,
    output_path: str = "tags.txt",
):
    seen_lines = set()
    count = 0


    with open(output_path, 'w', encoding='utf-8') as fout:
        # refined data
        if data_dir:
            data_dir = Path(data_dir)
            metadata_dir = data_dir / "metadata"
            image_files = [
                f for f in os.listdir(data_dir)
                if f.lower().endswith(('.png', '.jpg', 'jpeg'))
            ]

            for img_file in image_files:
                metadata_path = metadata_dir / (Path(img_file).stem + '.json')
                if not metadata_path.exists():
                    continue

                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    raw_tags = list(set(metadata.get('tags', [])))
                    if not raw_tags:
                        continue

                    cleaned_tags = []
                    for tag in raw_tags:
                        if not tag:
                            continue
                        cleaned_tags.append(tag)

                    cleaned_tags = list(dict.fromkeys(cleaned_tags))
                    if not cleaned_tags:
                        continue

                    line = ' '.join(cleaned_tags)
                    if line not in seen_lines:
                        fout.write(line + '\n')
                        seen_lines.add(line)
                        count += 1

                except Exception as e:
                    print(f"[ERROR] (refined) {img_file}: {e}")

    print(f"[DONE] Generated {count} unique translated tag lines in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/usb/refined_images", help="Dataset root directory (with images and metadata/)")
    parser.add_argument("--output_path", type=str, default="tags.txt", help="Where to save the tag text file")

    args = parser.parse_args()

    generate_tags_file(
        args.data_dir,
        args.output_path,
    )


