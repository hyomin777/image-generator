import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
from pathlib import Path
import argparse
from utils.translator import translate, load_cache, save_cache


def generate_tags_file(data_dir: str, output_path: str = "tags.txt", max_tags=10):
    metadata_dir = Path(data_dir) / "metadata"
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    seen_lines = set()
    count = 0

    load_cache()

    with open(output_path, 'w', encoding='utf-8') as fout:
        for img_file in image_files:
            metadata_path = metadata_dir / (Path(img_file).stem + '.json')
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                raw_tags = list(set(metadata.get('tags', [])))[:max_tags]
                if not raw_tags:
                    continue

                translated_tags = [translate(tag, target_lang='en') for tag in raw_tags]
                translated_tags = list(dict.fromkeys(translated_tags))

                line = ' '.join(translated_tags).strip()
                if line and line not in seen_lines:
                    fout.write(line + '\n')
                    seen_lines.add(line)
                    count += 1

            except Exception as e:
                print(f"[ERROR] {img_file}: {e}")

    save_cache()

    print(f"[DONE] Generated {count} unique translated tag lines in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root directory (with images and metadata/)")
    parser.add_argument("--output_path", type=str, default="tags.txt", help="Where to save the tag text file")
    parser.add_argument("--max_tags", type=int, default=10, help="Maximum tags per image")
    args = parser.parse_args()

    generate_tags_file(args.data_dir, args.output_path, args.max_tags)
