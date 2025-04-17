import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
import argparse
from pathlib import Path
from collections import Counter
from utils.translator import translate, load_cache, save_cache


def collect_tag_frequencies(data_dir: str, out_json="tag_freq.json"):
    metadata_dir = Path(data_dir) / "metadata"
    image_files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    tag_counter = Counter()
    for img_file in image_files:
        meta_path = metadata_dir / (Path(img_file).stem + '.json')
        if not meta_path.exists():
            continue

        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            raw_tags = metadata.get('tags', [])
            for tag in raw_tags:
                tag = tag.strip().lower()
                if tag:
                    tag_counter[tag] += 1
        except Exception as e:
            print(f"Error reading {img_file}: {e}")

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(dict(tag_counter), f, ensure_ascii=False, indent=2)
    print(f"[DONE] Collected frequencies of {len(tag_counter)} tags → {out_json}")


def process_metadata(metadata_path, tag_freq_dict=None, min_freq=5, use_title=True, translate_tags=True):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    raw_title = metadata.get('title', '') if use_title else ''
    raw_tags = list(set(metadata.get('tags', [])))
    if not raw_title and not raw_tags:
        return None

    cleaned_tags = []
    if raw_title:
        cleaned_tags.append(raw_title)
        if translate_tags:
            cleaned_tags.append(translate(raw_title))

    for tag in raw_tags:
        if not tag:
            continue
        if tag_freq_dict and tag_freq_dict.get(tag, 0) < min_freq:
            continue
        cleaned_tags.append(tag)
        if translate_tags:
            cleaned_tags.append(translate(tag))

    return list(dict.fromkeys(cleaned_tags))


def generate_tags_file(
    data_dir: str,
    freq_json: str = "tag_freq.json",
    output_path: str = "tags.txt",
    min_freq: int = 5,
    refined_data_dir: str = None
):
    with open(freq_json, 'r', encoding='utf-8') as f:
        tag_freq_dict = json.load(f)

    image_files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(('.png', '.jpg', 'jpeg'))
    ]
    metadata_dir = Path(data_dir) / "metadata"

    seen_lines = set()
    count = 0

    load_cache()

    with open(output_path, 'w', encoding='utf-8') as fout:
        # image
        for img_file in image_files:
            metadata_path = metadata_dir / (Path(img_file).stem + '.json')
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                raw_title = metadata.get('title', '')
                raw_tags = list(set(metadata.get('tags', [])))
                if not raw_title and not raw_tags:
                    continue

                cleaned_tags = []
                if raw_title:
                    cleaned_tags.append(raw_title)
                    cleaned_tags.append(translate(raw_title))

                for tag in raw_tags:
                    if not tag:
                        continue

                    freq = tag_freq_dict.get(tag, 0)
                    if freq < min_freq:
                        continue

                    cleaned_tags.append(tag)
                    cleaned_tags.append(translate(tag))

                cleaned_tags = list(dict.fromkeys(cleaned_tags))
                if not cleaned_tags:
                    continue

                line = ' '.join(cleaned_tags)
                if line not in seen_lines:
                    fout.write(line + '\n')
                    seen_lines.add(line)
                    count += 1

            except Exception as e:
                print(f"[ERROR] {img_file}: {e}")

        # refined data
        if refined_data_dir:
            refined_data_dir = Path(refined_data_dir)
            refined_metadata_dir = refined_data_dir / "metadata"
            refined_image_files = [
                f for f in os.listdir(refined_data_dir)
                if f.lower().endswith(('.png', '.jpg', 'jpeg'))
            ]

            for img_file in refined_image_files:
                metadata_path = refined_metadata_dir / (Path(img_file).stem + '.json')
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

    save_cache()
    print(f"[DONE] Generated {count} unique translated tag lines in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root directory (with images and metadata/)")
    parser.add_argument("--output_path", type=str, default="tags.txt", help="Where to save the tag text file")
    parser.add_argument("--freq_json", type=str, default="tag_freq.json", help="Where to save/load tag frequency data")
    parser.add_argument("--refined_data_dir", type=str, default=None, help="Optional refined data directory")

    args = parser.parse_args()

    collect_tag_frequencies(args.data_dir)
    generate_tags_file(
        args.data_dir,
        args.freq_json,
        args.output_path,
        refined_data_dir=args.refined_data_dir
    )

