import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
from pathlib import Path
from utils.translator import translate, load_cache, save_cache


def build_translation_cache(data_dir: Path):
    metadata_dir = data_dir / "metadata"
    filenames = [f for f in os.listdir(data_dir) if f.endswith((".jpg", ".jpeg", ".png"))]

    load_cache()

    total = len(filenames)
    print(f"[build_cache] Found {total} image files")

    for i, filename in enumerate(filenames):
        metadata_path = metadata_dir / (Path(filename).stem + ".json")
        if not metadata_path.exists():
            continue

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        raw_title = metadata.get('title', '')
        raw_tags = list(set(metadata.get("tags", [])))
        if not raw_title and not raw_tags:
            continue
        if raw_title:
            raw_tags.append(raw_title)

        # translate each tag and store in cache
        _ = [translate(tag) for tag in raw_tags]

        if (i + 1) % 1000 == 0:
            save_cache()
            print(f"/n/n[build_cache] Processed {i+1}/{total} images/n/n")

    save_cache()
    print("[build_cache] Cache build complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to image dataset")
    args = parser.parse_args()

    build_translation_cache(Path(args.data_dir))

