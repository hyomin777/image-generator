import os
import json
import shutil
from pathlib import Path
from PIL import Image
import pytesseract


def has_text(img_path, threshold=20):
    text = pytesseract.image_to_string(Image.open(img_path))
    return len(text.strip()) >= threshold

def is_low_quality(img_path, min_size=512):
    image = Image.open(img_path)
    return image.width < min_size or image.height < min_size

def is_manga_ratio(img_path):
    image = Image.open(img_path)
    aspect = image.width / image.height
    return aspect < 0.7 or aspect > 1.8

def should_filter(img_path, tag_text):
    try:
        if has_text(img_path): return True, "text"
        if is_low_quality(img_path): return True, "low_quality"
        if is_manga_ratio(img_path): return True, "manga_ratio"
        return False, "ok"
    except Exception as e:
        return True, f"error:{str(e)}"

def main(image_dir):
    image_dir = Path(image_dir)
    metadata_dir = image_dir / "metadata"
    filtered_dir = image_dir / "filtered"
    filtered_metadata_dir = filtered_dir / "metadata"

    filtered_dir.mkdir(exist_ok=True)
    filtered_metadata_dir.mkdir(exist_ok=True)

    filtered = []

    for img_file in image_dir.glob("*.jpg"):
        meta_path = metadata_dir / f"{img_file.stem}.json"
        if not meta_path.exists():
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        tags = meta.get("tags", [])
        title = meta.get("title", "")
        tag_text = title + " " + " ".join(tags)

        drop, reason = should_filter(img_file, tag_text)
        if drop:
            print(f"[FILTERED] {img_file.name}: {reason}")
            filtered.append((img_file.name, reason))

            # Move image
            shutil.move(str(img_file), str(filtered_dir / img_file.name))
            # Move metadata
            shutil.move(str(meta_path), str(filtered_metadata_dir / meta_path.name))

    print(f"\nTotal filtered: {len(filtered)}")
    with open("filtered_images.json", "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])

