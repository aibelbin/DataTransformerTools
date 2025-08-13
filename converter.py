#!/usr/bin/env python3
"""
Convert custom document layout annotations to YOLOv8 dataset format.

Input structure (required):
  raw_data/
    images/  (PNG images)
    jsons/   (annotation JSON files)

Annotation JSON schema example:
{
  "file_name": "doc_04953.png",
  "annotations": [
      {"bbox": [x_min, y_min, width, height], "category_id": int},
      ...
  ]
}

Bounding boxes are absolute pixel values.
Class mapping:
 0 Background
 1 Text
 2 Title
 3 List
 4 Table
 5 Figure

Output structure produced:
 datasets/
   images/
     train/
     val/
   labels/
     train/
     val/

Each label file <image_stem>.txt contains lines:
  class_id x_center y_center width height
with coordinates normalized to [0,1] relative to image width & height.

No training code included.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
from PIL import Image
from sklearn.model_selection import train_test_split
import argparse
import sys

CLASS_IDS = {0,1,2,3,4,5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Custom JSON -> YOLOv8 dataset converter")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root (default: current directory)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Proportion of images for training set (default 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--output", type=Path, default=Path("datasets"), help="Output datasets directory root")
    return parser.parse_args()


def load_annotations(json_dir: Path) -> Dict[str, Any]:
    ann_map: Dict[str, Any] = {}
    for jf in sorted(json_dir.glob("*.json")):
        try:
            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)
            fn = data.get("file_name")
            if not fn:
                print(f"[WARN] Missing file_name in {jf}", file=sys.stderr)
                continue
            ann_map[fn] = data
        except Exception as e:
            print(f"[ERROR] Failed to read {jf}: {e}", file=sys.stderr)
    return ann_map


def to_yolo_line(bbox: List[float], cls_id: int, img_w: int, img_h: int) -> str:
    x_min, y_min, w, h = bbox
    # Clip to bounds
    x_min = max(0.0, min(x_min, img_w))
    y_min = max(0.0, min(y_min, img_h))
    w = max(0.0, min(w, img_w - x_min))
    h = max(0.0, min(h, img_h - y_min))
    x_center = (x_min + w / 2.0) / img_w if img_w else 0.0
    y_center = (y_min + h / 2.0) / img_h if img_h else 0.0
    w_n = w / img_w if img_w else 0.0
    h_n = h / img_h if img_h else 0.0
    return f"{cls_id} {x_center:.6f} {y_center:.6f} {w_n:.6f} {h_n:.6f}"


def write_label(path: Path, lines: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def split_files(files: List[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    if not files:
        return [], []
    train, val = train_test_split(files, train_size=train_ratio, random_state=seed, shuffle=True)
    return list(train), list(val)


def copy_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def convert(args: argparse.Namespace):
    raw_images = args.root / "raw_data" / "images"
    raw_jsons = args.root / "raw_data" / "jsons"

    # Fallback to legacy directory names if the expected structure is missing
    if not raw_images.exists() or not raw_jsons.exists():
        legacy_images = args.root / "raw_images"
        legacy_jsons = args.root / "raw_json"
        if legacy_images.exists() and legacy_jsons.exists():
            print("[INFO] Using legacy directories raw_images/ and raw_json/ as input.")
            raw_images = legacy_images
            raw_jsons = legacy_jsons
        else:
            print(f"[ERROR] Required directories not found. Expecting either:\n  {raw_images} and {raw_jsons}\n  or legacy {legacy_images} and {legacy_jsons}")
            return 1

    ann_map = load_annotations(raw_jsons)
    image_files = {p.name: p for p in raw_images.glob("*.png")}

    all_image_names = sorted(image_files.keys())
    train_names, val_names = split_files(all_image_names, args.train_ratio, args.seed)
    train_set = set(train_names)

    out_images_train = args.output / "images" / "train"
    out_images_val = args.output / "images" / "val"
    out_labels_train = args.output / "labels" / "train"
    out_labels_val = args.output / "labels" / "val"

    stats = {"images": 0, "labels": 0, "boxes": 0, "missing_json": 0}

    for img_name in all_image_names:
        img_path = image_files[img_name]
        json_data = ann_map.get(img_name)

        is_train = img_name in train_set
        img_dst = (out_images_train if is_train else out_images_val) / img_name
        label_dst = (out_labels_train if is_train else out_labels_val) / f"{Path(img_name).stem}.txt"

        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:
            print(f"[ERROR] Cannot open image {img_path}: {e}", file=sys.stderr)
            continue

        lines: List[str] = []
        if json_data is None:
            stats["missing_json"] += 1
        else:
            for ann in json_data.get("annotations", []):
                bbox = ann.get("bbox")
                cls_id = ann.get("category_id")
                if (not isinstance(bbox, list)) or len(bbox) != 4:
                    print(f"[WARN] Bad bbox in {img_name}: {bbox}", file=sys.stderr)
                    continue
                if cls_id not in CLASS_IDS:
                    print(f"[WARN] Unknown class id {cls_id} in {img_name} skipping", file=sys.stderr)
                    continue
                lines.append(to_yolo_line(bbox, cls_id, w, h))
        write_label(label_dst, lines)
        copy_image(img_path, img_dst)

        stats["images"] += 1
        stats["labels"] += 1
        stats["boxes"] += len(lines)

    print("Conversion complete")
    for k,v in stats.items():
        print(f"  {k}: {v}")
    print(f"Train images: {len(train_names)} | Val images: {len(val_names)}")
    print(f"Output root: {args.output.resolve()}")
    return 0


def main():
    args = parse_args()
    exit(convert(args))

if __name__ == "__main__":  # pragma: no cover
    main()
