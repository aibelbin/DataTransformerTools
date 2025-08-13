#!/usr/bin/env python3
"""
Convert YOLOv8 detection prediction TXT files into competition JSON format.

Input:
  --images  : Folder with original PNG images (used for dimensions)
  --labels  : Folder containing YOLO prediction .txt files (one per image) with lines:
              class_id x_center y_center width height confidence
              (all coords normalized 0-1; confidence in [0,1])
  --output  : Destination folder for JSON files (one per image)
  --conf    : Confidence threshold (default 0.25)

Output JSON per image (example):
{
  "file_name": "doc_04953.png",
  "annotations": [
    {"bbox": [56.28, 94.35, 240.55, 257.81], "category_id": 1, "category_name": "Text"},
    ...
  ],
  "corruption": {"type": "none", "severity": 0}
}

Notes:
 - If a label file is missing, an empty annotations list is written.
 - Bounding boxes converted back to absolute pixel coordinates: [x_min, y_min, width, height].
 - Coordinates are clipped to image bounds.
 - Supports images with no detections (produces empty annotations array).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import sys

CLASS_MAP = {
    0: "Background",
    1: "Text",
    2: "Title",
    3: "List",
    4: "Table",
    5: "Figure",
}
VALID_CLASS_IDS = set(CLASS_MAP.keys())

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO predictions -> competition JSON converter")
    parser.add_argument("--images", type=Path, required=True, help="Path to folder with PNG images")
    parser.add_argument("--labels", type=Path, required=True, help="Path to folder with YOLO prediction .txt files")
    parser.add_argument("--output", type=Path, required=True, help="Output folder for JSON files")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25)")
    parser.add_argument("--round", type=int, default=2, help="Decimal places to round bbox values (default 2)")
    return parser.parse_args()


def load_image_size(img_path: Path):
    with Image.open(img_path) as im:
        return im.size  # (w,h)


def yolo_to_bbox_abs(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int):
    x_min = (xc - w / 2.0) * img_w
    y_min = (yc - h / 2.0) * img_h
    bw = w * img_w
    bh = h * img_h
    # Clip
    x_min = max(0.0, min(x_min, img_w))
    y_min = max(0.0, min(y_min, img_h))
    bw = max(0.0, min(bw, img_w - x_min))
    bh = max(0.0, min(bh, img_h - y_min))
    return [x_min, y_min, bw, bh]


def parse_prediction_line(line: str):
    parts = line.strip().split()
    if len(parts) not in (6,):  # expecting 6 fields
        return None
    try:
        cls_id = int(parts[0])
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        conf = float(parts[5])
        return cls_id, xc, yc, w, h, conf
    except ValueError:
        return None


def process_image(img_path: Path, labels_dir: Path, conf_thr: float, round_dp: int) -> Dict[str, Any]:
    img_w, img_h = load_image_size(img_path)
    label_file = labels_dir / f"{img_path.stem}.txt"
    annotations = []
    if label_file.exists():
        try:
            with label_file.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    if not raw_line.strip():
                        continue
                    parsed = parse_prediction_line(raw_line)
                    if not parsed:
                        print(f"[WARN] Skipping malformed line in {label_file}: {raw_line.strip()}", file=sys.stderr)
                        continue
                    cls_id, xc, yc, w, h, conf = parsed
                    if conf < conf_thr:
                        continue
                    if cls_id not in VALID_CLASS_IDS:
                        print(f"[WARN] Unknown class id {cls_id} in {label_file}", file=sys.stderr)
                        continue
                    bbox = yolo_to_bbox_abs(xc, yc, w, h, img_w, img_h)
                    if round_dp >= 0:
                        bbox = [round(v, round_dp) for v in bbox]
                    annotations.append({
                        "bbox": bbox,
                        "category_id": cls_id,
                        "category_name": CLASS_MAP[cls_id],
                    })
        except Exception as e:
            print(f"[ERROR] Failed reading {label_file}: {e}", file=sys.stderr)
    else:
        # No label file -> empty annotations list
        pass
    return {
        "file_name": img_path.name,
        "annotations": annotations,
        "corruption": {"type": "none", "severity": 0},
    }


def main():
    args = parse_args()
    images_dir: Path = args.images
    labels_dir: Path = args.labels
    output_dir: Path = args.output

    if not images_dir.exists():
        print(f"[ERROR] Images directory not found: {images_dir}")
        sys.exit(1)
    if not labels_dir.exists():
        print(f"[ERROR] Labels directory not found: {labels_dir}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        print(f"[WARN] No PNG images found in {images_dir}")
    first_json_path = None
    count = 0

    for img_path in image_paths:
        data = process_image(img_path, labels_dir, args.conf, args.round)
        json_path = output_dir / f"{img_path.stem}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if first_json_path is None:
            first_json_path = json_path
        count += 1

    print(f"Processed {count} images. JSON files saved in {output_dir.resolve() }.")
    if first_json_path:
        print(f"First JSON: {first_json_path.resolve()}")


if __name__ == "__main__":  # pragma: no cover
    main()
