from __future__ import annotations

import csv
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE = PROJECT_ROOT / "derived" / "pose_annotation_test"
OUTPUT = PROJECT_ROOT / "derived" / "pose_keypoint_crops"
IMAGES = SOURCE / "images"
LABELS = SOURCE / "detection_prelabels"
MANIFEST = SOURCE / "annotation_manifest.csv"
SEED = 20260818


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def read_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = line.split()
        boxes.append((int(values[0]), *map(float, values[1:5])))
    return boxes


def assign_splits(rows: list[dict[str, str]]) -> dict[str, str]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("near_duplicate_group") or row["image"]].append(row)
    items = list(grouped.items())
    random.Random(SEED).shuffle(items)
    target_train = round(sum(len(group) for _, group in items) * 0.8)
    train_count = 0
    result = {}
    for group_id, group in items:
        split = "train" if train_count < target_train else "val"
        result[group_id] = split
        if split == "train":
            train_count += len(group)
    return result


def main() -> None:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    by_image = {row["image"]: row for row in source_rows}
    group_splits = assign_splits(source_rows)

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    (OUTPUT / "images").mkdir(parents=True)
    (OUTPUT / "pose_labels_template").mkdir(parents=True)
    (OUTPUT / "previews").mkdir(parents=True)

    crop_rows = []
    preview_items = []
    for image_path in sorted(IMAGES.glob("*.jpg")):
        source_meta = by_image[image_path.name]
        group_id = source_meta.get("near_duplicate_group") or image_path.name
        split = group_splits[group_id]
        boxes = read_boxes(LABELS / f"{image_path.stem}.txt")
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            for instance_index, (class_id, cx, cy, bw, bh) in enumerate(boxes):
                x1 = (cx - bw / 2) * image.width
                y1 = (cy - bh / 2) * image.height
                x2 = (cx + bw / 2) * image.width
                y2 = (cy + bh / 2) * image.height
                pad_x = max(32.0, (x2 - x1) * 0.15)
                pad_y = max(16.0, (y2 - y1) * 0.05)
                crop_x1 = max(0, int(x1 - pad_x))
                crop_y1 = max(0, int(y1 - pad_y))
                crop_x2 = min(image.width, int(x2 + pad_x + 0.999))
                crop_y2 = min(image.height, int(y2 + pad_y + 0.999))
                crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                crop_name = f"{image_path.stem}__i{instance_index:02d}.jpg"
                crop_path = OUTPUT / "images" / split / crop_name
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_path, quality=95, subsampling=0)

                roi_x1 = max(0.0, x1 - crop_x1)
                roi_y1 = max(0.0, y1 - crop_y1)
                roi_x2 = min(float(crop.width), x2 - crop_x1)
                roi_y2 = min(float(crop.height), y2 - crop_y1)
                roi_cx = (roi_x1 + roi_x2) / 2 / crop.width
                roi_cy = (roi_y1 + roi_y2) / 2 / crop.height
                roi_w = (roi_x2 - roi_x1) / crop.width
                roi_h = (roi_y2 - roi_y1) / crop.height

                # YOLO Pose: class bbox + K0(TL), K1(TR), K2(BR), K3(BL), K4(level).
                # All keypoints start as invisible placeholders and must be annotated.
                template = [str(class_id), f"{roi_cx:.8f}", f"{roi_cy:.8f}", f"{roi_w:.8f}", f"{roi_h:.8f}"]
                template.extend(["0", "0", "0"] * 5)
                label_path = OUTPUT / "pose_labels_template" / split / f"{Path(crop_name).stem}.txt"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(" ".join(template) + "\n", encoding="utf-8")

                crop_rows.append({
                    "crop_image": crop_name,
                    "split_candidate": split,
                    "source_image": image_path.name,
                    "source_instance_index": instance_index,
                    "near_duplicate_group": group_id,
                    "crop_x1": crop_x1,
                    "crop_y1": crop_y1,
                    "crop_x2": crop_x2,
                    "crop_y2": crop_y2,
                    "source_bbox_x": f"{cx:.8f}",
                    "source_bbox_y": f"{cy:.8f}",
                    "source_bbox_w": f"{bw:.8f}",
                    "source_bbox_h": f"{bh:.8f}",
                    "k0_top_left_x": "",
                    "k0_top_left_y": "",
                    "k0_visibility": "",
                    "k1_top_right_x": "",
                    "k1_top_right_y": "",
                    "k1_visibility": "",
                    "k2_bottom_right_x": "",
                    "k2_bottom_right_y": "",
                    "k2_visibility": "",
                    "k3_bottom_left_x": "",
                    "k3_bottom_left_y": "",
                    "k3_visibility": "",
                    "k4_level_x": "",
                    "k4_level_y": "",
                    "k4_visibility": "",
                    "display_color_pair": "",
                    "lighting": "",
                    "occlusion": "",
                    "view_angle": "",
                    "review_status": "needs_annotation",
                })
                if len(preview_items) < 36:
                    preview_items.append((crop_name, crop.copy(), (roi_x1, roi_y1, roi_x2, roi_y2)))

    fields = list(crop_rows[0].keys())
    with (OUTPUT / "annotation_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(crop_rows)

    tile_w, tile_h = 240, 360
    sheet = Image.new("RGB", (tile_w * 6, tile_h * 6), "white")
    font = ImageFont.load_default()
    for index, (name, crop, box) in enumerate(preview_items):
        draw = ImageDraw.Draw(crop)
        draw.rectangle(box, outline="lime", width=max(2, crop.width // 100))
        crop.thumbnail((tile_w - 8, tile_h - 28))
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        tile.paste(crop, ((tile_w - crop.width) // 2, 22))
        ImageDraw.Draw(tile).text((4, 4), name, fill="black", font=font)
        sheet.paste(tile, ((index % 6) * tile_w, (index // 6) * tile_h))
    sheet.save(OUTPUT / "previews" / "crop_preview.jpg", quality=92)

    readme = """# Five-keypoint ROI annotation dataset

Each image is a clean high-resolution ROI cropped from a ground-truth detection box.
The green boxes in `previews/crop_preview.jpg` are preview-only and are not present in training images.

Keypoint order (clockwise corners):
0. K0 top-left
1. K1 top-right
2. K2 bottom-right
3. K3 bottom-left
4. K4 intersection of the flap-state boundary and the gauge centerline

Visibility: 2=visible, 1=occluded but reliably located, 0=unknown. Never guess an unknown point.
`pose_labels_template` contains zero/invisible placeholders; it is NOT trainable until annotation is complete.
The source set was formerly a detection test subset. Once used for pose training it must not be reported as a blind test set.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    counts = Counter(row["split_candidate"] for row in crop_rows)
    print(f"source_images={len(list(IMAGES.glob('*.jpg')))}")
    print(f"roi_instances={len(crop_rows)}")
    print(f"split_candidates={dict(counts)}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
