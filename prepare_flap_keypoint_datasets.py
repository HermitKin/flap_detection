from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "object detection" / "datasets" / "flap"
DEFAULT_FULL = ROOT / "object detection" / "datasets" / "flap_keypoint_full"
DEFAULT_QUICK = ROOT / "object detection" / "datasets" / "flap_keypoint_quick300"
SEED = 20260820
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
QUICK_COUNTS = {"train": 210, "val": 60, "test": 30}
KPT_COUNT = 4


def read_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) < 5:
            raise ValueError(f"{path}:{line_number}: expected at least 5 fields")
        class_id = int(float(values[0]))
        coords = tuple(map(float, values[1:5]))
        if not all(math.isfinite(value) for value in coords):
            raise ValueError(f"{path}:{line_number}: non-finite box")
        boxes.append((class_id, *coords))
    return boxes


def frame_key(path: Path) -> tuple[str, int]:
    match = re.match(r"^(.*)_(\d+)$", path.stem)
    if not match:
        return path.stem, 0
    return match.group(1), int(match.group(2))


def make_scene_groups(images: list[Path], max_frame_gap: int = 2) -> dict[str, str]:
    """Keep consecutive frames together so near-duplicate scenes cannot cross splits."""
    by_prefix: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in images:
        prefix, frame = frame_key(path)
        by_prefix[prefix].append((frame, path))

    result = {}
    group_index = 0
    for prefix, items in sorted(by_prefix.items()):
        previous_frame = None
        for frame, path in sorted(items):
            if previous_frame is None or frame - previous_frame > max_frame_gap:
                group_index += 1
            result[path.name] = f"{prefix}__scene_{group_index:04d}"
            previous_frame = frame
    return result


def assign_group_splits(group_weights: dict[str, int]) -> dict[str, str]:
    """Greedily approach 70/20/10 while never splitting a scene group."""
    total = sum(group_weights.values())
    targets = {split: total * ratio for split, ratio in SPLIT_RATIOS.items()}
    counts = {split: 0 for split in SPLIT_RATIOS}
    rng = random.Random(SEED)
    groups = list(group_weights.items())
    rng.shuffle(groups)
    groups.sort(key=lambda item: item[1], reverse=True)
    assignments = {}
    for group_id, weight in groups:
        best_split = min(
            SPLIT_RATIOS,
            key=lambda split: (
                ((counts[split] + weight - targets[split]) / max(targets[split], 1.0)) ** 2
                - ((counts[split] - targets[split]) / max(targets[split], 1.0)) ** 2,
                counts[split] / max(targets[split], 1.0),
            ),
        )
        assignments[group_id] = best_split
        counts[best_split] += weight
    return assignments


def crop_box(
    image: np.ndarray,
    box: tuple[int, float, float, float, float],
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[float, float, float, float]]:
    _, cx, cy, bw, bh = box
    height, width = image.shape[:2]
    x1 = (cx - bw / 2) * width
    y1 = (cy - bh / 2) * height
    x2 = (cx + bw / 2) * width
    y2 = (cy + bh / 2) * height
    x1, x2 = sorted((float(np.clip(x1, 0, width)), float(np.clip(x2, 0, width))))
    y1, y2 = sorted((float(np.clip(y1, 0, height)), float(np.clip(y2, 0, height))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("degenerate GT box")

    pad_x = max(24.0, (x2 - x1) * 0.15)
    pad_y = max(12.0, (y2 - y1) * 0.05)
    crop_x1 = max(0, int(math.floor(x1 - pad_x)))
    crop_y1 = max(0, int(math.floor(y1 - pad_y)))
    crop_x2 = min(width, int(math.ceil(x2 + pad_x)))
    crop_y2 = min(height, int(math.ceil(y2 + pad_y)))
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    roi_x1 = max(0.0, x1 - crop_x1)
    roi_y1 = max(0.0, y1 - crop_y1)
    roi_x2 = min(float(crop.shape[1]), x2 - crop_x1)
    roi_y2 = min(float(crop.shape[0]), y2 - crop_y1)
    roi_cx = (roi_x1 + roi_x2) / (2 * crop.shape[1])
    roi_cy = (roi_y1 + roi_y2) / (2 * crop.shape[0])
    roi_w = (roi_x2 - roi_x1) / crop.shape[1]
    roi_h = (roi_y2 - roi_y1) / crop.shape[0]
    return crop, (crop_x1, crop_y1, crop_x2, crop_y2), (roi_cx, roi_cy, roi_w, roi_h)


def diversity_feature(crop: np.ndarray, source_box: tuple[int, float, float, float, float]) -> np.ndarray:
    """Compact appearance/geometry descriptor used only for diverse sampling."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (12, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 180]).ravel()
    hist_s = cv2.calcHist([hsv], [1], None, [6], [0, 256]).ravel()
    hist_v = cv2.calcHist([hsv], [2], None, [6], [0, 256]).ravel()
    histogram = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    histogram /= max(float(histogram.sum()), 1.0)
    edges = cv2.Canny(gray, 60, 150)
    edge_thumb = cv2.resize(edges, (6, 12), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    _, cx, cy, bw, bh = source_box
    geometry = np.array([
        cx, cy, bw, bh, bw / max(bh, 1e-6),
        crop.shape[1] / max(crop.shape[0], 1),
    ], dtype=np.float32)
    return np.concatenate([thumb.ravel(), edge_thumb.ravel(), histogram, geometry])


def pose_template(class_id: int, roi_box: tuple[float, float, float, float]) -> str:
    fields = [str(class_id), *(f"{value:.8f}" for value in roi_box)]
    fields.extend(["0", "0", "0"] * KPT_COUNT)
    return " ".join(fields) + "\n"


def write_yaml(root: Path) -> None:
    text = """path: .
train: images/train
val: images/val
test: images/test

kpt_shape: [4, 3]
flip_idx: [1, 0, 3, 2]

names:
  0: flap
"""
    (root / "data.yaml").write_text(text, encoding="utf-8")


def write_readme(root: Path, quick: bool) -> None:
    kind = "300-image diverse quick-validation subset" if quick else "complete GT-instance crop set"
    text = f"""# Flap four-keypoint annotation dataset

This is the {kind}, generated from detection GT without modifying the source dataset.

Each source GT instance becomes one padded ROI image. Splits are scene-grouped at 70/20/10;
consecutive source frames and all instances from one source image stay in the same split.

Keypoint order:
0. effective-range top-left
1. effective-range top-right
2. effective-range bottom-right
3. effective-range bottom-left

`labels` contains YOLO-Pose templates with four invisible `(0,0,0)` placeholders.
These files are NOT trainable until the four effective-range corner keypoints are manually annotated.
`detection_labels` preserves the GT box transformed into the crop coordinate system.
See `annotation_manifest.csv` for source provenance and crop coordinates.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def save_preview(rows: list[dict], output: Path, limit: int = 48) -> None:
    chosen = rows[:limit]
    tile_w, tile_h, columns = 220, 320, 8
    rows_count = math.ceil(len(chosen) / columns)
    sheet = Image.new("RGB", (tile_w * columns, tile_h * rows_count), "white")
    font = ImageFont.load_default()
    for index, row in enumerate(chosen):
        with Image.open(row["absolute_image_path"]) as opened:
            image = opened.convert("RGB")
        image.thumbnail((tile_w - 8, tile_h - 32))
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        tile.paste(image, ((tile_w - image.width) // 2, 26))
        draw = ImageDraw.Draw(tile)
        draw.text((4, 4), f"{row['split']} | {row['crop_image']}", fill="black", font=font)
        sheet.paste(tile, ((index % columns) * tile_w, (index // columns) * tile_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def write_manifest(rows: list[dict], path: Path) -> None:
    public_rows = [{key: value for key, value in row.items() if key not in {"feature", "absolute_image_path"}} for row in rows]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_rows[0].keys()))
        writer.writeheader()
        writer.writerows(public_rows)


def farthest_sample(rows: list[dict], count: int) -> list[dict]:
    """Select diverse ROIs while allowing at most one instance per source image."""
    if count == 0:
        return []
    if len({row["source_image"] for row in rows}) < count:
        raise ValueError(f"not enough unique source images for diverse sample of {count}")
    matrix = np.vstack([row["feature"] for row in rows]).astype(np.float32)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    matrix = np.clip((matrix - mean) / np.maximum(std, 1e-4), -5, 5)
    start = int(np.argmin(np.sum(matrix * matrix, axis=1)))
    selected_indices = [start]
    used_sources = {rows[start]["source_image"]}
    min_distance = np.sum((matrix - matrix[start]) ** 2, axis=1)
    while len(selected_indices) < count:
        scores = min_distance.copy()
        for index, row in enumerate(rows):
            if row["source_image"] in used_sources:
                scores[index] = -1
        next_index = int(np.argmax(scores))
        if scores[next_index] < 0:
            raise RuntimeError("diversity sampler exhausted unique sources")
        selected_indices.append(next_index)
        used_sources.add(rows[next_index]["source_image"])
        distance = np.sum((matrix - matrix[next_index]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
    return [rows[index] for index in selected_indices]


def copy_quick_subset(selected: list[dict], quick_root: Path) -> list[dict]:
    quick_rows = []
    for row in selected:
        split = row["split"]
        image_target = quick_root / "images" / split / row["crop_image"]
        pose_target = quick_root / "labels" / split / f"{Path(row['crop_image']).stem}.txt"
        detection_target = quick_root / "detection_labels" / split / f"{Path(row['crop_image']).stem}.txt"
        for target in (image_target, pose_target, detection_target):
            target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["absolute_image_path"], image_target)
        full_root = Path(row["absolute_image_path"]).parents[2]
        shutil.copy2(full_root / "labels" / split / pose_target.name, pose_target)
        shutil.copy2(full_root / "detection_labels" / split / detection_target.name, detection_target)
        copied = dict(row)
        copied["absolute_image_path"] = str(image_target.resolve())
        quick_rows.append(copied)
    return quick_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full and diverse-300 keypoint ROI datasets from detection GT.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--full-output", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--quick-output", type=Path, default=DEFAULT_QUICK)
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Regenerate files in the two known output directories without deleting the source dataset.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    full_root = args.full_output.resolve()
    quick_root = args.quick_output.resolve()
    for target in (full_root, quick_root):
        if target.exists() and not args.overwrite_existing:
            raise FileExistsError(f"refusing to overwrite existing output: {target}")

    image_dir, label_dir = source / "images", source / "labels"
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"no images found: {image_dir}")
    missing_labels = [path.name for path in images if not (label_dir / f"{path.stem}.txt").exists()]
    if missing_labels:
        raise FileNotFoundError(f"missing labels for {len(missing_labels)} images: {missing_labels[:5]}")

    scene_by_image = make_scene_groups(images)
    boxes_by_image = {path.name: read_boxes(label_dir / f"{path.stem}.txt") for path in images}
    group_weights: dict[str, int] = defaultdict(int)
    for image_name, boxes in boxes_by_image.items():
        group_weights[scene_by_image[image_name]] += len(boxes)
    group_splits = assign_group_splits(group_weights)

    rows = []
    errors = []
    for image_index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            errors.append({"source_image": image_path.name, "error": "cannot_read_image"})
            continue
        scene_group = scene_by_image[image_path.name]
        split = group_splits.get(scene_group, "train")
        for instance_index, box in enumerate(boxes_by_image[image_path.name]):
            try:
                crop, crop_xyxy, roi_box = crop_box(image, box)
            except ValueError as exc:
                errors.append({"source_image": image_path.name, "instance_index": instance_index, "error": str(exc)})
                continue
            crop_name = f"{image_path.stem}__i{instance_index:02d}.jpg"
            image_target = full_root / "images" / split / crop_name
            pose_target = full_root / "labels" / split / f"{Path(crop_name).stem}.txt"
            detection_target = full_root / "detection_labels" / split / pose_target.name
            for target in (image_target, pose_target, detection_target):
                target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(image_target), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"failed to save crop: {image_target}")
            class_id = box[0]
            pose_target.write_text(pose_template(class_id, roi_box), encoding="utf-8")
            detection_target.write_text(
                f"{class_id} " + " ".join(f"{value:.8f}" for value in roi_box) + "\n",
                encoding="utf-8",
            )
            rows.append({
                "crop_image": crop_name,
                "split": split,
                "source_image": image_path.name,
                "source_instance_index": instance_index,
                "scene_group": scene_group,
                "source_class": class_id,
                "source_bbox_cx": f"{box[1]:.8f}",
                "source_bbox_cy": f"{box[2]:.8f}",
                "source_bbox_w": f"{box[3]:.8f}",
                "source_bbox_h": f"{box[4]:.8f}",
                "crop_x1": crop_xyxy[0],
                "crop_y1": crop_xyxy[1],
                "crop_x2": crop_xyxy[2],
                "crop_y2": crop_xyxy[3],
                "annotation_status": "needs_4_keypoints",
                "feature": diversity_feature(crop, box),
                "absolute_image_path": str(image_target.resolve()),
            })
        if image_index % 200 == 0:
            print(f"processed source images: {image_index}/{len(images)}")

    if errors:
        (full_root / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    if not rows:
        raise RuntimeError("no valid GT instances were produced")
    write_manifest(rows, full_root / "annotation_manifest.csv")
    write_yaml(full_root)
    write_readme(full_root, quick=False)
    preview_rows = sorted(rows, key=lambda row: (row["split"], row["scene_group"], row["crop_image"]))
    save_preview(preview_rows, full_root / "preview.jpg")

    quick_selected = []
    for split, count in QUICK_COUNTS.items():
        candidates = [row for row in rows if row["split"] == split]
        quick_selected.extend(farthest_sample(candidates, count))
    quick_rows = copy_quick_subset(quick_selected, quick_root)
    quick_rows.sort(key=lambda row: (row["split"], row["crop_image"]))
    write_manifest(quick_rows, quick_root / "annotation_manifest.csv")
    write_yaml(quick_root)
    write_readme(quick_root, quick=True)
    save_preview(quick_rows, quick_root / "preview.jpg")

    source_empty = sum(not boxes for boxes in boxes_by_image.values())
    stats = {
        "source_images": len(images),
        "source_empty_labels": source_empty,
        "valid_roi_instances": len(rows),
        "scene_groups": len(set(scene_by_image.values())),
        "full_split_counts": dict(Counter(row["split"] for row in rows)),
        "full_unique_source_images": len({row["source_image"] for row in rows}),
        "quick_total": len(quick_rows),
        "quick_split_counts": dict(Counter(row["split"] for row in quick_rows)),
        "quick_unique_source_images": len({row["source_image"] for row in quick_rows}),
        "invalid_instances": len(errors),
        "seed": SEED,
    }
    for root in (full_root, quick_root):
        (root / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"full_output={full_root}")
    print(f"quick_output={quick_root}")


if __name__ == "__main__":
    main()
