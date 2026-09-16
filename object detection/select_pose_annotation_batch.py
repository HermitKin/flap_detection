from __future__ import annotations

import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "datasets" / "flap_data"
IMAGES = ROOT / "all_images"
LABELS = ROOT / "derived" / "detection_labels_clipped"
OUT = ROOT / "derived"
BATCH = OUT / "annotation_batch_250"
BATCH_SIZE = 250
NEAR_DUPLICATE_MAD = 0.03
SEED = 42


def read_boxes(stem: str) -> list[tuple[float, float, float, float]]:
    boxes = []
    for line in (LABELS / f"{stem}.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            _, x, y, w, h = map(float, line.split())
            boxes.append((x, y, w, h))
    return boxes


def fps(features: np.ndarray, count: int, initial: list[int] | None = None) -> list[int]:
    selected = list(dict.fromkeys(initial or []))
    if not selected:
        selected = [int(np.argmax(np.sum((features - features.mean(axis=0)) ** 2, axis=1)))]
    min_dist = np.full(len(features), np.inf, dtype=np.float32)
    for idx in selected:
        min_dist = np.minimum(min_dist, np.sum((features - features[idx]) ** 2, axis=1))
    min_dist[selected] = -1
    while len(selected) < min(count, len(features)):
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        min_dist = np.minimum(min_dist, np.sum((features - features[idx]) ** 2, axis=1))
        min_dist[selected] = -1
    return selected


def main() -> None:
    image_paths = sorted(IMAGES.glob("*.jpg"))
    records = []
    raw_visual = []
    previous_gray = None
    group_id = 0
    previous_size = None

    for image_path in image_paths:
        with Image.open(image_path) as image:
            size = image.size
            rgb = np.asarray(image.convert("RGB").resize((16, 16)), dtype=np.float32) / 255.0
        gray = rgb.mean(axis=2)
        adjacent_mad = None if previous_gray is None else float(np.mean(np.abs(gray - previous_gray)))
        if previous_gray is not None and (size != previous_size or adjacent_mad > NEAR_DUPLICATE_MAD):
            group_id += 1
        boxes = read_boxes(image_path.stem)
        box_feature = np.mean(boxes, axis=0).tolist() if boxes else [0.0] * 4
        records.append({
            "image": image_path.name,
            "width": size[0],
            "height": size[1],
            "box_count": len(boxes),
            "near_duplicate_group": f"ndg_{group_id:04d}",
            "adjacent_thumbnail_mad": "" if adjacent_mad is None else f"{adjacent_mad:.6f}",
            "scene_id_manual": "",
            "split_manual": "",
        })
        geom = np.asarray([
            size[0] / max(size), size[1] / max(size), np.log2(max(size) / 512),
            len(boxes) / 3, *box_feature,
        ], dtype=np.float32)
        raw_visual.append(np.concatenate((rgb.reshape(-1), geom)))
        previous_gray, previous_size = gray, size

    with (OUT / "scene_group_candidates.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    positive_indices = [i for i, r in enumerate(records) if r["box_count"] > 0]
    features = np.stack([raw_visual[i] for i in positive_indices])
    rng = np.random.default_rng(SEED)
    projection = rng.normal(size=(features.shape[1], 48)).astype(np.float32)
    features = features @ projection
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)

    resolution_seeds = []
    seen_sizes = set()
    for local_idx, global_idx in enumerate(positive_indices):
        key = (records[global_idx]["width"], records[global_idx]["height"])
        if key not in seen_sizes:
            resolution_seeds.append(local_idx)
            seen_sizes.add(key)
    multi = [i for i, global_idx in enumerate(positive_indices) if records[global_idx]["box_count"] > 1]
    if multi:
        multi_features = features[multi]
        multi_picks = fps(multi_features, min(30, len(multi)))
        resolution_seeds.extend(multi[i] for i in multi_picks)
    selected_local = fps(features, BATCH_SIZE, resolution_seeds)
    selected_global = [positive_indices[i] for i in selected_local]

    image_out = BATCH / "images"
    label_out = BATCH / "detection_prelabels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    selected_rows = []
    for idx in selected_global:
        record = records[idx].copy()
        source_image = IMAGES / record["image"]
        target_image = image_out / source_image.name
        if not target_image.exists():
            try:
                os.link(source_image, target_image)
            except OSError:
                shutil.copy2(source_image, target_image)
        source_label = LABELS / f"{source_image.stem}.txt"
        shutil.copy2(source_label, label_out / source_label.name)
        selected_rows.append(record)

    with (BATCH / "batch_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(selected_rows)

    preview_rows = [selected_rows[i] for i in np.linspace(0, len(selected_rows) - 1, 25, dtype=int)]
    cell_w, cell_h = 384, 240
    preview = Image.new("RGB", (cell_w * 5, cell_h * 5), "white")
    for slot, record in enumerate(preview_rows):
        image_path = image_out / record["image"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            for x, y, w, h in read_boxes(image_path.stem):
                draw.rectangle(
                    ((x - w / 2) * image.width, (y - h / 2) * image.height,
                     (x + w / 2) * image.width, (y + h / 2) * image.height),
                    outline="red", width=max(2, image.width // 600),
                )
            image.thumbnail((cell_w, cell_h - 20))
            tile = Image.new("RGB", (cell_w, cell_h), "white")
            tile.paste(image, ((cell_w - image.width) // 2, 18))
            ImageDraw.Draw(tile).text((4, 2), record["image"], fill="black", font=ImageFont.load_default())
            preview.paste(tile, ((slot % 5) * cell_w, (slot // 5) * cell_h))
    preview.save(BATCH / "batch_preview.jpg", quality=92)

    group_sizes = defaultdict(int)
    for record in records:
        group_sizes[record["near_duplicate_group"]] += 1
    print(f"images={len(records)}")
    print(f"near_duplicate_groups={len(group_sizes)}, largest_group={max(group_sizes.values())}")
    print(f"annotation_batch={len(selected_rows)}, multi_target={sum(r['box_count'] > 1 for r in selected_rows)}")
    print(f"batch_dir={BATCH}")


if __name__ == "__main__":
    main()
