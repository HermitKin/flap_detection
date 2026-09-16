from __future__ import annotations

import csv
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat


ROOT = Path(__file__).resolve().parent / "datasets" / "flap_data"
IMAGES = ROOT / "all_images"
LABELS = (
    ROOT / "derived" / "detection_labels_clipped"
    if (ROOT / "derived" / "detection_labels_clipped").exists()
    else ROOT / "all_labels"
)
GROUPS_CSV = ROOT / "derived" / "scene_group_candidates.csv"
SPLIT_ROOT = ROOT / "split_7_2_1"
POSE_TEST = ROOT / "derived" / "pose_annotation_test"
RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
SEED = 20260818
NEAR_DUPLICATE_MAD = 0.03


def read_boxes(image_name: str) -> list[tuple[float, float, float, float]]:
    path = LABELS / f"{Path(image_name).stem}.txt"
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            _, x, y, w, h = map(float, line.split()[:5])
            boxes.append((x, y, w, h))
    return boxes


def visual_features(image_name: str, boxes: list[tuple[float, float, float, float]]) -> dict[str, object]:
    with Image.open(IMAGES / image_name) as image:
        rgb = image.convert("RGB")
        thumb = rgb.resize((128, 128))
        hsv = np.asarray(thumb.convert("HSV"), dtype=np.float32)
        gray = thumb.convert("L")
        brightness = float(ImageStat.Stat(gray).mean[0] / 255)
        saturation = float(hsv[..., 1].mean() / 255)
        hue = float(hsv[..., 0].mean() / 255)
        edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        sharpness = float(edges.var() / (255**2))
        highlight = float((np.asarray(gray) > 242).mean())
    if boxes:
        ratios = [h / max(w, 1e-9) for _, _, w, h in boxes]
        areas = [w * h for _, _, w, h in boxes]
        ratio = float(np.median(ratios))
        orientation = "vertical" if ratio > 1.5 else "horizontal" if ratio < 0.67 else "inclined_or_square"
        mean_area = float(np.mean(areas))
    else:
        orientation, mean_area = "negative", 0.0
    return {
        "brightness": brightness,
        "saturation": saturation,
        "hue": hue,
        "sharpness": sharpness,
        "highlight_ratio": highlight,
        "orientation": orientation,
        "distance_hint": "far" if boxes and mean_area < 0.012 else "near_or_medium",
        "target_count": len(boxes),
    }


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def load_or_build_group_rows() -> list[dict[str, str]]:
    if GROUPS_CSV.exists():
        with GROUPS_CSV.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    base_rows = []
    previous_gray = None
    previous_size = None
    group_id = 0
    for image_path in sorted(IMAGES.glob("*.jpg")):
        with Image.open(image_path) as image:
            size = image.size
            gray = np.asarray(image.convert("L").resize((16, 16)), dtype=np.float32) / 255
        adjacent_mad = None if previous_gray is None else float(np.mean(np.abs(gray - previous_gray)))
        if previous_gray is not None and (size != previous_size or adjacent_mad > NEAR_DUPLICATE_MAD):
            group_id += 1
        base_rows.append({
            "image": image_path.name,
            "width": str(size[0]),
            "height": str(size[1]),
            "box_count": str(len(read_boxes(image_path.name))),
            "near_duplicate_group": f"ndg_{group_id:04d}",
            "adjacent_thumbnail_mad": "" if adjacent_mad is None else f"{adjacent_mad:.6f}",
            "scene_id_manual": "",
            "split_manual": "",
        })
        previous_gray, previous_size = gray, size
    return base_rows


def group_stratum(rows: list[dict[str, object]]) -> tuple[object, ...]:
    positive = [row for row in rows if int(row["target_count"]) > 0]
    sample = positive or rows
    mean = lambda key: float(np.mean([float(row[key]) for row in sample]))
    return (
        any(int(row["target_count"]) > 1 for row in sample),
        any(row["distance_hint"] == "far" for row in sample),
        min(3, int(mean("brightness") * 4)),
        min(2, int(mean("saturation") * 3)),
        min(2, int(mean("hue") * 3)),
        min(2, int(mean("sharpness") * 180)),
        min(2, int(mean("highlight_ratio") * 15)),
    )


def choose_split(groups: dict[str, list[dict[str, object]]]) -> dict[str, str]:
    total = sum(len(rows) for rows in groups.values())
    targets = {split: total * ratio for split, ratio in RATIOS.items()}
    counts = Counter()
    assignment: dict[str, str] = {}
    rng = random.Random(SEED)
    items = list(groups.items())
    rng.shuffle(items)
    stratum_totals = Counter(group_stratum(rows) for _, rows in items)
    stratum_counts: dict[str, Counter] = {split: Counter() for split in RATIOS}
    # Allocate rare visual/geometry strata first, while also respecting image totals.
    items.sort(key=lambda item: (stratum_totals[group_stratum(item[1])], -len(item[1])))
    for group, rows in items:
        size = len(rows)
        stratum = group_stratum(rows)
        split = max(
            RATIOS,
            key=lambda name: (
                2 * (targets[name] - counts[name]) / max(targets[name], 1)
                + (stratum_totals[stratum] * RATIOS[name] - stratum_counts[name][stratum])
                / max(stratum_totals[stratum] * RATIOS[name], 1)
            ),
        )
        assignment[group] = split
        counts[split] += size
        stratum_counts[split][stratum] += 1
    return assignment


def main() -> None:
    base_rows = load_or_build_group_rows()
    rows = []
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for base in base_rows:
        boxes = read_boxes(base["image"])
        row: dict[str, object] = {**base, **visual_features(base["image"], boxes)}
        rows.append(row)
        groups[base["near_duplicate_group"]].append(row)

    assignment = choose_split(groups)
    for row in rows:
        row["split"] = assignment[str(row["near_duplicate_group"])]

    # Recreate only generated split products; original images and labels remain untouched.
    if SPLIT_ROOT.exists():
        shutil.rmtree(SPLIT_ROOT)
    if POSE_TEST.exists():
        shutil.rmtree(POSE_TEST)

    for row in rows:
        split = str(row["split"])
        image_name = str(row["image"])
        stem = Path(image_name).stem
        link_or_copy(IMAGES / image_name, SPLIT_ROOT / "images" / split / image_name)
        link_or_copy(LABELS / f"{stem}.txt", SPLIT_ROOT / "labels" / split / f"{stem}.txt")

    fields = list(rows[0].keys())
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    with (SPLIT_ROOT / "split_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    yaml_text = """path: /workspace/datasets/flap_data/split_7_2_1
train: images/train
val: images/val
test: images/test

names:
  0: flap
"""
    (SPLIT_ROOT / "data.yaml").write_text(yaml_text, encoding="utf-8")

    test_rows = [row for row in rows if row["split"] == "test" and int(row["target_count"]) > 0]
    for row in test_rows:
        image_name = str(row["image"])
        stem = Path(image_name).stem
        link_or_copy(IMAGES / image_name, POSE_TEST / "images" / image_name)
        link_or_copy(LABELS / f"{stem}.txt", POSE_TEST / "detection_prelabels" / f"{stem}.txt")
    POSE_TEST.mkdir(parents=True, exist_ok=True)
    pose_fields = fields + ["display_color_pair_manual", "lighting_manual", "occlusion_manual", "pose_review_status"]
    with (POSE_TEST / "annotation_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pose_fields)
        writer.writeheader()
        for row in test_rows:
            writer.writerow({
                **row,
                "display_color_pair_manual": "",
                "lighting_manual": "",
                "occlusion_manual": "",
                "pose_review_status": "needs_5_keypoints",
            })

    split_counts = Counter(str(row["split"]) for row in rows)
    group_sets = {
        split: {str(row["near_duplicate_group"]) for row in rows if row["split"] == split}
        for split in RATIOS
    }
    assert not (group_sets["train"] & group_sets["val"])
    assert not (group_sets["train"] & group_sets["test"])
    assert not (group_sets["val"] & group_sets["test"])
    print(f"images={len(rows)}, groups={len(groups)}")
    print("split_counts=" + ", ".join(f"{key}:{split_counts[key]}" for key in RATIOS))
    print(f"pose_test_images={len(test_rows)}")
    print("test_orientation=" + str(Counter(str(row["orientation"]) for row in test_rows)))
    print("test_distance=" + str(Counter(str(row["distance_hint"]) for row in test_rows)))
    print("test_target_count=" + str(Counter("multi" if int(row["target_count"]) > 1 else "single" for row in test_rows)))
    print("group_leakage=0")


if __name__ == "__main__":
    main()
