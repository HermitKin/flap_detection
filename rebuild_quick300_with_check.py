from __future__ import annotations

import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from prepare_flap_keypoint_datasets import diversity_feature, farthest_sample


ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "object detection" / "datasets"
FULL = DATASETS / "flap_keypoint_full"
CHECK = DATASETS / "check"
QUICK = DATASETS / "flap_keypoint_quick300"
UNANNOTATED = DATASETS / "flap_keypoint_quick300_unannotated"
SEED = 20260820
SPLITS = ("train", "val", "test")
EXTRA_COUNTS = {"train": 83, "val": 60, "test": 30}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_by_stem(root: Path, pattern: str) -> dict[str, Path]:
    result = {}
    for path in sorted(root.rglob(pattern)):
        if path.stem in result:
            raise ValueError(f"duplicate stem under {root}: {path.stem}")
        result[path.stem] = path
    return result


def frame_number(stem: str) -> int | None:
    match = re.search(r"_(\d+)__i\d+$", stem)
    return int(match.group(1)) if match else None


def check_pose_label(path: Path) -> None:
    lines = [line.split() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines or any(len(line) != 17 for line in lines):
        raise ValueError(f"invalid four-keypoint label: {path}")


def allocate_group_pools(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["scene_group"]].append(row)
    items = list(groups.items())
    random.Random(SEED).shuffle(items)
    items.sort(key=lambda item: len({row["source_image"] for row in item[1]}), reverse=True)
    pools = {split: [] for split in SPLITS}
    for _, group in items:
        if all(len({row["source_image"] for row in pools[s]}) >= EXTRA_COUNTS[s] for s in SPLITS):
            break
        split = min(
            SPLITS,
            key=lambda name: len({row["source_image"] for row in pools[name]}) / EXTRA_COUNTS[name],
        )
        pools[split].extend(group)
    for split, count in EXTRA_COUNTS.items():
        unique_sources = len({row["source_image"] for row in pools[split]})
        if unique_sources < count:
            raise RuntimeError(f"insufficient {split} pool: {unique_sources} unique sources for {count}")
    return pools


def add_features(rows: list[dict]) -> None:
    for row in rows:
        image = cv2.imread(row["absolute_image_path"])
        if image is None:
            raise OSError(f"cannot read image: {row['absolute_image_path']}")
        source_box = (
            int(row["source_class"]),
            float(row["source_bbox_cx"]),
            float(row["source_bbox_cy"]),
            float(row["source_bbox_w"]),
            float(row["source_bbox_h"]),
        )
        row["feature"] = diversity_feature(image, source_box)


def build_temp_roots() -> tuple[Path, Path]:
    quick_temp = DATASETS / ".flap_keypoint_quick300_building"
    unannotated_temp = DATASETS / ".flap_keypoint_quick300_unannotated_building"
    for path in (quick_temp, unannotated_temp):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    return quick_temp, unannotated_temp


def copy_triplet(image: Path, pose: Path, detection: Path, root: Path, split: str) -> None:
    targets = (
        (image, root / "images" / split / image.name),
        (pose, root / "labels" / split / pose.name),
        (detection, root / "detection_labels" / split / detection.name),
    )
    for source, target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def replace_directory(temp: Path, target: Path) -> None:
    if target.parent.resolve() != DATASETS.resolve() or target.name not in {
        "flap_keypoint_quick300", "flap_keypoint_quick300_unannotated"
    }:
        raise ValueError(f"unsafe replacement target: {target}")
    backup = target.with_name(f".{target.name}_backup")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        target.rename(backup)
    temp.rename(target)
    if backup.exists():
        shutil.rmtree(backup)


def main() -> None:
    full_rows = read_manifest(FULL / "annotation_manifest.csv")
    full_by_stem = {Path(row["crop_image"]).stem: row for row in full_rows}
    full_images = collect_by_stem(FULL / "images", "*.jpg")
    full_pose = collect_by_stem(FULL / "labels", "*.txt")
    full_detection = collect_by_stem(FULL / "detection_labels", "*.txt")
    check_images = collect_by_stem(CHECK / "images", "*.jpg")
    check_pose = collect_by_stem(CHECK / "labels", "*.txt")
    if set(check_images) != set(check_pose):
        raise ValueError("check images and labels do not match")
    for path in check_pose.values():
        check_pose_label(path)

    check_stems = set(check_images)
    matched_check = check_stems & set(full_by_stem)
    forced_groups = {full_by_stem[stem]["scene_group"] for stem in matched_check}
    missing_frames = {frame_number(stem) for stem in check_stems - matched_check}
    missing_frames.discard(None)
    candidates = []
    for row in full_rows:
        stem = Path(row["crop_image"]).stem
        number = frame_number(stem)
        if stem in check_stems or row["scene_group"] in forced_groups:
            continue
        if number is not None and any(abs(number - value) <= 2 for value in missing_frames):
            continue
        candidate = dict(row)
        candidate["absolute_image_path"] = str(full_images[stem])
        candidates.append(candidate)

    pools = allocate_group_pools(candidates)
    selected_extra = {}
    for split in SPLITS:
        add_features(pools[split])
        selected_extra[split] = farthest_sample(pools[split], EXTRA_COUNTS[split])

    quick_temp, unannotated_temp = build_temp_roots()
    manifest_fields = list(full_rows[0]) + ["pose_annotation_status"]
    quick_rows: list[dict[str, str]] = []
    unannotated_rows: list[dict[str, str]] = []

    for stem in sorted(check_stems):
        pose_path = check_pose[stem]
        if stem in full_detection:
            detection_path = full_detection[stem]
        else:
            detection_path = quick_temp / ".derived_detection" / f"{stem}.txt"
            detection_path.parent.mkdir(parents=True, exist_ok=True)
            first = pose_path.read_text(encoding="utf-8-sig").splitlines()[0].split()[:5]
            detection_path.write_text(" ".join(first) + "\n", encoding="utf-8")
        copy_triplet(check_images[stem], pose_path, detection_path, quick_temp, "train")
        base = dict(full_by_stem.get(stem, {}))
        base.update({
            "crop_image": check_images[stem].name,
            "split": "train",
            "source_image": base.get("source_image", f"{stem.split('__i')[0]}.jpg"),
            "source_instance_index": base.get("source_instance_index", stem.rsplit("__i", 1)[-1]),
            "scene_group": base.get("scene_group", f"check_only_{stem.split('__i')[0]}"),
            "source_class": "0",
            "annotation_status": "annotated_4_keypoints",
            "pose_annotation_status": "annotated_check",
        })
        quick_rows.append({field: base.get(field, "") for field in manifest_fields})

    for split in SPLITS:
        for row in selected_extra[split]:
            stem = Path(row["crop_image"]).stem
            copy_triplet(full_images[stem], full_pose[stem], full_detection[stem], quick_temp, split)
            clean = {field: row.get(field, "") for field in manifest_fields}
            clean["split"] = split
            clean["annotation_status"] = "needs_4_keypoints"
            clean["pose_annotation_status"] = "unannotated_placeholder"
            quick_rows.append(clean)
            copy_triplet(full_images[stem], full_pose[stem], full_detection[stem], unannotated_temp, split)
            unannotated_rows.append(dict(clean))

    quick_rows.sort(key=lambda row: (SPLITS.index(row["split"]), row["crop_image"]))
    unannotated_rows.sort(key=lambda row: (SPLITS.index(row["split"]), row["crop_image"]))
    write_manifest(quick_rows, quick_temp / "annotation_manifest.csv")
    write_manifest(unannotated_rows, unannotated_temp / "annotation_manifest.csv")
    for root in (quick_temp, unannotated_temp):
        shutil.copy2(FULL / "data.yaml", root / "data.yaml")

    quick_counts = Counter(row["split"] for row in quick_rows)
    unannotated_counts = Counter(row["split"] for row in unannotated_rows)
    stats = {
        "quick_total": len(quick_rows),
        "quick_split_counts": dict(quick_counts),
        "check_in_quick_train": len(check_stems),
        "check_missing_from_full_but_included_from_check": len(check_stems - set(full_by_stem)),
        "unannotated_total": len(unannotated_rows),
        "unannotated_split_counts": dict(unannotated_counts),
        "check_in_unannotated": 0,
        "seed": SEED,
    }
    for root in (quick_temp, unannotated_temp):
        (root / "selection_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (quick_temp / "README.md").write_text(
        "# Quick300 with check fixed in train\n\n"
        "All 127 manually annotated check images are in train. Train/val/test contains 210/60/30 images.\n"
        "Only check-derived pose labels are real; the other 173 labels are invisible placeholders.\n",
        encoding="utf-8",
    )
    (unannotated_temp / "README.md").write_text(
        "# Unannotated part of Quick300\n\n"
        "This contains the 173 Quick300 images not present in check, preserving their Quick300 splits.\n"
        "Pose labels are invisible placeholders and require annotation.\n",
        encoding="utf-8",
    )

    if quick_counts != Counter({"train": 210, "val": 60, "test": 30}):
        raise RuntimeError(f"unexpected quick split counts: {quick_counts}")
    if len(unannotated_rows) != 173:
        raise RuntimeError(f"unexpected unannotated count: {len(unannotated_rows)}")
    replace_directory(quick_temp, QUICK)
    replace_directory(unannotated_temp, UNANNOTATED)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"quick={QUICK}")
    print(f"unannotated={UNANNOTATED}")


if __name__ == "__main__":
    main()
