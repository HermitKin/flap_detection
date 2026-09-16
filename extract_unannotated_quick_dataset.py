from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECK = ROOT / "object detection" / "datasets" / "check"
QUICK = ROOT / "object detection" / "datasets" / "flap_keypoint_quick300"
OUTPUT = ROOT / "object detection" / "datasets" / "flap_keypoint_quick300_unannotated"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def image_map(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in result:
            raise ValueError(f"duplicate image stem: {path.stem}")
        result[path.stem] = path
    return result


def label_map(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*.txt")):
        if path.stem in result:
            raise ValueError(f"duplicate label stem: {path.stem}")
        result[path.stem] = path
    return result


def validate_pose_label(path: Path) -> int:
    lines = [line.split() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty pose label: {path}")
    for line_number, fields in enumerate(lines, start=1):
        if len(fields) != 17:  # class + bbox(4) + four (x,y,visibility) keypoints
            raise ValueError(f"{path}:{line_number}: expected 17 fields, got {len(fields)}")
        values = [float(value) for value in fields]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: non-finite value")
        if int(values[0]) != 0:
            raise ValueError(f"{path}:{line_number}: expected class 0, got {values[0]}")
        normalized = values[1:5] + [value for index, value in enumerate(values[5:]) if index % 3 != 2]
        if any(value < 0 or value > 1 for value in normalized):
            raise ValueError(f"{path}:{line_number}: normalized coordinate outside [0,1]")
        visibility = values[7::3]
        if any(value not in (0, 1, 2) for value in visibility):
            raise ValueError(f"{path}:{line_number}: visibility must be 0, 1, or 2")
    return len(lines)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")

    check_images = image_map(CHECK / "images")
    check_labels = label_map(CHECK / "labels")
    missing_check_labels = sorted(set(check_images) - set(check_labels))
    orphan_check_labels = sorted(set(check_labels) - set(check_images))
    if missing_check_labels or orphan_check_labels:
        raise ValueError(
            f"check pairing error: missing_labels={missing_check_labels[:5]}, "
            f"orphan_labels={orphan_check_labels[:5]}"
        )
    for path in check_labels.values():
        validate_pose_label(path)

    quick_images: dict[str, tuple[str, Path]] = {}
    for split in SPLITS:
        for stem, path in image_map(QUICK / "images" / split).items():
            if stem in quick_images:
                raise ValueError(f"quick image occurs in multiple splits: {stem}")
            quick_images[stem] = (split, path)

    quick_pose = label_map(QUICK / "labels")
    quick_detection = label_map(QUICK / "detection_labels")
    if set(quick_images) != set(quick_pose) or set(quick_images) != set(quick_detection):
        raise ValueError("quick300 image/pose-template/detection-label stems do not match")

    annotated_overlap = sorted(set(quick_images) & set(check_labels))
    remaining = sorted(set(quick_images) - set(check_labels))
    for stem in remaining:
        split, source_image = quick_images[stem]
        destinations = (
            (source_image, OUTPUT / "images" / split / source_image.name),
            (quick_pose[stem], OUTPUT / "labels" / split / quick_pose[stem].name),
            (quick_detection[stem], OUTPUT / "detection_labels" / split / quick_detection[stem].name),
        )
        for source, destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    manifest_path = QUICK / "annotation_manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    filtered_rows = [row for row in rows if Path(row["crop_image"]).stem in remaining]
    if len(filtered_rows) != len(remaining):
        raise ValueError("filtered manifest does not match remaining images")
    with (OUTPUT / "annotation_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(filtered_rows[0]))
        writer.writeheader()
        writer.writerows(filtered_rows)

    shutil.copy2(QUICK / "data.yaml", OUTPUT / "data.yaml")
    split_counts = Counter(quick_images[stem][0] for stem in remaining)
    stats = {
        "check_images_with_valid_pose_labels": len(check_images),
        "check_images_also_in_quick300": len(annotated_overlap),
        "check_images_not_in_quick300": len(check_images) - len(annotated_overlap),
        "quick300_images": len(quick_images),
        "remaining_unannotated_images": len(remaining),
        "remaining_split_counts": {split: split_counts[split] for split in SPLITS},
        "excluded_annotated_stems": annotated_overlap,
    }
    (OUTPUT / "selection_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# Unannotated remainder of flap_keypoint_quick300\n\n"
        "Images whose stems have valid four-keypoint labels in `../check` were excluded.\n"
        "The original quick300 train/val/test assignment is preserved.\n"
        "`labels` still contains invisible placeholder labels and must be replaced by manual annotations.\n"
        "`detection_labels` contains the transformed detection GT boxes.\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
