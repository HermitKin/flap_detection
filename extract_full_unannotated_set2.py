from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FULL = ROOT / "object detection" / "datasets" / "flap_keypoint_full"
SET2 = ROOT / "object detection" / "datasets" / "set2"
OUTPUT = ROOT / "object detection" / "datasets" / "flap_keypoint_full_unannotated"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def unique_files(root: Path, extensions: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stem in result:
            raise ValueError(f"duplicate stem {path.stem}: {result[path.stem]} and {path}")
        result[path.stem] = path
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")

    full_images: dict[str, tuple[str, Path]] = {}
    full_pose: dict[str, Path] = {}
    full_detection: dict[str, Path] = {}
    for split in SPLITS:
        for stem, path in unique_files(FULL / "images" / split, IMAGE_EXTENSIONS).items():
            if stem in full_images:
                raise ValueError(f"image occurs in multiple splits: {stem}")
            full_images[stem] = (split, path)
        for stem, path in unique_files(FULL / "labels" / split, {".txt"}).items():
            if stem in full_pose:
                raise ValueError(f"pose label occurs in multiple splits: {stem}")
            full_pose[stem] = path
        for stem, path in unique_files(FULL / "detection_labels" / split, {".txt"}).items():
            if stem in full_detection:
                raise ValueError(f"detection label occurs in multiple splits: {stem}")
            full_detection[stem] = path

    if set(full_images) != set(full_pose) or set(full_images) != set(full_detection):
        raise ValueError("full image, pose-label, and detection-label stems do not match")

    set2_labels = unique_files(SET2 / "labels", {".txt"})
    annotated_overlap = sorted(set(full_images) & set(set2_labels))
    remaining = sorted(set(full_images) - set(set2_labels))

    for stem in remaining:
        split, image = full_images[stem]
        for source, subdir in (
            (image, "images"),
            (full_pose[stem], "labels"),
            (full_detection[stem], "detection_labels"),
        ):
            destination = OUTPUT / subdir / split / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    with (FULL / "annotation_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    filtered_rows = [row for row in rows if Path(row["crop_image"]).stem in set(remaining)]
    if len(filtered_rows) != len(remaining):
        raise ValueError(
            f"filtered manifest mismatch: rows={len(filtered_rows)}, remaining={len(remaining)}"
        )
    with (OUTPUT / "annotation_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(filtered_rows[0]))
        writer.writeheader()
        writer.writerows(filtered_rows)

    shutil.copy2(FULL / "data.yaml", OUTPUT / "data.yaml")
    split_counts = Counter(full_images[stem][0] for stem in remaining)
    stats = {
        "full_images": len(full_images),
        "set2_label_files": len(set2_labels),
        "excluded_annotated_images_in_full": len(annotated_overlap),
        "set2_labels_not_in_full": len(set(set2_labels) - set(full_images)),
        "remaining_unannotated_images": len(remaining),
        "remaining_split_counts": {split: split_counts[split] for split in SPLITS},
        "excluded_annotated_stems": annotated_overlap,
    }
    (OUTPUT / "selection_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
