from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "object detection" / "datasets"
SET2 = DATASETS / "set2"
SOURCE = DATASETS / "flap"
OUTPUT = DATASETS / "set2_source_originals"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
INSTANCE_SUFFIX = re.compile(r"__i\d+$")


def index_files(root: Path, extensions: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stem in result:
            raise ValueError(f"duplicate stem {path.stem}: {result[path.stem]} and {path}")
        result[path.stem] = path
    return result


def validate_detection_label(path: Path) -> int:
    lines = [line.split() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty detection label: {path}")
    for line_number, fields in enumerate(lines, start=1):
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields, got {len(fields)}")
        values = [float(value) for value in fields]
        if int(values[0]) != 0:
            raise ValueError(f"{path}:{line_number}: expected class 0, got {values[0]}")
        if any(value < 0 or value > 1 for value in values[1:]):
            raise ValueError(f"{path}:{line_number}: coordinate outside [0, 1]")
    return len(lines)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")

    source_images = index_files(SOURCE / "images", IMAGE_EXTENSIONS)
    source_labels = index_files(SOURCE / "labels", {".txt"})
    if set(source_images) != set(source_labels):
        raise ValueError(
            "source image/label mismatch: "
            f"images_without_labels={sorted(set(source_images) - set(source_labels))[:10]}, "
            f"labels_without_images={sorted(set(source_labels) - set(source_images))[:10]}"
        )

    source_to_crops: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    source_split: dict[str, str] = {}
    for split in SPLITS:
        for crop in sorted((SET2 / "images" / split).glob("*")):
            if not crop.is_file() or crop.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            source_stem = INSTANCE_SUFFIX.sub("", crop.stem)
            previous_split = source_split.setdefault(source_stem, split)
            if previous_split != split:
                raise ValueError(
                    f"source {source_stem} occurs in both {previous_split} and {split}"
                )
            source_to_crops[source_stem].append((split, crop))

    found = sorted(set(source_to_crops) & set(source_images))
    missing = sorted(set(source_to_crops) - set(source_images))
    object_counts: dict[str, int] = {}
    for stem in found:
        split = source_split[stem]
        image = source_images[stem]
        label = source_labels[stem]
        object_counts[stem] = validate_detection_label(label)
        for source_path, subdir in ((image, "images"), (label, "labels")):
            destination = OUTPUT / subdir / split / source_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    mapping_rows = []
    for stem in sorted(source_to_crops):
        for split, crop in source_to_crops[stem]:
            mapping_rows.append(
                {
                    "set2_split": split,
                    "set2_crop": crop.relative_to(SET2).as_posix(),
                    "source_stem": stem,
                    "source_found": stem in source_images,
                    "output_image": (
                        (Path("images") / split / source_images[stem].name).as_posix()
                        if stem in source_images
                        else ""
                    ),
                    "output_label": (
                        (Path("labels") / split / source_labels[stem].name).as_posix()
                        if stem in source_labels
                        else ""
                    ),
                }
            )
    with (OUTPUT / "source_mapping.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)

    (OUTPUT / "missing_source_stems.txt").write_text(
        "".join(f"{stem}\n" for stem in missing), encoding="utf-8"
    )
    (OUTPUT / "data.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: flap\n",
        encoding="utf-8",
    )

    split_counts = Counter(source_split[stem] for stem in found)
    stats = {
        "set2_crop_images": sum(len(crops) for crops in source_to_crops.values()),
        "unique_source_stems_requested": len(source_to_crops),
        "source_images_and_labels_found": len(found),
        "source_stems_missing_from_flap": len(missing),
        "output_split_counts": {split: split_counts[split] for split in SPLITS},
        "source_detection_objects": sum(object_counts.values()),
        "source_images_with_multiple_objects": sum(count > 1 for count in object_counts.values()),
        "missing_source_stems": missing,
    }
    (OUTPUT / "build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
