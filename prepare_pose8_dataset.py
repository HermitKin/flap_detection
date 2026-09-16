from __future__ import annotations

import os
import random
import re
import shutil
from pathlib import Path


SOURCE = Path("/workspace/datasets/testset")
OUTPUT = Path("/workspace/datasets/testset_pose8")
SOURCE_IMAGES = SOURCE / "images" / "train"
SOURCE_LABELS = SOURCE / "labels" / "train"
SEED = 20260819


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def parse_pose4(line: str) -> tuple[int, list[float], list[tuple[float, float, int]]]:
    values = line.split()
    if len(values) != 17:
        raise ValueError(f"expected 17 fields for 4-point pose, got {len(values)}")
    class_id = int(values[0])
    bbox = list(map(float, values[1:5]))
    raw = list(map(float, values[5:]))
    points = [(raw[i], raw[i + 1], int(raw[i + 2])) for i in range(0, 12, 3)]
    return class_id, bbox, points


def frame_number(path: Path) -> int:
    match = re.search(r"_(\d+)__i\d+$", path.stem)
    if not match:
        raise ValueError(f"cannot parse frame number: {path.name}")
    return int(match.group(1))


def split_groups(images: list[Path]) -> dict[str, str]:
    groups: list[list[Path]] = []
    for image in sorted(images, key=frame_number):
        if not groups or frame_number(image) - frame_number(groups[-1][-1]) > 2:
            groups.append([])
        groups[-1].append(image)
    rng = random.Random(SEED)
    isolated = groups[1:]
    rng.shuffle(isolated)
    ordered = [groups[0], *isolated]  # keep the 0006-0010 near-duplicate sequence together
    target_val = max(1, round(len(images) * 0.2))
    val_groups: list[list[Path]] = []
    val_count = 0
    for group in reversed(ordered):
        if val_count >= target_val:
            break
        val_groups.append(group)
        val_count += len(group)
    val_names = {image.name for group in val_groups for image in group}
    return {image.name: ("val" if image.name in val_names else "train") for image in images}


def main() -> None:
    images = sorted(SOURCE_IMAGES.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"no images found in {SOURCE_IMAGES}")
    split_by_name = split_groups(images)
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    merged_count = 0
    corrected_source_classes = 0
    for image in images:
        label_path = SOURCE_LABELS / f"{image.stem}.txt"
        records = [parse_pose4(line) for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(records) != 2:
            raise ValueError(f"{label_path.name}: expected outer+inner skeleton, got {len(records)}")
        # CVAT class ids are inconsistent in a few files. Geometry is unambiguous:
        # the outer skeleton owns the larger bbox, the inner/counter skeleton the smaller bbox.
        records.sort(key=lambda item: item[1][2] * item[1][3], reverse=True)
        outer, inner = records
        if outer[0] != 1 or inner[0] != 0:
            corrected_source_classes += 1
        outer_points = outer[2]
        inner_points = inner[2]
        points8 = outer_points + inner_points
        if any(not (0 <= x <= 1 and 0 <= y <= 1 and v in (0, 1, 2)) for x, y, v in points8):
            raise ValueError(f"{label_path.name}: invalid keypoint coordinate or visibility")

        split = split_by_name[image.name]
        target_image = OUTPUT / "images" / split / image.name
        target_label = OUTPUT / "labels" / split / label_path.name
        link_or_copy(image, target_image)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        fields = ["0", *(f"{value:.8f}" for value in outer[1])]
        for x, y, visibility in points8:
            fields.extend((f"{x:.8f}", f"{y:.8f}", str(visibility)))
        target_label.write_text(" ".join(fields) + "\n", encoding="utf-8")
        merged_count += 1

    yaml_text = """path: /workspace/datasets/testset_pose8
train: images/train
val: images/val

kpt_shape: [8, 3]
flip_idx: [1, 0, 3, 2, 5, 4, 7, 6]

names:
  0: flap
"""
    (OUTPUT / "data.yaml").write_text(yaml_text, encoding="utf-8")
    train_count = sum(split == "train" for split in split_by_name.values())
    val_count = len(images) - train_count
    print(f"source_images={len(images)}")
    print(f"merged_pose8_labels={merged_count}")
    print(f"train={train_count}, val={val_count}")
    print(f"source_class_order_corrected={corrected_source_classes}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
