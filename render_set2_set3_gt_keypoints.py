from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent / "object detection" / "datasets"
DATASETS = ("set2", "set3")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
POINT_COLORS = ((255, 60, 60), (0, 220, 255), (100, 255, 80), (255, 80, 220))
INSTANCE_COLORS = ((0, 255, 255), (255, 80, 80), (80, 255, 120), (255, 100, 255))


def find_image(directory: Path, stem: str) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def draw_point(draw, point, color, radius, label=None, hollow=False):
    x, y = point
    box = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(box, fill=(20, 20, 20) if hollow else color, outline=(255, 255, 255), width=2)
    if label:
        draw.text((x + radius + 2, y - radius), label, fill=color, font=ImageFont.load_default(), stroke_width=2, stroke_fill=(0, 0, 0))


def render_set2(image: Image.Image, label_path: Path) -> tuple[Image.Image, int, int]:
    result = image.convert("RGB")
    draw = ImageDraw.Draw(result)
    width, height = result.size
    radius = max(4, round(min(width, height) / 55))
    line_width = max(3, radius // 2)
    instance_count = 0
    point_count = 0
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 17:
            raise ValueError(f"{label_path}:{line_number}: expected 17 fields, got {len(fields)}")
        values = [float(value) for value in fields]
        points = []
        for index in range(4):
            x, y, visibility = values[5 + index * 3 : 8 + index * 3]
            points.append((x * width, y * height, int(visibility)))
        for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
            if points[first][2] > 0 and points[second][2] > 0:
                draw.line((points[first][0], points[first][1], points[second][0], points[second][1]), fill=(255, 230, 0), width=line_width)
        for index, (x, y, visibility) in enumerate(points):
            if visibility <= 0:
                continue
            draw_point(draw, (x, y), POINT_COLORS[index], radius, f"P{index + 1}", hollow=visibility == 1)
            point_count += 1
        instance_count += 1
    return result, instance_count, point_count


def render_set3(image: Image.Image, label_path: Path) -> tuple[Image.Image, int, int]:
    result = image.convert("RGB")
    draw = ImageDraw.Draw(result)
    width, height = result.size
    radius = max(3, round(min(width, height) / 80))
    line_width = max(2, radius // 2)
    instance_count = 0
    point_count = 0
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        coordinates = [float(value) for value in fields[1:]]
        if len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError(f"{label_path}:{line_number}: invalid polygon coordinates")
        points = [(coordinates[i] * width, coordinates[i + 1] * height) for i in range(0, len(coordinates), 2)]
        color = INSTANCE_COLORS[instance_count % len(INSTANCE_COLORS)]
        draw.line(points + [points[0]], fill=color, width=line_width, joint="curve")
        for point_index, point in enumerate(points, 1):
            draw_point(draw, point, color, radius, str(point_index))
            point_count += 1
        instance_count += 1
    return result, instance_count, point_count


def render_dataset(dataset: str) -> dict:
    root = ROOT / dataset
    output = root / "gt_keypoints"
    if output.exists():
        # This directory is generated exclusively by this script; rebuild it atomically enough
        # for a corrected rerun without touching the source images or labels.
        shutil.rmtree(output)
    image_count = instance_count = point_count = 0
    split_counts = {}
    skipped_non_keypoint_labels = []
    for split in ("train", "val", "test"):
        split_count = 0
        label_dir = root / "labels" / split
        for label_path in sorted(label_dir.glob("*.txt")) if label_dir.is_dir() else []:
            lines = [line for line in label_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if any(len(line.split()) != 17 for line in lines):
                skipped_non_keypoint_labels.append(label_path.relative_to(root).as_posix())
                continue
            image_path = find_image(root / "images" / split, label_path.stem)
            if image_path is None:
                raise FileNotFoundError(f"no image for {label_path}")
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            rendered, instances, points = render_set2(image, label_path)
            destination = output / split / image_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            rendered.save(destination, quality=95)
            image_count += 1
            split_count += 1
            instance_count += instances
            point_count += points
        split_counts[split] = split_count
    stats = {
        "dataset": dataset,
        "rendered_images": image_count,
        "instances": instance_count,
        "visible_keypoints_or_polygon_vertices": point_count,
        "split_counts": split_counts,
        "format": "4 pose keypoints",
        "skipped_non_keypoint_labels": skipped_non_keypoint_labels,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "render_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    print(json.dumps([render_dataset(name) for name in DATASETS], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
