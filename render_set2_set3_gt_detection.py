from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent / "object detection" / "datasets"
DATASETS = ("set2", "set3")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
COLORS = ((255, 55, 55), (0, 210, 255), (80, 230, 100), (255, 80, 220))


def find_image(image_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def clamp(value: float, upper: int) -> int:
    return max(0, min(upper - 1, round(value)))


def parse_boxes(label_path: Path, dataset: str, width: int, height: int):
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            class_id = int(float(fields[0]))
            values = [float(value) for value in fields[1:]]
        except (ValueError, IndexError) as error:
            raise ValueError(f"invalid label at {label_path}:{line_number}") from error

        if dataset == "set2":
            if len(fields) < 5:
                raise ValueError(f"bbox missing at {label_path}:{line_number}")
            cx, cy, box_width, box_height = values[:4]
            x1 = (cx - box_width / 2) * width
            y1 = (cy - box_height / 2) * height
            x2 = (cx + box_width / 2) * width
            y2 = (cy + box_height / 2) * height
        else:
            if len(values) < 6 or len(values) % 2:
                raise ValueError(f"invalid polygon at {label_path}:{line_number}")
            xs = values[0::2]
            ys = values[1::2]
            x1, x2 = min(xs) * width, max(xs) * width
            y1, y2 = min(ys) * height, max(ys) * height

        boxes.append(
            (
                class_id,
                clamp(x1, width),
                clamp(y1, height),
                clamp(x2, width),
                clamp(y2, height),
            )
        )
    return boxes


def draw_boxes(image: Image.Image, boxes) -> Image.Image:
    result = image.convert("RGB")
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    line_width = max(3, round(min(result.size) / 180))
    for object_index, (class_id, x1, y1, x2, y2) in enumerate(boxes, 1):
        color = COLORS[class_id % len(COLORS)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        text = f"GT flap #{object_index}"
        text_box = draw.textbbox((0, 0), text, font=font, stroke_width=1)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_y = max(0, y1 - text_height - 6)
        draw.rectangle((x1, text_y, x1 + text_width + 8, text_y + text_height + 6), fill=color)
        draw.text((x1 + 4, text_y + 2), text, fill=(0, 0, 0), font=font, stroke_width=1)
    return result


def render_dataset(dataset: str) -> dict:
    root = ROOT / dataset
    output = root / "gt_detection"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    rendered = 0
    objects = 0
    split_counts = {}
    missing_images = []
    for split in ("train", "val", "test"):
        label_dir = root / "labels" / split
        labels = sorted(label_dir.glob("*.txt")) if label_dir.is_dir() else []
        split_rendered = 0
        for label_path in labels:
            image_path = find_image(root / "images" / split, label_path.stem)
            if image_path is None:
                missing_images.append(f"{split}/{label_path.stem}")
                continue
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            boxes = parse_boxes(label_path, dataset, image.width, image.height)
            destination = output / split / image_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            draw_boxes(image, boxes).save(destination, quality=95)
            rendered += 1
            split_rendered += 1
            objects += len(boxes)
        split_counts[split] = split_rendered

    stats = {
        "dataset": dataset,
        "rendered_images": rendered,
        "rendered_objects": objects,
        "split_counts": split_counts,
        "missing_images_for_labels": missing_images,
        "note": "Only images with GT label files are rendered.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "render_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def main() -> None:
    all_stats = [render_dataset(dataset) for dataset in DATASETS]
    print(json.dumps(all_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
