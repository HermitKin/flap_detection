"""Validate YOLO polygon labels and render random mask overlays for visual QA."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check and visualize a YOLO instance-segmentation dataset")
    parser.add_argument("--data", type=Path, default=Path("dataset-seg/data.yaml"))
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/dataset-check"))
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict, Path]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    root = Path(config.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    return config, root


def color_for_class(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(class_id + 2026)
    return tuple(int(value) for value in rng.integers(64, 256, size=3))


def read_image(path: Path) -> np.ndarray | None:
    """Read an image through NumPy because cv2.imread can fail on Windows Unicode paths."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image through NumPy because cv2.imwrite can fail on Windows Unicode paths."""
    extension = path.suffix or ".jpg"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"Cannot encode output image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    config, root = load_config(data_path)
    names = config.get("names") or {}
    class_count = len(names)
    split_value = config.get(args.split)
    if not split_value:
        raise ValueError(f"Split '{args.split}' is not defined in {data_path}")

    image_dir = root / split_value
    label_dir = root / "labels" / args.split
    images = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    errors: list[str] = []
    object_count = 0
    for image_path in images:
        relative_image = image_path.relative_to(image_dir)
        label_path = (label_dir / relative_image).with_suffix(".txt")
        if not label_path.is_file():
            errors.append(f"missing label: {label_path}")
            continue
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            fields = line.split()
            try:
                class_id = int(float(fields[0]))
                coordinates = np.asarray([float(value) for value in fields[1:]], dtype=np.float32)
            except (ValueError, IndexError):
                errors.append(f"invalid numbers: {label_path}:{line_number}")
                continue
            if len(fields) < 7 or coordinates.size % 2:
                errors.append(f"invalid polygon length: {label_path}:{line_number}")
            elif not 0 <= class_id < class_count:
                errors.append(f"class out of range: {label_path}:{line_number}")
            elif np.any((coordinates < 0) | (coordinates > 1)):
                errors.append(f"coordinates outside [0,1]: {label_path}:{line_number}")
            else:
                object_count += 1

    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"Found {len(errors)} dataset errors:\n{preview}")

    random.seed(args.seed)
    selected = random.sample(images, min(args.samples, len(images)))
    output_dir = args.output.resolve() / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in selected:
        image = read_image(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]
        overlay = image.copy()
        relative_image = image_path.relative_to(image_dir)
        label_path = (label_dir / relative_image).with_suffix(".txt")
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = line.split()
            class_id = int(float(fields[0]))
            points = np.asarray([float(value) for value in fields[1:]], dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= width
            points[:, 1] *= height
            polygon = np.round(points).astype(np.int32)
            color = color_for_class(class_id)
            cv2.fillPoly(overlay, [polygon], color)
            cv2.polylines(image, [polygon], True, color, 2)
        rendered = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
        write_image(output_dir / f"{image_path.stem}.jpg", rendered)

    print(f"Dataset valid: {len(images)} images, {object_count} instance masks")
    print(f"Visual samples: {output_dir}")


if __name__ == "__main__":
    main()
