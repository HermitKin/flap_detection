"""Convert COCO polygon/RLE instance annotations to a YOLO segmentation dataset.

This converter is intended for datasets with real instance masks, including
iSAID patches and BDD100K annotations converted to COCO format. It does not
generate pseudo-labels with SAM.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert COCO instance masks to YOLO polygon labels")
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional category names to retain; omitted means all COCO categories",
    )
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How output images reference source images",
    )
    parser.add_argument("--min-area", type=float, default=4.0, help="Minimum instance-mask area in pixels")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Polygon simplification tolerance in pixels; use 0 to disable",
    )
    return parser.parse_args()


def read_coco(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValueError(f"Missing COCO key '{key}' in {path}")
    return data


def build_category_mapping(categories: list[dict], selected_names: list[str] | None) -> tuple[dict[int, int], dict[int, str]]:
    categories_by_id = {int(category["id"]): str(category["name"]) for category in categories}
    if selected_names:
        missing = sorted(set(selected_names) - set(categories_by_id.values()))
        if missing:
            raise ValueError(f"Requested categories not found in COCO JSON: {missing}")
        ordered = [(category_id, name) for category_id, name in categories_by_id.items() if name in selected_names]
        ordered.sort(key=lambda item: selected_names.index(item[1]))
    else:
        ordered = sorted(categories_by_id.items())
    category_to_class = {category_id: class_id for class_id, (category_id, _) in enumerate(ordered)}
    class_names = {class_id: name for class_id, (_, name) in enumerate(ordered)}
    return category_to_class, class_names


def decode_rle(segmentation: dict, height: int, width: int) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as error:
        raise RuntimeError(
            "RLE masks require pycocotools. Install it in the training environment with "
            "'pip install pycocotools>=2.0.7'."
        ) from error
    rle = segmentation
    if isinstance(rle.get("counts"), list):
        rle = mask_utils.frPyObjects(rle, height, width)
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8)


def simplify_contour(contour: np.ndarray, epsilon: float) -> np.ndarray:
    contour = contour.reshape(-1, 2).astype(np.float32)
    if epsilon > 0 and len(contour) >= 3:
        contour = cv2.approxPolyDP(contour.reshape(-1, 1, 2), epsilon, True).reshape(-1, 2)
    return contour


def merge_contours(contours: list[np.ndarray]) -> np.ndarray:
    """Join disconnected visible regions with the shortest bridges for YOLO's one-polygon format."""
    contours = [contour for contour in contours if len(contour) >= 3]
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    contours.sort(key=lambda contour: cv2.contourArea(contour.astype(np.float32)), reverse=True)
    merged = contours[0]
    for contour in contours[1:]:
        distances = np.sum((merged[:, None, :] - contour[None, :, :]) ** 2, axis=2)
        merged_index, contour_index = np.unravel_index(np.argmin(distances), distances.shape)
        rotated = np.concatenate((contour[contour_index:], contour[: contour_index + 1]), axis=0)
        merged = np.concatenate(
            (merged[: merged_index + 1], rotated, merged[merged_index : merged_index + 1], merged[merged_index + 1 :]),
            axis=0,
        )
    return merged


def mask_to_polygon(mask: np.ndarray, epsilon: float, min_area: float) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    usable = [
        simplify_contour(contour, epsilon)
        for contour in contours
        if cv2.contourArea(contour) >= min_area and len(contour) >= 3
    ]
    polygon = merge_contours(usable)
    if len(polygon) < 3:
        return np.empty((0, 2), dtype=np.float32)
    return polygon


def segmentation_to_polygon(
    segmentation: object,
    height: int,
    width: int,
    epsilon: float,
    min_area: float,
) -> np.ndarray:
    # Polygon COCO annotations (such as iSAID) can be converted directly without
    # allocating a full-resolution binary mask for every one of the many instances.
    if isinstance(segmentation, list):
        polygons = segmentation
        if polygons and isinstance(polygons[0], (int, float)):
            polygons = [polygons]
        contours: list[np.ndarray] = []
        for polygon in polygons:
            points = np.asarray(polygon, dtype=np.float32)
            if points.size < 6 or points.size % 2:
                continue
            contour = points.reshape(-1, 2)
            if abs(cv2.contourArea(contour)) < min_area:
                continue
            contour = simplify_contour(contour, epsilon)
            if len(contour) >= 3:
                contours.append(contour)
        return merge_contours(contours)

    if isinstance(segmentation, dict):
        return mask_to_polygon(decode_rle(segmentation, height, width), epsilon, min_area)

    return np.empty((0, 2), dtype=np.float32)


def place_image(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Output image already exists: {target}")
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def find_source_image(images_root: Path, file_name: str) -> Path:
    candidate = images_root / Path(file_name)
    if candidate.is_file():
        return candidate
    flat_candidate = images_root / Path(file_name).name
    if flat_candidate.is_file():
        return flat_candidate
    raise FileNotFoundError(f"Image listed in COCO JSON was not found under {images_root}: {file_name}")


def convert_split(
    split: str,
    data: dict,
    images_root: Path,
    output_root: Path,
    category_to_class: dict[int, int],
    image_mode: str,
    epsilon: float,
    min_area: float,
) -> dict[str, int]:
    output_images = output_root / "images" / split
    output_labels = output_root / "labels" / split
    if output_images.exists() and any(output_images.rglob("*")):
        raise FileExistsError(f"Output image directory is not empty: {output_images}")
    if output_labels.exists() and any(output_labels.rglob("*.txt")):
        raise FileExistsError(f"Output label directory is not empty: {output_labels}")

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    stats = {"images": 0, "instances": 0, "crowd": 0, "filtered": 0, "invalid_masks": 0}
    for image_info in data["images"]:
        image_id = int(image_info["id"])
        width = int(image_info["width"])
        height = int(image_info["height"])
        file_name = str(image_info["file_name"]).replace("\\", "/")
        relative_image = Path(file_name)
        if relative_image.is_absolute() or ".." in relative_image.parts:
            relative_image = Path(relative_image.name)

        source_image = find_source_image(images_root, file_name)
        target_image = output_images / relative_image
        target_label = (output_labels / relative_image).with_suffix(".txt")
        target_label.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            category_id = int(annotation["category_id"])
            if category_id not in category_to_class:
                stats["filtered"] += 1
                continue
            if annotation.get("iscrowd", False):
                stats["crowd"] += 1
                continue
            if float(annotation.get("area", min_area)) < min_area:
                stats["filtered"] += 1
                continue

            polygon = segmentation_to_polygon(
                annotation.get("segmentation"),
                height=height,
                width=width,
                epsilon=epsilon,
                min_area=min_area,
            )
            if len(polygon) < 3:
                stats["invalid_masks"] += 1
                continue
            polygon[:, 0] = np.clip(polygon[:, 0] / width, 0.0, 1.0)
            polygon[:, 1] = np.clip(polygon[:, 1] / height, 0.0, 1.0)
            values = [str(category_to_class[category_id])]
            values.extend(f"{coordinate:.6f}" for coordinate in polygon.reshape(-1))
            lines.append(" ".join(values))
            stats["instances"] += 1

        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        place_image(source_image, target_image, image_mode)
        stats["images"] += 1

    return stats


def validate_categories(data: dict, reference_names: dict[int, str], source: Path) -> None:
    available = {int(category["id"]): str(category["name"]) for category in data["categories"]}
    reference_values = set(reference_names.values())
    if not reference_values.issubset(set(available.values())):
        missing = sorted(reference_values - set(available.values()))
        raise ValueError(f"Validation JSON is missing training categories {missing}: {source}")


def main() -> None:
    args = parse_args()
    output_root = args.output.resolve()
    if output_root.exists() and any(output_root.rglob("*")):
        raise FileExistsError(f"Output directory is not empty; choose a new path: {output_root}")

    train_json = args.train_json.resolve()
    val_json = args.val_json.resolve()
    train_data = read_coco(train_json)
    category_to_class, class_names = build_category_mapping(train_data["categories"], args.classes)
    if not class_names:
        raise ValueError("No categories remain after filtering")

    train_stats = convert_split(
        "train",
        train_data,
        args.train_images.resolve(),
        output_root,
        category_to_class,
        args.image_mode,
        args.epsilon,
        args.min_area,
    )
    del train_data

    val_data = read_coco(val_json)
    validate_categories(val_data, class_names, val_json)
    val_name_to_id = {str(category["name"]): int(category["id"]) for category in val_data["categories"]}
    val_category_to_class = {val_name_to_id[name]: class_id for class_id, name in class_names.items()}
    val_stats = convert_split(
        "val",
        val_data,
        args.val_images.resolve(),
        output_root,
        val_category_to_class,
        args.image_mode,
        args.epsilon,
        args.min_area,
    )

    config = {
        "path": output_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
    }
    with (output_root / "data.yaml").open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    print(f"Classes: {class_names}")
    print(f"Train: {train_stats}")
    print(f"Val:   {val_stats}")
    print(f"YOLO dataset: {output_root}")
    print(f"Next: python check_seg_dataset.py --data \"{output_root / 'data.yaml'}\" --split train")


if __name__ == "__main__":
    main()
