"""Mask/polygon conversion helpers for the SAM3 annotation GUI."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def mask_to_polygons(
    mask: np.ndarray,
    *,
    min_area: float = 40.0,
    epsilon_ratio: float = 0.0015,
) -> list[np.ndarray]:
    """Convert one binary mask to exterior OpenCV polygons.

    YOLO segmentation labels cannot represent holes, so only exterior contours are
    exported. Small disconnected specks are discarded.
    """
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[np.ndarray] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True).reshape(-1, 2)
        if len(polygon) >= 3:
            polygons.append(polygon.astype(np.float32))
    return sorted(polygons, key=cv2.contourArea, reverse=True)


def masks_to_yolo_lines(
    masks: list[np.ndarray],
    *,
    class_id: int = 0,
    min_area: float = 40.0,
    epsilon_ratio: float = 0.0015,
) -> list[str]:
    """Serialize masks as YOLO instance-segmentation polygon rows."""
    lines: list[str] = []
    for mask in masks:
        height, width = mask.shape[:2]
        # A YOLO row represents one instance and cannot encode disconnected pieces.
        # SAM occasionally leaves tiny remote islands, so keep the largest exterior
        # contour instead of turning those islands into fake gauge instances.
        polygons = mask_to_polygons(mask, min_area=min_area, epsilon_ratio=epsilon_ratio)
        if not polygons:
            continue
        normalized = polygons[0] / np.array([width, height], dtype=np.float32)
        normalized = np.clip(normalized, 0.0, 1.0)
        coords = " ".join(f"{value:.6f}" for value in normalized.reshape(-1))
        lines.append(f"{class_id} {coords}")
    return lines


def load_yolo_masks(label_path: Path, image_shape: tuple[int, int]) -> list[np.ndarray]:
    """Load existing YOLO polygons so interrupted annotation can be resumed."""
    if not label_path.exists():
        return []
    height, width = image_shape
    masks: list[np.ndarray] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2:
            continue
        try:
            values = np.asarray([float(v) for v in fields[1:]], dtype=np.float32).reshape(-1, 2)
        except ValueError:
            continue
        points = np.rint(values * np.array([width, height], dtype=np.float32)).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        masks.append(mask.astype(bool))
    return masks


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0
