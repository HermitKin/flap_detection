"""Deterministic, public-safe gauge geometry for documentation and smoke tests.

The box and corners are KNOWN construction geometry, never model predictions.
Rectification and reading use the same production service as the desktop GUI.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from services import ObjectDetectionService, ReadingService, draw_pose_results, write_image


def make_gauge(level: float = 0.62, colour: str = "red") -> np.ndarray:
    if not 0.15 <= level <= 0.85:
        raise ValueError("Synthetic demo level must be between 0.15 and 0.85.")
    if colour not in {"red", "blue"}:
        raise ValueError("Synthetic demo colour must be red or blue.")
    height, width = 560, 160
    image = np.full((height, width, 3), (223, 230, 235), dtype=np.uint8)
    image[:, :12] = (105, 120, 128)
    image[:, -12:] = (105, 120, 128)
    image[:, 48:98] = (241, 242, 244)
    boundary = round(height * (1 - level))
    bottom_colour = (46, 62, 211) if colour == "red" else (206, 109, 42)
    image[boundary:, 48:98] = bottom_colour
    # One-pixel flap joints remain short enough for the continuity algorithm.
    for y in range(14, height, 18):
        image[y:y + 1, 48:98] = (180, 188, 195)
    for value in range(0, 101, 10):
        y = int(np.clip(round((1 - value / 100) * height), 12, height - 10))
        cv2.line(image, (106, y), (116, y), (75, 90, 105), 1)
        cv2.putText(image, str(value), (119, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (55, 65, 75), 1, cv2.LINE_AA)
    return image


def make_scene(level: float = 0.62, colour: str = "red") -> tuple[np.ndarray, list, list]:
    gauge = make_gauge(level, colour)
    quad = np.array([[215, 66], [374, 110], [319, 645], [146, 576]], np.float32)
    source = np.array([[0, 0], [159, 0], [159, 559], [0, 559]], np.float32)
    matrix = cv2.getPerspectiveTransform(source, quad)
    scene = np.full((710, 520, 3), (32, 43, 56), np.uint8)
    for x in range(0, 520, 40):
        cv2.line(scene, (x, 0), (x, 709), (41, 54, 68), 1)
    warped = cv2.warpPerspective(gauge, matrix, (520, 710))
    mask = cv2.warpPerspective(np.full(gauge.shape[:2], 255, np.uint8), matrix, (520, 710))
    scene[mask > 0] = warped[mask > 0]
    box = [float(quad[:, 0].min() - 8), float(quad[:, 1].min() - 8),
           float(quad[:, 0].max() + 8), float(quad[:, 1].max() + 8)]
    return scene, quad.tolist(), box


def run_example(workdir: Path, level: float = 0.62, colour: str = "red") -> dict:
    scene, quad, box = make_scene(level, colour)
    source = write_image(workdir / f"synthetic_{colour}_{round(level * 100):02d}.png", scene)
    service = ReadingService(ObjectDetectionService(), None)
    record = service.read(source, 0, 100, False, existing_quads=[quad])
    annotated = draw_pose_results(scene, [{
        "box_xyxy": box, "keypoints": quad,
        "provenance": "known_synthetic_geometry",
        "keypoint_confidences": [1.0] * 4, "class_name": "known geometry",
    }])
    return {"source": source, "scene": scene, "quad": quad, "box": box,
            "annotated": annotated, "record": record, "expected": level * 100}
