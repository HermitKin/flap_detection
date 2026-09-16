from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from config import (
    OBJECT_CONFIDENCE,
    OBJECT_IMGSZ,
    OBJECT_MODEL,
    OUTPUT_DIR,
    POSE_CONFIDENCE,
    POSE_IMGSZ,
    POSE_MODEL,
)
from direct_hsv_reader import analyze_direct_flap_strip

ProgressCallback = Callable[[int, int, str], None]
APP_ROOT = Path(__file__).resolve().parent
OBJECT_INFERENCE_HELPER = APP_ROOT / "legacy" / "detect_gauge_yolo.py"
POSE_INFERENCE_HELPER = APP_ROOT / "legacy" / "detect_pose_keypoints_yolo.py"
KEYPOINT_NAMES_4 = ("TL", "TR", "BR", "BL")
KEYPOINT_COLOURS = (
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
)


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"无法编码图片：{path}")
    encoded.tofile(path)
    return path.resolve()


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path.resolve()


def run_json_inference(command: list[str], marker: str) -> object:
    """Run YOLO outside the GUI process and decode its marked JSON output."""
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        process = subprocess.run(
            command,
            cwd=str(APP_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
            creationflags=creation_flags,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("YOLO 推理超过 300 秒，已终止本次检测。") from exc

    marker_line = next(
        (line for line in reversed(process.stdout.splitlines()) if line.startswith(marker)),
        None,
    )
    if process.returncode != 0 or marker_line is None:
        details = process.stderr.strip() or process.stdout.strip() or "无错误输出"
        raise RuntimeError(
            f"YOLO 子进程执行失败（退出码 {process.returncode}）：{details}"
        )
    try:
        return json.loads(marker_line[len(marker):])
    except json.JSONDecodeError as exc:
        raise RuntimeError("YOLO 子进程返回的数据不是有效 JSON。") from exc


def clamp_text_origin(
    image: np.ndarray,
    text: str,
    preferred_x: int,
    preferred_y: int,
    font_scale: float,
    thickness: int,
) -> tuple[int, int]:
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x = int(np.clip(preferred_x, 0, max(0, image.shape[1] - text_width - 1)))
    y = int(np.clip(preferred_y, text_height + 1, image.shape[0] - baseline - 1))
    return x, y


def draw_pose_results(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw pose records without importing Ultralytics into the GUI process."""
    annotated = image.copy()
    narrow = annotated.shape[1] < 300
    font_scale = 0.45 if narrow else 0.55
    thickness = 1 if narrow else 2
    for detection in detections:
        box = detection.get("box_xyxy", detection.get("box"))
        if box is not None:
            x1, y1, x2, y2 = np.round(np.asarray(box)).astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
            if not narrow:
                confidence = float(detection.get("box_confidence", 0.0))
                label = f"{detection.get('class_name', 'flap')} {confidence:.3f}"
                origin = clamp_text_origin(
                    annotated, label, x1, y1 - 6, font_scale, thickness
                )
                cv2.putText(
                    annotated, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 0), thickness, cv2.LINE_AA,
                )

        points = np.asarray(detection.get("keypoints", []), dtype=np.float32)
        confidences = np.asarray(
            detection.get("keypoint_confidences", [1.0] * len(points)),
            dtype=np.float32,
        )
        if points.shape == (4, 2):
            cv2.polylines(
                annotated, [np.round(points).astype(np.int32)], True,
                (255, 0, 255), 3, cv2.LINE_AA,
            )
        for point_index, ((x, y), point_confidence) in enumerate(
            zip(points, confidences)
        ):
            centre = (int(round(float(x))), int(round(float(y))))
            colour = KEYPOINT_COLOURS[point_index % len(KEYPOINT_COLOURS)]
            cv2.circle(annotated, centre, 7, colour, -1, cv2.LINE_AA)
            point_name = (
                KEYPOINT_NAMES_4[point_index]
                if len(points) == 4 else f"K{point_index}"
            )
            point_label = point_name if narrow else f"{point_name} {point_confidence:.3f}"
            preferred_y = centre[1] + 18 if point_index < 2 else centre[1] - 8
            origin = clamp_text_origin(
                annotated, point_label, centre[0] + 5, preferred_y,
                font_scale, thickness,
            )
            cv2.putText(
                annotated, point_label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, colour, thickness, cv2.LINE_AA,
            )
    return annotated


@dataclass
class DetectionRecord:
    source_path: Path
    original: np.ndarray
    annotated: np.ndarray
    detections: list[dict]
    task: str
    model_path: Path | None = None
    saved_image: Path | None = None
    saved_json: Path | None = None
    crops: list[np.ndarray] = field(default_factory=list)
    saved_crops: list[Path] = field(default_factory=list)


@dataclass
class ReadingRecord:
    source_path: Path
    original: np.ndarray
    unrectified_visualization: np.ndarray
    rectified_visualization: np.ndarray
    flap_strip_visualization: np.ndarray
    annotated: np.ndarray
    hsv_visualization: np.ndarray
    reports: list[dict]
    unrectified_reports: list[dict]
    scale_min: float
    scale_max: float
    source_mode: str
    saved_input: Path | None = None
    saved_image: Path | None = None
    saved_detection_result: Path | None = None
    saved_unrectified: Path | None = None
    saved_rectified: Path | None = None
    saved_flap_strip: Path | None = None
    saved_hsv: Path | None = None
    saved_json: Path | None = None


class ObjectDetectionService:
    def __init__(self, model_path: Path = OBJECT_MODEL) -> None:
        self.model_path = Path(model_path).resolve()

    def set_model_path(self, model_path: Path) -> Path:
        selected = Path(model_path).expanduser().resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"目标检测模型不存在：{selected}")
        self.model_path = selected
        return selected

    def detect(
        self,
        image_paths: list[Path],
        confidence: float = OBJECT_CONFIDENCE,
        progress: ProgressCallback | None = None,
    ) -> list[DetectionRecord]:
        if not image_paths:
            return []
        model_path = Path(self.model_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"目标检测模型不存在：{model_path}")
        if not OBJECT_INFERENCE_HELPER.is_file():
            raise FileNotFoundError(f"目标检测子进程脚本不存在：{OBJECT_INFERENCE_HELPER}")
        records = []
        total = len(image_paths)
        for index, image_path in enumerate(image_paths, start=1):
            image_path = image_path.resolve()
            original = read_image(image_path)
            raw_detections = run_json_inference(
                [
                    sys.executable,
                    str(OBJECT_INFERENCE_HELPER),
                    str(model_path),
                    str(image_path),
                    str(float(confidence)),
                    str(int(OBJECT_IMGSZ)),
                ],
                "YOLO_RESULT_JSON=",
            )
            if not isinstance(raw_detections, list):
                raise RuntimeError("目标检测子进程返回的数据格式错误。")
            detections = []
            for detection_id, item in enumerate(raw_detections, start=1):
                box = item.get("box_xyxy", item.get("box"))
                if box is None:
                    continue
                score = float(item.get("confidence", item.get("box_confidence", 0.0)))
                detections.append({
                    "detection_id": detection_id,
                    "class_id": int(item.get("class_id", 0)),
                    "class_name": str(item.get("class_name", "flap")),
                    "box": [float(value) for value in box],
                    "box_xyxy": [float(value) for value in box],
                    "confidence": score,
                    "box_confidence": score,
                })
            annotated = self._draw(original, detections)
            records.append(DetectionRecord(
                source_path=image_path,
                original=original,
                annotated=annotated,
                detections=detections,
                task="object_detection",
                model_path=model_path,
            ))
            if progress:
                progress(index, total, image_path.name)
        return records

    @staticmethod
    def _draw(image: np.ndarray, detections: list[dict]) -> np.ndarray:
        annotated = image.copy()
        for detection in detections:
            x1, y1, x2, y2 = np.round(detection["box_xyxy"]).astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 60), 3)
            label = f"{detection['class_name']} {detection['confidence']:.3f}"
            text_y = max(18, y1 - 7)
            cv2.putText(annotated, label, (max(0, x1), text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 60), 2, cv2.LINE_AA)
        return annotated

    def save_record(self, record: DetectionRecord) -> tuple[Path, Path]:
        root = OUTPUT_DIR / "object_detection"
        image_path = write_image(root / "detected_images" / f"{record.source_path.stem}.png",
                                 record.annotated)
        payload = {
            "source_image": str(record.source_path),
            "source_size": [record.original.shape[1], record.original.shape[0]],
            "model": str((record.model_path or self.model_path).resolve()),
            "detections": record.detections,
            "detected_image": str(image_path),
        }
        json_path = write_json(root / "data" / f"{record.source_path.stem}.json", payload)
        record.saved_image, record.saved_json = image_path, json_path
        return image_path, json_path

    @staticmethod
    def crop_record(record: DetectionRecord) -> list[np.ndarray]:
        height, width = record.original.shape[:2]
        crops = []
        for detection in record.detections:
            x1, y1, x2, y2 = (int(round(value)) for value in detection["box_xyxy"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                crops.append(record.original[y1:y2, x1:x2].copy())
        record.crops = crops
        return crops

    @staticmethod
    def save_crops(record: DetectionRecord) -> list[Path]:
        if not record.crops:
            raise ValueError("请先点击 ROI 裁剪。")
        crop_dir = OUTPUT_DIR / record.task / "roi_crops"
        saved = []
        for index, crop in enumerate(record.crops, start=1):
            saved.append(write_image(
                crop_dir / f"{record.source_path.stem}_roi_{index:02d}.png", crop
            ))
        record.saved_crops = saved
        return saved


class PoseDetectionService:
    def __init__(self, model_path: Path = POSE_MODEL) -> None:
        self.model_path = Path(model_path).resolve()

    def set_model_path(self, model_path: Path) -> Path:
        selected = Path(model_path).expanduser().resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"关键点模型不存在：{selected}")
        self.model_path = selected
        return selected

    def detect(
        self,
        image_paths: list[Path],
        confidence: float = POSE_CONFIDENCE,
        progress: ProgressCallback | None = None,
    ) -> list[DetectionRecord]:
        if not image_paths:
            return []
        model_path = Path(self.model_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"关键点模型不存在：{model_path}")
        if not POSE_INFERENCE_HELPER.is_file():
            raise FileNotFoundError(f"关键点检测子进程脚本不存在：{POSE_INFERENCE_HELPER}")
        records = []
        total = len(image_paths)
        for index, image_path in enumerate(image_paths, start=1):
            image_path = image_path.resolve()
            original = read_image(image_path)
            detections = run_json_inference(
                [
                    sys.executable,
                    str(POSE_INFERENCE_HELPER),
                    str(model_path),
                    str(image_path),
                    str(float(confidence)),
                    str(int(POSE_IMGSZ)),
                    "--no-save",
                ],
                "POSE_RESULT_JSON=",
            )
            if not isinstance(detections, list):
                raise RuntimeError("关键点检测子进程返回的数据格式错误。")
            record = DetectionRecord(
                source_path=image_path,
                original=original,
                annotated=draw_pose_results(original, detections),
                detections=detections,
                task="keypoint_detection",
                model_path=model_path,
            )
            # Requirement: every pose image and its keypoint data are persisted.
            self.save_record(record)
            records.append(record)
            if progress:
                progress(index, total, image_path.name)
        return records

    def save_record(self, record: DetectionRecord) -> tuple[Path, Path]:
        root = OUTPUT_DIR / "keypoint_detection"
        image_path = write_image(root / "detected_images" / f"{record.source_path.stem}.png",
                                 record.annotated)
        payload = {
            "source_image": str(record.source_path),
            "source_filename": record.source_path.name,
            "source_stem": record.source_path.stem,
            "source_size": [record.original.shape[1], record.original.shape[0]],
            "model": str((record.model_path or self.model_path).resolve()),
            "keypoint_definition_for_4_points": ["TL", "TR", "BR", "BL"],
            "coordinate_system": {
                "pixel_origin": "top-left",
                "pixel_x_direction": "right",
                "pixel_y_direction": "down",
                "normalized_range": [0.0, 1.0],
            },
            "detections": record.detections,
            "detected_image": str(image_path),
        }
        json_path = write_json(root / "keypoints" / f"{record.source_path.stem}.json", payload)
        record.saved_image, record.saved_json = image_path, json_path
        return image_path, json_path

    @staticmethod
    def crop_record(record: DetectionRecord) -> list[np.ndarray]:
        height, width = record.original.shape[:2]
        crops = []
        for detection in record.detections:
            box = detection.get("box_xyxy", detection.get("box"))
            if box is None:
                continue
            x1, y1, x2, y2 = (int(round(value)) for value in box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                crops.append(record.original[y1:y2, x1:x2].copy())
        record.crops = crops
        return crops

    @staticmethod
    def save_crops(record: DetectionRecord) -> list[Path]:
        return ObjectDetectionService.save_crops(record)


class ReadingService:
    def __init__(
        self,
        object_service: ObjectDetectionService,
        pose_service: PoseDetectionService | None = None,
    ) -> None:
        self.object_service = object_service
        self.pose_service = pose_service

    def read(
        self,
        image_path: Path,
        scale_min: float,
        scale_max: float,
        call_model: bool,
        existing_boxes: list[list[float]] | None = None,
        existing_quads: list[list[list[float]]] | None = None,
    ) -> ReadingRecord:
        if scale_max <= scale_min:
            raise ValueError("量程上限必须大于量程下限。")
        image = read_image(image_path)
        if call_model:
            if self.pose_service is None:
                raise ValueError("关键点模型服务未配置，无法执行透视校正。")
            detection_record = self.pose_service.detect([image_path])[0]
            boxes = [detection["box_xyxy"] for detection in detection_record.detections]
            quads = [
                detection["roi_quad_tl_tr_br_bl"]
                for detection in detection_record.detections
                if detection.get("roi_quad_tl_tr_br_bl")
            ]
            source_mode = "重新调用关键点模型并透视校正"
        elif existing_quads:
            quads = existing_quads
            boxes = existing_boxes or []
            source_mode = "使用第二页无标注图和关键点进行透视校正"
        elif existing_boxes:
            boxes = existing_boxes
            quads = []
            source_mode = "使用已有检测数据"
        else:
            boxes = [[0.0, 0.0, float(image.shape[1]), float(image.shape[0])]]
            quads = []
            source_mode = "当前图片直接作为 ROI"

        work_items: list[dict] = []
        if quads:
            for quad in quads:
                rectified, matrix = self._rectify_quad(image, quad)
                unrectified_roi, unrectified_box = self._crop_quad_bounding_box(image, quad)
                work_items.append({
                    "roi": rectified,
                    "unrectified_roi": unrectified_roi,
                    "unrectified_box": unrectified_box,
                    "quad": [[float(value) for value in point] for point in quad],
                    "box": None,
                    "perspective_matrix": matrix.astype(float).tolist(),
                    "perspective_applied": True,
                })
        else:
            image_height, image_width = image.shape[:2]
            for raw_box in boxes:
                x1, y1, x2, y2 = (int(round(value)) for value in raw_box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image_width, x2), min(image_height, y2)
                if x2 - x1 < 7 or y2 - y1 < 40:
                    continue
                work_items.append({
                    "roi": image[y1:y2, x1:x2].copy(),
                    "unrectified_roi": image[y1:y2, x1:x2].copy(),
                    "unrectified_box": [x1, y1, x2, y2],
                    "quad": None,
                    "box": [x1, y1, x2, y2],
                    "perspective_matrix": None,
                    "perspective_applied": False,
                })

        if not work_items:
            raise ValueError("关键点或检测框没有生成有效的液位计 ROI。")

        reports = []
        unrectified_reports = []
        hsv_views: list[np.ndarray] = []
        flap_strip_views: list[np.ndarray] = []
        unrectified_views: list[np.ndarray] = []
        rectified_views: list[np.ndarray] = []
        annotated_views: list[np.ndarray] = []
        formal_errors: list[str] = []
        for gauge_id, item in enumerate(work_items, start=1):
            roi = item["roi"]
            raw_roi = item["unrectified_roi"]
            raw_annotated = raw_roi.copy()
            try:
                raw_analysis = analyze_direct_flap_strip(raw_roi)
                raw_sx1, raw_sx2 = raw_analysis.x1, raw_analysis.x2
                raw_level_y = raw_analysis.level_y
                raw_peak_z = raw_analysis.confidence
                raw_ratio = (raw_roi.shape[0] - raw_level_y) / max(
                    raw_roi.shape[0], 1
                )
                raw_reading = float(np.clip(
                    scale_min + raw_ratio * (scale_max - scale_min), scale_min, scale_max
                ))
                cv2.rectangle(
                    raw_annotated,
                    (raw_sx1, raw_level_y),
                    (raw_sx2 - 1, raw_roi.shape[0] - 1),
                    (255, 255, 0),
                    1,
                )
                cv2.line(
                    raw_annotated,
                    (0, raw_level_y),
                    (raw_roi.shape[1] - 1, raw_level_y),
                    (0, 165, 255),
                    2,
                )
                cv2.putText(
                    raw_annotated,
                    f"RAW #{gauge_id} {raw_reading:.1f}",
                    (4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 165, 255),
                    1,
                    cv2.LINE_AA,
                )
                unrectified_reports.append({
                    "gauge_id": gauge_id,
                    "source_box_xyxy": item["unrectified_box"],
                    "roi_size": [raw_roi.shape[1], raw_roi.shape[0]],
                    "flap_strip_xyxy": [
                        raw_sx1, raw_level_y, raw_sx2, raw_roi.shape[0]
                    ],
                    "coloured_interval_y": [raw_level_y, raw_roi.shape[0]],
                    "coloured_height": raw_roi.shape[0] - raw_level_y,
                    "full_scale_height": raw_roi.shape[0],
                    "boundary_y": raw_level_y,
                    "level_ratio": round(float(raw_ratio), 6),
                    "reading": round(raw_reading, 3),
                    "change_confidence_z": round(float(raw_peak_z), 3),
                    "continuity_confidence": round(float(raw_peak_z), 3),
                    "boundary_method": raw_analysis.metrics["method"],
                    "continuity_metrics": raw_analysis.metrics,
                    "perspective_applied": False,
                    "strip_selection_mode": "bottom_connected_coloured_interval_only",
                })
            except ValueError as exc:
                cv2.putText(
                    raw_annotated,
                    "RAW: no boundary",
                    (4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                unrectified_reports.append({
                    "gauge_id": gauge_id,
                    "source_box_xyxy": item["unrectified_box"],
                    "roi_size": [raw_roi.shape[1], raw_roi.shape[0]],
                    "perspective_applied": False,
                    "error": str(exc),
                })
            unrectified_views.append(raw_annotated)
            try:
                analysis = analyze_direct_flap_strip(roi)
            except ValueError as exc:
                formal_errors.append(f"液位计 {gauge_id}: {exc}")
                continue
            sx1, sx2 = analysis.x1, analysis.x2
            level_y = analysis.level_y
            peak_z = analysis.confidence
            ratio = (roi.shape[0] - level_y) / max(roi.shape[0], 1)
            reading = float(np.clip(scale_min + ratio * (scale_max - scale_min),
                                    scale_min, scale_max))
            colour_segment = roi[level_y:, sx1:sx2].copy()
            colour_segment_mask = analysis.bottom_connected_mask[level_y:]
            hsv_views.append(
                self._make_hsv_visualization(
                    colour_segment,
                    gauge_id,
                    colour_segment_mask,
                    level_y,
                    roi.shape[0],
                )
            )
            flap_strip_views.append(colour_segment)
            rectified_views.append(roi.copy())
            annotated_roi = roi.copy()
            cv2.rectangle(
                annotated_roi,
                (sx1, level_y),
                (sx2 - 1, roi.shape[0] - 1),
                (255, 255, 0),
                1,
            )
            cv2.line(
                annotated_roi, (0, level_y), (roi.shape[1] - 1, level_y), (0, 255, 255), 2
            )
            label = f"#{gauge_id} {reading:.1f}"
            cv2.putText(annotated_roi, label, (5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            annotated_views.append(annotated_roi)
            reports.append({
                "gauge_id": gauge_id,
                "gauge_box_xyxy": item["box"],
                "source_quad_tl_tr_br_bl": item["quad"],
                "perspective_applied": item["perspective_applied"],
                "perspective_matrix": item["perspective_matrix"],
                "rectified_size": [roi.shape[1], roi.shape[0]],
                "flap_strip_xyxy": [sx1, level_y, sx2, roi.shape[0]],
                "flap_strip_xyxy_rectified": [sx1, level_y, sx2, roi.shape[0]],
                "coloured_interval_y_rectified": [level_y, roi.shape[0]],
                "coloured_height": roi.shape[0] - level_y,
                "full_scale_height": roi.shape[0],
                "boundary_y_image": level_y,
                "boundary_y_rectified": level_y,
                "level_ratio": round(float(ratio), 6),
                "reading": round(reading, 3),
                "change_confidence_z": round(float(peak_z), 3),
                "continuity_confidence": round(float(peak_z), 3),
                "boundary_method": analysis.metrics["method"],
                "continuity_metrics": analysis.metrics,
                "hsv_channels": ["H", "S", "V", "BOTTOM_STATE"],
                "strip_selection_mode": "bottom_connected_coloured_interval_only",
                "strip_localization": analysis.metrics["localization"],
                "full_rectified_height_is_scale_range": True,
            })
        if not reports:
            detail = "；".join(formal_errors)
            raise ValueError(
                "没有找到满足底部起始和连续性条件的液位状态。"
                + (f" {detail}" if detail else "")
            )
        unrectified_visualization = self._stack_images(unrectified_views)
        rectified_visualization = self._stack_images(rectified_views)
        flap_strip_visualization = self._stack_images(flap_strip_views)
        annotated = self._stack_images(annotated_views)
        hsv_visualization = self._stack_hsv_views(hsv_views)
        return ReadingRecord(
            source_path=image_path.resolve(), original=image,
            unrectified_visualization=unrectified_visualization,
            rectified_visualization=rectified_visualization,
            flap_strip_visualization=flap_strip_visualization,
            annotated=annotated,
            hsv_visualization=hsv_visualization,
            reports=reports, unrectified_reports=unrectified_reports,
            scale_min=scale_min, scale_max=scale_max,
            source_mode=source_mode,
        )

    @staticmethod
    def _rectify_quad(
        image: np.ndarray,
        raw_quad: list[list[float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(raw_quad, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("四关键点必须是 TL、TR、BR、BL 顺序的 4×2 有效坐标。")
        points[:, 0] = np.clip(points[:, 0], 0, image.shape[1] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, image.shape[0] - 1)
        tl, tr, br, bl = points
        target_width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
        target_height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
        if target_width < 7 or target_height < 40:
            raise ValueError(
                f"关键点围成的区域过小：{target_width}×{target_height}。"
            )
        destination = np.asarray([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(points, destination)
        rectified = cv2.warpPerspective(
            image,
            matrix,
            (target_width, target_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rectified, matrix

    @staticmethod
    def _crop_quad_bounding_box(
        image: np.ndarray,
        raw_quad: list[list[float]],
    ) -> tuple[np.ndarray, list[int]]:
        """Crop the same keypoint target without applying perspective correction."""
        points = np.asarray(raw_quad, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("无法根据无效关键点生成未校正对比图。")
        x1 = max(0, int(np.floor(points[:, 0].min())))
        y1 = max(0, int(np.floor(points[:, 1].min())))
        x2 = min(image.shape[1], int(np.ceil(points[:, 0].max())) + 1)
        y2 = min(image.shape[0], int(np.ceil(points[:, 1].max())) + 1)
        if x2 - x1 < 7 or y2 - y1 < 40:
            raise ValueError(f"关键点外接矩形过小：{x2 - x1}×{y2 - y1}。")
        return image[y1:y2, x1:x2].copy(), [x1, y1, x2, y2]

    @staticmethod
    def _stack_images(images: list[np.ndarray]) -> np.ndarray:
        if len(images) == 1:
            return images[0]
        max_width = max(image.shape[1] for image in images)
        padded = []
        for image in images:
            right = max_width - image.shape[1]
            padded.append(cv2.copyMakeBorder(
                image, 0, 8, 0, right, cv2.BORDER_CONSTANT, value=(25, 25, 25)
            ))
        return cv2.vconcat(padded)

    @staticmethod
    def _make_hsv_visualization(
        strip: np.ndarray,
        gauge_id: int,
        bottom_connected_mask: np.ndarray,
        source_level_y: int,
        full_scale_height: int,
    ) -> np.ndarray:
        """Show only the extracted bottom-connected coloured indication interval."""
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        full = np.full_like(hue, 255)
        hue_colour = cv2.cvtColor(
            cv2.merge([hue, full, full]), cv2.COLOR_HSV2BGR
        )
        saturation_gray = cv2.cvtColor(saturation, cv2.COLOR_GRAY2BGR)
        value_gray = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
        bottom_state = np.zeros_like(value_gray)
        bottom_rows = np.asarray(bottom_connected_mask).reshape(-1).astype(bool)
        bottom_state[bottom_rows, :, 1] = 220
        bottom_state[~bottom_rows, :, 0] = 150

        source_height = strip.shape[0]
        panel_width = max(100, min(180, strip.shape[1] * 6))
        panels = []
        for title, panel in (
            ("HUE", hue_colour),
            ("SATURATION", saturation_gray),
            ("VALUE", value_gray),
            ("BOTTOM STATE", bottom_state),
        ):
            resized = cv2.resize(
                panel, (panel_width, source_height), interpolation=cv2.INTER_NEAREST
            )
            cv2.rectangle(
                resized, (0, 0), (panel_width - 1, source_height - 1), (0, 255, 255), 1
            )
            cv2.rectangle(resized, (0, 0), (panel_width - 1, 24), (0, 0, 0), -1)
            cv2.putText(
                resized,
                title,
                (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            boundary_display_y = min(24, source_height - 1)
            cv2.line(
                resized,
                (0, boundary_display_y),
                (panel_width - 1, boundary_display_y),
                (0, 0, 255),
                2,
            )
            panels.append(resized)
        combined = cv2.hconcat(panels)
        header = np.zeros((58, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            header,
            f"Gauge #{gauge_id}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            header,
            (
                f"Colour interval y={source_level_y}:{full_scale_height}; "
                f"height={strip.shape[0]}/{full_scale_height}"
            ),
            (8, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return cv2.vconcat([header, combined])

    @staticmethod
    def _stack_hsv_views(views: list[np.ndarray]) -> np.ndarray:
        if len(views) == 1:
            return views[0]
        max_width = max(view.shape[1] for view in views)
        padded = []
        for view in views:
            right = max_width - view.shape[1]
            padded.append(cv2.copyMakeBorder(
                view, 0, 8, 0, right, cv2.BORDER_CONSTANT, value=(25, 25, 25)
            ))
        return cv2.vconcat(padded)

    @staticmethod
    def save_record(record: ReadingRecord) -> tuple[Path, Path]:
        root = OUTPUT_DIR / "read_roi"
        input_path = write_image(
            root / "input_roi" / f"{record.source_path.stem}_input.png",
            record.original,
        )
        # Keep the old images path for compatibility and also save an explicitly
        # named detection result so the page-3 output is unambiguous.
        legacy_image_path = write_image(
            root / "images" / f"{record.source_path.stem}.png",
            record.annotated,
        )
        detection_result_path = write_image(
            root / "detection_results" / f"{record.source_path.stem}_detection_result.png",
            record.annotated,
        )
        unrectified_path = write_image(
            root / "unrectified_comparison" / f"{record.source_path.stem}_raw.png",
            record.unrectified_visualization,
        )
        rectified_path = write_image(
            root / "rectified_images" / f"{record.source_path.stem}_rectified.png",
            record.rectified_visualization,
        )
        flap_strip_path = write_image(
            root / "flap_color_strips" / f"{record.source_path.stem}_strip.png",
            record.flap_strip_visualization,
        )
        hsv_path = write_image(
            root / "hsv_results" / f"{record.source_path.stem}_hsv.png",
            record.hsv_visualization,
        )
        payload = {
            "source_image": str(record.source_path),
            "source_size": [record.original.shape[1], record.original.shape[0]],
            "scale_range": [record.scale_min, record.scale_max],
            "source_mode": record.source_mode,
            "gauges": record.reports,
            "unrectified_comparison": record.unrectified_reports,
            "input_roi_image": str(input_path),
            "unrectified_result_image": str(unrectified_path),
            "rectified_image": str(rectified_path),
            "flap_color_strip_image": str(flap_strip_path),
            "detection_result_image": str(detection_result_path),
            "result_image": str(detection_result_path),
            "legacy_result_image": str(legacy_image_path),
            "hsv_result_image": str(hsv_path),
        }
        json_path = write_json(root / "data" / f"{record.source_path.stem}.json", payload)
        record.saved_input = input_path
        record.saved_image = detection_result_path
        record.saved_detection_result = detection_result_path
        record.saved_unrectified = unrectified_path
        record.saved_rectified = rectified_path
        record.saved_flap_strip = flap_strip_path
        record.saved_hsv = hsv_path
        record.saved_json = json_path
        return detection_result_path, json_path


def load_boxes_from_json(path: Path) -> list[list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    detections = payload.get("detections")
    if detections is None and "gauges" in payload:
        detections = payload["gauges"]
    boxes = []
    for detection in detections or []:
        box = (
            detection.get("box_xyxy")
            or detection.get("box")
            or detection.get("gauge_box_xyxy")
        )
        if box and len(box) == 4:
            boxes.append([float(value) for value in box])
    return boxes


def load_quads_from_json(path: Path) -> list[list[list[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    quads = []
    for detection in payload.get("detections", []):
        quad = (
            detection.get("roi_quad_tl_tr_br_bl")
            or detection.get("keypoints_xy_pixels")
            or detection.get("keypoints")
        )
        if quad and len(quad) == 4 and all(len(point) == 2 for point in quad):
            quads.append([
                [float(point[0]), float(point[1])] for point in quad
            ])
    return quads
