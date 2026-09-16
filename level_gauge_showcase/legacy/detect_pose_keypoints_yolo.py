"""YOLO Pose inference, annotated-image output, and ROI keypoint export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

import cv2
import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[1]
DEFAULT_OUTPUT_ROOT = SCRIPT_ROOT / "pose_detection_results"

# 1. 关键点模型权重
MODEL_PATH = Path(
    PROJECT_ROOT / "object detection" / "runs" / "train"
    / "flap_pose4_with_yolo11s-set2" / "weights" / "best.pt"
)
# 2. 待检测图片文件夹
INPUT_PATH = Path(
    PROJECT_ROOT / "object detection" / "datasets" / "flap_keypoint_full" / "images" / "test"
)
# ===================================================================

# 一般无需修改。检测结果输出到 pose_detection_results 下。
CONFIDENCE = 0.20
INFERENCE_IMAGE_SIZE = 960
INFERENCE_BATCH_SIZE = 16
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
KEYPOINT_NAMES_4 = ("TL", "TR", "BR", "BL")
KEYPOINT_COLOURS = (
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO Pose and save an annotated image plus ROI keypoint JSON."
    )
    # All four positionals are optional so the script can be run directly after
    # editing MODEL_PATH and INPUT_PATH above. Existing callers may still pass
    # positional overrides without any interface change.
    parser.add_argument("model", nargs="?", default=str(MODEL_PATH), help="YOLO Pose weights")
    parser.add_argument(
        "image", nargs="?", default=str(INPUT_PATH),
        help="input image or a folder containing images",
    )
    parser.add_argument(
        "confidence", nargs="?", type=float, default=CONFIDENCE,
        help="box confidence threshold",
    )
    parser.add_argument(
        "imgsz", nargs="?", type=int, default=INFERENCE_IMAGE_SIZE,
        help="YOLO inference image size",
    )
    parser.add_argument(
        "--batch", type=int, default=INFERENCE_BATCH_SIZE,
        help="inference batch size when input is a folder",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="recursively scan subfolders",
    )
    parser.add_argument(
        "--output-root", default=str(DEFAULT_OUTPUT_ROOT),
        help="default: pose_detection_results beside this script",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="keep stdout JSON only; do not write image/keypoint files",
    )
    return parser.parse_args()


def class_name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


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
    annotated = image.copy()
    narrow = annotated.shape[1] < 300
    font_scale = 0.45 if narrow else 0.55
    thickness = 1 if narrow else 2

    for detection in detections:
        box = np.asarray(detection["box"], dtype=np.float32)
        x1, y1, x2, y2 = np.round(box).astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
        box_label = f"{detection['class_name']} {detection['box_confidence']:.3f}"
        if not narrow:
            label_origin = clamp_text_origin(
                annotated, box_label, x1, y1 - 6, font_scale, thickness
            )
            cv2.putText(
                annotated, box_label, label_origin,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0),
                thickness, cv2.LINE_AA,
            )

        points = np.asarray(detection["keypoints"], dtype=np.float32)
        confidences = np.asarray(detection["keypoint_confidences"], dtype=np.float32)
        if len(points) == 4:
            cv2.polylines(
                annotated, [np.round(points).astype(np.int32)], True,
                (255, 0, 255), 3, cv2.LINE_AA,
            )
        for point_index, ((x, y), point_confidence) in enumerate(zip(points, confidences)):
            centre = (int(round(float(x))), int(round(float(y))))
            colour = KEYPOINT_COLOURS[point_index % len(KEYPOINT_COLOURS)]
            cv2.circle(annotated, centre, 7, colour, -1, cv2.LINE_AA)
            point_name = (
                KEYPOINT_NAMES_4[point_index]
                if len(points) == 4 else f"K{point_index}"
            )
            point_label = point_name if narrow else f"{point_name} {point_confidence:.3f}"
            preferred_y = centre[1] + 18 if point_index < 2 else centre[1] - 8
            point_origin = clamp_text_origin(
                annotated, point_label, centre[0] + 5, preferred_y,
                font_scale, thickness,
            )
            cv2.putText(
                annotated, point_label, point_origin,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour,
                thickness, cv2.LINE_AA,
            )
    return annotated


def write_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"cannot encode output image: {path}")
    encoded.tofile(path)


def build_detection_records(prediction: object) -> list[dict]:
    records: list[dict] = []
    if prediction.boxes is None or prediction.keypoints is None:
        return records

    boxes = prediction.boxes.xyxy.cpu().numpy()
    box_confidences = prediction.boxes.conf.cpu().numpy()
    classes = prediction.boxes.cls.cpu().numpy().astype(int)
    keypoints = prediction.keypoints.xy.cpu().numpy()
    keypoints_normalized = prediction.keypoints.xyn.cpu().numpy()
    keypoint_confidences = (
        None if prediction.keypoints.conf is None
        else prediction.keypoints.conf.cpu().numpy()
    )

    for index in range(len(boxes)):
        class_id = int(classes[index])
        point_count = len(keypoints[index])
        confidences = (
            np.ones(point_count, dtype=np.float32)
            if keypoint_confidences is None
            else keypoint_confidences[index]
        )
        point_names = (
            list(KEYPOINT_NAMES_4)
            if point_count == 4
            else [f"K{point_index}" for point_index in range(point_count)]
        )
        records.append({
            "detection_id": index + 1,
            "class_id": class_id,
            "class_name": class_name(prediction.names, class_id),
            "box": boxes[index].astype(float).tolist(),
            "box_xyxy": boxes[index].astype(float).tolist(),
            "box_confidence": float(box_confidences[index]),
            # Keep backward-compatible fields for existing callers.
            "keypoints": keypoints[index].astype(float).tolist(),
            "keypoint_confidences": confidences.astype(float).tolist(),
            # Explicit fields for ROI perspective conversion.
            "keypoint_order": point_names,
            "keypoints_xy_pixels": keypoints[index].astype(float).tolist(),
            "keypoints_xy_normalized": keypoints_normalized[index].astype(float).tolist(),
            "roi_quad_tl_tr_br_bl": (
                keypoints[index].astype(float).tolist() if point_count == 4 else None
            ),
            "roi_perspective_ready": point_count == 4,
        })
    return records


def save_outputs(
    output_root: Path,
    image_path: Path,
    model_path: Path,
    prediction: object,
    detections: list[dict],
) -> tuple[Path, Path]:
    detected_image_dir = output_root / "detected_images"
    keypoint_data_dir = output_root / "keypoints"
    detected_image_dir.mkdir(parents=True, exist_ok=True)
    keypoint_data_dir.mkdir(parents=True, exist_ok=True)

    # Both folders use the input image stem for one-to-one matching.
    detected_image_path = detected_image_dir / f"{image_path.stem}.png"
    keypoint_json_path = keypoint_data_dir / f"{image_path.stem}.json"
    annotated = draw_pose_results(prediction.orig_img, detections)
    write_png(detected_image_path, annotated)

    image_height, image_width = prediction.orig_img.shape[:2]
    payload = {
        "source_image": str(image_path.resolve()),
        "source_filename": image_path.name,
        "source_stem": image_path.stem,
        "source_size": [int(image_width), int(image_height)],
        "model": str(model_path.resolve()),
        "keypoint_definition_for_4_points": list(KEYPOINT_NAMES_4),
        "coordinate_system": {
            "pixel_origin": "top-left",
            "pixel_x_direction": "right",
            "pixel_y_direction": "down",
            "normalized_range": [0.0, 1.0],
        },
        "detections": detections,
        "detected_image": str(detected_image_path.resolve()),
    }
    keypoint_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return detected_image_path.resolve(), keypoint_json_path.resolve()


def discover_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise SystemExit(f"unsupported image extension: {input_path}")
        return [input_path.resolve()]
    if not input_path.is_dir():
        raise SystemExit(f"input image/folder not found: {input_path}")
    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    images = sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"no supported images found in: {input_path}")
    return images


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    input_path = Path(args.image).resolve()
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")
    image_paths = discover_images(input_path, args.recursive)

    # Prevent silent overwrites when two input files share the same stem.
    stems = [path.stem.casefold() for path in image_paths]
    duplicate_stems = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicate_stems and not args.no_save:
        raise SystemExit(
            "duplicate image stems would overwrite outputs: " + ", ".join(duplicate_stems)
        )

    model = YOLO(str(model_path))
    predictions = model.predict(
        source=[str(path) for path in image_paths],
        conf=args.confidence,
        imgsz=args.imgsz,
        batch=max(1, args.batch),
        iou=0.7,
        max_det=10,
        verbose=False,
        stream=True,
    )

    output_root = Path(args.output_root).resolve()
    batch_records = []
    for index, (image_path, prediction) in enumerate(zip(image_paths, predictions), start=1):
        detections = build_detection_records(prediction)
        item = {
            "source_image": str(image_path),
            "detections": len(detections),
            "perspective_ready_detections": sum(
                bool(detection["roi_perspective_ready"]) for detection in detections
            ),
        }
        if not args.no_save:
            detected_image_path, keypoint_json_path = save_outputs(
                output_root, image_path, model_path, prediction, detections,
            )
            item["detected_image"] = str(detected_image_path)
            item["keypoint_json"] = str(keypoint_json_path)
        batch_records.append(item)
        print(
            f"[{index}/{len(image_paths)}] {image_path.name}: "
            f"detections={len(detections)}"
        )

        # Keep the exact single-image marker/interface used by
        # rectify_before_reading.py and other existing callers.
        if len(image_paths) == 1:
            if not args.no_save:
                print(f"POSE_DETECTED_IMAGE={item['detected_image']}")
                print(f"POSE_KEYPOINT_JSON={item['keypoint_json']}")
            print("POSE_RESULT_JSON=" + json.dumps(detections, ensure_ascii=True))

    if not args.no_save:
        summary = {
            "input": str(input_path),
            "model": str(model_path),
            "images": len(batch_records),
            "images_with_detections": sum(item["detections"] > 0 for item in batch_records),
            "total_detections": sum(item["detections"] for item in batch_records),
            "output_structure": {
                "detected_images": str((output_root / "detected_images").resolve()),
                "keypoints": str((output_root / "keypoints").resolve()),
            },
            "items": batch_records,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("========== 批量关键点检测完成 ==========")
        print(f"图片数量: {summary['images']}")
        print(f"检测成功图片: {summary['images_with_detections']}")
        print(f"检测目标总数: {summary['total_detections']}")
        print(f"检测图目录: {output_root / 'detected_images'}")
        print(f"关键点目录: {output_root / 'keypoints'}")
        print(f"汇总报告: {summary_path}")


if __name__ == "__main__":
    main()
