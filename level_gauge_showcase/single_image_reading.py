"""服务器端单张图片：Pose、ROI、透视前后液位读数。"""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path
from typing import Any

from ultralytics import YOLO

import remote_batch_reading as batch
from services import ObjectDetectionService, ReadingService, draw_pose_results
from legacy.detect_pose_keypoints_yolo import build_detection_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单张磁翻板液位计关键点检测与读数")
    parser.add_argument("image", nargs="?", type=Path, help="单张图片路径")
    parser.add_argument("--scale-min", type=float, help="量程下限；省略时交互输入")
    parser.add_argument("--scale-max", type=float, help="量程上限；省略时交互输入")
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=Path(
            "/workspace/runs/pose/flap_pose4_with_yolo11s-fullset/weights/best.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/level_gauge_batch/single_results"),
    )
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--reader", choices=("app", "strict", "legacy"), default="app"
    )
    return parser.parse_args()


def container_path(path: Path) -> Path:
    """Allow pasting either the host bind path or the /workspace path."""
    text = str(path).strip().strip('"').replace("\\", "/")
    host_prefix = os.environ.get("LEVEL_GAUGE_HOST_PREFIX", "/host/project").rstrip("/")
    if text == host_prefix or text.startswith(host_prefix + "/"):
        text = "/workspace" + text[len(host_prefix):]
    return Path(text).resolve()


def prompt_image(value: Path | None) -> Path:
    if value is None:
        text = input("请输入单张图片路径: ").strip()
        if not text:
            raise SystemExit("未输入图片路径。")
        value = Path(text)
    path = container_path(value)
    if not path.is_file():
        raise SystemExit(f"图片不存在：{path}")
    return path


def prompt_scale(value: float | None, label: str, default: float | None) -> float:
    if value is not None:
        return float(value)
    suffix = f"（回车默认 {default:g}）" if default is not None else ""
    while True:
        text = input(f"{label}{suffix}: ").strip()
        if not text and default is not None:
            return default
        try:
            return float(text)
        except ValueError:
            print("请输入有效数字。")


def main() -> None:
    args = parse_args()
    image_path = prompt_image(args.image)
    scale_min = prompt_scale(args.scale_min, "请输入量程下限", 0.0)
    scale_max = prompt_scale(args.scale_max, "请输入量程上限", None)
    if scale_max <= scale_min:
        raise SystemExit("量程上限必须大于量程下限。")

    model_path = container_path(args.pose_model)
    if not model_path.is_file():
        raise SystemExit(f"关键点模型不存在：{model_path}")
    output_root = container_path(args.output)
    image_dir = output_root / image_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)
    reader_name = batch.configure_reader(args.reader)

    print("========== 单张液位计检测 ==========")
    print(f"图片：{image_path}")
    print(f"量程：{scale_min:g} ～ {scale_max:g}")
    print(f"关键点模型：{model_path}")
    print(f"读数算法：{reader_name}")

    payload: dict[str, Any] = {
        "image_name": image_path.name,
        "source_image": str(image_path),
        "scale_range": [scale_min, scale_max],
        "actual_reading": None,
        "pose_model": str(model_path),
        "pose_confidence": args.confidence,
        "pose_imgsz": args.imgsz,
        "device": str(args.device),
        "reading_algorithm": reader_name,
        "status": "processing",
        "pose_detections": [],
        "gauges": [],
        "outputs": {},
    }
    try:
        original = batch.read_image(image_path)
        model = YOLO(str(model_path))
        prediction = model.predict(
            source=str(image_path),
            conf=args.confidence,
            imgsz=args.imgsz,
            device=args.device,
            iou=0.7,
            max_det=10,
            verbose=False,
        )[0]
        detections = build_detection_records(prediction)
        payload["pose_detections"] = detections
        payload["outputs"]["pose_detected_image"] = batch.write_image(
            image_dir / "pose_detected.png",
            draw_pose_results(original, detections),
        )
        payload["outputs"]["pose_json"] = batch.write_json(
            image_dir / "pose_keypoints.json",
            {
                "source_image": str(image_path),
                "source_size": [original.shape[1], original.shape[0]],
                "model": str(model_path),
                "confidence": args.confidence,
                "imgsz": args.imgsz,
                "keypoint_order": ["TL", "TR", "BR", "BL"],
                "detections": detections,
            },
        )
        reading_service = ReadingService(ObjectDetectionService(), None)
        if not detections:
            payload["status"] = "pose_not_detected"
        for gauge_id, detection in enumerate(detections, start=1):
            gauge_dir = image_dir / f"gauge_{gauge_id:02d}"
            gauge: dict[str, Any] = {
                "gauge_id": gauge_id,
                "status": "processing",
                "detection": detection,
            }
            try:
                roi, local_quad, source_box = batch.crop_detection(original, detection)
                roi_path = Path(batch.write_image(gauge_dir / "01_roi.png", roi))
                record = reading_service.read(
                    roi_path,
                    scale_min,
                    scale_max,
                    call_model=False,
                    existing_quads=[local_quad],
                )
                gauge.update({
                    "status": "ok",
                    "source_box_xyxy": source_box,
                    "local_quad_tl_tr_br_bl": local_quad,
                    "rectified_report": record.reports[0] if record.reports else None,
                    "unrectified_report": (
                        record.unrectified_reports[0]
                        if record.unrectified_reports else None
                    ),
                    "outputs": batch.save_reading_outputs(gauge_dir, roi, record),
                })
            except Exception as exc:
                gauge.update({
                    "status": "reading_failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
            payload["gauges"].append(gauge)
        if detections:
            payload["status"] = (
                "ok"
                if any(gauge["status"] == "ok" for gauge in payload["gauges"])
                else "reading_failed"
            )
    except Exception as exc:
        payload.update({
            "status": "pose_inference_failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })

    result_path = image_dir / "result.json"
    batch.write_json(result_path, payload)
    csv_path = image_dir / "reading_statistics.csv"
    saved_rows = batch.rows_from_saved_result(result_path)
    row_map = {
        (row["image_name"], str(row["gauge_id"])): row for row in saved_rows
    }
    old_rows = batch.load_existing_rows(csv_path)
    batch.save_csv(csv_path, row_map, old_rows)

    print(f"状态：{payload['status']}")
    for gauge in payload["gauges"]:
        raw = (gauge.get("unrectified_report") or {}).get("reading")
        rectified = (gauge.get("rectified_report") or {}).get("reading")
        print(
            f"液位计 {gauge['gauge_id']}：{gauge['status']}；"
            f"未透视={raw}；透视后={rectified}"
        )
    print(f"结果 JSON：{result_path}")
    print(f"结果 CSV：{csv_path}")


if __name__ == "__main__":
    main()
