"""服务器端关键点定位、ROI 裁剪、透视前后读数交互式批处理。

该脚本不启动 GUI，关键点模型只加载一次。每张图片提示手动量程，处理后立即
保存独立 JSON 和汇总 CSV；actual_reading 列留空供人工填写。再次运行默认
跳过已经完成的图片，并保留 CSV 中已经填写的 actual_reading。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

from ultralytics import YOLO
import cv2
import numpy as np

import services
from services import ObjectDetectionService, ReadingService, draw_pose_results
from legacy.detect_pose_keypoints_yolo import build_detection_records


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CSV_FIELDS = [
    "image_name",
    "actual_reading",
    "scale_min",
    "scale_max",
    "pose_status",
    "pose_detections",
    "gauge_id",
    "box_confidence",
    "min_keypoint_confidence",
    "mean_keypoint_confidence",
    "reading_status",
    "unrectified_reading",
    "rectified_reading",
    "rectified_minus_unrectified",
    "absolute_difference",
    "absolute_difference_full_scale_percent",
    "unrectified_boundary_y",
    "rectified_boundary_y",
    "roi_path",
    "pose_json",
    "reading_json",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量关键点检测、ROI裁剪及透视前后液位读数"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/workspace/datasets/flap_keypoint_full/images/test"),
        help="测试图片目录",
    )
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=Path(
            "/workspace/runs/pose/flap_pose4_with_yolo11s-fullset/weights/best.pt"
        ),
        help="YOLO Pose 权重",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/level_gauge_batch/batch_results"),
        help="批处理输出目录",
    )
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0", help="Ultralytics device，例如 0 或 cpu")
    parser.add_argument(
        "--reader",
        choices=("app", "strict", "legacy"),
        default="app",
        help="app=沿用 services.py 当前读数入口；strict/legacy 可显式指定",
    )
    parser.add_argument(
        "--range-csv",
        type=Path,
        help="可选量程表，列名 image_name,scale_min,scale_max；未命中仍会询问",
    )
    parser.add_argument("--start", help="从指定文件名或 stem 开始")
    parser.add_argument("--no-resume", action="store_true", help="重新处理已有结果")
    return parser.parse_args()


def configure_reader(name: str) -> str:
    if name == "strict":
        from direct_hsv_reader import analyze_direct_flap_strip

        services.analyze_direct_flap_strip = analyze_direct_flap_strip
    elif name == "legacy":
        from legacy_candidate_reader import analyze_direct_flap_strip

        services.analyze_direct_flap_strip = analyze_direct_flap_strip
    function = services.analyze_direct_flap_strip
    return f"{function.__module__}.{function.__name__}"


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"无法编码图片：{path}")
    encoded.tofile(path)
    return str(path.resolve())


def write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path.resolve())


def discover_images(source: Path) -> list[Path]:
    if not source.is_dir():
        raise FileNotFoundError(f"测试图片目录不存在：{source}")
    return sorted(
        path.resolve()
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_ranges(path: Path | None) -> dict[str, tuple[float, float]]:
    if path is None:
        return {}
    mapping: dict[str, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("image_name") or row.get("image") or "").strip()
            if not name:
                continue
            try:
                lower = float(row["scale_min"])
                upper = float(row["scale_max"])
            except (KeyError, TypeError, ValueError):
                continue
            if upper > lower:
                mapping[name] = (lower, upper)
                mapping[Path(name).stem] = (lower, upper)
    return mapping


def prompt_number(label: str, default: float | None) -> float | str:
    default_text = f"，回车沿用 {default:g}" if default is not None else ""
    while True:
        text = input(f"{label}{default_text}（q退出，s跳过）: ").strip()
        if text.lower() in {"q", "s"}:
            return text.lower()
        if not text and default is not None:
            return default
        try:
            return float(text)
        except ValueError:
            print("请输入有效数字、q 或 s。")


def prompt_range(
    image: Path,
    index: int,
    total: int,
    previous: tuple[float, float] | None,
    configured: dict[str, tuple[float, float]],
) -> tuple[float, float] | str:
    preset = configured.get(image.name) or configured.get(image.stem)
    if preset is not None:
        print(f"\n[{index}/{total}] {image.name}，量程表：{preset[0]:g} ～ {preset[1]:g}")
        return preset
    print(f"\n[{index}/{total}] {image.name}")
    lower_default = previous[0] if previous else 0.0
    upper_default = previous[1] if previous else None
    lower = prompt_number("请输入量程下限", lower_default)
    if isinstance(lower, str):
        return lower
    upper = prompt_number("请输入量程上限", upper_default)
    if isinstance(upper, str):
        return upper
    if upper <= lower:
        print("量程上限必须大于量程下限，请重新输入。")
        return prompt_range(image, index, total, previous, configured)
    return lower, upper


def load_existing_rows(csv_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            (row.get("image_name", ""), row.get("gauge_id", "")): row
            for row in csv.DictReader(handle)
        }


def save_csv(
    csv_path: Path,
    rows: dict[tuple[str, str], dict[str, Any]],
    old_rows: dict[tuple[str, str], dict[str, str]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda row: (row["image_name"], str(row["gauge_id"])))
    for row in ordered:
        key = (str(row["image_name"]), str(row["gauge_id"]))
        # 人工填写的真实读数在续跑或重算时不得被覆盖。
        row["actual_reading"] = old_rows.get(key, {}).get(
            "actual_reading", row.get("actual_reading", "")
        )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def crop_detection(
    image: np.ndarray, detection: dict[str, Any]
) -> tuple[np.ndarray, list[list[float]], list[int]]:
    box = detection.get("box_xyxy") or detection.get("box")
    quad = detection.get("roi_quad_tl_tr_br_bl") or detection.get("keypoints")
    if not box or len(box) != 4 or not quad or len(quad) != 4:
        raise ValueError("检测结果缺少有效框或4个关键点。")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(float(value))) for value in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 7 or y2 - y1 < 40:
        raise ValueError(f"ROI过小：[{x1}, {y1}, {x2}, {y2}]")
    local_quad = [
        [float(point[0]) - x1, float(point[1]) - y1]
        for point in quad
    ]
    return image[y1:y2, x1:x2].copy(), local_quad, [x1, y1, x2, y2]


def confidence_values(detection: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    box_confidence = finite(
        detection.get("box_confidence", detection.get("confidence"))
    )
    values = [
        value
        for value in (finite(item) for item in detection.get("keypoint_confidences", []))
        if value is not None
    ]
    return (
        box_confidence,
        min(values) if values else None,
        sum(values) / len(values) if values else None,
    )


def save_reading_outputs(
    gauge_dir: Path,
    roi: np.ndarray,
    record: services.ReadingRecord,
) -> dict[str, str]:
    return {
        "roi": write_image(gauge_dir / "01_roi.png", roi),
        "unrectified": write_image(
            gauge_dir / "02_unrectified_reading.png",
            record.unrectified_visualization,
        ),
        "rectified": write_image(
            gauge_dir / "03_rectified.png", record.rectified_visualization
        ),
        "reading_result": write_image(
            gauge_dir / "04_reading_result.png", record.annotated
        ),
        "colour_interval": write_image(
            gauge_dir / "05_colour_interval.png", record.flap_strip_visualization
        ),
        "hsv": write_image(gauge_dir / "06_hsv.png", record.hsv_visualization),
    }


def row_from_gauge(
    image_name: str,
    scale_min: float,
    scale_max: float,
    detections: int,
    gauge_id: int,
    detection: dict[str, Any],
    gauge_payload: dict[str, Any],
    pose_json: str,
    reading_json: str,
) -> dict[str, Any]:
    box_confidence, min_kpt, mean_kpt = confidence_values(detection)
    rectified = gauge_payload.get("rectified_report") or {}
    raw = gauge_payload.get("unrectified_report") or {}
    rectified_reading = finite(rectified.get("reading"))
    raw_reading = finite(raw.get("reading"))
    difference = (
        rectified_reading - raw_reading
        if rectified_reading is not None and raw_reading is not None
        else None
    )
    span = scale_max - scale_min
    return {
        "image_name": image_name,
        "actual_reading": "",
        "scale_min": scale_min,
        "scale_max": scale_max,
        "pose_status": "ok",
        "pose_detections": detections,
        "gauge_id": gauge_id,
        "box_confidence": box_confidence,
        "min_keypoint_confidence": min_kpt,
        "mean_keypoint_confidence": mean_kpt,
        "reading_status": gauge_payload["status"],
        "unrectified_reading": raw_reading,
        "rectified_reading": rectified_reading,
        "rectified_minus_unrectified": difference,
        "absolute_difference": abs(difference) if difference is not None else None,
        "absolute_difference_full_scale_percent": (
            abs(difference) / span * 100.0 if difference is not None and span > 0 else None
        ),
        "unrectified_boundary_y": raw.get("boundary_y"),
        "rectified_boundary_y": rectified.get("boundary_y_rectified"),
        "roi_path": gauge_payload.get("outputs", {}).get("roi", ""),
        "pose_json": pose_json,
        "reading_json": reading_json,
        "error": gauge_payload.get("error", ""),
    }


def rows_from_saved_result(result_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    rows = []
    detections = payload.get("pose_detections", [])
    for gauge in payload.get("gauges", []):
        detection_index = int(gauge.get("gauge_id", 1)) - 1
        detection = detections[detection_index] if detection_index < len(detections) else {}
        rows.append(row_from_gauge(
            payload["image_name"],
            float(payload["scale_range"][0]),
            float(payload["scale_range"][1]),
            len(detections),
            int(gauge.get("gauge_id", 1)),
            detection,
            gauge,
            payload.get("outputs", {}).get("pose_json", ""),
            str(result_path.resolve()),
        ))
    if not rows:
        rows.append({
            "image_name": payload.get("image_name", result_path.parent.name),
            "actual_reading": "",
            "scale_min": (payload.get("scale_range") or [None, None])[0],
            "scale_max": (payload.get("scale_range") or [None, None])[1],
            "pose_status": payload.get("status", "failed"),
            "pose_detections": len(detections),
            "gauge_id": 0,
            "reading_status": payload.get("status", "failed"),
            "reading_json": str(result_path.resolve()),
            "error": payload.get("error", ""),
        })
    return rows


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    model_path = args.pose_model.resolve()
    output = args.output.resolve()
    if not model_path.is_file():
        raise SystemExit(f"关键点模型不存在：{model_path}")
    if not 0.0 < args.confidence <= 1.0:
        raise SystemExit("confidence 必须在 0 到 1 之间。")
    images = discover_images(source)
    if args.start:
        start = args.start.casefold()
        images = [
            image for image in images
            if image.name.casefold() >= start or image.stem.casefold() >= start
        ]
    if not images:
        raise SystemExit("没有找到待处理图片。")

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "reading_statistics.csv"
    old_rows = load_existing_rows(csv_path)
    all_rows: dict[tuple[str, str], dict[str, Any]] = dict(old_rows)
    ranges = load_ranges(args.range_csv.resolve() if args.range_csv else None)
    reader_name = configure_reader(args.reader)

    print("========== 服务器批量液位读数 ==========")
    print(f"测试图片：{source}")
    print(f"图片数量：{len(images)}")
    print(f"关键点模型：{model_path}")
    print(f"读数算法：{reader_name}")
    print(f"输出目录：{output}")
    print("提示：量程输入时 q=保存并退出，s=跳过当前图片。")
    print("正在加载关键点模型……")

    model = YOLO(str(model_path))
    reading_service = ReadingService(ObjectDetectionService(), None)
    previous_range: tuple[float, float] | None = None

    for index, image_path in enumerate(images, start=1):
        image_dir = output / image_path.stem
        result_path = image_dir / "result.json"
        if result_path.exists() and not args.no_resume:
            try:
                for row in rows_from_saved_result(result_path):
                    all_rows[(row["image_name"], str(row["gauge_id"]))] = row
                print(f"[{index}/{len(images)}] 已完成，跳过：{image_path.name}")
                continue
            except Exception as exc:
                print(f"已有结果损坏，将重新处理：{result_path}；{exc}")

        selected_range = prompt_range(
            image_path, index, len(images), previous_range, ranges
        )
        if selected_range == "q":
            save_csv(csv_path, all_rows, old_rows)
            print(f"已保存当前进度：{csv_path}")
            return
        if selected_range == "s":
            payload = {
                "image_name": image_path.name,
                "source_image": str(image_path),
                "status": "skipped_by_user",
                "pose_model": str(model_path),
                "reading_algorithm": reader_name,
                "pose_detections": [],
                "gauges": [],
            }
            write_json(result_path, payload)
            row = rows_from_saved_result(result_path)[0]
            all_rows[(image_path.name, "0")] = row
            save_csv(csv_path, all_rows, old_rows)
            continue

        scale_min, scale_max = selected_range
        previous_range = selected_range
        image_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "image_name": image_path.name,
            "source_image": str(image_path),
            "scale_range": [scale_min, scale_max],
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
            original = read_image(image_path)
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
            pose_annotated = draw_pose_results(original, detections)
            payload["outputs"]["pose_detected_image"] = write_image(
                image_dir / "pose_detected.png", pose_annotated
            )
            pose_json_path = write_json(
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
            payload["outputs"]["pose_json"] = pose_json_path
            if not detections:
                payload["status"] = "pose_not_detected"
            else:
                for gauge_id, detection in enumerate(detections, start=1):
                    gauge_dir = image_dir / f"gauge_{gauge_id:02d}"
                    gauge_payload: dict[str, Any] = {
                        "gauge_id": gauge_id,
                        "status": "processing",
                        "detection": detection,
                    }
                    try:
                        roi, local_quad, source_box = crop_detection(original, detection)
                        roi_path = Path(write_image(gauge_dir / "01_roi.png", roi))
                        record = reading_service.read(
                            roi_path,
                            scale_min,
                            scale_max,
                            call_model=False,
                            existing_boxes=None,
                            existing_quads=[local_quad],
                        )
                        outputs = save_reading_outputs(gauge_dir, roi, record)
                        gauge_payload.update({
                            "status": "ok",
                            "source_box_xyxy": source_box,
                            "local_quad_tl_tr_br_bl": local_quad,
                            "rectified_report": record.reports[0] if record.reports else None,
                            "unrectified_report": (
                                record.unrectified_reports[0]
                                if record.unrectified_reports else None
                            ),
                            "outputs": outputs,
                        })
                    except Exception as exc:
                        gauge_payload.update({
                            "status": "reading_failed",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        })
                    payload["gauges"].append(gauge_payload)
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

        reading_json = write_json(result_path, payload)
        new_rows = rows_from_saved_result(result_path)
        for row in new_rows:
            row["reading_json"] = reading_json
            all_rows[(row["image_name"], str(row["gauge_id"]))] = row
        save_csv(csv_path, all_rows, old_rows)
        ok_count = sum(gauge.get("status") == "ok" for gauge in payload["gauges"])
        print(
            f"完成：{image_path.name}；Pose目标={len(payload['pose_detections'])}；"
            f"成功读数={ok_count}；状态={payload['status']}"
        )
        print(f"单图 JSON：{result_path}")

    save_csv(csv_path, all_rows, old_rows)
    summary = {
        "source": str(source),
        "pose_model": str(model_path),
        "reading_algorithm": reader_name,
        "image_count": len(images),
        "csv": str(csv_path),
        "output": str(output),
    }
    write_json(output / "batch_summary.json", summary)
    print("========== 全部处理完成 ==========")
    print(f"统计 CSV：{csv_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断。已完成图片的数据此前已经逐张保存。")
        sys.exit(130)
