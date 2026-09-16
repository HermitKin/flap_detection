"""汇总展示程序的目标检测、关键点检测和 ReadROI 读数结果。

只使用已经保存的 JSON，不重新运行模型。人工真实读数填入
ground_truth_readings.csv 后，再运行本脚本即可得到 MAE/RMSE/命中率等指标。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECT_DIR = ROOT / "outputs" / "object_detection" / "data"
DEFAULT_POSE_DIR = ROOT / "outputs" / "keypoint_detection" / "keypoints"
DEFAULT_READING_DIR = ROOT / "outputs" / "read_roi" / "data"
DEFAULT_GT = Path(__file__).resolve().parent / "ground_truth_readings.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def rounded(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def xyxy_iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def resolve_yolo_label(source: Path) -> Path | None:
    parts = list(source.parts)
    indices = [index for index, part in enumerate(parts) if part.lower() == "images"]
    for index in reversed(indices):
        candidate_parts = parts.copy()
        candidate_parts[index] = "labels"
        candidate = Path(*candidate_parts).with_suffix(".txt")
        if candidate.exists():
            return candidate
    return None


def read_gt_boxes(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        cx, cy, bw, bh = map(float, fields[1:5])
        boxes.append([
            (cx - bw / 2.0) * width,
            (cy - bh / 2.0) * height,
            (cx + bw / 2.0) * width,
            (cy + bh / 2.0) * height,
        ])
    return boxes


def greedy_match(
    predictions: list[list[float]], ground_truth: list[list[float]], threshold: float
) -> tuple[int, int, int, list[float]]:
    pairs = sorted(
        (
            (xyxy_iou(prediction, target), prediction_index, target_index)
            for prediction_index, prediction in enumerate(predictions)
            for target_index, target in enumerate(ground_truth)
        ),
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    matched_ious: list[float] = []
    for iou, prediction_index, target_index in pairs:
        if iou < threshold:
            break
        if prediction_index in used_predictions or target_index in used_targets:
            continue
        used_predictions.add(prediction_index)
        used_targets.add(target_index)
        matched_ious.append(iou)
    true_positive = len(matched_ious)
    return (
        true_positive,
        len(predictions) - true_positive,
        len(ground_truth) - true_positive,
        matched_ious,
    )


def evaluate_object_detection(data_dir: Path, iou_threshold: float) -> dict[str, Any]:
    files = sorted(data_dir.glob("*.json")) if data_dir.exists() else []
    attempted = detected_images = total_detections = 0
    confidence_values: list[float] = []
    evaluated_with_gt = tp = fp = fn = 0
    matched_ious: list[float] = []
    rows: list[dict[str, Any]] = []
    for path in files:
        record = load_json(path)
        detections = record.get("detections") or []
        attempted += 1
        detected_images += bool(detections)
        total_detections += len(detections)
        predictions: list[list[float]] = []
        for detection in detections:
            box = detection.get("box_xyxy") or detection.get("box")
            if box and len(box) == 4:
                predictions.append([float(item) for item in box])
            confidence = finite_number(
                detection.get("confidence", detection.get("box_confidence"))
            )
            if confidence is not None:
                confidence_values.append(confidence)

        source = Path(record.get("source_image", ""))
        size = record.get("source_size") or [0, 0]
        label = resolve_yolo_label(source) if source else None
        row_tp = row_fp = row_fn = None
        row_mean_iou = None
        if label and len(size) >= 2:
            gt_boxes = read_gt_boxes(label, int(size[0]), int(size[1]))
            row_tp, row_fp, row_fn, row_ious = greedy_match(
                predictions, gt_boxes, iou_threshold
            )
            evaluated_with_gt += 1
            tp += row_tp
            fp += row_fp
            fn += row_fn
            matched_ious.extend(row_ious)
            row_mean_iou = mean(row_ious) if row_ious else None
        rows.append({
            "result_json": str(path.resolve()),
            "source_image": str(source),
            "detections": len(detections),
            "has_ground_truth": bool(label),
            "tp": row_tp,
            "fp": row_fp,
            "fn": row_fn,
            "mean_matched_iou": rounded(row_mean_iou),
        })

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "attempted_images": attempted,
        "images_with_detections": detected_images,
        "image_detection_rate": rounded(safe_div(detected_images, attempted)),
        "image_miss_rate": rounded(safe_div(attempted - detected_images, attempted)),
        "total_detections": total_detections,
        "mean_detection_confidence": rounded(mean(confidence_values)) if confidence_values else None,
        "gt_evaluated_images": evaluated_with_gt,
        "iou_threshold": iou_threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": rounded(precision),
        "recall": rounded(recall),
        "f1": rounded(f1),
        "false_discovery_rate": rounded(safe_div(fp, tp + fp)),
        "miss_rate": rounded(safe_div(fn, tp + fn)),
        "mean_matched_iou": rounded(mean(matched_ious)) if matched_ious else None,
        "rows": rows,
        "note": "TP/FP/FN 仅统计能够找到 YOLO GT 标签的图片；mAP 需要完整置信度序列，不能由 GUI 筛选后的 JSON 可靠还原。",
    }


def evaluate_pose_detection(data_dir: Path, confidence_threshold: float) -> dict[str, Any]:
    files = sorted(data_dir.glob("*.json")) if data_dir.exists() else []
    attempted = detected_images = detections_total = perspective_ready = 0
    box_confidences: list[float] = []
    min_keypoint_confidences: list[float] = []
    mean_keypoint_confidences: list[float] = []
    low_confidence_detections = 0
    rows: list[dict[str, Any]] = []
    for path in files:
        record = load_json(path)
        detections = record.get("detections") or []
        attempted += 1
        detected_images += bool(detections)
        detections_total += len(detections)
        for detection in detections:
            box_confidence = finite_number(
                detection.get("box_confidence", detection.get("confidence"))
            )
            if box_confidence is not None:
                box_confidences.append(box_confidence)
            confidences = [
                number
                for number in (
                    finite_number(item)
                    for item in detection.get("keypoint_confidences", [])
                )
                if number is not None
            ]
            minimum = min(confidences) if confidences else None
            average = mean(confidences) if confidences else None
            if minimum is not None:
                min_keypoint_confidences.append(minimum)
                low_confidence_detections += minimum < confidence_threshold
            if average is not None:
                mean_keypoint_confidences.append(average)
            ready = bool(detection.get("roi_perspective_ready"))
            perspective_ready += ready
            rows.append({
                "result_json": str(path.resolve()),
                "source_image": record.get("source_image", ""),
                "detection_id": detection.get("detection_id"),
                "box_confidence": rounded(box_confidence),
                "min_keypoint_confidence": rounded(minimum),
                "mean_keypoint_confidence": rounded(average),
                "perspective_ready": ready,
                "low_keypoint_confidence": (
                    minimum < confidence_threshold if minimum is not None else True
                ),
            })
    return {
        "attempted_images": attempted,
        "images_with_detections": detected_images,
        "image_detection_rate": rounded(safe_div(detected_images, attempted)),
        "image_miss_rate": rounded(safe_div(attempted - detected_images, attempted)),
        "total_detections": detections_total,
        "perspective_ready_detections": perspective_ready,
        "perspective_ready_rate": rounded(safe_div(perspective_ready, detections_total)),
        "keypoint_confidence_threshold": confidence_threshold,
        "low_keypoint_confidence_detections": low_confidence_detections,
        "low_keypoint_confidence_rate": rounded(
            safe_div(low_confidence_detections, detections_total)
        ),
        "mean_box_confidence": rounded(mean(box_confidences)) if box_confidences else None,
        "mean_min_keypoint_confidence": (
            rounded(mean(min_keypoint_confidences)) if min_keypoint_confidences else None
        ),
        "mean_keypoint_confidence": (
            rounded(mean(mean_keypoint_confidences)) if mean_keypoint_confidences else None
        ),
        "rows": rows,
        "note": "这里统计运行成功率和置信度质量；关键点 PCK/NME 需要与同一坐标系下的人工关键点 GT 配对。",
    }


def reading_rows(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = sorted(data_dir.glob("*.json")) if data_dir.exists() else []
    for path in files:
        record = load_json(path)
        source_image = record.get("source_image", "")
        scale = record.get("scale_range") or [None, None]
        raw_by_id = {
            item.get("gauge_id"): item
            for item in record.get("unrectified_comparison", [])
        }
        for gauge in record.get("gauges", []):
            gauge_id = gauge.get("gauge_id", 1)
            raw = raw_by_id.get(gauge_id, {})
            rectified = finite_number(gauge.get("reading"))
            unrectified = finite_number(raw.get("reading"))
            lower = finite_number(scale[0] if len(scale) > 0 else None)
            upper = finite_number(scale[1] if len(scale) > 1 else None)
            span = upper - lower if lower is not None and upper is not None else None
            sample_id = f"{path.stem}__g{int(gauge_id):02d}"
            delta = (
                rectified - unrectified
                if rectified is not None and unrectified is not None
                else None
            )
            rows.append({
                "sample_id": sample_id,
                "result_json": str(path.resolve()),
                "source_image": source_image,
                "gauge_id": gauge_id,
                "scale_min": lower,
                "scale_max": upper,
                "scale_span": span,
                "unrectified_reading": unrectified,
                "rectified_reading": rectified,
                "rectified_minus_unrectified": delta,
                "abs_rectified_unrectified_difference": abs(delta) if delta is not None else None,
                "unrectified_success": unrectified is not None,
                "rectified_success": rectified is not None,
                "perspective_applied": gauge.get("perspective_applied", False),
                "rectified_confidence": finite_number(gauge.get("continuity_confidence")),
                "unrectified_confidence": finite_number(raw.get("continuity_confidence")),
                "actual_reading": None,
                "notes": "",
            })
    return rows


def load_or_create_ground_truth(path: Path, rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    existing: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("sample_id"):
                    existing[row["sample_id"]] = row
    fieldnames = [
        "sample_id", "source_image", "gauge_id", "scale_min", "scale_max",
        "actual_reading", "include", "notes",
    ]
    merged: list[dict[str, Any]] = []
    for row in rows:
        previous = existing.get(row["sample_id"], {})
        merged.append({
            "sample_id": row["sample_id"],
            "source_image": row["source_image"],
            "gauge_id": row["gauge_id"],
            "scale_min": row["scale_min"],
            "scale_max": row["scale_max"],
            "actual_reading": previous.get("actual_reading", ""),
            "include": previous.get("include", "1"),
            "notes": previous.get("notes", ""),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)
    return {row["sample_id"]: row for row in merged}


def regression_metrics(
    rows: Iterable[dict[str, Any]], prediction_field: str
) -> dict[str, Any] | None:
    pairs: list[tuple[float, float, float | None]] = []
    for row in rows:
        actual = finite_number(row.get("actual_reading"))
        predicted = finite_number(row.get(prediction_field))
        if actual is not None and predicted is not None:
            pairs.append((predicted, actual, finite_number(row.get("scale_span"))))
    if not pairs:
        return None
    errors = [prediction - actual for prediction, actual, _ in pairs]
    absolute_errors = [abs(value) for value in errors]
    squared_errors = [value * value for value in errors]
    percentage_errors = [
        abs(prediction - actual) / abs(actual)
        for prediction, actual, _ in pairs
        if abs(actual) > 1e-12
    ]
    normalized_errors = [
        abs(prediction - actual) / span
        for prediction, actual, span in pairs
        if span is not None and span > 0
    ]
    result: dict[str, Any] = {
        "samples": len(pairs),
        "mae": rounded(mean(absolute_errors)),
        "rmse": rounded(math.sqrt(mean(squared_errors))),
        "median_absolute_error": rounded(median(absolute_errors)),
        "max_absolute_error": rounded(max(absolute_errors)),
        "bias": rounded(mean(errors)),
        "mape": rounded(mean(percentage_errors)) if percentage_errors else None,
        "normalized_mae_full_scale": (
            rounded(mean(normalized_errors)) if normalized_errors else None
        ),
    }
    for tolerance in (0.01, 0.02, 0.05):
        eligible = [
            abs(prediction - actual) <= tolerance * span
            for prediction, actual, span in pairs
            if span is not None and span > 0
        ]
        result[f"within_{int(tolerance * 100)}pct_full_scale_rate"] = (
            rounded(mean(eligible)) if eligible else None
        )
    result["misread_rate_over_5pct_full_scale"] = (
        rounded(mean(value > 0.05 for value in normalized_errors))
        if normalized_errors else None
    )
    return result


def evaluate_readings(
    data_dir: Path, ground_truth_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = reading_rows(data_dir)
    ground_truth = load_or_create_ground_truth(ground_truth_path, rows)
    for row in rows:
        gt = ground_truth.get(row["sample_id"], {})
        include = str(gt.get("include", "1")).strip().lower() not in {"0", "false", "no"}
        row["actual_reading"] = finite_number(gt.get("actual_reading")) if include else None
        row["notes"] = gt.get("notes", "")

    paired = [
        row for row in rows
        if row["unrectified_reading"] is not None and row["rectified_reading"] is not None
    ]
    differences = [row["rectified_minus_unrectified"] for row in paired]
    absolute_differences = [abs(value) for value in differences]
    normalized_differences = [
        abs(row["rectified_minus_unrectified"]) / row["scale_span"]
        for row in paired
        if row["scale_span"] is not None and row["scale_span"] > 0
    ]
    largest_disagreements = [
        {
            "sample_id": row["sample_id"],
            "unrectified_reading": rounded(row["unrectified_reading"], 3),
            "rectified_reading": rounded(row["rectified_reading"], 3),
            "absolute_difference": rounded(
                row["abs_rectified_unrectified_difference"], 3
            ),
            "absolute_difference_full_scale": rounded(
                row["abs_rectified_unrectified_difference"] / row["scale_span"]
            ) if row["scale_span"] is not None and row["scale_span"] > 0 else None,
        }
        for row in sorted(
            paired,
            key=lambda item: item["abs_rectified_unrectified_difference"],
            reverse=True,
        )[:5]
    ]
    actual_count = sum(row["actual_reading"] is not None for row in rows)
    rectified_metrics = regression_metrics(rows, "rectified_reading")
    unrectified_metrics = regression_metrics(rows, "unrectified_reading")
    improved = worsened = unchanged = 0
    error_reductions: list[float] = []
    for row in rows:
        actual = row["actual_reading"]
        raw = row["unrectified_reading"]
        rectified = row["rectified_reading"]
        if actual is None or raw is None or rectified is None:
            continue
        raw_error = abs(raw - actual)
        rectified_error = abs(rectified - actual)
        error_reductions.append(raw_error - rectified_error)
        if rectified_error < raw_error - 1e-9:
            improved += 1
        elif rectified_error > raw_error + 1e-9:
            worsened += 1
        else:
            unchanged += 1

    summary = {
        "saved_result_files": len(list(data_dir.glob("*.json"))) if data_dir.exists() else 0,
        "successful_rectified_readings": sum(row["rectified_success"] for row in rows),
        "successful_unrectified_readings": sum(row["unrectified_success"] for row in rows),
        "paired_readings": len(paired),
        "mean_rectified_minus_unrectified": rounded(mean(differences)) if differences else None,
        "mean_absolute_rectified_unrectified_difference": (
            rounded(mean(absolute_differences)) if absolute_differences else None
        ),
        "median_absolute_rectified_unrectified_difference": (
            rounded(median(absolute_differences)) if absolute_differences else None
        ),
        "max_absolute_rectified_unrectified_difference": (
            rounded(max(absolute_differences)) if absolute_differences else None
        ),
        "mean_absolute_difference_full_scale": (
            rounded(mean(normalized_differences)) if normalized_differences else None
        ),
        "paired_within_1pct_full_scale_rate": (
            rounded(mean(value <= 0.01 for value in normalized_differences))
            if normalized_differences else None
        ),
        "paired_within_2pct_full_scale_rate": (
            rounded(mean(value <= 0.02 for value in normalized_differences))
            if normalized_differences else None
        ),
        "paired_within_5pct_full_scale_rate": (
            rounded(mean(value <= 0.05 for value in normalized_differences))
            if normalized_differences else None
        ),
        "largest_rectified_unrectified_disagreements": largest_disagreements,
        "ground_truth_samples": actual_count,
        "unrectified_vs_ground_truth": unrectified_metrics,
        "rectified_vs_ground_truth": rectified_metrics,
        "rectification_improved_samples": improved,
        "rectification_worsened_samples": worsened,
        "rectification_unchanged_samples": unchanged,
        "mean_absolute_error_reduction_after_rectification": (
            rounded(mean(error_reductions)) if error_reductions else None
        ),
        "note": (
            "未透视与透视后的差值只衡量一致性，不代表准确性。"
            "只有填写人工真实读数后，才能计算 MAE、RMSE、误读率和矫正是否改善。"
        ),
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percent(value: float | None) -> str:
    return "不可计算" if value is None else f"{value * 100:.2f}%"


def number(value: Any, digits: int = 3) -> str:
    return "不可计算" if value is None else f"{float(value):.{digits}f}"


def build_markdown(summary: dict[str, Any], gt_path: Path) -> str:
    detection = summary["object_detection"]
    pose = summary["keypoint_detection"]
    reading = summary["reading"]
    lines = [
        "# 磁翻板液位计识别统计报告",
        "",
        "## 1. 目标检测",
        "",
        f"- 已运行图片：{detection['attempted_images']} 张",
        f"- 检出率：{percent(detection['image_detection_rate'])}",
        f"- 图片漏检率：{percent(detection['image_miss_rate'])}",
        f"- 可与 GT 配对：{detection['gt_evaluated_images']} 张",
        f"- Precision@IoU0.5：{percent(detection['precision'])}",
        f"- Recall@IoU0.5：{percent(detection['recall'])}",
        f"- F1@IoU0.5：{percent(detection['f1'])}",
        f"- 疑似误检数 FP：{detection['fp']}；漏检数 FN：{detection['fn']}",
        f"- 匹配框平均 IoU：{number(detection['mean_matched_iou'])}",
        "",
        "## 2. 关键点检测",
        "",
        f"- 已运行图片：{pose['attempted_images']} 张",
        f"- 图片检出率：{percent(pose['image_detection_rate'])}",
        f"- 可透视变换率：{percent(pose['perspective_ready_rate'])}",
        f"- 低关键点置信度率：{percent(pose['low_keypoint_confidence_rate'])}",
        f"- 平均框置信度：{number(pose['mean_box_confidence'])}",
        f"- 每目标最小关键点置信度均值：{number(pose['mean_min_keypoint_confidence'])}",
        "",
        "## 3. ReadROI 读数",
        "",
        f"- 已保存读数 JSON：{reading['saved_result_files']} 个",
        f"- 透视后成功读数：{reading['successful_rectified_readings']} 个",
        f"- 未透视成功读数：{reading['successful_unrectified_readings']} 个",
        f"- 可成对比较：{reading['paired_readings']} 个",
        f"- 透视后减未透视平均差：{number(reading['mean_rectified_minus_unrectified'])}",
        f"- 两者平均绝对差：{number(reading['mean_absolute_rectified_unrectified_difference'])}",
        f"- 两者中位绝对差：{number(reading['median_absolute_rectified_unrectified_difference'])}",
        f"- 两者最大绝对差：{number(reading['max_absolute_rectified_unrectified_difference'])}",
        f"- 平均绝对差/满量程：{percent(reading['mean_absolute_difference_full_scale'])}",
        f"- 差值在满量程 ±1% 内：{percent(reading['paired_within_1pct_full_scale_rate'])}",
        f"- 差值在满量程 ±2% 内：{percent(reading['paired_within_2pct_full_scale_rate'])}",
        f"- 差值在满量程 ±5% 内：{percent(reading['paired_within_5pct_full_scale_rate'])}",
        "",
        "### 差异最大的样本",
        "",
    ]
    for item in reading["largest_rectified_unrectified_disagreements"]:
        lines.append(
            f"- `{item['sample_id']}`：未透视 {number(item['unrectified_reading'])}，"
            f"透视后 {number(item['rectified_reading'])}，绝对差 {number(item['absolute_difference'])}，"
            f"占满量程 {percent(item['absolute_difference_full_scale'])}。"
        )
    lines.extend([
        "",
        "## 4. 与人工真实读数比较",
        "",
        f"- 已填写真实读数：{reading['ground_truth_samples']} 个",
    ])
    for title, key in (
        ("未透视", "unrectified_vs_ground_truth"),
        ("透视后", "rectified_vs_ground_truth"),
    ):
        metrics = reading[key]
        if metrics is None:
            lines.append(f"- {title}：尚无人工真值，MAE/RMSE/误读率不可计算。")
        else:
            lines.extend([
                f"- {title} MAE：{number(metrics['mae'])}",
                f"- {title} RMSE：{number(metrics['rmse'])}",
                f"- {title} 满量程归一化 MAE：{percent(metrics['normalized_mae_full_scale'])}",
                f"- {title} ±5% 满量程命中率：{percent(metrics['within_5pct_full_scale_rate'])}",
                f"- {title} 误读率（误差 >5% 满量程）：{percent(metrics['misread_rate_over_5pct_full_scale'])}",
            ])
    lines.extend([
        "",
        "## 5. 评价原则",
        "",
        "- 检出率用于描述有没有输出目标，Precision/Recall/F1 才用于区分误检和漏检。",
        "- 未透视读数与透视后读数接近，只能说明两条流程一致，不能说明读数正确。",
        "- 读数准确性以人工真实读数为基准，主指标建议使用 MAE、RMSE、满量程归一化 MAE 和 ±5% 满量程命中率。",
        "- 当前失败操作没有统一保存尝试记录，因此“ReadROI 总成功率”的分母还不完整；后续批量评测应固定测试清单并保存成功和失败状态。",
        "",
        f"人工真值表：`{gt_path}`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="统计液位计完整流程指标")
    parser.add_argument("--object-dir", type=Path, default=DEFAULT_OBJECT_DIR)
    parser.add_argument("--pose-dir", type=Path, default=DEFAULT_POSE_DIR)
    parser.add_argument("--reading-dir", type=Path, default=DEFAULT_READING_DIR)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--keypoint-confidence", type=float, default=0.2)
    args = parser.parse_args()

    object_metrics = evaluate_object_detection(args.object_dir.resolve(), args.iou)
    pose_metrics = evaluate_pose_detection(
        args.pose_dir.resolve(), args.keypoint_confidence
    )
    reading_metrics, rows = evaluate_readings(
        args.reading_dir.resolve(), args.ground_truth.resolve()
    )
    summary = {
        "object_detection": {key: value for key, value in object_metrics.items() if key != "rows"},
        "keypoint_detection": {key: value for key, value in pose_metrics.items() if key != "rows"},
        "reading": reading_metrics,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "object_detection_details.csv", object_metrics["rows"])
    write_csv(output_dir / "keypoint_detection_details.csv", pose_metrics["rows"])
    write_csv(output_dir / "reading_details.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = build_markdown(summary, args.ground_truth.resolve())
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"\n汇总 JSON: {output_dir / 'summary.json'}")
    print(f"逐图明细: {output_dir / 'reading_details.csv'}")


if __name__ == "__main__":
    main()
