"""Export step-by-step diagnostics for one saved ReadROI JSON result."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from direct_hsv_reader import (
    _hsv_embedding,
    analyze_direct_flap_strip,
    find_bottom_connected_boundary,
)
from services import ReadingService, read_image, write_image, write_json


APP_ROOT = Path(__file__).resolve().parent


def collect_column_trace(image: np.ndarray) -> dict:
    height, width = image.shape[:2]
    embedding = _hsv_embedding(image)
    local_width = max(3, int(round(width * 0.10)) | 1)
    local_width = min(local_width, width if width % 2 else width - 1)
    half = local_width // 2
    boundary_map = np.full(width, np.nan, dtype=np.float32)
    confidence_map = np.zeros(width, dtype=np.float32)
    separation_map = np.zeros(width, dtype=np.float32)
    failures: Counter[str] = Counter()

    for x in range(width):
        left = max(0, x - half)
        right = min(width, x + half + 1)
        features = np.mean(embedding[:, left:right], axis=1).astype(np.float32)
        sigma = max(1.0, height * 0.005)
        kernel = max(3, int(round(sigma * 6)) | 1)
        features = cv2.GaussianBlur(
            features, (1, kernel), sigmaX=0, sigmaY=sigma
        )
        try:
            level_y, _, _, confidence, metrics = find_bottom_connected_boundary(
                features
            )
        except ValueError as exc:
            message = str(exc)
            if "过于接近" in message:
                reason = "top_bottom_too_similar"
            elif "底部参考区" in message:
                reason = "bottom_seed_inconsistent"
            elif "端点" in message:
                reason = "boundary_near_endpoint"
            elif "重现" in message:
                reason = "bottom_state_reappears_after_gap"
            elif "一致率" in message:
                reason = "top_bottom_agreement_too_low"
            else:
                reason = "other"
            failures[reason] += 1
            continue
        boundary_map[x] = level_y
        confidence_map[x] = max(0.0, confidence)
        separation_map[x] = float(metrics["state_separation"])

    valid = np.isfinite(boundary_map) & (confidence_map >= 2.5)
    tolerance_rows = max(4, int(round(height * 0.025)))
    maximum_gap = max(1, int(round(width * 0.03)))
    minimum_width = max(3, int(round(width * 0.08)))
    maximum_width = max(minimum_width, int(round(width * 0.65)))
    groups: list[list[int]] = []
    current: list[int] = []
    for x in range(width):
        if not valid[x]:
            continue
        if not current:
            current = [x]
            continue
        gap = x - current[-1] - 1
        reference_y = float(np.median(boundary_map[current]))
        if gap <= maximum_gap and abs(float(boundary_map[x]) - reference_y) <= tolerance_rows:
            current.append(x)
        else:
            groups.append(current)
            current = [x]
    if current:
        groups.append(current)

    group_records = []
    for group in groups:
        span = group[-1] - group[0] + 1
        group_records.append({
            "x1": group[0],
            "x2": group[-1] + 1,
            "span": span,
            "valid_columns": len(group),
            "boundary_median_y": float(np.median(boundary_map[group])),
            "boundary_std_rows": float(np.std(boundary_map[group])),
            "median_confidence": float(np.median(confidence_map[group])),
            "passes_width_filter": minimum_width <= span <= maximum_width,
        })
    return {
        "image_size": [width, height],
        "local_window_width": local_width,
        "confidence_threshold": 2.5,
        "boundary_tolerance_rows": tolerance_rows,
        "maximum_invalid_gap_columns": maximum_gap,
        "minimum_candidate_width": minimum_width,
        "maximum_candidate_width": maximum_width,
        "successful_columns_before_confidence": int(np.count_nonzero(np.isfinite(boundary_map))),
        "valid_columns": int(np.count_nonzero(valid)),
        "failure_counts": dict(failures),
        "groups": group_records,
        "boundary_map": boundary_map,
        "confidence_map": confidence_map,
        "separation_map": separation_map,
        "valid_mask": valid,
    }


def draw_quad(image: np.ndarray, quad: list[list[float]]) -> np.ndarray:
    result = image.copy()
    points = np.asarray(quad, dtype=np.float32)
    points[:, 0] = np.clip(points[:, 0], 0, image.shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, image.shape[0] - 1)
    points_i = np.round(points).astype(np.int32)
    cv2.polylines(result, [points_i], True, (255, 0, 255), 2, cv2.LINE_AA)
    for name, point in zip(("TL", "TR", "BR", "BL"), points_i):
        cv2.circle(result, tuple(point), 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            result, name, tuple(point), cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (0, 255, 255), 1, cv2.LINE_AA,
        )
    return result


def draw_column_trace(
    image: np.ndarray,
    trace: dict,
    analysis,
) -> np.ndarray:
    scale_x = max(4, int(np.ceil(320 / image.shape[1])))
    result = cv2.resize(
        image,
        (image.shape[1] * scale_x, image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    boundary_map = trace["boundary_map"]
    confidence_map = trace["confidence_map"]
    valid = trace["valid_mask"]
    for x in range(image.shape[1]):
        xx1, xx2 = x * scale_x, (x + 1) * scale_x - 1
        if np.isfinite(boundary_map[x]):
            colour = (0, 220, 0) if valid[x] else (0, 140, 255)
            y = int(round(float(boundary_map[x])))
            cv2.circle(result, ((xx1 + xx2) // 2, y), 2, colour, -1)
        if valid[x]:
            cv2.line(result, (xx1, 0), (xx1, 12), (0, 220, 0), 1)
    x1 = analysis.x1 * scale_x
    x2 = analysis.x2 * scale_x - 1
    cv2.rectangle(
        result, (x1, analysis.level_y),
        (x2, result.shape[0] - 1), (255, 255, 0), 2,
    )
    cv2.line(
        result, (0, analysis.level_y),
        (result.shape[1] - 1, analysis.level_y), (0, 165, 255), 2,
    )
    label = (
        f"selected x={analysis.x1}:{analysis.x2}, y={analysis.level_y}; "
        f"valid={trace['valid_columns']}/{image.shape[1]}"
    )
    cv2.rectangle(result, (0, 0), (result.shape[1] - 1, 30), (0, 0, 0), -1)
    cv2.putText(
        result, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (0, 255, 255), 1, cv2.LINE_AA,
    )
    return result


def draw_score_trace(image: np.ndarray, analysis) -> np.ndarray:
    strip = image[:, analysis.x1:analysis.x2]
    strip_width = max(90, strip.shape[1] * 8)
    strip_view = cv2.resize(
        strip, (strip_width, image.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    plot_width = 260
    plot = np.zeros((image.shape[0], plot_width, 3), dtype=np.uint8)
    score = np.asarray(analysis.bottom_state_score, dtype=np.float32)
    limit = max(float(np.percentile(np.abs(score), 98)), 0.25)
    centre = plot_width // 2
    cv2.line(plot, (centre, 0), (centre, plot.shape[0] - 1), (90, 90, 90), 1)
    points = []
    for y, value in enumerate(score):
        x = int(round(centre + np.clip(value / limit, -1.0, 1.0) * (centre - 8)))
        points.append((x, y))
    cv2.polylines(plot, [np.asarray(points, dtype=np.int32)], False, (0, 255, 255), 1)
    cv2.line(
        strip_view, (0, analysis.level_y),
        (strip_view.shape[1] - 1, analysis.level_y), (0, 165, 255), 2,
    )
    cv2.line(
        plot, (0, analysis.level_y),
        (plot.shape[1] - 1, analysis.level_y), (0, 165, 255), 2,
    )
    combined = cv2.hconcat([strip_view, plot])
    header = np.zeros((46, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header, "selected strip | row score: left=upper, right=bottom",
        (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
        (0, 255, 255), 1, cv2.LINE_AA,
    )
    return cv2.vconcat([header, combined])


def serializable_trace(trace: dict) -> dict:
    result = {}
    for key, value in trace.items():
        if isinstance(value, np.ndarray):
            result[key] = value.astype(float).tolist()
        else:
            result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path, help="saved outputs/read_roi/data JSON")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.json_path.read_text(encoding="utf-8"))
    source_path = Path(payload["source_image"])
    report = payload["gauges"][0]
    quad = report.get("source_quad_tl_tr_br_bl")
    if not quad:
        raise SystemExit("The report has no perspective quad.")
    image = read_image(source_path)
    rectified, _ = ReadingService._rectify_quad(image, quad)
    raw_roi, raw_box = ReadingService._crop_quad_bounding_box(image, quad)
    raw_analysis = analyze_direct_flap_strip(raw_roi)
    rectified_analysis = analyze_direct_flap_strip(rectified)
    raw_trace = collect_column_trace(raw_roi)
    rectified_trace = collect_column_trace(rectified)

    output = args.output or (
        APP_ROOT / "outputs" / "read_roi_debug" / source_path.stem
    )
    output.mkdir(parents=True, exist_ok=True)
    write_image(output / "01_input_quad.png", draw_quad(image, quad))
    write_image(output / "02_unrectified_bbox.png", raw_roi)
    write_image(
        output / "03_unrectified_columns.png",
        draw_column_trace(raw_roi, raw_trace, raw_analysis),
    )
    write_image(
        output / "04_unrectified_selected_score.png",
        draw_score_trace(raw_roi, raw_analysis),
    )
    write_image(output / "05_rectified.png", rectified)
    write_image(
        output / "06_rectified_columns.png",
        draw_column_trace(rectified, rectified_trace, rectified_analysis),
    )
    write_image(
        output / "07_rectified_selected_score.png",
        draw_score_trace(rectified, rectified_analysis),
    )
    trace_payload = {
        "source_image": str(source_path),
        "source_size": [image.shape[1], image.shape[0]],
        "quad_tl_tr_br_bl": quad,
        "unrectified_box_xyxy": raw_box,
        "unrectified": {
            **serializable_trace(raw_trace),
            "selected_x1": raw_analysis.x1,
            "selected_x2": raw_analysis.x2,
            "selected_boundary_y": raw_analysis.level_y,
            "selected_confidence": raw_analysis.confidence,
        },
        "rectified": {
            **serializable_trace(rectified_trace),
            "selected_x1": rectified_analysis.x1,
            "selected_x2": rectified_analysis.x2,
            "selected_boundary_y": rectified_analysis.level_y,
            "selected_confidence": rectified_analysis.confidence,
        },
    }
    write_json(output / "debug_trace.json", trace_payload)
    print(f"DEBUG_OUTPUT={output.resolve()}")
    print(
        "RAW:", raw_analysis.x1, raw_analysis.x2, raw_analysis.level_y,
        f"valid={raw_trace['valid_columns']}/{raw_roi.shape[1]}",
    )
    print(
        "RECTIFIED:", rectified_analysis.x1, rectified_analysis.x2,
        rectified_analysis.level_y,
        f"valid={rectified_trace['valid_columns']}/{rectified.shape[1]}",
    )


if __name__ == "__main__":
    main()
