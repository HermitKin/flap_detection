"""单张图片磁翻板液位计检测、透视校正与 HSV 连续状态读数。

直接运行时先弹出图片选择框，再在命令行输入量程。脚本调用 YOLO Pose
四关键点模型，透视校正后在完整宽度中自适应提取磁翻板颜色区域，
再从底部向上追踪连续颜色状态，并保存所有处理图和 JSON 数据。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent

# 可直接修改下面三个参数。
MODEL_PATH = (
    PROJECT_ROOT
    / "object detection"
    / "runs"
    / "train"
    / "flap_pose4_with_yolo11s-set2"
    / "weights"
    / "best.pt"
)
OUTPUT_ROOT = PROJECT_ROOT / "single_image_level_results"
POSE_HELPER = PROJECT_ROOT / "detect_pose_keypoints_yolo.py"
POSE_IMGSZ = 960
POSE_CONFIDENCE = 0.20

# 连续颜色状态判断参数。
END_MARGIN_RATIO = 0.05
REFERENCE_RATIO = 0.10
ALLOWED_GAP_RATIO = 0.025
MINIMUM_AGREEMENT = 0.70

KEYPOINT_NAMES = ("TL", "TR", "BR", "BL")
KEYPOINT_COLOURS = (
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
)


@dataclass
class FlapAnalysis:
    x1: int
    x2: int
    level_y: int
    bottom_state_score: np.ndarray
    bottom_connected_mask: np.ndarray
    localization_profile: np.ndarray
    localization_mask: np.ndarray
    confidence: float
    metrics: dict


def read_image(path: Path) -> np.ndarray:
    """支持中文路径读取图片。"""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> Path:
    """支持中文路径保存图片。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"无法编码图片：{path}")
    encoded.tofile(path)
    return path.resolve()


def put_text_clamped(
    image: np.ndarray,
    text: str,
    preferred_xy: tuple[int, int],
    font_scale: float,
    colour: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """绘制文字并限制在图像范围内，避免窄图文字被截断。"""
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x = int(np.clip(preferred_xy[0], 0, max(0, image.shape[1] - text_width - 1)))
    y = int(
        np.clip(
            preferred_xy[1],
            text_height + 1,
            max(text_height + 1, image.shape[0] - baseline - 1),
        )
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def choose_image() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="选择一张待检测的磁翻板液位计图片",
        filetypes=[
            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("所有文件", "*.*"),
        ],
    )
    root.destroy()
    return Path(selected) if selected else None


def prompt_scale_range(
    default_min: float = 0.0,
    default_max: float = 100.0,
) -> tuple[float, float]:
    while True:
        try:
            minimum_text = input(
                f"请输入量程下限（直接回车默认 {default_min:g}）: "
            ).strip()
            maximum_text = input(
                f"请输入量程上限（直接回车默认 {default_max:g}）: "
            ).strip()
            minimum = float(minimum_text) if minimum_text else default_min
            maximum = float(maximum_text) if maximum_text else default_max
        except ValueError:
            print("量程必须是数字，请重新输入。")
            continue
        if maximum <= minimum:
            print("量程上限必须大于量程下限，请重新输入。")
            continue
        return minimum, maximum


def detect_pose_in_subprocess(
    image_path: Path,
    model_path: Path,
    confidence: float,
    imgsz: int,
) -> list[dict]:
    """隔离 YOLO/PyTorch，避免 Windows Anaconda 的 OpenMP DLL 冲突。"""
    if not POSE_HELPER.is_file():
        raise ValueError(f"关键点推理助手不存在：{POSE_HELPER}")
    process = subprocess.run(
        [
            sys.executable,
            str(POSE_HELPER),
            str(model_path),
            str(image_path),
            str(confidence),
            str(imgsz),
            "--no-save",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    marker = "POSE_RESULT_JSON="
    result_line = next(
        (line for line in process.stdout.splitlines() if line.startswith(marker)),
        None,
    )
    if process.returncode != 0 or result_line is None:
        detail = process.stderr.strip() or process.stdout.strip() or "未知推理错误"
        raise ValueError(f"关键点模型推理失败：{detail}")
    return json.loads(result_line[len(marker) :])


def hsv_embedding(image: np.ndarray) -> np.ndarray:
    """将像素转换为循环 Hue、饱和度和亮度特征。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_angle = hsv[:, :, 0] * (2.0 * np.pi / 180.0)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    return np.stack(
        [
            np.cos(hue_angle) * saturation,
            np.sin(hue_angle) * saturation,
            saturation,
            value,
        ],
        axis=2,
    ).astype(np.float32)


def extract_hsv_row_features(strip: np.ndarray) -> np.ndarray:
    """每行提取循环 Hue、饱和度和亮度四维颜色特征。"""
    features = np.mean(hsv_embedding(strip), axis=1).astype(np.float32)

    sigma = max(1.0, strip.shape[0] * 0.005)
    kernel = max(3, int(round(sigma * 6)) | 1)
    return cv2.GaussianBlur(features, (1, kernel), sigmaX=0, sigmaY=sigma)


def find_bottom_connected_boundary(
    features: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, float, dict]:
    """从有效底部向上追踪连续颜色状态，返回第一次真实断开的位置。"""
    height = len(features)
    margin = max(4, int(round(height * END_MARGIN_RATIO)))
    reference_rows = max(12, int(round(height * REFERENCE_RATIO)))
    if 2 * (margin + reference_rows) >= height:
        raise ValueError("颜色带高度不足，无法建立顶部和底部颜色参考。")

    usable_bottom = height - margin
    top_reference = np.median(
        features[margin : margin + reference_rows], axis=0
    )
    bottom_reference = np.median(
        features[usable_bottom - reference_rows : usable_bottom], axis=0
    )
    state_axis = bottom_reference - top_reference
    state_separation = float(np.linalg.norm(state_axis))
    if state_separation < 0.05:
        raise ValueError("顶部和底部 HSV 状态过于接近，无法确认颜色变化。")

    midpoint = (top_reference + bottom_reference) * 0.5
    score = ((features - midpoint) @ state_axis) / max(
        float(state_axis @ state_axis), 1e-6
    )
    raw_bottom_mask = (score > 0).astype(np.uint8).reshape(-1, 1)

    allowed_gap = max(3, int(round(height * ALLOWED_GAP_RATIO)) | 1)
    connected_mask = cv2.morphologyEx(
        raw_bottom_mask,
        cv2.MORPH_CLOSE,
        np.ones((allowed_gap, 1), dtype=np.uint8),
    ).ravel()

    seed_slice = raw_bottom_mask[
        usable_bottom - reference_rows : usable_bottom
    ]
    bottom_seed_fraction = float(np.mean(seed_slice))
    if bottom_seed_fraction < MINIMUM_AGREEMENT:
        raise ValueError(
            "底部颜色状态不稳定："
            f"底部参考区一致率 {bottom_seed_fraction:.1%}，"
            f"要求至少 {MINIMUM_AGREEMENT:.0%}。"
        )

    y = usable_bottom - 1
    while y >= margin and connected_mask[y] != 0:
        y -= 1
    level_y = y + 1

    minimum_state_rows = max(12, int(round(height * 0.05)))
    if (
        level_y <= margin + minimum_state_rows
        or level_y >= usable_bottom - minimum_state_rows
    ):
        raise ValueError("连续颜色分界过于靠近量程端点，结果无效。")

    top_values = score[margin:level_y]
    bottom_values = score[level_y:usable_bottom]
    top_agreement = float(np.mean(top_values < 0))
    bottom_agreement = float(np.mean(bottom_values > 0))

    # 大段底部状态在分界上方重新出现时，通常是遮挡或多重伪分界。
    top_raw_mask = raw_bottom_mask.ravel()[margin:level_y]
    longest_reappearance = 0
    current_run = 0
    for is_bottom_like in top_raw_mask:
        if is_bottom_like:
            current_run += 1
            longest_reappearance = max(longest_reappearance, current_run)
        else:
            current_run = 0
    if longest_reappearance > allowed_gap:
        raise ValueError(
            "底部颜色状态断开后又大段出现，可能存在遮挡或多重分界："
            f"重现 {longest_reappearance} 行，允许 {allowed_gap} 行。"
        )

    if (
        top_agreement < MINIMUM_AGREEMENT
        or bottom_agreement < MINIMUM_AGREEMENT
    ):
        raise ValueError(
            "没有形成稳定的上下连续颜色区域："
            f"上部一致率 {top_agreement:.1%}，"
            f"下部一致率 {bottom_agreement:.1%}。"
        )

    top_median = float(np.median(top_values))
    bottom_median = float(np.median(bottom_values))
    top_mad = float(np.median(np.abs(top_values - top_median)))
    bottom_mad = float(np.median(np.abs(bottom_values - bottom_median)))
    pooled_mad = max(1.4826 * (top_mad + bottom_mad) * 0.5, 0.05)
    confidence = (
        (bottom_median - top_median)
        / pooled_mad
        * min(top_agreement, bottom_agreement)
    )
    metrics = {
        "method": "bottom_seeded_continuous_hsv",
        "margin_rows": margin,
        "reference_rows": reference_rows,
        "allowed_gap_rows": allowed_gap,
        "state_separation": round(state_separation, 6),
        "bottom_seed_fraction": round(bottom_seed_fraction, 6),
        "top_state_agreement": round(top_agreement, 6),
        "bottom_state_agreement": round(bottom_agreement, 6),
        "longest_bottom_reappearance_rows": longest_reappearance,
    }
    return (
        level_y,
        score.astype(np.float32),
        connected_mask.astype(bool),
        float(confidence),
        metrics,
    )


def locate_colour_strip_by_dense_columns(
    image: np.ndarray,
) -> tuple[list[dict], np.ndarray, np.ndarray, dict]:
    """逐列追踪底部颜色状态，再组合分界高度稳定的连续列。"""
    height, width = image.shape[:2]
    if height < 40 or width < 7:
        raise ValueError(f"透视校正区域过小：{width}×{height}")
    embedding = hsv_embedding(image)
    local_width = max(3, int(round(width * 0.10)) | 1)
    local_width = min(local_width, width if width % 2 else width - 1)
    half = local_width // 2
    boundary_map = np.full(width, np.nan, dtype=np.float32)
    confidence_map = np.zeros(width, dtype=np.float32)
    separation_map = np.zeros(width, dtype=np.float32)

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
            level_y, _, _, confidence, boundary_metrics = (
                find_bottom_connected_boundary(features)
            )
        except ValueError:
            continue
        boundary_map[x] = level_y
        confidence_map[x] = max(0.0, confidence)
        separation_map[x] = float(boundary_metrics["state_separation"])

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

    candidates: list[dict] = []
    for group in groups:
        x1, x2 = group[0], group[-1] + 1
        span = x2 - x1
        if span < minimum_width or span > maximum_width:
            continue
        boundaries = boundary_map[group]
        boundary_std = float(np.std(boundaries))
        stability = max(0.0, 1.0 - boundary_std / max(tolerance_rows, 1))
        median_confidence = float(np.median(confidence_map[group]))
        centre = (x1 + x2) / (2.0 * width)
        centre_prior = 0.90 + 0.10 * (
            1.0 - min(abs(centre - 0.5) * 2.0, 1.0)
        )
        localization_confidence = (
            span / max(width, 1)
            * np.log1p(median_confidence)
            * max(stability, 0.10)
            * centre_prior
            * 10.0
        )
        inside_separation = float(np.median(separation_map[group]))
        outside_valid = valid.copy()
        outside_valid[x1:x2] = False
        outside_values = separation_map[outside_valid]
        outside_separation = (
            float(np.median(outside_values)) if len(outside_values) else 0.0
        )
        candidates.append(
            {
                "x1": x1,
                "x2": x2,
                "inside_separation": inside_separation,
                "outside_separation": outside_separation,
                "contrast": inside_separation - outside_separation,
                "localization_confidence": float(localization_confidence),
                "boundary_median_y": float(np.median(boundaries)),
                "boundary_std_rows": boundary_std,
                "valid_column_count": len(group),
            }
        )
    if not candidates:
        raise ValueError(
            "全宽逐列分析没有形成横向连续且分界高度稳定的翻片颜色区域。"
        )
    metrics = {
        "method": "dense_column_bottom_connected_colour_region",
        "search_width": width,
        "local_window_width": local_width,
        "boundary_tolerance_rows": tolerance_rows,
        "maximum_invalid_gap_columns": maximum_gap,
        "valid_column_count": int(np.count_nonzero(valid)),
        "candidate_count": len(candidates),
        "fixed_center_used": False,
        "fixed_width_used": False,
    }
    return candidates, np.nan_to_num(boundary_map, nan=-1.0), valid, metrics


def analyze_flap_strip(rectified: np.ndarray) -> FlapAnalysis:
    candidates, profile, _, localization_metrics = locate_colour_strip_by_dense_columns(
        rectified
    )
    accepted = []
    errors = []
    for candidate in candidates:
        x1, x2 = candidate["x1"], candidate["x2"]
        features = extract_hsv_row_features(rectified[:, x1:x2])
        try:
            boundary = find_bottom_connected_boundary(features)
        except ValueError as exc:
            errors.append(f"x={x1}:{x2}: {exc}")
            continue
        quality = max(boundary[3], 0.01) * max(
            candidate["localization_confidence"], 0.01
        )
        accepted.append((quality, candidate, features, boundary))
    if not accepted:
        detail = "；".join(errors)
        raise ValueError(
            "自适应颜色带候选均未形成可靠的底部连续状态。"
            + (f" {detail}" if detail else "")
        )

    _, selected, features, boundary = max(accepted, key=lambda item: item[0])
    x1, x2 = selected["x1"], selected["x2"]
    level_y, score, connected_mask, confidence, boundary_metrics = boundary
    localization_mask = np.zeros(rectified.shape[1], dtype=bool)
    localization_mask[x1:x2] = True
    metrics = {
        "method": "adaptive_full_width_colour_region_then_bottom_continuity",
        "localization": {
            **localization_metrics,
            "selected_x1": x1,
            "selected_x2": x2,
            "selected_width": x2 - x1,
            "selected_width_ratio": round((x2 - x1) / rectified.shape[1], 6),
            "inside_state_separation": round(selected["inside_separation"], 6),
            "outside_state_separation": round(selected["outside_separation"], 6),
            "boundary_median_y": round(selected["boundary_median_y"], 3),
            "boundary_std_rows": round(selected["boundary_std_rows"], 3),
            "valid_column_count": selected["valid_column_count"],
            "localization_confidence": round(
                selected["localization_confidence"], 6
            ),
        },
        **boundary_metrics,
    }
    return FlapAnalysis(
        x1=x1,
        x2=x2,
        level_y=level_y,
        bottom_state_score=score,
        bottom_connected_mask=connected_mask,
        localization_profile=profile,
        localization_mask=localization_mask,
        confidence=confidence,
        metrics=metrics,
    )


def validate_and_clip_quad(
    raw_quad: np.ndarray,
    image_shape: tuple[int, ...],
) -> np.ndarray:
    quad = np.asarray(raw_quad, dtype=np.float32).copy()
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError("关键点不是有效的 4×2 坐标。")
    quad[:, 0] = np.clip(quad[:, 0], 0, image_shape[1] - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, image_shape[0] - 1)
    polygon = np.round(quad).astype(np.int32)
    if not cv2.isContourConvex(polygon):
        raise ValueError("四关键点 TL、TR、BR、BL 没有构成凸四边形。")
    if abs(float(cv2.contourArea(quad))) < 100.0:
        raise ValueError("四关键点围成的区域过小。")
    return quad


def rectify_quad(
    image: np.ndarray,
    raw_quad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    quad = validate_and_clip_quad(raw_quad, image.shape)
    tl, tr, br, bl = quad
    target_width = int(
        round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    )
    target_height = int(
        round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    )
    if target_width < 7 or target_height < 40:
        raise ValueError(f"关键点区域过小：{target_width}×{target_height}")
    destination = np.asarray(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, destination)
    rectified = cv2.warpPerspective(
        image,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rectified, matrix


def draw_pose_detection(
    image: np.ndarray,
    box: np.ndarray,
    quad: np.ndarray,
    keypoint_confidences: np.ndarray,
    box_confidence: float,
    gauge_id: int,
) -> None:
    narrow = image.shape[1] < 300
    font_scale = 0.34 if narrow else 0.4
    x1, y1, x2, y2 = np.round(box).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
    polygon = np.round(quad).astype(np.int32)
    cv2.polylines(image, [polygon], True, (255, 0, 255), 2, cv2.LINE_AA)
    for index, (point, point_confidence) in enumerate(
        zip(polygon, keypoint_confidences)
    ):
        centre = (int(point[0]), int(point[1]))
        colour = KEYPOINT_COLOURS[index]
        cv2.circle(image, centre, 5, colour, -1, cv2.LINE_AA)
        point_label = KEYPOINT_NAMES[index]
        if not narrow:
            point_label += f" {point_confidence:.2f}"
        put_text_clamped(
            image,
            point_label,
            (centre[0] + 4, max(12, centre[1] - 5)),
            font_scale=font_scale,
            colour=colour,
            thickness=1,
        )
    box_label = (
        f"#{gauge_id} {box_confidence:.2f}"
        if narrow
        else f"#{gauge_id} pose {box_confidence:.3f}"
    )
    put_text_clamped(
        image,
        box_label,
        (max(0, x1), max(16, y1 - 6)),
        font_scale=0.4 if narrow else 0.5,
        colour=(255, 255, 0),
        thickness=1,
    )


def create_hsv_visualization(
    strip: np.ndarray,
    connected_mask: np.ndarray,
    gauge_id: int,
    source_level_y: int,
    full_scale_height: int,
) -> np.ndarray:
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    full = np.full_like(hue, 255)
    hue_colour = cv2.cvtColor(
        cv2.merge([hue, full, full]), cv2.COLOR_HSV2BGR
    )
    saturation_gray = cv2.cvtColor(saturation, cv2.COLOR_GRAY2BGR)
    value_gray = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
    bottom_state = np.zeros_like(value_gray)
    bottom_rows = np.asarray(connected_mask).reshape(-1).astype(bool)
    bottom_state[bottom_rows, :, 1] = 220
    bottom_state[~bottom_rows, :, 0] = 150

    height = strip.shape[0]
    panel_width = max(100, min(180, strip.shape[1] * 6))
    panels = []
    for title, panel in (
        ("HUE", hue_colour),
        ("SATURATION", saturation_gray),
        ("VALUE", value_gray),
        ("BOTTOM STATE", bottom_state),
    ):
        resized = cv2.resize(
            panel, (panel_width, height), interpolation=cv2.INTER_NEAREST
        )
        cv2.rectangle(resized, (0, 0), (panel_width - 1, height - 1), (0, 255, 255), 1)
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
        boundary_display_y = min(24, height - 1)
        cv2.line(
            resized, (0, boundary_display_y),
            (panel_width - 1, boundary_display_y), (0, 0, 255), 2
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


def create_colour_region_visualization(
    rectified: np.ndarray,
    analysis: FlapAnalysis,
) -> np.ndarray:
    """高亮自适应检测出的颜色带，非选中区域转为暗灰色。"""
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    background = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    background = (background.astype(np.float32) * 0.42).astype(np.uint8)
    result = background
    result[analysis.level_y :, analysis.x1 : analysis.x2] = rectified[
        analysis.level_y :, analysis.x1 : analysis.x2
    ]
    cv2.rectangle(
        result,
        (analysis.x1, analysis.level_y),
        (analysis.x2 - 1, rectified.shape[0] - 1),
        (0, 255, 0),
        1,
    )
    return result


def project_boundary_to_source(
    perspective_matrix: np.ndarray,
    rectified_width: int,
    level_y: int,
) -> np.ndarray:
    inverse = np.linalg.inv(perspective_matrix)
    line = np.asarray(
        [[[0.0, float(level_y)], [float(rectified_width - 1), float(level_y)]]],
        dtype=np.float32,
    )
    return cv2.perspectiveTransform(line, inverse)[0]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="单张图片磁翻板关键点检测、透视校正和 HSV 液位读数"
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="不填写时弹出图片选择框",
    )
    parser.add_argument("--model", default=str(MODEL_PATH), help="YOLO Pose 权重")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT), help="输出根目录")
    parser.add_argument("--confidence", type=float, default=POSE_CONFIDENCE)
    parser.add_argument("--imgsz", type=int, default=POSE_IMGSZ)
    parser.add_argument("--scale-min", type=float)
    parser.add_argument("--scale-max", type=float)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    image_path = Path(args.image) if args.image else choose_image()
    if image_path is None:
        print("未选择图片，程序结束。")
        return
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise SystemExit(f"图片不存在：{image_path}")

    if args.scale_min is None or args.scale_max is None:
        scale_min, scale_max = prompt_scale_range(
            args.scale_min if args.scale_min is not None else 0.0,
            args.scale_max if args.scale_max is not None else 100.0,
        )
    else:
        scale_min, scale_max = args.scale_min, args.scale_max
        if scale_max <= scale_min:
            raise SystemExit("量程上限必须大于量程下限。")

    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise SystemExit(f"关键点模型不存在：{model_path}")

    image = read_image(image_path)
    output_dir = Path(args.output_root).resolve() / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在调用关键点模型：{model_path}")
    detections = detect_pose_in_subprocess(
        image_path, model_path, args.confidence, args.imgsz
    )
    if not detections:
        raise SystemExit("没有检测到液位计或四关键点。")

    pose_view = image.copy()
    final_view = image.copy()
    reports: list[dict] = []
    errors: list[str] = []

    for index, detection in enumerate(detections):
        gauge_id = index + 1
        raw_keypoints = np.asarray(
            detection.get("roi_quad_tl_tr_br_bl")
            or detection.get("keypoints_xy_pixels")
            or detection.get("keypoints")
            or [],
            dtype=np.float32,
        )
        if raw_keypoints.shape != (4, 2):
            errors.append(f"液位计 {gauge_id}: 关键点数量不是4个。")
            continue
        box = np.asarray(
            detection.get("box_xyxy") or detection.get("box"), dtype=np.float32
        )
        box_confidence = float(
            detection.get("box_confidence", detection.get("confidence", 0.0))
        )
        point_confidences = np.asarray(
            detection.get("keypoint_confidences", [1.0] * 4), dtype=np.float32
        )
        draw_pose_detection(
            pose_view,
            box,
            raw_keypoints,
            point_confidences,
            box_confidence,
            gauge_id,
        )
        try:
            quad = validate_and_clip_quad(raw_keypoints, image.shape)
            rectified, matrix = rectify_quad(image, quad)
            analysis = analyze_flap_strip(rectified)
        except ValueError as exc:
            errors.append(f"液位计 {gauge_id}: {exc}")
            continue

        ratio = (rectified.shape[0] - analysis.level_y) / max(
            rectified.shape[0], 1
        )
        reading = float(
            np.clip(
                scale_min + ratio * (scale_max - scale_min),
                scale_min,
                scale_max,
            )
        )

        draw_pose_detection(
            final_view,
            box,
            quad,
            point_confidences,
            box_confidence,
            gauge_id,
        )

        boundary_source = project_boundary_to_source(
            matrix, rectified.shape[1], analysis.level_y
        )
        p1, p2 = np.round(boundary_source).astype(int)
        cv2.line(final_view, tuple(p1), tuple(p2), (0, 255, 255), 3, cv2.LINE_AA)
        label_xy = (int(p1[0]), max(18, int(p1[1]) - 8))
        reading_label = (
            f"{reading:.2f}"
            if image.shape[1] < 300
            else f"#{gauge_id} level={reading:.2f}"
        )
        put_text_clamped(
            final_view,
            reading_label,
            label_xy,
            font_scale=0.5 if image.shape[1] < 300 else 0.55,
            colour=(0, 255, 255),
            thickness=2,
        )

        marked_rectified = rectified.copy()
        cv2.rectangle(
            marked_rectified,
            (analysis.x1, analysis.level_y),
            (analysis.x2 - 1, rectified.shape[0] - 1),
            (255, 255, 0),
            1,
        )
        cv2.line(
            marked_rectified,
            (0, analysis.level_y),
            (rectified.shape[1] - 1, analysis.level_y),
            (0, 255, 255),
            2,
        )
        cv2.putText(
            marked_rectified,
            f"level={reading:.2f}",
            (4, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        gauge_dir = output_dir / f"gauge_{gauge_id:02d}"
        rectified_path = write_image(
            gauge_dir / "02_rectified.png", rectified
        )
        colour_region_path = write_image(
            gauge_dir / "02_colour_region.png",
            create_colour_region_visualization(rectified, analysis),
        )
        strip_path = write_image(
            gauge_dir / "03_flap_strip.png",
            rectified[analysis.level_y :, analysis.x1 : analysis.x2],
        )
        hsv_path = write_image(
            gauge_dir / "04_hsv_continuity.png",
            create_hsv_visualization(
                rectified[analysis.level_y :, analysis.x1 : analysis.x2],
                analysis.bottom_connected_mask[analysis.level_y :],
                gauge_id,
                analysis.level_y,
                rectified.shape[0],
            ),
        )
        reading_path = write_image(
            gauge_dir / "05_rectified_reading.png", marked_rectified
        )

        reports.append(
            {
                "gauge_id": gauge_id,
                "box_xyxy": box.astype(float).tolist(),
                "box_confidence": box_confidence,
                "keypoint_order": list(KEYPOINT_NAMES),
                "keypoints_xy_pixels": quad.astype(float).tolist(),
                "keypoint_confidences": point_confidences.astype(float).tolist(),
                "perspective_matrix": matrix.astype(float).tolist(),
                "rectified_size": [rectified.shape[1], rectified.shape[0]],
                "flap_strip_xyxy_rectified": [
                    analysis.x1,
                    analysis.level_y,
                    analysis.x2,
                    rectified.shape[0],
                ],
                "coloured_interval_y_rectified": [
                    analysis.level_y,
                    rectified.shape[0],
                ],
                "coloured_height": rectified.shape[0] - analysis.level_y,
                "full_scale_height": rectified.shape[0],
                "boundary_y_rectified": analysis.level_y,
                "boundary_line_source_xy": boundary_source.astype(float).tolist(),
                "level_ratio": round(float(ratio), 6),
                "reading": round(reading, 3),
                "continuity_confidence": round(analysis.confidence, 3),
                "continuity_metrics": analysis.metrics,
                "strip_localization_metrics": analysis.metrics["localization"],
                "rectified_image": str(rectified_path),
                "colour_region_image": str(colour_region_path),
                "flap_strip_image": str(strip_path),
                "hsv_continuity_image": str(hsv_path),
                "rectified_reading_image": str(reading_path),
            }
        )

    pose_path = write_image(output_dir / "01_pose_detection.png", pose_view)
    if not reports:
        detail = "\n".join(errors) if errors else "没有有效的检测结果。"
        raise SystemExit(f"没有得到可靠读数：\n{detail}\n关键点图：{pose_path}")

    result_path = write_image(output_dir / "detection_result.png", final_view)
    json_path = output_dir / "readings.json"
    json_path.write_text(
        json.dumps(
            {
                "source_image": str(image_path),
                "source_size": [image.shape[1], image.shape[0]],
                "result_size": [final_view.shape[1], final_view.shape[0]],
                "model": str(model_path),
                "pose_imgsz": args.imgsz,
                "pose_confidence_threshold": args.confidence,
                "scale_range": [scale_min, scale_max],
                "strip_localization_mode": "dense_column_bottom_connected_colour_region",
                "fixed_center_or_width_used": False,
                "gauges": reports,
                "skipped_errors": errors,
                "pose_detection_image": str(pose_path),
                "result_image": str(result_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n========== 单张图片液位检测结果 ==========")
    print(f"选择的图片: {image_path}")
    print(f"设置的量程: {scale_min:g} ～ {scale_max:g}")
    print(f"检测到并成功读取: {len(reports)} 个液位计")
    for report in reports:
        metrics = report["continuity_metrics"]
        print(f"\n液位计 {report['gauge_id']}")
        print(f"  检测框: {report['box_xyxy']}")
        print(f"  YOLO置信度: {report['box_confidence']:.4f}")
        print(f"  四关键点: {report['keypoints_xy_pixels']}")
        print(f"  校正尺寸: {report['rectified_size'][0]}×{report['rectified_size'][1]}")
        print(f"  自适应翻片颜色带: {report['flap_strip_xyxy_rectified']}")
        localization = report["strip_localization_metrics"]
        print(f"  颜色带定位置信度: {localization['localization_confidence']:.3f}")
        print(f"  分界位置 y: {report['boundary_y_rectified']}")
        print(f"  底部种子一致率: {metrics['bottom_seed_fraction']:.1%}")
        print(f"  上部状态一致率: {metrics['top_state_agreement']:.1%}")
        print(f"  下部状态一致率: {metrics['bottom_state_agreement']:.1%}")
        print(f"  连续性置信度: {report['continuity_confidence']:.3f}")
        print(f"  当前读数: {report['reading']:.3f}")
    if errors:
        print("\n跳过的检测：")
        for error in errors:
            print(f"  - {error}")
    print(f"\n原图尺寸结果图: {result_path}")
    print(f"关键点检测图: {pose_path}")
    print(f"JSON报告: {json_path.resolve()}")
    print(f"全部分步结果目录: {output_dir.resolve()}")
    print("==========================================")


if __name__ == "__main__":
    main()
