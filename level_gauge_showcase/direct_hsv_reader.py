"""Adaptive colour-region reader for a rectified magnetic-flap gauge.

The keypoint rectangle defines the full scale height. A dense per-column map is
built by tracing bottom-connected HSV states over the whole rectified width.
Adjacent columns with a stable boundary form the flap region; no fixed centre
or fixed width crop is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DirectFlapAnalysis:
    x1: int
    x2: int
    features: np.ndarray
    feature_names: list[str]
    level_y: int
    bottom_state_score: np.ndarray
    bottom_connected_mask: np.ndarray
    localization_profile: np.ndarray
    localization_mask: np.ndarray
    confidence: float
    metrics: dict


def _hsv_embedding(image: np.ndarray) -> np.ndarray:
    """Map BGR pixels to circular-Hue, saturation and value coordinates."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    angle = hsv[:, :, 0] * (2.0 * np.pi / 180.0)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    return np.stack(
        [
            np.cos(angle) * saturation,
            np.sin(angle) * saturation,
            saturation,
            value,
        ],
        axis=2,
    ).astype(np.float32)


def extract_hsv_row_features(strip: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Describe each row using colour-only HSV features in circular Hue space."""
    embedding = _hsv_embedding(strip)
    features = np.mean(embedding, axis=1).astype(np.float32)
    sigma = max(1.0, strip.shape[0] * 0.005)
    kernel = max(3, int(round(sigma * 6)) | 1)
    features = cv2.GaussianBlur(features, (1, kernel), sigmaX=0, sigmaY=sigma)
    return features, ["H cos", "H sin", "S mean", "V mean"]


def find_bottom_connected_boundary(
    features: np.ndarray,
    margin_ratio: float = 0.05,
    reference_ratio: float = 0.10,
    allowed_gap_ratio: float = 0.025,
    minimum_agreement: float = 0.70,
) -> tuple[int, np.ndarray, np.ndarray, float, dict]:
    """Find one lower colour state connected to the effective bottom."""
    height = len(features)
    margin = max(4, int(round(height * margin_ratio)))
    reference_rows = max(12, int(round(height * reference_ratio)))
    if 2 * (margin + reference_rows) >= height:
        raise ValueError("颜色带高度不足，无法建立顶部和底部颜色参考。")

    usable_bottom = height - margin
    top_reference = np.median(features[margin : margin + reference_rows], axis=0)
    bottom_reference = np.median(
        features[usable_bottom - reference_rows : usable_bottom], axis=0
    )
    state_axis = bottom_reference - top_reference
    state_separation = float(np.linalg.norm(state_axis))
    if state_separation < 0.05:
        raise ValueError("顶部和底部 HSV 状态过于接近，无法确认连续颜色变化。")

    midpoint = (top_reference + bottom_reference) * 0.5
    bottom_state_score = ((features - midpoint) @ state_axis) / max(
        float(state_axis @ state_axis), 1e-6
    )
    raw_bottom_mask = (bottom_state_score > 0).astype(np.uint8).reshape(-1, 1)
    allowed_gap = max(3, int(round(height * allowed_gap_ratio)) | 1)
    connected_mask = cv2.morphologyEx(
        raw_bottom_mask,
        cv2.MORPH_CLOSE,
        np.ones((allowed_gap, 1), dtype=np.uint8),
    ).ravel()

    bottom_seed_fraction = float(
        np.mean(raw_bottom_mask[usable_bottom - reference_rows : usable_bottom])
    )
    if bottom_seed_fraction < minimum_agreement:
        raise ValueError(
            f"底部颜色状态不连续：底部参考区一致率仅 {bottom_seed_fraction:.1%}。"
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

    top_values = bottom_state_score[margin:level_y]
    bottom_values = bottom_state_score[level_y:usable_bottom]
    top_agreement = float(np.mean(top_values < 0))
    bottom_agreement = float(np.mean(bottom_values > 0))
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
            "底部颜色状态在断开后又大段出现，可能存在遮挡或多重分界："
            f"重现 {longest_reappearance} 行，允许 {allowed_gap} 行。"
        )
    if top_agreement < minimum_agreement or bottom_agreement < minimum_agreement:
        raise ValueError(
            "颜色状态没有形成稳定的上、下连续区域："
            f"上部一致率 {top_agreement:.1%}，下部一致率 {bottom_agreement:.1%}。"
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
        "boundary_method": "bottom_seeded_continuous_hsv",
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
        bottom_state_score.astype(np.float32),
        connected_mask.astype(bool),
        float(confidence),
        metrics,
    )


def locate_colour_strip_by_dense_columns(
    image: np.ndarray,
) -> tuple[list[dict], np.ndarray, np.ndarray, dict]:
    """Build a dense per-column boundary map and group stable adjacent columns."""
    height, width = image.shape[:2]
    if height < 40 or width < 7:
        raise ValueError(f"校正后的液位计区域过小：{width}×{height}。")

    embedding = _hsv_embedding(image)
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
        centre_prior = 0.90 + 0.10 * (1.0 - min(abs(centre - 0.5) * 2.0, 1.0))
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
    profile = np.nan_to_num(boundary_map, nan=-1.0)
    return candidates, profile, valid, metrics


def analyze_direct_flap_strip(image: np.ndarray) -> DirectFlapAnalysis:
    """Find the real colour band across the full width, then read its boundary."""
    candidates, profile, high_response, localization_metrics = (
        locate_colour_strip_by_dense_columns(image)
    )
    accepted: list[tuple[float, dict, np.ndarray, list[str], tuple]] = []
    errors: list[str] = []
    for candidate in candidates:
        x1, x2 = candidate["x1"], candidate["x2"]
        features, names = extract_hsv_row_features(image[:, x1:x2])
        try:
            boundary = find_bottom_connected_boundary(features)
        except ValueError as exc:
            errors.append(f"x={x1}:{x2}: {exc}")
            continue
        _, _, _, continuity_confidence, _ = boundary
        quality = max(continuity_confidence, 0.01) * max(
            candidate["localization_confidence"], 0.01
        )
        accepted.append((quality, candidate, features, names, boundary))

    if not accepted:
        detail = "；".join(errors)
        raise ValueError(
            "自适应颜色带候选均未形成可靠的底部连续状态。"
            + (f" {detail}" if detail else "")
        )

    _, selected, features, names, boundary = max(accepted, key=lambda item: item[0])
    level_y, score, connected_mask, confidence, boundary_metrics = boundary
    x1, x2 = selected["x1"], selected["x2"]
    selected_mask = np.zeros(image.shape[1], dtype=bool)
    selected_mask[x1:x2] = True
    metrics = {
        "method": "adaptive_colour_region_then_bottom_continuity",
        "localization": {
            **localization_metrics,
            "selected_x1": x1,
            "selected_x2": x2,
            "selected_width": x2 - x1,
            "selected_width_ratio": round((x2 - x1) / image.shape[1], 6),
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
    return DirectFlapAnalysis(
        x1=x1,
        x2=x2,
        features=features,
        feature_names=names,
        level_y=level_y,
        bottom_state_score=score,
        bottom_connected_mask=connected_mask,
        localization_profile=profile,
        localization_mask=selected_mask,
        confidence=confidence,
        metrics=metrics,
    )
