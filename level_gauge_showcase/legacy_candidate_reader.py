"""加连续性约束之前的 76 候选条带 HSV 读数算法（独立对照版）。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DirectFlapAnalysis:
    """与 GUI 当前读数函数兼容的统一分析结果。"""

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


WIDTH_RATIOS = (0.20, 0.24, 0.28, 0.32)
CENTER_RATIOS = np.linspace(0.28, 0.72, 19)


def _smooth_rows(values: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    kernel = max(3, int(round(sigma * 6)) | 1)
    return cv2.GaussianBlur(
        values.astype(np.float32), (1, kernel), sigmaX=0, sigmaY=sigma
    )


def extract_legacy_row_features(
    image: np.ndarray, x1: int, x2: int
) -> tuple[np.ndarray, list[str]]:
    strip = image[:, x1:x2]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
    horizontal_gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    hue_angle = hsv[:, :, 0] * (2.0 * np.pi / 180.0)
    saturation_weight = hsv[:, :, 1] / 255.0
    features = np.column_stack([
        (np.cos(hue_angle) * saturation_weight).mean(axis=1),
        (np.sin(hue_angle) * saturation_weight).mean(axis=1),
        hsv[:, :, 1].mean(axis=1),
        hsv[:, :, 1].std(axis=1),
        hsv[:, :, 2].mean(axis=1),
        hsv[:, :, 2].std(axis=1),
        gray.std(axis=1),
        horizontal_gradient.mean(axis=1),
    ])
    names = [
        "H cos", "H sin", "S mean", "S std",
        "V mean", "V std", "gray std", "edge energy",
    ]
    features = _smooth_rows(features)
    median = np.median(features, axis=0)
    mad = np.median(np.abs(features - median), axis=0)
    normalized = (features - median) / np.maximum(1.4826 * mad, 1.0)
    return np.clip(normalized, -8.0, 8.0).astype(np.float32), names


def find_persistent_change_peak(
    features: np.ndarray, margin_ratio: float = 0.05
) -> tuple[int, np.ndarray]:
    height = len(features)
    window = max(12, round(height * 0.08))
    cumulative = np.vstack([
        np.zeros((1, features.shape[1]), dtype=np.float32),
        np.cumsum(features, axis=0),
    ])
    score = np.zeros(height, dtype=np.float32)
    start = max(window, round(height * margin_ratio))
    end = min(height - window, round(height * (1.0 - margin_ratio)))
    for y in range(start, end):
        above = (cumulative[y] - cumulative[y - window]) / window
        below = (cumulative[y + window] - cumulative[y]) / window
        score[y] = np.linalg.norm(above - below)
    score = cv2.GaussianBlur(
        score.reshape(-1, 1), (1, 9), sigmaX=0, sigmaY=2
    ).ravel()
    return int(np.argmax(score)), score


def _peak_z_score(score: np.ndarray, level_y: int) -> float:
    valid = score[score > 0]
    if len(valid) == 0:
        return 0.0
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    return (float(score[level_y]) - median) / max(1.4826 * mad, 0.1)


def analyze_legacy_candidate_strips(image: np.ndarray) -> DirectFlapAnalysis:
    """Evaluate the historical 4 widths x 19 centres candidate grid."""
    height, width = image.shape[:2]
    if height < 40 or width < 7:
        raise ValueError(f"候选条带法输入过小：{width}×{height}。")
    candidates = []
    for width_ratio in WIDTH_RATIOS:
        strip_width = max(7, round(width * width_ratio))
        for center_ratio in CENTER_RATIOS:
            center_x = round(width * float(center_ratio))
            x1 = max(0, center_x - strip_width // 2)
            x2 = min(width, x1 + strip_width)
            x1 = max(0, x2 - strip_width)
            if x2 - x1 < 7:
                continue
            features, names = extract_legacy_row_features(image, x1, x2)
            level_y, score = find_persistent_change_peak(features)
            peak_z = _peak_z_score(score, level_y)
            actual_center = (x1 + x2) / (2.0 * width)
            center_prior = max(0.65, 1.0 - abs(actual_center - 0.5))
            width_prior = max(0.75, 1.0 - abs(width_ratio - 0.26))
            quality = peak_z * center_prior * width_prior
            candidates.append({
                "quality": float(quality),
                "x1": x1,
                "x2": x2,
                "features": features,
                "names": names,
                "level_y": level_y,
                "score": score,
                "peak_z": peak_z,
                "width_ratio": width_ratio,
                "center_ratio": float(center_ratio),
            })
    if not candidates:
        raise ValueError("旧版候选条带法没有生成有效候选。")
    selected = max(candidates, key=lambda item: item["quality"])
    if selected["peak_z"] <= 0:
        raise ValueError("旧版候选条带法没有找到有效变化峰。")
    x1, x2, level_y = selected["x1"], selected["x2"], selected["level_y"]
    selected_columns = np.zeros(width, dtype=bool)
    selected_columns[x1:x2] = True
    bottom_mask = np.arange(height) >= level_y
    return DirectFlapAnalysis(
        x1=x1,
        x2=x2,
        features=selected["features"],
        feature_names=selected["names"],
        level_y=level_y,
        bottom_state_score=selected["score"].astype(np.float32),
        bottom_connected_mask=bottom_mask,
        localization_profile=np.zeros(width, dtype=np.float32),
        localization_mask=selected_columns,
        confidence=float(selected["peak_z"]),
        metrics={
            "method": "legacy_76_candidate_change_peak",
            "boundary_method": "persistent_multivariate_change_peak",
            "localization": {
                "method": "legacy_4_widths_x_19_centres",
                "candidate_count": len(candidates),
                "selected_x1": x1,
                "selected_x2": x2,
                "selected_width": x2 - x1,
                "selected_width_ratio": round((x2 - x1) / width, 6),
                "selected_center_ratio": round(
                    (x1 + x2) / (2.0 * width), 6
                ),
                "requested_width_ratio": selected["width_ratio"],
                "requested_center_ratio": selected["center_ratio"],
                "localization_confidence": round(selected["quality"], 6),
                "fixed_center_used": False,
                "candidate_grid_used": True,
            },
            "peak_z": round(float(selected["peak_z"]), 6),
        },
    )


# 与 services.py 当前入口同名。若要手动接入 GUI，只需修改一行导入，
# services.py 中后续调用、绘图、读数换算和保存代码均无需改动。
analyze_direct_flap_strip = analyze_legacy_candidate_strips


def _read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix or ".png", image)
    if not success:
        raise ValueError(f"无法编码图片：{path}")
    encoded.tofile(path)
    return path.resolve()


def _choose_image() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="选择已经裁剪或透视校正的液位计 ROI",
        filetypes=[
            ("图片", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("所有文件", "*.*"),
        ],
    )
    root.destroy()
    return Path(path).resolve() if path else None


def _read_scale(value: float | None, prompt: str, default: float) -> float:
    if value is not None:
        return float(value)
    text = input(f"{prompt}（回车默认 {default:g}）: ").strip()
    return default if not text else float(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="旧版无底部连续约束的76候选条带液位读数"
    )
    parser.add_argument("image", nargs="?", help="已裁剪或透视校正的液位计ROI")
    parser.add_argument("--scale-min", type=float)
    parser.add_argument("--scale-max", type=float)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    image_path = Path(args.image).resolve() if args.image else _choose_image()
    if image_path is None:
        raise SystemExit("未选择图片。")
    scale_min = _read_scale(args.scale_min, "请输入量程下限", 0.0)
    scale_max = _read_scale(args.scale_max, "请输入量程上限", 100.0)
    if scale_max <= scale_min:
        raise SystemExit("量程上限必须大于量程下限。")

    image = _read_image(image_path)
    analysis = analyze_legacy_candidate_strips(image)
    height = image.shape[0]
    ratio = (height - analysis.level_y) / max(height, 1)
    reading = float(np.clip(
        scale_min + ratio * (scale_max - scale_min), scale_min, scale_max
    ))

    annotated = image.copy()
    cv2.rectangle(
        annotated,
        (analysis.x1, analysis.level_y),
        (analysis.x2 - 1, height - 1),
        (255, 255, 0),
        2,
    )
    cv2.line(
        annotated,
        (0, analysis.level_y),
        (image.shape[1] - 1, analysis.level_y),
        (0, 255, 255),
        2,
    )
    cv2.putText(
        annotated,
        f"level={reading:.3f}",
        (4, max(20, analysis.level_y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else image_path.parent / "legacy_candidate_results" / image_path.stem
    )
    result_path = _write_image(output_dir / "detection_result.png", annotated)
    colour_path = _write_image(
        output_dir / "colour_interval.png",
        image[analysis.level_y:, analysis.x1:analysis.x2],
    )
    report = {
        "source_image": str(image_path),
        "method": "legacy_76_candidate_change_peak_without_continuity_constraints",
        "scale_range": [scale_min, scale_max],
        "image_size": [image.shape[1], image.shape[0]],
        "colour_interval_xyxy": [
            analysis.x1, analysis.level_y, analysis.x2, height
        ],
        "boundary_y": analysis.level_y,
        "coloured_height": height - analysis.level_y,
        "level_ratio": round(float(ratio), 6),
        "reading": round(reading, 3),
        "peak_z": round(float(analysis.confidence), 6),
        "metrics": analysis.metrics,
        "detection_result": str(result_path),
        "colour_interval_image": str(colour_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "reading.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("========== 旧版76候选条带读数 ==========")
    print(f"图片: {image_path}")
    print(f"颜色区间: {report['colour_interval_xyxy']}")
    print(f"分界 y: {analysis.level_y}")
    print(f"液位比例: {ratio:.6f}")
    print(f"当前读数: {reading:.3f}")
    print(f"检测结果: {result_path}")
    print(f"JSON: {report_path.resolve()}")


if __name__ == "__main__":
    main()
