"""Detect the magnetic-flap strip by scanning and rectify that strip."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from rectify_before_reading import (
    DEFAULT_MODEL,
    choose_image_with_dialog,
    four_point_rectify,
    read_image,
    run_pose,
    select_best_gauge,
    write_image,
)
from read_level_roi import auto_find_flap_strip


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "flap_strip_rectification_results"


def rotate_bound(image: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate without clipping and return the source-to-rotated affine matrix."""
    height, width = image.shape[:2]
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    cosine = abs(float(matrix[0, 0]))
    sine = abs(float(matrix[0, 1]))
    target_width = int(math.ceil(height * sine + width * cosine))
    target_height = int(math.ceil(height * cosine + width * sine))
    matrix[0, 2] += target_width / 2.0 - centre[0]
    matrix[1, 2] += target_height / 2.0 - centre[1]
    rotated = cv2.warpAffine(
        image, matrix, (target_width, target_height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, matrix


def detect_pose_with_rotation_fallback(
    image: np.ndarray,
    image_path: Path,
    model_path: Path,
    confidence: float,
    imgsz: int,
    max_rotation: int,
    rotation_step: int,
) -> tuple[dict, float, list[dict]]:
    """Detect directly, then try rotated views only when direct detection fails."""
    attempts: list[dict] = []
    try:
        direct = select_best_gauge(run_pose(image_path, model_path, confidence, imgsz))
        attempts.append({"angle": 0.0, "status": "detected", "box_confidence": direct["box_confidence"]})
        return direct, 0.0, attempts
    except (RuntimeError, ValueError) as exc:
        attempts.append({"angle": 0.0, "status": "failed", "reason": str(exc)})

    angles = []
    for value in range(rotation_step, max_rotation + 1, rotation_step):
        angles.extend((-value, value))
    candidates = []
    with tempfile.TemporaryDirectory(prefix="flap_pose_rotation_") as temporary:
        temporary_dir = Path(temporary)
        for angle in angles:
            rotated, affine = rotate_bound(image, float(angle))
            rotated_path = temporary_dir / f"rotation_{angle:+d}.png"
            write_image(rotated_path, rotated)
            try:
                detection = select_best_gauge(
                    run_pose(rotated_path, model_path, confidence, imgsz)
                )
            except (RuntimeError, ValueError) as exc:
                attempts.append({"angle": float(angle), "status": "failed", "reason": str(exc)})
                continue

            keypoint_confidences = np.asarray(
                detection.get("keypoint_confidences", [1.0] * 4), dtype=np.float64
            )
            quality = float(detection["box_confidence"]) * float(keypoint_confidences.mean())
            rotated_points = np.asarray(detection["keypoints"], dtype=np.float32)
            inverse_affine = cv2.invertAffineTransform(affine)
            source_points = cv2.transform(
                rotated_points.reshape(1, 4, 2), inverse_affine
            ).reshape(4, 2)
            mapped = dict(detection)
            mapped["keypoints"] = source_points.tolist()
            attempts.append({
                "angle": float(angle), "status": "detected",
                "box_confidence": float(detection["box_confidence"]),
                "quality": quality,
            })
            candidates.append((quality, mapped, float(angle)))

    if not candidates:
        raise RuntimeError(
            f"原图及 ±{max_rotation}° 旋转搜索均未检测到四关键点液位计。"
        )
    _, best_detection, best_angle = max(candidates, key=lambda item: item[0])
    return best_detection, best_angle, attempts


def smooth_1d(values: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel_size = max(3, int(kernel_size) | 1)
    return cv2.GaussianBlur(values.astype(np.float32).reshape(1, -1),
                            (kernel_size, 1), 0).ravel()


def scan_flap_columns(
    image: np.ndarray,
    pose_points: np.ndarray,
) -> tuple[int, int, int, int, np.ndarray, float, float]:
    """Find the flap span using the reader's HSV change-point window scan."""
    height, width = image.shape[:2]
    tl, tr, br, bl = pose_points
    top_y = max(0, int(round((tl[1] + tr[1]) / 2.0)))
    bottom_y = min(height - 1, int(round((bl[1] + br[1]) / 2.0)))
    if bottom_y - top_y < 40:
        raise ValueError("Pose 给出的有效高度过小。")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    horizontal_edges = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    trim = max(2, int(round((bottom_y - top_y) * 0.03)))
    column_energy = horizontal_edges[top_y + trim:bottom_y - trim + 1].mean(axis=0)
    column_energy = smooth_1d(column_energy, 5)

    # Reuse read_level_roi.py's colour-independent candidate scan. It compares
    # upper/lower HSV appearance changes over many strip widths and centres, so
    # dense scale ticks alone cannot win merely by having high edge energy.
    strip_left, strip_right_exclusive, _, _, _, _, peak_z = auto_find_flap_strip(image)
    left = int(strip_left)
    right = int(strip_right_exclusive - 1)
    if right - left + 1 < 7:
        raise ValueError("HSV扫描得到的翻片带宽度不可信。")
    # Kept only for the diagnostic profile; selection is based on peak_z above.
    threshold = float(np.median(column_energy[left:right + 1]))
    return left, right, top_y, bottom_y, column_energy, threshold, float(peak_z)


def fit_vertical_edge_ransac(
    edge: np.ndarray,
    gradient_x: np.ndarray,
    initial_x: int,
    top_y: int,
    bottom_y: int,
    band: int,
    seed: int,
) -> tuple[float, float, dict]:
    """Fit x = slope*y + intercept near one scanned strip boundary."""
    height, width = edge.shape
    x1 = max(0, initial_x - band)
    x2 = min(width - 1, initial_x + band)
    region_edge = edge[top_y:bottom_y + 1, x1:x2 + 1] > 0
    region_gradient = gradient_x[top_y:bottom_y + 1, x1:x2 + 1]
    gradient_limit = max(12.0, float(np.percentile(region_gradient, 55)))
    ys_local, xs_local = np.nonzero(region_edge & (region_gradient >= gradient_limit))
    xs = xs_local.astype(np.float64) + x1
    ys = ys_local.astype(np.float64) + top_y
    if len(xs) < 20:
        return 0.0, float(initial_x), {
            "status": "fallback_scan_boundary", "support_points": int(len(xs)), "inliers": 0,
        }

    rng = np.random.default_rng(seed)
    gauge_height = max(1.0, float(bottom_y - top_y))
    best_inliers = None
    best_score = -1e18
    best_model = (0.0, float(initial_x))
    for _ in range(1400):
        indices = rng.choice(len(xs), 2, replace=False)
        y_a, y_b = ys[indices[0]], ys[indices[1]]
        if abs(y_b - y_a) < gauge_height * 0.12:
            continue
        slope = (xs[indices[1]] - xs[indices[0]]) / (y_b - y_a)
        if abs(slope) > 0.10:
            continue
        intercept = xs[indices[0]] - slope * y_a
        residual = np.abs(xs - (slope * ys + intercept))
        inliers = residual <= 1.35
        count = int(inliers.sum())
        if count < 12:
            continue
        coverage = float(np.ptp(ys[inliers])) / gauge_height
        middle_x = slope * ((top_y + bottom_y) / 2.0) + intercept
        offset = abs(middle_x - initial_x)
        score = count + 80.0 * min(coverage, 1.0) - 4.0 * offset
        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_model = (float(slope), float(intercept))

    if best_inliers is None:
        return 0.0, float(initial_x), {
            "status": "fallback_scan_boundary", "support_points": int(len(xs)), "inliers": 0,
        }

    # Closed-form least-squares refinement on RANSAC inliers.
    y_in = ys[best_inliers]
    x_in = xs[best_inliers]
    y_mean = float(y_in.mean())
    x_mean = float(x_in.mean())
    denominator = float(np.sum((y_in - y_mean) ** 2))
    slope = 0.0 if denominator < 1e-8 else float(
        np.sum((y_in - y_mean) * (x_in - x_mean)) / denominator
    )
    intercept = x_mean - slope * y_mean
    fitted_top = slope * top_y + intercept
    fitted_middle = slope * ((top_y + bottom_y) / 2.0) + intercept
    fitted_bottom = slope * bottom_y + intercept
    if (abs(slope) > 0.10
            or abs(fitted_top - initial_x) > band
            or abs(fitted_middle - initial_x) > band
            or abs(fitted_bottom - initial_x) > band):
        return 0.0, float(initial_x), {
            "status": "fallback_scan_boundary", "support_points": int(len(xs)),
            "inliers": int(best_inliers.sum()),
        }
    return slope, intercept, {
        "status": "ransac_refined",
        "support_points": int(len(xs)),
        "inliers": int(best_inliers.sum()),
        "slope_dx_per_dy": round(slope, 6),
        "angle_from_vertical_deg": round(math.degrees(math.atan(slope)), 4),
    }


def profile_visualization(
    image: np.ndarray,
    energy: np.ndarray,
    threshold: float,
    left: int,
    right: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    chart_width = max(360, width * 3)
    chart_height = 260
    chart = np.full((chart_height, chart_width, 3), 255, dtype=np.uint8)
    maximum = max(float(energy.max()), 1.0)
    points = []
    for x, value in enumerate(energy):
        chart_x = int(round(x * (chart_width - 1) / max(width - 1, 1)))
        chart_y = chart_height - 25 - int(round(value / maximum * (chart_height - 50)))
        points.append((chart_x, chart_y))
    cv2.polylines(chart, [np.array(points, np.int32)], False, (180, 60, 20), 2, cv2.LINE_AA)
    threshold_y = chart_height - 25 - int(round(threshold / maximum * (chart_height - 50)))
    cv2.line(chart, (0, threshold_y), (chart_width - 1, threshold_y), (0, 0, 255), 1)
    for boundary, colour in ((left, (0, 180, 0)), (right, (0, 180, 0))):
        chart_x = int(round(boundary * (chart_width - 1) / max(width - 1, 1)))
        cv2.line(chart, (chart_x, 0), (chart_x, chart_height - 1), colour, 2)
    cv2.putText(chart, "Column horizontal-texture energy", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(chart, f"detected x={left}..{right}", (8, chart_height - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 0), 1, cv2.LINE_AA)
    return chart


def draw_scan_region(image: np.ndarray, left: int, right: int, top: int, bottom: int) -> np.ndarray:
    canvas = image.copy()
    overlay = image.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 255), -1)
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 255, 255), 2)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan the flap strip and rectify by its two long edges.")
    parser.add_argument("image", nargs="?", help="省略时弹出图片选择框")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--rotation-search-max", type=int, default=30,
                        help="原图Pose失败时自动搜索的最大旋转角度")
    parser.add_argument("--rotation-search-step", type=int, default=10)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else choose_image_with_dialog()
    if image_path is None:
        print("未选择图片。")
        return
    image_path = image_path.resolve()
    model_path = Path(args.model).resolve()
    image = read_image(image_path)

    detection, pose_detection_rotation, pose_attempts = detect_pose_with_rotation_fallback(
        image, image_path, model_path, args.confidence, args.imgsz,
        args.rotation_search_max, args.rotation_search_step,
    )
    pose_points = np.asarray(detection["keypoints"], dtype=np.float32)

    # Stage 1: use Pose only to establish an orientation-normalized coordinate
    # system. Column scanning is never performed directly in the tilted source.
    coarse_gauge, coarse_matrix, coarse_size, coarse_side_lengths = four_point_rectify(
        image, pose_points, None, None
    )
    coarse_width, coarse_height = coarse_size
    canonical_pose_points = np.array([
        [0, 0],
        [coarse_width - 1, 0],
        [coarse_width - 1, coarse_height - 1],
        [0, coarse_height - 1],
    ], dtype=np.float32)
    left_scan, right_scan, top_y, bottom_y, energy, threshold, scan_peak_z = scan_flap_columns(
        coarse_gauge, canonical_pose_points
    )

    gray = cv2.cvtColor(coarse_gauge, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (3, 3), 0)
    edge = cv2.Canny(blurred, 45, 130, L2gradient=True)
    gradient_x = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))
    band = max(4, int(round((right_scan - left_scan + 1) * 0.22)))
    left_slope, left_intercept, left_info = fit_vertical_edge_ransac(
        edge, gradient_x, left_scan, top_y, bottom_y, band, 20260825
    )
    right_slope, right_intercept, right_info = fit_vertical_edge_ransac(
        edge, gradient_x, right_scan, top_y, bottom_y, band, 20260826
    )

    flap_points_coarse = np.array([
        [left_slope * top_y + left_intercept, top_y],
        [right_slope * top_y + right_intercept, top_y],
        [right_slope * bottom_y + right_intercept, bottom_y],
        [left_slope * bottom_y + left_intercept, bottom_y],
    ], dtype=np.float32)
    top_width = float(np.linalg.norm(flap_points_coarse[1] - flap_points_coarse[0]))
    bottom_width = float(np.linalg.norm(flap_points_coarse[2] - flap_points_coarse[3]))
    if min(top_width, bottom_width) < 5:
        raise SystemExit("拟合后的翻片区域宽度过小，拒绝透视矫正。")

    scan_overlay = draw_scan_region(coarse_gauge, left_scan, right_scan, top_y, bottom_y)
    fit_overlay = coarse_gauge.copy()
    cv2.line(fit_overlay,
             (round(left_slope * top_y + left_intercept), top_y),
             (round(left_slope * bottom_y + left_intercept), bottom_y),
             (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(fit_overlay,
             (round(right_slope * top_y + right_intercept), top_y),
             (round(right_slope * bottom_y + right_intercept), bottom_y),
             (0, 255, 255), 2, cv2.LINE_AA)
    coarse_quadrilateral = fit_overlay.copy()
    cv2.polylines(coarse_quadrilateral, [np.round(flap_points_coarse).astype(np.int32)], True,
                  (255, 0, 255), 2, cv2.LINE_AA)
    for name, (x, y) in zip(("TL", "TR", "BR", "BL"), flap_points_coarse):
        point = (round(float(x)), round(float(y)))
        cv2.circle(coarse_quadrilateral, point, 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(coarse_quadrilateral, name, (point[0] + 3, max(14, point[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

    # Map the scanned flap quadrilateral back into the untouched source image,
    # then warp from the source once to avoid quality loss from double resampling.
    inverse_coarse_matrix = np.linalg.inv(coarse_matrix)
    flap_points_source = cv2.perspectiveTransform(
        flap_points_coarse.reshape(1, 4, 2), inverse_coarse_matrix
    ).reshape(4, 2).astype(np.float32)

    pose_overlay = image.copy()
    cv2.polylines(pose_overlay, [np.round(pose_points).astype(np.int32)], True,
                  (255, 255, 0), 2, cv2.LINE_AA)
    source_quadrilateral = pose_overlay.copy()
    cv2.polylines(source_quadrilateral, [np.round(flap_points_source).astype(np.int32)], True,
                  (255, 0, 255), 3, cv2.LINE_AA)
    for name, (x, y) in zip(("TL", "TR", "BR", "BL"), flap_points_source):
        point = (round(float(x)), round(float(y)))
        cv2.circle(source_quadrilateral, point, 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(source_quadrilateral, name, (point[0] + 3, max(14, point[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

    rectified, matrix, rectified_size, side_lengths = four_point_rectify(
        image, flap_points_source, None, None
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root).resolve() / f"{image_path.stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "01_original.png": image,
        "02_pose_quadrilateral.png": pose_overlay,
        "03_coarse_pose_rectification.png": coarse_gauge,
        "04_oriented_column_scan_profile.png": profile_visualization(
            coarse_gauge, energy, threshold, left_scan, right_scan
        ),
        "05_scanned_flap_in_coarse_view.png": scan_overlay,
        "06_coarse_clahe.png": clahe,
        "07_coarse_canny.png": edge,
        "08_fitted_edges_in_coarse_view.png": fit_overlay,
        "09_flap_quad_in_coarse_view.png": coarse_quadrilateral,
        "10_flap_quad_mapped_to_source.png": source_quadrilateral,
        "11_rectified_flap_from_source.png": rectified,
    }
    for filename, output_image in outputs.items():
        write_image(output_dir / filename, output_image)

    report = {
        "source": str(image_path),
        "method": "Pose coarse homography -> orientation-normalized scan -> inverse-map flap quad -> one source warp",
        "pose_detection_rotation_deg": pose_detection_rotation,
        "pose_rotation_attempts": pose_attempts,
        "pose_keypoints_tl_tr_br_bl": np.round(pose_points, 3).tolist(),
        "coarse_pose_rectified_size": list(coarse_size),
        "coarse_pose_side_lengths": coarse_side_lengths,
        "coarse_pose_homography": np.round(coarse_matrix, 9).tolist(),
        "scanned_columns_in_coarse_view": [left_scan, right_scan],
        "flap_scan_change_confidence_z": round(scan_peak_z, 4),
        "vertical_extent_in_coarse_view": [top_y, bottom_y],
        "edge_search_band_px": band,
        "left_edge": left_info,
        "right_edge": right_info,
        "flap_points_in_coarse_view": np.round(flap_points_coarse, 3).tolist(),
        "flap_points_in_source_tl_tr_br_bl": np.round(flap_points_source, 3).tolist(),
        "side_lengths": side_lengths,
        "homography": np.round(matrix, 9).tolist(),
        "rectified_size": list(rectified_size),
        "reading_performed": False,
    }
    (output_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("========== 按翻片区域透视矫正 ==========")
    print(f"Pose检测采用的临时旋转角度: {pose_detection_rotation:.1f}°")
    top_centre = (pose_points[0] + pose_points[1]) / 2.0
    bottom_centre = (pose_points[2] + pose_points[3]) / 2.0
    source_axis_angle = math.degrees(math.atan2(
        float(bottom_centre[0] - top_centre[0]),
        float(bottom_centre[1] - top_centre[1]),
    ))
    print(f"原图仪表轴线相对竖直角度: {source_axis_angle:.3f}°")
    print(f"粗矫正坐标尺寸: {coarse_width}x{coarse_height}")
    print(f"粗矫正坐标中的扫描列: x={left_scan}..{right_scan}")
    print(f"翻片HSV状态变化置信度: {scan_peak_z:.3f}")
    print(f"粗矫正坐标中的纵向范围: y={top_y}..{bottom_y}")
    print(f"左边拟合: {left_info}")
    print(f"右边拟合: {right_info}")
    print(f"反投影到原图的翻片四角: {np.round(flap_points_source, 2).tolist()}")
    print(f"矫正尺寸: {rectified_size[0]}x{rectified_size[1]}")
    print(f"输出目录: {output_dir}")

    if not args.no_show:
        for title, key in (
            ("Pose-normalized coarse gauge", "03_coarse_pose_rectification.png"),
            ("Oriented scan in coarse view", "05_scanned_flap_in_coarse_view.png"),
            ("Flap quad mapped to source", "10_flap_quad_mapped_to_source.png"),
            ("Rectified flap from source", "11_rectified_flap_from_source.png"),
        ):
            cv2.imshow(title, outputs[key])
            cv2.waitKey(0)
            cv2.destroyWindow(title)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
