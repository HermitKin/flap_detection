"""Select an image, detect four Pose keypoints, and rectify the gauge."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = (
    PROJECT_ROOT / "object detection" / "runs" / "train"
    / "flap_pose4_with_yolo11s-set2" / "weights" / "best.pt"
)
POSE_HELPER = PROJECT_ROOT / "detect_pose_keypoints_yolo.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "perspective_results"
POINT_NAMES = ("TL", "TR", "BR", "BL")
POINT_COLOURS = (
    (0, 0, 255),      # TL red
    (0, 255, 255),    # TR yellow
    (255, 0, 0),      # BR blue
    (0, 255, 0),      # BL green
)


def choose_image_with_dialog() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="选择待关键点检测和透视矫正的图片",
        filetypes=[
            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("所有文件", "*.*"),
        ],
    )
    root.destroy()
    return Path(selected) if selected else None


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"无法读取图片: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    extension = path.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError(f"无法编码结果图片: {path}")
    encoded.tofile(path)


def run_pose(image_path: Path, model_path: Path, confidence: float, imgsz: int) -> list[dict]:
    process = subprocess.run(
        [
            sys.executable,
            str(POSE_HELPER),
            str(model_path),
            str(image_path),
            str(confidence),
            str(imgsz),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    marker = "POSE_RESULT_JSON="
    result_line = next(
        (line for line in process.stdout.splitlines() if line.startswith(marker)),
        None,
    )
    if process.returncode != 0 or result_line is None:
        details = process.stderr.strip() or process.stdout.strip() or "未知 YOLO Pose 错误"
        raise RuntimeError(f"关键点模型推理失败:\n{details}")
    return json.loads(result_line[len(marker):])


def select_best_gauge(detections: list[dict]) -> dict:
    candidates = []
    for detection in detections:
        points = detection.get("keypoints", [])
        if detection.get("class_id") != 0 or len(points) != 4:
            continue
        confidences = detection.get("keypoint_confidences", [1.0] * 4)
        quality = float(detection["box_confidence"]) * float(np.mean(confidences))
        candidates.append((quality, detection))
    if not candidates:
        raise RuntimeError("没有检测到类别 0 且包含四个关键点的液位计。")
    return max(candidates, key=lambda item: item[0])[1]


def order_points_toolight(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order arbitrary detector points as TL, TR, BR, BL like Toolight.

    Toolight chooses TL/BR from the minimum/maximum ``x + y`` values, then
    chooses TR from the remaining point with the maximum ``x - y`` value.
    The final remaining point is BL. The returned indices let callers keep
    keypoint confidences aligned with the reordered coordinates.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("关键点不是有效的 4x2 坐标。")

    indices = np.arange(4)
    coordinate_sum = points[:, 0] + points[:, 1]
    tl_index = int(np.argmin(coordinate_sum))
    br_index = int(np.argmax(coordinate_sum))
    remaining = indices[(indices != tl_index) & (indices != br_index)]
    if tl_index == br_index or len(remaining) != 2:
        raise ValueError("四个检测点无法唯一确定左上角和右下角。")

    x_minus_y = points[remaining, 0] - points[remaining, 1]
    tr_position = int(np.argmax(x_minus_y))
    tr_index = int(remaining[tr_position])
    bl_index = int(remaining[1 - tr_position])
    order = np.array([tl_index, tr_index, br_index, bl_index], dtype=np.int64)
    if len(np.unique(order)) != 4:
        raise ValueError("四个检测点无法按 TL/TR/BR/BL 唯一排序。")
    return points[order].astype(np.float32), order


def validate_tl_tr_br_bl(points: np.ndarray, image_shape: tuple[int, ...]) -> None:
    """Validate the dataset-defined TL, TR, BR, BL keypoint order."""
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("关键点不是有效的 4x2 坐标。")
    contour = points.reshape(-1, 1, 2).astype(np.float32)
    area = abs(float(cv2.contourArea(contour)))
    if area < 20.0:
        raise ValueError(f"四关键点面积过小，无法透视矫正: {area:.2f}")
    if not cv2.isContourConvex(np.round(contour).astype(np.int32)):
        raise ValueError("TL/TR/BR/BL 组成的四边形不是凸四边形。")
    height, width = image_shape[:2]
    margin_x, margin_y = width * 0.10, height * 0.10
    if ((points[:, 0] < -margin_x) | (points[:, 0] > width + margin_x)
            | (points[:, 1] < -margin_y) | (points[:, 1] > height + margin_y)).any():
        raise ValueError("关键点超出合理图像范围。")


def draw_keypoint_positions(
    image: np.ndarray,
    points: np.ndarray,
    confidences: np.ndarray,
    box: np.ndarray,
    box_confidence: float,
    connect: bool,
) -> np.ndarray:
    canvas = image.copy()
    x1, y1, x2, y2 = np.round(box).astype(int)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)
    if connect:
        cv2.polylines(
            canvas,
            [np.round(points).astype(np.int32)],
            True,
            (255, 0, 255),
            3,
            cv2.LINE_AA,
        )
    for index, ((x, y), confidence) in enumerate(zip(points, confidences)):
        point = (int(round(float(x))), int(round(float(y))))
        cv2.circle(canvas, point, 7, POINT_COLOURS[index], -1, cv2.LINE_AA)
        label = (
            POINT_NAMES[index]
            if canvas.shape[1] < 300
            else f"{POINT_NAMES[index]} {confidence:.3f} ({point[0]},{point[1]})"
        )
        font_scale = 0.50 if canvas.shape[1] < 300 else 0.52
        thickness = 1 if canvas.shape[1] < 300 else 2
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        text_x = int(np.clip(point[0] + 6, 0, max(0, canvas.shape[1] - text_width - 1)))
        preferred_y = point[1] + text_height + 8 if index < 2 else point[1] - 8
        text_y = int(np.clip(preferred_y, text_height + 1, canvas.shape[0] - 2))
        cv2.putText(
            canvas, label, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, POINT_COLOURS[index], thickness, cv2.LINE_AA,
        )
    if canvas.shape[1] >= 300:
        cv2.putText(
            canvas, f"Pose box confidence={box_confidence:.3f}", (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA,
        )
    return canvas


def solve_inverse_homography(
    destination_rectangle: np.ndarray,
    source_quadrilateral: np.ndarray,
) -> np.ndarray:
    """Solve the destination-to-source homography used for inverse sampling."""
    rows: list[list[float]] = []
    values: list[float] = []
    for (x, y), (source_x, source_y) in zip(
        destination_rectangle.astype(np.float64),
        source_quadrilateral.astype(np.float64),
    ):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -source_x * x, -source_x * y])
        values.append(source_x)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -source_y * x, -source_y * y])
        values.append(source_y)
    try:
        coefficients = np.linalg.solve(
            np.asarray(rows, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("四点形成的单应性矩阵为奇异矩阵，无法矫正。") from exc
    return np.append(coefficients, 1.0).reshape(3, 3)


def inverse_map_bilinear(
    image: np.ndarray,
    destination_to_source: np.ndarray,
    output_width: int,
    output_height: int,
    chunk_rows: int = 64,
) -> np.ndarray:
    """Apply Toolight-style inverse mapping with exact bilinear interpolation."""
    source_height, source_width = image.shape[:2]
    if source_width < 2 or source_height < 2:
        raise ValueError("原图宽高必须至少为 2 像素。")
    output = np.zeros((output_height, output_width, *image.shape[2:]), dtype=np.uint8)
    source = image.astype(np.float64, copy=False)
    x_coordinates = np.arange(output_width, dtype=np.float64)[None, :]
    matrix = np.asarray(destination_to_source, dtype=np.float64)

    for row_start in range(0, output_height, chunk_rows):
        row_end = min(row_start + chunk_rows, output_height)
        y_coordinates = np.arange(row_start, row_end, dtype=np.float64)[:, None]
        denominator = (
            matrix[2, 0] * x_coordinates
            + matrix[2, 1] * y_coordinates
            + matrix[2, 2]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            source_x = (
                matrix[0, 0] * x_coordinates
                + matrix[0, 1] * y_coordinates
                + matrix[0, 2]
            ) / denominator
            source_y = (
                matrix[1, 0] * x_coordinates
                + matrix[1, 1] * y_coordinates
                + matrix[1, 2]
            ) / denominator

        valid = (
            np.isfinite(source_x)
            & np.isfinite(source_y)
            & (source_x >= 0.0)
            & (source_x < source_width - 1)
            & (source_y >= 0.0)
            & (source_y < source_height - 1)
        )
        safe_source_x = np.where(np.isfinite(source_x), source_x, 0.0)
        safe_source_y = np.where(np.isfinite(source_y), source_y, 0.0)
        x0 = np.clip(np.floor(safe_source_x), 0, source_width - 2).astype(np.int64)
        y0 = np.clip(np.floor(safe_source_y), 0, source_height - 2).astype(np.int64)
        fraction_x = source_x - x0
        fraction_y = source_y - y0

        top_left = source[y0, x0]
        top_right = source[y0, x0 + 1]
        bottom_left = source[y0 + 1, x0]
        bottom_right = source[y0 + 1, x0 + 1]
        if image.ndim == 3:
            fraction_x = fraction_x[..., None]
            fraction_y = fraction_y[..., None]
            valid_for_pixels = valid[..., None]
        else:
            valid_for_pixels = valid
        interpolated = (
            top_left * (1.0 - fraction_x) * (1.0 - fraction_y)
            + top_right * fraction_x * (1.0 - fraction_y)
            + bottom_left * (1.0 - fraction_x) * fraction_y
            + bottom_right * fraction_x * fraction_y
        )
        # JavaScript's Math.round for non-negative channel values is floor(x + 0.5).
        rounded = np.floor(np.clip(interpolated, 0.0, 255.0) + 0.5).astype(np.uint8)
        output[row_start:row_end] = np.where(valid_for_pixels, rounded, 0)
    return output


def four_point_rectify(
    image: np.ndarray,
    points_tl_tr_br_bl: np.ndarray,
    forced_width: int | None,
    forced_height: int | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], dict]:
    ordered_points, _ = order_points_toolight(points_tl_tr_br_bl)
    tl, tr, br, bl = ordered_points
    top_width = float(np.linalg.norm(tr - tl))
    bottom_width = float(np.linalg.norm(br - bl))
    left_height = float(np.linalg.norm(bl - tl))
    right_height = float(np.linalg.norm(br - tr))

    average_width = (top_width + bottom_width) / 2.0
    average_height = (left_height + right_height) / 2.0
    # Match JavaScript Math.round rather than Python's ties-to-even round().
    output_width = forced_width or max(2, int(np.floor(average_width + 0.5)))
    output_height = forced_height or max(2, int(np.floor(average_height + 0.5)))
    if output_width < 2 or output_height < 2:
        raise ValueError(f"无效输出尺寸: {output_width}x{output_height}")

    # Match Toolight: solve a rectangle-to-source matrix and inverse-map every
    # destination pixel. It uses width/height (not width-1/height-1) as the
    # far rectangle coordinates.
    destination = np.array([
        [0, 0],
        [output_width, 0],
        [output_width, output_height],
        [0, output_height],
    ], dtype=np.float64)
    inverse_mapping_matrix = solve_inverse_homography(destination, ordered_points)
    rectified = inverse_map_bilinear(
        image, inverse_mapping_matrix, output_width, output_height
    )
    # Preserve the historical contract used by rectify_by_flap_strip.py: the
    # returned matrix maps source coordinates to destination coordinates.
    matrix = np.linalg.inv(inverse_mapping_matrix)
    matrix /= matrix[2, 2]
    measurements = {
        "top_width": round(top_width, 3),
        "bottom_width": round(bottom_width, 3),
        "left_height": round(left_height, 3),
        "right_height": round(right_height, 3),
        "estimated_width_average": round(average_width, 3),
        "estimated_height_average": round(average_height, 3),
    }
    return rectified, matrix, (output_width, output_height), measurements


def make_preview(image: np.ndarray, max_width: int = 1100, max_height: int = 850) -> np.ndarray:
    """Resize only a GUI preview; processing and saved images remain unchanged."""
    height, width = image.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                      interpolation=cv2.INTER_AREA)


def show_step(window_name: str, image: np.ndarray) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    preview = make_preview(image)
    cv2.imshow(window_name, preview)
    cv2.resizeWindow(window_name, preview.shape[1], preview.shape[0])
    print(f"已显示：{window_name}。按任意键继续。")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def unique_output_directory(root: Path, image_stem: str) -> Path:
    preferred = root / image_stem
    if not preferred.exists():
        return preferred
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"{image_stem}_{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在 read_level_roi.py 之前执行四关键点透视矫正。"
    )
    parser.add_argument("image", nargs="?", help="省略时首先弹出图片选择框")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO Pose 权重")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--rect-width", type=int, help="可选：强制标准矩形宽度")
    parser.add_argument("--rect-height", type=int, help="可选：强制标准矩形高度")
    parser.add_argument("--no-show", action="store_true", help="只保存，不弹出步骤预览")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else choose_image_with_dialog()
    if image_path is None:
        print("未选择图片，程序结束。")
        return
    image_path = image_path.resolve()
    model_path = Path(args.model).resolve()
    if not image_path.is_file():
        raise SystemExit(f"图片不存在: {image_path}")
    if not model_path.is_file():
        raise SystemExit(f"模型不存在: {model_path}")
    if not POSE_HELPER.is_file():
        raise SystemExit(f"Pose 辅助程序不存在: {POSE_HELPER}")

    image = read_image(image_path)
    print("正在进行 YOLO Pose 四关键点识别...")
    detection = select_best_gauge(run_pose(image_path, model_path, args.confidence, args.imgsz))
    detected_points = np.asarray(detection["keypoints"], dtype=np.float32)
    detected_confidences = np.asarray(
        detection["keypoint_confidences"], dtype=np.float32
    )
    points, point_order = order_points_toolight(detected_points)
    confidences = detected_confidences[point_order]
    box = np.asarray(detection["box"], dtype=np.float32)
    box_confidence = float(detection["box_confidence"])
    validate_tl_tr_br_bl(points, image.shape)

    keypoints_only = draw_keypoint_positions(
        image, points, confidences, box, box_confidence, connect=False
    )
    quadrilateral = draw_keypoint_positions(
        image, points, confidences, box, box_confidence, connect=True
    )

    print("\n识别到的关键点（原图坐标）:")
    for name, (x, y), confidence in zip(POINT_NAMES, points, confidences):
        print(f"  {name}: ({x:.2f}, {y:.2f}), confidence={confidence:.4f}")
    if float(confidences.min()) < 0.20:
        print(f"警告：最低关键点置信度仅 {float(confidences.min()):.4f}，请重点检查关键点图。")

    if not args.no_show:
        show_step("Step 1 - Pose keypoint positions", keypoints_only)
        show_step("Step 2 - Four-keypoint quadrilateral", quadrilateral)

    rectified, matrix, rectified_size, measurements = four_point_rectify(
        image, points, args.rect_width, args.rect_height
    )
    inverse_matrix = np.linalg.inv(matrix)
    inverse_matrix /= inverse_matrix[2, 2]

    output_dir = unique_output_directory(Path(args.output_root).resolve(), image_path.stem)
    output_dir.mkdir(parents=True, exist_ok=False)
    original_path = output_dir / "01_original.png"
    rectified_path = output_dir / "03_rectified_rectangle.png"
    # Diagnostic overlays are display-only. Never bake detection boxes,
    # keypoints, labels, or quadrilateral lines into saved image artifacts.
    write_image(original_path, image)
    write_image(rectified_path, rectified)

    report = {
        "source": str(image_path),
        "model": str(model_path),
        "original_size": [int(image.shape[1]), int(image.shape[0])],
        "box_xyxy": np.round(box, 3).tolist(),
        "box_confidence": round(box_confidence, 6),
        "keypoint_order": list(POINT_NAMES),
        "detected_keypoints_raw": np.round(detected_points, 3).tolist(),
        "geometry_sort_indices_tl_tr_br_bl": point_order.tolist(),
        "keypoints_tl_tr_br_bl": np.round(points, 3).tolist(),
        "keypoint_confidences": np.round(confidences, 6).tolist(),
        "source_side_lengths": measurements,
        "destination_sampling_rectangle": [
            [0, 0],
            [rectified_size[0], 0],
            [rectified_size[0], rectified_size[1]],
            [0, rectified_size[1]],
        ],
        "homography_source_to_destination": np.round(matrix, 9).tolist(),
        "homography_destination_to_source": np.round(inverse_matrix, 9).tolist(),
        "rectification_method": (
            "Toolight-style geometric point ordering; opposite-side average dimensions; "
            "destination-to-source homography; inverse mapping with bilinear interpolation"
        ),
        "rectified_size": list(rectified_size),
        "source_image_resized_or_overwritten": False,
        "saved_original_image": str(original_path),
        "diagnostic_overlays_saved": False,
        "reading_performed": False,
        "rectified_image": str(rectified_path),
    }
    report_path = output_dir / "result.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_show:
        show_step("Step 3 - Rectified standard rectangle", rectified)
        cv2.destroyAllWindows()

    print("\n========== 四关键点透视矫正结果 ==========")
    print(f"输入图片: {image_path}")
    print(f"输入尺寸: {image.shape[1]}x{image.shape[0]}（原图未修改）")
    print(f"关键点框置信度: {box_confidence:.4f}")
    print(f"标准矩形尺寸: {rectified_size[0]}x{rectified_size[1]}")
    print(f"未标注原图: {original_path}")
    print("关键点和四点连线仅用于屏幕预览，不写入保存图片。")
    print(f"矫正结果图: {rectified_path}")
    print(f"JSON报告: {report_path}")
    print("下一步可将 03_rectified_rectangle.png 作为 read_level_roi.py 的输入。")
    print("==========================================")


if __name__ == "__main__":
    main()
