"""Colour-independent reader for a cropped magnetic-flap level gauge."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "object detection" / "runs" / "train" / "flap_pose4_with_yolo11n-set2" / "weights" / "best.pt"
YOLO_HELPER = PROJECT_ROOT / "detect_gauge_yolo.py"


def smooth_columns(values: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    kernel = max(3, int(round(sigma * 6)) | 1)
    return cv2.GaussianBlur(values.astype(np.float32), (1, kernel), sigmaX=0, sigmaY=sigma)


def extract_row_features(image: np.ndarray, x1: int, x2: int) -> tuple[np.ndarray, list[str]]:
    """Describe each row in HSV space without assuming a display colour."""
    strip = image[:, x1:x2]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))

    # Hue is circular in OpenCV: 0 and 179 are neighbours. Encode it as sin/cos
    # and down-weight hue when saturation is low, where hue is not meaningful.
    hue_angle = hsv[:, :, 0] * (2.0 * np.pi / 180.0)
    saturation_weight = hsv[:, :, 1] / 255.0
    hue_cos = np.cos(hue_angle) * saturation_weight
    hue_sin = np.sin(hue_angle) * saturation_weight

    features = np.column_stack([
        hue_cos.mean(axis=1),
        hue_sin.mean(axis=1),
        hsv[:, :, 1].mean(axis=1),
        hsv[:, :, 1].std(axis=1),
        hsv[:, :, 2].mean(axis=1),
        hsv[:, :, 2].std(axis=1),
        gray.std(axis=1),
        grad_x.mean(axis=1),
    ])
    names = ["H cos", "H sin", "S mean", "S std", "V mean", "V std",
             "gray std", "edge energy"]

    features = smooth_columns(features)
    median = np.median(features, axis=0)
    mad = np.median(np.abs(features - median), axis=0)
    features = (features - median) / np.maximum(1.4826 * mad, 1.0)
    return np.clip(features, -8.0, 8.0), names


def find_change_point(features: np.ndarray, margin_ratio: float = 0.05) -> tuple[int, np.ndarray]:
    """Find the split with the largest persistent multivariate appearance change."""
    height = len(features)
    window = max(12, round(height * 0.08))
    cumulative = np.vstack([np.zeros((1, features.shape[1])), np.cumsum(features, axis=0)])
    score = np.zeros(height, dtype=np.float32)

    start = max(window, round(height * margin_ratio))
    end = min(height - window, round(height * (1.0 - margin_ratio)))
    for y in range(start, end):
        above = (cumulative[y] - cumulative[y - window]) / window
        below = (cumulative[y + window] - cumulative[y]) / window
        score[y] = np.linalg.norm(above - below)

    score = cv2.GaussianBlur(score.reshape(-1, 1), (1, 9), sigmaX=0, sigmaY=2).ravel()
    return int(np.argmax(score)), score


def auto_find_flap_strip(image: np.ndarray) -> tuple[int, int, np.ndarray, list[str], int, np.ndarray, float]:
    """Search the gauge ROI for the vertical strip with the clearest state boundary."""
    height, width = image.shape[:2]
    if height < 40 or width < 7:
        raise ValueError(f"Gauge ROI is too small: {width}x{height}")

    candidates: list[tuple[float, int, int, np.ndarray, list[str], int, np.ndarray, float]] = []
    width_ratios = (0.20, 0.24, 0.28, 0.32)
    center_ratios = np.linspace(0.28, 0.72, 19)
    for width_ratio in width_ratios:
        strip_width = max(7, round(width * width_ratio))
        for center_ratio in center_ratios:
            center_x = round(width * center_ratio)
            x1 = max(0, center_x - strip_width // 2)
            x2 = min(width, x1 + strip_width)
            x1 = max(0, x2 - strip_width)
            if x2 - x1 < 7:
                continue

            features, names = extract_row_features(image, x1, x2)
            level_y, score = find_change_point(features)
            valid_score = score[score > 0]
            median = float(np.median(valid_score))
            mad = float(np.median(np.abs(valid_score - median)))
            peak_z = (float(score[level_y]) - median) / max(1.4826 * mad, 0.1)

            # The strip is normally near the gauge centre, but the prior is soft
            # enough to handle an off-centre tube after perspective correction.
            actual_center = (x1 + x2) / (2.0 * width)
            center_prior = max(0.65, 1.0 - abs(actual_center - 0.5))
            width_prior = max(0.75, 1.0 - abs(width_ratio - 0.26))
            quality = peak_z * center_prior * width_prior
            candidates.append((quality, x1, x2, features, names, level_y, score, peak_z))

    if not candidates:
        raise ValueError("No valid flap-strip candidates")
    _, x1, x2, features, names, level_y, score, peak_z = max(candidates, key=lambda item: item[0])
    return x1, x2, features, names, level_y, score, peak_z


def detect_gauges(image_path: Path, image: np.ndarray, model_path: Path, confidence: float) -> list[dict]:
    """Detect gauge boxes with YOLO; accept an already-cropped tall gauge as fallback."""
    try:
        process = subprocess.run(
            [sys.executable, str(YOLO_HELPER), str(model_path), str(image_path), str(confidence)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        marker = "YOLO_RESULT_JSON="
        result_line = next((line for line in process.stdout.splitlines() if line.startswith(marker)), None)
        if process.returncode != 0 or result_line is None:
            details = process.stderr.strip() or process.stdout.strip() or "unknown YOLO error"
            raise RuntimeError(details)
        raw_boxes = json.loads(result_line[len(marker):])
        boxes = []
        for item in raw_boxes:
            xyxy, conf = item["box"], item["confidence"]
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            if x2 - x1 >= 7 and y2 - y1 >= 40:
                boxes.append({"box": [x1, y1, x2, y2], "detection_confidence": float(conf)})
        if boxes:
            return sorted(boxes, key=lambda item: item["detection_confidence"], reverse=True)
    except Exception as exc:
        print(f"Warning: YOLO detection unavailable: {exc}")

    height, width = image.shape[:2]
    if height / max(width, 1) >= 2.0:
        return [{"box": [0, 0, width, height], "detection_confidence": None}]
    return []


def choose_image_with_dialog() -> Path | None:
    """Open the native image-selection dialog."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="选择待检测的液位计图片",
        filetypes=[
            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("所有文件", "*.*"),
        ],
    )
    root.destroy()
    return Path(selected) if selected else None


def prompt_scale_range(default_min: float = 0.0, default_max: float = 100.0) -> tuple[float, float]:
    """Read the physical scale range from CMD after the image is selected."""
    while True:
        try:
            min_text = input(f"请输入量程下限（直接回车默认 {default_min:g}）: ").strip()
            max_text = input(f"请输入量程上限（直接回车默认 {default_max:g}）: ").strip()
            scale_min = float(min_text) if min_text else default_min
            scale_max = float(max_text) if max_text else default_max
        except ValueError:
            print("量程必须是数字，请重新输入。")
            continue
        if scale_max <= scale_min:
            print("量程上限必须大于量程下限，请重新输入。")
            continue
        return scale_min, scale_max


def save_debug_images(
    image: np.ndarray,
    x1: int,
    x2: int,
    features: np.ndarray,
    names: list[str],
    score: np.ndarray,
    level_y: int,
    reading: float,
    output_dir: Path,
) -> list[Path]:
    # Import plotting only after YOLO inference. This avoids a Windows OpenMP
    # runtime collision between PyTorch and Matplotlib's numerical backend.
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    paths: list[Path] = []

    marked_roi = image.copy()
    cv2.rectangle(marked_roi, (x1, 0), (x2 - 1, image.shape[0] - 1), (0, 255, 0), 1)
    p = output_dir / "01_central_strip.png"
    cv2.imwrite(str(p), marked_roi)
    paths.append(p)

    strip = image[:, x1:x2]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    fig, axes = plt.subplots(1, 3, figsize=(6, 8), sharey=True)
    channel_names = ["Hue H", "Saturation S", "Brightness V"]
    cmaps = ["hsv", "gray", "gray"]
    for i, ax in enumerate(axes):
        ax.imshow(hsv[:, :, i], cmap=cmaps[i], aspect="auto",
                  vmin=0, vmax=179 if i == 0 else 255)
        ax.axhline(level_y, color="cyan", linewidth=1)
        ax.set_title(channel_names[i])
        ax.set_xticks([])
    axes[0].set_ylabel("Image row y")
    fig.suptitle("Central strip in HSV channels")
    fig.tight_layout()
    p = output_dir / "02_hsv_channels.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(7, 8))
    selected = list(range(len(names)))
    for i in selected:
        ax.plot(features[:, i], np.arange(len(features)), label=names[i], linewidth=1)
    ax.axhline(level_y, color="red", linestyle="--", label=f"boundary y={level_y}")
    ax.invert_yaxis()
    ax.set(xlabel="Robust normalized feature", ylabel="Image row y",
           title="HSV row appearance features")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = output_dir / "03_row_features.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.plot(score, np.arange(len(score)), color="#2060a8", linewidth=1.5)
    ax.axhline(level_y, color="red", linestyle="--", label=f"maximum y={level_y}")
    ax.invert_yaxis()
    ax.set(xlabel="Multivariate change score", ylabel="Image row y",
           title="Upper/lower appearance change-point score")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = output_dir / "04_change_score.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    # Keep the result at exactly the same pixel size as the input. The numerical
    # result is printed to stdout because a 49-pixel-wide image cannot hold text.
    result = image.copy()
    cv2.line(result, (0, level_y), (image.shape[1] - 1, level_y), (0, 255, 255), 1)
    p = output_dir / "05_final_reading.png"
    cv2.imwrite(str(p), result)
    paths.append(p)

    fig, axes = plt.subplots(1, 3, figsize=(12, 8), gridspec_kw={"width_ratios": [1, 2.3, 2.3]})
    axes[0].imshow(rgb)
    axes[0].axhline(level_y, color="yellow", linewidth=2)
    axes[0].set_title("Input + boundary")
    axes[1].plot(features[:, 0], np.arange(len(features)), label="H cos")
    axes[1].plot(features[:, 1], np.arange(len(features)), label="H sin")
    axes[1].plot(features[:, 2], np.arange(len(features)), label="S mean")
    axes[1].plot(features[:, 4], np.arange(len(features)), label="V mean")
    axes[1].axhline(level_y, color="red", linestyle="--")
    axes[1].invert_yaxis()
    axes[1].set_title("Appearance features")
    axes[1].legend()
    axes[2].plot(score, np.arange(len(score)))
    axes[2].axhline(level_y, color="red", linestyle="--")
    axes[2].invert_yaxis()
    axes[2].set_title(f"Change score -> {reading:.1f}")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    p = output_dir / "06_pipeline_summary.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive magnetic-flap gauge reader.")
    parser.add_argument("image", nargs="?", help="omit this argument to open an image-selection dialog")
    parser.add_argument("--output-dir", help="default: level_results/<image name>")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO gauge detector weights")
    parser.add_argument("--confidence", type=float, default=0.20, help="YOLO confidence threshold")
    parser.add_argument("--scale-min", type=float, help="scale lower limit; prompted in CMD if omitted")
    parser.add_argument("--scale-max", type=float, help="scale upper limit; prompted in CMD if omitted")
    parser.add_argument("--debug", action="store_true", help="save the six step-by-step analysis images")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else choose_image_with_dialog()
    if image_path is None:
        print("No image selected.")
        return
    if args.scale_min is None or args.scale_max is None:
        prompted_min, prompted_max = prompt_scale_range(
            default_min=args.scale_min if args.scale_min is not None else 0.0,
            default_max=args.scale_max if args.scale_max is not None else 100.0,
        )
        scale_min, scale_max = prompted_min, prompted_max
    else:
        scale_min, scale_max = args.scale_min, args.scale_max
    if scale_max <= scale_min:
        raise SystemExit("Scale maximum must be greater than scale minimum.")
    image_path = image_path.resolve()
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Cannot read image: {image_path}")

    output_dir = (Path(args.output_dir) if args.output_dir else
                  PROJECT_ROOT / "level_results" / image_path.stem).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detections = detect_gauges(image_path, image, Path(args.model), args.confidence)
    if not detections:
        raise SystemExit("No gauge detected. Try a clearer image or lower --confidence.")

    annotated = image.copy()
    reports = []
    for gauge_id, detection in enumerate(detections, start=1):
        gx1, gy1, gx2, gy2 = detection["box"]
        gauge = image[gy1:gy2, gx1:gx2]
        try:
            sx1, sx2, features, names, level_y, score, peak_z = auto_find_flap_strip(gauge)
        except ValueError as exc:
            print(f"Gauge {gauge_id} skipped: {exc}")
            continue

        # Calibrate from the actual YOLO gauge ROI instead of reusing pixel
        # anchors measured on a particular reference image. This assumes the
        # ROI top/bottom correspond to the physical scale maximum/minimum.
        scale_top_y = 0.0
        scale_bottom_y = float(gauge.shape[0] - 1)
        level_ratio = (scale_bottom_y - level_y) / (scale_bottom_y - scale_top_y)
        reading = scale_min + level_ratio * (scale_max - scale_min)
        reading = float(np.clip(reading, scale_min, scale_max))

        if args.debug:
            debug_dir = output_dir / f"gauge_{gauge_id:02d}"
            save_debug_images(gauge, sx1, sx2, features, names, score, level_y, reading, debug_dir)

        global_level_y = gy1 + level_y
        cv2.rectangle(annotated, (gx1, gy1), (gx2 - 1, gy2 - 1), (0, 255, 0), 2)
        cv2.rectangle(annotated, (gx1 + sx1, gy1), (gx1 + sx2 - 1, gy2 - 1), (255, 255, 0), 1)
        cv2.line(annotated, (gx1, global_level_y), (gx2 - 1, global_level_y), (0, 255, 255), 2)
        if image.shape[1] >= 150:
            label = f"#{gauge_id} level={reading:.1f}"
            text_y = max(14, gy1 - 6)
            cv2.putText(annotated, label, (gx1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)

        reports.append({
            "gauge_id": gauge_id,
            "gauge_box_xyxy": [gx1, gy1, gx2, gy2],
            "flap_strip_xyxy": [gx1 + sx1, gy1, gx1 + sx2, gy2],
            "boundary_y_local": level_y,
            "boundary_y_image": global_level_y,
            "scale_anchor_y_local": [scale_top_y, scale_bottom_y],
            "level_ratio": round(float(level_ratio), 6),
            "reading": round(reading, 3),
            "change_confidence_z": round(float(peak_z), 3),
            "detection_confidence": detection["detection_confidence"],
        })

    if not reports:
        raise SystemExit("Gauge was found, but no reliable flap region was detected.")

    result_path = output_dir / "detection_result.png"
    report_path = output_dir / "readings.json"
    cv2.imwrite(str(result_path), annotated)
    report_path.write_text(json.dumps({
        "source": str(image_path),
        "input_size": [image.shape[1], image.shape[0]],
        "result_size": [annotated.shape[1], annotated.shape[0]],
        "scale_range": [scale_min, scale_max],
        "gauges": reports,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========== 液位检测结果 ==========")
    print(f"选择的图片: {image_path}")
    print(f"设置的量程: {scale_min:g} ～ {scale_max:g}")
    print(f"检测到液位计: {len(reports)} 个")
    for report in reports:
        print(f"\n液位计 {report['gauge_id']}")
        print(f"  仪表框: {report['gauge_box_xyxy']}")
        print(f"  翻片区域: {report['flap_strip_xyxy']}")
        print(f"  液面分界 y: {report['boundary_y_image']}")
        print(f"  YOLO置信度: {report['detection_confidence']}")
        print(f"  变化点置信度: {report['change_confidence_z']:.2f}")
        print(f"  当前读数: {report['reading']:.1f}")
    print(f"\n输入/结果尺寸: {image.shape[1]}x{image.shape[0]}（未改变）")
    print(f"结果图片: {result_path}")
    print(f"JSON报告: {report_path}")
    print("==================================")


if __name__ == "__main__":
    main()
