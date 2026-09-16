"""Generate step-by-step figures for the automatic flap-strip search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from legacy.read_level_roi import extract_row_features, find_change_point


WIDTH_RATIOS = (0.20, 0.24, 0.28, 0.32)
CENTER_RATIOS = np.linspace(0.28, 0.72, 19)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix or ".png", image)
    if not success:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_first_box(json_path: Path) -> list[float]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    detections = payload.get("detections", [])
    if not detections:
        raise ValueError("The detection JSON contains no target.")
    box = detections[0].get("box_xyxy", detections[0].get("box"))
    if box is None or len(box) != 4:
        raise ValueError("The first target contains no valid box.")
    return [float(value) for value in box]


def scan_candidates(roi: np.ndarray) -> list[dict]:
    height, width = roi.shape[:2]
    records = []
    for width_ratio in WIDTH_RATIOS:
        strip_width = max(7, round(width * width_ratio))
        for center_ratio in CENTER_RATIOS:
            center_x = round(width * center_ratio)
            x1 = max(0, center_x - strip_width // 2)
            x2 = min(width, x1 + strip_width)
            x1 = max(0, x2 - strip_width)
            features, names = extract_row_features(roi, x1, x2)
            level_y, score = find_change_point(features)
            valid_score = score[score > 0]
            median = float(np.median(valid_score))
            mad = float(np.median(np.abs(valid_score - median)))
            peak_z = (float(score[level_y]) - median) / max(1.4826 * mad, 0.1)
            actual_center = (x1 + x2) / (2.0 * width)
            center_prior = max(0.65, 1.0 - abs(actual_center - 0.5))
            width_prior = max(0.75, 1.0 - abs(width_ratio - 0.26))
            quality = peak_z * center_prior * width_prior
            records.append({
                "width_ratio": float(width_ratio),
                "center_ratio": float(center_ratio),
                "x1": int(x1),
                "x2": int(x2),
                "level_y": int(level_y),
                "peak_z": float(peak_z),
                "center_prior": float(center_prior),
                "width_prior": float(width_prior),
                "quality": float(quality),
                "features": features,
                "feature_names": names,
                "change_score": score,
            })
    return records


def save_original_roi(image: np.ndarray, box: list[int], path: Path) -> None:
    annotated = image.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(annotated, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
    cv2.putText(annotated, "YOLO gauge ROI", (4, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    write_image(path, annotated)


def save_candidate_figure(roi: np.ndarray, output_path: Path) -> None:
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 4, figsize=(12, 10), constrained_layout=True)
    sample_centres = CENTER_RATIOS[::3]
    for axis, width_ratio in zip(axes, WIDTH_RATIOS):
        axis.imshow(rgb)
        strip_width = max(7, round(roi.shape[1] * width_ratio))
        colours = plt.cm.turbo(np.linspace(0.05, 0.95, len(sample_centres)))
        for centre, colour in zip(sample_centres, colours):
            center_x = round(roi.shape[1] * centre)
            x1 = max(0, center_x - strip_width // 2)
            x2 = min(roi.shape[1], x1 + strip_width)
            x1 = max(0, x2 - strip_width)
            axis.axvspan(x1, x2, facecolor=colour, edgecolor=colour, alpha=0.18)
            axis.axvline((x1 + x2) / 2, color=colour, linewidth=0.7)
        axis.set_title(f"width = {width_ratio:.0%}\n19 centre positions")
        axis.set_xlim(0, roi.shape[1] - 1)
        axis.set_ylim(roi.shape[0] - 1, 0)
        axis.set_xlabel("ROI x")
    axes[0].set_ylabel("ROI y")
    fig.suptitle("Vertical candidate strips scanned inside the detected gauge ROI", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_heatmap(records: list[dict], best: dict, output_path: Path) -> None:
    quality = np.asarray([record["quality"] for record in records], dtype=np.float32)
    quality = quality.reshape(len(WIDTH_RATIOS), len(CENTER_RATIOS))
    fig, axis = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    image = axis.imshow(quality, aspect="auto", cmap="viridis")
    best_row = WIDTH_RATIOS.index(best["width_ratio"])
    best_col = int(np.argmin(np.abs(CENTER_RATIOS - best["center_ratio"])))
    axis.scatter([best_col], [best_row], marker="x", s=180, linewidths=3,
                 color="red", label="selected maximum")
    axis.set_xticks(range(len(CENTER_RATIOS)))
    axis.set_xticklabels([f"{value:.2f}" for value in CENTER_RATIOS], rotation=45)
    axis.set_yticks(range(len(WIDTH_RATIOS)))
    axis.set_yticklabels([f"{value:.0%}" for value in WIDTH_RATIOS])
    axis.set_xlabel("candidate centre / ROI width")
    axis.set_ylabel("candidate width / ROI width")
    axis.set_title("Candidate quality = change peak z-score × centre prior × width prior")
    axis.legend(loc="upper right")
    fig.colorbar(image, ax=axis, label="quality score")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_selected_strip(roi: np.ndarray, best: dict, output_path: Path) -> None:
    x1, x2, level_y = best["x1"], best["x2"], best["level_y"]
    annotated = roi.copy()
    cv2.rectangle(annotated, (x1, 0), (x2 - 1, roi.shape[0] - 1), (0, 255, 0), 2)
    cv2.line(annotated, (0, level_y), (roi.shape[1] - 1, level_y), (0, 0, 255), 2)
    strip = roi[:, x1:x2]
    enlarged_width = max(120, strip.shape[1] * 6)
    enlarged = cv2.resize(strip, (enlarged_width, strip.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.line(enlarged, (0, level_y), (enlarged_width - 1, level_y), (0, 0, 255), 2)

    fig, axes = plt.subplots(1, 2, figsize=(8, 10), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.5]})
    axes[0].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Selected in ROI\nx={x1}:{x2}, local y={level_y}")
    axes[1].imshow(cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Selected strip enlarged\nred line = strongest change")
    for axis in axes:
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_change_score(roi: np.ndarray, best: dict, output_path: Path) -> None:
    strip = roi[:, best["x1"]:best["x2"]]
    score = best["change_score"]
    rows = np.arange(len(score))
    fig, axes = plt.subplots(1, 2, figsize=(8, 9), constrained_layout=True)
    axes[0].imshow(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB), aspect="auto")
    axes[0].axhline(best["level_y"], color="red", linewidth=2)
    axes[0].set_title("Selected strip")
    axes[0].set_ylabel("local row y")
    axes[1].plot(score, rows, color="navy")
    axes[1].axhline(best["level_y"], color="red", linewidth=2,
                    label=f"maximum y={best['level_y']}")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("upper/lower feature-change score")
    axes[1].set_ylabel("local row y")
    axes[1].set_title(f"Change score, peak z={best['peak_z']:.2f}")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("detection_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    image = read_image(args.image)
    raw_box = load_first_box(args.detection_json)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in raw_box]
    box = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
    roi = image[box[1]:box[3], box[0]:box[2]]
    records = scan_candidates(roi)
    best = max(records, key=lambda record: record["quality"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_original_roi(image, box, args.output_dir / "01_detected_gauge_roi.png")
    save_candidate_figure(roi, args.output_dir / "02_vertical_candidate_scan.png")
    save_heatmap(records, best, args.output_dir / "03_candidate_quality_heatmap.png")
    save_selected_strip(roi, best, args.output_dir / "04_selected_strip.png")
    save_change_score(roi, best, args.output_dir / "05_selected_strip_change_score.png")

    write_json(args.output_dir / "strip_search_explanation.json", {
        "source_image": str(args.image.resolve()),
        "gauge_box_xyxy": box,
        "roi_size": [roi.shape[1], roi.shape[0]],
        "candidate_count": len(records),
        "width_ratios": list(WIDTH_RATIOS),
        "center_ratio_range": [float(CENTER_RATIOS[0]), float(CENTER_RATIOS[-1])],
        "selected": {key: value for key, value in best.items()
                     if key not in {"features", "feature_names", "change_score"}},
    })
    print(json.dumps({
        "candidate_count": len(records),
        "gauge_box_xyxy": box,
        "roi_size": [roi.shape[1], roi.shape[0]],
        "selected_x": [best["x1"], best["x2"]],
        "selected_y": best["level_y"],
        "selected_peak_z": round(best["peak_z"], 3),
        "selected_quality": round(best["quality"], 3),
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
