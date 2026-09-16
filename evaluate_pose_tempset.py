from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


WEIGHTS = Path("/workspace/runs/pose/flap_pose8_yolo11n/weights/best.pt")
SOURCE = Path("/workspace/datasets/tempset")
OUTPUT = Path("/workspace/runs/pose_eval/tempset_best")
POINTS_DIR = OUTPUT / "visualizations" / "points_only"
LINES_DIR = OUTPUT / "visualizations" / "with_geometry"
CONF = 0.25
KPT_CONF = 0.25

OUTER_COLOR = (70, 255, 70)   # BGR
INNER_COLOR = (255, 220, 0)
BOX_COLOR = (40, 160, 255)


def convex(points: np.ndarray) -> bool:
    return bool(cv2.isContourConvex(points.astype(np.float32)))


def geometry_status(points: np.ndarray, confidences: np.ndarray, width: int, height: int) -> tuple[bool, str, float, float]:
    if len(points) != 8 or len(confidences) != 8:
        return False, "not_8_keypoints", 0.0, 0.0
    if float(confidences.min()) < KPT_CONF:
        return False, "low_keypoint_confidence", 0.0, 0.0
    if np.any(points[:, 0] <= 0) or np.any(points[:, 1] <= 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
        return False, "keypoint_outside_or_zero", 0.0, 0.0
    outer, inner = points[:4], points[4:]
    outer_area = abs(float(cv2.contourArea(outer.astype(np.float32))))
    inner_area = abs(float(cv2.contourArea(inner.astype(np.float32))))
    if outer_area < 1 or inner_area < 1:
        return False, "degenerate_quad", outer_area, inner_area
    if not convex(outer) or not convex(inner):
        return False, "non_convex_quad", outer_area, inner_area
    if inner_area >= outer_area:
        return False, "inner_not_smaller", outer_area, inner_area
    inside = [cv2.pointPolygonTest(outer.astype(np.float32), tuple(map(float, point)), False) >= 0 for point in inner]
    if not all(inside):
        return False, "inner_not_inside_outer", outer_area, inner_area
    return True, "ok", outer_area, inner_area


def draw_prediction(image: np.ndarray, box: np.ndarray, box_conf: float, points: np.ndarray,
                    confidences: np.ndarray, connect: bool) -> None:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)
    cv2.putText(image, f"flap {box_conf:.2f}", (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, BOX_COLOR, 1, cv2.LINE_AA)
    radius = max(3, round(min(image.shape[:2]) / 45))
    valid = (
        (confidences >= KPT_CONF)
        & (points[:, 0] > 0) & (points[:, 1] > 0)
        & (points[:, 0] < image.shape[1]) & (points[:, 1] < image.shape[0])
    )
    if connect:
        for color, offset in ((OUTER_COLOR, 0), (INNER_COLOR, 4)):
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                ia, ib = offset + a, offset + b
                if valid[ia] and valid[ib]:
                    cv2.line(image, tuple(points[ia].astype(int)), tuple(points[ib].astype(int)), color, 2, cv2.LINE_AA)
    for index, (point, confidence) in enumerate(zip(points, confidences)):
        if not valid[index]:
            continue
        color = OUTER_COLOR if index < 4 else INNER_COLOR
        center = tuple(point.astype(int))
        cv2.circle(image, center, radius, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, center, max(2, radius - 1), color, -1, cv2.LINE_AA)


def main() -> None:
    image_paths = sorted(path for path in SOURCE.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not image_paths:
        raise FileNotFoundError(f"no images in {SOURCE}")
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    POINTS_DIR.mkdir(parents=True)
    LINES_DIR.mkdir(parents=True)

    model = YOLO(str(WEIGHTS))
    rows = []
    speeds = []
    results = model.predict(source=[str(path) for path in image_paths], imgsz=960, conf=CONF,
                            device=0, stream=True, verbose=False)
    for result in results:
        image_name = Path(result.path).name
        points_image = result.orig_img.copy()
        lines_image = result.orig_img.copy()
        detection_count = len(result.boxes)
        speeds.append(sum(float(result.speed.get(key, 0.0)) for key in ("preprocess", "inference", "postprocess")))
        if detection_count == 0:
            rows.append({"image": image_name, "detection_index": "", "detection_count": 0,
                         "box_conf": "", "mean_kpt_conf": "", "min_kpt_conf": "",
                         "geometry_ok": 0, "geometry_status": "no_detection",
                         "outer_area_px2": "", "inner_area_px2": ""})
        else:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            box_confs = result.boxes.conf.detach().cpu().numpy()
            keypoints = result.keypoints.xy.detach().cpu().numpy()
            if result.keypoints.conf is None:
                keypoint_confs = np.ones(keypoints.shape[:2], dtype=np.float32)
            else:
                keypoint_confs = result.keypoints.conf.detach().cpu().numpy()
            for index in range(detection_count):
                points = keypoints[index]
                kconf = keypoint_confs[index]
                ok, status, outer_area, inner_area = geometry_status(
                    points, kconf, result.orig_img.shape[1], result.orig_img.shape[0]
                )
                rows.append({
                    "image": image_name,
                    "detection_index": index,
                    "detection_count": detection_count,
                    "box_conf": f"{float(box_confs[index]):.6f}",
                    "mean_kpt_conf": f"{float(kconf.mean()):.6f}",
                    "min_kpt_conf": f"{float(kconf.min()):.6f}",
                    "geometry_ok": int(ok),
                    "geometry_status": status,
                    "outer_area_px2": f"{outer_area:.3f}",
                    "inner_area_px2": f"{inner_area:.3f}",
                })
                draw_prediction(points_image, xyxy[index], float(box_confs[index]), points, kconf, connect=False)
                draw_prediction(lines_image, xyxy[index], float(box_confs[index]), points, kconf, connect=True)
        cv2.imwrite(str(POINTS_DIR / image_name), points_image)
        cv2.imwrite(str(LINES_DIR / image_name), lines_image)

    with (OUTPUT / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_image_counts = {}
    for row in rows:
        per_image_counts[row["image"]] = int(row["detection_count"])
    detection_rows = [row for row in rows if row["detection_index"] != ""]
    status_counts = {}
    for row in detection_rows:
        status_counts[row["geometry_status"]] = status_counts.get(row["geometry_status"], 0) + 1
    # Run once more from the saved CSV-level inference is unnecessary: collect detailed
    # confidence/zero-point diagnostics from a lightweight second prediction stream.
    per_kpt_conf = [[] for _ in range(8)]
    per_kpt_zero = [0] * 8
    for result in model.predict(source=[str(path) for path in image_paths], imgsz=960, conf=CONF,
                                device=0, stream=True, verbose=False):
        if result.keypoints is None or result.keypoints.conf is None:
            continue
        pts_all = result.keypoints.xy.detach().cpu().numpy()
        conf_all = result.keypoints.conf.detach().cpu().numpy()
        for pts, confs in zip(pts_all, conf_all):
            for index in range(min(8, len(pts))):
                per_kpt_conf[index].append(float(confs[index]))
                per_kpt_zero[index] += int(float(pts[index, 0]) <= 0 or float(pts[index, 1]) <= 0)
    summary = {
        "weights": str(WEIGHTS),
        "source": str(SOURCE),
        "images": len(image_paths),
        "images_no_detection": sum(count == 0 for count in per_image_counts.values()),
        "images_one_detection": sum(count == 1 for count in per_image_counts.values()),
        "images_multiple_detections": sum(count > 1 for count in per_image_counts.values()),
        "total_detections": len(detection_rows),
        "mean_box_conf": float(np.mean([float(row["box_conf"]) for row in detection_rows])) if detection_rows else 0.0,
        "mean_keypoint_conf": float(np.mean([float(row["mean_kpt_conf"]) for row in detection_rows])) if detection_rows else 0.0,
        "geometry_status_counts": status_counts,
        "per_keypoint_mean_conf": [float(np.mean(values)) if values else 0.0 for values in per_kpt_conf],
        "per_keypoint_zero_or_clipped_count": per_kpt_zero,
        "mean_total_ms_per_image": float(np.mean(speeds)),
        "approx_fps": 1000.0 / float(np.mean(speeds)) if speeds and np.mean(speeds) > 0 else 0.0,
        "note": "No GT labels were present; accuracy, precision, recall, mAP, OKS and pixel error cannot be computed.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
