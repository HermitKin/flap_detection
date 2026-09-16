from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "datasets" / "flap_data"
IMAGES = ROOT / "all_images"
LABELS = ROOT / "all_labels"
DERIVED = ROOT / "derived"
CLEAN_LABELS = DERIVED / "detection_labels_clipped"


def clip_box(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
    x2, y2 = min(1.0, x + w / 2), min(1.0, y + h / 2)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def main() -> None:
    CLEAN_LABELS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    clipped = 0
    instance_count = 0

    for image_path in sorted(IMAGES.glob("*.jpg")):
        label_path = LABELS / f"{image_path.stem}.txt"
        output_lines: list[str] = []
        instances = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                cls_s, x_s, y_s, w_s, h_s = line.split()
                cls = int(cls_s)
                original = tuple(map(float, (x_s, y_s, w_s, h_s)))
                cleaned = clip_box(*original)
                clipped += int(any(abs(a - b) > 1e-12 for a, b in zip(original, cleaned)))
                output_lines.append(f"{cls} " + " ".join(f"{value:.8f}" for value in cleaned))
                instances.append((cls, *cleaned))
        (CLEAN_LABELS / f"{image_path.stem}.txt").write_text(
            "\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8"
        )

        if not instances:
            rows.append({
                "image": image_path.name,
                "instance_id": "",
                "is_negative": 1,
                "review_status": "confirmed_negative_or_occluded",
            })
            continue

        for instance_id, (cls, x, y, w, h) in enumerate(instances):
            instance_count += 1
            rows.append({
                "image": image_path.name,
                "instance_id": instance_id,
                "is_negative": 0,
                "class_id": cls,
                "bbox_x": f"{x:.8f}",
                "bbox_y": f"{y:.8f}",
                "bbox_w": f"{w:.8f}",
                "bbox_h": f"{h:.8f}",
                "scene_id": "",
                "camera_id": "",
                "equipment_id": "",
                "split": "",
                "display_color_pair": "",
                "active_side": "",
                "scale_min": "",
                "scale_max": "",
                "unit": "",
                "kpt_top_left_x": "",
                "kpt_top_left_y": "",
                "kpt_top_right_x": "",
                "kpt_top_right_y": "",
                "kpt_bottom_left_x": "",
                "kpt_bottom_left_y": "",
                "kpt_bottom_right_x": "",
                "kpt_bottom_right_y": "",
                "kpt_level_x": "",
                "kpt_level_y": "",
                "review_status": "needs_pose_annotation",
            })

    fieldnames = [
        "image", "instance_id", "is_negative", "class_id",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "scene_id", "camera_id", "equipment_id", "split",
        "display_color_pair", "active_side", "scale_min", "scale_max", "unit",
        "kpt_top_left_x", "kpt_top_left_y",
        "kpt_top_right_x", "kpt_top_right_y",
        "kpt_bottom_left_x", "kpt_bottom_left_y",
        "kpt_bottom_right_x", "kpt_bottom_right_y",
        "kpt_level_x", "kpt_level_y", "review_status",
    ]
    with (DERIVED / "annotation_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"clean_labels={CLEAN_LABELS}")
    print(f"instances={instance_count}, clipped_boxes={clipped}, manifest_rows={len(rows)}")


if __name__ == "__main__":
    main()
