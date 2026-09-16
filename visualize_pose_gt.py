from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "object detection" / "datasets" / "flap_cleaned"
IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels" 
GT_DIR = ROOT / "gt"
POINTS_DIR = GT_DIR / "keypoints_only"
LINES_DIR = GT_DIR / "keypoints_with_lines"

# class 0/1/2: clearly separated colors. Visibility 1 is drawn hollow.
COLORS = [(255, 70, 70), (0, 220, 255), (120, 255, 70), (255, 80, 220)]
KPT_COUNT = 5


def parse_label(path: Path, width: int, height: int):
    instances = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if not values:
            continue
        expected = 5 + KPT_COUNT * 3
        if len(values) != expected:
            raise ValueError(f"{path.name}: expected {expected} fields, got {len(values)}")
        class_id = int(values[0])
        raw = list(map(float, values[5:]))
        points = []
        for index in range(KPT_COUNT):
            x, y, visibility = raw[index * 3 : index * 3 + 3]
            points.append((x * width, y * height, int(visibility)))
        instances.append((class_id, points))
    return instances


def draw_point(draw: ImageDraw.ImageDraw, x: float, y: float, visibility: int, color, radius: int):
    box = (x - radius, y - radius, x + radius, y + radius)
    if visibility == 1:
        draw.ellipse(box, fill="black", outline=color, width=max(2, radius // 2))
    else:
        draw.ellipse(box, fill=color, outline="black", width=max(1, radius // 3))


def annotate(image: Image.Image, instances, connect: bool) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    radius = max(3, round(min(image.size) / 45))
    line_width = max(2, radius // 2)
    for instance_index, (class_id, points) in enumerate(instances):
        color = COLORS[class_id % len(COLORS)]
        if connect:
            # The first four points follow TL -> TR -> BR -> BL. Never connect K4.
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                xa, ya, va = points[a]
                xb, yb, vb = points[b]
                if va > 0 and vb > 0:
                    draw.line((xa, ya, xb, yb), fill=color, width=line_width)
        for point_index, (x, y, visibility) in enumerate(points):
            if visibility <= 0:
                continue
            draw_point(draw, x, y, visibility, color, radius)
    return result


def main() -> None:
    if GT_DIR.exists():
        shutil.rmtree(GT_DIR)
    POINTS_DIR.mkdir(parents=True)
    LINES_DIR.mkdir(parents=True)

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    missing = []
    instance_count = 0
    visible_counts = [0, 0, 0]
    for image_path in image_paths:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing.append(image_path.name)
            continue
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        instances = parse_label(label_path, image.width, image.height)
        instance_count += len(instances)
        for _, points in instances:
            for _, _, visibility in points:
                if 0 <= visibility <= 2:
                    visible_counts[visibility] += 1
        annotate(image, instances, connect=False).save(POINTS_DIR / image_path.name, quality=95)
        annotate(image, instances, connect=True).save(LINES_DIR / image_path.name, quality=95)

    if missing:
        raise FileNotFoundError(f"missing labels for {len(missing)} images: {missing[:5]}")
    print(f"images={len(image_paths)}")
    print(f"instances={instance_count}")
    print(f"visibility_v0_v1_v2={visible_counts}")
    print(f"keypoints_only={POINTS_DIR}")
    print(f"keypoints_with_lines={LINES_DIR}")


if __name__ == "__main__":
    main()
