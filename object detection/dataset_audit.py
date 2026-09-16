from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "datasets" / "flap_data"
IMAGES = ROOT / "all_images"
LABELS = ROOT / "all_labels"
OUT = ROOT / "audit"


def quantiles(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {str(q): round(float(np.quantile(a, q)), 6) for q in (0, .05, .25, .5, .75, .95, 1)}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    images = sorted(IMAGES.glob("*.jpg"))
    records: list[dict] = []
    dims: Counter[tuple[int, int]] = Counter()
    classes: Counter[int] = Counter()
    invalid: list[str] = []
    missing: list[str] = []
    empty: list[str] = []
    thumbs: list[np.ndarray] = []

    for image_path in images:
        with Image.open(image_path) as im:
            width, height = im.size
            dims[(width, height)] += 1
            thumbs.append(np.asarray(im.convert("L").resize((32, 32)), dtype=np.float32))
        label_path = LABELS / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing.append(image_path.name)
            continue
        boxes = []
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                invalid.append(f"{label_path.name}:{line_no}: field_count={len(fields)}")
                continue
            cls, x, y, w, h = int(fields[0]), *map(float, fields[1:])
            classes[cls] += 1
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1 and
                    x - w / 2 >= 0 and y - h / 2 >= 0 and x + w / 2 <= 1 and y + h / 2 <= 1):
                invalid.append(f"{label_path.name}:{line_no}: out_of_bounds")
            boxes.append((cls, x, y, w, h))
        if not boxes:
            empty.append(label_path.name)
        records.append({"image": image_path.name, "width": width, "height": height, "boxes": boxes})

    all_boxes = [box for r in records for box in r["boxes"]]
    adjacent_mad = [float(np.mean(np.abs(thumbs[i] - thumbs[i - 1])) / 255.0) for i in range(1, len(thumbs))]
    report = {
        "image_count": len(images),
        "label_count": len(list(LABELS.glob("*.txt"))),
        "image_dimensions": {f"{w}x{h}": n for (w, h), n in dims.items()},
        "missing_labels": missing,
        "empty_labels": empty,
        "invalid_annotations": invalid,
        "class_instances": dict(sorted(classes.items())),
        "boxes_per_image": dict(sorted(Counter(len(r["boxes"]) for r in records).items())),
        "box_x_center_quantiles": quantiles([b[1] for b in all_boxes]),
        "box_y_center_quantiles": quantiles([b[2] for b in all_boxes]),
        "box_width_quantiles": quantiles([b[3] for b in all_boxes]),
        "box_height_quantiles": quantiles([b[4] for b in all_boxes]),
        "box_aspect_h_over_w_quantiles": quantiles([b[4] / b[3] for b in all_boxes]),
        "box_area_quantiles": quantiles([b[3] * b[4] for b in all_boxes]),
        "adjacent_frame_thumbnail_mad": {
            "mean": round(mean(adjacent_mad), 6),
            "median": round(median(adjacent_mad), 6),
            "quantiles": quantiles(adjacent_mad),
            "fraction_below_0.01": round(sum(x < .01 for x in adjacent_mad) / len(adjacent_mad), 6),
            "fraction_below_0.03": round(sum(x < .03 for x in adjacent_mad) / len(adjacent_mad), 6),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    indices = sorted(set(np.linspace(0, len(records) - 1, 12, dtype=int).tolist()))
    cell_w, cell_h = 480, 300
    sheet = Image.new("RGB", (cell_w * 3, cell_h * 4), "white")
    for slot, idx in enumerate(indices):
        record = records[idx]
        image_path = IMAGES / record["image"]
        with Image.open(image_path) as source:
            source = source.convert("RGB")
            draw = ImageDraw.Draw(source)
            for cls, x, y, w, h in record["boxes"]:
                x1, y1 = (x - w / 2) * source.width, (y - h / 2) * source.height
                x2, y2 = (x + w / 2) * source.width, (y + h / 2) * source.height
                draw.rectangle((x1, y1, x2, y2), outline="red", width=max(3, source.width // 500))
                draw.text((x1 + 4, max(0, y1 - 18)), f"class {cls}", fill="red", font=ImageFont.load_default())
            source.thumbnail((cell_w, cell_h - 24))
            tile = Image.new("RGB", (cell_w, cell_h), "white")
            tile.paste(source, ((cell_w - source.width) // 2, 20))
            ImageDraw.Draw(tile).text((6, 3), record["image"], fill="black", font=ImageFont.load_default())
            sheet.paste(tile, ((slot % 3) * cell_w, (slot // 3) * cell_h))
    sheet.save(OUT / "contact_sheet.jpg", quality=92)

    review_records = [r for r in records if len(r["boxes"]) != 1][:16]
    if review_records:
        review_sheet = Image.new("RGB", (cell_w * 4, cell_h * 4), "white")
        for slot, record in enumerate(review_records):
            image_path = IMAGES / record["image"]
            with Image.open(image_path) as source:
                source = source.convert("RGB")
                draw = ImageDraw.Draw(source)
                for cls, x, y, w, h in record["boxes"]:
                    x1, y1 = (x - w / 2) * source.width, (y - h / 2) * source.height
                    x2, y2 = (x + w / 2) * source.width, (y + h / 2) * source.height
                    draw.rectangle((x1, y1, x2, y2), outline="red", width=max(3, source.width // 500))
                source.thumbnail((cell_w, cell_h - 24))
                tile = Image.new("RGB", (cell_w, cell_h), "white")
                tile.paste(source, ((cell_w - source.width) // 2, 20))
                ImageDraw.Draw(tile).text(
                    (6, 3), f"{record['image']} boxes={len(record['boxes'])}", fill="black", font=ImageFont.load_default()
                )
                review_sheet.paste(tile, ((slot % 4) * cell_w, (slot // 4) * cell_h))
        review_sheet.save(OUT / "empty_and_multi_box_review.jpg", quality=92)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"contact_sheet={OUT / 'contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
