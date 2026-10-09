"""Build reproducible documentation figures/GIFs from synthetic inputs only.

GUI screenshots are captured separately from ``app.py --demo``. This builder
never reads datasets, weights, private outputs, or desktop screenshots.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "level_gauge_showcase"))
from direct_hsv_reader import analyze_direct_flap_strip
from synthetic_demo import run_example

BG = "#0b1220"
CARD = "#142033"
LINE = "#2b3d52"
TEXT = "#eef5ff"
MUTED = "#a0b5cc"
TEAL = "#43dfb5"
AMBER = "#ffca6a"


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def text(draw, position, value, size=18, colour=TEXT, bold=False):
    draw.text(position, value, font=font(size, bold), fill=colour)


def image_panel(canvas, bgr, bounds):
    x, y, width, height = bounds
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas.paste(image, (x + (width - image.width) // 2, y + (height - image.height) // 2))


def arrow(draw, start, end, colour=TEAL):
    draw.line([start, end], fill=colour, width=3)
    x, y = end
    draw.polygon([(x, y), (x - 10, y - 6), (x - 10, y + 6)], fill=colour)


def pipeline_card(example, active=3):
    canvas = Image.new("RGB", (1200, 680), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (32, 22), "FLAP DETECTION", 33, bold=True)
    text(draw, (32, 68), "Geometry → rectification → bottom-connected HSV → level reading", 18, MUTED)
    record = example["record"]
    stages = [example["scene"], example["annotated"], record.rectified_visualization, record.annotated]
    labels = [("01 / SCENE", "Synthetic gauge"), ("02 / CORNERS", "Known TL / TR / BR / BL"),
              ("03 / RECTIFY", "Production homography"), ("04 / READ", "Production HSV algorithm")]
    for i, (view, (title, subtitle)) in enumerate(zip(stages, labels)):
        left = 32 + i * 292
        border = TEAL if i <= active else LINE
        draw.rounded_rectangle((left, 118, left + 264, 574), 15, fill=CARD, outline=border, width=2)
        text(draw, (left + 15, 135), title, 19, border, True)
        text(draw, (left + 15, 169), subtitle, 13, MUTED)
        if i <= active:
            image_panel(canvas, view, (left + 16, 205, 232, 350))
        else:
            text(draw, (left + 82, 370), "NEXT", 22, LINE, True)
        if i < 3:
            arrow(draw, (left + 272, 344), (left + 285, 344))
    report = record.reports[0]
    text(draw, (32, 600), f"Known input: {example['expected']:.1f} / 100", 22, MUTED)
    text(draw, (435, 600), f"Computed: {report['reading']:.2f} / 100", 22, TEAL, True)
    text(draw, (32, 644), "SYNTHETIC EXPLANATION · corners are construction geometry, not YOLO predictions", 15, MUTED)
    return canvas


def algorithm_card(example):
    record = example["record"]
    analysis = analyze_direct_flap_strip(record.rectified_visualization)
    canvas = Image.new("RGB", (1200, 670), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (32, 24), "HOW THE READER FINDS THE BOUNDARY", 29, bold=True)
    text(draw, (32, 73), "Circular Hue features · adaptive columns · one stable bottom-connected state", 18, MUTED)
    draw.rounded_rectangle((32, 125, 315, 610), 15, fill=CARD)
    image_panel(canvas, record.annotated, (60, 163, 226, 415))
    text(draw, (53, 137), "RECTIFIED ROI", 16, TEAL, True)
    # Graph of the score calculated by production code, not invented values.
    gx, gy, gw, gh = 388, 185, 710, 234
    scores = analysis.bottom_state_score
    limit = max(0.7, float(np.max(np.abs(scores))) * 1.1)
    draw.line((gx, gy + gh / 2, gx + gw, gy + gh / 2), fill=LINE, width=2)
    points = [(gx + i / (len(scores) - 1) * gw,
               gy + gh / 2 - float(v) / limit * (gh / 2 - 10)) for i, v in enumerate(scores)]
    draw.line(points, fill=TEAL, width=3)
    bx = gx + analysis.level_y / (len(scores) - 1) * gw
    draw.line((bx, gy - 12, bx, gy + gh + 9), fill=AMBER, width=2)
    text(draw, (gx, 132), "BOTTOM-STATE PROJECTION q(y)", 18, TEAL, True)
    text(draw, (gx, 154), "q > 0: bottom-like     q < 0: top-like", 16, MUTED)
    text(draw, (gx, 435), "TOP / y = 0", 14, MUTED)
    text(draw, (gx + gw - 160, 435), "BOTTOM / y = H", 14, MUTED)
    text(draw, (gx, 479), f"Boundary y* = {analysis.level_y}     Selected columns [{analysis.x1}, {analysis.x2})", 18, AMBER)
    text(draw, (gx, 524), "r = (H - y*) / H", 24, TEXT, True)
    text(draw, (gx, 563), f"L = Lmin + r × (Lmax - Lmin) = {record.reports[0]['reading']:.2f}", 20, TEAL)
    text(draw, (32, 632), "The full rectified height is the scale reference; the coloured segment is the measured interval.", 16, MUTED)
    return canvas


def architecture_card():
    canvas = Image.new("RGB", (1200, 480), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (32, 24), "FROM IMAGE TO AN AUDITABLE READING", 30, bold=True)
    titles = ["INPUT", "YOLO", "GEOMETRY", "HSV", "OUTPUT"]
    lines = [["Single / batch", "Clean pixel image"], ["Object box", "Four-corner Pose"],
             ["ROI coordinates", "Perspective warp"], ["Dense columns", "Bottom continuity"],
             ["Level + diagnostics", "PNG / JSON / CSV"]]
    for i, title in enumerate(titles):
        x = 32 + i * 234
        draw.rounded_rectangle((x, 133, x + 205, 325), 16, fill=CARD, outline=LINE, width=2)
        text(draw, (x + 17, 159), f"0{i + 1}", 21, TEAL, True)
        text(draw, (x + 17, 199), title, 21, TEXT, True)
        for j, line in enumerate(lines[i]):
            text(draw, (x + 17, 247 + j * 26), line, 15, MUTED)
        if i < 4:
            arrow(draw, (x + 211, 226), (x + 229, 226))
    text(draw, (32, 376), "Desktop UI: Tkinter + worker thread    |    Model inference: isolated Python subprocess", 19, MUTED)
    text(draw, (32, 417), "PUBLIC DEMO bypasses YOLO with known geometry; production rectification and reading remain active.", 16, TEAL)
    return canvas


def save_gif(frames, output, durations):
    # A shared palette prevents colour flicker between frames.
    palette = frames[-1].quantize(colors=128)
    converted = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    converted[0].save(output, save_all=True, append_images=converted[1:],
                      duration=durations, loop=0, optimize=True, disposal=2)


def build(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        example = run_example(Path(temp))
        pipeline_card(example).save(output / "pipeline.png")
        algorithm_card(example).save(output / "algorithm.png")
        architecture_card().save(output / "architecture.png")
        save_gif([pipeline_card(example, i) for i in range(4)], output / "pipeline.gif", [950, 950, 950, 2200])
        levels = [0.35, 0.44, 0.53, 0.62, 0.70, 0.78, 0.70, 0.62, 0.53, 0.44]
        dynamic = [pipeline_card(run_example(Path(temp), level)) for level in levels]
        save_gif(dynamic, output / "level-sweep.gif", [300] * len(dynamic))
        cases = []
        for colour in ("red", "blue"):
            for level in (0.35, 0.62, 0.78):
                sample = run_example(Path(temp), level, colour)
                actual = sample["record"].reports[0]["reading"]
                cases.append({"colour": colour, "expected": level * 100, "computed": actual,
                              "absolute_error": round(abs(actual - level * 100), 3)})
        summary = {"provenance": "synthetic geometry; no YOLO inference; not a real-world benchmark",
                   "scale_range": [0, 100], "cases": cases,
                   "mean_absolute_error": round(sum(c["absolute_error"] for c in cases) / len(cases), 3)}
        (output / "synthetic-results.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "assets")
    build(parser.parse_args().output.resolve())
