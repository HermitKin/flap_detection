"""Draw a rectangle and let SAM2 create a YOLO liquid-gauge segmentation label.

This tool mirrors the hook annotation workflow: every edit and navigation action
is written immediately, and the exact current image is restored after restart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sam3_level_annotator.geometry import mask_to_polygons
    from sam3_level_annotator.sam2_backend import Sam2Backend
else:
    from .geometry import mask_to_polygons
    from .sam2_backend import Sam2Backend


WINDOW_NAME = "Level Gauge SAM2 Annotator (CPU)"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FOOTER_HEIGHT = 78


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def load_polygons(label_path: Path, shape: tuple[int, int]) -> list[np.ndarray]:
    height, width = shape
    if not label_path.exists():
        return []
    polygons: list[np.ndarray] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 7 or (len(fields) - 1) % 2:
            print(f"WARNING: ignored malformed label row {label_path}:{line_number}", file=sys.stderr)
            continue
        try:
            values = np.asarray([float(value) for value in fields[1:]], dtype=np.float64)
        except ValueError:
            print(f"WARNING: ignored non-numeric label row {label_path}:{line_number}", file=sys.stderr)
            continue
        polygon = values.reshape(-1, 2) * np.asarray([width, height])
        polygons.append(polygon.astype(np.float32))
    return polygons


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding=encoding)
    temporary.replace(path)


def save_polygons(label_path: Path, polygons: list[np.ndarray], shape: tuple[int, int]) -> None:
    height, width = shape
    lines = []
    for polygon in polygons:
        normalized = polygon.astype(np.float64) / np.asarray([width, height])
        normalized = np.clip(normalized, 0.0, 1.0)
        lines.append("0 " + " ".join(f"{value:.6f}" for value in normalized.reshape(-1)))
    atomic_write_text(label_path, "\n".join(lines) + ("\n" if lines else ""), encoding="ascii")


class Annotator:
    def __init__(
        self,
        images_root: Path,
        labels_root: Path,
        images: list[Path],
        backend: Sam2Backend,
        progress_file: Path,
        max_width: int,
        max_height: int,
    ) -> None:
        self.images_root = images_root
        self.labels_root = labels_root
        self.images = images
        self.backend = backend
        self.progress_file = progress_file
        self.max_width = max_width
        self.max_height = max_height
        self.index = 0
        self.image = np.empty((0, 0, 3), np.uint8)
        self.polygons: list[np.ndarray] = []
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None
        self.scale = 1.0
        self.message = "Draw a rectangle around one liquid level gauge"
        self.reviewed = False
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse)
        self.load(self.load_progress())

    @property
    def image_path(self) -> Path:
        return self.images[self.index]

    @property
    def label_path(self) -> Path:
        relative = self.image_path.relative_to(self.images_root)
        return self.labels_root / relative.parent / f"{relative.stem}.txt"

    def sample_key(self, path: Path | None = None) -> str:
        return (path or self.image_path).relative_to(self.images_root).as_posix()

    def load_progress(self) -> int:
        if not self.progress_file.exists():
            return 0
        try:
            progress = json.loads(self.progress_file.read_text(encoding="utf-8"))
            sample = str(progress.get("sample", ""))
            for index, path in enumerate(self.images):
                if self.sample_key(path) == sample:
                    print(f"Resuming at {index + 1}/{len(self.images)}: {sample}")
                    return index
            return min(max(int(progress.get("index", 0)), 0), len(self.images) - 1)
        except (OSError, ValueError, TypeError):
            return 0

    def save_progress(self, action: str) -> None:
        payload = {
            "index": self.index,
            "sample": self.sample_key(),
            "image": self.image_path.resolve().as_posix(),
            "label": self.label_path.resolve().as_posix(),
            "instances": len(self.polygons),
            "reviewed": self.reviewed,
            "last_action": action,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_write_text(
            self.progress_file,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )

    def persist(self, action: str) -> None:
        """Atomically write the current YOLO label and resume cursor."""
        save_polygons(self.label_path, self.polygons, self.image.shape[:2])
        self.reviewed = True
        self.save_progress(action)
        self.message = f"Auto-saved: {len(self.polygons)} instance(s)"

    def load(self, index: int) -> None:
        self.index = max(0, min(index, len(self.images) - 1))
        self.image = read_image(self.image_path)
        self.polygons = load_polygons(self.label_path, self.image.shape[:2])
        self.drag_start = self.drag_end = None
        self.reviewed = self.label_path.exists()
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.message = "Loading image for SAM2..."
        self.show()
        cv2.waitKey(1)
        self.backend.set_image(Image.fromarray(rgb))
        self.message = (
            "Loaded existing label; draw another box"
            if self.polygons
            else "Draw a rectangle around one liquid level gauge"
        )
        self.save_progress("load")

    def navigate(self, step: int) -> None:
        # Edits are already persisted immediately. Merely passing an untouched
        # image must not silently turn it into a reviewed negative sample.
        self.save_progress("navigate")
        target = max(0, min(self.index + step, len(self.images) - 1))
        if target != self.index:
            self.load(target)

    def segment_box(self, display_box: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = [round(value / self.scale) for value in display_box]
        height, width = self.image.shape[:2]
        x1, x2 = sorted((int(np.clip(x1, 0, width - 1)), int(np.clip(x2, 0, width - 1))))
        y1, y2 = sorted((int(np.clip(y1, 0, height - 1)), int(np.clip(y2, 0, height - 1))))
        if x2 - x1 < 5 or y2 - y1 < 5:
            self.message = "Box is too small"
            return
        self.message = "SAM2 is segmenting on CPU..."
        self.show()
        cv2.waitKey(1)
        try:
            predictions = self.backend.predict_box((x1, y1, x2, y2))
        except Exception as exc:
            self.message = f"SAM2 failed: {exc}"
            print(f"SAM2 inference failed: {exc}", file=sys.stderr)
            return
        if not predictions:
            self.message = "No mask found; draw the box again"
            return
        candidates = mask_to_polygons(predictions[0].mask, min_area=30, epsilon_ratio=0.0025)
        if not candidates:
            self.message = "Mask was invalid; draw the box again"
            return
        self.polygons.append(candidates[0])
        self.persist("add_instance")
        self.message = f"Added instance {len(self.polygons)} and auto-saved"

    def undo(self) -> None:
        if self.polygons:
            self.polygons.pop()
            self.persist("undo")
            self.message = "Removed last instance and auto-saved"

    def clear(self) -> None:
        self.polygons.clear()
        self.persist("clear")
        self.message = "Cleared image and saved an empty label"

    def mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        display_height = round(self.image.shape[0] * self.scale)
        if y >= display_height:
            return
        display_width = round(self.image.shape[1] * self.scale)
        x = int(np.clip(x, 0, display_width - 1))
        y = int(np.clip(y, 0, display_height - 1))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = self.drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            sx, sy = self.drag_start
            self.drag_start = self.drag_end = None
            self.segment_box((min(sx, x), min(sy, y), max(sx, x), max(sy, y)))

    def show(self) -> None:
        height, width = self.image.shape[:2]
        self.scale = min(1.0, self.max_width / width, self.max_height / height)
        draw_width, draw_height = max(1, round(width * self.scale)), max(1, round(height * self.scale))
        canvas = cv2.resize(self.image, (draw_width, draw_height), interpolation=cv2.INTER_AREA)
        overlay = canvas.copy()
        display_polygons = [np.rint(poly * self.scale).astype(np.int32) for poly in self.polygons]
        for polygon in display_polygons:
            cv2.fillPoly(overlay, [polygon], (0, 220, 0))
        canvas = cv2.addWeighted(overlay, 0.38, canvas, 0.62, 0)
        for number, polygon in enumerate(display_polygons, start=1):
            cv2.polylines(canvas, [polygon], True, (0, 255, 255), 2, cv2.LINE_AA)
            center = np.rint(polygon.mean(axis=0)).astype(int)
            cv2.putText(canvas, str(number), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        (0, 0, 255), 2, cv2.LINE_AA)
        if self.drag_start is not None and self.drag_end is not None:
            cv2.rectangle(canvas, self.drag_start, self.drag_end, (255, 0, 255), 2)

        footer = np.zeros((FOOTER_HEIGHT, draw_width, 3), dtype=np.uint8)
        state = "reviewed" if self.reviewed else "unreviewed"
        title = (
            f"{self.index + 1}/{len(self.images)}  {self.sample_key()}  "
            f"masks={len(self.polygons)}  {state}"
        )
        cv2.putText(footer, title[:150], (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(footer, "N/Right next | P/Left previous | Z undo | C clear | S save | Q quit",
                    (7, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(footer, self.message[:150], (7, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (0, 200, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, np.vstack([canvas, footer]))

    def run(self) -> None:
        while True:
            self.show()
            key = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self.save_progress("close")
                break
            if key in (ord("q"), 27):
                self.save_progress("quit")
                break
            if key in (ord("n"), 83):
                self.navigate(1)
            elif key in (ord("p"), 81):
                self.navigate(-1)
            elif key == ord("s"):
                self.persist("manual_save")
            elif key == ord("z"):
                self.undo()
            elif key == ord("c"):
                self.clear()
        cv2.destroyAllWindows()


def collect_images(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="image root; subfolders are preserved")
    parser.add_argument("--labels", type=Path, required=True, help="YOLO segmentation label root")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--model", default="sam2_t.pt", help="SAM2 tiny checkpoint path or name")
    parser.add_argument("--imgsz", type=int, default=640, help="SAM2 inference size; 512 is faster, 640 is finer")
    parser.add_argument("--progress", type=Path, help="resume JSON; default is beside labels")
    parser.add_argument("--no-recursive", action="store_true", help="only read images directly in --images")
    parser.add_argument("--max-width", type=int, default=1500)
    parser.add_argument("--max-height", type=int, default=820)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_root = args.images.resolve()
    labels_root = args.labels.resolve()
    if not images_root.is_dir():
        raise SystemExit(f"Image directory not found: {images_root}")
    images = collect_images(images_root, recursive=not args.no_recursive)
    if not images:
        raise SystemExit(f"No supported images found under: {images_root}")
    progress_file = (args.progress or (labels_root.parent / "level_gauge_annotator_progress.json")).resolve()
    print(f"Loading SAM2 {args.model} on {args.device}; images={len(images)}", flush=True)
    backend = Sam2Backend(args.model, args.device, args.imgsz)
    Annotator(
        images_root, labels_root, images, backend, progress_file,
        args.max_width, args.max_height,
    ).run()


if __name__ == "__main__":
    main()
