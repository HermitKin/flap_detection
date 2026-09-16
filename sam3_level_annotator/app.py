"""Tkinter GUI for semi-automatic liquid-level-gauge segmentation annotation."""

from __future__ import annotations

import argparse
import json
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sam3_level_annotator.geometry import load_yolo_masks, mask_iou, masks_to_yolo_lines
    from sam3_level_annotator.sam3_backend import Prediction, Sam3Backend
else:
    from .geometry import load_yolo_masks, mask_iou, masks_to_yolo_lines
    from .sam3_backend import Prediction, Sam3Backend


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = [(0, 208, 255), (255, 99, 71), (80, 220, 120), (190, 110, 255), (255, 210, 50)]


@dataclass(frozen=True)
class Annotation:
    mask: np.ndarray
    score: float | None
    source: str


class LevelGaugeAnnotator(tk.Tk):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.title("SAM3 液位计半自动分割标注")
        self.geometry("1450x900")
        self.minsize(1000, 650)

        self.image_dir = args.images.resolve() if args.images else None
        self.label_dir = args.labels.resolve() if args.labels else None
        self.images: list[Path] = []
        self.index = -1
        self.image: Image.Image | None = None
        self.annotations: list[Annotation] = []
        self.history: list[list[Annotation]] = []
        self.selected = -1
        self.dirty = False
        self.busy = False
        self.box_start: tuple[float, float] | None = None
        self.box_item: int | None = None
        self.display_scale = 1.0
        self.display_offset = (0.0, 0.0)
        self.tk_image: ImageTk.PhotoImage | None = None
        self.backend = Sam3Backend(args.device, args.checkpoint)

        self.prompt_var = tk.StringVar(value=args.prompt)
        self.threshold_var = tk.DoubleVar(value=args.threshold)
        self.box_mode_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="请选择图片目录")
        self.progress_var = tk.StringVar(value="0 / 0")

        self._build_ui()
        self._bind_keys()
        if self.image_dir:
            self.open_directory(self.image_dir)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="打开目录", command=self.choose_directory).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="上一张 A", command=self.previous_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="下一张 D", command=self.next_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="保存 Ctrl+S", command=self.save).pack(side=tk.LEFT, padx=3)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(toolbar, text="英文提示词:").pack(side=tk.LEFT)
        ttk.Entry(toolbar, textvariable=self.prompt_var, width=25).pack(side=tk.LEFT, padx=4)
        ttk.Label(toolbar, text="阈值:").pack(side=tk.LEFT)
        ttk.Spinbox(toolbar, from_=0.0, to=1.0, increment=0.05, textvariable=self.threshold_var, width=5).pack(
            side=tk.LEFT, padx=4
        )
        self.auto_button = ttk.Button(toolbar, text="自动识别并分割 R", command=self.run_text_prediction)
        self.auto_button.pack(side=tk.LEFT, padx=4)
        ttk.Label(toolbar, textvariable=self.progress_var).pack(side=tk.RIGHT, padx=6)

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        canvas_frame = ttk.Frame(body)
        side = ttk.Frame(body, padding=8, width=280)
        body.add(canvas_frame, weight=5)
        body.add(side, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="#202124", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.render())
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-3>", self.on_right_click)

        ttk.Label(side, text="实例列表").pack(anchor=tk.W)
        self.listbox = tk.Listbox(side, height=18, exportselection=False)
        self.listbox.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)
        ttk.Checkbutton(side, text="拖框补标模式", variable=self.box_mode_var).pack(anchor=tk.W, pady=3)
        ttk.Label(side, text="在漏检液位计外围拖一个框，SAM3 会补充分割。", wraplength=250).pack(anchor=tk.W)
        ttk.Button(side, text="删除选中 Del", command=self.delete_selected).pack(fill=tk.X, pady=(12, 3))
        ttk.Button(side, text="撤销 Ctrl+Z", command=self.undo).pack(fill=tk.X, pady=3)
        ttk.Button(side, text="清空本图", command=self.clear_annotations).pack(fill=tk.X, pady=3)
        ttk.Separator(side).pack(fill=tk.X, pady=12)
        ttk.Label(
            side,
            text="操作：\n• R：按提示词自动分割\n• 拖框：补标单个漏检目标\n• 单击 Mask：选中\n• 右键 / Del：删除\n• A / D：上一张 / 下一张\n• Ctrl+S：保存 YOLO 多边形",
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

        status = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=4)
        status.pack(fill=tk.X, side=tk.BOTTOM)

    def _bind_keys(self) -> None:
        self.bind("<Control-s>", lambda _event: self.save())
        self.bind("<Control-z>", lambda _event: self.undo())
        self.bind("<Delete>", lambda _event: self.delete_selected())
        self.bind("<Key-r>", lambda _event: self.run_text_prediction())
        self.bind("<Key-a>", lambda _event: self.previous_image())
        self.bind("<Key-d>", lambda _event: self.next_image())

    def choose_directory(self) -> None:
        selected = filedialog.askdirectory(title="选择待标注图片目录")
        if selected:
            self.open_directory(Path(selected))

    def open_directory(self, directory: Path) -> None:
        images = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            messagebox.showwarning("没有图片", f"目录中没有支持的图片：\n{directory}")
            return
        self.image_dir = directory.resolve()
        self.label_dir = self.label_dir or (self.image_dir.parent / "labels")
        self.images = images
        self.load_image(0)

    def load_image(self, index: int) -> None:
        if not self.images or not 0 <= index < len(self.images) or not self._may_leave_image():
            return
        path = self.images[index]
        try:
            with Image.open(path) as loaded:
                self.image = ImageOps.exif_transpose(loaded).convert("RGB")
        except Exception as exc:
            messagebox.showerror("读取失败", f"无法读取 {path.name}\n{exc}")
            return
        self.index = index
        label_path = self._label_path(path)
        masks = load_yolo_masks(label_path, (self.image.height, self.image.width))
        self.annotations = [Annotation(mask, None, "existing") for mask in masks]
        self.history.clear()
        self.selected = -1
        self.dirty = False
        self._update_listbox()
        self.progress_var.set(f"{index + 1} / {len(self.images)}")
        self.status_var.set(f"{path.name}  |  {self.image.width}×{self.image.height}  |  已有 {len(masks)} 个实例")
        self.render()

    def _may_leave_image(self) -> bool:
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel("尚未保存", "当前图片的标注已修改，是否先保存？")
        if answer is None:
            return False
        if answer:
            return self.save()
        return True

    def previous_image(self) -> None:
        if self.index > 0 and not self.busy:
            self.load_image(self.index - 1)

    def next_image(self) -> None:
        if self.index + 1 < len(self.images) and not self.busy:
            self.load_image(self.index + 1)

    def run_text_prediction(self) -> None:
        if self.image is None or self.busy:
            return
        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("缺少提示词", "请输入英文目标词，例如 liquid level gauge。")
            return
        try:
            threshold = float(self.threshold_var.get())
        except (ValueError, tk.TclError):
            messagebox.showwarning("阈值错误", "阈值应为 0 到 1 的小数。")
            return
        if not 0.0 <= threshold <= 1.0:
            messagebox.showwarning("阈值错误", "阈值应为 0 到 1 的小数。")
            return
        image = self.image.copy()

        def work() -> list[Prediction]:
            self.backend.set_image(image)
            return self.backend.predict_text(prompt, threshold)

        self._run_in_background("SAM3 正在自动识别液位计…", work, self._replace_predictions)

    def _replace_predictions(self, predictions: list[Prediction]) -> None:
        self._remember()
        self.annotations = [Annotation(p.mask, p.score, p.source) for p in predictions]
        self.selected = 0 if self.annotations else -1
        self.dirty = True
        self._refresh(f"自动识别完成：{len(predictions)} 个候选实例，请人工检查后保存")

    def _add_box_prediction(self, predictions: list[Prediction]) -> None:
        if not predictions:
            self.status_var.set("拖框区域未得到有效 Mask，请扩大或调整框后重试")
            return
        prediction = predictions[0]
        if any(mask_iou(item.mask, prediction.mask) > 0.85 for item in self.annotations):
            self.status_var.set("该目标与已有实例高度重合，未重复添加")
            return
        self._remember()
        self.annotations.append(Annotation(prediction.mask, prediction.score, prediction.source))
        self.selected = len(self.annotations) - 1
        self.dirty = True
        self._refresh("拖框补标完成")

    def _run_in_background(self, status: str, function, on_success) -> None:
        self.busy = True
        self.auto_button.state(["disabled"])
        self.status_var.set(status + "（首次运行会下载/加载模型）")

        def target() -> None:
            try:
                result = function()
            except Exception as exc:
                detail = traceback.format_exc()
                self.after(0, lambda: self._background_error(exc, detail))
            else:
                self.after(0, lambda: self._background_success(result, on_success))

        threading.Thread(target=target, daemon=True).start()

    def _background_success(self, result, callback) -> None:
        self.busy = False
        self.auto_button.state(["!disabled"])
        callback(result)

    def _background_error(self, exc: Exception, detail: str) -> None:
        self.busy = False
        self.auto_button.state(["!disabled"])
        self.status_var.set("SAM3 推理失败")
        messagebox.showerror("SAM3 推理失败", f"{exc}\n\n详细信息已输出到终端。")
        print(detail, file=sys.stderr)

    def on_left_press(self, event: tk.Event) -> None:
        if self.image is None or self.busy:
            return
        if self.box_mode_var.get():
            self.box_start = (event.x, event.y)
            self.box_item = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#ffeb3b", width=2)
        else:
            self.select_at(event.x, event.y)

    def on_left_drag(self, event: tk.Event) -> None:
        if self.box_start and self.box_item:
            self.canvas.coords(self.box_item, self.box_start[0], self.box_start[1], event.x, event.y)

    def on_left_release(self, event: tk.Event) -> None:
        if not self.box_start:
            return
        start = self.box_start
        self.box_start = None
        if self.box_item:
            self.canvas.delete(self.box_item)
            self.box_item = None
        x1, y1 = self._canvas_to_image(*start)
        x2, y2 = self._canvas_to_image(event.x, event.y)
        box = (int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2)))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            self.select_at(event.x, event.y)
            return
        image = self.image.copy()

        def work() -> list[Prediction]:
            self.backend.set_image(image)
            return self.backend.predict_box(box)

        self._run_in_background("SAM3 正在按框补标…", work, self._add_box_prediction)

    def on_right_click(self, event: tk.Event) -> None:
        self.select_at(event.x, event.y)
        self.delete_selected()

    def select_at(self, canvas_x: float, canvas_y: float) -> None:
        x, y = self._canvas_to_image(canvas_x, canvas_y)
        ix, iy = int(x), int(y)
        self.selected = -1
        if self.image and 0 <= ix < self.image.width and 0 <= iy < self.image.height:
            for index in range(len(self.annotations) - 1, -1, -1):
                if self.annotations[index].mask[iy, ix]:
                    self.selected = index
                    break
        self._update_listbox()
        self.render()

    def on_list_select(self, _event: tk.Event) -> None:
        selection = self.listbox.curselection()
        if selection:
            self.selected = int(selection[0])
            self.render()

    def delete_selected(self) -> None:
        if 0 <= self.selected < len(self.annotations):
            self._remember()
            del self.annotations[self.selected]
            self.selected = min(self.selected, len(self.annotations) - 1)
            self.dirty = True
            self._refresh("已删除选中实例")

    def clear_annotations(self) -> None:
        if self.annotations and messagebox.askyesno("确认清空", "清空当前图片的全部实例？可用 Ctrl+Z 撤销。"):
            self._remember()
            self.annotations.clear()
            self.selected = -1
            self.dirty = True
            self._refresh("当前图片标注已清空")

    def _remember(self) -> None:
        self.history.append(list(self.annotations))
        self.history = self.history[-20:]

    def undo(self) -> None:
        if self.history:
            self.annotations = self.history.pop()
            self.selected = min(self.selected, len(self.annotations) - 1)
            self.dirty = True
            self._refresh("已撤销")

    def save(self) -> bool:
        if self.image is None or self.index < 0:
            return False
        assert self.label_dir is not None
        self.label_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.images[self.index]
        label_path = self._label_path(image_path)
        lines = masks_to_yolo_lines([item.mask for item in self.annotations])
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        meta = {
            "image": image_path.name,
            "class_names": {"0": "liquid_level_gauge"},
            "instances": [
                {"score": item.score, "source": item.source, "area": int(item.mask.sum())}
                for item in self.annotations
            ],
        }
        metadata_dir = self.label_dir.parent / "annotation_meta"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / f"{image_path.stem}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.dirty = False
        self.status_var.set(f"已保存 {len(self.annotations)} 个实例 → {label_path}")
        return True

    def _label_path(self, image_path: Path) -> Path:
        assert self.label_dir is not None
        return self.label_dir / f"{image_path.stem}.txt"

    def _refresh(self, status: str) -> None:
        self.status_var.set(status)
        self._update_listbox()
        self.render()

    def _update_listbox(self) -> None:
        self.listbox.delete(0, tk.END)
        for index, item in enumerate(self.annotations):
            score = "人工/已有" if item.score is None else f"置信度 {item.score:.3f}"
            self.listbox.insert(tk.END, f"实例 {index + 1}  |  {score}")
        if 0 <= self.selected < len(self.annotations):
            self.listbox.selection_set(self.selected)
            self.listbox.see(self.selected)

    def render(self) -> None:
        self.canvas.delete("image")
        if self.image is None:
            return
        canvas_w, canvas_h = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.display_scale = min(canvas_w / self.image.width, canvas_h / self.image.height)
        draw_w = max(1, round(self.image.width * self.display_scale))
        draw_h = max(1, round(self.image.height * self.display_scale))
        offset_x, offset_y = (canvas_w - draw_w) / 2, (canvas_h - draw_h) / 2
        self.display_offset = (offset_x, offset_y)

        base = np.asarray(self.image.resize((draw_w, draw_h), Image.Resampling.LANCZOS)).copy()
        for index, annotation in enumerate(self.annotations):
            mask = cv2.resize(annotation.mask.astype(np.uint8), (draw_w, draw_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            color = np.asarray(COLORS[index % len(COLORS)], dtype=np.uint8)
            alpha = 0.58 if index == self.selected else 0.35
            base[mask] = ((1 - alpha) * base[mask] + alpha * color).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(base, contours, -1, tuple(int(v) for v in color), 3 if index == self.selected else 1)
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(base))
        self.canvas.create_image(offset_x, offset_y, image=self.tk_image, anchor=tk.NW, tags="image")
        self.canvas.tag_lower("image")

    def _canvas_to_image(self, x: float, y: float) -> tuple[float, float]:
        ox, oy = self.display_offset
        return (x - ox) / self.display_scale, (y - oy) / self.display_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 liquid-level-gauge segmentation annotator")
    parser.add_argument("--images", type=Path, help="directory containing images")
    parser.add_argument("--labels", type=Path, help="YOLO segmentation label output directory")
    parser.add_argument("--prompt", default="liquid level gauge", help="English SAM3 concept prompt")
    parser.add_argument("--threshold", type=float, default=0.45, help="text prediction score threshold")
    parser.add_argument("--device", default="cuda", help="SAM3 device, e.g. cuda or cpu")
    parser.add_argument("--checkpoint", type=Path, help="optional local sam3.pt checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = LevelGaugeAnnotator(args)
    app.mainloop()


if __name__ == "__main__":
    main()
