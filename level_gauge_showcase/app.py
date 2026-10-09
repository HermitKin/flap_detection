from __future__ import annotations

import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from config import IMAGE_EXTENSIONS, OBJECT_CONFIDENCE, POSE_CONFIDENCE
from services import (
    DetectionRecord,
    ObjectDetectionService,
    PoseDetectionService,
    ReadingRecord,
    ReadingService,
    load_boxes_from_json,
    load_quads_from_json,
    read_image,
)


IMAGE_FILE_TYPES = [
    ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
    ("所有文件", "*.*"),
]

MODEL_FILE_TYPES = [
    ("模型文件", "*.pt *.onnx *.engine *.torchscript *.yaml *.yml"),
    ("PyTorch 权重", "*.pt"),
    ("所有文件", "*.*"),
]


class ImagePanel(ttk.LabelFrame):
    """A resize-aware image panel that keeps the source image unchanged."""

    def __init__(self, parent: tk.Widget, title: str) -> None:
        super().__init__(parent, text=title, padding=6)
        self._image = None
        self._photo = None
        self.label = ttk.Label(self, text="暂无图片", anchor="center")
        self.label.pack(fill="both", expand=True)
        self.label.bind("<Configure>", self._redraw)

    def show(self, image) -> None:
        self._image = None if image is None else image.copy()
        self._redraw()

    def clear(self) -> None:
        self._image = None
        self._photo = None
        self.label.configure(image="", text="暂无图片")

    def _redraw(self, _event=None) -> None:
        if self._image is None:
            return
        width = max(self.label.winfo_width() - 8, 120)
        height = max(self.label.winfo_height() - 8, 120)
        rgb = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((width, height), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)
        self.label.configure(image=self._photo, text="")


class DetectionTab(ttk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        app: "LevelGaugeShowcaseApp",
        mode: str,
    ) -> None:
        super().__init__(parent, padding=10)
        self.app = app
        self.mode = mode
        self.service = app.object_service if mode == "object" else app.pose_service
        self.default_model_path = Path(self.service.model_path).resolve()
        self.default_confidence = OBJECT_CONFIDENCE if mode == "object" else POSE_CONFIDENCE
        self.records: list[DetectionRecord] = []
        self.selected_paths: list[Path] = []
        self.index = -1
        self.busy = False

        self.confidence_var = tk.StringVar(value=f"{self.default_confidence:.2f}")
        self.status_var = tk.StringVar(value="请选择一张或多张图片。")
        self.counter_var = tk.StringVar(value="0 / 0")
        self.detail_var = tk.StringVar(value="检测结果：暂无")
        self.model_path_var = tk.StringVar(value=str(self.default_model_path))

        self._build()

    @property
    def title(self) -> str:
        return "液柱计检测" if self.mode == "object" else "关键点检测"

    def _build(self) -> None:
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Button(toolbar, text="选择单张图片", command=self.choose_one).pack(side="left", padx=3)
        ttk.Button(toolbar, text="选择多张图片", command=self.choose_many).pack(side="left", padx=3)
        if self.mode == "pose":
            ttk.Button(
                toolbar,
                text="读取第1页 ROI 结果",
                command=self.choose_object_rois,
            ).pack(side="left", padx=(12, 3))
        ttk.Button(toolbar, text="开始检测", command=self.start_detection).pack(side="left", padx=(12, 3))
        ttk.Label(toolbar, text="置信度：").pack(side="left", padx=(14, 2))
        ttk.Entry(toolbar, textvariable=self.confidence_var, width=7).pack(side="left")
        ttk.Button(toolbar, text="上一张", command=lambda: self.move(-1)).pack(side="right", padx=3)
        ttk.Button(toolbar, text="下一张", command=lambda: self.move(1)).pack(side="right", padx=3)
        ttk.Label(toolbar, textvariable=self.counter_var).pack(side="right", padx=10)

        model_bar = ttk.Frame(self)
        model_bar.pack(fill="x", pady=(0, 8))
        ttk.Label(model_bar, text=f"当前{self.title}模型：").pack(
            side="left", padx=(3, 4)
        )
        ttk.Entry(
            model_bar,
            textvariable=self.model_path_var,
            state="readonly",
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(model_bar, text="选择模型", command=self.choose_model).pack(
            side="left", padx=3
        )
        ttk.Button(
            model_bar,
            text="恢复默认模型",
            command=self.restore_default_model,
        ).pack(side="left", padx=3)

        viewer = ttk.Panedwindow(self, orient="horizontal")
        viewer.pack(fill="both", expand=True)
        self.original_panel = ImagePanel(viewer, "原始图片")
        self.result_panel = ImagePanel(viewer, "检测结果")
        viewer.add(self.original_panel, weight=1)
        viewer.add(self.result_panel, weight=1)

        info = ttk.Frame(self)
        info.pack(fill="x", pady=(8, 4))
        ttk.Label(info, textvariable=self.status_var).pack(side="left")
        ttk.Label(info, textvariable=self.detail_var).pack(side="right")

        actions = ttk.Frame(self)
        actions.pack(fill="x", pady=(4, 0))
        ttk.Label(actions, text="当前图片操作：").pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="查看检测结果", command=self.show_details).pack(side="left", padx=3)
        ttk.Button(actions, text="ROI 裁剪", command=self.crop_current).pack(side="left", padx=3)
        ttk.Button(actions, text="保存全部检测数据", command=self.save_all).pack(side="right", padx=3)
        ttk.Button(actions, text="保存当前 ROI", command=self.save_crops).pack(side="right", padx=3)

    def choose_model(self) -> None:
        if self.busy:
            messagebox.showwarning(
                "正在检测", "当前检测尚未结束，不能切换模型。", parent=self
            )
            return
        current = Path(self.service.model_path)
        selected_path = filedialog.askopenfilename(
            title=f"选择{self.title}模型",
            initialdir=str(current.parent if current.parent.exists() else Path.cwd()),
            filetypes=MODEL_FILE_TYPES,
        )
        if not selected_path:
            return
        try:
            selected = self.service.set_model_path(Path(selected_path))
        except Exception as exc:
            messagebox.showerror("模型切换失败", str(exc), parent=self)
            return
        self.model_path_var.set(str(selected))
        self.status_var.set(
            f"已切换{self.title}模型：{selected.name}。下一次检测开始生效。"
        )

    def restore_default_model(self) -> None:
        if self.busy:
            messagebox.showwarning(
                "正在检测", "当前检测尚未结束，不能切换模型。", parent=self
            )
            return
        try:
            selected = self.service.set_model_path(self.default_model_path)
        except Exception as exc:
            messagebox.showerror("恢复默认模型失败", str(exc), parent=self)
            return
        self.model_path_var.set(str(selected))
        self.status_var.set(f"已恢复默认{self.title}模型：{selected.name}。")

    def choose_one(self) -> None:
        path = filedialog.askopenfilename(title="选择待检测图片", filetypes=IMAGE_FILE_TYPES)
        if path:
            self._set_selected([Path(path)])

    def choose_many(self) -> None:
        paths = filedialog.askopenfilenames(title="选择一张或多张待检测图片", filetypes=IMAGE_FILE_TYPES)
        if paths:
            self._set_selected([Path(path) for path in paths])

    def choose_object_rois(self) -> None:
        object_records = self.app.object_tab.records
        if not object_records:
            messagebox.showwarning(
                "第1页没有数据", "请先在第1页完成液柱计检测。", parent=self
            )
            return

        roi_paths: list[Path] = []
        try:
            for record in object_records:
                # Direct page-to-page transfer: use page 1's in-memory records
                # and their saved ROI paths instead of opening another dialog.
                if not record.crops:
                    self.app.object_service.crop_record(record)
                if record.crops and not record.saved_crops:
                    self.app.object_service.save_crops(record)
                roi_paths.extend(path for path in record.saved_crops if path.exists())
        except Exception as exc:
            messagebox.showerror("读取第1页 ROI 失败", str(exc), parent=self)
            return

        if not roi_paths:
            messagebox.showwarning(
                "没有可用 ROI", "第1页检测记录中没有可裁剪的目标。", parent=self
            )
            return

        self._set_selected(roi_paths)
        self.status_var.set(
            f"已直接读取第1页 {len(object_records)} 张检测记录中的 "
            f"{len(roi_paths)} 个 ROI，点击“开始检测”。"
        )

    def _set_selected(self, paths: list[Path]) -> None:
        self.selected_paths = [path for path in paths if path.suffix.lower() in IMAGE_EXTENSIONS]
        self.records = []
        self.index = -1
        self.counter_var.set(f"0 / {len(self.selected_paths)}")
        self.status_var.set(f"已选择 {len(self.selected_paths)} 张图片，点击“开始检测”。")
        self.detail_var.set("检测结果：暂无")
        if self.selected_paths:
            try:
                self.original_panel.show(read_image(self.selected_paths[0]))
                self.result_panel.clear()
            except Exception as exc:
                messagebox.showerror("读取失败", str(exc), parent=self)

    def start_detection(self) -> None:
        if self.busy:
            return
        if not self.selected_paths:
            messagebox.showwarning("尚未选择图片", "请先选择一张或多张图片。", parent=self)
            return
        try:
            confidence = float(self.confidence_var.get())
            if not 0.0 < confidence <= 1.0:
                raise ValueError
        except ValueError:
            messagebox.showerror("参数错误", "置信度必须是 0 到 1 之间的数字。", parent=self)
            return

        self.busy = True
        self.records = []
        self.index = -1
        self.status_var.set(
            f"正在使用 {Path(self.service.model_path).name} 加载模型并开始检测……"
        )
        paths = list(self.selected_paths)
        threading.Thread(
            target=self._detect_worker,
            args=(paths, confidence),
            daemon=True,
        ).start()

    def _detect_worker(self, paths: list[Path], confidence: float) -> None:
        try:
            total = len(paths)
            for number, path in enumerate(paths, start=1):
                record = self.service.detect([path], confidence=confidence)[0]
                self.after(0, self._accept_record, record, number, total)
            self.after(0, self._finish_detection, None)
        except Exception as exc:
            details = f"{exc}\n\n{traceback.format_exc()}"
            self.after(0, self._finish_detection, details)

    def _accept_record(self, record: DetectionRecord, number: int, total: int) -> None:
        # Both detection pages persist every ROI immediately after inference.
        # Cropping always uses record.original, never the resized GUI preview.
        try:
            # Pose records have already been saved inside PoseDetectionService;
            # object records are saved here as soon as their inference finishes.
            if self.mode == "object":
                self.service.save_record(record)
            crops = self.service.crop_record(record)
            if crops:
                self.service.save_crops(record)
        except Exception as exc:
            self.status_var.set(f"ROI 自动保存失败：{exc}")
        self.records.append(record)
        self.index = len(self.records) - 1
        self.status_var.set(f"正在检测 {number}/{total}：{record.source_path.name}")
        self.show_current()

    def _finish_detection(self, error: str | None) -> None:
        self.busy = False
        if error:
            self.status_var.set("检测中止。")
            messagebox.showerror("检测失败", error, parent=self)
            return
        if self.mode == "pose":
            roi_count = sum(len(record.saved_crops) for record in self.records)
            auto_text = f"；关键点图、JSON 和 {roi_count} 个 ROI 已自动保存"
        else:
            roi_count = sum(len(record.saved_crops) for record in self.records)
            auto_text = f"；检测图、JSON 和 {roi_count} 个 ROI 已自动保存"
        self.status_var.set(f"检测完成，共 {len(self.records)} 张{auto_text}。")

    def current_record(self) -> DetectionRecord | None:
        if 0 <= self.index < len(self.records):
            return self.records[self.index]
        return None

    def show_current(self) -> None:
        record = self.current_record()
        if record is None:
            return
        self.original_panel.show(record.original)
        self.result_panel.show(record.annotated)
        self.counter_var.set(f"{self.index + 1} / {len(self.records)}")
        noun = "目标" if self.mode == "object" else "关键点目标"
        self.detail_var.set(f"{record.source_path.name}；检测到 {len(record.detections)} 个{noun}")

    def show_details(self) -> None:
        record = self.current_record()
        if record is None:
            messagebox.showwarning("没有结果", "请先完成检测。", parent=self)
            return

        lines = [
            f"图片：{record.source_path}",
            f"使用模型：{record.model_path or self.service.model_path}",
            f"原始尺寸：{record.original.shape[1]} × {record.original.shape[0]}",
            f"检测目标数量：{len(record.detections)}",
            "",
        ]
        if not record.detections:
            lines.append("未检测到目标。")
        for index, detection in enumerate(record.detections, start=1):
            box = detection.get("box_xyxy", detection.get("box", []))
            confidence = detection.get("confidence", detection.get("box_confidence"))
            lines.extend([
                f"目标 {index}",
                f"  类别：{detection.get('class_name', detection.get('class_id', '-'))}",
                f"  检测框位置 [x1, y1, x2, y2]：{self._format_values(box)}",
                f"  检测框置信度：{float(confidence):.6f}" if confidence is not None
                else "  检测框置信度：无",
            ])
            points = detection.get("keypoints_xy_pixels", detection.get("keypoints"))
            if points:
                names = detection.get("keypoint_order") or [
                    f"K{point_index}" for point_index in range(len(points))
                ]
                point_confidences = detection.get("keypoint_confidences") or []
                lines.append("  关键点：")
                for point_index, point in enumerate(points):
                    point_name = names[point_index] if point_index < len(names) else f"K{point_index}"
                    point_confidence = (
                        point_confidences[point_index]
                        if point_index < len(point_confidences) else None
                    )
                    confidence_text = (
                        f"，置信度={float(point_confidence):.6f}"
                        if point_confidence is not None else ""
                    )
                    lines.append(
                        f"    {point_name}: ({float(point[0]):.2f}, "
                        f"{float(point[1]):.2f}){confidence_text}"
                    )
            lines.append("")

        window = tk.Toplevel(self)
        window.title(f"{self.title}详情 - {record.source_path.name}")
        window.geometry("1100x720")
        window.minsize(800, 520)
        window.transient(self.winfo_toplevel())

        content = ttk.Panedwindow(window, orient="horizontal")
        content.pack(fill="both", expand=True, padx=10, pady=10)
        result_panel = ImagePanel(content, "检测结果图")
        result_panel.show(record.annotated)
        content.add(result_panel, weight=3)

        detail_frame = ttk.LabelFrame(content, text="检测数据", padding=6)
        text = tk.Text(detail_frame, wrap="word", font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(detail_frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")
        content.add(detail_frame, weight=2)

    @staticmethod
    def _format_values(values) -> str:
        if not values:
            return "无"
        return "[" + ", ".join(f"{float(value):.2f}" for value in values) + "]"

    def move(self, delta: int) -> None:
        if not self.records:
            return
        self.index = (self.index + delta) % len(self.records)
        self.show_current()

    def crop_current(self) -> None:
        record = self.current_record()
        if record is None:
            messagebox.showwarning("没有结果", "请先完成检测。", parent=self)
            return
        crops = self.service.crop_record(record)
        if not crops:
            messagebox.showwarning("无法裁剪", "当前图片没有检测目标。", parent=self)
            return
        self.result_panel.show(crops[0])
        self.detail_var.set(f"已裁剪 {len(crops)} 个 ROI；当前展示第 1 个")

    def save_all(self) -> None:
        if not self.records:
            messagebox.showwarning("没有结果", "请先完成检测。", parent=self)
            return
        try:
            saved = [self.service.save_record(record) for record in self.records]
            crop_count = 0
            for record in self.records:
                if not record.crops:
                    self.service.crop_record(record)
                if record.crops:
                    crop_count += len(self.service.save_crops(record))
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc), parent=self)
            return
        folder = saved[0][0].parent.parent
        self.status_var.set(
            f"已保存 {len(saved)} 张检测图、数据和 {crop_count} 个 ROI：{folder}"
        )
        messagebox.showinfo(
            "保存完成",
            f"已保存 {len(saved)} 组检测结果和 {crop_count} 个 ROI。\n{folder}",
            parent=self,
        )

    def save_crops(self) -> None:
        record = self.current_record()
        if record is None:
            messagebox.showwarning("没有结果", "请先完成检测。", parent=self)
            return
        try:
            paths = self.service.save_crops(record)
        except Exception as exc:
            messagebox.showwarning("无法保存", str(exc), parent=self)
            return
        self.status_var.set(f"已保存 {len(paths)} 个 ROI：{paths[0].parent}")
        messagebox.showinfo("保存完成", f"已保存 {len(paths)} 个 ROI。\n{paths[0].parent}", parent=self)


class ReadingTab(ttk.Frame):
    def __init__(self, parent: tk.Widget, app: "LevelGaugeShowcaseApp") -> None:
        super().__init__(parent, padding=10)
        self.app = app
        self.image_path: Path | None = None
        self.existing_boxes: list[list[float]] = []
        self.existing_quads: list[list[list[float]]] = []
        self.record: ReadingRecord | None = None
        self.busy = False

        self.mode_var = tk.StringVar(value="model")
        self.scale_min_var = tk.StringVar(value="0")
        self.scale_max_var = tk.StringVar(value="100")
        self.source_var = tk.StringVar(value="尚未选择图片")
        self.status_var = tk.StringVar(value="选择图片或从前两个页面载入当前结果。")
        self.result_var = tk.StringVar(value="当前读数：暂无")
        self._build()

    def _build(self) -> None:
        source_bar = ttk.Frame(self)
        source_bar.pack(fill="x", pady=(0, 6))
        ttk.Button(source_bar, text="选择图片", command=self.choose_image).pack(side="left", padx=3)
        ttk.Button(source_bar, text="使用目标检测当前图片/数据", command=self.use_object_current).pack(side="left", padx=3)
        ttk.Button(source_bar, text="使用关键点检测当前图片/数据", command=self.use_pose_current).pack(side="left", padx=3)
        ttk.Button(source_bar, text="读取检测 JSON", command=self.choose_json).pack(side="left", padx=3)
        ttk.Label(source_bar, textvariable=self.source_var).pack(side="right", padx=5)

        settings = ttk.LabelFrame(self, text="读数设置", padding=7)
        settings.pack(fill="x", pady=(0, 8))
        ttk.Radiobutton(
            settings, text="重新调用关键点模型并透视校正", variable=self.mode_var, value="model"
        ).pack(side="left", padx=5)
        ttk.Radiobutton(
            settings, text="使用已有检测数据（无数据时整图作为 ROI）",
            variable=self.mode_var, value="existing"
        ).pack(side="left", padx=5)
        ttk.Label(settings, text="量程下限：").pack(side="left", padx=(18, 2))
        ttk.Entry(settings, textvariable=self.scale_min_var, width=8).pack(side="left")
        ttk.Label(settings, text="量程上限：").pack(side="left", padx=(8, 2))
        ttk.Entry(settings, textvariable=self.scale_max_var, width=8).pack(side="left")
        ttk.Button(settings, text="开始读数", command=self.start_reading).pack(side="right", padx=5)

        viewer = ttk.Panedwindow(self, orient="horizontal")
        viewer.pack(fill="both", expand=True)
        self.original_panel = ImagePanel(viewer, "无标注关键点裁切输入")
        self.unrectified_panel = ImagePanel(viewer, "未透视校正读数（对比）")
        self.rectified_panel = ImagePanel(viewer, "透视校正结果")
        self.result_panel = ImagePanel(viewer, "ReadROI 结果")
        viewer.add(self.original_panel, weight=1)
        viewer.add(self.unrectified_panel, weight=1)
        viewer.add(self.rectified_panel, weight=1)
        viewer.add(self.result_panel, weight=1)

        footer = ttk.Frame(self)
        footer.pack(fill="x", pady=(8, 0))
        ttk.Label(footer, textvariable=self.status_var).pack(side="left")
        ttk.Label(footer, textvariable=self.result_var).pack(side="left", padx=25)
        ttk.Button(footer, text="保存读数结果", command=self.save_result).pack(side="right")
        ttk.Button(
            footer, text="查看检测结果", command=self.show_hsv_result
        ).pack(side="right", padx=8)

    def _load_source(
        self,
        path: Path,
        boxes: list[list[float]],
        description: str,
        quads: list[list[list[float]]] | None = None,
    ) -> None:
        try:
            image = read_image(path)
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc), parent=self)
            return
        self.image_path = path.resolve()
        self.existing_boxes = boxes
        self.existing_quads = quads or []
        self.record = None
        self.original_panel.show(image)
        self.unrectified_panel.clear()
        self.rectified_panel.clear()
        self.result_panel.clear()
        self.source_var.set(path.name)
        self.status_var.set(description)
        self.result_var.set("当前读数：暂无")

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(title="选择待读数图片", filetypes=IMAGE_FILE_TYPES)
        if path:
            self._load_source(
                Path(path), [], "已载入无标注图片，可调用关键点模型后透视校正。", []
            )

    def _use_detection_record(
        self,
        record: DetectionRecord | None,
        source_name: str,
        service,
    ) -> None:
        if record is None:
            messagebox.showwarning("没有当前结果", f"{source_name}页面还没有检测结果。", parent=self)
            return
        boxes = []
        quads = []
        for detection in record.detections:
            box = detection.get("box_xyxy", detection.get("box"))
            if box:
                boxes.append([float(value) for value in box])
            quad = detection.get("roi_quad_tl_tr_br_bl")
            if quad:
                quads.append([
                    [float(point[0]), float(point[1])] for point in quad
                ])

        if quads:
            # Page 3 receives page 2's clean detection-box crop, not page 2's
            # original input and not its annotated display image. Coordinates
            # must therefore be translated into this crop's local system.
            try:
                if not record.crops:
                    service.crop_record(record)
                if record.crops and (
                    not record.saved_crops
                    or any(not path.exists() for path in record.saved_crops)
                ):
                    service.save_crops(record)
            except Exception as exc:
                messagebox.showerror("读取第二页裁剪结果失败", str(exc), parent=self)
                return

            valid_targets = []
            image_height, image_width = record.original.shape[:2]
            for detection in record.detections:
                box = detection.get("box_xyxy", detection.get("box"))
                quad = detection.get("roi_quad_tl_tr_br_bl")
                if not box or not quad:
                    continue
                x1, y1, x2, y2 = (int(round(float(value))) for value in box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image_width, x2), min(image_height, y2)
                if x2 > x1 and y2 > y1:
                    valid_targets.append((detection, [x1, y1, x2, y2], quad))

            available_count = min(len(valid_targets), len(record.saved_crops))
            if available_count == 0:
                messagebox.showwarning(
                    "没有第二页裁剪结果", "关键点检测目标没有生成有效 ROI。", parent=self
                )
                return

            detection, crop_box, source_quad = valid_targets[0]
            crop_path = record.saved_crops[0]
            crop_image = record.crops[0]
            x1, y1, _, _ = crop_box
            local_quad = [[
                float(point[0]) - x1,
                float(point[1]) - y1,
            ] for point in source_quad]
            crop_height, crop_width = crop_image.shape[:2]
            self.mode_var.set("existing")
            self._load_source(
                crop_path,
                [[0.0, 0.0, float(crop_width), float(crop_height)]],
                f"已直接载入第二页目标 #{detection.get('detection_id', 1)} 的无标注裁剪 ROI，"
                "关键点已转换为裁剪图局部坐标；读数时将先透视校正。",
                [local_quad],
            )
            return

        # Object-detection records have no perspective quad. Keep the processed
        # result path as a compatibility fallback for the first-page shortcut.
        if record.saved_image is None or not record.saved_image.exists():
            service.save_record(record)
        processed_path = record.saved_image
        if processed_path is None:
            messagebox.showerror("结果不存在", "无法取得右侧处理结果图片。", parent=self)
            return
        self.mode_var.set("existing")
        self._load_source(
            processed_path,
            boxes,
            f"已载入{source_name}右侧处理图及数据，共 {len(boxes)} 个 ROI。",
            [],
        )

    def use_object_current(self) -> None:
        self._use_detection_record(
            self.app.object_tab.current_record(), "目标检测", self.app.object_service
        )

    def use_pose_current(self) -> None:
        self._use_detection_record(
            self.app.pose_tab.current_record(), "关键点检测", self.app.pose_service
        )

    def choose_json(self) -> None:
        path = filedialog.askopenfilename(title="选择检测数据", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            boxes = load_boxes_from_json(Path(path))
            quads = load_quads_from_json(Path(path))
        except Exception as exc:
            messagebox.showerror("数据读取失败", str(exc), parent=self)
            return
        if not self.image_path:
            messagebox.showinfo("继续选择图片", "JSON 已读取，请再选择它对应的原始图片。", parent=self)
            image_path = filedialog.askopenfilename(title="选择 JSON 对应的原始图片", filetypes=IMAGE_FILE_TYPES)
            if not image_path:
                return
            self._load_source(
                Path(image_path), boxes,
                f"已载入 JSON：{len(boxes)} 个框、{len(quads)} 组关键点。",
                quads,
            )
        else:
            self.existing_boxes = boxes
            self.existing_quads = quads
            self.mode_var.set("existing")
            self.status_var.set(
                f"已载入 JSON：{len(boxes)} 个框、{len(quads)} 组关键点。"
            )

    def start_reading(self) -> None:
        if self.busy:
            return
        if self.image_path is None:
            messagebox.showwarning("尚未选择图片", "请先选择待读数图片。", parent=self)
            return
        try:
            scale_min = float(self.scale_min_var.get())
            scale_max = float(self.scale_max_var.get())
            if scale_max <= scale_min:
                raise ValueError("量程上限必须大于量程下限。")
        except ValueError as exc:
            messagebox.showerror("量程错误", str(exc) or "请输入有效数字。", parent=self)
            return
        self.busy = True
        self.status_var.set("正在根据关键点透视校正，并在校正图上定位翻片区域……")
        image_path = self.image_path
        call_model = self.mode_var.get() == "model"
        boxes = list(self.existing_boxes)
        quads = [
            [[float(value) for value in point] for point in quad]
            for quad in self.existing_quads
        ]
        threading.Thread(
            target=self._reading_worker,
            args=(image_path, scale_min, scale_max, call_model, boxes, quads),
            daemon=True,
        ).start()

    def _reading_worker(
        self,
        image_path: Path,
        scale_min: float,
        scale_max: float,
        call_model: bool,
        boxes: list[list[float]],
        quads: list[list[list[float]]],
    ) -> None:
        try:
            record = self.app.reading_service.read(
                image_path, scale_min, scale_max, call_model, boxes, quads
            )
            self.after(0, self._finish_reading, record, None)
        except Exception as exc:
            self.after(0, self._finish_reading, None, f"{exc}\n\n{traceback.format_exc()}")

    def _finish_reading(self, record: ReadingRecord | None, error: str | None) -> None:
        self.busy = False
        if error or record is None:
            self.status_var.set("读数失败。")
            messagebox.showerror("读数失败", error or "未知错误", parent=self)
            return
        self.record = record
        self.unrectified_panel.show(record.unrectified_visualization)
        self.rectified_panel.show(record.rectified_visualization)
        self.result_panel.show(record.annotated)
        values = "，".join(f"#{item['gauge_id']} {item['reading']:.1f}" for item in record.reports)
        raw_values = "，".join(
            f"#{item['gauge_id']} {item['reading']:.1f}"
            for item in record.unrectified_reports if "reading" in item
        )
        comparison = f"；未校正对比：{raw_values}" if raw_values else ""
        self.result_var.set(f"校正后读数：{values}{comparison}")
        try:
            image_path, json_path = self.app.reading_service.save_record(record)
        except Exception as exc:
            self.status_var.set(f"读数完成，但自动保存失败：{exc}")
            messagebox.showwarning("自动保存失败", str(exc), parent=self)
            return
        self.status_var.set(
            f"读数完成并已自动保存；方式：{record.source_mode}；目录：{image_path.parent.parent}"
        )

    def show_hsv_result(self) -> None:
        if self.record is None:
            messagebox.showwarning("没有结果", "请先完成 ReadROI 读数。", parent=self)
            return

        window = tk.Toplevel(self)
        window.title(f"HSV 颜色处理结果 - {self.record.source_path.name}")
        hsv_height, hsv_width = self.record.hsv_visualization.shape[:2]
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        popup_width = min(max(hsv_width + 100, 700), screen_width - 100)
        popup_height = min(max(hsv_height + 170, 620), screen_height - 100)
        window.geometry(f"{popup_width}x{popup_height}")
        window.minsize(620, 500)
        window.transient(self.winfo_toplevel())

        panel = ImagePanel(
            window,
            "底部连续有色示数段 HSV 结果（仅展示分界至底部）",
        )
        panel.pack(fill="both", expand=True, padx=10, pady=(10, 5))
        panel.show(self.record.hsv_visualization)

        details = []
        for report in self.record.reports:
            details.append(
                f"液位计 #{report['gauge_id']}：读数={report['reading']:.1f}，"
                f"校正图分界 y={report.get('boundary_y_rectified', report['boundary_y_image'])}，"
                f"翻片区域={report['flap_strip_xyxy']}，"
                f"置信度={report['continuity_confidence']:.3f}"
            )
        ttk.Label(
            window,
            text=(
                "HSV 图仅显示自动提取的有色示数段；关键点矩形总高度只用于量程换算。\n"
                + "\n".join(details)
            ),
            anchor="w",
            justify="left",
            padding=(12, 6, 12, 10),
        ).pack(fill="x")

    def save_result(self) -> None:
        if self.record is None:
            messagebox.showwarning("没有结果", "请先完成读数。", parent=self)
            return
        try:
            image_path, json_path = self.app.reading_service.save_record(self.record)
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc), parent=self)
            return
        self.status_var.set(f"读数结果已保存：{image_path.parent.parent}")
        messagebox.showinfo(
            "保存完成",
            f"第三页输入 ROI：{self.record.saved_input}\n"
            f"未校正对比图：{self.record.saved_unrectified}\n"
            f"透视校正图：{self.record.saved_rectified}\n"
            f"ReadROI 检测结果图：{image_path}\n"
            f"自适应翻片颜色带：{self.record.saved_flap_strip}\n"
            f"HSV处理图：{self.record.saved_hsv}\n数据：{json_path}",
            parent=self,
        )


class LevelGaugeShowcaseApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("磁翻板液位计智能检测与读数展示")
        self.geometry("1360x840")
        self.minsize(1060, 680)

        self.object_service = ObjectDetectionService()
        self.pose_service = PoseDetectionService()
        self.reading_service = ReadingService(self.object_service, self.pose_service)

        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("TNotebook.Tab", padding=(20, 9), font=("Microsoft YaHei UI", 10))

        header = ttk.Frame(self, padding=(12, 10, 12, 4))
        header.pack(fill="x")
        ttk.Label(
            header,
            text="磁翻板液位计检测、关键点定位与 ReadROI 读数",
            font=("Microsoft YaHei UI", 16, "bold"),
        ).pack(side="left")
        ttk.Label(
            header,
            text="前两页可切换模型；检测数据自动记录实际权重",
            foreground="#555555",
        ).pack(side="right")

        notebook = ttk.Notebook(self)
        self.notebook = notebook
        notebook.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.object_tab = DetectionTab(notebook, self, mode="object")
        self.pose_tab = DetectionTab(notebook, self, mode="pose")
        self.reading_tab = ReadingTab(notebook, self)
        notebook.add(self.object_tab, text="1  液柱计检测")
        notebook.add(self.pose_tab, text="2  关键点检测")
        notebook.add(self.reading_tab, text="3  ReadROI 读数")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="磁翻板液位计检测与读数展示")
    parser.add_argument("--demo", action="store_true", help="加载无权重的合成几何演示")
    args = parser.parse_args()
    app = LevelGaugeShowcaseApp()
    if args.demo:
        from public_demo import populate_demo

        populate_demo(app)
    app.mainloop()


if __name__ == "__main__":
    main()
