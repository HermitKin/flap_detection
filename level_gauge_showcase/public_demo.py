"""Load public synthetic examples into the actual three-tab desktop application."""

from pathlib import Path
from tkinter import ttk

import cv2

from services import DetectionRecord
from synthetic_demo import run_example


def populate_demo(app) -> None:
    import services

    workdir = Path(__file__).resolve().parents[1] / ".demo-work"
    services.OUTPUT_DIR = workdir / "outputs"
    app.title("Flap Detection | 合成几何演示（无 YOLO 推理）")
    banner = ttk.Label(app, text="合成案例 · 框/四角来自已知几何 · 透视校正与 HSV 读数真实计算",
                       foreground="#14745c", font=("Microsoft YaHei UI", 10, "bold"))
    banner.pack(before=app.notebook, fill="x", padx=14, pady=3)

    for level, colour in [(0.35, "red"), (0.62, "red"), (0.78, "blue")]:
        example = run_example(workdir, level, colour)
        box = example["box"]
        object_view = example["scene"].copy()
        x1, y1, x2, y2 = [round(v) for v in box]
        cv2.rectangle(object_view, (x1, y1), (x2, y2), (255, 220, 65), 2)
        detection = {"box_xyxy": box, "class_name": "synthetic gauge",
                     "provenance": "known_synthetic_geometry"}
        for tab, image, points, task in [
            (app.object_tab, object_view, [detection], "object_detection"),
            (app.pose_tab, example["annotated"], [{**detection,
                "keypoints": example["quad"], "keypoint_order": ["TL", "TR", "BR", "BL"],
                "roi_quad_tl_tr_br_bl": example["quad"]}], "keypoint_detection"),
        ]:
            record = DetectionRecord(example["source"], example["scene"], image, points, task)
            tab.records.append(record)
            tab.service.crop_record(record)
            tab.service.save_crops(record)

    for tab in (app.object_tab, app.pose_tab):
        tab.index = 1
        tab.show_current()
        tab.model_path_var.set("合成已知几何（没有模型置信度，不调用 YOLO）")
        tab.status_var.set("已载入 3 个合成案例；可用上一张/下一张浏览。")
        # Keep actual-model and file-selection controls out of the synthetic workflow.
        for widget in descendants(tab):
            if isinstance(widget, ttk.Button) and widget.cget("text") in {
                "选择单张图片", "选择多张图片", "开始检测", "选择模型", "恢复默认模型",
                "读取第1页 ROI 结果", "查看检测结果", "保存检测结果",
            }:
                widget.state(["disabled"])

    app.reading_tab.use_pose_current()
    for widget in descendants(app.reading_tab):
        if isinstance(widget, ttk.Radiobutton) and widget.cget("value") == "model":
            widget.state(["disabled"])
    app.reading_tab.start_reading()
    def refresh_status():
        if app.reading_tab.busy:
            app.after(150, refresh_status)
        elif app.reading_tab.record:
            app.reading_tab.status_var.set("合成案例读数完成；透视校正与颜色分析由项目算法真实计算。")
    app.after(150, refresh_status)


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)
