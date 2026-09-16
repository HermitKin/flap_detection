"""Validate a trained YOLO instance-segmentation checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO11 instance segmentation")
    parser.add_argument("--model", type=Path, default=Path("runs/segment/yolo11m-seg/weights/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("dataset-seg/data.yaml"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/segment-val")
    parser.add_argument("--name", default="yolo11m-seg-val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    data_path = args.data.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Segmentation dataset YAML not found: {data_path}")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if model.task != "segment":
        raise ValueError(f"Expected a segmentation checkpoint, but model task is '{model.task}'")

    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    print("\nValidation summary")
    print(f"Box  mAP50:    {metrics.box.map50:.4f}")
    print(f"Box  mAP50-95: {metrics.box.map:.4f}")
    print(f"Mask mAP50:    {metrics.seg.map50:.4f}")
    print(f"Mask mAP50-95: {metrics.seg.map:.4f}")


if __name__ == "__main__":
    main()
