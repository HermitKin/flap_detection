"""Train a YOLO11 instance-segmentation model on a custom dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11 instance segmentation")
    parser.add_argument("--data", type=Path, default=Path("dataset-seg/data.yaml"))
    parser.add_argument("--model", default="yolo11m-seg.pt", help="Use a -seg checkpoint, not a detection checkpoint")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="For example 0, 0,1 or cpu")
    parser.add_argument("--project", default="runs/segment")
    parser.add_argument("--name", default="yolo11m-seg")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--resume", type=Path, default=None, help="Path to a segmentation last.pt checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Segmentation dataset YAML not found: {data_path}\n"
            "Run prepare_coco_seg_dataset.py first."
        )

    # Import only when training is ready to start so --help and path validation
    # work even if the currently active Python environment is not the CUDA environment.
    from ultralytics import YOLO

    if args.resume:
        resume_path = args.resume.resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        model = YOLO(str(resume_path))
        model.train(resume=True)
        return

    if "-seg" not in Path(args.model).stem:
        raise ValueError("--model must be an instance-segmentation model such as yolo11m-seg.pt")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        cache=args.cache,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        close_mosaic=10,
        workers=args.workers,
        device=args.device,
        optimizer="SGD",
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
