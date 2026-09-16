import os


# Ultralytics launches fresh Python processes for multi-GPU DDP. Ensure those
# workers import the same project source tree as this entry point.
os.environ["PYTHONPATH"] = os.pathsep.join(
    path for path in ("/workspace", os.environ.get("PYTHONPATH")) if path
)

from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("/workspace/yolo11n.pt")
    model.train(
        data="/workspace/datasets/flap_data/data.yaml",
        imgsz=640,
        epochs=300,
        # Two-GPU DistributedDataParallel training. The batch size is global,
        # so each RTX A6000 processes 32 images per step.
        device=[0, 1],
        batch=64,
        workers=8,
        optimizer="SGD",
        project="/workspace/runs/train",
        name="flap_yolo11n",
        exist_ok=False,
        resume=False,
    )
