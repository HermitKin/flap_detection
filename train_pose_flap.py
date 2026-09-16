import os


# Keep multi-GPU DDP workers on the same project Ultralytics source tree.
os.environ["PYTHONPATH"] = os.pathsep.join(
    path for path in ("/workspace", os.environ.get("PYTHONPATH")) if path
)

from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("/workspace/yolo11n-pose.pt")
    model.train(
        data="/workspace/datasets/testset_pose8/data.yaml",
        imgsz=960,
        epochs=200,
        device=[0, 1],
        batch=16,
        workers=4,
        optimizer="SGD",
        project="/workspace/runs/pose",
        name="flap_pose8_yolo11n",
        exist_ok=False,
        resume=False,
        plots=True,
    )
