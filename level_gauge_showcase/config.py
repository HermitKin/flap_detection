from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
MODEL_DIR = APP_ROOT / "models"
OUTPUT_DIR = APP_ROOT / "outputs"

OBJECT_MODEL = MODEL_DIR / "object_detector.pt"
POSE_MODEL = MODEL_DIR / "pose_keypoints.pt"

OBJECT_CONFIDENCE = 0.25
OBJECT_IMGSZ = 640
POSE_CONFIDENCE = 0.20
POSE_IMGSZ = 960

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
