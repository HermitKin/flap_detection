"""Isolated YOLO inference helper used to avoid Windows OpenMP DLL conflicts."""

import json
import sys

from ultralytics import YOLO


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: detect_gauge_yolo.py MODEL IMAGE CONFIDENCE")
    model_path, image_path, confidence = sys.argv[1], sys.argv[2], float(sys.argv[3])
    prediction = YOLO(model_path).predict(image_path, conf=confidence, verbose=False)[0]
    boxes = [
        {"box": xyxy.tolist(), "confidence": float(conf)}
        for xyxy, conf in zip(prediction.boxes.xyxy.cpu().numpy(), prediction.boxes.conf.cpu().numpy())
    ]
    print("YOLO_RESULT_JSON=" + json.dumps(boxes, ensure_ascii=True))


if __name__ == "__main__":
    main()
