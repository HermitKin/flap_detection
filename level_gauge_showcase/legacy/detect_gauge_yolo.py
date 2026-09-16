"""Isolated YOLO inference helper used to avoid Windows OpenMP DLL conflicts."""

import json
import sys

from ultralytics import YOLO


def main() -> None:
    if len(sys.argv) not in (4, 5):
        raise SystemExit("usage: detect_gauge_yolo.py MODEL IMAGE CONFIDENCE [IMGSZ]")
    model_path, image_path, confidence = sys.argv[1], sys.argv[2], float(sys.argv[3])
    image_size = int(sys.argv[4]) if len(sys.argv) == 5 else 640
    prediction = YOLO(model_path).predict(
        image_path,
        conf=confidence,
        imgsz=image_size,
        iou=0.7,
        max_det=20,
        verbose=False,
    )[0]
    detections = []
    if prediction.boxes is not None:
        boxes = prediction.boxes.xyxy.cpu().numpy()
        confidences = prediction.boxes.conf.cpu().numpy()
        classes = prediction.boxes.cls.cpu().numpy().astype(int)
        for box, score, class_id in zip(boxes, confidences, classes):
            names = prediction.names
            if isinstance(names, dict):
                name = str(names.get(int(class_id), class_id))
            elif isinstance(names, (list, tuple)) and 0 <= int(class_id) < len(names):
                name = str(names[int(class_id)])
            else:
                name = str(class_id)
            detections.append({
                "box": box.astype(float).tolist(),
                "box_xyxy": box.astype(float).tolist(),
                "confidence": float(score),
                "class_id": int(class_id),
                "class_name": name,
            })
    print("YOLO_RESULT_JSON=" + json.dumps(detections, ensure_ascii=True))


if __name__ == "__main__":
    main()
