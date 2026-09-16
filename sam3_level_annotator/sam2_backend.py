"""CPU-friendly adapter around Ultralytics SAM2 for rectangle prompts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .sam3_backend import Prediction


class Sam2Backend:
    """Run the tiny SAM2 checkpoint using an image and one XYXY box prompt."""

    def __init__(self, model_path: str | Path = "sam2_t.pt", device: str = "cpu", imgsz: int = 640) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.imgsz = imgsz
        self.model = None
        self._image: np.ndarray | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            from ultralytics import SAM
        except ImportError as exc:
            raise RuntimeError("未安装 ultralytics，请执行：pip install ultralytics") from exc
        self.model = SAM(self.model_path)

    def set_image(self, image: Image.Image) -> None:
        self.load()
        rgb = np.asarray(image.convert("RGB"))
        self._image = np.ascontiguousarray(rgb[:, :, ::-1])

    def predict_box(self, box_xyxy: tuple[int, int, int, int]) -> list[Prediction]:
        if self._image is None:
            raise RuntimeError("请先打开图片。")
        result = self.model.predict(
            self._image,
            bboxes=[list(box_xyxy)],
            device=self.device,
            imgsz=self.imgsz,
            retina_masks=True,
            verbose=False,
        )[0]
        if result.masks is None or len(result.masks.data) == 0:
            return []
        masks = result.masks.data.detach().cpu().numpy()
        scores = np.ones(len(masks), dtype=np.float32)
        if result.boxes is not None and result.boxes.conf is not None and len(result.boxes.conf) == len(masks):
            scores = result.boxes.conf.detach().float().cpu().numpy()
        x1, y1, x2, y2 = box_xyxy
        return sorted(
            [
                Prediction(
                    mask=mask > 0.5,
                    score=float(score),
                    box=(float(x1), float(y1), float(x2), float(y2)),
                    source="sam2_box",
                )
                for mask, score in zip(masks, scores)
                if (mask > 0.5).any()
            ],
            key=lambda item: item.score,
            reverse=True,
        )
