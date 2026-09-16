"""Thin, lazy-loaded adapter around Meta's official SAM 3 package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Prediction:
    mask: np.ndarray
    score: float
    box: tuple[float, float, float, float]
    source: str


class Sam3Backend:
    """Load SAM3 on first use and expose text/box prediction as NumPy arrays."""

    def __init__(self, device: str = "cuda", checkpoint: Path | None = None) -> None:
        self.device = device
        self.checkpoint = checkpoint
        self.processor: Any = None
        self._image: Image.Image | None = None
        self._image_state: dict[str, Any] | None = None

    @property
    def loaded(self) -> bool:
        return self.processor is not None

    def load(self) -> None:
        if self.loaded:
            return
        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise RuntimeError(
                "未安装 Meta 官方 SAM3。请先按 README 创建 sam3 环境并执行 pip install -e ."
            ) from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("选择了 CUDA，但 PyTorch 未检测到可用的 NVIDIA GPU。")
        kwargs: dict[str, Any] = {"device": self.device}
        if self.checkpoint:
            kwargs.update(checkpoint_path=str(self.checkpoint), load_from_HF=False)
        model = build_sam3_image_model(**kwargs)
        # Sam3Processor defaults to CUDA independently of the model, so pass the
        # requested device explicitly (important for --device cpu).
        self.processor = Sam3Processor(model, device=self.device)

    def set_image(self, image: Image.Image) -> None:
        self.load()
        self._image = image.convert("RGB")
        # Cache the expensive image backbone once. Prompt-specific state below is
        # shallow-copied so multiple boxes do not accumulate geometric prompts.
        self._image_state = self.processor.set_image(self._image)

    def predict_text(self, prompt: str, threshold: float) -> list[Prediction]:
        self._require_image()
        # The official processor filters candidates internally; set its threshold
        # before inference so GUI values below the default 0.5 actually take effect.
        self.processor.set_confidence_threshold(threshold)
        state = self._fresh_state()
        output = self.processor.set_text_prompt(state=state, prompt=prompt)
        return self._extract(output, threshold=threshold, source=f"text:{prompt}")

    def predict_box(self, box_xyxy: tuple[int, int, int, int]) -> list[Prediction]:
        """Segment the object inside an absolute-pixel user box."""
        self._require_image()
        width, height = self._image.size
        x1, y1, x2, y2 = box_xyxy
        normalized_cxcywh = [
            ((x1 + x2) / 2) / width,
            ((y1 + y2) / 2) / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        ]
        self.processor.set_confidence_threshold(0.0)
        state = self._fresh_state()
        output = self.processor.add_geometric_prompt(
            box=normalized_cxcywh, label=True, state=state
        )
        predictions = self._extract(output, threshold=0.0, source="box")
        if not predictions:
            return []
        return [max(predictions, key=lambda pred: _box_iou(pred.box, box_xyxy))]

    def _require_image(self) -> None:
        if self._image is None or self._image_state is None:
            raise RuntimeError("请先打开图片。")

    def _fresh_state(self) -> dict[str, Any]:
        """Reuse image tensors but isolate dictionaries mutated by the processor."""
        assert self._image_state is not None
        state = dict(self._image_state)
        state["backbone_out"] = dict(self._image_state["backbone_out"])
        return state

    @staticmethod
    def _extract(output: dict[str, Any], threshold: float, source: str) -> list[Prediction]:
        masks = _to_numpy(output["masks"])
        boxes = _to_numpy(output["boxes"])
        scores = _to_numpy(output["scores"]).reshape(-1)
        if masks.ndim == 4:
            masks = masks[:, 0]
        results: list[Prediction] = []
        for mask, box, score in zip(masks, boxes, scores):
            value = float(score)
            binary = mask > 0.5
            if value >= threshold and binary.any():
                results.append(
                    Prediction(
                        mask=binary,
                        score=value,
                        box=tuple(float(v) for v in box),
                        source=source,
                    )
                )
        return sorted(results, key=lambda item: item.score, reverse=True)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _box_iou(first: tuple[float, float, float, float], second: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0
