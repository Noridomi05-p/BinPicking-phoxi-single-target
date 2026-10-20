from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .image_utils import resize_mask_nearest
from .types import InstanceDetection


class DetectorError(RuntimeError):
    pass


class YOLOInstanceSegmenter:
    def __init__(self, weights: str | Path, settings: dict[str, Any]) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError(
                "Khong import duoc ultralytics. Hay kich hoat venv va chay: pip install -e ."
            ) from exc

        self.weights = str(weights)
        weight_path = Path(self.weights)
        is_local_path = weight_path.is_absolute() or len(weight_path.parts) > 1
        if is_local_path and not weight_path.exists():
            raise DetectorError(f"Khong tim thay model: {self.weights}")
        self.model = YOLO(self.weights)
        self.settings = settings

    def predict(self, image_bgr: np.ndarray) -> list[InstanceDetection]:
        height, width = image_bgr.shape[:2]
        predict_kwargs: dict[str, Any] = {
            "source": image_bgr,
            "conf": float(self.settings.get("confidence", 0.35)),
            "iou": float(self.settings.get("iou", 0.60)),
            "imgsz": int(self.settings.get("image_size", 1024)),
            "retina_masks": True,
            "verbose": False,
        }
        device = self.settings.get("device")
        if device is not None and str(device).strip():
            predict_kwargs["device"] = device
        results = self.model.predict(**predict_kwargs)
        if not results:
            return []
        result = results[0]
        if result.boxes is None or result.masks is None:
            return []

        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        scores = result.boxes.conf.detach().cpu().numpy().astype(float)
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        masks = result.masks.data.detach().cpu().numpy()
        names = result.names

        detections: list[InstanceDetection] = []
        count = min(len(classes), len(scores), len(boxes), len(masks))
        for index in range(count):
            class_id = int(classes[index])
            if isinstance(names, dict):
                class_name = str(names.get(class_id, class_id))
            else:
                class_name = str(names[class_id])
            x1, y1, x2, y2 = boxes[index]
            bbox = (
                int(np.clip(round(x1), 0, width - 1)),
                int(np.clip(round(y1), 0, height - 1)),
                int(np.clip(round(x2), 0, width - 1)),
                int(np.clip(round(y2), 0, height - 1)),
            )
            mask = resize_mask_nearest(masks[index] > 0.5, height, width)
            detections.append(
                InstanceDetection(
                    instance_id=index,
                    class_id=class_id,
                    class_name=class_name,
                    score=float(scores[index]),
                    bbox_xyxy=bbox,
                    mask=mask,
                )
            )
        return detections
