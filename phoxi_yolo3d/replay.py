"""Portable frozen-mask replay: no GPU, YOLO inference, or camera required."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from .storage import load_capture
from .types import InstanceDetection


class FrozenSegmenter:
    def __init__(self, detections):
        self.detections = detections

    def predict(self, _image):
        return self.detections


def load_frozen_inspection(directory):
    root = Path(directory).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, digest in manifest.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"INSPECTION_FILE_MISSING: {name}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"INSPECTION_HASH_MISMATCH: {name}")
    frame = load_capture(root / "scans" / "frame.npz")
    record = json.loads((root / "result.json").read_text(encoding="utf-8"))
    for key in ("frame_id", "timestamp_utc"):
        if record.get(key) is not None and record[key] != getattr(frame, key):
            raise ValueError(f"INSPECTION_FRAME_MISMATCH: {key}")
    detections, seen = [], set()
    with np.load(root / "masks.npz", allow_pickle=False) as masks:
        for obj in record.get("objects", []) + record.get("rejected", []):
            identifier = int(obj["instance_id"])
            if identifier in seen:
                raise ValueError("DUPLICATE_INSTANCE_ID")
            seen.add(identifier)
            mask = np.array(masks[f"instance_{identifier}"], dtype=bool)
            if mask.shape != frame.texture_bgr.shape[:2]:
                raise ValueError("MASK_FRAME_SHAPE_MISMATCH")
            detections.append(InstanceDetection(identifier, int(obj["class_id"]), str(obj["class_name"]),
                float(obj["confidence"]), tuple(obj["bbox_xyxy"]), mask))
    return frame, FrozenSegmenter(detections)
