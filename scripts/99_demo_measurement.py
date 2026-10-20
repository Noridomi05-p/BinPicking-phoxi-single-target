#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from phoxi_yolo3d.config import load_config, project_path
from phoxi_yolo3d.pipeline import PhoXiYoloPipeline
from phoxi_yolo3d.storage import save_bgr_png, save_capture
from phoxi_yolo3d.types import Frame3D, InstanceDetection


class KnownMaskSegmenter:
    def __init__(self, detections):
        self.detections = detections

    def predict(self, image_bgr):
        del image_bgr
        return self.detections


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def main() -> int:
    config = load_config()
    height, width = 480, 640
    yy, xx = np.mgrid[:height, :width]
    mask_a = ((xx - 220) / 105) ** 2 + ((yy - 225) / 75) ** 2 <= 1
    mask_b = ((xx - 420) / 85) ** 2 + ((yy - 245) / 115) ** 2 <= 1
    texture = np.full((height, width, 3), 55, dtype=np.uint8)
    texture[mask_a] = (60, 180, 245)
    texture[mask_b] = (220, 120, 70)
    z = np.full((height, width), 1100.0, dtype=np.float32)
    z[mask_a] = 700.0 + 10.0 * (xx[mask_a] - 220) / 105
    z[mask_b] = 860.0 + 18.0 * (yy[mask_b] - 245) / 115
    fx = fy = 520.0
    x = (xx - width / 2) * z / fx
    y = (yy - height / 2) * z / fy
    cloud = np.stack([x, y, z], axis=2).astype(np.float32)
    confidence = np.full((height, width), 0.95, dtype=np.float32)
    cloud[:8, :8] = 0

    frame = Frame3D(
        texture_bgr=texture,
        point_cloud_mm=cloud,
        depth_mm=z,
        confidence_map=confidence,
        frame_id=1,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        info={"source": "synthetic_demo"},
    )
    detections = [
        InstanceDetection(0, 0, "vat_mau_a", 0.99, bbox(mask_a), mask_a),
        InstanceDetection(1, 1, "vat_mau_b", 0.98, bbox(mask_b), mask_b),
    ]
    results = PhoXiYoloPipeline(
        KnownMaskSegmenter(detections), config["measurement"]
    ).process(frame)
    output = project_path("demo_output", config)
    output.mkdir(parents=True, exist_ok=True)
    save_capture(frame, output / "raw", stem="synthetic", save_3d=True)
    records = [item.to_dict() for item in results]
    (output / "measurements.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        from phoxi_yolo3d.visualization import draw_results

        save_bgr_png(output / "synthetic_annotated.png", draw_results(texture, results))
    except (ImportError, RuntimeError) as exc:
        print(f"Bo qua anh overlay: {exc}")
    print(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"Demo da luu tai: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

