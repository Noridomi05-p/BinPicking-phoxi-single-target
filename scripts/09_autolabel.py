#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phoxi_yolo3d.config import load_class_names, load_config, project_path, validate_model_classes
from phoxi_yolo3d.dataset import list_images, write_yolo_segmentation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dung model pilot tao polygon goi y; van can nguoi dung review"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    class_names = load_class_names()
    cfg = config["dataset"]
    images_dir = project_path(cfg["raw_images"], config)
    manual_dir = project_path(cfg["raw_labels"], config)
    auto_dir = project_path(cfg["auto_labels"], config)
    weights = Path(args.weights).expanduser().resolve() if args.weights else project_path(config["yolo"]["weights"], config)
    if not weights.exists():
        raise SystemExit("Can train model pilot truoc khi auto-label.")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    try:
        validate_model_classes(model.names, class_names)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    auto_dir.mkdir(parents=True, exist_ok=True)
    summary = {"created": 0, "skipped_manual": 0, "skipped_existing": 0, "instances": 0}
    yolo_cfg = config["yolo"]
    for image_path in list_images(images_dir):
        manual_path = manual_dir / f"{image_path.stem}.txt"
        output_path = auto_dir / f"{image_path.stem}.txt"
        if manual_path.exists():
            summary["skipped_manual"] += 1
            continue
        if output_path.exists() and not args.overwrite:
            summary["skipped_existing"] += 1
            continue
        predictions = model.predict(
            source=str(image_path),
            conf=args.confidence,
            iou=float(yolo_cfg["iou"]),
            imgsz=int(yolo_cfg["image_size"]),
            retina_masks=True,
            verbose=False,
        )
        objects = []
        if predictions and predictions[0].masks is not None and predictions[0].boxes is not None:
            classes = predictions[0].boxes.cls.detach().cpu().numpy().astype(int)
            polygons = predictions[0].masks.xyn
            for class_id, polygon in zip(classes, polygons):
                polygon = np.asarray(polygon, dtype=float)
                if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
                    continue
                objects.append((int(class_id), polygon.reshape(-1).tolist()))
        write_yolo_segmentation(output_path, objects)
        summary["created"] += 1
        summary["instances"] += len(objects)
        print(f"{image_path.name}: {len(objects)} goi y")

    (auto_dir / "autolabel_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Tiep theo: python scripts/03_annotate.py de review va luu nhan thu cong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

