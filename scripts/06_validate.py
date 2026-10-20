#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phoxi_yolo3d.config import load_class_names, load_config, project_path


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Danh gia model tren val/test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    load_class_names()
    config = load_config(args.config)
    yolo_cfg = config["yolo"]
    dataset_root = project_path(config["dataset"]["prepared_dir"], config)
    dataset_yaml = dataset_root / "dataset.yaml"
    split = args.split
    if split == "test" and not any((dataset_root / "images" / "test").glob("*")):
        print("Test split dang rong, chuyen sang val.")
        split = "val"
    weights = Path(args.weights).expanduser().resolve() if args.weights else project_path(yolo_cfg["weights"], config)
    if not weights.exists():
        raise SystemExit(f"Khong tim thay model: {weights}")

    from ultralytics import YOLO

    kwargs = {
        "data": str(dataset_yaml),
        "split": split,
        "imgsz": int(yolo_cfg["image_size"]),
        "plots": True,
        "verbose": True,
    }
    device = args.device if args.device is not None else yolo_cfg.get("device")
    if device is not None and str(device).strip():
        kwargs["device"] = device
    metrics = YOLO(str(weights)).val(**kwargs)
    report = {
        "weights": str(weights),
        "split": split,
        "results": json_safe(getattr(metrics, "results_dict", {})),
        "box_map50_95": float(metrics.box.map) if getattr(metrics, "box", None) else None,
        "box_map50": float(metrics.box.map50) if getattr(metrics, "box", None) else None,
        "mask_map50_95": float(metrics.seg.map) if getattr(metrics, "seg", None) else None,
        "mask_map50": float(metrics.seg.map50) if getattr(metrics, "seg", None) else None,
        "save_dir": str(getattr(metrics, "save_dir", "")),
    }
    report_dir = project_path("reports", config)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "validation_metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Da luu: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

