#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from phoxi_yolo3d.config import load_class_names, load_config, project_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Huan luyen YOLO instance segmentation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    load_class_names()
    config = load_config(args.config)
    yolo_cfg = config["yolo"]
    train_cfg = config["training"]
    dataset_yaml = project_path(config["dataset"]["prepared_dir"], config) / "dataset.yaml"
    if not dataset_yaml.exists():
        raise SystemExit("Chua co dataset.yaml. Chay scripts/04_prepare_dataset.py truoc.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Chua cai ultralytics. Chay: pip install -e .") from exc

    model_source = args.model or yolo_cfg["base_model"]
    model = YOLO(model_source)
    kwargs = {
        "data": str(dataset_yaml),
        "epochs": args.epochs or int(train_cfg["epochs"]),
        "imgsz": args.imgsz or int(yolo_cfg["image_size"]),
        "batch": args.batch or int(train_cfg["batch"]),
        "workers": int(train_cfg["workers"]),
        "patience": int(train_cfg["patience"]),
        "seed": int(train_cfg["seed"]),
        "project": str(project_path(train_cfg["output_dir"], config)),
        "name": str(train_cfg["run_name"]),
        "plots": True,
        "resume": args.resume,
    }
    device = args.device if args.device is not None else yolo_cfg.get("device")
    if device is not None and str(device).strip():
        kwargs["device"] = device
    result = model.train(**kwargs)
    save_dir = Path(result.save_dir)
    best_source = save_dir / "weights" / "best.pt"
    if not best_source.exists():
        raise SystemExit(f"Train xong nhung khong tim thay {best_source}")
    best_target = project_path(yolo_cfg["weights"], config)
    best_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_source, best_target)
    print(f"Model tot nhat: {best_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

