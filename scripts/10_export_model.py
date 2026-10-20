#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from phoxi_yolo3d.config import load_config, project_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Xuat model YOLO sang ONNX/TensorRT/OpenVINO")
    parser.add_argument("--config", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--format", default="onnx", choices=("onnx", "engine", "openvino"))
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    yolo_cfg = config["yolo"]
    weights = Path(args.weights).expanduser().resolve() if args.weights else project_path(yolo_cfg["weights"], config)
    if not weights.exists():
        raise SystemExit(f"Khong tim thay {weights}")

    from ultralytics import YOLO

    kwargs = {
        "format": args.format,
        "imgsz": args.imgsz or int(yolo_cfg["image_size"]),
        "half": args.half,
        "dynamic": args.format == "onnx",
    }
    if args.device:
        kwargs["device"] = args.device
    exported = YOLO(str(weights)).export(**kwargs)
    print(f"Da xuat: {exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

