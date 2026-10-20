#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys

from phoxi_yolo3d.camera import PhoXiCamera
from phoxi_yolo3d.config import load_config


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "CHUA_CAI"


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiem tra moi truong PhoXi YOLO 3D")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-refresh", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    camera_config = config["camera"]

    print("=== HE THONG ===")
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PHOXI_CONTROL_PATH: {os.getenv('PHOXI_CONTROL_PATH', 'CHUA_DAT')}")
    for package in ("phoxi-api", "numpy", "opencv-python", "ultralytics", "torch"):
        print(f"{package}: {package_version(package)}")

    try:
        control = PhoXiCamera.create_control(camera_config.get("api_lib_path"))
        print("\n=== PHOXI CONTROL ===")
        print(f"Running: {control.is_phoxicontrol_running()}")
        if not control.is_phoxicontrol_running():
            print("LOI: Hay mo PhoXi Control roi chay lai.")
            return 2
        print(f"PhoXi Control version: {control.get_phoxicontrol_version()}")
        print(f"PhoXi API version: {control.get_api_version()}")
        print(f"Authenticated: {control.is_authenticated()}")
        devices = control.get_device_list(refresh=not args.no_refresh)
        print(f"\n=== CAMERA ({len(devices)}) ===")
        print(json.dumps(devices, ensure_ascii=False, indent=2, default=str))
        if not devices:
            print("LOI: Khong tim thay camera. Kiem tra nguon/cap Ethernet/IP.")
            return 3
        print("\nKET LUAN: Lop giao tiep PhoXi san sang.")
        return 0
    except Exception as exc:
        print(f"\nLOI: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

