#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from phoxi_yolo3d.camera import PhoXiCamera
from phoxi_yolo3d.config import load_config, project_path
from phoxi_yolo3d.storage import frame_statistics, save_capture
from phoxi_yolo3d.image_utils import prepare_frame_texture


def main() -> int:
    parser = argparse.ArgumentParser(description="Chup bo anh tu PhoXi bang software trigger")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--mode", choices=("manual", "interval"), default="manual")
    parser.add_argument("--interval", type=float, default=None)
    parser.add_argument("--save-3d", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.count <= 0:
        raise SystemExit("--count phai > 0")

    config = load_config(args.config)
    camera_cfg = config["camera"]
    texture_cfg = config["texture"]
    capture_cfg = config["capture"]
    output_dir = project_path(capture_cfg["output_dir"], config)
    save_3d = capture_cfg.get("save_3d", False) if args.save_3d is None else args.save_3d
    interval = args.interval if args.interval is not None else capture_cfg["interval_seconds"]

    camera = PhoXiCamera(
        device_id=args.device_id or camera_cfg.get("device_id"),
        address=camera_cfg.get("address"),
        api_lib_path=camera_cfg.get("api_lib_path"),
        refresh_discovery=bool(camera_cfg.get("refresh_discovery", True)),
        timeout_ms=int(camera_cfg.get("timeout_ms", 30_000)),
        force_camera_space=bool(camera_cfg.get("force_camera_space", True)),
        texture_source=camera_cfg.get("texture_source"),
        led_power=camera_cfg.get("led_power"),
        texture_lower_percentile=float(texture_cfg.get("lower_percentile", 1.0)),
        texture_upper_percentile=float(texture_cfg.get("upper_percentile", 99.5)),
        roi_xyxy=camera_cfg.get("roi_xyxy"),
        calibration_id=camera_cfg.get("calibration_id"),
    )

    print(f"Se luu anh vao: {output_dir / 'images'}")
    print(f"Luu 3D: {save_3d}")
    with camera:
        print(f"Da ket noi: {camera.device_id}")
        if camera.applied_capture_settings:
            print(f"Cau hinh texture: {camera.applied_capture_settings}")
        if camera.component_errors:
            print(f"Canh bao output tuy chon: {camera.component_errors}")
        if camera.coordinate_space_errors:
            print(f"Thong tin CameraSpace: {camera.coordinate_space_errors}")
        for index in range(args.count):
            if args.mode == "manual":
                command = input(
                    f"\n[{index + 1}/{args.count}] Sap xep vat, giu yen roi Enter de chup (q de dung): "
                ).strip().lower()
                if command == "q":
                    break
            frame = camera.capture()
            prepare_frame_texture(frame, texture_cfg)
            paths = save_capture(frame, output_dir, save_3d=save_3d)
            stats = frame_statistics(frame)
            print(
                f"Da chup {paths['image'].name} | "
                f"valid3D={stats['valid_3d_ratio'] * 100:.1f}%"
            )
            if args.mode == "interval" and index + 1 < args.count:
                time.sleep(max(0.0, interval))
    print("Hoan thanh thu du lieu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

