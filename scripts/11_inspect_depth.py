#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from phoxi_yolo3d.storage import load_capture
from phoxi_yolo3d.visualization import depth_preview


def robust_value(frame, u: int, v: int, radius: int = 2):
    height, width = frame.texture_bgr.shape[:2]
    if not (0 <= u < width and 0 <= v < height):
        return None
    x1, x2 = max(0, u - radius), min(width, u + radius + 1)
    y1, y2 = max(0, v - radius), min(height, v + radius + 1)
    points = frame.point_cloud_mm[y1:y2, x1:x2].reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    points = points[valid]
    if not len(points):
        return None
    center = np.median(points, axis=0)
    return center, float(np.median(np.linalg.norm(points, axis=1)))


def pane_pixel(x: int, y: int, width: int, height: int):
    """OpenCV mouse coordinates are in the displayed image (before window scaling)."""
    return (x % width, y) if 0 <= x < 2 * width and 0 <= y < height else (-1, -1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Di chuot de xem X,Y,Z,D tren file scan NPZ")
    parser.add_argument("scan")
    args = parser.parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Can cai opencv-python.") from exc

    frame = load_capture(args.scan)
    texture = frame.texture_bgr.copy()
    depth = depth_preview(frame.depth_mm, frame.point_cloud_mm)
    state = {"u": texture.shape[1] // 2, "v": texture.shape[0] // 2}
    window = "PhoXi XYZ - di chuot, q de thoat"

    def mouse(event, x, y, flags, param):
        del event, flags, param
        state["u"], state["v"] = pane_pixel(x, y, texture.shape[1], texture.shape[0])

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
    while True:
        view = texture.copy()
        u, v = state["u"], state["v"]
        value = robust_value(frame, u, v)
        if value is None:
            text = f"pixel=({u},{v}) INVALID"
        else:
            xyz, distance = value
            text = (
                f"pixel=({u},{v}) X={xyz[0]:.2f} Y={xyz[1]:.2f} "
                f"Z={xyz[2]:.2f} D={distance:.2f} mm"
            )
        cv2.circle(view, (u, v), 6, (0, 0, 255), 2)
        cv2.rectangle(view, (0, 0), (view.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(view, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        combined = np.hstack([view, depth])
        cv2.imshow(window, combined)
        if (cv2.waitKey(20) & 0xFF) == ord("q"):
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

