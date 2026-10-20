"""Texture diagnostic and approximate grasp direction overlay on observed pixels."""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy.spatial import cKDTree

from .grasp_data import PreparedGraspScene


def draw_grasp_diagnostics(scene: PreparedGraspScene, rows: np.ndarray,
                           associations: list[dict[str, Any]]) -> np.ndarray:
    import cv2
    canvas = scene.frame.texture_bgr.copy()
    for target in scene.targets:
        contours, _ = cv2.findContours(target.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 255, 0) if target.stats["eligible"] else (0, 128, 255), 2)
        # Magenta marks removed/invalid/eroded pixels for reviewing the cleaning threshold.
        dropped = target.mask & ~target.clean_mask
        canvas[dropped] = (canvas[dropped] * .5 + np.array([255, 0, 255]) * .5).astype(np.uint8)
        yy, xx = np.nonzero(target.mask)
        if len(xx):
            label = f"{target.instance_id}: {target.class_name} pts={len(target.points_m)}"
            cv2.putText(canvas, label, (int(xx.min()), max(int(yy.min())-6, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1, cv2.LINE_AA)
    valid = np.isfinite(scene.frame.point_cloud_mm).all(axis=2) & (scene.frame.point_cloud_mm[..., 2] > 0)
    pixels_yx = np.argwhere(valid)
    points_m = scene.frame.point_cloud_mm[valid] / 1000.0
    if len(rows) and len(points_m):
        tree = cKDTree(points_m)
        seen = set()
        for row, association in zip(rows, associations):
            instance_id = association["instance_id"]
            if instance_id in seen:
                continue
            seen.add(instance_id)
            seed = row[13:16]
            distance, index = tree.query(seed)
            if distance > .02:
                continue
            y, x = pixels_yx[index]
            cv2.circle(canvas, (int(x), int(y)), 5, (0, 0, 255), -1)
            # Nearest observed XYZ provides a diagnostic overlay, not calibrated projection.
            end = seed + row[4:13].reshape(3, 3)[:, 1] * .025
            end_distance, end_index = tree.query(end)
            ey, ex = pixels_yx[end_index]
            if end_distance < .025:
                cv2.arrowedLine(canvas, (int(x), int(y)), (int(ex), int(ey)), (255, 255, 0), 2)
            cv2.putText(canvas, f"grasp {row[0]:.2f} W={row[1]*1000:.1f}mm",
                        (int(x)+8, int(y)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
    footer = np.zeros((55, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(footer, "Magenta=excluded pixels | Arrow=approx. jaw direction", (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    cv2.putText(footer, "CAMERA PROPOSALS ONLY - ROBOT TCP / COLLISION GEOMETRY NOT CALIBRATED", (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX, .40, (0, 190, 255), 1)
    return np.vstack((canvas, footer))
