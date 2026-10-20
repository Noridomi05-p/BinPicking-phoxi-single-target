from __future__ import annotations

from typing import Any

import numpy as np

from .image_utils import erode_mask, resize_mask_nearest
from .types import Frame3D, InstanceMeasurement


def summarize_mask_3d(frame: Frame3D, mask: np.ndarray, settings: dict[str, Any]) -> InstanceMeasurement:
    """Coarse statistics only; no component fitting or grasp geometry here."""
    mask = resize_mask_nearest(mask, *frame.texture_bgr.shape[:2])
    count = int(mask.sum())
    points = frame.point_cloud_mm[mask]
    keep = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    keep &= points[:, 2] >= float(settings.get("min_z_mm", 0))
    keep &= points[:, 2] <= float(settings.get("max_z_mm", np.inf))
    if settings.get("confidence_min") is not None:
        if frame.confidence_map is None:
            keep[:] = False
        else:
            conf = np.squeeze(frame.confidence_map)[mask]
            keep &= np.isfinite(conf) & (conf >= float(settings["confidence_min"]))
    points = points[keep]
    enough = len(points) >= int(settings.get("min_valid_points", 80))
    result = InstanceMeasurement(enough, None if enough else "TOO_FEW_POINTS", len(points), count,
                                 len(points)/max(count, 1), measurement_kind="coarse")
    if enough:
        result.center_xyz_mm = tuple(float(v) for v in np.median(points, axis=0))
        result.depth_z_median_mm = float(np.median(points[:, 2]))
        result.distance_3d_median_mm = float(np.median(np.linalg.norm(points, axis=1)))
        result.depth_z_p10_mm, result.depth_z_p90_mm = map(float, np.percentile(points[:, 2], [10, 90]))
    return result


def measure_mask_3d(
    frame: Frame3D,
    mask: np.ndarray,
    settings: dict[str, Any],
) -> InstanceMeasurement:
    """Measure the visible surface selected by one instance mask."""

    height, width = frame.texture_bgr.shape[:2]
    original_mask = resize_mask_nearest(mask, height, width)
    mask_pixels = int(original_mask.sum())
    if mask_pixels == 0:
        return InstanceMeasurement(False, "mask_rong", 0, 0, 0.0)

    ys, xs = np.nonzero(original_mask)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    original_roi = original_mask[y1:y2, x1:x2]

    min_points = int(settings.get("min_valid_points", 80))
    radius = int(settings.get("mask_erode_px", 3))
    working_mask = erode_mask(original_roi, radius)
    if int(working_mask.sum()) < min_points:
        working_mask = original_roi

    cloud_roi = np.asarray(frame.point_cloud_mm[y1:y2, x1:x2], dtype=np.float32)
    confidence_roi = (
        None
        if frame.confidence_map is None
        else np.squeeze(frame.confidence_map)[y1:y2, x1:x2]
    )
    confidence_min = settings.get("confidence_min")

    def select_points(candidate_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        selected = cloud_roi[candidate_mask]
        valid_points = np.isfinite(selected).all(axis=1)
        valid_points &= selected[:, 2] > 0
        valid_points &= selected[:, 2] >= float(settings.get("min_z_mm", 0.0))
        valid_points &= selected[:, 2] <= float(settings.get("max_z_mm", np.inf))
        selected_confidence = None
        if confidence_roi is not None:
            selected_confidence = np.asarray(confidence_roi[candidate_mask])
            if confidence_min is not None:
                valid_points &= np.isfinite(selected_confidence)
                valid_points &= selected_confidence >= float(confidence_min)
            selected_confidence = selected_confidence[valid_points]
        return selected[valid_points], selected_confidence

    points, confidence_values = select_points(working_mask)
    if points.shape[0] < min_points and not np.array_equal(working_mask, original_roi):
        points, confidence_values = select_points(original_roi)

    raw_valid_count = int(points.shape[0])
    if raw_valid_count < min_points:
        return InstanceMeasurement(
            valid=False,
            reason=f"khong_du_diem_3d:{raw_valid_count}<{min_points}",
            point_count=raw_valid_count,
            mask_pixel_count=mask_pixels,
            valid_ratio=float(raw_valid_count / max(mask_pixels, 1)),
        )

    trim_percent = float(settings.get("trim_percent", 1.0))
    if 0 < trim_percent < 49 and points.shape[0] >= 100:
        low, high = np.percentile(points[:, 2], [trim_percent, 100.0 - trim_percent])
        keep = (points[:, 2] >= low) & (points[:, 2] <= high)
        trimmed = points[keep]
        if trimmed.shape[0] >= min_points:
            points = trimmed
            if confidence_values is not None:
                confidence_values = confidence_values[keep]

    ranges = np.linalg.norm(points, axis=1)
    center = np.median(points, axis=0)
    p05_xyz = np.percentile(points, 5, axis=0)
    p95_xyz = np.percentile(points, 95, axis=0)
    confidence_median = None
    if confidence_values is not None:
        conf_values = confidence_values[np.isfinite(confidence_values)]
        if conf_values.size:
            confidence_median = float(np.median(conf_values))

    return InstanceMeasurement(
        valid=True,
        reason=None,
        point_count=int(points.shape[0]),
        mask_pixel_count=mask_pixels,
        valid_ratio=float(raw_valid_count / max(mask_pixels, 1)),
        center_xyz_mm=tuple(float(value) for value in center),
        depth_z_median_mm=float(np.median(points[:, 2])),
        depth_z_p10_mm=float(np.percentile(points[:, 2], 10)),
        depth_z_p90_mm=float(np.percentile(points[:, 2], 90)),
        distance_3d_median_mm=float(np.median(ranges)),
        nearest_surface_mm=float(
            np.percentile(ranges, float(settings.get("nearest_percentile", 5.0)))
        ),
        visible_extent_xyz_mm=tuple(float(value) for value in (p95_xyz - p05_xyz)),
        confidence_median=confidence_median,
    )
