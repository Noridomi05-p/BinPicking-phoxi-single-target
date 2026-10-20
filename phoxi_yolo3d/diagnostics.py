from __future__ import annotations

from typing import Any

import numpy as np

from .types import Frame3D, InstanceResult


def frame_diagnostics(frame: Frame3D, measurement: dict[str, Any], postprocessor) -> dict:
    cloud = frame.point_cloud_mm
    z = cloud[:, :, 2]
    raw_valid = np.isfinite(cloud).all(axis=2) & (z > 0)
    usable = raw_valid & (z >= float(measurement.get("min_z_mm", 0)))
    usable &= z <= float(measurement.get("max_z_mm", np.inf))
    roi_mask = np.ones(z.shape, dtype=bool)
    if postprocessor.roi is not None:
        roi_mask = postprocessor.roi.mask(z.shape)
    count = int(roi_mask.sum())

    def ratio(mask):
        return float(np.count_nonzero(mask & roi_mask) / max(count, 1))

    result = {
        "roi_pixels": count,
        "pointcloud_valid_ratio_roi": ratio(raw_valid),
        "pointcloud_in_range_ratio_roi": ratio(usable),
        "pointcloud_invalid_ratio_roi": 1.0 - ratio(raw_valid),
        "background_valid_ratio_roi": None,
        "paired_valid_ratio_roi": None,
        "depthmap_valid_ratio_roi": None,
    }
    if frame.depth_mm is not None:
        depth = np.squeeze(frame.depth_mm)
        result["depthmap_valid_ratio_roi"] = ratio(np.isfinite(depth) & (depth > 0))
    reference = postprocessor.surface_reference
    if reference is not None and reference.depth_mm.shape == z.shape:
        background = np.isfinite(reference.depth_mm) & (reference.depth_mm > 0)
        background &= reference.depth_mm >= float(measurement.get("min_z_mm", 0))
        background &= reference.depth_mm <= float(measurement.get("max_z_mm", np.inf))
        result["background_valid_ratio_roi"] = ratio(background)
        result["paired_valid_ratio_roi"] = ratio(usable & background)
    return result


def result_counts(results: list[InstanceResult], rejected: list[InstanceResult]) -> dict:
    return {
        "object_count": len(results),
        "verified_3d_count": sum(item.verified_3d for item in results),
        "unverified_count": sum(not item.verified_3d for item in results),
        "rejected_count": len(rejected),
    }
