from __future__ import annotations

from typing import Any

import numpy as np

from .types import InstanceResult


def suppress_duplicate_masks(
    results: list[InstanceResult], settings: dict[str, Any]
) -> tuple[list[InstanceResult], list[InstanceResult]]:
    """Suppress overlapping same-class masks, preserving disjoint objects.

    This cannot repair one object split into non-overlapping masks. Do not
    merge masks just because they have the same class or nearby centroids.
    """
    if not bool(settings.get("enabled", True)):
        return results, []
    iou_threshold = float(settings.get("mask_iou", 0.45))
    containment_threshold = float(settings.get("containment", 0.80))
    kept: list[InstanceResult] = []
    rejected: list[InstanceResult] = []
    for candidate in sorted(results, key=lambda item: item.detection.score, reverse=True):
        mask = np.asarray(candidate.detection.mask, dtype=bool)
        candidate_area = int(mask.sum())
        for previous in kept:
            if candidate.detection.class_id != previous.detection.class_id:
                continue
            previous_mask = np.asarray(previous.detection.mask, dtype=bool)
            if mask.shape != previous_mask.shape:
                continue
            intersection = int(np.count_nonzero(mask & previous_mask))
            if not intersection:
                continue
            previous_area = int(previous_mask.sum())
            iou = intersection / max(1, candidate_area + previous_area - intersection)
            containment = intersection / max(1, min(candidate_area, previous_area))
            if iou >= iou_threshold or containment >= containment_threshold:
                decision = candidate.postprocess
                decision.valid = False
                decision.reason = "DUPLICATE_MASK"
                decision.duplicate_of_instance_id = previous.detection.instance_id
                decision.duplicate_mask_iou = iou
                decision.duplicate_containment = containment
                rejected.append(candidate)
                break
        else:
            kept.append(candidate)
    return kept, rejected
