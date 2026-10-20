"""Prepare aligned PhoXi XYZ and saved instance masks for offline grasp proposals."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .image_utils import erode_mask
from .roi import PolygonROI
from .storage import load_capture
from .types import Frame3D


@dataclass
class GraspTarget:
    instance_id: int
    class_name: str
    yolo_confidence: float | None
    mask: np.ndarray
    clean_mask: np.ndarray
    points_m: np.ndarray
    stats: dict[str, Any]


@dataclass
class PreparedGraspScene:
    frame: Frame3D
    targets: list[GraspTarget]
    workspace_mask: np.ndarray
    network_mask: np.ndarray
    network_points_m: np.ndarray
    network_colors_rgb: np.ndarray
    collision_points_m: np.ndarray
    warnings: list[str]
    source_paths: dict[str, str | None]


def remove_small_3d_components(
    cloud_mm: np.ndarray, mask: np.ndarray, settings: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """4-neighbour pixel graph; edges require nearby real XYZ, never fill holes."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    mask = np.asarray(mask, dtype=bool)
    if cloud_mm.shape != (*mask.shape, 3):
        raise ValueError("Mask va point cloud khong cung kich thuoc.")
    inner = erode_mask(mask, int(settings.get("mask_erode_px", 3)))
    valid = inner & np.isfinite(cloud_mm).all(axis=2) & (cloud_mm[..., 2] > 0)
    count = int(valid.sum())
    result = np.zeros_like(valid)
    stats: dict[str, Any] = {
        "mask_pixels": int(mask.sum()),
        "inner_mask_pixels": int(inner.sum()),
        "valid_points_before_cleanup": count,
        "valid_ratio_inner_mask": count / max(int(inner.sum()), 1),
    }
    if count:
        ids = np.full(mask.shape, -1, dtype=np.int32)
        ids[valid] = np.arange(count)
        jump_mm = float(settings.get("neighbor_jump_mm", 12.0))
        sources, destinations = [], []
        for a, b in ((np.s_[:, :-1], np.s_[:, 1:]),
                     (np.s_[:-1, :], np.s_[1:, :])):
            neighbours = valid[a] & valid[b]
            delta = cloud_mm[a][neighbours] - cloud_mm[b][neighbours]
            nearby = np.linalg.norm(delta, axis=1) <= jump_mm
            sources.append(ids[a][neighbours][nearby])
            destinations.append(ids[b][neighbours][nearby])
        rows, cols = np.concatenate(sources), np.concatenate(destinations)
        graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                           shape=(count, count)).tocsr()
        n_components, labels = connected_components(graph, directed=False)
        sizes = np.bincount(labels)
        cutoff = max(int(settings.get("min_component_points", 64)),
                     int(np.ceil(count * float(settings.get("min_component_fraction", .0025)))))
        result[valid] = sizes[labels] >= cutoff
        stats.update(component_count=int(n_components), component_min_points=cutoff,
                     largest_component_points=int(sizes.max()))
    kept = int(result.sum())
    ratio = kept / max(count, 1)
    reasons = []
    if kept < int(settings.get("min_target_points", 200)):
        reasons.append("TOO_FEW_TARGET_POINTS")
    if ratio < float(settings.get("min_retained_ratio", .5)):
        reasons.append("TOO_MANY_DISCONNECTED_POINTS")
    if stats["valid_ratio_inner_mask"] < float(settings.get("min_valid_mask_ratio", .3)):
        reasons.append("TOO_MUCH_MISSING_DEPTH")
    stats.update(clean_points=kept, removed_points=count-kept, retained_ratio=ratio,
                 eligible=not reasons, reasons=reasons)
    return result, stats


def _read_object_metadata(path: Path | None, frame: Frame3D) -> dict[int, dict[str, Any]] | None:
    if path is None:
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    # result.json written by save_inspection represents exactly one frozen scan.
    if not isinstance(record, dict) or not isinstance(record.get("objects"), list):
        raise ValueError("--result can result.json cua inspection, khong phai offline_results.json.")
    for key in ("frame_id", "timestamp_utc"):
        saved = record.get(key)
        current = getattr(frame, key)
        if saved is not None and current is not None and saved != current:
            raise ValueError(f"result.json thuoc scan khac: {key} khong khop.")
    ids = [int(item["instance_id"]) for item in record["objects"]]
    if len(ids) != len(set(ids)):
        raise ValueError("result.json co instance_id trung lap.")
    return {int(item["instance_id"]): item for item in record["objects"]}


def prepare_scene(
    scan_path: Path, masks_path: Path, settings: dict[str, Any],
    result_path: Path | None = None, roi_path: Path | None = None,
    instance_ids: list[int] | None = None,
) -> PreparedGraspScene:
    # The persisted coordinate declarations are authoritative; XYZ is already aligned.
    with np.load(scan_path, allow_pickle=False) as arrays:
        meta = json.loads(str(arrays["metadata_json"].item()))
    if meta.get("coordinate_space") != "camera" or meta.get("distance_unit") != "millimetre":
        raise ValueError("Scan phai khai bao coordinate_space=camera, distance_unit=millimetre.")
    frame = load_capture(scan_path)
    shape = frame.texture_bgr.shape[:2]
    workspace = PolygonROI.load(roi_path).mask(shape) if roi_path else np.ones(shape, bool)
    objects = _read_object_metadata(result_path, frame)
    warnings: list[str] = []
    if roi_path is None:
        warnings.append("NO_ROI: network workspace is the full saved frame; camera crop may hide walls.")
    if objects is None and instance_ids is None:
        raise ValueError("Thieu result.json: can --instance-ids (chi chon vat da xac nhan).")
    if objects is None:
        warnings.append("NO_RESULT_METADATA: explicit instance IDs selected; no saved 3D verification/class labels.")
    ids = instance_ids if instance_ids is not None else sorted(objects or {})
    if len(ids) != len(set(ids)) or any(i < 0 for i in ids):
        raise ValueError("instance IDs phai duy nhat va >= 0.")
    targets = []
    removal = np.zeros(shape, bool)
    with np.load(masks_path, allow_pickle=False) as masks:
        for instance_id in ids:
            key = f"instance_{instance_id}"
            if key not in masks:
                raise ValueError(f"Khong co mask {key} trong {masks_path}")
            obj = (objects or {}).get(instance_id)
            if objects is not None and obj is None:
                raise ValueError(f"Instance {instance_id} khong thuoc OBJECTS cua result.json.")
            binary = np.asarray(masks[key], dtype=bool)
            if binary.shape != shape:
                raise ValueError(f"{key}: mask {binary.shape} khac scan {shape}")
            inside = binary & workspace
            clean_mask, stats = remove_small_3d_components(frame.point_cloud_mm, inside, settings)
            if obj is not None and not obj.get("verified_3d", False):
                stats["eligible"] = False
                stats["reasons"].append("NOT_3D_VERIFIED_IN_INSPECTION")
            removal |= binary & ~clean_mask
            targets.append(GraspTarget(
                instance_id=instance_id,
                class_name=str(obj.get("class_name", key)) if obj else key,
                yolo_confidence=float(obj["confidence"]) if obj else None,
                mask=binary, clean_mask=clean_mask,
                points_m=frame.point_cloud_mm[clean_mask].astype(np.float32) / 1000.0,
                stats=stats,
            ))
    valid = np.isfinite(frame.point_cloud_mm).all(axis=2) & (frame.point_cloud_mm[..., 2] > 0)
    network_mask = valid & workspace & ~removal
    # Keep the complete observed scene, including bin walls, for collision checks.
    # Removed target outliers are retained here: noisy evidence must not become free space.
    return PreparedGraspScene(
        frame, targets, workspace, network_mask,
        frame.point_cloud_mm[network_mask].astype(np.float32) / 1000.0,
        frame.texture_bgr[network_mask][:, ::-1].astype(np.float32) / 255.0,
        frame.point_cloud_mm[valid].astype(np.float32) / 1000.0,
        warnings,
        {"scan": str(scan_path), "masks": str(masks_path),
         "result": str(result_path) if result_path else None,
         "roi": str(roi_path) if roi_path else None},
    )


def sample_real_points(points_m: np.ndarray, number: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if points_m.ndim != 2 or points_m.shape[1] != 3 or not len(points_m):
        raise ValueError("Khong co diem that trong workspace.")
    if number < 2048:
        raise ValueError("Baseline can num_points >= 2048; khuyen nghi 20000.")
    rng = np.random.default_rng(seed)
    if len(points_m) >= number:
        indices = rng.choice(len(points_m), number, replace=False)
    else:
        indices = np.concatenate((np.arange(len(points_m)),
                                  rng.choice(len(points_m), number-len(points_m), replace=True)))
        rng.shuffle(indices)
    return np.ascontiguousarray(points_m[indices], dtype=np.float32), indices
