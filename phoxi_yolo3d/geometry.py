"""Millimetre geometry. No robot transforms and no synthetic depth filling."""
from __future__ import annotations

import numpy as np


def unit(vector):
    v = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("DEGENERATE_VECTOR")
    return v / norm


def toward_camera(normal, point):
    n = unit(normal)
    return -n if np.dot(n, -np.asarray(point)) < 0 else n


def fit_plane(points, inlier_mm=1.5, min_ratio=.8, max_rms_mm=1.0, seed=42):
    """RANSAC + SVD; return signed n.X+d, positive toward camera origin."""
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
    if len(points) < 20:
        raise ValueError("PLANE_TOO_FEW_POINTS")
    rng = np.random.default_rng(seed)
    if len(points) > 8000:
        points = points[rng.choice(len(points), 8000, replace=False)]
    best = np.zeros(len(points), bool)
    for _ in range(100):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b-a, c-a)
        if np.linalg.norm(n) < 1e-8:
            continue
        n = unit(n)
        keep = np.abs((points-a) @ n) <= inlier_mm
        if keep.sum() > best.sum():
            best = keep
    if best.sum() < 20 or best.mean() < min_ratio:
        raise ValueError("PLANE_LOW_INLIER_RATIO")
    for _ in range(2):
        center = points[best].mean(axis=0)
        _, singular, vt = np.linalg.svd(points[best]-center, full_matrices=False)
        if singular[1] < 1e-5:
            raise ValueError("PLANE_COLLINEAR_POINTS")
        normal = toward_camera(vt[-1], center)
        d = -float(center @ normal)
        best = np.abs(points @ normal + d) <= inlier_mm
    ratio = float(best.mean())
    rms = float(np.sqrt(np.mean((points[best] @ normal + d)**2)))
    if ratio < min_ratio or rms > max_rms_mm:
        raise ValueError("PLANE_POOR_FIT")
    return {"normal": normal.tolist(), "offset_mm": d, "inlier_ratio": ratio,
            "rms_mm": rms, "sample_points": len(points), "origin": "explicit_floor_roi"}


def frame_identity(frame):
    """Only acquisition geometry/settings; never frame number or wall-clock time."""
    data = frame.info.get("acquisition", {})
    return {"shape_hw": list(frame.texture_bgr.shape[:2]),
            **{k: data.get(k) for k in ("device_id", "coordinate_space", "roi_xyxy",
              "source_shape_hw", "calibration_id", "texture_source", "led_power")}}


def identity_status(saved, frame):
    current = frame_identity(frame)
    if not saved:
        return "UNBOUND"
    for key in set(saved) | set(current):
        if saved.get(key) != current.get(key):
            return "MISMATCH"
    if not saved.get("device_id") or not saved.get("calibration_id") or saved.get("coordinate_space") != "CameraSpace/PrimaryCamera":
        return "UNBOUND"
    return "MATCH"


def plane_basis(normal):
    n = unit(normal)
    reference = np.array([1., 0., 0.]) if abs(n[0]) < .9 else np.array([0., 1., 0.])
    u = unit(reference-n*np.dot(reference, n))
    return u, unit(np.cross(n, u))


def rotation_for_jaws(normal, closing):
    """Columns x=tangent, y=jaw closing, z=approach into the object."""
    z = -unit(normal)
    y = unit(np.asarray(closing)-z*np.dot(closing, z))
    x = unit(np.cross(y, z))
    return np.column_stack((x, y, z))


def to_graspnet_rotation(rotation_camera):
    """GraspNet columns: approach, jaw-closing, height. Proper SO(3) change."""
    r = np.asarray(rotation_camera)
    return np.column_stack((r[:, 2], r[:, 1], -r[:, 0]))
