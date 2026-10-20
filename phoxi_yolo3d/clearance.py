"""Conservative observed approach check, never a certificate of robot safety.

Boxes must bound the REAL open gripper including linkage movement. A point
cloud without points in a box is not proof of free space: every camera ray
through that box must have an observed surface behind its far boundary.
Closing contact, held-object retreat and the robot arm remain unverified.
"""
from __future__ import annotations

from itertools import product
import numpy as np


REQUIRED_DIMENSIONS = ("finger_thickness_mm", "finger_span_mm", "finger_length_mm",
    "palm_span_mm", "palm_width_mm", "palm_depth_mm", "insertion_depth_mm",
    "approach_mm", "vertical_motion_envelope_mm")


def fit_projection(frame, max_error_px=.5):
    """Fit only if organized XYZ supports one pinhole projection to subpixel error.

    This is for scene diagnostics, not camera/robot calibration. Validate on
    ALL valid pixels, not just the fit sample; never invent rays when it fails.
    """
    cloud = frame.point_cloud_mm
    valid = np.isfinite(cloud).all(axis=2) & (cloud[..., 2] > 0)
    y, x = np.nonzero(valid)
    points = cloud[valid].astype(np.float64)
    if len(points) < 30:
        return None
    rays = np.column_stack((points[:, :2]/points[:, 2:3], np.ones(len(points))))
    pixels = np.column_stack((x, y))
    index = np.linspace(0, len(points)-1, min(4000, len(points))).astype(int)
    coefficients, _, rank, _ = np.linalg.lstsq(rays[index], pixels[index], rcond=None)
    if rank != 3:
        return None
    error = np.linalg.norm(rays @ coefficients-pixels, axis=1)
    if error.max() > max_error_px:
        return None
    k = np.vstack((coefficients.T, [0., 0., 1.]))
    if abs(np.linalg.det(k)) < 1e-8:
        return None
    return {"matrix": k, "inverse": np.linalg.inv(k),
            "max_error_px": float(error.max()), "source": "fitted_organized_xyz_diagnostic"}


def approach_boxes(opening_mm, settings):
    """Three conservative boxes swept along straight -z -> +z approach."""
    t, span, length = (float(settings[k]) for k in REQUIRED_DIMENSIONS[:3])
    insertion = float(settings["insertion_depth_mm"])
    if insertion >= length:
        raise ValueError("INSERTION_EXCEEDS_FINGER_LENGTH")
    approach = float(settings["approach_mm"])
    margin = float(settings.get("safety_margin_mm", 2.0))
    vertical = float(settings["vertical_motion_envelope_mm"])
    boxes = []
    for sign in (-1, 1):
        y0, y1 = sorted((sign*opening_mm/2, sign*(opening_mm/2+t)))
        low = np.array([-span/2, y0, insertion-length-approach-vertical])-margin
        high = np.array([span/2, y1, insertion+vertical])+margin
        boxes.append((f"finger_{sign:+d}", low, high))
    low = np.array([-float(settings["palm_span_mm"])/2,
                    -float(settings["palm_width_mm"])/2,
                    insertion-length-float(settings["palm_depth_mm"])-approach-vertical])-margin
    high = np.array([float(settings["palm_span_mm"])/2,
                     float(settings["palm_width_mm"])/2, insertion-length+vertical])+margin
    boxes.append(("palm", low, high))
    return boxes


def check_boxes(frame, anchor, rotation, boxes, projection):
    """Check all measured XYZ, including rejected masks and out-of-range returns."""
    cloud = frame.point_cloud_mm
    valid = np.isfinite(cloud).all(axis=2) & (cloud[..., 2] > 0)
    local = (cloud[valid]-anchor) @ rotation
    diagnostics = []
    for name, low, high in boxes:
        count = int(np.count_nonzero(np.all((local >= low) & (local <= high), axis=1)))
        diagnostics.append({"part": name, "observed_collision_points": count})
    if any(row["observed_collision_points"] for row in diagnostics):
        return {"status": "BLOCKED", "reason": "OBSERVED_VOLUME_COLLISION", "parts": diagnostics}
    if projection is None:
        return {"status": "UNKNOWN", "reason": "PROJECTION_UNVERIFIED", "parts": diagnostics}

    h, w = valid.shape
    k, inverse = projection["matrix"], projection["inverse"]
    unknown_total, checked_total = 0, 0
    for (name, low, high), diag in zip(boxes, diagnostics):
        corners_local = np.array(list(product(*zip(low, high))))
        corners = corners_local @ rotation.T + anchor
        if np.any(corners[:, 2] <= 0):
            return {"status": "UNKNOWN", "reason": "VOLUME_BEHIND_CAMERA", "parts": diagnostics}
        uvw = corners @ k.T
        uv = uvw[:, :2]/uvw[:, 2:3]
        # A one-pixel guard covers the residual fitted projection error.
        if np.any(uv.min(axis=0) < 1) or uv[:, 0].max() > w-2 or uv[:, 1].max() > h-2:
            return {"status": "UNKNOWN", "reason": "VOLUME_OUTSIDE_FOV", "parts": diagnostics}
        lo = np.floor(uv.min(axis=0)-1).astype(int)
        hi = np.ceil(uv.max(axis=0)+1).astype(int)
        yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        rays = np.column_stack((xx.ravel(), yy.ravel(), np.ones(xx.size))) @ inverse.T
        directions = rays @ rotation
        origin = -np.asarray(anchor) @ rotation
        near = np.full(len(rays), -np.inf)
        far = np.full(len(rays), np.inf)
        intersects = np.ones(len(rays), bool)
        for axis in range(3):
            parallel = np.abs(directions[:, axis]) < 1e-12
            intersects &= ~(parallel & ((origin[axis] < low[axis]) | (origin[axis] > high[axis])))
            a = np.full(len(rays), -np.inf)
            b = np.full(len(rays), np.inf)
            moving = ~parallel
            t0 = (low[axis]-origin[axis])/directions[moving, axis]
            t1 = (high[axis]-origin[axis])/directions[moving, axis]
            a[moving], b[moving] = np.minimum(t0, t1), np.maximum(t0, t1)
            near, far = np.maximum(near, a), np.minimum(far, b)
        intersects &= (far >= np.maximum(near, 0))
        y, x = yy.ravel()[intersects], xx.ravel()[intersects]
        exit_z = far[intersects]
        # Conservative pixel neighbourhood: no hole or foreground pixel may
        # be silently skipped at the projected box edge.
        free = np.ones(len(x), bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                sy, sx = np.clip(y+dy, 0, h-1), np.clip(x+dx, 0, w-1)
                free &= valid[sy, sx] & (cloud[sy, sx, 2] > exit_z+1.0)
        unknown = int(np.count_nonzero(~free))
        unknown_total += unknown
        checked_total += len(x)
        diag.update(checked_rays=len(x), unknown_or_occluded_rays=unknown)
    if unknown_total or not checked_total:
        return {"status": "UNKNOWN", "reason": "UNOBSERVED_OR_OCCLUDED_VOLUME", "parts": diagnostics}
    return {"status": "OBSERVED_APPROACH_CLEAR", "reason": None, "parts": diagnostics,
            "projection_max_error_px": projection["max_error_px"]}


def check_approach(frame, anchor, rotation, opening_mm, settings, projection=None):
    incomplete = [k for k in REQUIRED_DIMENSIONS if settings.get(k) is None]
    if not settings.get("enabled", True):
        result = {"status": "UNKNOWN", "reason": "CLEARANCE_DISABLED"}
    elif incomplete or not settings.get("geometry_verified", False):
        result = {"status": "UNKNOWN", "reason": "GRIPPER_GEOMETRY_UNCONFIGURED", "missing": incomplete}
    else:
        try:
            boxes = approach_boxes(opening_mm, settings)
            result = check_boxes(frame, np.asarray(anchor), np.asarray(rotation), boxes, projection)
        except ValueError as exc:
            result = {"status": "UNKNOWN", "reason": str(exc)}
    result.update(model="configured_open_gripper_swept_boxes",
                  closing_status="NOT_VERIFIED_LINKAGE_AND_CONTACT",
                  retreat_status="NOT_VERIFIED_HELD_OBJECT",
                  arm_status="NOT_VERIFIED", robot_ready=False)
    return result
