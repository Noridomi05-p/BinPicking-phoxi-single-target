"""Robotiq 2F-85 grasp indicator drawn directly on the selected object.

Measured pose and opening determine the jaw markers. Their short bars are a
2D annotation, not a calibrated Robotiq mesh. Display projection is isolated
from collision checks. No detached inset and no invented off-plane direction.
"""
from __future__ import annotations

import numpy as np


PINK = (235, 70, 235)
YELLOW = (40, 230, 255)
AXES = (("x", (60, 80, 255)), ("y jaw", (70, 235, 80)),
        ("z approach", (255, 160, 40)))


def fit_display_projection(frame, anchor_uv):
    """Robust diagnostic pinhole fit with independent pixel validation.

    Allow small residuals for illustration, then verify the actual anchor.
    This is intentionally separate from clearance.fit_projection, whose
    stricter all-pixel check and collision decisions remain unchanged.
    """
    cloud = frame.point_cloud_mm
    valid = np.isfinite(cloud).all(axis=2) & (cloud[..., 2] > 0)
    yy, xx = np.nonzero(valid)
    if len(xx) < 100:
        return None
    # Separate deterministic training and validation samples of measured XYZ.
    count = min(len(xx), 12000)
    index = np.linspace(0, len(xx)-1, count).astype(int)
    points = cloud[yy[index], xx[index]].astype(float)
    pixels = np.column_stack((xx[index], yy[index]))
    rays = np.column_stack((points[:, :2]/points[:, 2:3], np.ones(count)))
    train = np.arange(count) % 2 == 0
    keep = train.copy()
    try:
        for _ in range(3):
            coefficients, _, rank, _ = np.linalg.lstsq(rays[keep], pixels[keep], rcond=None)
            if rank != 3:
                return None
            errors = np.linalg.norm(rays @ coefficients-pixels, axis=1)
            median = float(np.median(errors[train]))
            mad = float(np.median(np.abs(errors[train]-median)))
            keep = train & (errors <= max(1.0, median+4*1.4826*mad))
            if keep.sum() < 40 or keep.sum() < .85*train.sum():
                return None
        p95 = float(np.percentile(errors[~train], 95))
        if p95 > 2.0:
            return None
        matrix = np.vstack((coefficients.T, [0., 0., 1.]))
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
            return None
        u, v = map(int, anchor_uv)
        if not (0 <= u < cloud.shape[1] and 0 <= v < cloud.shape[0] and valid[v, u]):
            return None
        uvw = matrix @ cloud[v, u]
        error = float(np.linalg.norm(uvw[:2]/uvw[2]-[u, v]))
        if not np.isfinite(error) or error > 2.0:
            return None
        return {"matrix": matrix, "p95_error_px": p95, "anchor_error_px": error,
                "purpose": "DISPLAY_ONLY_NOT_FOR_COLLISION"}
    except np.linalg.LinAlgError:
        return None


def fit_surface_projection(frame, pose, target_mask):
    """Fit only the selected surface's tangent plane to its observed pixels.

    A homography can describe the two in-plane jaw positions even when a
    whole-frame pinhole fit fails. It cannot project an off-plane approach
    vector, a finger volume or an unseen robot body. Never used for clearance.
    """
    import cv2
    if target_mask is None or target_mask.shape != frame.point_cloud_mm.shape[:2]:
        return None
    anchor = np.asarray(pose["surface_anchor_xyz_mm"], dtype=float)
    rotation = np.asarray(pose["rotation_camera"], dtype=float)
    cloud = frame.point_cloud_mm
    valid = target_mask.astype(bool) & np.isfinite(cloud).all(axis=2) & (cloud[..., 2] > 0)
    yy, xx = np.nonzero(valid)
    if len(xx) < 80:
        return None
    local = (cloud[yy, xx].astype(float)-anchor) @ rotation
    # Use measured points near the anchor's local surface, not adjacent layers.
    radius = max(20., float(pose["proposed_opening_mm"])*.65)
    keep = (np.abs(local[:, 2]) <= 3.) & (np.linalg.norm(local[:, :2], axis=1) <= radius)
    local, pixels = local[keep, :2], np.column_stack((xx, yy))[keep]
    if len(local) < 80 or np.linalg.eigvalsh(np.cov(local.T))[0] < .25:
        return None
    step = max(1, (len(local)+5999)//6000)
    local, pixels = local[::step], pixels[::step]
    train = np.arange(len(local)) % 3 != 0
    try:
        matrix, inliers = cv2.findHomography(local[train], pixels[train], cv2.RANSAC,
            1.5, maxIters=1000, confidence=.995)
        if matrix is None or not np.isfinite(matrix).all() or inliers.mean() < .85:
            return None
        def mapped(points):
            homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
            if np.any(np.abs(homogeneous[:, 2]) < 1e-8):
                return None
            return homogeneous[:, :2]/homogeneous[:, 2:3]
        observed = mapped(local[~train])
        if observed is None:
            return None
        p95 = float(np.percentile(np.linalg.norm(observed-pixels[~train], axis=1), 95))
        center = mapped(np.zeros((1, 2)))
        if center is None:
            return None
        error = float(np.linalg.norm(center[0]-pose["surface_anchor_uv"]))
        if not np.isfinite([p95, error]).all() or p95 > 2.0 or error > 2.0:
            return None
        # Reject a degenerate or extremely anisotropic local image mapping.
        if abs(matrix[2, 2]) < 1e-8:
            return None
        jacobian = (matrix[:2, :2]*matrix[2, 2] -
                    np.outer(matrix[:2, 2], matrix[2, :2]))/matrix[2, 2]**2
        scales = np.linalg.svd(jacobian, compute_uv=False)
        if scales[-1] < 1e-3 or scales[0] > 100 or scales[0]/scales[-1] > 50:
            return None
        opening = float(pose["proposed_opening_mm"])
        extent = np.max(np.abs(local[:, 1]))
        if opening/2 > extent + max(12., opening*.35):
            return None
        return {"kind": "LOCAL_SURFACE", "homography": matrix,
                "p95_error_px": p95, "anchor_error_px": error,
                "purpose": "DISPLAY_ONLY_NOT_FOR_COLLISION"}
    except (cv2.error, np.linalg.LinAlgError):
        return None


def display_projection(frame, pose, target_mask=None):
    # Private per-frame cache; neither the saved frame metadata nor the plan
    # is mutated by redraw. Pose changes invalidate a local surface fit.
    key = (id(frame.point_cloud_mm), id(target_mask), tuple(pose["surface_anchor_uv"]),
           tuple(pose["surface_anchor_xyz_mm"]), tuple(np.asarray(pose["rotation_camera"]).ravel()))
    cached = getattr(frame, "_picking_display_projection", None)
    if cached is None or cached[0] != key:
        u, v = map(int, pose["surface_anchor_uv"])
        cloud = frame.point_cloud_mm
        consistent = (0 <= u < cloud.shape[1] and 0 <= v < cloud.shape[0] and
            np.isfinite(cloud[v, u]).all() and cloud[v, u, 2] > 0 and
            np.linalg.norm(cloud[v, u]-pose["surface_anchor_xyz_mm"]) <= .25)
        fitted = None
        if consistent:
            fitted = fit_display_projection(frame, pose["surface_anchor_uv"])
            if fitted is None:
                fitted = fit_surface_projection(frame, pose, target_mask)
        cached = (key, fitted)
        frame._picking_display_projection = cached
    return cached[1]


def grasp_symbol_geometry(pose):
    """Two parallel pad markers; opening is metric, bar length is illustrative."""
    opening = float(pose["proposed_opening_mm"])
    span = float(np.clip(opening*.45, 12., 32.))
    return np.array([[0., 0., 0.], [0., -opening/2, 0.], [0., opening/2, 0.],
        [-span/2, -opening/2, 0.], [span/2, -opening/2, 0.],
        [-span/2, opening/2, 0.], [span/2, opening/2, 0.]])


def project_points(points, projection):
    points = np.asarray(points, dtype=float)
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        return None
    uvw = points @ projection["matrix"].T
    uv = uvw[:, :2]/uvw[:, 2:3]
    if not np.isfinite(uv).all() or np.max(np.abs(uv)) > 1e6:
        return None
    return uv


def _label(canvas, text, pixel, colour, scale=.43):
    import cv2
    width, height = canvas.shape[1], canvas.shape[0]
    tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    x = int(np.clip(pixel[0], 3, max(3, width-tw-4)))
    y = int(np.clip(pixel[1], 15, max(15, height-5)))
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def project_local_points(points, projection, pose):
    points = np.asarray(points, dtype=float)
    if projection.get("kind") != "LOCAL_SURFACE":
        return project_points(points @ np.asarray(pose["rotation_camera"]).T+
                              np.asarray(pose["surface_anchor_xyz_mm"]), projection)
    # This fallback is valid on the observed surface plane only.
    if not np.isfinite(points).all() or np.any(np.abs(points[:, 2]) > 1e-6):
        return None
    matrix = projection["homography"]
    uvw = np.column_stack((points[:, :2], np.ones(len(points)))) @ matrix.T
    if np.any(uvw[:, 2]*matrix[2, 2] <= 1e-8):
        return None
    uv = uvw[:, :2]/uvw[:, 2:3]
    if not np.isfinite(uv).all() or np.max(np.abs(uv)) > 1e6:
        return None
    return uv


def _draw_on_target(canvas, pose, projection, top):
    import cv2
    local = grasp_symbol_geometry(pose)
    uv = project_local_points(local, projection, pose)
    if uv is None or not 4 <= np.linalg.norm(uv[1]-uv[2]) <= max(canvas.shape[:2])*1.2:
        return False
    pixels = np.rint(uv).astype(int)
    center, left, right = pixels[:3]
    # High-contrast parallel finger markers, directly beside the chosen mask.
    for i, j in ((3, 4), (5, 6)):
        cv2.line(canvas, tuple(pixels[i]), tuple(pixels[j]), (0, 0, 0), 8, cv2.LINE_AA)
        cv2.line(canvas, tuple(pixels[i]), tuple(pixels[j]), PINK, 4, cv2.LINE_AA)
    cv2.line(canvas, tuple(left), tuple(right), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(canvas, tuple(left), tuple(right), AXES[1][1], 2, cv2.LINE_AA)
    for end in (left, right):
        # Arrows point inward; their connecting line is the closing AXIS.
        tip = np.rint(center+.18*(end-center)).astype(int)
        cv2.arrowedLine(canvas, tuple(end), tuple(tip), AXES[1][1], 2,
                        cv2.LINE_AA, tipLength=.20)
    # Only a verified global model can show an off-plane approach arrow.
    pre = project_local_points([[0, 0, -min(30., pose["proposed_opening_mm"]*.5)]], projection, pose)
    if pre is not None and np.linalg.norm(pre[0]-center) >= 6:
        start = tuple(np.rint(pre[0]).astype(int))
        cv2.arrowedLine(canvas, start, tuple(center), AXES[2][1], 3, cv2.LINE_AA, tipLength=.20)
    else:
        # Symbol for approach at C; no invented screen direction for a normal
        # that could not be projected from the available measurements.
        cv2.circle(canvas, tuple(center), 10, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(canvas, tuple(center), 10, AXES[2][1], 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(center), 4, YELLOW, -1, cv2.LINE_AA)
    _label(canvas, "C", center+[-17, -12], YELLOW)
    # Keep the opening and angle beside the grasp, rather than in an inset.
    label_y = int(pixels[:, 1].max())+22
    if label_y+20 >= canvas.shape[0]:
        label_y = max(top+17, int(pixels[:, 1].min())-34)
    label_x = int(center[0])-90
    _label(canvas, f"Robotiq 2F-85 | w={pose['proposed_opening_mm']:.1f} mm",
           (label_x, label_y), PINK, .43)
    angle = pose.get("jaw_angle_plane_deg")
    reference = "floor" if pose.get("jaw_angle_reference") == "floor_plane" else "cam XY"
    if angle is not None:
        _label(canvas, f"jaw/{reference}={angle:.1f} deg", (label_x, label_y+19), AXES[1][1], .43)
    return True


def draw_gripper_overlay(canvas, frame, pose, target_mask=None, top=90):
    import cv2
    projection = display_projection(frame, pose, target_mask)
    drawn = projection is not None and _draw_on_target(canvas, pose, projection, top)
    if not drawn:
        pixel = tuple(map(int, pose["surface_anchor_uv"]))
        cv2.circle(canvas, pixel, 5, YELLOW, -1)
        _label(canvas, "GRASP PROJECTION UNVERIFIED", np.asarray(pixel)+[-100, 25], YELLOW)
    return {"object_overlay_drawn": bool(drawn), "inset_drawn": False,
            "projection_status": (projection.get("kind", "DISPLAY_FIT") if drawn else "UNVERIFIED"),
            "gripper_model": "ROBOTIQ_2F85_PARALLEL_JAW_SYMBOL",
            "robot_ready": False}
