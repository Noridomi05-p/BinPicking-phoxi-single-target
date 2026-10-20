"""Detailed work for one ranked object at a time; other scene points stay intact."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .geometry import fit_plane, plane_basis, rotation_for_jaws, toward_camera, unit
from .image_utils import erode_mask, resize_mask_nearest


def largest_surface_component(cloud, valid, jump_mm):
    count = int(valid.sum())
    if not count:
        raise ValueError("NO_VALID_3D")
    ids = np.full(valid.shape, -1, np.int32)
    ids[valid] = np.arange(count)
    sources, destinations = [], []
    for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        both = valid[a] & valid[b]
        near = np.linalg.norm(cloud[a][both]-cloud[b][both], axis=1) <= jump_mm
        sources.append(ids[a][both][near])
        destinations.append(ids[b][both][near])
    rows, cols = np.concatenate(sources), np.concatenate(destinations)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(count, count)).tocsr()
    number, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    chosen = int(np.argmax(sizes))
    selected = np.zeros_like(valid)
    selected[valid] = labels == chosen
    return selected, {"component_count": int(number), "retained_ratio": float(sizes[chosen]/count),
                      "discarded_for_geometry": int(count-sizes[chosen]),
                      "discarded_points_still_used_for_collision": True}


def surface_anchor(cloud, mask, core_fraction=.8):
    import cv2
    # Explicit border handles a mask touching all four image edges.
    distance = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    core = mask & (distance >= float(distance.max())*core_fraction)
    y, x = np.nonzero(core)
    if not len(x):
        raise ValueError("NO_CORE_POINT")
    points = cloud[core]
    robust_center = np.median(points, axis=0)
    index = int(np.argmin(np.sum((points-robust_center)**2, axis=1)))
    # Return one measured pixel exactly, not a component-wise synthetic median.
    return points[index].astype(np.float64), (int(x[index]), int(y[index])), float(distance[y[index], x[index]])


def local_surface_normal(frame, mask, anchor, settings):
    all_points = frame.point_cloud_mm[mask]
    local = np.linalg.norm(all_points-anchor, axis=1) <= float(settings.get("normal_radius_mm", 10.))
    points = all_points[local]
    minimum = int(settings.get("min_normal_points", 20))
    if len(points) < minimum:
        raise ValueError("NORMAL_TOO_FEW_LOCAL_POINTS")
    rms_limit = float(settings.get("max_normal_rms_mm", 1.5))
    fit = fit_plane(points, inlier_mm=rms_limit*2, min_ratio=.7, max_rms_mm=rms_limit)
    svd_normal = np.asarray(fit["normal"])
    source = "local_RANSAC_SVD"
    normal = svd_normal
    if frame.normal_map is not None and settings.get("normal_map_enabled", True):
        normals = frame.normal_map[mask][local].astype(float)
        lengths = np.linalg.norm(normals, axis=1)
        good = np.isfinite(normals).all(axis=1) & (lengths > 1e-8)
        normals = normals[good] / lengths[good, None]
        positions = points[good]
        normals[np.sum(normals * -positions, axis=1) < 0] *= -1
        cosine = np.cos(np.deg2rad(float(settings.get("normal_inlier_deg", 20.))))
        agree = normals @ svd_normal >= cosine
        if agree.sum() >= minimum and agree.mean() >= .7:
            normal = toward_camera(np.mean(normals[agree], axis=0), anchor)
            source = "NormalMap_checked_against_local_plane"
        elif len(normals):
            source = "local_RANSAC_SVD_NormalMap_disagrees"
    return normal, {"normal_source": source, "local_plane_rms_mm": fit["rms_mm"],
                    "local_normal_points": len(points)}


def generate_poses(frame, result, settings, measurement_settings, floor_plane=None):
    import cv2
    cloud = frame.point_cloud_mm
    mask = resize_mask_nearest(result.detection.mask, *cloud.shape[:2])
    if np.any(mask[[0, -1], :]) or np.any(mask[:, [0, -1]]):
        raise ValueError("MASK_TOUCHES_CROP_BOUNDARY")
    valid = mask & np.isfinite(cloud).all(axis=2) & (cloud[..., 2] > 0)
    valid &= cloud[..., 2] >= float(measurement_settings.get("min_z_mm", 0))
    valid &= cloud[..., 2] <= float(measurement_settings.get("max_z_mm", np.inf))
    confidence_min = measurement_settings.get("confidence_min")
    if confidence_min is not None:
        if frame.confidence_map is None:
            raise ValueError("CONFIDENCE_MAP_MISSING")
        conf = np.squeeze(frame.confidence_map)
        valid &= np.isfinite(conf) & (conf >= float(confidence_min))
    clean, stats = largest_surface_component(cloud, valid, float(settings.get("neighbor_jump_mm", 6.)))
    if stats["retained_ratio"] < float(settings.get("min_component_ratio", .7)):
        raise ValueError("MULTIPLE_3D_SURFACES")
    if clean.sum() < int(settings.get("min_points", 80)):
        raise ValueError("TOO_FEW_CLEAN_POINTS")
    core_mask = erode_mask(clean, int(measurement_settings.get("mask_erode_px", 3)))
    if core_mask.sum() < int(settings.get("min_normal_points", 20)):
        raise ValueError("CORE_TOO_THIN_OR_SPARSE")
    anchor, pixel, core_radius = surface_anchor(cloud, core_mask, float(settings.get("core_fraction", .8)))
    normal, normal_stats = local_surface_normal(frame, clean, anchor, settings)
    vertical = np.asarray(floor_plane["normal"]) if floor_plane else np.array([0., 0., -1.])
    tilt = float(np.rad2deg(np.arccos(np.clip(np.dot(normal, unit(vertical)), -1, 1))))
    if tilt > float(settings.get("max_tilt_deg", 25.)):
        raise ValueError("SURFACE_TILT_EXCEEDED")
    points = cloud[clean].astype(float)
    shape = settings.get("class_shapes", {}).get(result.detection.class_name, "unknown")
    # Local upper face for box orientation; full component remains the width
    # bound and collision evidence. Never quantile-trim to the nearest Z layer.
    orient_points = points
    if shape in ("box", "cube"):
        face = np.abs((points-anchor) @ normal) <= float(settings.get("plane_face_mm", 2.))
        if face.sum() < int(settings.get("min_points", 80)):
            raise ValueError("TOP_FACE_UNCERTAIN")
        orient_points = points[face]
    u, v = plane_basis(normal)
    planar = np.column_stack(((orient_points-anchor) @ u, (orient_points-anchor) @ v)).astype(np.float32)
    limit = int(settings.get("max_geometry_points", 12000))
    step = max(1, (len(planar) + limit - 1)//limit)
    # A strided NumPy view is not a continuous OpenCV point vector. In
    # particular, dense real scans used to fail inside convexHull here.
    sample = np.ascontiguousarray(planar[::step], dtype=np.float32)
    if len(sample) < 3 or not np.isfinite(sample).all():
        raise ValueError("INVALID_PLANAR_POINTS")
    rect = cv2.minAreaRect(sample)
    corners = cv2.boxPoints(rect)
    edges = [corners[1]-corners[0], corners[2]-corners[1]]
    edges.sort(key=np.linalg.norm)
    if np.linalg.norm(edges[0]) < .5:
        raise ValueError("DEGENERATE_VISIBLE_EXTENT")
    closing_directions = [unit(e[0]*u+e[1]*v) for e in edges]
    # Symmetry does not determine a unique yaw. Test more orientations rather
    # than turning an unstable PCA vector into a confident wrist rotation.
    ratio = float(np.linalg.norm(edges[1])/np.linalg.norm(edges[0]))
    if ratio < 1.15 or shape in ("cube", "cylinder", "unknown"):
        closing_directions.extend((unit(closing_directions[0]+closing_directions[1]),
                                   unit(closing_directions[0]-closing_directions[1])))
    poses = []
    reference_u, reference_v = plane_basis(vertical)
    zero_axis = "camera_x" if abs(unit(vertical)[0]) < .9 else "camera_y"
    for closing in closing_directions:
        rotation = rotation_for_jaws(normal, closing)
        projections = (points-anchor) @ closing
        visible_span = float(np.ptp(projections))
        centered_bound = float(2*np.max(np.abs(projections)))
        width = centered_bound + float(settings.get("width_uncertainty_mm", 3.))
        opening = width + 2*float(settings.get("opening_gap_mm", 2.))
        if width > float(settings.get("max_object_width_mm", 75.)) or opening > float(settings.get("max_opening_mm", 85.)):
            continue
        poses.append({
            "surface_anchor_xyz_mm": anchor.tolist(), "surface_anchor_uv": list(pixel),
            "surface_normal_camera": normal.tolist(), "rotation_camera": rotation.tolist(),
            "axis_convention": "columns: x=tangent, y=jaw_closing, z=approach",
            "translation_semantics": "measured surface anchor; NOT robot TCP or GraspNet reference point",
            "visible_width_mm": visible_span, "centered_width_bound_mm": centered_bound,
            "estimated_width_with_margin_mm": width, "proposed_opening_mm": opening,
            "full_object_width_known": False, "core_radius_px": core_radius,
            "tilt_deg": tilt, "tilt_reference": "floor_normal" if floor_plane else "camera_optical_axis",
            "jaw_angle_plane_deg": float(np.rad2deg(np.arctan2(closing @ reference_v, closing @ reference_u)) % 180.),
            "jaw_angle_reference": "floor_plane" if floor_plane else "camera_xy",
            "jaw_angle_zero_axis": zero_axis,
            "jaw_angle_semantics": "jaw axis modulo 180 deg; zero=jaw_angle_zero_axis projected into reference plane, positive toward cross(reference normal, zero); NOT robot Euler angles",
            "shape_hint": shape, "yaw_ambiguous": ratio < 1.15,
            "cylinder_contact_mode": "UNVERIFIED" if shape == "cylinder" else None,
            **stats, **normal_stats, "robot_ready": False})
    if not poses:
        raise ValueError("NO_ORIENTATION_WITHIN_OPENING_LIMIT")
    return poses
