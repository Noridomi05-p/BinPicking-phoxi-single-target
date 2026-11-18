"""Fail early for misspelled gates; never silently change threshold semantics."""
from __future__ import annotations

import math


SURFACE_KEYS = set("""enabled path capture_frames reference_min_valid_fraction
mask_erode_px min_mask_pixels min_valid_points min_height_mm noise_sigma_multiplier
height_percentile min_raised_ratio_mask min_raised_points min_valid_ratio min_z_mm
max_z_mm height_mode require_reference_identity floor_roi_path
plane_inlier_mm plane_min_inlier_ratio plane_max_rms_mm""".split())
PICKING_KEYS = set("""enabled max_candidates min_valid_ratio min_points require_plane
require_roi require_reference_identity max_frame_age_ms neighbor_jump_mm
min_component_ratio core_fraction normal_radius_mm min_normal_points
normal_inlier_deg max_normal_rms_mm max_tilt_deg plane_face_mm
opening_gap_mm width_uncertainty_mm max_object_width_mm max_opening_mm
max_geometry_points normal_map_enabled class_shapes clearance robot_target""".split())
CLEARANCE_KEYS = set("""enabled geometry_verified finger_thickness_mm finger_span_mm
finger_length_mm palm_span_mm palm_width_mm palm_depth_mm insertion_depth_mm
approach_mm safety_margin_mm vertical_motion_envelope_mm projection_max_error_px""".split())


def _keys(value, allowed, path):
    if not isinstance(value, dict):
        raise ValueError(f"{path} phai la mapping YAML.")
    unknown = set(value) - allowed
    if unknown:
        hint = " Doi min_raised_ratio thanh min_raised_ratio_mask (mau so = toan mask)." if "min_raised_ratio" in unknown else ""
        raise ValueError(f"Khoa khong hop le o {path}: {sorted(unknown)}.{hint}")


def _number(section, key, low=None, high=None, integer=False, nullable=False):
    if key not in section or (nullable and section[key] is None):
        return
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} phai la so huu han.")
    if integer and int(value) != value:
        raise ValueError(f"{key} phai la so nguyen.")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{key} ngoai khoang [{low}, {high}].")


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Cau hinh phai la mapping YAML.")
    for name in ("camera", "texture", "measurement", "postprocess", "runtime", "picking"):
        if name in config and not isinstance(config[name], dict):
            raise ValueError(f"{name} phai la mapping YAML.")
    surface = config.get("postprocess", {}).get("surface", {})
    _keys(surface, SURFACE_KEYS, "postprocess.surface")
    for key in ("min_raised_ratio_mask", "min_valid_ratio", "reference_min_valid_fraction", "plane_min_inlier_ratio"):
        _number(surface, key, 0, 1)
    for key in ("min_valid_points", "min_raised_points", "min_mask_pixels", "capture_frames"):
        _number(surface, key, 1, integer=True)
    _number(surface, "mask_erode_px", 0, integer=True)
    _number(surface, "height_percentile", 0, 100)
    for key in ("min_height_mm", "noise_sigma_multiplier", "plane_inlier_mm", "plane_max_rms_mm"):
        _number(surface, key, 0.000001)
    if surface.get("height_mode", "camera_z") not in ("camera_z", "plane"):
        raise ValueError("height_mode chi nhan camera_z hoac plane.")
    picking = config.get("picking", {})
    _keys(picking, PICKING_KEYS, "picking")
    robot = picking.get("robot_target", {})
    _keys(robot, set("enabled calibration_path tcp_name tcp_flange_xyzabc contact_center_flange_mm contact_reference_opening_mm fixed_opening_mm open_position_request grasp_depth_mm pregrasp_distance_mm rotation_proposal_tcp measurement_note".split()), "picking.robot_target")
    if "enabled" in robot and not isinstance(robot["enabled"], bool):
        raise ValueError("robot_target.enabled phai la true/false")
    for key in ("fixed_opening_mm", "contact_reference_opening_mm"):
        _number(robot, key, .001, 85)
    _number(robot, "open_position_request", 0, 255, integer=True)
    _number(robot, "grasp_depth_mm", 0, 100)
    _number(robot, "pregrasp_distance_mm", .001, 500)
    for key in ("min_valid_ratio", "min_component_ratio", "core_fraction"):
        _number(picking, key, 0.000001, 1)
    for key in ("max_candidates", "min_points", "min_normal_points", "max_geometry_points"):
        _number(picking, key, 1, integer=True)
    for key in ("normal_inlier_deg", "max_tilt_deg"):
        _number(picking, key, 0.000001, 89)
    for key in ("max_frame_age_ms", "neighbor_jump_mm", "normal_radius_mm", "max_normal_rms_mm", "plane_face_mm", "max_object_width_mm", "max_opening_mm"):
        _number(picking, key, 0.000001)
    for key in ("opening_gap_mm", "width_uncertainty_mm"):
        _number(picking, key, 0)
    _number(picking, "max_opening_mm", .000001, 85.)
    if picking.get("max_object_width_mm", 75.) > picking.get("max_opening_mm", 85.):
        raise ValueError("max_object_width_mm vuot max_opening_mm cua Robotiq 2F-85.")
    clearance = picking.get("clearance", {})
    _keys(clearance, CLEARANCE_KEYS, "picking.clearance")
    for key in CLEARANCE_KEYS - {"enabled", "geometry_verified"}:
        _number(clearance, key, 0 if key in ("safety_margin_mm", "vertical_motion_envelope_mm") else .000001, nullable=True)
    for section in (surface, picking, clearance):
        for key in ("enabled", "require_plane", "require_roi", "require_reference_identity", "normal_map_enabled", "geometry_verified"):
            if key in section and not isinstance(section[key], bool):
                raise ValueError(f"{key} phai la true/false, khong phai chuoi.")
    shapes = picking.get("class_shapes", {})
    if not isinstance(shapes, dict) or any(v not in ("box", "cube", "cylinder", "unknown") for v in shapes.values()):
        raise ValueError("class_shapes chi nhan box/cube/cylinder/unknown.")
    texture = config.get("texture", {})
    clahe = texture.get("clahe", {})
    _keys(clahe, {"enabled", "clip_limit", "tile_grid_size"}, "texture.clahe")
    _number(clahe, "clip_limit", .000001)
    if "enabled" in clahe and not isinstance(clahe["enabled"], bool):
        raise ValueError("texture.clahe.enabled phai la true/false.")
    _number(texture, "lower_percentile", 0, 100)
    _number(texture, "upper_percentile", 0, 100)
    if texture.get("lower_percentile", 1) >= texture.get("upper_percentile", 99.5):
        raise ValueError("lower_percentile phai nho hon upper_percentile.")
    grid = clahe.get("tile_grid_size", [8, 8])
    if not isinstance(grid, (list, tuple)) or len(grid) != 2 or any(type(v) is not int or v < 1 for v in grid):
        raise ValueError("tile_grid_size can 2 so nguyen duong.")
