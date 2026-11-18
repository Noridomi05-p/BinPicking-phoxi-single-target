"""Rank cheaply, evaluate one candidate at a time, expose one final proposal."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
import numpy as np

from .clearance import check_approach, fit_projection
from .geometry import identity_status
from .target_geometry import generate_poses


def freshness(frame, max_age_ms, offline=False):
    now = datetime.now(timezone.utc)
    stamp = frame.info.get("capture_started_utc", frame.timestamp_utc)
    result = {"evaluated_at_utc": now.isoformat(), "captured_at_utc": stamp,
              "valid_until_utc": None, "age_ms": None, "status": "UNKNOWN"}
    try:
        captured = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if captured.tzinfo is None:
            raise ValueError("timestamp needs timezone")
        age = (now-captured).total_seconds()*1000
        result.update(age_ms=age, valid_until_utc=(captured+timedelta(milliseconds=max_age_ms)).isoformat(),
                      status="FRESH" if 0 <= age <= max_age_ms else "FUTURE_CLOCK" if age < 0 else "STALE")
    except (ValueError, TypeError, OverflowError):
        pass
    if offline:
        result["status"] = "OFFLINE_REPLAY"
    return result


def plan_single_target(frame, results, settings, measurement_settings, postprocessor=None, offline=False):
    import cv2
    robot_settings = settings.get("robot_target", {})
    if robot_settings.get("enabled", False):
        settings = dict(settings)
        from .robot_target import scalar
        settings["max_opening_mm"] = min(float(settings.get("max_opening_mm", 85)),
                                         scalar(robot_settings, "fixed_opening_mm", 1, 85))
    started = time.perf_counter()
    output = {"enabled": True, "status": "NO_TARGET", "target_instance_id": None,
              "pose": None, "ranking": [], "attempts": [], "robot_ready": False,
              "coordinate_space": "camera", "distance_unit": "millimetre",
              "backend": "deterministic_surface_geometry",
              "automatic_mask_merge": False,
              "freshness": freshness(frame, float(settings.get("max_frame_age_ms", 5000)), offline)}
    reference = getattr(postprocessor, "surface_reference", None)
    plane = reference.plane if reference else None
    output["reference"] = {"identity_status": identity_status(reference.identity, frame) if reference else "MISSING",
                           "floor_plane_available": plane is not None}
    reasons = []
    if settings.get("require_roi", True) and (getattr(postprocessor, "roi", None) is None
                                             or not getattr(postprocessor, "roi_enabled", False)):
        reasons.append("ROI_MISSING")
    if settings.get("require_plane", True) and plane is None:
        reasons.append("FLOOR_PLANE_MISSING")
    if settings.get("require_reference_identity", True) and (
            reference is None or identity_status(reference.identity, frame) != "MATCH"):
        reasons.append("REFERENCE_IDENTITY_UNVERIFIED")
    if not offline and output["freshness"]["status"] != "FRESH":
        reasons.append("FRAME_"+output["freshness"]["status"])
    if reasons:
        output.update(reasons=reasons, timings_ms={"total": (time.perf_counter()-started)*1000})
        return output
    candidates = []
    for item in results:
        m, post = item.measurement, item.postprocess
        eligibility = []
        if not item.verified_3d:
            eligibility.append("NOT_3D_VERIFIED")
        if m.point_count < int(settings.get("min_points", 80)):
            eligibility.append("TOO_FEW_POINTS")
        if m.valid_ratio < float(settings.get("min_valid_ratio", .5)):
            eligibility.append("LOW_3D_COVERAGE")
        # Normalize only for ranking; it is not a probability of grasp success.
        height = post.surface_height_percentile_mm or 0.
        score = .45*min(1., m.valid_ratio) + .35*float(np.clip(height/100., 0, 1)) + .20*item.detection.score
        row = {"instance_id": item.detection.instance_id, "class_name": item.detection.class_name,
               "pick_score": float(score), "height_percentile_mm": float(height),
               "height_mode": post.surface_height_mode, "valid_ratio": m.valid_ratio,
               "eligible": not eligibility, "reasons": eligibility,
               "full_object_integrity": "UNVERIFIED"}
        output["ranking"].append(row)
        if not eligibility:
            candidates.append((score, item))
    candidates.sort(key=lambda pair: (-pair[0], pair[1].detection.instance_id))
    output["ranking"].sort(key=lambda row: (-row["pick_score"], row["instance_id"]))
    ranked_at = time.perf_counter()
    fallback = None
    projection = None
    projection_done = False
    clearance_settings = settings.get("clearance", {})
    for score, item in candidates[:int(settings.get("max_candidates", 3))]:
        attempt = {"instance_id": item.detection.instance_id, "pick_score": float(score), "poses": []}
        output["attempts"].append(attempt)
        try:
            poses = generate_poses(frame, item, settings, measurement_settings, plane)
        except ValueError as exc:
            attempt["reason"] = str(exc)
            continue
        except cv2.error as exc:
            # Reject the candidate, never fabricate a pose or reuse the
            # previous frame when an OpenCV geometry operation fails.
            attempt.update(reason="OPENCV_GEOMETRY_ERROR", detail=str(exc))
            continue
        if clearance_settings.get("geometry_verified", False) and not projection_done:
            projection = fit_projection(frame, float(clearance_settings.get("projection_max_error_px", .5)))
            projection_done = True
        for pose in poses:
            if robot_settings.get("enabled", False):
                pose["required_opening_mm"] = pose["proposed_opening_mm"]
                pose["proposed_opening_mm"] = float(robot_settings["fixed_opening_mm"])
            pose["clearance"] = check_approach(frame, pose["surface_anchor_xyz_mm"], pose["rotation_camera"],
                                              pose["proposed_opening_mm"], clearance_settings, projection)
            attempt["poses"].append(pose)
            status = pose["clearance"]["status"]
            if status == "BLOCKED":
                continue
            if fallback is None:
                fallback = (item.detection.instance_id, pose)
            if status == "OBSERVED_APPROACH_CLEAR":
                output.update(status="APPROACH_CHECKED_PROPOSAL", target_instance_id=item.detection.instance_id, pose=pose)
                break
        if output["target_instance_id"] is not None:
            break
        if fallback and fallback[1]['clearance'].get('reason') in (
                'GRIPPER_GEOMETRY_UNCONFIGURED', 'CLEARANCE_DISABLED', 'PROJECTION_UNVERIFIED'):
            # These are frame/model limitations shared by every target. Keep
            # one preview instead of repeating expensive geometry pointlessly.
            break
    if output["target_instance_id"] is None and fallback:
        output.update(status="PREVIEW_UNVALIDATED", target_instance_id=fallback[0], pose=fallback[1])
    # Heavy processing itself may consume the frame's time budget.
    output["freshness"] = freshness(frame, float(settings.get("max_frame_age_ms", 5000)), offline)
    if not offline and output["freshness"]["status"] != "FRESH":
        output.update(status="STALE_PROPOSAL", target_instance_id=None, pose=None)
    output["timings_ms"] = {"rank": (ranked_at-started)*1000,
                            "geometry_and_clearance": (time.perf_counter()-ranked_at)*1000,
                            "total": (time.perf_counter()-started)*1000}
    if robot_settings.get("enabled", False):
        from .robot_target import attach_preview
        return attach_preview(frame, output, robot_settings)
    return output
