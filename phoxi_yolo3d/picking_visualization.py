from __future__ import annotations

import numpy as np

from .picking import freshness
from .picking_overlay import draw_gripper_overlay
from .visualization import draw_results


def visible_results(results, picking, debug=False):
    if debug or not picking.get("enabled", False):
        return results
    return [r for r in results if r.detection.instance_id == picking.get("target_instance_id")]


def draw_picking(frame, results, rejected, picking, *, debug=False, roi=None, alpha=.22, fill_masks=True, live=False, max_age_ms=5000):
    """Keep the captured target visible; frame expiry only changes its viewing badge.

    The planner owns target validity. Redrawing a held image must neither erase
    its inspection result nor renew the capture timestamp or robot permission.
    """
    import cv2
    report = dict(picking)
    fresh = freshness(frame, max_age_ms) if live and report.get("enabled") else None
    canvas = draw_results(frame.texture_bgr, visible_results(results, report, debug),
        rejected_results=rejected, debug_rejected=debug, roi=roi, alpha=alpha, fill_masks=fill_masks)
    if not report.get("enabled"):
        return canvas
    pose = report.get("pose")
    overlay = None
    freshness_height = 20 if fresh is not None else 0
    if pose and report.get("target_instance_id") is not None:
        selected = next((r for r in results if r.detection.instance_id == report["target_instance_id"]), None)
        overlay = draw_gripper_overlay(canvas, frame, pose,
            target_mask=selected.detection.mask if selected else None, top=70+freshness_height)
    text = f"PICK {report.get('status')} | target={report.get('target_instance_id')} | CAMERA PROPOSAL"
    if not pose and report.get("reasons"):
        text += " | " + report["reasons"][0]
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], (70 if pose else 28) + freshness_height), (20, 20, 20), -1)
    cv2.putText(canvas, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 210, 255), 1, cv2.LINE_AA)
    if fresh is not None:
        age = "?" if fresh["age_ms"] is None else f"{fresh['age_ms']/1000:.1f}"
        if fresh["status"] == "FRESH":
            age_label = f"FRAME {age}s / {max_age_ms/1000:.1f}s | FRESH"
            age_colour = (150, 220, 150)
        else:
            age_label = f"FRAME {age}s | {fresh['status']} - VIEW ONLY | Space/N: recapture"
            age_colour = (0, 150, 255)
        cv2.putText(canvas, age_label, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, .45, age_colour, 1, cv2.LINE_AA)
    if pose:
        p = np.asarray(pose["surface_anchor_xyz_mm"])
        label = f"surface XYZ={p[0]:.1f},{p[1]:.1f},{p[2]:.1f} mm | opening={pose['proposed_opening_mm']:.1f} | tilt={pose['tilt_deg']:.1f}"
        cv2.putText(canvas, label, (8, 40 + freshness_height), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
        angle = pose.get("jaw_angle_plane_deg")
        reference = "floor" if pose.get("jaw_angle_reference") == "floor_plane" else "camera XY"
        angle_text = f"jaw/{reference}={angle:.1f} deg" if angle is not None else "jaw angle: see axes"
        legend = "Robotiq 2F-85 | Pink: jaws | Green: close"
        if overlay and not overlay['object_overlay_drawn']:
            legend = "2D projection unverified | S: save inspection"
        cv2.putText(canvas, angle_text+" | "+legend, (8, 62 + freshness_height),
                    cv2.FONT_HERSHEY_SIMPLEX, .40, (235, 200, 235), 1, cv2.LINE_AA)
    return canvas
