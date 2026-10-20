from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .types import InstanceResult


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Can cai opencv-python de hien thi ket qua.") from exc
    return cv2


def instance_color(class_name: str, instance_id: int) -> tuple[int, int, int]:
    digest = hashlib.sha256(f"{class_name}:{instance_id}".encode()).digest()
    # BGR, kept away from very dark colours.
    return tuple(int(70 + value % 170) for value in digest[:3])


def draw_results(
    image_bgr: np.ndarray,
    results: list[InstanceResult],
    alpha: float = 0.22,
    fill_masks: bool = True,
    rejected_results: list[InstanceResult] | None = None,
    debug_rejected: bool = False,
    roi: Any | None = None,
    status_lines: list[str] | None = None,
) -> np.ndarray:
    cv2 = _cv2()
    canvas = image_bgr.copy()
    if fill_masks and results:
        colour_layer = canvas.copy()
        for item in results:
            detection = item.detection
            colour = instance_color(detection.class_name, detection.instance_id)
            colour_layer[detection.mask] = colour
        canvas = cv2.addWeighted(colour_layer, alpha, canvas, 1.0 - alpha, 0)

    def put_lines(
        lines: list[str],
        anchor: tuple[int, int],
        colour: tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        x = int(np.clip(anchor[0], 4, max(4, canvas.shape[1] - 5)))
        first_y = int(np.clip(anchor[1], 20, max(20, canvas.shape[0] - 5)))
        for line_index, line in enumerate(lines):
            y = int(np.clip(first_y + line_index * 20, 20, canvas.shape[0] - 5))
            text_width = cv2.getTextSize(line, font, scale, 1)[0][0]
            draw_x = min(x, max(4, canvas.shape[1] - text_width - 4))
            cv2.putText(
                canvas,
                line,
                (draw_x, y),
                font,
                scale,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                line,
                (draw_x, y),
                font,
                scale,
                colour,
                1,
                cv2.LINE_AA,
            )

    def contour_and_anchor(mask: np.ndarray):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return contours, (4, 20)
        largest = max(contours, key=cv2.contourArea)
        top_index = int(np.argmin(largest[:, 0, 1]))
        top_x, top_y = (int(value) for value in largest[top_index, 0])
        if top_y >= 48:
            anchor = (top_x, top_y - 8)
        else:
            anchor = (top_x, top_y + 22)
        return contours, anchor

    for item in results:
        detection = item.detection
        measurement = item.measurement
        colour = instance_color(detection.class_name, detection.instance_id)
        contours, anchor = contour_and_anchor(detection.mask)
        cv2.drawContours(canvas, contours, -1, colour, 3, cv2.LINE_AA)
        if measurement.valid:
            measure_line = (
                f"Z={measurement.depth_z_median_mm:.1f} "
                f"D={measurement.distance_3d_median_mm:.1f} mm"
            )
        else:
            measure_line = (
                f"2D OK | 3D WAIT: {measurement.point_count} points "
                f"({measurement.valid_ratio:.1%} mask)"
            )
        lines = [
            f"#{detection.instance_id} {detection.class_name} {detection.score:.2f}",
            measure_line,
        ]
        post = item.postprocess
        if (post.surface_status == "PASS" and post.surface_height_max_mm is not None
                and post.surface_height_p99_mm is not None):
            lines.append(
                f"Hmax={post.surface_height_max_mm:.1f} "
                f"H99={post.surface_height_p99_mm:.1f} mm"
            )
        if post.surface_raised_ratio_mask is not None:
            lines.append(
                f"SURFACE {post.surface_status}: raised={post.surface_raised_ratio_mask:.1%} mask"
            )
        else:
            lines.append(f"SURFACE {post.surface_status}")
        if debug_rejected:
            lines.append(f"MEASURE: {measurement.point_count} pts | {measurement.reason or 'OK'}")
            if post.surface_mask_pixels:
                lines.append(
                    f"PAIRS={post.surface_valid_points}/{post.surface_mask_pixels} "
                    f"RAISED={post.surface_raised_points}"
                )
            if post.reason:
                lines.append(post.reason)
        put_lines(lines, anchor, colour)

    if debug_rejected:
        for item in rejected_results or []:
            detection = item.detection
            contours, anchor = contour_and_anchor(detection.mask)
            cv2.drawContours(canvas, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)
            reason = item.postprocess.reason or "REJECTED"
            debug_lines = [
                f"REJECT {detection.class_name} {detection.score:.2f}",
                reason,
            ]
            if item.postprocess.roi_overlap is not None:
                debug_lines.append(f"ROI={item.postprocess.roi_overlap:.2f}")
            if item.postprocess.surface_height_percentile_mm is not None:
                debug_lines.append(
                    f"H={item.postprocess.surface_height_percentile_mm:.1f} mm"
                )
            if item.postprocess.duplicate_of_instance_id is not None:
                debug_lines.append(f"same as #{item.postprocess.duplicate_of_instance_id}")
            put_lines(debug_lines, anchor, (0, 0, 255))

    if roi is not None:
        canvas = roi.draw(canvas)
    if status_lines:
        put_lines(status_lines, (12, 56), (255, 255, 255))
    return canvas


def depth_preview(depth_mm: np.ndarray | None, point_cloud_mm: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    # Show the exact source used by the 3D gates (camera-space point-cloud Z).
    depth = np.asarray(point_cloud_mm[:, :, 2], dtype=np.float32)
    valid = np.isfinite(point_cloud_mm).all(axis=2) & (depth > 0)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        low, high = np.percentile(depth[valid], [2, 98])
        if high > low:
            # Near objects are warm, far objects are cool after inversion.
            scaled = 255.0 - np.clip((depth - low) * 255.0 / (high - low), 0, 255)
            gray[valid] = scaled[valid].astype(np.uint8)
    colour = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colour[~valid] = 0
    return colour


def make_dashboard(
    annotated_bgr: np.ndarray,
    depth_bgr: np.ndarray,
    max_width: int = 1800,
    footer_lines: list[str] | None = None,
    max_height: int = 950,
) -> np.ndarray:
    cv2 = _cv2()
    if depth_bgr.shape[:2] != annotated_bgr.shape[:2]:
        depth_bgr = cv2.resize(
            depth_bgr,
            (annotated_bgr.shape[1], annotated_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    left = annotated_bgr.copy()
    right = depth_bgr.copy()
    dashboard = np.hstack([left, right])
    header = np.full((40, dashboard.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(header, "YOLO MASK + 3D", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(header, "POINTCLOUD Z | BLACK = NO 3D", (left.shape[1] + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    dashboard = np.vstack([header, dashboard])
    if footer_lines:
        footer = np.full((24 * len(footer_lines) + 12, dashboard.shape[1], 3), 25, dtype=np.uint8)
        for index, line in enumerate(footer_lines):
            cv2.putText(footer, line, (12, 24 * index + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        dashboard = np.vstack([dashboard, footer])
    scale = min(1.0, max_width / dashboard.shape[1], max_height / dashboard.shape[0])
    if scale < 1.0:
        dashboard = cv2.resize(
            dashboard,
            (int(round(dashboard.shape[1] * scale)), int(round(dashboard.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return dashboard
