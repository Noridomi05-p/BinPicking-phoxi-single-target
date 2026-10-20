from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Can cai opencv-python de ve ROI.") from exc
    return cv2


@dataclass
class PolygonROI:
    """Polygon ROI stored as normalized (x, y) coordinates in [0, 1]."""

    points_normalized: list[tuple[float, float]]

    def __post_init__(self) -> None:
        if len(self.points_normalized) < 3:
            raise ValueError("ROI can it nhat 3 diem.")
        cleaned: list[tuple[float, float]] = []
        for point in self.points_normalized:
            if len(point) != 2:
                raise ValueError("Moi diem ROI phai gom x,y.")
            x, y = float(point[0]), float(point[1])
            if not np.isfinite(x) or not np.isfinite(y):
                raise ValueError("Toa do ROI phai la so huu han.")
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                raise ValueError("Toa do ROI chuan hoa phai nam trong [0,1].")
            cleaned.append((x, y))
        self.points_normalized = cleaned

    @classmethod
    def from_pixels(
        cls,
        points: Sequence[Sequence[int | float]],
        image_shape: Sequence[int],
    ) -> "PolygonROI":
        height, width = int(image_shape[0]), int(image_shape[1])
        if height < 1 or width < 1:
            raise ValueError(f"Kich thuoc anh ROI khong hop le: {(height, width)}")
        x_scale = max(width - 1, 1)
        y_scale = max(height - 1, 1)
        normalized = [
            (
                float(np.clip(float(point[0]) / x_scale, 0.0, 1.0)),
                float(np.clip(float(point[1]) / y_scale, 0.0, 1.0)),
            )
            for point in points
        ]
        return cls(normalized)

    def points_pixels(self, image_shape: Sequence[int]) -> np.ndarray:
        height, width = int(image_shape[0]), int(image_shape[1])
        x_scale = max(width - 1, 1)
        y_scale = max(height - 1, 1)
        points = [
            (
                int(round(x * x_scale)),
                int(round(y * y_scale)),
            )
            for x, y in self.points_normalized
        ]
        return np.asarray(points, dtype=np.int32)

    def mask(self, image_shape: Sequence[int]) -> np.ndarray:
        height, width = int(image_shape[0]), int(image_shape[1])
        points = self.points_pixels((height, width))
        try:
            cv2 = _cv2()
        except RuntimeError:
            # Vectorised ray casting keeps non-GUI tests usable without OpenCV.
            yy, xx = np.mgrid[:height, :width]
            inside = np.zeros((height, width), dtype=bool)
            xj, yj = points[-1]
            for xi, yi in points:
                crosses = (yi > yy) != (yj > yy)
                denominator = float(yj - yi)
                if denominator == 0.0:
                    denominator = 1e-12
                edge_x = (xj - xi) * (yy - yi) / denominator + xi
                inside ^= crosses & (xx <= edge_x)
                xj, yj = xi, yi
            return inside
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [points], 1)
        return roi_mask.astype(bool)

    def evaluate_mask(self, object_mask: np.ndarray) -> tuple[float, bool]:
        binary = np.asarray(object_mask, dtype=bool)
        pixel_count = int(binary.sum())
        if binary.ndim != 2 or pixel_count == 0:
            return 0.0, False
        roi_mask = self.mask(binary.shape)
        overlap = float(np.logical_and(binary, roi_mask).sum() / pixel_count)
        # A concave mask's mean pixel can lie in a hole or another object.
        # This is only an ROI gate; the 3D grasp anchor is computed separately.
        from scipy.ndimage import distance_transform_edt
        ys, xs = np.nonzero(binary)
        y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
        distances = distance_transform_edt(np.pad(binary[y0:y1, x0:x1], 1))[1:-1, 1:-1]
        cy, cx = np.unravel_index(np.argmax(distances), distances.shape)
        center_x, center_y = int(cx+x0), int(cy+y0)
        return overlap, bool(roi_mask[center_y, center_x])

    def draw(
        self,
        image_bgr: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
        label: str = "ROI",
    ) -> np.ndarray:
        cv2 = _cv2()
        canvas = image_bgr.copy()
        points = self.points_pixels(canvas.shape[:2])
        cv2.polylines(canvas, [points], True, color, thickness, cv2.LINE_AA)
        anchor = tuple(int(value) for value in points[0])
        text_y = max(22, anchor[1] - 8)
        cv2.putText(
            canvas,
            label,
            (anchor[0], text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (anchor[0], text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        return canvas

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "coordinate_system": "normalized_xy",
            "points": [[float(x), float(y)] for x, y in self.points_normalized],
        }
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "PolygonROI":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        points = payload.get("points")
        if not isinstance(points, list):
            raise ValueError(f"File ROI khong co danh sach points: {source}")
        return cls([(float(point[0]), float(point[1])) for point in points])


def edit_polygon_roi(
    image_bgr: np.ndarray,
    initial_roi: PolygonROI | None = None,
    window_name: str = "Ve ROI long thung",
) -> PolygonROI | None:
    """Blocking OpenCV editor. Returns None when the edit is cancelled."""

    cv2 = _cv2()
    image = np.asarray(image_bgr, dtype=np.uint8)
    points: list[tuple[int, int]] = []
    closed = False
    if initial_roi is not None:
        points = [tuple(int(v) for v in point) for point in initial_roi.points_pixels(image.shape[:2])]
        closed = True

    def on_mouse(event, x, y, _flags, _userdata) -> None:
        nonlocal closed
        if event == cv2.EVENT_LBUTTONDOWN and not closed:
            points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and len(points) >= 3:
            closed = True

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while True:
            canvas = image.copy()
            overlay = canvas.copy()
            if len(points) >= 3 and closed:
                cv2.fillPoly(
                    overlay,
                    [np.asarray(points, dtype=np.int32)],
                    (0, 255, 255),
                )
                canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)
            if len(points) >= 2:
                cv2.polylines(
                    canvas,
                    [np.asarray(points, dtype=np.int32)],
                    closed,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            for index, point in enumerate(points):
                cv2.circle(canvas, point, 5, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    str(index + 1),
                    (point[0] + 7, point[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            instructions = [
                "Left click: them diem | Right click/Enter: dong ROI",
                "U: lui diem | C: ve lai | S: luu | Esc/Q: huy",
            ]
            for index, line in enumerate(instructions):
                origin = (12, 26 + index * 24)
                cv2.putText(
                    canvas,
                    line,
                    origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    line,
                    origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10) and len(points) >= 3:
                closed = True
            elif key in (ord("u"), 8, 127):
                if closed:
                    closed = False
                elif points:
                    points.pop()
            elif key == ord("c"):
                points.clear()
                closed = False
            elif key == ord("s") and closed and len(points) >= 3:
                return PolygonROI.from_pixels(points, image.shape[:2])
            elif key in (27, ord("q")):
                return None
    finally:
        cv2.setMouseCallback(window_name, lambda *_args: None)
        cv2.destroyWindow(window_name)
