"""Checkerboard pose from aligned texture + metric organized PhoXi XYZ.

No pinhole intrinsics are invented. At each subpixel corner, a robust local
affine fit interpolates the measured XYZ. A rigid metric fit estimates the
board pose; scale and 3D residual gates reject inaccurate / sparse boards.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from .math3d import points, rigid_fit


@dataclass(frozen=True)
class BoardSpec:
    cols: int = 15
    rows: int = 10
    square_mm: float = 16.0

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (self.cols,self.rows)):
            raise ValueError("Số góc phải là số nguyên.")
        if not 3 <= self.cols <= 40 or not 3 <= self.rows <= 40:
            raise ValueError("Số góc trong mỗi chiều phải từ 3 đến 40.")
        if not np.isfinite(self.square_mm) or self.square_mm <= 0:
            raise ValueError("Cạnh ô phải > 0 mm.")

    def model(self) -> np.ndarray:
        x, y = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
        return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))) * self.square_mm

    def to_dict(self):
        return {"type": "checkerboard", "counts": "inner_corners", **asdict(self),
                "origin": "physically_marked_first_inner_corner", "axes": "x=columns,y=rows,z=x_cross_y"}


@dataclass
class BoardDetection:
    corners_uv: np.ndarray
    corners_camera_mm: np.ndarray
    valid: np.ndarray
    inliers: np.ndarray
    T_camera_board: np.ndarray
    rms_mm: float
    max_error_mm: float
    measured_square_mm: float
    flipped: bool

    def to_dict(self):
        return {"T_camera_board": self.T_camera_board.tolist(), "rms_mm": self.rms_mm,
                "max_error_mm": self.max_error_mm, "measured_square_mm": self.measured_square_mm,
                "valid_corners": int(self.valid.sum()), "inlier_corners": int(self.inliers.sum()),
                "total_corners": len(self.valid), "origin_flipped_180": self.flipped,
                "method": "organized_xyz_local_affine_then_rigid_3d"}


def interpolate_corner(cloud, uv, radius=5, min_points=12, max_rms_mm=.5):
    """Interpolate XYZ at the exact 2D corner, never median XYZ of an offset patch."""
    h, w = cloud.shape[:2]
    u, v = map(float, uv)
    x0, x1 = max(0, int(np.floor(u)) - radius), min(w, int(np.floor(u)) + radius + 2)
    y0, y1 = max(0, int(np.floor(v)) - radius), min(h, int(np.floor(v)) + radius + 2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    xyz = cloud[y0:y1, x0:x1].reshape(-1, 3).astype(float)
    design = np.column_stack((xx.ravel() - u, yy.ravel() - v, np.ones(xx.size)))
    good = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
    if good.sum() < min_points:
        raise ValueError("Thiếu điểm 3D quanh góc bảng.")
    keep = good.copy()
    for _ in range(4):
        a, b = design[keep], xyz[keep]
        if len(a) < min_points or np.linalg.matrix_rank(a) < 3:
            raise ValueError("Điểm 3D quanh góc không đủ phân bố theo hai chiều.")
        coef = np.linalg.lstsq(a, b, rcond=None)[0]
        error = np.linalg.norm(xyz[good] - design[good] @ coef, axis=1)
        med = np.median(error)
        limit = max(.08, float(med + 3 * 1.4826 * np.median(np.abs(error - med))))
        next_keep = good.copy()
        next_keep[good] = error <= limit
        if np.array_equal(keep, next_keep):
            break
        keep = next_keep
    a, b = design[keep], xyz[keep]
    if len(a) < min_points or np.linalg.matrix_rank(a) < 3:
        raise ValueError("Không đủ điểm 3D sau loại nhiễu cục bộ.")
    # The corner must be bracketed, not extrapolated across a missing quadrant.
    if not (a[:, 0].min() < 0 < a[:, 0].max() and a[:, 1].min() < 0 < a[:, 1].max()):
        raise ValueError("Góc nằm ngoài vùng 3D quan sát được.")
    coef = np.linalg.lstsq(a, b, rcond=None)[0]
    rms = float(np.sqrt(np.mean(np.sum((b - a @ coef) ** 2, axis=1))))
    if rms > max_rms_mm:
        raise ValueError("3D quanh góc bảng không phẳng / nhiễu quá lớn.")
    return coef[2]


def estimate_board(frame, spec: BoardSpec, flipped=False, settings=None, corners_uv=None):
    import cv2
    settings = settings or {}
    frame.validate()
    if corners_uv is None:
        gray = cv2.cvtColor(frame.texture_bgr, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, (spec.cols, spec.rows), flags=flags)
        if not found:
            raise ValueError("Không thấy đủ góc checkerboard. Kiểm tra số GÓC TRONG, ánh sáng và toàn bảng.")
        uv = corners.reshape(-1, 2).astype(float)
    else:
        uv = np.asarray(corners_uv, float).reshape(-1, 2).copy()
    if len(uv) != spec.cols * spec.rows or not np.isfinite(uv).all():
        raise ValueError("Số góc ảnh không khớp cấu hình bảng.")
    if flipped:
        uv = uv[::-1].copy()
    xyz = np.full((len(uv), 3), np.nan)
    valid = np.zeros(len(uv), bool)
    for i, corner in enumerate(uv):
        try:
            xyz[i] = interpolate_corner(frame.point_cloud_mm, corner,
                radius=int(settings.get("corner_radius_px", 5)),
                max_rms_mm=float(settings.get("local_rms_max_mm", .5)))
            valid[i] = True
        except ValueError:
            continue
    minimum = max(6, min(12, len(uv)))
    if valid.sum() < minimum or valid.mean() < float(settings.get("min_3d_fraction", .70)):
        raise ValueError(f"Chỉ {valid.sum()}/{len(uv)} góc có 3D đủ tốt; cần ≥70%. Không điền 3D giả.")
    model = spec.model()
    indices = np.flatnonzero(valid)
    rng = np.random.default_rng(541)
    best = np.zeros(len(uv), bool)
    threshold = float(settings.get("corner_inlier_mm", 1.2))
    for _ in range(160):
        subset = rng.choice(indices, 3, replace=False)
        try:
            candidate = rigid_fit(model[subset], xyz[subset])
        except ValueError:
            continue
        error = np.linalg.norm(points(candidate, model[valid]) - xyz[valid], axis=1)
        keep = np.zeros(len(uv), bool)
        keep[valid] = error < threshold
        if keep.sum() > best.sum():
            best = keep
    if best.sum() < minimum or best.sum() / valid.sum() < .75:
        raise ValueError("Các góc 3D không tạo thành một bảng chuẩn cứng; kiểm tra bề mặt / kích thước ô.")
    t = rigid_fit(model[best], xyz[best])
    errors = np.linalg.norm(points(t, model[best]) - xyz[best], axis=1)
    rms = float(np.sqrt(np.mean(errors ** 2)))
    if rms > float(settings.get("board_rms_max_mm", 1.0)):
        raise ValueError(f"Sai số fit bảng {rms:.3f} mm quá lớn.")
    # Use physically adjacent corners, not a fitted similarity scale.
    square_lengths = []
    for row in range(spec.rows):
        for col in range(spec.cols):
            i = row * spec.cols + col
            for j in ([i + 1] if col + 1 < spec.cols else []) + ([i + spec.cols] if row + 1 < spec.rows else []):
                if best[i] and best[j]:
                    square_lengths.append(np.linalg.norm(xyz[i] - xyz[j]))
    if not square_lengths:
        raise ValueError("Không đủ cạnh ô để kiểm tra kích thước bảng.")
    measured = float(np.median(square_lengths))
    if abs(measured / spec.square_mm - 1) > float(settings.get("square_relative_tolerance", .03)):
        raise ValueError(f"Cạnh ô đo 3D khoảng {measured:.2f} mm, cấu hình {spec.square_mm:.2f} mm. Kiểm tra bản in.")
    return BoardDetection(uv, xyz, valid, best, t, rms, float(errors.max()), measured, bool(flipped))


def overlay(frame, detection: BoardDetection | None, spec: BoardSpec):
    import cv2
    image = frame.texture_bgr.copy()
    if detection is None:
        return image
    uv = np.rint(detection.corners_uv).astype(int)
    h, w = image.shape[:2]
    radius = max(2, round(w / 700))
    for i, p in enumerate(uv):
        color = (80, 220, 80) if detection.inliers[i] else (20, 90, 240)
        cv2.circle(image, tuple(p), radius, color, -1, cv2.LINE_AA)
    origin = uv[0]
    cv2.circle(image, tuple(origin), radius * 4, (0, 220, 255), 2, cv2.LINE_AA)
    for index, label, color in ((min(3, spec.cols - 1), 'X', (60, 80, 255)),
                                (min(3, spec.rows - 1) * spec.cols, 'Y', (80, 230, 90))):
        cv2.arrowedLine(image, tuple(origin), tuple(uv[index]), color, max(2, radius), cv2.LINE_AA, tipLength=.2)
        xy = np.clip(uv[index] + [8, -8], [0, 15], [w - 30, h - 10])
        cv2.putText(image, label, tuple(xy), cv2.FONT_HERSHEY_SIMPLEX, max(.5, w / 2000), color, 2)
    return image
