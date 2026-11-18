"""All transforms use column vectors; T_A_B maps coordinates B -> A, in mm."""
from __future__ import annotations

import warnings
import numpy as np
from scipy.spatial.transform import Rotation


def check_transform(value) -> np.ndarray:
    t = np.asarray(value, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("Ma trận phải là 4×4 và chỉ chứa số hữu hạn.")
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("Hàng cuối ma trận phải là [0, 0, 0, 1].")
    if not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(t[:3, :3]), 1, atol=1e-5
    ):
        raise ValueError("Phần quay phải thuộc SO(3), không phản chiếu / đổi tỉ lệ.")
    return t


def transform(rotation=None, translation=None) -> np.ndarray:
    t = np.eye(4)
    if rotation is not None:
        t[:3, :3] = rotation
    if translation is not None:
        t[:3, 3] = translation
    return check_transform(t)


def inverse(t) -> np.ndarray:
    t = check_transform(t)
    r = t[:3, :3].T
    return transform(r, -r @ t[:3, 3])


def pose_zyz(values) -> np.ndarray:
    """Doosan intrinsic Z-Y'-Z'', millimetres / degrees (not XYZ / RPY)."""
    p = np.asarray(values, dtype=float)
    if p.shape != (6,) or not np.isfinite(p).all():
        raise ValueError("Pose cần đúng 6 số hữu hạn: X Y Z A B C.")
    return transform(Rotation.from_euler("ZYZ", p[3:], degrees=True).as_matrix(), p[:3])


def as_pose_zyz(t) -> list[float]:
    t = check_transform(t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # ZYZ gimbal lock: equivalent angles.
        angles = Rotation.from_matrix(t[:3, :3]).as_euler("ZYZ", degrees=True)
    return np.r_[t[:3, 3], angles].tolist()


def parse_pose(text: str) -> list[float]:
    import re
    tokens = re.split(r"[\s,;]+", text.strip().strip("[]()"))
    try:
        values = [float(s) for s in tokens]
    except ValueError as exc:
        raise ValueError("Nhập 6 số, ngăn bằng khoảng trắng; dùng dấu chấm thập phân.") from exc
    pose_zyz(values)
    return values


def points(t, xyz) -> np.ndarray:
    t = check_transform(t)
    p = np.asarray(xyz, dtype=float)
    return p @ t[:3, :3].T + t[:3, 3]


def rigid_fit(source, target) -> np.ndarray:
    a, b = np.asarray(source, float), np.asarray(target, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 3:
        raise ValueError("Ghép cặp 3D cần ít nhất 3 điểm tương ứng.")
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    u, s, vt = np.linalg.svd((a - ac).T @ (b - bc))
    if s[1] < 1e-9:
        raise ValueError("Các điểm thẳng hàng; không xác định được tư thế bảng.")
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ correction @ u.T
    return transform(r, bc - r @ ac)


def pose_error(a, b) -> tuple[float, float]:
    a, b = check_transform(a), check_transform(b)
    return (float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            float(np.rad2deg(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())))
