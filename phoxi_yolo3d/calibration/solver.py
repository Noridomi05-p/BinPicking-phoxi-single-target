"""Joint eye-to-hand calibration: F_i Y = X C_i.

F_i=T_base_flange, C_i=T_camera_board, X=T_base_camera,
Y=T_flange_board. OpenCV hand-eye initializers receive inverse(F_i), NOT F_i.
Absolute SE(3) consistency is then refined with a robust least-squares loss.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .math3d import check_transform, inverse, pose_error, transform


def rotation_coverage(poses):
    rotations = np.array([check_transform(t)[:3, :3] for t in poses])
    vectors = np.array([Rotation.from_matrix(a.T @ b).as_rotvec()
                        for a, b in combinations(rotations, 2)])
    s = np.linalg.svd(vectors, compute_uv=False) / np.sqrt(max(1, len(vectors)))
    s = np.pad(s, (0, max(0, 3 - len(s))))
    return {"independent_axis_rms_deg": np.rad2deg(s).tolist(),
            "max_pair_rotation_deg": float(np.rad2deg(np.linalg.norm(vectors, axis=1).max()))}


def _mean_transform(values):
    values = np.asarray(values)
    rotations = Rotation.from_matrix(values[:, :3, :3])
    matrices = rotations.as_matrix()
    distances = np.array([[Rotation.from_matrix(a.T @ b).magnitude() for b in matrices] for a in matrices])
    medoid = np.argmin(np.median(distances, axis=1))
    errors = distances[medoid]
    keep = errors <= max(np.deg2rad(5), float(np.median(errors) * 3 + 1e-7))
    return transform(Rotation.from_matrix(matrices[keep]).mean().as_matrix(),
                     np.median(values[keep, :3, 3], axis=0))


def _pack(x, y):
    return np.r_[Rotation.from_matrix(x[:3, :3]).as_rotvec(), x[:3, 3],
                 Rotation.from_matrix(y[:3, :3]).as_rotvec(), y[:3, 3]]


def _unpack(p):
    return (transform(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:6]),
            transform(Rotation.from_rotvec(p[6:9]).as_matrix(), p[9:12]))


def _residual(p, f, c, rotation_weight_mm):
    x, y = _unpack(p)
    left, right = f @ y, x @ c
    delta_r = np.swapaxes(left[:, :3, :3], 1, 2) @ right[:, :3, :3]
    rotation_error = Rotation.from_matrix(delta_r).as_rotvec() * rotation_weight_mm
    return np.column_stack((left[:, :3, 3] - right[:, :3, 3], rotation_error)).ravel()


def _rows(samples, x, y):
    rows = []
    for s in samples:
        if not s.get("enabled", True):
            continue
        left = check_transform(s["T_base_flange"]) @ y
        right = x @ check_transform(s["T_camera_board"])
        mm, deg = pose_error(left, right)
        rows.append({"id": s["id"], "role": s["role"], "position_error_mm": mm,
                     "rotation_error_deg": deg,
                     "position_delta_base_mm": (right[:3, 3] - left[:3, 3]).tolist()})
    return rows


def _summary(rows):
    if not rows:
        return {"count": 0, "position_rms_mm": None, "position_max_mm": None,
                "rotation_rms_deg": None, "rotation_max_deg": None}
    mm = np.array([r["position_error_mm"] for r in rows])
    deg = np.array([r["rotation_error_deg"] for r in rows])
    return {"count": len(rows), "position_rms_mm": float(np.sqrt(np.mean(mm ** 2))),
            "position_max_mm": float(mm.max()), "rotation_rms_deg": float(np.sqrt(np.mean(deg ** 2))),
            "rotation_max_deg": float(deg.max())}


def solve(samples, settings=None):
    import cv2
    settings = settings or {}
    fit = [s for s in samples if s.get("enabled", True) and s["role"] == "fit"]
    if len(fit) < int(settings.get("min_fit_samples", 6)):
        raise ValueError("Cần ít nhất 6 mẫu TÍNH; đề xuất 15–20 mẫu và ≥5 mẫu kiểm chứng riêng.")
    f = np.array([check_transform(s["T_base_flange"]) for s in fit])
    c = np.array([check_transform(s["T_camera_board"]) for s in fit])
    coverage = rotation_coverage(f)
    if coverage["independent_axis_rms_deg"][1] < float(settings.get("min_second_axis_deg", 3)):
        raise ValueError("Hướng robot thiếu biến thiên quanh hai trục độc lập. Thu thêm mẫu có góc nghiêng khác nhau.")
    fi = np.array([inverse(t) for t in f])
    candidates = []
    weight = float(settings.get("rotation_weight_mm", 100.0))
    rng = np.random.default_rng(309)
    trials = [(np.arange(len(f)), method) for method in
              (cv2.CALIB_HAND_EYE_PARK, cv2.CALIB_HAND_EYE_HORAUD, cv2.CALIB_HAND_EYE_TSAI)]
    if len(f) >= 9:
        trials += [(rng.choice(len(f), 6, replace=False), cv2.CALIB_HAND_EYE_PARK) for _ in range(12)]
    for index, method in trials:
        try:
            r, t = cv2.calibrateHandEye(list(fi[index, :3, :3]), list(fi[index, :3, 3]),
                                      list(c[index, :3, :3]), list(c[index, :3, 3]), method=method)
            x = transform(r, np.asarray(t).ravel())
            y = _mean_transform(fi @ x @ c)
            p = _pack(x, y)
            errors = _residual(p, f, c, weight).reshape(-1, 6)
            candidates.append((float(np.median(np.linalg.norm(errors, axis=1))), p))
        except (ValueError, np.linalg.LinAlgError, cv2.error):
            continue
    if not candidates:
        raise ValueError("Không khởi tạo được calibration. Kiểm tra ZYZ, pose mặt bích / BASE và gốc bảng.")
    p0 = min(candidates, key=lambda pair: pair[0])[1]
    optimization = least_squares(_residual, p0, args=(f, c, weight), loss="soft_l1",
                                 f_scale=float(settings.get("robust_scale_mm", 1.0)),
                                 x_scale=np.r_[np.full(3, .2), np.full(3, 200.),
                                               np.full(3, .2), np.full(3, 100.)],
                                 max_nfev=1000, ftol=1e-11, xtol=1e-11, gtol=1e-10)
    if not optimization.success:
        raise ValueError("Bộ tối ưu chưa hội tụ; không xuất ma trận để sử dụng.")
    # Scale columns so rotation and translation parameters have comparable units.
    jac = optimization.jac * np.r_[np.full(3, .01), np.ones(3), np.full(3, .01), np.ones(3)]
    singular = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-7))
    if rank < 12:
        raise ValueError("Dữ liệu suy biến: ma trận chưa xác định duy nhất. Thu thêm tư thế đa hướng.")
    x, y = _unpack(optimization.x)
    rows = _rows(samples, x, y)
    fit_summary = _summary([r for r in rows if r["role"] == "fit"])
    validation = _summary([r for r in rows if r["role"] == "validation"])
    pos_limit = float(settings.get("max_position_error_mm", 2.0))
    rot_limit = float(settings.get("max_rotation_error_deg", 1.0))
    min_validation = int(settings.get("min_validation_samples", 5))
    issues = []
    for row in rows:
        row["review_required"] = row["position_error_mm"] > pos_limit or row["rotation_error_deg"] > rot_limit
    if any(r["review_required"] for r in rows if r["role"] == "fit"):
        issues.append("FIT_RESIDUAL_EXCEEDS_USER_LIMIT")
    if validation["count"] < min_validation:
        issues.append("MORE_INDEPENDENT_VALIDATION_REQUIRED")
    if any(r["review_required"] for r in rows if r["role"] == "validation"):
        issues.append("VALIDATION_RESIDUAL_EXCEEDS_USER_LIMIT")
    return {"schema_version": 1, "method": "eye_to_hand_3d_board_joint_se3",
            "equation": "T_base_flange[i] @ T_flange_board = T_base_camera @ T_camera_board[i]",
            "distance_unit": "millimetre", "robot_angle_unit": "degree", "robot_euler": "intrinsic_ZYZ",
            "robot_reference": "flange", "robot_frame": "BASE", "coordinate_space": "CameraSpace/PrimaryCamera",
            "T_base_camera": x.tolist(), "T_flange_board": y.tolist(),
            "fit": fit_summary, "validation": validation, "residuals": rows,
            "coverage": coverage, "jacobian_rank": rank,
            "quality_limits": {"position_max_mm": pos_limit, "rotation_max_deg": rot_limit,
                               "min_validation_samples": min_validation},
            "status": "VERIFIED_WITHIN_CONFIGURED_LIMITS" if not issues else "REVIEW_REQUIRED",
            "issues": issues, "robot_ready": False}
