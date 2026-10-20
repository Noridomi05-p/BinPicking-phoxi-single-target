"""Filter proposals, associate target evidence and export camera frame transforms."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .grasp_data import GraspTarget


def proposal_is_valid(row: np.ndarray) -> bool:
    if row.shape != (17,) or not np.isfinite(row).all():
        return False
    r = row[4:13].reshape(3, 3)
    return bool(row[1] > 0 and row[2] > 0 and row[3] > 0
                and np.allclose(r.T @ r, np.eye(3), atol=1e-3)
                and abs(np.linalg.det(r)-1.0) < 1e-3)


def filter_and_associate(
    proposals: np.ndarray, targets: list[GraspTarget],
    gripper: dict[str, Any], settings: dict[str, Any],
    observed_points_m: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    proposals = np.asarray(proposals, dtype=np.float64)
    if proposals.ndim != 2 or proposals.shape[1] != 17:
        raise ValueError("GraspNet decode phai tra mang N x 17.")
    usable = [t for t in targets if t.stats["eligible"] and len(t.points_m)]
    trees = [cKDTree(t.points_m) for t in usable]
    observed_tree = cKDTree(observed_points_m) if observed_points_m is not None and len(observed_points_m) else None
    rejected: Counter[str] = Counter()
    selected, associations = [], []
    candidate_counts: Counter[int] = Counter()
    order = np.argsort(-np.nan_to_num(proposals[:, 0], nan=-np.inf))
    for index in order:
        row = proposals[index]
        if not proposal_is_valid(row):
            rejected["INVALID_POSE"] += 1
            continue
        if row[0] < float(settings.get("min_score", .1)):
            rejected["LOW_GRASP_SCORE"] += 1
            continue
        command_width_mm = row[1]*1000.0 + float(gripper.get("opening_margin_mm", 4.0))
        if command_width_mm > float(gripper.get("max_opening_mm", 85.0)):
            rejected["TOO_WIDE_FOR_GRIPPER"] += 1
            continue
        if not trees:
            rejected["NO_ELIGIBLE_TARGET"] += 1
            continue
        origin, rotation = row[13:16], row[4:13].reshape(3, 3)
        distances = np.array([tree.query(origin)[0] for tree in trees])
        winner = int(np.argmin(distances))
        if distances[winner]*1000.0 > float(settings.get("seed_max_distance_mm", 10.0)):
            rejected["SEED_NOT_ON_TARGET"] += 1
            continue
        back = float(settings.get("closing_region_back_mm", 40.0)) / 1000.0
        radius = np.linalg.norm([max(back, row[3]), row[1]/2, row[2]/2]) + 1e-6
        supports = []
        for target, tree in zip(usable, trees):
            nearby = tree.query_ball_point(origin, radius)
            local = (target.points_m[nearby] - origin) @ rotation
            inside = ((local[:, 0] >= -back) & (local[:, 0] <= row[3])
                      & (np.abs(local[:, 1]) <= row[1]/2)
                      & (np.abs(local[:, 2]) <= row[2]/2))
            supports.append(int(inside.sum()))
        support, total = supports[winner], sum(supports)
        if observed_tree is not None:
            neighbours = observed_tree.query_ball_point(origin, radius)
            local = (observed_points_m[neighbours] - origin) @ rotation
            observed_inside = ((local[:, 0] >= -back) & (local[:, 0] <= row[3])
                               & (np.abs(local[:, 1]) <= row[1]/2)
                               & (np.abs(local[:, 2]) <= row[2]/2))
            # Unlabelled, rejected or eroded evidence also counts; it is not free space.
            total = max(total, int(observed_inside.sum()))
        if support < int(settings.get("min_target_support_points", 30)):
            rejected["TOO_FEW_TARGET_POINTS_BETWEEN_JAWS"] += 1
            continue
        purity = support / max(total, 1)
        if purity < float(settings.get("min_target_purity", .8)):
            rejected["MULTIPLE_TARGETS_BETWEEN_JAWS"] += 1
            continue
        target = usable[winner]
        if candidate_counts[target.instance_id] >= int(settings.get("candidates_per_object", 60)):
            rejected["TARGET_CANDIDATE_QUOTA"] += 1
            continue
        selected.append(row.copy())
        associations.append({"instance_id": target.instance_id,
                             "class_name": target.class_name,
                             "yolo_confidence": target.yolo_confidence,
                             "target_support_points": support,
                             "target_purity": purity,
                             "seed_distance_mm": float(distances[winner]*1000),
                             "suggested_opening_mm": float(command_width_mm)})
        candidate_counts[target.instance_id] += 1
        if len(selected) >= int(settings.get("max_candidates", 300)):
            break
    return np.asarray(selected, dtype=np.float64).reshape(-1, 17), associations, dict(rejected)


def diversify_per_object(
    rows: np.ndarray, associations: list[dict[str, Any]], settings: dict[str, Any]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Use jaw-symmetric rotation distance; independent quotas per target."""
    selected: list[int] = []
    per_object = int(settings.get("per_object", 5))
    translation_threshold = float(settings.get("nms_translation_mm", 15.0)) / 1000
    rotation_threshold = np.deg2rad(float(settings.get("nms_rotation_deg", 20.0)))
    symmetry = np.diag([1., -1., -1.])
    for i in np.argsort(-rows[:, 0]):
        same = [j for j in selected if associations[j]["instance_id"] == associations[i]["instance_id"]]
        if len(same) >= per_object:
            continue
        rotation = rows[i, 4:13].reshape(3, 3)
        duplicate = False
        for j in same:
            if np.linalg.norm(rows[i, 13:16] - rows[j, 13:16]) > translation_threshold:
                continue
            other = rows[j, 4:13].reshape(3, 3)
            cosine = max((np.trace(rotation.T @ other)-1)/2,
                         (np.trace(rotation.T @ other @ symmetry)-1)/2)
            if np.arccos(np.clip(cosine, -1, 1)) <= rotation_threshold:
                duplicate = True
                break
        if not duplicate:
            selected.append(int(i))
    return rows[selected].copy(), [associations[i] for i in selected]


def camera_grasp_record(row: np.ndarray, association: dict[str, Any]) -> dict[str, Any]:
    if not proposal_is_valid(row):
        raise ValueError("Khong xuat pose khong hop le.")
    transform = np.eye(4)
    transform[:3, :3] = row[4:13].reshape(3, 3)
    transform[:3, 3] = row[13:16]
    return {**association, "grasp_score": float(row[0]),
            "grasp_width_mm": float(row[1]*1000), "grasp_height_mm": float(row[2]*1000),
            "grasp_depth_mm": float(row[3]*1000),
            "translation_camera_mm": (row[13:16]*1000).tolist(),
            "rotation_camera_grasp": transform[:3, :3].tolist(),
            "T_camera_grasp_m": transform.tolist(),
            "translation_semantics": "GraspNet reference point, NOT calibrated robot TCP",
            "collision_model": "graspnet_generic_parallel_jaw",
            "robot_ready": False}
