#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from phoxi_yolo3d.grasp_data import prepare_scene, sample_real_points
from phoxi_yolo3d.grasp_selection import camera_grasp_record, diversify_per_object, filter_and_associate
from phoxi_yolo3d.grasp_visualization import draw_grasp_diagnostics
from phoxi_yolo3d.storage import save_bgr_png, utc_stem


def resolve_project(path: str) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="GraspNet offline tren scan + masks PhoXi da luu")
    parser.add_argument("scan", help="Duong dan scans/frame.npz cua inspection")
    parser.add_argument("--config", default=str(ROOT / "config/grasp.yaml"))
    parser.add_argument("--masks", default=None)
    parser.add_argument("--result", default=None, help="result.json tu cung inspection (khong dung offline_results.json)")
    parser.add_argument("--roi", default=None, help="ROI json cua cung scan/crop")
    parser.add_argument("--instance-ids", type=int, nargs="+", default=None)
    parser.add_argument("--prepare-only", action="store_true", help="Kiem tra input; khong can torch/GPU/checkpoint")
    parser.add_argument("--allow-full-frame", action="store_true", help="Cho phep inference khi khong co ROI")
    parser.add_argument("--visualize", action="store_true", help="Mo Open3D sau khi inference")
    parser.add_argument("--repo", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).expanduser().read_text(encoding="utf-8"))
    validate_settings(settings)
    scan = Path(args.scan).expanduser().resolve()
    inspection = scan.parent.parent if scan.parent.name == "scans" else scan.parent
    masks = Path(args.masks).expanduser().resolve() if args.masks else inspection / "masks.npz"
    result = Path(args.result).expanduser().resolve() if args.result else inspection / "result.json"
    if not result.is_file():
        if args.result:
            parser.error(f"Khong co --result {result}")
        result = None
    roi = Path(args.roi).expanduser().resolve() if args.roi else inspection / "reference/bin_roi.json"
    if not roi.is_file():
        if args.roi:
            parser.error(f"Khong co --roi {roi}")
        roi = None
    if roi is None and not args.prepare_only and not args.allow_full_frame:
        parser.error("Thieu ROI: dung --roi <bin_roi.json> hoac cho phep ro rang --allow-full-frame.")
    started = time.perf_counter()
    scene = prepare_scene(scan, masks, settings["data"], result, roi, args.instance_ids)
    number = args.num_points if args.num_points is not None else int(settings["backend"]["num_points"])
    sampled, indices = sample_real_points(scene.network_points_m, number,
                                         int(settings["backend"].get("seed", 42)))
    rows = np.empty((0, 17), dtype=np.float64)
    associations: list[dict] = []
    report = {
        "mode": "PREPARE_ONLY" if args.prepare_only else "GRASPNET_CAMERA_PROPOSALS",
        "robot_ready": False, "coordinate_space": "camera", "point_units": "metre",
        "source_paths": scene.source_paths, "settings": settings,
        "sampled_points": number, "sampled_unique_points": int(len(np.unique(indices))),
        "workspace_points": len(scene.network_points_m),
        "observed_collision_points": len(scene.collision_points_m),
        "warnings": scene.warnings + [
            "Missing/unobserved depth is not free space.",
            "Generic jaw collision model is NOT the Robotiq 2F-85 CAD/linkage/mount.",
            "No hand-eye/base calibration, grasp-to-TCP calibration, robot IK or path check.",
        ],
        "targets": [{"instance_id": t.instance_id, "class_name": t.class_name,
                     "yolo_confidence": t.yolo_confidence, **t.stats} for t in scene.targets],
        "grasps": [], "collision_model": "graspnet_generic_parallel_jaw",
    }
    if not args.prepare_only:
        if not any(t.stats["eligible"] for t in scene.targets):
            report["warnings"].append("No eligible target: no network inference was run.")
        else:
            from phoxi_yolo3d.grasp_backend import GraspNetBackend
            repo = resolve_project(args.repo or settings["backend"]["repo"])
            checkpoint = resolve_project(args.checkpoint or settings["backend"]["checkpoint"])
            backend = GraspNetBackend(repo, checkpoint, settings["backend"])
            proposals, profile = backend.predict(sampled, scene.network_colors_rgb[indices])
            rows, associations, rejections = filter_and_associate(
                proposals, scene.targets, settings["gripper"], settings["selection"],
                scene.collision_points_m)
            collided = backend.collision_mask(rows, scene.collision_points_m, settings["collision"],
                                               float(settings["gripper"]["opening_margin_mm"]))
            rejections["GENERIC_JAW_COLLISION"] = int(collided.sum())
            associations = [a for a, bad in zip(associations, collided) if not bad]
            rows, associations = diversify_per_object(rows[~collided], associations, settings["selection"])
            report.update(backend_profile=profile, proposal_count=len(proposals), rejected_counts=rejections,
                          resolved_backend={"repo": str(repo), "checkpoint": str(checkpoint)})
    report["grasps"] = [camera_grasp_record(row, association) for row, association in zip(rows, associations)]
    report["total_ms"] = (time.perf_counter()-started)*1000
    output = Path(args.output).expanduser().resolve() if args.output else resolve_project(settings["output_dir"]) / utc_stem("grasp")
    # Avoid overwriting an inspection or a previous proposal run.
    output.mkdir(parents=True, exist_ok=False)
    (output / "grasp_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(output / "grasps.npy", rows)
    np.savez_compressed(output / "prepared_points.npz", sampled_points_m=sampled,
                        sampled_colors_rgb=scene.network_colors_rgb[indices],
                        **{f"target_{t.instance_id}_points_m": t.points_m for t in scene.targets},
                        **{f"target_{t.instance_id}_clean_mask": t.clean_mask for t in scene.targets})
    save_bgr_png(output / "grasp_preview.png", draw_grasp_diagnostics(scene, rows, associations))
    for target in scene.targets:
        status = "ELIGIBLE" if target.stats["eligible"] else ",".join(target.stats["reasons"])
        count = sum(a["instance_id"] == target.instance_id for a in associations)
        print(f"{target.instance_id} {target.class_name}: points={len(target.points_m)} {status}, grasps={count}")
    print(f"Da luu: {output}\nCAMERA PROPOSALS ONLY; robot_ready=False")
    if args.visualize and not args.prepare_only and "backend_profile" in report:
        backend.show_3d(scene.network_points_m, scene.network_colors_rgb, rows)
    return 0


def validate_settings(settings: dict) -> None:
    for group in ("backend", "data", "gripper", "selection", "collision"):
        if not isinstance(settings.get(group), dict):
            raise ValueError(f"Config thieu nhom {group}")
    positive = {"backend": ["num_points", "num_views"],
                "data": ["neighbor_jump_mm", "min_component_points", "min_target_points"],
                "gripper": ["max_opening_mm"],
                "selection": ["seed_max_distance_mm", "min_target_support_points", "closing_region_back_mm",
                              "max_candidates", "candidates_per_object", "per_object", "nms_translation_mm", "nms_rotation_deg"],
                "collision": ["voxel_size_mm", "approach_distance_mm", "threshold", "chunk_size"]}
    for group, keys in positive.items():
        for key in keys:
            if not np.isfinite(float(settings[group][key])) or float(settings[group][key]) <= 0:
                raise ValueError(f"{group}.{key} phai la so > 0")
    ratios = {"data": ["min_component_fraction", "min_retained_ratio", "min_valid_mask_ratio"],
              "selection": ["min_target_purity", "min_score"]}
    for group, keys in ratios.items():
        for key in keys:
            if not 0 <= float(settings[group][key]) <= 1:
                raise ValueError(f"{group}.{key} phai nam trong [0,1]")
    margin = float(settings["gripper"]["opening_margin_mm"])
    if not 0 <= margin < float(settings["gripper"]["max_opening_mm"]):
        raise ValueError("opening_margin_mm phai >= 0 va nho hon max_opening_mm")
    if float(settings["gripper"]["max_opening_mm"]) > 85:
        raise ValueError("Profile robotiq_2f85_standard_fingers gioi han 85 mm.")
    if int(settings["data"]["mask_erode_px"]) < 0:
        raise ValueError("mask_erode_px phai >= 0")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as exc:
        print(f"LOI: {exc}", file=sys.stderr)
        raise SystemExit(2)
