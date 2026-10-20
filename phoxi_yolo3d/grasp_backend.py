"""Lazy adapter for an independently installed official GraspNet baseline."""
from __future__ import annotations

import importlib
import inspect
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


class GraspBackendError(RuntimeError):
    pass


def _configure_repo(repo: Path) -> None:
    if not (repo / "models" / "graspnet.py").is_file():
        raise GraspBackendError(f"Thieu baseline tai {repo}; xem CAI_DAT_GRASPNET.md.")
    for suffix in ("", "models", "utils", "pointnet2"):
        directory = str(repo / suffix)
        if directory not in sys.path:
            sys.path.insert(0, directory)


def _imports(repo: Path):
    _configure_repo(repo)
    try:
        import torch
        from graspnetAPI import GraspGroup
        network = importlib.import_module("graspnet")
        collision = importlib.import_module("collision_detector")
    except Exception as exc:
        raise GraspBackendError(
            "Khong import duoc GraspNet/PointNet++/KNN/GraspNetAPI. "
            "Chay scripts/14_check_grasp_env.py va doc huong dan moi truong. "
            f"Loi goc: {type(exc).__name__}: {exc}"
        ) from exc
    if not Path(network.__file__).resolve().is_relative_to(repo.resolve()):
        raise GraspBackendError("Module graspnet da import tu repo khac; chay script trong process moi.")
    return torch, GraspGroup, network, collision.ModelFreeCollisionDetector


def _command_version(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {"returncode": result.returncode,
                "output": (result.stdout + result.stderr).strip()[:6000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def environment_report(repo: Path, checkpoint: Path, device_name: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "repo": str(repo), "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "nvidia_smi": _command_version(["nvidia-smi"]),
        "nvcc": _command_version(["nvcc", "--version"]),
        "gcc": _command_version(["gcc", "--version"]),
        "imports_ok": False, "cuda_ok": False, "full_forward_tested": False,
    }
    try:
        import torch
        report["torch"] = {"version": torch.__version__, "built_cuda": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            device = torch.device(device_name)
            properties = torch.cuda.get_device_properties(device)
            probe = torch.ones((8, 8), device=device)
            _ = probe @ probe
            torch.cuda.synchronize(device)
            report["torch"].update(gpu=properties.name, total_vram_mib=properties.total_memory/1024**2,
                                   capability=list(torch.cuda.get_device_capability(device)),
                                   compiled_architectures=torch.cuda.get_arch_list())
            report["cuda_ok"] = True
        else:
            report["cuda_error"] = "torch.cuda.is_available() = False"
    except Exception as exc:
        report["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        _imports(repo)
        report["imports_ok"] = True
    except Exception as exc:
        report["import_error"] = str(exc)
    # Import or a CUDA matrix multiply alone does not validate custom extension kernels.
    report["ready_for_smoke_test"] = bool(report["cuda_ok"] and report["imports_ok"]
                                           and report["checkpoint_exists"])
    return report


class GraspNetBackend:
    def __init__(self, repo: Path, checkpoint: Path, settings: dict[str, Any]):
        self.torch, self.group_type, model_module, self.collision_type = _imports(repo)
        if not self.torch.cuda.is_available():
            raise GraspBackendError("Baseline nay can CUDA + extension; CPU chi ho tro --prepare-only.")
        name = str(settings.get("device", "cuda:0"))
        if not name.startswith("cuda"):
            raise GraspBackendError("device phai la cuda:N trong ban tich hop nay.")
        self.device = self.torch.device(name)
        if not checkpoint.is_file():
            raise GraspBackendError(f"Thieu pretrained checkpoint: {checkpoint}")
        self.net = model_module.GraspNet(
            input_feature_dim=0, num_view=int(settings.get("num_views", 300)),
            num_angle=12, num_depth=4, cylinder_radius=.05,
            hmin=-.02, hmax_list=[.01, .02, .03, .04], is_training=False,
        )
        # Explicit CPU load reduces startup VRAM; unpickle only trusted official weights.
        load_options: dict[str, Any] = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(self.torch.load).parameters:
            load_options["weights_only"] = True
        try:
            state = self.torch.load(str(checkpoint), **load_options)
            self.net.load_state_dict(state["model_state_dict"], strict=True)
            self.net.to(self.device).eval()
        except Exception as exc:
            raise GraspBackendError(f"Khong load duoc checkpoint/model: {exc}") from exc
        self.decode = model_module.pred_decode

    def predict(self, points_m: np.ndarray, colors_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        tensor = self.torch.from_numpy(np.ascontiguousarray(points_m[None], dtype=np.float32))
        self.torch.cuda.synchronize(self.device)
        self.torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        try:
            with self.torch.no_grad():
                outputs = self.net({"point_clouds": tensor.to(self.device), "cloud_colors": colors_rgb})
                decoded = self.decode(outputs)[0].detach().cpu().numpy()
            self.torch.cuda.synchronize(self.device)
        except RuntimeError as exc:
            raise GraspBackendError(
                "Forward CUDA that bai. Neu OOM, giam num_points xuong 16384/8192; "
                "neu extension/kernel loi, can build lai dung PyTorch/CUDA/architecture. "
                f"Loi goc: {exc}"
            ) from exc
        profile = {"inference_ms": (time.perf_counter()-start)*1000,
                   "cuda_peak_allocated_mib": self.torch.cuda.max_memory_allocated(self.device)/1024**2,
                   "cuda_peak_reserved_mib": self.torch.cuda.max_memory_reserved(self.device)/1024**2,
                   "gpu": self.torch.cuda.get_device_name(self.device),
                   "sampled_points": int(len(points_m)), "batch": 1,
                   "full_forward_tested": True}
        return np.asarray(decoded, dtype=np.float64).reshape(-1, 17), profile

    def collision_mask(self, rows: np.ndarray, scene_m: np.ndarray,
                       settings: dict[str, Any], opening_margin_mm: float) -> np.ndarray:
        if not len(rows):
            return np.zeros(0, dtype=bool)
        detector = self.collision_type(scene_m, voxel_size=float(settings["voxel_size_mm"])/1000)
        step = int(settings.get("chunk_size", 8))
        result = []
        for begin in range(0, len(rows), step):
            widened = rows[begin:begin+step].copy()
            widened[:, 1] += opening_margin_mm / 1000.0
            groups = self.group_type(widened)
            mask = detector.detect(groups,
                                   approach_dist=float(settings["approach_distance_mm"])/1000,
                                   collision_thresh=float(settings["threshold"]))
            result.extend(np.asarray(mask, dtype=bool).tolist())
        return np.asarray(result, dtype=bool)

    def show_3d(self, points_m: np.ndarray, colors_rgb: np.ndarray, rows: np.ndarray) -> None:
        import open3d as o3d
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points_m)
        cloud.colors = o3d.utility.Vector3dVector(colors_rgb)
        geometries = self.group_type(rows).to_open3d_geometry_list() if len(rows) else []
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
        print("3D preview: generic parallel jaw, NOT exact Robotiq mesh/TCP.")
        o3d.visualization.draw_geometries([cloud, axes, *geometries], window_name="Camera grasp proposals")
