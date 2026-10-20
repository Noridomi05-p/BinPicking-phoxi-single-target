from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .image_utils import erode_mask, resize_mask_nearest
from .types import Frame3D
from .geometry import fit_plane, frame_identity, identity_status


def frame_depth_mm(frame: Frame3D) -> np.ndarray:
    """Return an aligned 2D camera-Z map in millimetres."""

    # PointCloud[:, :, 2] is explicitly camera-space Z. Using it avoids any
    # ambiguity between axial depth and Euclidean range in vendor DepthMap data.
    depth = np.asarray(frame.point_cloud_mm[:, :, 2], dtype=np.float32)
    if depth.shape != frame.texture_bgr.shape[:2]:
        raise ValueError(
            "Depth khong thang hang voi texture: "
            f"{depth.shape} != {frame.texture_bgr.shape[:2]}"
        )
    return depth


@dataclass
class SurfaceCheck:
    valid: bool
    reason: str | None
    valid_ratio: float
    raised_ratio: float | None = None
    height_percentile_mm: float | None = None
    height_max_mm: float | None = None
    height_p99_mm: float | None = None
    height_threshold_mm: float | None = None
    status: str = "UNKNOWN"
    mask_pixels: int = 0
    valid_points: int = 0
    raised_points: int = 0
    raised_ratio_mask: float | None = None
    height_mode: str = "camera_z"
    reference_status: str = "UNBOUND"


@dataclass
class SurfaceReference:
    """Median empty-bin depth and its robust temporal noise, in millimetres."""

    depth_mm: np.ndarray
    noise_sigma_mm: np.ndarray
    valid_count: np.ndarray
    frame_count: int
    created_utc: str
    identity: dict = field(default_factory=dict)
    plane: dict | None = None
    plane_error: str | None = None

    def __post_init__(self) -> None:
        self.depth_mm = np.asarray(self.depth_mm, dtype=np.float32)
        self.noise_sigma_mm = np.asarray(self.noise_sigma_mm, dtype=np.float32)
        self.valid_count = np.asarray(self.valid_count, dtype=np.uint16)
        if self.depth_mm.ndim != 2:
            raise ValueError(f"Surface depth phai la anh 2D, nhan {self.depth_mm.shape}")
        if self.noise_sigma_mm.shape != self.depth_mm.shape:
            raise ValueError("Surface noise va depth khong cung kich thuoc.")
        if self.valid_count.shape != self.depth_mm.shape:
            raise ValueError("Surface valid_count va depth khong cung kich thuoc.")
        if int(self.frame_count) < 1:
            raise ValueError("Surface frame_count phai >= 1.")
        if self.plane is not None:
            normal = np.asarray(self.plane.get("normal"), dtype=float)
            if (normal.shape != (3,) or not np.isfinite(normal).all()
                    or abs(np.linalg.norm(normal)-1.) > 1e-4
                    or not np.isfinite(self.plane.get("offset_mm", np.nan))
                    or not np.isfinite(self.plane.get("rms_mm", np.nan))
                    or self.plane["rms_mm"] < 0):
                raise ValueError("Mat phang nen khong hop le.")

    @classmethod
    def from_frames(
        cls,
        frames: Iterable[Frame3D],
        min_z_mm: float = 0.0,
        max_z_mm: float = np.inf,
        min_valid_fraction: float = 0.50,
        plane_roi: np.ndarray | None = None,
        plane_settings: dict | None = None,
    ) -> "SurfaceReference":
        depths: list[np.ndarray] = []
        expected_shape: tuple[int, int] | None = None
        identity = None
        plane_points = []
        for frame in frames:
            frame.validate()
            depth = frame_depth_mm(frame).astype(np.float32, copy=True)
            current_identity = frame_identity(frame)
            if identity is None:
                identity = current_identity
            elif identity != current_identity:
                raise ValueError("Cac frame nen khac camera/crop/he toa do/hieu chuan.")
            if expected_shape is None:
                expected_shape = depth.shape
            elif depth.shape != expected_shape:
                raise ValueError(
                    f"Cac frame nen khong cung kich thuoc: {depth.shape} != {expected_shape}"
                )
            valid = np.isfinite(frame.point_cloud_mm).all(axis=2) & (depth > 0)
            valid &= depth >= float(min_z_mm)
            valid &= depth <= float(max_z_mm)
            depth[~valid] = np.nan
            depths.append(depth)
            if plane_roi is not None:
                if plane_roi.shape != depth.shape:
                    raise ValueError("Floor ROI khac kich thuoc frame.")
                pts = frame.point_cloud_mm[valid & plane_roi]
                if len(pts):
                    plane_points.append(pts[::max(1, len(pts)//2000)])

        if not depths:
            raise ValueError("Khong co frame nao de tao be mat thung.")
        stack = np.stack(depths, axis=0)
        valid_count = np.isfinite(stack).sum(axis=0)
        required_count = max(1, int(np.ceil(len(depths) * float(min_valid_fraction))))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(stack, axis=0)
            mad = np.nanmedian(np.abs(stack - median[None, :, :]), axis=0)
        noise_sigma = 1.4826 * mad
        insufficient = valid_count < required_count
        median[insufficient] = np.nan
        noise_sigma[insufficient] = np.nan
        plane, plane_error = None, None
        if plane_roi is not None:
            settings = plane_settings or {}
            try:
                plane = fit_plane(np.concatenate(plane_points) if plane_points else np.empty((0, 3)),
                    inlier_mm=float(settings.get("plane_inlier_mm", 1.5)),
                    min_ratio=float(settings.get("plane_min_inlier_ratio", .8)),
                    max_rms_mm=float(settings.get("plane_max_rms_mm", 1.0)))
            except ValueError as exc:
                plane_error = str(exc)
        return cls(
            depth_mm=median.astype(np.float32),
            noise_sigma_mm=noise_sigma.astype(np.float32),
            valid_count=valid_count.astype(np.uint16),
            frame_count=len(depths),
            created_utc=datetime.now(timezone.utc).isoformat(),
            identity=identity or {}, plane=plane, plane_error=plane_error,
        )

    def evaluate_mask(
        self,
        frame: Frame3D,
        mask: np.ndarray,
        settings: dict[str, Any],
    ) -> SurfaceCheck:
        current = frame_depth_mm(frame)
        if current.shape != self.depth_mm.shape:
            return SurfaceCheck(
                valid=True,
                reason="SURFACE_SHAPE_MISMATCH",
                valid_ratio=0.0,
            )

        binding = identity_status(self.identity, frame)
        if binding == "MISMATCH" or (settings.get("require_reference_identity", False) and binding != "MATCH"):
            return SurfaceCheck(True, "SURFACE_IDENTITY_" + binding, 0.0, reference_status=binding)
        height_mode = settings.get("height_mode", "camera_z")
        if height_mode == "plane" and self.plane is None:
            return SurfaceCheck(True, "FLOOR_PLANE_MISSING", 0.0, reference_status=binding, height_mode="plane")

        object_mask = resize_mask_nearest(mask, *current.shape)
        erode_px = int(settings.get("mask_erode_px", 3))
        working_mask = erode_mask(object_mask, erode_px)
        min_mask_pixels = int(settings.get("min_mask_pixels", 30))
        if int(working_mask.sum()) < min_mask_pixels:
            working_mask = object_mask
        # The denominator is the COMPLETE segmentation mask, including holes
        # in the sensor data. Erosion only excludes unreliable edge evidence.
        mask_count = int(object_mask.sum())
        if mask_count == 0:
            return SurfaceCheck(False, "EMPTY_MASK", 0.0, status="REJECT")

        valid = working_mask.copy()
        valid &= np.isfinite(frame.point_cloud_mm).all(axis=2) & (current > 0)
        valid &= np.isfinite(self.depth_mm) & (self.depth_mm > 0)
        min_z = float(settings.get("min_z_mm", 0.0))
        max_z = float(settings.get("max_z_mm", np.inf))
        valid &= current >= min_z
        valid &= current <= max_z
        valid &= self.depth_mm >= min_z
        valid &= self.depth_mm <= max_z
        valid_count = int(valid.sum())
        valid_ratio = float(valid_count / max(mask_count, 1))
        minimum_valid_ratio = float(settings.get("min_valid_ratio", 0.50))
        minimum_valid_points = int(settings.get("min_valid_points", 30))
        if valid_count < minimum_valid_points:
            return SurfaceCheck(
                valid=True,
                reason="INSUFFICIENT_3D",
                valid_ratio=valid_ratio,
                mask_pixels=mask_count,
                valid_points=valid_count,
            )

        # In CameraSpace, a real object is closer than the empty-bin surface,
        # therefore Z_background - Z_current is positive.
        heights = self.depth_mm[valid] - current[valid]
        gate_heights = heights
        local_noise = self.noise_sigma_mm[valid]
        if height_mode == "plane":
            normal = np.asarray(self.plane["normal"])
            points = frame.point_cloud_mm[valid]
            heights = points @ normal + float(self.plane["offset_mm"])
            # Detect change against the empty-bin surface too: static walls
            # above the floor must not become new raised objects.
            ray_scale = (points / points[:, 2:3]) @ normal
            gate_heights = (current[valid] - self.depth_mm[valid]) * ray_scale
            local_noise = np.hypot(local_noise * np.abs(ray_scale), self.plane["rms_mm"])
        minimum_height = float(settings.get("min_height_mm", 5.0))
        noise_multiplier = float(settings.get("noise_sigma_multiplier", 3.0))
        local_noise = np.where(np.isfinite(local_noise), local_noise, np.inf)
        thresholds = np.maximum(minimum_height, noise_multiplier * local_noise)
        threshold = float(np.median(thresholds))
        percentile = float(settings.get("height_percentile", 90.0))
        height_value = float(np.percentile(heights, percentile))
        # The literal maximum can be a single noisy depth return. Keep it
        # alongside P99 so operators can see whether a spike dominates it.
        height_max_mm = float(np.max(heights))
        height_p99_mm = float(np.percentile(heights, 99.0))
        raised_points = int(np.count_nonzero(gate_heights >= thresholds))
        raised_ratio = float(raised_points / valid_count)
        raised_ratio_mask = float(raised_points / mask_count)
        minimum_raised_ratio = float(settings.get("min_raised_ratio_mask", 0.02))
        minimum_raised_points = int(settings.get("min_raised_points", 30))
        if (raised_points >= minimum_raised_points
                and raised_ratio_mask >= minimum_raised_ratio):
            status, reason, keep = "PASS", None, True
        elif valid_ratio >= minimum_valid_ratio:
            # Missing depth is not floor evidence. Reject only when enough of
            # the mask has measured current AND background points.
            status, reason, keep = "REJECT", "NEAR_BIN_SURFACE", False
        else:
            status, reason, keep = "UNKNOWN", "INSUFFICIENT_3D", True
        return SurfaceCheck(
            valid=keep,
            reason=reason,
            valid_ratio=valid_ratio,
            raised_ratio=raised_ratio,
            height_percentile_mm=height_value,
            height_max_mm=height_max_mm,
            height_p99_mm=height_p99_mm,
            height_threshold_mm=threshold,
            status=status,
            mask_pixels=mask_count,
            valid_points=valid_count,
            raised_points=raised_points,
            raised_ratio_mask=raised_ratio_mask,
            height_mode=height_mode, reference_status=binding,
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": 2,
            "frame_count": int(self.frame_count),
            "created_utc": self.created_utc,
            "shape_hw": list(self.depth_mm.shape),
            "distance_unit": "millimetre",
            "identity": self.identity,
            "plane": self.plane,
            "plane_error": self.plane_error,
        }
        np.savez_compressed(
            output,
            depth_mm=self.depth_mm,
            noise_sigma_mm=self.noise_sigma_mm,
            valid_count=self.valid_count,
            metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "SurfaceReference":
        source = Path(path)
        with np.load(source, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            return cls(
                depth_mm=np.array(data["depth_mm"], dtype=np.float32, copy=True),
                noise_sigma_mm=np.array(
                    data["noise_sigma_mm"], dtype=np.float32, copy=True
                ),
                valid_count=np.array(data["valid_count"], dtype=np.uint16, copy=True),
                frame_count=int(metadata["frame_count"]),
                created_utc=str(metadata["created_utc"]),
                identity=metadata.get("identity", {}), plane=metadata.get("plane"),
                plane_error=metadata.get("plane_error"),
            )


def capture_surface_reference(
    camera,
    frame_count: int,
    min_z_mm: float,
    max_z_mm: float,
    min_valid_fraction: float = 0.50,
    progress: Callable[[int, int], None] | None = None,
    floor_roi=None,
    plane_settings: dict | None = None,
) -> SurfaceReference:
    """Capture fresh frames from an opened camera while the bin is empty."""

    count = max(1, int(frame_count))
    frames: list[Frame3D] = []
    for index in range(count):
        frames.append(camera.capture())
        if progress is not None:
            progress(index + 1, count)
    return SurfaceReference.from_frames(
        frames,
        min_z_mm=min_z_mm,
        max_z_mm=max_z_mm,
        min_valid_fraction=min_valid_fraction,
        plane_roi=None if floor_roi is None else floor_roi.mask(frames[0].texture_bgr.shape[:2]),
        plane_settings=plane_settings,
    )
