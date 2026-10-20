from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class Frame3D:
    """One organized PhoXi frame in camera coordinates (millimetres)."""

    texture_bgr: np.ndarray
    point_cloud_mm: np.ndarray
    texture_raw: np.ndarray | None = None
    depth_mm: np.ndarray | None = None
    confidence_map: np.ndarray | None = None
    normal_map: np.ndarray | None = None
    frame_id: int | None = None
    timestamp_utc: str | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.texture_bgr.ndim != 3 or self.texture_bgr.shape[2] != 3:
            raise ValueError(
                f"texture_bgr phai co shape (H,W,3), nhan {self.texture_bgr.shape}"
            )
        height, width = self.texture_bgr.shape[:2]
        if self.point_cloud_mm.shape != (height, width, 3):
            raise ValueError(
                "PointCloud khong thang hang voi Texture: "
                f"texture={(height, width)}, point_cloud={self.point_cloud_mm.shape}"
            )
        for name, array in (
            ("DepthMap", self.depth_mm),
            ("ConfidenceMap", self.confidence_map),
        ):
            if array is not None and np.squeeze(array).shape != (height, width):
                raise ValueError(
                    f"{name} khong thang hang voi Texture: {np.squeeze(array).shape} != {(height, width)}"
                )
        if self.normal_map is not None and self.normal_map.shape != (height, width, 3):
            raise ValueError(
                f"NormalMap phai co shape {(height, width, 3)}, nhan {self.normal_map.shape}"
            )


@dataclass
class InstanceDetection:
    instance_id: int
    class_id: int
    class_name: str
    score: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray = field(repr=False)


@dataclass
class InstanceMeasurement:
    valid: bool
    reason: str | None
    point_count: int
    mask_pixel_count: int
    valid_ratio: float
    center_xyz_mm: tuple[float, float, float] | None = None
    depth_z_median_mm: float | None = None
    depth_z_p10_mm: float | None = None
    depth_z_p90_mm: float | None = None
    distance_3d_median_mm: float | None = None
    nearest_surface_mm: float | None = None
    visible_extent_xyz_mm: tuple[float, float, float] | None = None
    confidence_median: float | None = None
    measurement_kind: str = "full_visible_surface"


@dataclass
class PostprocessDecision:
    """Result of the 2D ROI and 3D bin-surface validation gates."""

    valid: bool = True
    reason: str | None = None
    roi_overlap: float | None = None
    roi_center_inside: bool | None = None
    surface_valid_ratio: float | None = None
    surface_raised_ratio: float | None = None
    surface_height_percentile_mm: float | None = None
    surface_height_max_mm: float | None = None
    surface_height_p99_mm: float | None = None
    surface_height_threshold_mm: float | None = None
    # valid means keep the 2D detection; it does not certify a 3D measurement.
    surface_status: str = "NOT_CHECKED"
    surface_mask_pixels: int = 0
    surface_valid_points: int = 0
    surface_raised_points: int = 0
    surface_raised_ratio_mask: float | None = None
    duplicate_of_instance_id: int | None = None
    duplicate_mask_iou: float | None = None
    duplicate_containment: float | None = None
    surface_height_mode: str = "camera_z"
    surface_reference_status: str = "UNBOUND"


@dataclass
class InstanceResult:
    detection: InstanceDetection
    measurement: InstanceMeasurement
    postprocess: PostprocessDecision = field(default_factory=PostprocessDecision)

    @property
    def verified_3d(self) -> bool:
        """Both the measurement and the above-background check are confirmed."""
        return (
            self.postprocess.valid
            and self.measurement.valid
            and self.postprocess.surface_status == "PASS"
        )

    def to_dict(self) -> dict[str, Any]:
        detection = asdict(self.detection)
        detection.pop("mask", None)
        return {
            "instance_id": detection.pop("instance_id"),
            "class_id": detection.pop("class_id"),
            "class_name": detection.pop("class_name"),
            "confidence": detection.pop("score"),
            "bbox_xyxy": list(detection.pop("bbox_xyxy")),
            "verified_3d": self.verified_3d,
            "measurement": asdict(self.measurement),
            "postprocess": asdict(self.postprocess),
        }
