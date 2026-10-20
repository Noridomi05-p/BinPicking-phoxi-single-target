from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import project_path
from .roi import PolygonROI
from .surface_reference import SurfaceReference
from .types import Frame3D, InstanceDetection, PostprocessDecision


class InstancePostProcessor:
    """Apply deterministic ROI and empty-bin 3D gates after YOLO."""

    def __init__(
        self,
        roi_settings: dict[str, Any] | None = None,
        surface_settings: dict[str, Any] | None = None,
        roi_path: str | Path | None = None,
        surface_path: str | Path | None = None,
    ) -> None:
        self.roi_settings = dict(roi_settings or {})
        self.surface_settings = dict(surface_settings or {})
        self.roi_path = None if roi_path is None else Path(roi_path)
        self.surface_path = None if surface_path is None else Path(surface_path)
        self.roi: PolygonROI | None = None
        self.surface_reference: SurfaceReference | None = None
        self.load_errors: list[str] = []
        self.reload()

    @property
    def roi_enabled(self) -> bool:
        return bool(self.roi_settings.get("enabled", True))

    @property
    def surface_enabled(self) -> bool:
        return bool(self.surface_settings.get("enabled", True))

    def reload(self) -> None:
        self.load_errors.clear()
        self.roi = None
        self.surface_reference = None
        if self.roi_enabled and self.roi_path is not None and self.roi_path.exists():
            try:
                self.roi = PolygonROI.load(self.roi_path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.load_errors.append(f"Khong doc duoc ROI {self.roi_path}: {exc}")
        if (
            self.surface_enabled
            and self.surface_path is not None
            and self.surface_path.exists()
        ):
            try:
                self.surface_reference = SurfaceReference.load(self.surface_path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.load_errors.append(
                    f"Khong doc duoc surface {self.surface_path}: {exc}"
                )

    def set_roi(self, roi: PolygonROI, save: bool = True) -> None:
        self.roi = roi
        if save:
            if self.roi_path is None:
                raise ValueError("Chua cau hinh duong dan file ROI.")
            roi.save(self.roi_path)

    def set_surface_reference(
        self, reference: SurfaceReference, save: bool = True
    ) -> None:
        self.surface_reference = reference
        if save:
            if self.surface_path is None:
                raise ValueError("Chua cau hinh duong dan surface reference.")
            reference.save(self.surface_path)

    def evaluate(
        self,
        frame: Frame3D,
        detection: InstanceDetection,
    ) -> PostprocessDecision:
        decision = PostprocessDecision(valid=True)

        if self.roi_enabled and self.roi is not None:
            overlap, center_inside = self.roi.evaluate_mask(detection.mask)
            decision.roi_overlap = overlap
            decision.roi_center_inside = center_inside
            minimum_overlap = float(self.roi_settings.get("min_overlap", 0.80))
            require_center = bool(self.roi_settings.get("require_center", True))
            if require_center and not center_inside:
                decision.valid = False
                decision.reason = "OUTSIDE_ROI"
                return decision
            if overlap < minimum_overlap:
                decision.valid = False
                decision.reason = "LOW_ROI_OVERLAP"
                return decision

        if self.surface_enabled and self.surface_reference is not None:
            check = self.surface_reference.evaluate_mask(
                frame, detection.mask, self.surface_settings
            )
            decision.surface_valid_ratio = check.valid_ratio
            decision.surface_raised_ratio = check.raised_ratio
            decision.surface_height_percentile_mm = check.height_percentile_mm
            decision.surface_height_max_mm = check.height_max_mm
            decision.surface_height_p99_mm = check.height_p99_mm
            decision.surface_height_threshold_mm = check.height_threshold_mm
            decision.surface_status = check.status
            decision.surface_mask_pixels = check.mask_pixels
            decision.surface_valid_points = check.valid_points
            decision.surface_raised_points = check.raised_points
            decision.surface_raised_ratio_mask = check.raised_ratio_mask
            decision.surface_height_mode = check.height_mode
            decision.surface_reference_status = check.reference_status
            decision.reason = check.reason
            if not check.valid:
                decision.valid = False
                decision.reason = check.reason
                return decision
        elif self.surface_enabled:
            decision.surface_status = "MISSING"
            decision.reason = "NO_SURFACE_REFERENCE"
        else:
            decision.surface_status = "OFF"

        return decision

    def status_lines(self) -> list[str]:
        if not self.roi_enabled:
            roi_status = "ROI: OFF"
        elif self.roi is None:
            roi_status = "ROI: MISSING (press R)"
        else:
            roi_status = "ROI: READY"

        if not self.surface_enabled:
            surface_status = "SURFACE: OFF"
        elif self.surface_reference is None:
            surface_status = "SURFACE: MISSING (empty bin + press B)"
        else:
            surface_status = (
                f"SURFACE: READY ({self.surface_reference.frame_count} frames)"
            )
            surface_status += " | FLOOR PLANE: " + ("READY" if self.surface_reference.plane else "MISSING (F then B)")
        return [roi_status, surface_status]


def build_postprocessor(config: dict[str, Any]) -> InstancePostProcessor:
    post_cfg = dict(config.get("postprocess", {}))
    roi_cfg = dict(post_cfg.get("roi", {}))
    surface_cfg = dict(post_cfg.get("surface", {}))
    measurement_cfg = dict(config.get("measurement", {}))
    for key in ("min_z_mm", "max_z_mm", "min_valid_points"):
        if key not in surface_cfg and key in measurement_cfg:
            surface_cfg[key] = measurement_cfg[key]

    roi_path = project_path(roi_cfg.get("path", "config/bin_roi.json"), config)
    surface_path = project_path(
        surface_cfg.get("path", "data/calibration/bin_surface.npz"), config
    )
    return InstancePostProcessor(
        roi_settings=roi_cfg,
        surface_settings=surface_cfg,
        roi_path=roi_path,
        surface_path=surface_path,
    )
