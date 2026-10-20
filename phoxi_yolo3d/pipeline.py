from __future__ import annotations

from typing import Any, Protocol
import time

from .measurement import measure_mask_3d, summarize_mask_3d
from .image_utils import prepare_frame_texture
from .duplicates import suppress_duplicate_masks
from .types import (
    Frame3D,
    InstanceDetection,
    InstanceResult,
    PostprocessDecision,
)


class Segmenter(Protocol):
    def predict(self, image_bgr) -> list[InstanceDetection]: ...


class PostProcessor(Protocol):
    def evaluate(
        self, frame: Frame3D, detection: InstanceDetection
    ) -> PostprocessDecision: ...


class PhoXiYoloPipeline:
    def __init__(
        self,
        segmenter: Segmenter,
        measurement_settings: dict[str, Any],
        postprocessor: PostProcessor | None = None,
        duplicate_settings: dict[str, Any] | None = None,
        picking_settings: dict[str, Any] | None = None,
        texture_settings: dict[str, Any] | None = None,
        offline: bool = False,
    ) -> None:
        self.segmenter = segmenter
        self.measurement_settings = measurement_settings
        self.postprocessor = postprocessor
        self.duplicate_settings = dict(duplicate_settings or {})
        self.last_rejected: list[InstanceResult] = []
        self.picking_settings = dict(picking_settings or {})
        self.texture_settings = texture_settings
        self.offline = offline
        self.last_picking: dict[str, Any] = {"enabled": False, "robot_ready": False}
        self.last_timings_ms: dict[str, float] = {}

    def process_all(
        self, frame: Frame3D
    ) -> tuple[list[InstanceResult], list[InstanceResult]]:
        """Return accepted and rejected instances without hiding debug evidence."""

        frame.validate()
        started = time.perf_counter()
        if self.texture_settings is not None:
            prepare_frame_texture(frame, self.texture_settings)
        prepared = time.perf_counter()
        detections = self.segmenter.predict(frame.texture_bgr)
        detected = time.perf_counter()
        accepted: list[InstanceResult] = []
        rejected: list[InstanceResult] = []
        for detection in detections:
            measure = summarize_mask_3d if self.picking_settings.get("enabled", False) else measure_mask_3d
            measurement = measure(
                frame, detection.mask, self.measurement_settings
            )
            decision = (
                PostprocessDecision()
                if self.postprocessor is None
                else self.postprocessor.evaluate(frame, detection)
            )
            result = InstanceResult(
                detection=detection,
                measurement=measurement,
                postprocess=decision,
            )
            if decision.valid:
                accepted.append(result)
            else:
                rejected.append(result)
        accepted, duplicates = suppress_duplicate_masks(accepted, self.duplicate_settings)
        rejected.extend(duplicates)
        self.last_rejected = rejected
        gated = time.perf_counter()
        if self.picking_settings.get("enabled", False):
            from .picking import plan_single_target
            self.last_picking = plan_single_target(frame, accepted, self.picking_settings,
                self.measurement_settings, self.postprocessor, self.offline)
            target_id = self.last_picking.get("target_instance_id")
            for item in accepted:
                if item.detection.instance_id == target_id:
                    item.measurement = measure_mask_3d(frame, item.detection.mask, self.measurement_settings)
        finished = time.perf_counter()
        self.last_timings_ms = {"texture": (prepared-started)*1000, "segmentation": (detected-prepared)*1000,
            "gates_and_measurements": (gated-detected)*1000, "picking_and_target_measurement": (finished-gated)*1000,
            "total": (finished-started)*1000}
        return accepted, rejected

    def process(self, frame: Frame3D) -> list[InstanceResult]:
        accepted, _ = self.process_all(frame)
        return accepted
