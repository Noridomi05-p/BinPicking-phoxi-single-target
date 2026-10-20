import tempfile
import unittest
from pathlib import Path

import numpy as np

from phoxi_yolo3d.pipeline import PhoXiYoloPipeline
from phoxi_yolo3d.postprocess import InstancePostProcessor
from phoxi_yolo3d.roi import PolygonROI
from phoxi_yolo3d.surface_reference import SurfaceReference
from phoxi_yolo3d.types import Frame3D, InstanceDetection


def make_frame(height=80, width=100, z_mm=700.0):
    yy, xx = np.mgrid[:height, :width]
    z = np.full((height, width), z_mm, dtype=np.float32)
    cloud = np.stack(
        [(xx - width / 2) * z / 500, (yy - height / 2) * z / 500, z],
        axis=2,
    ).astype(np.float32)
    return Frame3D(
        texture_bgr=np.zeros((height, width, 3), dtype=np.uint8),
        point_cloud_mm=cloud,
        depth_mm=z,
    )


class RoiAndSurfaceTests(unittest.TestCase):
    def test_roi_overlap_and_round_trip(self):
        roi = PolygonROI([(0.20, 0.20), (0.80, 0.20), (0.80, 0.80), (0.20, 0.80)])
        inside = np.zeros((100, 100), dtype=bool)
        inside[30:60, 30:60] = True
        outside = np.zeros((100, 100), dtype=bool)
        outside[0:10, 0:10] = True
        overlap, center_inside = roi.evaluate_mask(inside)
        self.assertGreater(overlap, 0.99)
        self.assertTrue(center_inside)
        overlap, center_inside = roi.evaluate_mask(outside)
        self.assertEqual(overlap, 0.0)
        self.assertFalse(center_inside)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roi.json"
            roi.save(path)
            loaded = PolygonROI.load(path)
        self.assertEqual(loaded.points_normalized, roi.points_normalized)

    def test_surface_rejects_floor_and_accepts_raised_object(self):
        background_frames = [make_frame(z_mm=700.0) for _ in range(5)]
        reference = SurfaceReference.from_frames(
            background_frames,
            min_z_mm=400,
            max_z_mm=1300,
        )
        mask = np.zeros((80, 100), dtype=bool)
        mask[20:60, 30:70] = True
        settings = {
            "min_z_mm": 400,
            "max_z_mm": 1300,
            "mask_erode_px": 2,
            "min_valid_points": 30,
            "min_valid_ratio": 0.5,
            "min_height_mm": 5.0,
            "noise_sigma_multiplier": 3.0,
            "height_percentile": 90.0,
            "min_raised_ratio": 0.2,
        }

        floor_check = reference.evaluate_mask(make_frame(z_mm=700.0), mask, settings)
        self.assertFalse(floor_check.valid)
        self.assertEqual(floor_check.reason, "NEAR_BIN_SURFACE")

        object_frame = make_frame(z_mm=700.0)
        object_frame.depth_mm[mask] = 675.0
        object_frame.point_cloud_mm[mask, 2] = 675.0
        object_check = reference.evaluate_mask(object_frame, mask, settings)
        self.assertTrue(object_check.valid)
        self.assertGreater(object_check.height_percentile_mm, 20.0)

    def test_pipeline_separates_accepted_and_rejected_instances(self):
        background = SurfaceReference.from_frames([make_frame() for _ in range(3)])
        object_mask = np.zeros((80, 100), dtype=bool)
        object_mask[15:40, 20:45] = True
        scratch_mask = np.zeros((80, 100), dtype=bool)
        scratch_mask[45:65, 55:80] = True
        frame = make_frame()
        frame.depth_mm[object_mask] = 675.0
        frame.point_cloud_mm[object_mask, 2] = 675.0

        detections = [
            InstanceDetection(0, 0, "object", 0.9, (20, 15, 44, 39), object_mask),
            InstanceDetection(1, 1, "scratch", 0.4, (55, 45, 79, 64), scratch_mask),
        ]

        class Segmenter:
            def predict(self, _image):
                return detections

        postprocessor = InstancePostProcessor(
            roi_settings={"enabled": True, "min_overlap": 0.8},
            surface_settings={
                "enabled": True,
                "min_z_mm": 400,
                "max_z_mm": 1300,
                "min_valid_points": 20,
                "min_valid_ratio": 0.5,
                "min_height_mm": 5.0,
                "min_raised_ratio": 0.2,
            },
        )
        postprocessor.roi = PolygonROI(
            [(0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)]
        )
        postprocessor.surface_reference = background
        pipeline = PhoXiYoloPipeline(
            Segmenter(),
            {"min_z_mm": 400, "max_z_mm": 1300, "min_valid_points": 20},
            postprocessor=postprocessor,
        )
        accepted, rejected = pipeline.process_all(frame)
        self.assertEqual([item.detection.class_name for item in accepted], ["object"])
        self.assertEqual([item.detection.class_name for item in rejected], ["scratch"])
        self.assertEqual(rejected[0].postprocess.reason, "NEAR_BIN_SURFACE")


if __name__ == "__main__":
    unittest.main()
