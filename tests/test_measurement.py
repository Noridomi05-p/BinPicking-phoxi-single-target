import unittest

import numpy as np

from phoxi_yolo3d.measurement import measure_mask_3d
from phoxi_yolo3d.types import Frame3D


def make_frame(height=100, width=120, z_mm=700.0):
    yy, xx = np.mgrid[:height, :width]
    z = np.full((height, width), z_mm, dtype=np.float32)
    cloud = np.stack(
        [(xx - width / 2) * z / 500, (yy - height / 2) * z / 500, z], axis=2
    ).astype(np.float32)
    return Frame3D(
        texture_bgr=np.zeros((height, width, 3), dtype=np.uint8),
        point_cloud_mm=cloud,
        depth_mm=z,
        confidence_map=np.ones((height, width), dtype=np.float32),
    )


class MeasurementTests(unittest.TestCase):
    def test_constant_surface_depth(self):
        frame = make_frame()
        mask = np.zeros((100, 120), dtype=bool)
        mask[20:80, 25:95] = True
        result = measure_mask_3d(
            frame,
            mask,
            {
                "min_z_mm": 400,
                "max_z_mm": 1300,
                "mask_erode_px": 3,
                "min_valid_points": 50,
                "nearest_percentile": 5,
                "trim_percent": 1,
                "confidence_min": None,
            },
        )
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.depth_z_median_mm, 700.0, places=4)
        self.assertGreater(result.distance_3d_median_mm, 700.0)
        self.assertGreater(result.point_count, 1000)

    def test_erosion_rejects_contaminated_mask_edge(self):
        frame = make_frame()
        mask = np.zeros((100, 120), dtype=bool)
        mask[20:80, 25:95] = True
        edge = mask.copy()
        edge[24:76, 29:91] = False
        frame.point_cloud_mm[edge, 2] = 1100.0
        result = measure_mask_3d(
            frame,
            mask,
            {
                "min_z_mm": 400,
                "max_z_mm": 1300,
                "mask_erode_px": 5,
                "min_valid_points": 50,
                "trim_percent": 0,
            },
        )
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.depth_z_median_mm, 700.0, places=4)

    def test_too_few_points_is_invalid(self):
        frame = make_frame(20, 20)
        mask = np.zeros((20, 20), dtype=bool)
        mask[1:3, 1:3] = True
        result = measure_mask_3d(
            frame,
            mask,
            {"min_z_mm": 400, "max_z_mm": 1300, "min_valid_points": 20},
        )
        self.assertFalse(result.valid)
        self.assertIn("khong_du_diem", result.reason)


if __name__ == "__main__":
    unittest.main()

