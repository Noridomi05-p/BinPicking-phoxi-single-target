import tempfile
import unittest
from pathlib import Path

import numpy as np

from phoxi_yolo3d.storage import load_capture, save_capture
from phoxi_yolo3d.types import Frame3D


class StorageTests(unittest.TestCase):
    def test_npz_round_trip(self):
        frame = Frame3D(
            texture_bgr=np.full((8, 12, 3), 127, dtype=np.uint8),
            texture_raw=np.full((8, 12), 2048, dtype=np.uint16),
            point_cloud_mm=np.ones((8, 12, 3), dtype=np.float32) * 600,
            depth_mm=np.full((8, 12), 600, dtype=np.float32),
            confidence_map=np.full((8, 12), 0.9, dtype=np.float32),
            frame_id=42,
            info={"hello": "world"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = save_capture(frame, Path(temporary), stem="sample", save_3d=True)
            loaded = load_capture(paths["scan"])
            np.testing.assert_array_equal(loaded.texture_bgr, frame.texture_bgr)
            np.testing.assert_allclose(loaded.point_cloud_mm, frame.point_cloud_mm)
            self.assertEqual(loaded.frame_id, 42)
            self.assertEqual(loaded.info["hello"], "world")


if __name__ == "__main__":
    unittest.main()

