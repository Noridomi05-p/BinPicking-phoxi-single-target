import unittest

import numpy as np

from phoxi_yolo3d.image_utils import normalize_texture


class NormalizeTextureTests(unittest.TestCase):
    def test_uint16_grayscale_becomes_three_channel_uint8(self):
        source = np.arange(100, dtype=np.uint16).reshape(10, 10) * 100
        result = normalize_texture(source, 0, 100)
        self.assertEqual(result.shape, (10, 10, 3))
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])

    def test_rgb_is_converted_to_bgr(self):
        source = np.zeros((2, 3, 3), dtype=np.uint8)
        source[:, :, 0] = 10
        source[:, :, 2] = 200
        result = normalize_texture(source)
        self.assertTrue(np.all(result[:, :, 0] == 200))
        self.assertTrue(np.all(result[:, :, 2] == 10))


if __name__ == "__main__":
    unittest.main()

