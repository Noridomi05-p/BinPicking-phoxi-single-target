import unittest

import numpy as np

from phoxi_yolo3d.diagnostics import frame_diagnostics, result_counts
from phoxi_yolo3d.pipeline import PhoXiYoloPipeline
from phoxi_yolo3d.postprocess import InstancePostProcessor
from phoxi_yolo3d.surface_reference import SurfaceReference
from phoxi_yolo3d.types import Frame3D, InstanceDetection


def frame(z):
    z = np.asarray(z, dtype=np.float32)
    cloud = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=2)
    return Frame3D(np.zeros((*z.shape, 3), dtype=np.uint8), cloud, depth_mm=z.copy())


class SparseDepthTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.ones((100, 100), dtype=bool)
        self.background = SurfaceReference.from_frames([frame(np.full((100, 100), 1000))])
        self.settings = {
            'mask_erode_px': 0, 'min_z_mm': 400, 'max_z_mm': 1300,
            'min_valid_points': 30, 'min_raised_points': 30,
            'min_raised_ratio_mask': 0.02, 'min_valid_ratio': 0.5,
            'min_height_mm': 5, 'noise_sigma_multiplier': 3,
        }

    def test_sparse_raised_surface_can_pass_without_half_mask_coverage(self):
        # Only 3% of the mask has depth, but those 300 measured points are raised.
        z = np.zeros((100, 100), dtype=np.float32)
        z[10:13, :] = 975
        check = self.background.evaluate_mask(frame(z), self.mask, self.settings)
        self.assertTrue(check.valid)
        self.assertEqual(check.status, 'PASS')
        self.assertAlmostEqual(check.raised_ratio_mask, 0.03)
        self.assertAlmostEqual(check.raised_ratio, 1.0)

    def test_denominator_is_full_mask_not_just_valid_points(self):
        z = np.zeros((100, 100), dtype=np.float32)
        z[10, :] = 975  # All 100 valid points are high; only 1% of full mask.
        check = self.background.evaluate_mask(frame(z), self.mask, self.settings)
        self.assertEqual(check.status, 'UNKNOWN')
        self.assertTrue(check.valid)
        self.assertAlmostEqual(check.raised_ratio_mask, 0.01)

    def test_enough_raised_evidence_is_not_vetoed_by_low_p90(self):
        z = np.full((100, 100), 1000, dtype=np.float32)
        z[10:13, :] = 975
        check = self.background.evaluate_mask(frame(z), self.mask, self.settings)
        self.assertEqual(check.status, 'PASS')
        self.assertAlmostEqual(check.raised_ratio_mask, .03)
        self.assertAlmostEqual(check.height_percentile_mm, 0)

    def test_height_max_p99_and_json_use_only_paired_valid_points(self):
        z = np.full((100, 100), 980, dtype=np.float32)
        z[10, 10] = 900  # One high outlier: 100 mm above the reference.
        z[20, 20] = 0  # No current 3D point.
        self.background.depth_mm[30, 30] = np.nan  # No reference point.
        detection = InstanceDetection(0, 2, 'khoi_tru', .9, (0, 0, 99, 99), self.mask)

        class Segmenter:
            def predict(self, _image):
                return [detection]

        post = InstancePostProcessor(
            roi_settings={'enabled': False}, surface_settings=self.settings,
        )
        post.surface_reference = self.background
        kept, rejected = PhoXiYoloPipeline(
            Segmenter(), {'min_valid_points': 80}, post,
        ).process_all(frame(z))
        self.assertFalse(rejected)
        self.assertTrue(kept[0].verified_3d)
        self.assertEqual(kept[0].postprocess.surface_valid_points, 9998)
        self.assertAlmostEqual(kept[0].postprocess.surface_height_max_mm, 100)
        self.assertAlmostEqual(kept[0].postprocess.surface_height_p99_mm, 20)
        payload = kept[0].to_dict()
        self.assertAlmostEqual(payload['postprocess']['surface_height_max_mm'], 100)
        self.assertAlmostEqual(payload['postprocess']['surface_height_p99_mm'], 20)
        self.assertAlmostEqual(payload['measurement']['distance_3d_median_mm'], 980)

    def test_zero_and_fourteen_points_stay_unknown_not_floor(self):
        for count in (0, 14):
            z = np.zeros((100, 100), dtype=np.float32)
            z.flat[:count] = 975
            check = self.background.evaluate_mask(frame(z), self.mask, self.settings)
            self.assertTrue(check.valid)
            self.assertEqual(check.status, 'UNKNOWN')
            self.assertEqual(check.reason, 'INSUFFICIENT_3D')

    def test_measured_floor_is_rejected(self):
        z = np.full((100, 100), 999, dtype=np.float32)
        check = self.background.evaluate_mask(frame(z), self.mask, self.settings)
        self.assertFalse(check.valid)
        self.assertEqual(check.reason, 'NEAR_BIN_SURFACE')

    def test_reference_holes_cannot_prove_or_disprove_object_height(self):
        reference = SurfaceReference.from_frames([frame(np.zeros((100, 100)))])
        check = reference.evaluate_mask(frame(np.full((100, 100), 975)), self.mask, self.settings)
        self.assertTrue(check.valid)
        self.assertEqual(check.status, 'UNKNOWN')
        self.assertEqual(check.valid_points, 0)

    def test_shape_mismatch_is_unknown_not_an_automatic_object_rejection(self):
        check = self.background.evaluate_mask(frame(np.ones((50, 50)) * 975), np.ones((50, 50)), self.settings)
        self.assertTrue(check.valid)
        self.assertEqual(check.status, 'UNKNOWN')
        self.assertEqual(check.reason, 'SURFACE_SHAPE_MISMATCH')

    def test_noise_threshold_prevents_noisy_floor_from_passing(self):
        self.background.noise_sigma_mm[:] = 3
        check = self.background.evaluate_mask(frame(np.full((100, 100), 994)), self.mask, self.settings)
        self.assertEqual(check.status, 'REJECT')
        self.assertAlmostEqual(check.height_threshold_mm, 9)

    def test_mask_border_does_not_provide_false_positive_evidence(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:90, 10:90] = True
        z = np.ones((100, 100), dtype=np.float32) * 1000
        z[10:12, 10:90] = 975
        check = self.background.evaluate_mask(frame(z), mask, {**self.settings, 'mask_erode_px': 3})
        self.assertEqual(check.status, 'REJECT')
        self.assertEqual(check.mask_pixels, 6400)

    def test_pipeline_keeps_zero_depth_label_but_never_marks_verified(self):
        detection = InstanceDetection(0, 3, 'khoi_lap_phuong', .98, (0, 0, 99, 99), self.mask)

        class Segmenter:
            def predict(self, _image):
                return [detection]

        post = InstancePostProcessor(roi_settings={'enabled': False}, surface_settings=self.settings)
        post.surface_reference = self.background
        pipeline = PhoXiYoloPipeline(Segmenter(), {'min_valid_points': 80}, post)
        kept, rejected = pipeline.process_all(frame(np.zeros((100, 100))))
        self.assertEqual(len(kept), 1)
        self.assertEqual(rejected, [])
        self.assertFalse(kept[0].measurement.valid)
        self.assertFalse(kept[0].verified_3d)
        self.assertEqual(result_counts(kept, rejected)['unverified_count'], 1)
        self.assertFalse(kept[0].to_dict()['verified_3d'])

    def test_diagnostics_distinguish_missing_depth_and_out_of_range(self):
        z = np.full((10, 10), 1000, dtype=np.float32)
        z[:2] = 0
        z[2:4] = 1400
        post = InstancePostProcessor(roi_settings={'enabled': False})
        stats = frame_diagnostics(frame(z), {'min_z_mm': 400, 'max_z_mm': 1300}, post)
        self.assertAlmostEqual(stats['pointcloud_valid_ratio_roi'], .8)
        self.assertAlmostEqual(stats['pointcloud_in_range_ratio_roi'], .6)
        self.assertIsNone(stats['background_valid_ratio_roi'])


if __name__ == '__main__':
    unittest.main()
