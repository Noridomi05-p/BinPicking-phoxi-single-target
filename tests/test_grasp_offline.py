import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from phoxi_yolo3d.grasp_data import GraspTarget, prepare_scene, remove_small_3d_components, sample_real_points
from phoxi_yolo3d.grasp_selection import camera_grasp_record, diversify_per_object, filter_and_associate
from phoxi_yolo3d.storage import save_capture
from phoxi_yolo3d.types import Frame3D


def test_scene():
    yy, xx = np.mgrid[:30, :30]
    cloud = np.stack(((xx-15)*1., (yy-15)*1., np.full(xx.shape, 1000.)), axis=-1).astype(np.float32)
    return Frame3D(np.zeros((30, 30, 3), dtype=np.uint8), cloud,
                   frame_id=8, timestamp_utc="2026-10-02T08:28:12Z")


def target(points, instance=0):
    return GraspTarget(instance, f"test_{instance}", .9, np.ones((2, 2), bool),
                       np.ones((2, 2), bool), np.asarray(points), {"eligible": True})


def proposal(width=.06, score=.9):
    row = np.zeros(17)
    row[:4] = [score, width, .02, .02]
    row[4:13] = np.eye(3).ravel()
    row[13:16] = [0, 0, 1]
    row[16] = -1
    return row


class GraspPreparationTests(unittest.TestCase):
    def test_disconnected_depth_spike_removed_without_height_cutoff(self):
        frame = test_scene()
        frame.point_cloud_mm[4:6, 4:6, 2] = 700
        frame.point_cloud_mm[10, 10] = np.nan
        cleaned, stats = remove_small_3d_components(frame.point_cloud_mm, np.ones((30, 30), bool),
                                                  {"mask_erode_px": 0, "min_component_points": 20})
        self.assertFalse(cleaned[4:6, 4:6].any())
        self.assertFalse(cleaned[10, 10])
        self.assertTrue(stats["eligible"])
        self.assertEqual(stats["removed_points"], 4)
        self.assertTrue(cleaned[20, 20])

    def test_large_raised_surface_kept(self):
        frame = test_scene()
        frame.point_cloud_mm[:10, :, 2] = 900
        cleaned, _ = remove_small_3d_components(frame.point_cloud_mm, np.ones((30, 30), bool),
                                              {"mask_erode_px": 0, "min_component_points": 20})
        self.assertTrue(cleaned.all())

    def test_empty_depth_does_not_pass(self):
        frame = test_scene()
        frame.point_cloud_mm[:] = 0
        clean, stats = remove_small_3d_components(frame.point_cloud_mm, np.ones((30, 30), bool), {})
        self.assertFalse(stats["eligible"])
        self.assertFalse(clean.any())

    def test_sampling_duplicates_real_points_only(self):
        points = np.array([[0, 0, 1], [0, .1, 1]], dtype=np.float32)
        sampled, ids = sample_real_points(points, 2048, 42)
        np.testing.assert_array_equal(sampled, points[ids])
        self.assertEqual(set(ids), {0, 1})
        self.assertRaises(ValueError, sample_real_points, np.empty((0, 3)), 2048, 42)

    def test_rejected_mask_not_selected_and_collision_cloud_not_cropped(self):
        frame = test_scene()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan = save_capture(frame, root, "frame", save_3d=True)["scan"]
            mask = np.zeros((30, 30), bool)
            mask[2:-2, 2:-2] = True
            np.savez(root / "masks.npz", instance_0=mask, instance_1=mask)
            metadata = {"frame_id": 8, "objects": [{"instance_id": 0, "class_name": "cube",
                        "confidence": .9, "verified_3d": True}], "rejected": [{"instance_id": 1}]}
            result = root / "result.json"
            result.write_text(json.dumps(metadata))
            scene = prepare_scene(scan, root / "masks.npz", {"mask_erode_px": 1}, result)
            self.assertEqual([t.instance_id for t in scene.targets], [0])
            self.assertEqual(len(scene.collision_points_m), 900)
            self.assertLess(len(scene.network_points_m), 900)
            np.testing.assert_allclose(scene.collision_points_m[:, 2], 1)
            with self.assertRaisesRegex(ValueError, "OBJECTS"):
                prepare_scene(scan, root / "masks.npz", {}, result, instance_ids=[1])
            metadata["frame_id"] = 9
            result.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "scan khac"):
                prepare_scene(scan, root / "masks.npz", {}, result)

    def test_metadata_required_or_explicit_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan = save_capture(test_scene(), root, "frame", save_3d=True)["scan"]
            np.savez(root / "masks.npz", instance_0=np.ones((30, 30), bool))
            with self.assertRaisesRegex(ValueError, "instance-ids"):
                prepare_scene(scan, root / "masks.npz", {})
            scene = prepare_scene(scan, root / "masks.npz", {}, instance_ids=[0])
            self.assertTrue(scene.targets[0].stats["eligible"])
            self.assertTrue(any("NO_RESULT" in w for w in scene.warnings))


class GraspSelectionTests(unittest.TestCase):
    def setUp(self):
        x, y = np.meshgrid(np.linspace(-.015, .015, 9), np.linspace(-.02, .02, 9))
        self.points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 1.)))
        self.target = target(self.points)
        self.gripper = {"max_opening_mm": 85, "opening_margin_mm": 4}
        self.settings = {"min_target_support_points": 10}

    def select(self, rows, targets=None, observed=None):
        return filter_and_associate(np.array(rows), targets or [self.target],
                                    self.gripper, self.settings, observed)

    def test_opening_limit_includes_margin_and_invalid_rotation_rejected(self):
        invalid = proposal()
        invalid[4] = 2
        rows, associations, rejected = self.select([proposal(.081), proposal(.082), invalid])
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(associations[0]["suggested_opening_mm"], 85)
        self.assertEqual(rejected["TOO_WIDE_FOR_GRIPPER"], 1)
        self.assertEqual(rejected["INVALID_POSE"], 1)

    def test_empty_or_wrong_target_rejected(self):
        self.assertEqual(len(self.select([proposal()], [target(self.points + [0, 0, .1])])[0]), 0)
        self.assertEqual(len(self.select([proposal()], [target(self.points[:2])])[0]), 0)

    def test_multiple_labels_or_unlabelled_observed_evidence_rejected(self):
        rows, _, reasons = self.select([proposal()], [self.target, target(self.points, 1)])
        self.assertEqual(len(rows), 0)
        self.assertEqual(reasons["MULTIPLE_TARGETS_BETWEEN_JAWS"], 1)
        extra = self.points.copy()
        extra[:, 2] += .005
        rows, _, reasons = self.select([proposal()], observed=np.vstack([self.points, extra]))
        self.assertEqual(len(rows), 0)
        self.assertEqual(reasons["MULTIPLE_TARGETS_BETWEEN_JAWS"], 1)

    def test_camera_transform_is_metres_and_never_robot_ready(self):
        row = proposal()
        record = camera_grasp_record(row, {"instance_id": 0})
        self.assertFalse(record["robot_ready"])
        self.assertEqual(record["translation_camera_mm"], [0, 0, 1000])
        self.assertEqual(np.asarray(record["T_camera_grasp_m"])[2, 3], 1)

    def test_jaw_symmetry_suppression_and_per_object_quota(self):
        a, b, c = proposal(), proposal(score=.8), proposal(score=.7)
        b[4:13] = np.diag([1, -1, -1]).ravel()
        rows, labels = diversify_per_object(np.array([a, b, c]),
                                            [{"instance_id": 0}, {"instance_id": 0}, {"instance_id": 1}],
                                            {"per_object": 1})
        self.assertEqual(len(rows), 2)
        self.assertEqual([l["instance_id"] for l in labels], [0, 1])


if __name__ == "__main__":
    unittest.main()
