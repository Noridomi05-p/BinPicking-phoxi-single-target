import copy
import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from phoxi_yolo3d.clearance import check_approach, check_boxes, fit_projection
from phoxi_yolo3d.config import load_config, validate_model_classes
from phoxi_yolo3d.config_validation import validate_config
from phoxi_yolo3d.geometry import fit_plane, rotation_for_jaws, to_graspnet_rotation
from phoxi_yolo3d.image_utils import prepare_frame_texture
from phoxi_yolo3d.measurement import summarize_mask_3d
from phoxi_yolo3d.picking import freshness, plan_single_target
from phoxi_yolo3d.picking_visualization import visible_results
from phoxi_yolo3d.pipeline import PhoXiYoloPipeline
from phoxi_yolo3d.postprocess import InstancePostProcessor
from phoxi_yolo3d.replay import FrozenSegmenter, load_frozen_inspection
from phoxi_yolo3d.roi import PolygonROI
from phoxi_yolo3d.surface_reference import SurfaceReference
from phoxi_yolo3d.target_geometry import generate_poses, surface_anchor
from phoxi_yolo3d.types import Frame3D, InstanceDetection, InstanceResult, PostprocessDecision


def synthetic_frame(z=800., height=160, width=200, normal=None, offset=None):
    yy, xx = np.mgrid[:height, :width]
    rays = np.stack(((xx-width/2)/300., (yy-height/2)/300., np.ones_like(xx)), axis=2)
    if normal is not None:
        depth = -float(offset)/(rays @ normal)
    else:
        depth = np.full((height, width), z)
    cloud = (rays*depth[..., None]).astype(np.float32)
    frame = Frame3D(np.full((height, width, 3), 100, np.uint8), cloud,
                    normal_map=np.broadcast_to([0., 0., -1.], cloud.shape).copy().astype(np.float32),
                    frame_id=12, timestamp_utc=datetime.now(timezone.utc).isoformat())
    frame.info['acquisition'] = {'device_id': 'TEST-CAMERA', 'coordinate_space': 'CameraSpace/PrimaryCamera',
        'roi_xyxy': [0, 0, width, height], 'source_shape_hw': [height, width], 'calibration_id': 'test-1',
        'texture_source': 'LED', 'led_power': 2048}
    return frame


def put_object(frame, identifier=0, region=(45, 65, 60, 90), z=740., name='hop_anibox'):
    y0, y1, x0, x1 = region
    mask = np.zeros(frame.texture_bgr.shape[:2], bool)
    mask[y0:y1, x0:x1] = True
    rays = frame.point_cloud_mm[mask]/frame.point_cloud_mm[mask, 2:3]
    frame.point_cloud_mm[mask] = rays*z
    return InstanceDetection(identifier, 0, name, .95, (x0, y0, x1, y1), mask)


def result_for(frame, detection, height=60.):
    measurement = summarize_mask_3d(frame, detection.mask, {'min_valid_points': 30})
    return InstanceResult(detection, measurement, PostprocessDecision(surface_status='PASS',
        surface_height_mode='plane', surface_height_percentile_mm=height))


def picking_settings():
    settings = copy.deepcopy(load_config('config/picking.yaml')['picking'])
    settings.update(require_roi=False, require_plane=False, require_reference_identity=False)
    return settings


def plane_post(background):
    post = InstancePostProcessor(roi_settings={'enabled': False}, surface_settings={
        'height_mode': 'plane', 'require_reference_identity': True, 'min_valid_points': 30,
        'mask_erode_px': 1})
    post.surface_reference = SurfaceReference.from_frames([background], plane_roi=np.ones(background.texture_bgr.shape[:2], bool))
    return post


class ConfigurationRegressionTests(unittest.TestCase):
    def test_old_threshold_key_is_not_silently_ignored(self):
        config = load_config()
        config['postprocess']['surface']['min_raised_ratio'] = .2
        with self.assertRaisesRegex(ValueError, 'min_raised_ratio_mask'):
            validate_config(config)

    def test_numeric_and_boolean_gates_reject_invalid_values(self):
        for key, value in [('max_candidates', 1.5), ('max_tilt_deg', 95), ('enabled', 'false'), ('min_valid_ratio', float('nan'))]:
            config = load_config()
            config['picking'][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_class_count_order_and_noncontiguous_ids(self):
        for names in ({0: 'a'}, {0: 'a', 1: 'b', 2: 'c'}, {1: 'a', 2: 'b'}, ['b', 'a']):
            with self.assertRaises(ValueError):
                validate_model_classes(names, ['a', 'b'])
        validate_model_classes({0: 'a', 1: 'b'}, ['a', 'b'])

    def test_mouse_right_pane_and_invalid_border(self):
        path = Path(__file__).parents[1] / 'scripts/11_inspect_depth.py'
        spec = importlib.util.spec_from_file_location('inspector', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.pane_pixel(230, 50, 200, 160), (30, 50))
        self.assertEqual(module.pane_pixel(400, 50, 200, 160), (-1, -1))
        self.assertIsNone(module.robust_value(synthetic_frame(), -1, 5))

    def test_clahe_repeated_replay_is_idempotent_and_raw_unchanged(self):
        frame = synthetic_frame()
        frame.texture_raw = np.arange(160*200, dtype=np.uint16).reshape(160, 200)
        raw = frame.texture_raw.copy()
        settings = {'clahe': {'enabled': True, 'clip_limit': 1.5, 'tile_grid_size': [8, 8]}}
        first = prepare_frame_texture(frame, settings).copy()
        second = prepare_frame_texture(frame, settings)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(raw, frame.texture_raw)


class ReferenceGeometryTests(unittest.TestCase):
    def test_tilted_floor_reports_perpendicular_height_and_round_trips(self):
        n = np.array([np.sin(np.deg2rad(25)), 0., -np.cos(np.deg2rad(25))])
        background = synthetic_frame(normal=n, offset=800.)
        reference = SurfaceReference.from_frames([background], plane_roi=np.ones((160, 200), bool))
        frame = copy.deepcopy(background)
        mask = np.zeros((160, 200), bool)
        mask[40:90, 60:110] = True
        rays = frame.point_cloud_mm[mask]/frame.point_cloud_mm[mask, 2:3]
        frame.point_cloud_mm[mask] = rays*((20-800)/(rays @ n))[:, None]
        check = reference.evaluate_mask(frame, mask, {'height_mode': 'plane', 'mask_erode_px': 0})
        self.assertEqual(check.status, 'PASS')
        self.assertAlmostEqual(check.height_percentile_mm, 20., places=2)
        axial = reference.evaluate_mask(frame, mask, {'mask_erode_px': 0})
        self.assertGreater(abs(axial.height_percentile_mm-20), 1.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'plane.npz'
            reference.save(path)
            loaded = SurfaceReference.load(path)
            self.assertEqual(loaded.identity, reference.identity)
            np.testing.assert_allclose(loaded.plane['normal'], n, atol=1e-5)

    def test_same_shape_different_crop_is_unknown(self):
        background = synthetic_frame()
        reference = SurfaceReference.from_frames([background])
        frame = copy.deepcopy(background)
        d = put_object(frame)
        frame.info['acquisition']['roi_xyxy'] = [1, 0, 201, 160]
        check = reference.evaluate_mask(frame, d.mask, {})
        self.assertEqual(check.reason, 'SURFACE_IDENTITY_MISMATCH')
        self.assertEqual(check.status, 'UNKNOWN')

    def test_legacy_reference_cannot_satisfy_strict_identity(self):
        reference = SurfaceReference.from_frames([synthetic_frame()])
        reference.identity = {}
        frame = synthetic_frame()
        d = put_object(frame)
        self.assertEqual(reference.evaluate_mask(frame, d.mask, {'require_reference_identity': True}).status, 'UNKNOWN')

    def test_unmoved_wall_above_floor_is_not_a_new_object(self):
        background = synthetic_frame()
        wall = put_object(background, z=740)
        floor = ~wall.mask
        ref = SurfaceReference.from_frames([background], plane_roi=floor)
        result = ref.evaluate_mask(background, wall.mask, {'height_mode': 'plane'})
        self.assertEqual(result.status, 'REJECT')
        self.assertGreater(result.height_percentile_mm, 50)

    def test_plane_fit_rejects_collinear_or_missing_data(self):
        for points in (np.zeros((30, 3)), np.column_stack((np.arange(50), np.zeros(50), np.ones(50)*800))):
            with self.assertRaises(ValueError):
                fit_plane(points)


class TargetGeometryTests(unittest.TestCase):
    def test_u_shaped_anchor_is_one_measured_surface_point(self):
        frame = synthetic_frame()
        mask = np.zeros((160, 200), bool)
        mask[30:90, 50:65] = True
        mask[30:90, 95:110] = True
        mask[75:90, 50:110] = True
        anchor, (u, v), _ = surface_anchor(frame.point_cloud_mm, mask)
        self.assertTrue(mask[v, u])
        np.testing.assert_array_equal(anchor, frame.point_cloud_mm[v, u])
        y, x = np.nonzero(mask)
        self.assertFalse(mask[round(y.mean()), round(x.mean())])

    def test_mixed_normal_signs_and_rotation_conventions(self):
        frame = synthetic_frame()
        d = put_object(frame)
        frame.normal_map[::2] *= -1
        poses = generate_poses(frame, result_for(frame, d), picking_settings(), {'mask_erode_px': 2})
        pose = poses[0]
        self.assertIn('NormalMap', pose['normal_source'])
        np.testing.assert_allclose(pose['surface_normal_camera'], [0, 0, -1], atol=1e-6)
        r = np.array(pose['rotation_camera'])
        np.testing.assert_allclose(r.T@r, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(np.linalg.det(r), 1.)
        g = to_graspnet_rotation(r)
        self.assertAlmostEqual(np.linalg.det(g), 1.)
        np.testing.assert_allclose(g[:, 0], r[:, 2])

    def test_large_normalmap_error_uses_measured_geometry(self):
        frame = synthetic_frame()
        d = put_object(frame)
        frame.normal_map[:] = [1, 0, 0]
        poses = generate_poses(frame, result_for(frame, d), picking_settings(), {})
        self.assertIn('disagrees', poses[0]['normal_source'])

    def test_lower_surface_contamination_removed_but_recorded(self):
        frame = synthetic_frame()
        d = put_object(frame, region=(40, 65, 60, 90))
        lower = np.zeros(d.mask.shape, bool)
        lower[60:65, 60:90] = True
        ray = frame.point_cloud_mm[lower]/frame.point_cloud_mm[lower, 2:3]
        frame.point_cloud_mm[lower] = ray*780.
        poses = generate_poses(frame, result_for(frame, d), picking_settings(), {'mask_erode_px': 2})
        self.assertEqual(poses[0]['surface_anchor_xyz_mm'][2], 740.)
        self.assertGreater(poses[0]['discarded_for_geometry'], 100)
        self.assertTrue(poses[0]['discarded_points_still_used_for_collision'])

    def test_square_top_retains_multiple_yaws(self):
        frame = synthetic_frame()
        d = put_object(frame, region=(40, 60, 60, 80), name='khoi_lap_phuong')
        poses = generate_poses(frame, result_for(frame, d), picking_settings(), {})
        self.assertGreaterEqual(len(poses), 4)
        self.assertTrue(all(p['yaw_ambiguous'] for p in poses))

    def test_opening_budget_and_crop_edge_are_rejections(self):
        frame = synthetic_frame()
        d = put_object(frame, region=(40, 78, 60, 98))
        with self.assertRaisesRegex(ValueError, 'OPENING_LIMIT'):
            generate_poses(frame, result_for(frame, d), picking_settings(), {})
        d = put_object(frame, region=(0, 30, 60, 80))
        with self.assertRaisesRegex(ValueError, 'CROP_BOUNDARY'):
            generate_poses(frame, result_for(frame, d), picking_settings(), {})

    def test_tilt_is_rejected_not_clamped(self):
        n = np.array([np.sin(np.deg2rad(35)), 0., -np.cos(np.deg2rad(35))])
        frame = synthetic_frame(normal=n, offset=600)
        mask = np.zeros((160, 200), bool)
        mask[40:70, 60:90] = True
        d = InstanceDetection(0, 0, 'hop_anibox', .9, (60, 40, 90, 70), mask)
        with self.assertRaisesRegex(ValueError, 'TILT'):
            generate_poses(frame, result_for(frame, d), picking_settings(), {})


class ClearanceTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_frame()
        self.anchor = np.array([0., 0., 700.])
        self.rotation = np.eye(3)
        self.boxes = [('test_finger', np.array([-10., -10., -20.]), np.array([10., 10., 10.]))]

    def check(self):
        return check_boxes(self.frame, self.anchor, self.rotation, self.boxes, fit_projection(self.frame))

    def test_empty_measured_approach_is_observed_clear(self):
        self.assertEqual(self.check()['status'], 'OBSERVED_APPROACH_CLEAR')

    def test_missing_ray_is_unknown_not_free(self):
        self.frame.point_cloud_mm[80, 100] = 0
        self.assertEqual(self.check()['status'], 'UNKNOWN')

    def test_foreground_occlusion_is_unknown_not_free(self):
        self.frame.point_cloud_mm[80, 100] = [0, 0, 650]
        self.assertEqual(self.check()['status'], 'UNKNOWN')

    def test_off_center_obstacle_inside_finger_volume_is_blocked(self):
        # Obstacle is away from the center ray. Two sample points could miss it.
        y, x = 82, 103
        self.frame.point_cloud_mm[y, x] = [(x-100)*700/300, (y-80)*700/300, 700]
        self.assertEqual(self.check()['status'], 'BLOCKED')

    def test_fov_boundary_cannot_be_certified(self):
        self.anchor[0] = 230.
        self.assertEqual(self.check()['reason'], 'VOLUME_OUTSIDE_FOV')

    def test_missing_dimensions_and_disabled_check_are_unknown(self):
        for settings in ({}, {'enabled': False}):
            result = check_approach(self.frame, self.anchor, self.rotation, 50., settings)
            self.assertEqual(result['status'], 'UNKNOWN')
            self.assertFalse(result['robot_ready'])

    def test_nonprojective_cloud_fails_visibility_inference(self):
        self.frame.point_cloud_mm[40, 50, 0] += 100
        self.assertIsNone(fit_projection(self.frame))


class SelectionTests(unittest.TestCase):
    def test_blocked_first_target_falls_back_and_display_is_one_object(self):
        frame = synthetic_frame()
        first = put_object(frame)
        second = put_object(frame, 1, (95, 115, 125, 155), 750)
        items = [result_for(frame, first, 60), result_for(frame, second, 50)]
        def clearance(_frame, anchor, *_args):
            return {'status': 'BLOCKED' if anchor[0] < 0 else 'OBSERVED_APPROACH_CLEAR', 'robot_ready': False}
        with patch('phoxi_yolo3d.picking.check_approach', side_effect=clearance):
            plan = plan_single_target(frame, items, picking_settings(), {}, offline=True)
        self.assertEqual(plan['target_instance_id'], 1)
        self.assertEqual(len(plan['attempts']), 2)
        self.assertEqual([x.detection.instance_id for x in visible_results(items, plan)], [1])
        self.assertEqual(len(visible_results(items, plan, debug=True)), 2)
        self.assertFalse(plan['robot_ready'])

    def test_missing_geometry_is_explicit_preview(self):
        frame = synthetic_frame()
        d = put_object(frame)
        plan = plan_single_target(frame, [result_for(frame, d)], picking_settings(), {}, offline=True)
        self.assertEqual(plan['status'], 'PREVIEW_UNVALIDATED')
        self.assertFalse(plan['pose']['full_object_width_known'])
        self.assertEqual(plan['pose']['clearance']['reason'], 'GRIPPER_GEOMETRY_UNCONFIGURED')
        json.dumps(plan, allow_nan=False)

    def test_sparse_detection_retained_but_not_selected(self):
        frame = synthetic_frame()
        d = put_object(frame)
        y, x = np.nonzero(d.mask)
        frame.point_cloud_mm[y[:400], x[:400]] = 0
        plan = plan_single_target(frame, [result_for(frame, d)], picking_settings(), {}, offline=True)
        self.assertIsNone(plan['target_instance_id'])
        self.assertIn('LOW_3D_COVERAGE', plan['ranking'][0]['reasons'])

    def test_stale_unknown_future_and_offline_timestamps(self):
        frame = synthetic_frame()
        frame.timestamp_utc = (datetime.now(timezone.utc)-timedelta(seconds=10)).isoformat()
        self.assertEqual(freshness(frame, 5000)['status'], 'STALE')
        d = put_object(frame)
        plan = plan_single_target(frame, [result_for(frame, d)], picking_settings(), {})
        self.assertIsNone(plan['target_instance_id'])
        self.assertEqual(freshness(frame, 5000, offline=True)['status'], 'OFFLINE_REPLAY')
        frame.timestamp_utc = 'invalid'
        self.assertEqual(freshness(frame, 5000)['status'], 'UNKNOWN')
        frame.timestamp_utc = (datetime.now(timezone.utc)+timedelta(seconds=10)).isoformat()
        self.assertEqual(freshness(frame, 5000)['status'], 'FUTURE_CLOCK')

    def test_pipeline_computes_full_measurement_only_for_selected_target(self):
        background = synthetic_frame()
        frame = copy.deepcopy(background)
        detections = [put_object(frame), put_object(frame, 1, (95, 115, 125, 155), 750)]
        pipeline = PhoXiYoloPipeline(FrozenSegmenter(detections), {'min_valid_points': 30}, plane_post(background),
                                   picking_settings=picking_settings(), offline=True)
        accepted, rejected = pipeline.process_all(frame)
        self.assertFalse(rejected)
        target = pipeline.last_picking['target_instance_id']
        self.assertIsNotNone(target)
        self.assertEqual(sum(x.measurement.measurement_kind == 'full_visible_surface' for x in accepted), 1)
        self.assertEqual(len(accepted), 2)


if __name__ == '__main__':
    unittest.main()
