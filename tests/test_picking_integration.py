import copy
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from phoxi_yolo3d.config import load_config
from phoxi_yolo3d.pipeline import PhoXiYoloPipeline
from phoxi_yolo3d.picking_visualization import draw_picking
from phoxi_yolo3d.replay import FrozenSegmenter, load_frozen_inspection
from phoxi_yolo3d.roi import PolygonROI
from test_picking import synthetic_frame, put_object, plane_post


ROOT = Path(__file__).parents[1]


def live_module():
    spec = importlib.util.spec_from_file_location('picking_live', ROOT / 'scripts/07_run_live.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene():
    background = synthetic_frame()
    frame = copy.deepcopy(background)
    detections = [put_object(frame), put_object(frame, 1, (95, 115, 125, 155), 750)]
    post = plane_post(background)
    post.roi = PolygonROI([(0, 0), (1, 0), (1, 1), (0, 1)])
    post.roi_settings['enabled'] = True
    config = load_config(ROOT / 'config/picking.yaml')
    return frame, detections, post, config


class PickingIntegrationTests(unittest.TestCase):
    def test_save_move_and_cli_replay_without_yolo_or_camera(self):
        frame, detections, post, config = scene()
        pipeline = PhoXiYoloPipeline(FrozenSegmenter(detections), config['measurement'], post,
                                    picking_settings=config['picking'], offline=True)
        accepted, rejected = pipeline.process_all(frame)
        self.assertIsNotNone(pipeline.last_picking['target_instance_id'])
        record = {'frame_id': frame.frame_id, 'timestamp_utc': frame.timestamp_utc,
                  'objects': [r.to_dict() for r in accepted], 'rejected': [r.to_dict() for r in rejected],
                  'picking': pipeline.last_picking, 'robot_ready': False}
        image = draw_picking(frame, accepted, rejected, pipeline.last_picking)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = live_module().save_inspection(Path(directory), frame, record, accepted, rejected, image, config, post)
            moved = Path(directory) / 'moved-inspection'
            shutil.move(str(root), moved)
            loaded, frozen = load_frozen_inspection(moved)
            np.testing.assert_array_equal(loaded.point_cloud_mm, frame.point_cloud_mm)
            self.assertEqual(len(frozen.predict(None)), 2)
            environment = {**os.environ, 'PYTHONPATH': str(ROOT)}
            completed = subprocess.run([sys.executable, str(ROOT/'scripts/15_replay_picking.py'), str(moved)],
                cwd=directory, env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((moved/'picking_replay/picking.json').read_text())
            self.assertEqual(report['picking']['target_instance_id'], pipeline.last_picking['target_instance_id'])
            self.assertEqual(report['picking']['freshness']['status'], 'OFFLINE_REPLAY')
            self.assertTrue((moved/'picking_replay/target.png').exists())
            with (moved/'masks.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'HASH_MISMATCH'):
                load_frozen_inspection(moved)

    def test_live_single_target_debug_and_save_do_not_recapture_or_rerun(self):
        frame, detections, post, config = scene()
        live = live_module()
        captures, predictions = [], []
        class Camera:
            device_id = 'TEST-CAMERA'
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *_args): pass
            def capture(self):
                captures.append(1)
                return copy.deepcopy(frame)
        class Segmenter:
            def __init__(self, *_args): pass
            def predict(self, _image):
                predictions.append(1)
                return detections
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root/'fake.pt'
            weights.touch()
            config['yolo']['weights'] = str(weights)
            config['runtime']['output_dir'] = str(root/'results')
            with patch.object(live, 'load_config', return_value=config), \
                 patch.object(live, 'build_postprocessor', return_value=post), \
                 patch.object(live, 'PhoXiCamera', Camera), \
                 patch.object(live, 'YOLOInstanceSegmenter', Segmenter), \
                 patch.object(cv2, 'imshow'), patch.object(cv2, 'destroyAllWindows'), \
                 patch.object(cv2, 'waitKey', side_effect=[ord('d'), ord('m'), ord('s'), ord('p'), -1, ord('q')]), \
                 patch('sys.argv', ['07_run_live.py']), redirect_stdout(io.StringIO()):
                self.assertEqual(live.main(), 0)
            self.assertEqual(len(captures), 1)
            self.assertEqual(len(predictions), 1)
            reports = list(root.glob('results/*/inspection_*/result.json'))
            self.assertEqual(len(reports), 1)
            record = json.loads(reports[0].read_text())
            self.assertIsNotNone(record['picking']['target_instance_id'])
            self.assertEqual(len(record['objects']), 2)
            self.assertFalse(record['robot_ready'])


if __name__ == '__main__':
    unittest.main()
