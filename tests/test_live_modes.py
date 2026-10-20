import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from phoxi_yolo3d.config import load_config
from phoxi_yolo3d.storage import load_capture
from phoxi_yolo3d.types import Frame3D, InstanceDetection

try:
    import cv2
except ImportError:
    cv2 = None


def load_live():
    spec = importlib.util.spec_from_file_location('live_under_test', Path(__file__).parents[1] / 'scripts' / '07_run_live.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(cv2 is None, 'OpenCV required for dashboard integration test')
class LiveModesTests(unittest.TestCase):
    def test_manual_debug_fill_pause_save_do_not_capture_again(self):
        live = load_live()
        camera_instances = []
        prediction_calls = []

        class Camera:
            device_id = 'MOCK'

            def __init__(self, **kwargs):
                self.calls = 0
                camera_instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def capture(self):
                self.calls += 1
                image = np.full((180, 320, 3), 100, dtype=np.uint8)
                cloud = np.zeros((180, 320, 3), dtype=np.float32)
                cloud[:, :, 2] = 1000
                return Frame3D(image, cloud, depth_mm=cloud[:, :, 2], frame_id=self.calls)

        class Segmenter:
            def __init__(self, *_args):
                pass

            def predict(self, image):
                prediction_calls.append(1)
                mask = np.zeros(image.shape[:2], dtype=bool)
                mask[60:100, 120:170] = True
                return [InstanceDetection(0, 3, 'khoi_lap_phuong', .98, (120, 60, 170, 100), mask)]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config()
            weights = root / 'dummy.pt'
            weights.touch()
            config['yolo']['weights'] = str(weights)
            config['runtime']['output_dir'] = str(root / 'results')
            config['runtime']['scan_mode'] = 'manual'
            config.setdefault('postprocess', {}).setdefault('roi', {})['enabled'] = False
            config['postprocess'].setdefault('surface', {})['enabled'] = False
            keys = [-1, ord('d'), ord('m'), ord('s'), ord('p'), -1, ord('p'), ord('n'), -1, ord('q')]
            with patch.object(live, 'load_config', return_value=config), \
                 patch.object(live, 'PhoXiCamera', Camera), \
                 patch.object(live, 'YOLOInstanceSegmenter', Segmenter), \
                 patch('sys.argv', ['07_run_live.py']), \
                 patch.object(cv2, 'imshow'), \
                 patch.object(cv2, 'waitKey', side_effect=keys), \
                 patch.object(cv2, 'destroyAllWindows'), redirect_stdout(io.StringIO()):
                self.assertEqual(live.main(), 0)
            self.assertEqual(camera_instances[0].calls, 2)
            self.assertEqual(len(prediction_calls), 2)
            scans = list(root.glob('results/*/inspection_*/scans/frame.npz'))
            self.assertEqual(len(scans), 1)
            self.assertEqual(load_capture(scans[0]).frame_id, 1)
            report = json.loads((scans[0].parents[1] / 'result.json').read_text())
            self.assertEqual(report['frame_id'], 1)
            self.assertEqual(report['verified_3d_count'], 0)
            self.assertTrue((scans[0].parents[1] / 'dashboard.png').is_file())


if __name__ == '__main__':
    unittest.main()
