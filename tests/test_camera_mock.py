import unittest
from unittest.mock import patch

import numpy as np

from phoxi_yolo3d.camera import PhoXiCamera


class FakeTriggerMode:
    SOFTWARE = "Software"


class FakePhoXiDeviceType:
    TriggerMode = FakeTriggerMode


class FakeFrame:
    Successful = True
    Messages = []
    Info = {"test": True}
    Texture = np.arange(80, dtype=np.uint16).reshape(8, 10)
    PointCloud = np.dstack(
        [
            np.zeros((8, 10), dtype=np.float32),
            np.zeros((8, 10), dtype=np.float32),
            np.full((8, 10), 650, dtype=np.float32),
        ]
    )
    DepthMap = np.full((8, 10), 650, dtype=np.float32)
    ConfidenceMap = np.ones((8, 10), dtype=np.float32)
    NormalMap = np.dstack(
        [
            np.zeros((8, 10), dtype=np.float32),
            np.zeros((8, 10), dtype=np.float32),
            np.ones((8, 10), dtype=np.float32),
        ]
    )


class FakeDevice:
    def __init__(self):
        self.logout_on_exit = False
        self.stop_acquisition_on_exit = False
        self.acquiring = False
        self.trigger_mode_value = None
        self.disconnected = False

    def info(self):
        return {"file_camera": False}

    def is_acquiring(self):
        return self.acquiring

    def stop_acquisition(self):
        self.acquiring = False

    def start_acquisition(self):
        self.acquiring = True

    def set_settings(self, settings):
        self.settings = settings
        return {}

    def set_frame_settings(self, settings):
        self.frame_settings = settings
        return {}

    def set_trigger_mode(self, value):
        self.trigger_mode_value = value

    def trigger_frame(self, wait_accept, wait_grabbing_end):
        self.trigger_args = (wait_accept, wait_grabbing_end)
        return 17

    def get_frame(self, frame_id, timeout_ms):
        self.get_frame_args = (frame_id, timeout_ms)
        return FakeFrame()

    def disconnect(self, logout, stop_acquisition):
        self.disconnected = True
        self.acquiring = False


class FakeControl:
    last_device = None

    def __init__(self, api_lib_path=None):
        self.api_lib_path = api_lib_path

    def is_phoxicontrol_running(self):
        return True

    def get_device_list(self, refresh=True):
        return [{"hw_identification": "PHOXI-M-TEST"}]

    def connect(self, device_id, timeout_ms=None, address=None):
        self.connect_args = (device_id, timeout_ms, address)
        FakeControl.last_device = FakeDevice()
        return FakeControl.last_device


class CameraMockTests(unittest.TestCase):
    @patch(
        "phoxi_yolo3d.camera._import_phoxi_api",
        return_value=(FakeControl, FakePhoXiDeviceType),
    )
    def test_discovery_connect_capture_and_disconnect(self, _patched):
        camera = PhoXiCamera(timeout_ms=1234)
        with camera:
            self.assertEqual(camera.device_id, "PHOXI-M-TEST")
            frame = camera.capture()
            self.assertEqual(frame.frame_id, 17)
            self.assertEqual(frame.texture_bgr.shape, (8, 10, 3))
            self.assertEqual(frame.point_cloud_mm.shape, (8, 10, 3))
            self.assertTrue(FakeControl.last_device.acquiring)
            self.assertEqual(FakeControl.last_device.trigger_mode_value, "Software")
        self.assertTrue(FakeControl.last_device.disconnected)

    @patch(
        "phoxi_yolo3d.camera._import_phoxi_api",
        return_value=(FakeControl, FakePhoXiDeviceType),
    )
    def test_capture_crops_every_aligned_component(self, _patched):
        camera = PhoXiCamera(roi_xyxy=(2, 1, 8, 7))
        with camera:
            frame = camera.capture()
        self.assertEqual(frame.texture_bgr.shape, (6, 6, 3))
        self.assertEqual(frame.texture_raw.shape, (6, 6))
        self.assertEqual(frame.point_cloud_mm.shape, (6, 6, 3))
        self.assertEqual(frame.depth_mm.shape, (6, 6))
        self.assertEqual(frame.confidence_map.shape, (6, 6))
        self.assertEqual(frame.normal_map.shape, (6, 6, 3))
        self.assertEqual(frame.info["source_shape_hw"], [8, 10])
        self.assertEqual(frame.info["roi_xyxy"], [2, 1, 8, 7])


if __name__ == "__main__":
    unittest.main()
