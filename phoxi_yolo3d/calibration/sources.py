"""Camera-only acquisition and explicit synthetic mode; no robot controller import."""
from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
from scipy.spatial.transform import Rotation

from ..types import Frame3D
from .board import BoardSpec
from .math3d import as_pose_zyz, inverse, transform


class PhoXiSource:
    def __init__(self, config):
        self.config, self.camera = config, None
        self.coordinate_verification = None

    @staticmethod
    def verify_coordinates(camera):
        """A fixed/read-only selector may reject writes while already correct.

        Verify BOTH actual values, rather than treating every set_settings error
        as proof that the current coordinate system is wrong.
        """
        expected = {"CoordinatesSettings/CoordinateSpace": "CameraSpace",
                    "CoordinatesSettings/CameraSpace": "PrimaryCamera"}
        write_errors = dict(camera.coordinate_space_errors or {})
        try:
            values, read_errors = camera.device.get_settings(list(expected))
        except Exception as exc:
            raise ValueError("Không đọc lại được hệ tọa độ PhoXi: " + str(exc) +
                             "; phản hồi khi ghi: " + repr(write_errors)) from exc
        values, read_errors = values or {}, read_errors or {}
        mismatch = {key: {"expected": want, "actual": values.get(key),
                          "read_error": read_errors.get(key)}
                    for key, want in expected.items()
                    if values.get(key) != want or read_errors.get(key)}
        if mismatch:
            raise ValueError("Hệ tọa độ đọc lại chưa đúng / chưa xác minh được: " + repr(mismatch) +
                             "; phản hồi khi ghi: " + repr(write_errors))
        return {"verified_by": "device_get_settings", "values": dict(values),
                "set_settings_messages": write_errors}

    def capture(self, spec=None):
        if self.camera is None:
            from ..camera import PhoXiCamera
            c, texture = self.config["camera"], self.config["texture"]
            camera = PhoXiCamera(device_id=c.get("device_id"), address=c.get("address"),
                api_lib_path=c.get("api_lib_path"), refresh_discovery=c.get("refresh_discovery", True),
                timeout_ms=int(c.get("timeout_ms", 30000)), force_camera_space=True,
                texture_source=c.get("texture_source"), led_power=c.get("led_power"),
                texture_lower_percentile=texture.get("lower_percentile", 1.),
                texture_upper_percentile=texture.get("upper_percentile", 99.5),
                roi_xyxy=None, calibration_id=c.get("calibration_id"))
            try:
                camera.open()
                if camera.device.info().get("file_camera", False):
                    raise ValueError("Hiệu chuẩn thật cần scanner vật lý. Dùng --diagnose-scan để kiểm tra NPZ đã ghi.")
                self.coordinate_verification = self.verify_coordinates(camera)
            except Exception:
                camera.close()
                raise
            self.camera = camera
        frame = self.camera.capture()
        frame.info["calibration_coordinate_verification"] = self.coordinate_verification
        return frame

    def close(self):
        if self.camera is not None:
            self.camera.close()
            self.camera = None
            self.coordinate_verification = None


class DemoSource:
    """Rendered board, measured-pixel geometry and known independent ground truth."""
    def __init__(self):
        self.index = 0
        self.T_base_camera = transform(Rotation.from_euler("x", 180, degrees=True).as_matrix(), [-400, 20, 1050])
        self.T_flange_board = transform(Rotation.from_euler("z", 20, degrees=True).as_matrix(), [30, -40, 120])

    def capture(self, spec: BoardSpec):
        import cv2
        i, self.index = self.index, self.index + 1
        r = Rotation.from_euler("XYZ", [16*np.sin(i*.9), 18*np.cos(i*.7), 12*np.sin(i*.5)], degrees=True).as_matrix()
        size = spec.square_mm
        w, h = 800, 600
        fx = 1150.
        k = np.array([[fx, 0, w/2], [0, fx, h/2], [0, 0, 1.]])
        # Keep the complete board visible across supported board sizes.
        distance = max(750., size * (spec.cols + 3) * fx / 620.) + 35*np.sin(i*.6)
        center = np.array([12*np.sin(i*.8), 15*np.cos(i*.5), distance])
        offset = np.array([(spec.cols - 1)*size/2, (spec.rows - 1)*size/2, 0.])
        t = center - r @ offset
        t_camera_board = transform(r, t)
        t_base_flange = self.T_base_camera @ t_camera_board @ inverse(self.T_flange_board)
        cell = 36
        canvas = np.full(((spec.rows+3)*cell, (spec.cols+3)*cell), 245, np.uint8)
        for y in range(spec.rows+1):
            for x in range(spec.cols+1):
                if (x+y) % 2 == 0:
                    canvas[(y+1)*cell:(y+2)*cell, (x+1)*cell:(x+2)*cell] = 15
        # Inner origin is at pixel (2*cell,2*cell) on the source canvas.
        homography = k @ np.column_stack((r[:, 0]*size/cell, r[:, 1]*size/cell,
                                         t - 2*size*r[:, 0] - 2*size*r[:, 1]))
        image = cv2.warpPerspective(canvas, homography, (w,h), flags=cv2.INTER_LINEAR, borderValue=80)
        mask = cv2.warpPerspective(np.full_like(canvas,255), homography, (w,h), flags=cv2.INTER_NEAREST) > 0
        yy, xx = np.mgrid[:h,:w]
        rays = np.stack(((xx-w/2)/fx, (yy-h/2)/fx, np.ones((h,w))), axis=-1)
        normal = r[:, 2]
        depth = np.full((h,w), distance + 200)
        depth[mask] = (normal @ t) / (rays[mask] @ normal)
        rng = np.random.default_rng(i+17)
        cloud = rays * depth[..., None]
        cloud[mask] += rng.normal(0,.025,size=(int(mask.sum()),3))
        stamp = datetime.now(timezone.utc).isoformat()
        projected_origin = k @ t
        info = {"calibration_demo": True, "capture_started_utc": stamp,
                "demo_robot_pose_xyzabc": as_pose_zyz(t_base_flange),
                "demo_origin_uv": (projected_origin[:2]/projected_origin[2]).tolist(),
                "demo_T_camera_board": t_camera_board.tolist(),
                "acquisition": {"device_id": "DEMO_PHOXI", "coordinate_space": "CameraSpace/PrimaryCamera",
                                "source_shape_hw": [h,w], "roi_xyxy": None, "calibration_id": "DEMO_ONLY"}}
        return Frame3D(cv2.cvtColor(image,cv2.COLOR_GRAY2BGR),cloud.astype(np.float32),
                       texture_raw=image,depth_mm=depth.astype(np.float32),frame_id=i,timestamp_utc=stamp,info=info)

    def close(self):
        pass
