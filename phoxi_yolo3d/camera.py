from __future__ import annotations

from datetime import datetime, timezone
from collections.abc import Sequence
from typing import Any

import numpy as np

from .image_utils import normalize_texture
from .types import Frame3D


class PhoXiCameraError(RuntimeError):
    pass


def _import_phoxi_api():
    try:
        from phoxi_api import PhoXiControl, PhoXiDevice
    except ImportError as exc:
        raise PhoXiCameraError(
            "Khong import duoc phoxi_api. Hay kich hoat venv va chay: pip install -e ."
        ) from exc
    return PhoXiControl, PhoXiDevice


def _extract_device_id(device_info: dict[str, Any]) -> str | None:
    candidate_keys = (
        "hw_identification",
        "HWIdentification",
        "hardware_identification",
        "HardwareIdentification",
        "device_id",
        "DeviceID",
        "id",
        "ID",
        "name",
        "Name",
    )
    for key in candidate_keys:
        value = device_info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class PhoXiCamera:
    """Small, explicit wrapper around the official phoxi-api package."""

    REQUIRED_COMPONENTS = ("PointCloud", "DepthMap", "Texture")
    OPTIONAL_COMPONENTS = ("ConfidenceMap", "NormalMap")

    def __init__(
        self,
        device_id: str | None = None,
        address: str | None = None,
        api_lib_path: str | None = None,
        refresh_discovery: bool = True,
        timeout_ms: int = 30_000,
        force_camera_space: bool = True,
        texture_source: str | None = None,
        led_power: int | None = None,
        texture_lower_percentile: float = 1.0,
        texture_upper_percentile: float = 99.5,
        roi_xyxy: Sequence[int] | None = None,
        calibration_id: str | None = None,
    ) -> None:
        self.device_id = device_id
        self.calibration_id = calibration_id
        self.address = address
        self.api_lib_path = api_lib_path
        self.refresh_discovery = refresh_discovery
        self.timeout_ms = int(timeout_ms)
        self.force_camera_space = bool(force_camera_space)
        self.texture_source = (
            None
            if texture_source is None or not str(texture_source).strip()
            else str(texture_source).strip()
        )
        self.led_power = None if led_power is None else int(led_power)
        if self.led_power is not None and not 0 <= self.led_power <= 4095:
            raise ValueError("camera.led_power phai nam trong khoang 0..4095.")
        self.texture_lower_percentile = float(texture_lower_percentile)
        self.texture_upper_percentile = float(texture_upper_percentile)
        if roi_xyxy is None:
            self.roi_xyxy: tuple[int, int, int, int] | None = None
        else:
            if len(roi_xyxy) != 4:
                raise ValueError("camera.roi_xyxy phai gom [x1, y1, x2, y2].")
            self.roi_xyxy = tuple(int(value) for value in roi_xyxy)
        self.control = None
        self.device = None
        self.component_errors: dict[str, Any] = {}
        self.coordinate_space_errors: dict[str, Any] = {}
        self.capture_setting_errors: dict[str, Any] = {}
        self.applied_capture_settings: dict[str, Any] = {}

    @staticmethod
    def create_control(api_lib_path: str | None = None):
        PhoXiControl, _ = _import_phoxi_api()
        return PhoXiControl(api_lib_path=api_lib_path)

    @classmethod
    def list_devices(
        cls, api_lib_path: str | None = None, refresh: bool = True
    ) -> list[dict[str, Any]]:
        control = cls.create_control(api_lib_path)
        if not control.is_phoxicontrol_running():
            raise PhoXiCameraError("PhoXi Control chua chay.")
        return control.get_device_list(refresh=refresh)

    def _resolve_device_id(self) -> str:
        if self.device_id:
            return self.device_id
        devices = self.control.get_device_list(refresh=self.refresh_discovery)
        if not devices:
            raise PhoXiCameraError(
                "PhoXi Control khong tim thay camera. Kiem tra nguon, cap Ethernet va IP."
            )
        if len(devices) > 1:
            ids = [_extract_device_id(item) for item in devices]
            raise PhoXiCameraError(
                "Tim thay nhieu camera/file-camera. Hay dat camera.device_id trong config: "
                + ", ".join(str(value) for value in ids)
            )
        resolved = _extract_device_id(devices[0])
        if not resolved:
            raise PhoXiCameraError(
                f"Khong doc duoc device ID tu thong tin camera: {devices[0]!r}"
            )
        return resolved

    def open(self) -> "PhoXiCamera":
        _, PhoXiDevice = _import_phoxi_api()
        self.control = self.create_control(self.api_lib_path)
        if not self.control.is_phoxicontrol_running():
            raise PhoXiCameraError(
                "PhoXi Control chua chay. Mo PhoXi Control roi chay lai chuong trinh."
            )
        self.device_id = self._resolve_device_id()
        if self.address:
            self.device = self.control.connect(
                self.device_id, address=self.address, timeout_ms=self.timeout_ms
            )
        else:
            self.device = self.control.connect(self.device_id, timeout_ms=self.timeout_ms)

        self.device.logout_on_exit = True
        self.device.stop_acquisition_on_exit = True
        if self.device.is_acquiring():
            self.device.stop_acquisition()

        info = self.device.info()
        if not bool(info.get("file_camera", False)):
            capture_settings: dict[str, Any] = {}
            if self.texture_source is not None:
                capture_settings["CapturingSettings/TextureSource"] = self.texture_source
            if self.led_power is not None:
                capture_settings["CapturingSettings/LEDPower"] = self.led_power
            if capture_settings:
                capture_errors = self.device.set_settings(capture_settings) or {}
                self.capture_setting_errors = dict(capture_errors)
                fatal_capture_errors = {
                    key: value
                    for key, value in self.capture_setting_errors.items()
                    if value
                }
                if fatal_capture_errors:
                    raise PhoXiCameraError(
                        "Khong ap dung duoc cau hinh TextureSource/LEDPower: "
                        f"{fatal_capture_errors}"
                    )
                self.applied_capture_settings = dict(capture_settings)
            if self.force_camera_space:
                coordinate_errors = self.device.set_settings(
                    {
                        "CoordinatesSettings/CoordinateSpace": "CameraSpace",
                        "CoordinatesSettings/CameraSpace": "PrimaryCamera",
                    }
                ) or {}
                self.coordinate_space_errors = dict(coordinate_errors)
                fatal_coordinate_errors = {
                    key: value
                    for key, value in self.coordinate_space_errors.items()
                    if key.endswith("/CoordinateSpace") and value
                }
                if fatal_coordinate_errors:
                    raise PhoXiCameraError(
                        "Khong chuyen duoc PhoXi ve CameraSpace; khong the bao dam "
                        f"khoang cach tu camera: {fatal_coordinate_errors}"
                    )
            requested = {
                name: True for name in self.REQUIRED_COMPONENTS + self.OPTIONAL_COMPONENTS
            }
            errors = self.device.set_frame_settings(requested) or {}
            self.component_errors = dict(errors)
            required_errors = {
                key: value
                for key, value in self.component_errors.items()
                if key in self.REQUIRED_COMPONENTS and value
            }
            if required_errors:
                raise PhoXiCameraError(
                    f"Khong bat duoc cac output bat buoc cua PhoXi: {required_errors}"
                )

        self.device.set_trigger_mode(PhoXiDevice.TriggerMode.SOFTWARE)
        self.device.start_acquisition()
        return self

    def close(self) -> None:
        if self.device is None:
            return
        try:
            self.device.disconnect(logout=True, stop_acquisition=True)
        finally:
            self.device = None

    def __enter__(self) -> "PhoXiCamera":
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def capture(self) -> Frame3D:
        if self.device is None:
            raise PhoXiCameraError("Camera chua duoc open().")
        capture_started = datetime.now(timezone.utc).isoformat()
        frame_id = self.device.trigger_frame(True, True)
        frame = self.device.get_frame(frame_id, timeout_ms=self.timeout_ms)
        if not bool(getattr(frame, "Successful", True)):
            messages = getattr(frame, "Messages", None)
            raise PhoXiCameraError(f"PhoXi chup frame that bai: {messages}")

        raw_texture = getattr(frame, "Texture", None)
        point_cloud = getattr(frame, "PointCloud", None)
        if raw_texture is None or point_cloud is None:
            raise PhoXiCameraError("Frame khong co Texture hoac PointCloud.")

        raw_texture = np.squeeze(np.array(raw_texture, copy=True))
        point_cloud = np.asarray(point_cloud, dtype=np.float32).copy()
        depth = getattr(frame, "DepthMap", None)
        confidence = getattr(frame, "ConfidenceMap", None)
        normals = getattr(frame, "NormalMap", None)

        depth_array = (
            None if depth is None else np.squeeze(np.asarray(depth)).astype(np.float32)
        )
        confidence_array = (
            None
            if confidence is None
            else np.squeeze(np.asarray(confidence)).astype(np.float32)
        )
        normal_array = (
            None if normals is None else np.asarray(normals, dtype=np.float32).copy()
        )

        source_height, source_width = point_cloud.shape[:2]
        if raw_texture.shape[:2] != (source_height, source_width):
            raise PhoXiCameraError(
                "Texture va PointCloud khong cung kich thuoc truoc khi crop: "
                f"texture={raw_texture.shape[:2]}, point_cloud={(source_height, source_width)}"
            )

        frame_info = dict(getattr(frame, "Info", {}) or {})
        frame_info["capture_started_utc"] = capture_started
        frame_info["acquisition"] = {
            "device_id": str(self.device_id),
            "coordinate_space": "CameraSpace/PrimaryCamera" if self.force_camera_space else "UNVERIFIED",
            "roi_xyxy": list(self.roi_xyxy) if self.roi_xyxy else None,
            "source_shape_hw": [source_height, source_width],
            "calibration_id": self.calibration_id,
            "texture_source": self.texture_source, "led_power": self.led_power,
        }
        if self.roi_xyxy is not None:
            x1, y1, x2, y2 = self.roi_xyxy
            if not (
                0 <= x1 < x2 <= source_width
                and 0 <= y1 < y2 <= source_height
            ):
                raise PhoXiCameraError(
                    "camera.roi_xyxy nam ngoai frame: "
                    f"roi={self.roi_xyxy}, frame={(source_width, source_height)}"
                )

            def crop(array: np.ndarray | None) -> np.ndarray | None:
                if array is None:
                    return None
                return np.ascontiguousarray(array[y1:y2, x1:x2, ...])

            raw_texture = crop(raw_texture)
            point_cloud = crop(point_cloud)
            depth_array = crop(depth_array)
            confidence_array = crop(confidence_array)
            normal_array = crop(normal_array)
            frame_info["source_shape_hw"] = [source_height, source_width]
            frame_info["roi_xyxy"] = [x1, y1, x2, y2]

        texture_bgr = normalize_texture(
            raw_texture,
            self.texture_lower_percentile,
            self.texture_upper_percentile,
        )

        captured = Frame3D(
            texture_bgr=texture_bgr,
            texture_raw=raw_texture,
            point_cloud_mm=point_cloud,
            depth_mm=depth_array,
            confidence_map=confidence_array,
            normal_map=normal_array,
            frame_id=int(frame_id),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            info=frame_info,
        )
        captured.validate()
        return captured
