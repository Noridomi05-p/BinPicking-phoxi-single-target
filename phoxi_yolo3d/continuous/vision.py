from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from ..camera import PhoXiCamera
from ..calibration.sources import PhoXiSource
from ..config import project_path
from ..detector import YOLOInstanceSegmenter
from ..pipeline import PhoXiYoloPipeline
from ..postprocess import build_postprocessor
from ..geometry import frame_identity, identity_status
from ..storage import _json_safe, save_bgr_png, save_capture
from ..visualization import draw_results, depth_preview, make_dashboard
from ..picking_visualization import draw_picking
from .contract import integer

CAMERA_KEYS = {
    'texture_source': 'CapturingSettings/TextureSource',
    'shutter_multiplier': 'CapturingSettings/ShutterMultiplier',
    'laser_power': 'CapturingSettings/LaserPower',
    'led_power': 'CapturingSettings/LEDPower',
}


def reference_diagnostics(frame, reference, cfg):
    saved = deepcopy(getattr(reference, 'identity', None) or {})
    actual = frame_identity(frame)
    differences = {key: dict(saved=saved.get(key), current=actual.get(key))
                   for key in sorted(set(saved) | set(actual)) if saved.get(key) != actual.get(key)}
    return dict(status=identity_status(saved, frame), differences=differences,
                saved_identity=saved, frame_identity=actual,
                reference_path=str(project_path(cfg.get('postprocess', {}).get('surface', {}).get('path',
                                               'data/calibration/bin_surface.npz'), cfg)),
                run_live_capture_settings={key: cfg['camera'].get(key) for key in ('texture_source', 'led_power')})


def override_settings(values):
    result = {}
    for key, path in CAMERA_KEYS.items():
        value = values[key]
        if key in ('laser_power', 'led_power'):
            value = integer(value, 0, 4095, key.upper())
        elif key == 'shutter_multiplier':
            value = integer(value, 1, 50, 'SHUTTER_MULTIPLIER')
        elif value not in ('LED', 'Computed', 'Laser'):
            raise ValueError('INVALID_TEXTURE_SOURCE')
        result[path] = value
    return result


def configure_camera(camera, policy):
    """No capture-setting writes in PhoXi Control mode; always verify metric frame."""
    coordinates = PhoXiSource.verify_coordinates(camera)
    paths = list(CAMERA_KEYS.values())
    writes = {}
    if policy['camera_settings_source'] == 'config':
        writes = override_settings(policy['camera_overrides'])
        camera.device.stop_acquisition()
        try:
            write_errors = camera.device.set_settings(writes) or {}
            values, errors = camera.device.get_settings(paths)
            values, errors = values or {}, errors or {}
            mismatch = {key: {'expected': value, 'actual': values.get(key),
                              'read_error': errors.get(key), 'write_message': write_errors.get(key)}
                        for key, value in writes.items()
                        if errors.get(key) or values.get(key) != value}
            if mismatch:
                raise ValueError('CAMERA_SETTINGS_NOT_VERIFIED:'+repr(mismatch))
        finally:
            camera.device.start_acquisition()
    else:
        try:
            values, errors = camera.device.get_settings(paths)
            values, errors = values or {}, errors or {}
        except Exception as exc:
            # Illumination readback is optional when deliberately preserving PhoXi Control.
            # Coordinate readback above is mandatory in both modes.
            values, errors = {}, {'capture_settings_readback': str(exc)}
            # Some SDKs reject a query containing an unsupported optional key.
            # Recover independently readable values without inventing a readback.
            for path in paths:
                try:
                    one, one_errors = camera.device.get_settings([path])
                    values.update(one or {})
                    errors.update(one_errors or {})
                except Exception as one_exc:
                    errors[path] = str(one_exc)
    return dict(source=policy['camera_settings_source'], values=values,
                read_messages=errors, writes=writes, coordinates=coordinates)


def open_configured_camera(cfg, policy):
    c, t = cfg['camera'], cfg['texture']
    camera = PhoXiCamera(device_id=c.get('device_id'), address=c.get('address'),
        api_lib_path=c.get('api_lib_path'), refresh_discovery=c.get('refresh_discovery', True),
        timeout_ms=int(c.get('timeout_ms', 30000)), force_camera_space=True,
        texture_source=None, led_power=None, roi_xyxy=c.get('roi_xyxy'),
        calibration_id=c.get('calibration_id'),
        texture_lower_percentile=t.get('lower_percentile', 1),
        texture_upper_percentile=t.get('upper_percentile', 99.5))
    try:
        camera.open()
        if camera.device.info().get('file_camera', False):
            raise ValueError('CONTINUOUS_REQUIRES_PHYSICAL_CAMERA')
        actual_settings = configure_camera(camera, policy)
        actual_settings['writes_on_open'] = dict(getattr(camera, 'applied_capture_settings', {}) or {})
        actual, errors = actual_settings['values'], actual_settings['read_messages']
        texture, led = actual.get(CAMERA_KEYS['texture_source']), actual.get(CAMERA_KEYS['led_power'])
        if (not isinstance(texture, str) or not texture or led is None
                or errors.get(CAMERA_KEYS['texture_source']) or errors.get(CAMERA_KEYS['led_power'])):
            raise ValueError('TEXTURE_SOURCE_LED_READBACK_REQUIRED_FOR_BACKGROUND_IDENTITY')
        camera.texture_source = texture
        camera.led_power = integer(led, 0, 4095, 'ACTUAL_LED_POWER')
    except BaseException:
        camera.close()
        raise
    return camera, actual_settings


def capture_verified_frame(camera, actual_settings):
    coordinates = PhoXiSource.verify_coordinates(camera)
    keys = [CAMERA_KEYS['texture_source'], CAMERA_KEYS['led_power']]
    values, errors = camera.device.get_settings(keys)
    values, errors = values or {}, errors or {}
    if (any(errors.get(k) or values.get(k) is None for k in keys)
            or not isinstance(values.get(keys[0]), str)):
        raise ValueError('FRAME_TEXTURE_LED_READBACK_NOT_VERIFIED')
    camera.texture_source = values[keys[0]]
    camera.led_power = integer(values[keys[1]], 0, 4095, 'ACTUAL_LED_POWER')
    frame = camera.capture()
    frame.info.setdefault('acquisition', {}).update(texture_source=camera.texture_source,
                                                 led_power=camera.led_power)
    frame.info['continuous_coordinate_verification'] = coordinates
    frame.info['continuous_capture_settings'] = actual_settings
    return frame


class LiveVision:
    """Constructing the object does not import/connect PhoXi or load YOLO."""
    def __init__(self, cfg, policy, output, emit=lambda *args: None):
        self.cfg, self.policy = deepcopy(cfg), deepcopy(policy)
        self.output, self.emit = Path(output), emit
        self.camera = None
        self.pipeline = None
        self.actual_settings = None
        self.index = 0

    def open(self):
        if self.camera is not None:
            return
        cfg = self.cfg
        weights = project_path(cfg['yolo']['weights'], cfg)
        if not weights.is_file():
            raise ValueError('MODEL_NOT_FOUND:'+str(weights))
        self.emit('STATUS', 'Đang nạp và làm nóng YOLO; robot chờ tại P_WAIT')
        post = build_postprocessor(cfg)
        if post.roi_enabled and post.roi is None:
            raise ValueError('ROI_MISSING: dùng run_live để thiết lập ROI trước')
        if post.surface_reference is None or post.surface_reference.plane is None:
            raise ValueError('FLOOR_REFERENCE_MISSING: cần nền và mặt phẳng đã thiết lập')
        segmenter = YOLOInstanceSegmenter(weights, cfg['yolo'])
        roi = cfg['camera'].get('roi_xyxy')
        h, w = (roi[3]-roi[1], roi[2]-roi[0]) if roi else (800, 800)
        segmenter.predict(np.zeros((h, w, 3), np.uint8))
        self.pipeline = PhoXiYoloPipeline(segmenter, cfg['measurement'], post,
            cfg.get('postprocess', {}).get('duplicates', {}), cfg['picking'], cfg['texture'])
        self.camera, self.actual_settings = open_configured_camera(cfg, self.policy)
        self.emit('CAMERA', self.actual_settings)

    def capture(self):
        self.open()
        frame = capture_verified_frame(self.camera, self.actual_settings)
        results, rejected = self.pipeline.process_all(frame)
        picking = deepcopy(self.pipeline.last_picking)
        post = self.pipeline.postprocessor
        reference = reference_diagnostics(frame, getattr(post, 'surface_reference', None), self.cfg)
        picking.setdefault('reference', {})['identity_differences'] = reference['differences']
        picking['reference']['identity_details'] = reference
        self.emit('REFERENCE', reference)
        annotated = draw_picking(frame, results, rejected, picking, roi=post.roi,
                                 live=True, max_age_ms=self.cfg['picking']['max_frame_age_ms'],
                                 debug=reference['status'] != 'MATCH')
        dashboard = make_dashboard(annotated, depth_preview(frame.depth_mm, frame.point_cloud_mm),
                                   max_width=1400, max_height=850,
                                   footer_lines=['CONTINUOUS | captured only at P_WAIT | robot_ready=False'])
        self.index += 1
        folder = self.output/('scan_%05d'%self.index)
        folder.mkdir(parents=True, exist_ok=True)
        record = dict(frame_id=frame.frame_id, timestamp_utc=frame.timestamp_utc,
                      captured_at_utc=frame.info.get('capture_started_utc'),
                      picking=picking, objects=[r.to_dict() for r in results],
                      rejected=[r.to_dict() for r in rejected], camera_settings=self.actual_settings,
                      acquisition_identity=reference['frame_identity'], reference_identity=reference,
                      robot_ready=False)
        (folder/'result.json').write_text(json.dumps(_json_safe(record), ensure_ascii=False, indent=2), encoding='utf-8')
        save_bgr_png(folder/'dashboard.png', dashboard)
        if self.policy.get('save_scan_npz', False):
            save_capture(frame, folder, stem='frame', save_3d=True)
        self.emit('IMAGE', dashboard)
        self.emit('SCAN', dict(frame_id=frame.frame_id, picking=picking, path=str(folder)))
        return frame, picking

    def close(self):
        if self.camera is not None:
            try:
                self.camera.close()
            finally:
                self.camera = None
                self.pipeline = None
