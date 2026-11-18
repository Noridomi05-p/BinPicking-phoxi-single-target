"""Capture empty-bin data with verified camera readback; never connects to a robot."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading

import numpy as np
import yaml

from ..config import project_path
from ..geometry import identity_status
from ..roi import PolygonROI
from ..storage import _json_safe, utc_stem
from ..surface_reference import SurfaceReference
from ..visualization import make_dashboard, depth_preview
from .vision import CAMERA_KEYS, open_configured_camera, capture_verified_frame, reference_diagnostics, override_settings


def save_camera_profile(path, source, values):
    """Persist only camera policy fields; preserve all robot/motion/gripper values."""
    path = Path(path)
    if source not in ('config', 'phoxicontrol'):
        raise ValueError('INVALID_CAMERA_SETTINGS_SOURCE')
    checked = override_settings(values)
    raw = path.read_bytes()
    document = yaml.safe_load(raw)
    document['continuous']['camera_settings_source'] = source
    document['continuous']['camera_overrides'] = {name:checked[key] for name,key in CAMERA_KEYS.items()}
    backup = path.with_name(path.stem+'_'+utc_stem('camera_backup')+path.suffix)
    with backup.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',prefix='.camera_profile_',suffix='.yaml',
                                         dir=path.parent,delete=False) as f:
            temporary = Path(f.name)
            yaml.safe_dump(document,f,allow_unicode=True,sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary,path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return backup


class BackgroundCancelled(Exception):
    pass


class BackgroundService:
    """The operator must empty the bin and keep the robot at P_WAIT before pressing Capture."""
    def __init__(self, cfg, policy, emit=lambda *args: None):
        self.cfg, self.policy, self.emit = deepcopy(cfg), deepcopy(policy), emit
        self.stop_event = threading.Event()

    def request_stop(self):
        self.stop_event.set()

    def run(self):
        camera, temporary = None, None
        destination = project_path(self.cfg['postprocess']['surface']['path'], self.cfg)
        try:
            settings = self.cfg['postprocess']['surface']
            floor_path = project_path(settings.get('floor_roi_path', 'config/bin_floor_roi.json'), self.cfg)
            if not floor_path.is_file():
                raise ValueError('FLOOR_ROI_MISSING: dùng F trong run_live để xác định vùng đáy trước')
            floor_roi = PolygonROI.load(floor_path)
            roi_path = project_path(self.cfg['postprocess']['roi']['path'], self.cfg)
            bin_roi = PolygonROI.load(roi_path)
            camera, actual = open_configured_camera(self.cfg, self.policy)
            self.emit('CAMERA', actual)
            expected_profile = {key:value for key,value in actual.get('values', {}).items()
                                if key in CAMERA_KEYS.values() and value is not None}
            count = max(1, int(settings.get('capture_frames', 15)))
            frames, seen = [], set()
            for index in range(count):
                if self.stop_event.is_set():
                    raise BackgroundCancelled()
                frame = capture_verified_frame(camera, actual)
                if expected_profile:
                    current, errors = camera.device.get_settings(list(expected_profile))
                    current, errors = current or {}, errors or {}
                    if any(errors.get(k) or current.get(k) != v for k,v in expected_profile.items()):
                        raise ValueError('BACKGROUND_CAMERA_PROFILE_CHANGED_DURING_CAPTURE')
                if frame.frame_id is None or frame.frame_id in seen:
                    raise ValueError('BACKGROUND_REUSED_OR_MISSING_FRAME_ID')
                seen.add(frame.frame_id)
                frames.append(frame)
                self.emit('STATUS', 'Đang chụp nền thùng trống %s/%s — không di chuyển camera, thùng hoặc robot'%(index+1,count))
            if self.stop_event.is_set():
                raise BackgroundCancelled()
            shape = frames[0].texture_bgr.shape[:2]
            measurement = self.cfg['measurement']
            reference = SurfaceReference.from_frames(frames,
                min_z_mm=float(measurement.get('min_z_mm', 0)),
                max_z_mm=float(measurement.get('max_z_mm', float('inf'))),
                min_valid_fraction=float(settings.get('reference_min_valid_fraction', .5)),
                plane_roi=floor_roi.mask(shape), plane_settings=settings)
            if reference.plane is None:
                raise ValueError('BACKGROUND_FLOOR_PLANE_NOT_VERIFIED:'+str(reference.plane_error))
            region = bin_roi.mask(shape)
            valid = np.isfinite(reference.depth_mm) & (reference.depth_mm > 0)
            coverage = np.count_nonzero(valid & region)/max(1, np.count_nonzero(region))
            if coverage < .5 or identity_status(reference.identity, frames[-1]) != 'MATCH':
                raise ValueError('BACKGROUND_COVERAGE_OR_IDENTITY_NOT_VERIFIED')
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix='.background_', suffix='.npz', dir=destination.parent, delete=False) as f:
                temporary = Path(f.name)
            reference.save(temporary)
            # Keep verified illumination outside identity so the existing core's identity
            # schema is unchanged and older SurfaceReference readers remain compatible.
            with np.load(temporary, allow_pickle=False) as saved:
                arrays = {key:np.array(saved[key],copy=True) for key in saved.files}
            metadata = json.loads(str(arrays['metadata_json'].item()))
            metadata['verified_capture_settings'] = _json_safe(actual)
            arrays['metadata_json'] = np.array(json.dumps(metadata, ensure_ascii=False))
            np.savez_compressed(temporary, **arrays)
            loaded = SurfaceReference.load(temporary)
            if loaded.identity != reference.identity or loaded.plane is None:
                raise ValueError('BACKGROUND_TEMP_FILE_NOT_VERIFIED')
            if self.stop_event.is_set():
                raise BackgroundCancelled()
            backup = None
            if destination.exists():
                backup = destination.with_name(destination.stem+'_'+utc_stem('backup')+'.npz')
                with destination.open('rb') as src, backup.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                    dst.flush()
                    os.fsync(dst.fileno())
            if self.stop_event.is_set():
                raise BackgroundCancelled()
            with temporary.open('rb') as f:
                os.fsync(f.fileno())
            os.replace(temporary, destination)
            temporary = None
            details = reference_diagnostics(frames[-1], reference, self.cfg)
            self.emit('REFERENCE', details)
            self.emit('IMAGE', make_dashboard(bin_roi.draw(frames[-1].texture_bgr),
                depth_preview(frames[-1].depth_mm, frames[-1].point_cloud_mm), max_width=1400, max_height=850))
            self.emit('BACKGROUND_SAVED', dict(path=str(destination), backup=str(backup) if backup else None,
                frames=count, coverage=coverage, identity=reference.identity, camera_settings=actual))
        except BackgroundCancelled:
            self.emit('STATUS', 'Đã hủy chụp nền; chưa thay tệp nền đang dùng')
        except BaseException as exc:
            self.emit('FAULT', dict(detail=str(exc), phase='BACKGROUND_CAPTURE', completed=0,
                                   output=str(destination.parent), motion_outcome_unconfirmed=False))
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as exc:
                    self.emit('FILE_CLEANUP_ERROR', str(exc))
            try:
                if camera is not None:
                    camera.close()
            except Exception as exc:
                self.emit('CAMERA_CLOSE_ERROR', str(exc))
            self.emit('RELEASED', 'Đã nhả camera sau tác vụ nền; chưa kết nối hay gửi lệnh robot')
