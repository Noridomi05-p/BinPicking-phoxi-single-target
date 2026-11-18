from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from ..config import load_config, project_path
from ..robot_target import tcp_preview

PROTOCOL = 'PHOXI_CONT_V1'
STEPS = ('OPEN_AT_WAIT', 'ABOVE_TARGET', 'PREGRASP', 'GRASP', 'CLOSE_GRIPPER',
         'RETREAT_TO_PREGRASP', 'LIFT_TO_TRANSFER_Z', 'WAIT_WITH_OBJECT',
         'DROP_UP_WITH_OBJECT', 'DROP', 'RELEASE_AT_DROP', 'DROP_UP_EMPTY', 'WAIT_EMPTY')


def finite(value, count):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError('INVALID_VECTOR_LENGTH')
    result = [float(v) for v in value]
    if not all(math.isfinite(v) for v in result):
        raise ValueError('NONFINITE_VALUE')
    return result


def number(value, low, high, name):
    v = float(value)
    if not math.isfinite(v) or not low <= v <= high:
        raise ValueError('INVALID_'+name)
    return v


def integer(value, low, high, name):
    v = number(value, low, high, name)
    if v != int(v):
        raise ValueError('INVALID_'+name)
    return int(v)


def safe_text(value):
    if not isinstance(value, str) or not value.strip() or not value.isascii() or any(c in value for c in '|\r\n'):
        raise ValueError('INVALID_PROTOCOL_TEXT')
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_configuration(vision_path, continuous_path):
    cfg = load_config(vision_path)
    policy = yaml.safe_load(Path(continuous_path).read_text(encoding='utf-8'))['continuous']
    if policy['camera_settings_source'] not in ('phoxicontrol', 'config'):
        raise ValueError('INVALID_CAMERA_SETTINGS_SOURCE')
    for k in ('max_cycles', 'no_target_retries'):
        policy[k] = integer(policy[k], 0, 1000000, k.upper())
    policy['stage_timeout_seconds'] = number(policy['stage_timeout_seconds'], 1, 3600, 'STAGE_TIMEOUT')
    policy['settle_seconds'] = number(policy['settle_seconds'], 0, 30, 'SETTLE_SECONDS')
    policy['port'] = integer(policy['port'], 1024, 65535, 'PORT')
    points_path = project_path(policy['points_path'], cfg)
    points = json.loads(points_path.read_text(encoding='utf-8'))
    contract = build_contract(cfg, policy, points)
    # Any change during a run requires an explicit stop/start; never silently reload geometry.
    watched = [Path(vision_path).resolve(), Path(continuous_path).resolve(), points_path,
               project_path(cfg['picking']['robot_target']['calibration_path'], cfg)]
    default = project_path('config/project.yaml', cfg)
    if default.exists():
        watched.append(default)
    for section, key in [('roi', 'path'), ('surface', 'path'), ('surface', 'floor_roi_path')]:
        value = cfg.get('postprocess', {}).get(section, {}).get(key)
        if value:
            path = project_path(value, cfg)
            if path.exists():
                watched.append(path)
    watched = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    return cfg, policy, contract, watched


def check_files(watched):
    for name, expected in watched.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('CONFIG_OR_REFERENCE_CHANGED_RESTART:'+name)


def build_contract(cfg, policy, taught):
    if not cfg['picking'].get('enabled'):
        raise ValueError('PICKING_DISABLED')
    rt = cfg['picking'].get('robot_target', {})
    if not rt.get('enabled'):
        raise ValueError('ROBOT_TARGET_DISABLED_USE_PICKING_ROBOT_CONFIG')
    if taught.get('frame') != 'BASE' or taught.get('euler') != 'intrinsic_ZYZ':
        raise ValueError('TAUGHT_POINTS_WRONG_FRAME')
    tcp_name = safe_text(rt['tcp_name'])
    if taught.get('tcp_name') != tcp_name:
        raise ValueError('TCP_NAME_CHANGED_RETEACH_FIXED_POINTS')
    tcp = finite(rt['tcp_flange_xyzabc'], 6)
    if not np.allclose(tcp, finite(taught['tcp_flange_xyzabc'], 6), atol=1e-6, rtol=0):
        raise ValueError('TCP_OFFSET_CHANGED_RETEACH_FIXED_POINTS')
    if rt['fixed_opening_mm'] != 75 or rt['open_position_request'] != 30:
        raise ValueError('CURRENT_COMMISSIONED_OPENING_IS_75_MM_PR_30')
    motion, gripper = deepcopy(policy['motion']), deepcopy(policy['gripper'])
    motion['transfer_tcp_z_mm'] = number(motion['transfer_tcp_z_mm'], 170, 1000, 'TRANSFER_HEIGHT')
    motion['max_descent_mm'] = number(motion['max_descent_mm'], 1, 100, 'MAX_DESCENT')
    for key in ('travel_velocity', 'travel_acceleration', 'approach_velocity', 'approach_acceleration'):
        values = finite(motion[key], 2)
        # Configuration limits, not a guarantee that a trajectory is collision-free.
        if not 0 < values[0] <= 1000 or not 0 < values[1] <= 180:
            raise ValueError('INVALID_MOTION_'+key)
        motion[key] = values
    if gripper['open_pr'] != rt['open_position_request']:
        raise ValueError('GRIPPER_OPENING_CONFIG_MISMATCH')
    for k in ('open_pr', 'close_pr'):
        gripper[k] = integer(gripper[k], 0, 255, k.upper())
    if gripper['close_pr'] <= gripper['open_pr']:
        raise ValueError('CLOSE_PR_MUST_EXCEED_OPEN_PR')
    gripper['speed_force_word'] = integer(gripper['speed_force_word'], 0, 65535, 'SPEED_FORCE_WORD')
    for k in ('status_signal', 'fault_echo_signal', 'position_current_signal'):
        safe_text(gripper[k])
    gripper['poll_seconds'] = number(gripper['poll_seconds'], .02, .5, 'POLL_SECONDS')
    gripper['timeout_seconds'] = number(gripper['timeout_seconds'], 1, 30, 'GRIPPER_TIMEOUT')
    gripper['position_tolerance_raw'] = integer(gripper['position_tolerance_raw'], 0, 10, 'POSITION_TOLERANCE')
    points = {}
    for name in ('P_WAIT', 'P_DROP_UP', 'P_DROP'):
        p = taught['points'][name]
        if p['solution_space'] != 2:
            raise ValueError('REQUIRES_TAUGHT_SOLUTION_SPACE_2')
        points[name] = dict(pose=finite(p['xyzabc'], 6), joints=finite(p['joints'], 6))
    if min(points['P_WAIT']['pose'][2], points['P_DROP_UP']['pose'][2]) < motion['transfer_tcp_z_mm']:
        raise ValueError('TRANSFER_HEIGHT_ABOVE_FIXED_ROUTE')
    if points['P_DROP']['pose'][2] >= points['P_DROP_UP']['pose'][2]:
        raise ValueError('DROP_MUST_BE_BELOW_DROP_UP')
    calibration_path = project_path(rt['calibration_path'], cfg)
    cal_raw = calibration_path.read_bytes()
    cal = json.loads(cal_raw)
    if cal.get('demo', True) or cal.get('status') != 'VERIFIED_WITHIN_CONFIGURED_LIMITS':
        raise ValueError('CALIBRATION_UNVERIFIED')
    contact = finite(rt['contact_center_flange_mm'], 3)
    max_age = number(cfg['picking']['max_frame_age_ms'], 500, 30000, 'FRAME_MAX_AGE')/1000
    max_tilt = number(cfg['picking'].get('max_tilt_deg', 25), 0, 45, 'MAX_TILT')
    return dict(protocol=PROTOCOL, controller_revision=2, tcp_name=tcp_name, tcp=tcp, points=points,
                contact_center_flange_mm=contact, robot_target=deepcopy(rt), motion=motion,
                gripper=gripper, calibration_sha256=hashlib.sha256(cal_raw).hexdigest(),
                max_frame_age_seconds=max_age, max_tilt_deg=max_tilt)


def timestamp_age(stamp, now=None):
    t = datetime.fromisoformat(str(stamp).replace('Z', '+00:00'))
    if t.tzinfo is None:
        raise ValueError('CAPTURE_TIMESTAMP_REQUIRES_TIMEZONE')
    age = ((now or datetime.now(timezone.utc))-t).total_seconds()
    if not math.isfinite(age) or age < -.05:
        raise ValueError('CAPTURE_TIME_IN_FUTURE')
    return max(0, age)


def make_job(frame, picking, cfg, contract, ready_time):
    if picking.get('target_instance_id') is None or picking.get('pose') is None:
        return None
    if picking.get('tcp_preview_error'):
        raise ValueError('INVALID_TCP_PREVIEW:'+str(picking['tcp_preview_error']))
    fresh = picking.get('freshness', {})
    stamp = fresh.get('captured_at_utc')
    if fresh.get('status') != 'FRESH' or fresh.get('offline'):
        raise ValueError('REQUIRES_FRESH_LIVE_CAPTURE')
    captured = datetime.fromisoformat(str(stamp).replace('Z', '+00:00'))
    if captured < ready_time:
        raise ValueError('CAPTURE_PRECEDES_ROBOT_READY')
    if timestamp_age(stamp) > contract['max_frame_age_seconds']:
        raise ValueError('CAPTURE_EXPIRED')
    if stamp != frame.info.get('capture_started_utc'):
        raise ValueError('FRAME_CAPTURE_TIMESTAMP_MISMATCH')
    preview = picking.get('tcp_preview')
    expected = tcp_preview(frame, picking['pose'], cfg['picking']['robot_target'])
    if not isinstance(preview, dict) or preview.get('source_frame_id') != frame.frame_id or frame.frame_id is None:
        raise ValueError('FRAME_ID_MISMATCH')
    if (preview.get('status') != 'TCP_PREVIEW_UNVALIDATED' or preview.get('robot_ready') is not False
            or preview.get('frame') != 'BASE' or preview.get('euler') != 'intrinsic_ZYZ'
            or preview.get('distance_unit') != 'millimetre'):
        raise ValueError('WRONG_PREVIEW_TYPE')
    for key in ('tcp_name', 'calibration_sha256'):
        if preview.get(key) != expected[key]:
            raise ValueError('PREVIEW_CONFIG_MISMATCH:'+key)
    if expected['calibration_sha256'] != contract['calibration_sha256']:
        raise ValueError('CALIBRATION_CHANGED')
    for key in ('tcp_xyzabc', 'pregrasp_xyzabc', 'tcp_flange_xyzabc', 'tcp_to_contact_tcp_mm'):
        if not np.allclose(finite(preview[key], len(expected[key])), expected[key], atol=1e-6, rtol=0):
            raise ValueError('PREVIEW_GEOMETRY_MISMATCH:'+key)
    if picking['pose'].get('clearance', {}).get('status') == 'BLOCKED':
        raise ValueError('APPROACH_BLOCKED')
    pre, grasp = expected['pregrasp_xyzabc'], expected['tcp_xyzabc']
    lift = pre.copy()
    lift[2] = contract['motion']['transfer_tcp_z_mm']
    validate_poses(pre, grasp, lift, contract)
    return dict(pregrasp=pre, grasp=grasp, lift=lift, captured_at_utc=stamp,
                frame_id=int(frame.frame_id), target_instance_id=picking['target_instance_id'],
                calibration_sha256=expected['calibration_sha256'],
                clearance=picking['pose'].get('clearance', {}), robot_ready=False)


def validate_poses(pre, grasp, lift, contract):
    pre, grasp, lift = finite(pre, 6), finite(grasp, 6), finite(lift, 6)
    if any(abs(v) > 2000 for pose in (pre, grasp, lift) for v in pose[:3]):
        raise ValueError('POSITION_OUTSIDE_SANITY_LIMIT')
    if not grasp[2] < pre[2] < lift[2]:
        raise ValueError('INVALID_HEIGHT_ORDER')
    if abs(lift[2]-contract['motion']['transfer_tcp_z_mm']) > .001:
        raise ValueError('INVALID_TRANSFER_HEIGHT')
    if max(abs(lift[i]-pre[i]) for i in (0, 1, 3, 4, 5)) > .001:
        raise ValueError('LIFT_MUST_KEEP_PREGRASP_XY_ABC')
    rp = Rotation.from_euler('ZYZ', pre[3:], degrees=True)
    rg = Rotation.from_euler('ZYZ', grasp[3:], degrees=True)
    if (rp.inv()*rg).magnitude() > 1e-5:
        raise ValueError('GRASP_PREGRASP_ORIENTATION_MISMATCH')
    axis = rp.as_matrix()[:, 2]
    delta = np.array(grasp[:3])-pre[:3]
    length = float(np.linalg.norm(delta))
    if not 1 <= length <= contract['motion']['max_descent_mm']:
        raise ValueError('INVALID_DESCENT_DISTANCE')
    if not np.allclose(delta, length*axis, atol=.002, rtol=0):
        raise ValueError('DESCENT_NOT_ALONG_TCP_Z')
    if -axis[2] < math.cos(math.radians(contract['max_tilt_deg'])):
        raise ValueError('APPROACH_TILT_OUTSIDE_LIMIT')
