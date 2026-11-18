"""Camera proposal -> BASE TCP preview. No robot motion or gripper I/O."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .calibration.math3d import check_transform, transform, as_pose_zyz, pose_zyz


def scalar(settings, key, lower, upper):
    value = float(settings[key])
    if not np.isfinite(value) or not lower <= value <= upper:
        raise ValueError('INVALID_' + key.upper())
    return value


def tcp_preview(frame, pose, settings, calibration=None):
    """Offsets are TCP -> contact, expressed in TCP axes; subtract, not add."""
    if calibration is None:
        path = Path(settings['calibration_path'])
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        raw = path.read_bytes()
        calibration = json.loads(raw)
    else:
        raw = json.dumps(calibration, sort_keys=True).encode()
    c = calibration
    if (c.get('demo', True) or c.get('status') != 'VERIFIED_WITHIN_CONFIGURED_LIMITS'
            or c.get('distance_unit') != 'millimetre'
            or c.get('coordinate_space') != 'CameraSpace/PrimaryCamera'
            or c.get('robot_frame') != 'BASE'):
        raise ValueError('CALIBRATION_UNVERIFIED_OR_WRONG_FRAME')
    identity = c.get('camera_identity', {})
    actual = frame.info.get('acquisition', {})
    for key in ('device_id', 'coordinate_space', 'calibration_id', 'source_shape_hw'):
        if identity.get(key) is None or actual.get(key) != identity[key]:
            raise ValueError('CALIBRATION_IDENTITY_MISMATCH:' + key)
    # Image cropping changes pixel indices, not metric CameraSpace coordinates.
    tbc = check_transform(c['T_base_camera'])
    opening = scalar(settings, 'fixed_opening_mm', 1, 85)
    required = float(pose.get('required_opening_mm', pose['proposed_opening_mm']))
    if not np.isfinite(required) or required <= 0:
        raise ValueError('INVALID_REQUIRED_OPENING')
    if required > opening + 1e-6:
        raise ValueError('FIXED_OPENING_INSUFFICIENT')
    depth = scalar(settings, 'grasp_depth_mm', 0, 100)
    approach = scalar(settings, 'pregrasp_distance_mm', 1, 500)
    tcp_name = str(settings['tcp_name'])
    if not tcp_name.strip() or any(ch in tcp_name for ch in '|\r\n'):
        raise ValueError('INVALID_TCP_NAME')
    # T_proposal_tcp describes the required TCP axes relative to proposal axes.
    align = check_transform(transform(settings['rotation_proposal_tcp']))[:3, :3]
    tcp_flange = pose_zyz(settings['tcp_flange_xyzabc'])
    contact_flange = np.asarray(settings['contact_center_flange_mm'], dtype=float)
    if contact_flange.shape != (3,) or not np.isfinite(contact_flange).all():
        raise ValueError('INVALID_CONTACT_CENTER')
    offset = tcp_flange[:3, :3].T @ (contact_flange - tcp_flange[:3, 3])
    ts = transform(pose['rotation_camera'], pose['surface_anchor_xyz_mm'])
    surface_base = tbc @ ts
    axis = surface_base[:3, 2]
    contact = surface_base[:3, 3] + depth * axis
    rotation = surface_base[:3, :3] @ align
    tcp = transform(rotation, contact - rotation @ offset)
    pre = tcp.copy()
    pre[:3, 3] -= approach * axis
    return dict(status='TCP_PREVIEW_UNVALIDATED', robot_ready=False, frame='BASE',
        source_frame_id=frame.frame_id if hasattr(frame, 'frame_id') else None,
        distance_unit='millimetre', angle_unit='degree', euler='intrinsic_ZYZ',
        tcp_name=tcp_name, tcp_flange_xyzabc=settings['tcp_flange_xyzabc'],
        tcp_xyzabc=as_pose_zyz(tcp), pregrasp_xyzabc=as_pose_zyz(pre),
        rotation_base_tcp=rotation.tolist(), approach_base=axis.tolist(),
        surface_anchor_base_mm=surface_base[:3, 3].tolist(), contact_center_base_mm=contact.tolist(),
        tcp_to_contact_tcp_mm=offset.tolist(), grasp_depth_mm=depth,
        actual_opening_mm=opening, open_position_request=int(settings['open_position_request']),
        contact_reference_opening_mm=settings['contact_reference_opening_mm'],
        calibration_sha256=hashlib.sha256(raw).hexdigest(),
        limitations=['CONTACT_CENTER_APPROXIMATE_AT_REFERENCE_OPENING',
                     'CLOSING_KINEMATICS_UNVERIFIED', 'ARM_AND_RETREAT_NOT_VERIFIED'])


def attach_preview(frame, output, settings):
    if not settings.get('enabled', False):
        return output
    output['tcp_preview'] = None
    if output.get('pose') is not None:
        try:
            output['tcp_preview'] = tcp_preview(frame, output['pose'], settings)
            output['tcp_preview']['source_freshness'] = dict(output.get('freshness', {}))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            output['tcp_preview_error'] = str(exc)
    return output
