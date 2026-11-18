#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from phoxi_yolo3d.acquisition import (
    CAPTURE_MODES, CaptureSchedule, SoftwareTriggerServer, TriggerBroker,
)
from phoxi_yolo3d.camera import PhoXiCamera
from phoxi_yolo3d.config import load_config, project_path
from phoxi_yolo3d.detector import YOLOInstanceSegmenter
from phoxi_yolo3d.diagnostics import frame_diagnostics, result_counts
from phoxi_yolo3d.pipeline import PhoXiYoloPipeline
from phoxi_yolo3d.postprocess import build_postprocessor
from phoxi_yolo3d.roi import PolygonROI, edit_polygon_roi
from phoxi_yolo3d.storage import save_bgr_png, save_capture, utc_stem
from phoxi_yolo3d.surface_reference import capture_surface_reference
from phoxi_yolo3d.visualization import depth_preview, draw_results, make_dashboard
from phoxi_yolo3d.picking_visualization import draw_picking


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def save_inspection(output, frame, record, results, rejected, dashboard, config, postprocessor):
    """Save evidence without triggering another frame or filling missing depth."""
    root = output / utc_stem('inspection')
    paths = save_capture(frame, root, stem='frame', save_3d=True)
    save_bgr_png(root / 'dashboard.png', dashboard)
    (root / 'result.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    np.savez_compressed(
        root / 'masks.npz',
        **{f'instance_{item.detection.instance_id}': item.detection.mask for item in results + rejected},
    )
    replay = {key: value for key, value in config.items() if not key.startswith('_')}
    replay = json.loads(json.dumps(replay))
    replay.setdefault('postprocess', {})['roi'] = dict(postprocessor.roi_settings)
    replay['postprocess']['surface'] = dict(postprocessor.surface_settings)
    if postprocessor.roi is not None:
        roi_path = root / 'reference' / 'bin_roi.json'
        postprocessor.roi.save(roi_path)
        replay['postprocess']['roi']['path'] = '@config/reference/bin_roi.json'
    else:
        replay['postprocess']['roi']['path'] = '@config/reference/missing_roi.json'
    if postprocessor.surface_reference is not None:
        surface_path = root / 'reference' / 'bin_surface.npz'
        postprocessor.surface_reference.save(surface_path)
        replay['postprocess']['surface']['path'] = '@config/reference/bin_surface.npz'
    else:
        replay['postprocess']['surface']['path'] = '@config/reference/missing_surface.npz'
    floor_source = project_path(postprocessor.surface_settings.get('floor_roi_path', 'config/bin_floor_roi.json'), config)
    if floor_source.exists():
        PolygonROI.load(floor_source).save(root / 'reference' / 'bin_floor_roi.json')
    replay['postprocess']['surface']['floor_roi_path'] = '@config/reference/bin_floor_roi.json'
    (root / 'replay.yaml').write_text(yaml.safe_dump(replay, sort_keys=False), encoding='utf-8')
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob('*') if p.is_file()}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Da luu anh + 3D + masks + ket qua + nen tham chieu: {root}')
    print(f"Xem lai: python scripts/08_run_offline.py {paths['scan']} --config {root / 'replay.yaml'}")
    print(f"Chay dung masks da luu (khong can YOLO): python scripts/15_replay_picking.py {root}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description='PhoXi: chup tung anh, lien tuc, chu ky, hoac tin hieu')
    parser.add_argument('--config', default=None)
    parser.add_argument('--device-id', default=None)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--count', type=int, default=0, help='0 = khong gioi han; >0 = thoat sau N anh')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--save-images', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--mode', choices=CAPTURE_MODES, default=None)
    parser.add_argument('--interval', type=float, default=None, help='So giay nghi SAU moi lan xu ly')
    parser.add_argument('--trigger-host', default=None)
    parser.add_argument('--trigger-port', type=int, default=None)
    parser.add_argument('--conf', type=float, default=None)
    parser.add_argument('--imgsz', type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    yolo_cfg, camera_cfg = config['yolo'], config['camera']
    texture_cfg, measurement_cfg = config['texture'], config['measurement']
    runtime_cfg = config['runtime']
    if args.conf is not None:
        if not 0 <= args.conf <= 1:
            parser.error('--conf phai nam trong [0, 1]')
        yolo_cfg['confidence'] = args.conf
    if args.imgsz is not None:
        if args.imgsz <= 0:
            parser.error('--imgsz phai > 0')
        yolo_cfg['image_size'] = args.imgsz
    mode = args.mode or runtime_cfg.get('scan_mode', 'manual')
    interval = args.interval if args.interval is not None else runtime_cfg.get('interval_seconds', 2.0)
    if interval <= 0 or args.count < 0:
        parser.error('--interval phai > 0; --count phai >= 0')
    if args.headless and mode == 'manual' and args.count != 1:
        parser.error('Headless manual dung --count 1; cho tin hieu dung --mode trigger')
    schedule = CaptureSchedule(mode, interval)
    trigger_host = args.trigger_host or runtime_cfg.get('trigger_host', '127.0.0.1')
    trigger_port = args.trigger_port if args.trigger_port is not None else int(runtime_cfg.get('trigger_port', 8765))
    weights = Path(args.weights).expanduser().resolve() if args.weights else project_path(yolo_cfg['weights'], config)
    if not weights.exists():
        raise SystemExit(f'Khong tim thay model {weights}. Hay train/copy best.pt truoc.')
    yolo_cfg['weights'] = str(weights)
    postprocessor = build_postprocessor(config)
    pipeline = PhoXiYoloPipeline(
        YOLOInstanceSegmenter(weights, yolo_cfg), measurement_cfg,
        postprocessor=postprocessor,
        duplicate_settings=config.get('postprocess', {}).get('duplicates', {}),
        picking_settings=config.get('picking', {}), texture_settings=texture_cfg,
    )
    camera = PhoXiCamera(
        device_id=args.device_id or camera_cfg.get('device_id'),
        address=camera_cfg.get('address'), api_lib_path=camera_cfg.get('api_lib_path'),
        refresh_discovery=bool(camera_cfg.get('refresh_discovery', True)),
        timeout_ms=int(camera_cfg.get('timeout_ms', 30_000)),
        force_camera_space=bool(camera_cfg.get('force_camera_space', True)),
        texture_source=camera_cfg.get('texture_source'), led_power=camera_cfg.get('led_power'),
        texture_lower_percentile=float(texture_cfg.get('lower_percentile', 1.0)),
        texture_upper_percentile=float(texture_cfg.get('upper_percentile', 99.5)),
        roi_xyxy=camera_cfg.get('roi_xyxy'),
        calibration_id=camera_cfg.get('calibration_id'),
    )
    session = datetime.now(timezone.utc).strftime('live_%Y%m%dT%H%M%S_%fZ')
    output = project_path(runtime_cfg['output_dir'], config) / session
    save_images = runtime_cfg.get('save_annotated', False) if args.save_images is None else args.save_images
    fill_masks = bool(runtime_cfg.get('mask_fill', True))
    debug_rejected = bool(runtime_cfg.get('debug_rejected', False))
    cv2 = None
    if not args.headless:
        import cv2 as cv2_module
        cv2 = cv2_module
    broker = TriggerBroker()
    trigger_server = None
    active_request = None

    def start_trigger():
        nonlocal trigger_server
        if trigger_server is None:
            trigger_server = SoftwareTriggerServer(broker, trigger_host, trigger_port).start()
            print(f'Software trigger TCP: {trigger_host}:{trigger_port} | TRIGGER + newline')

    for error in postprocessor.load_errors:
        print(f'Canh bao: {error}')
    for status in postprocessor.status_lines():
        print(status)
    print('1=manual 2=continuous 3=interval 4=trigger | Space/N=chup 1 anh P=tam dung')
    print('R=ROI F=vung day phang B=nen thung trong D=chi tiet M=to mask S=luu anh+3D Q/Esc=thoat')
    frame = None
    results, rejected = [], []
    record = {}
    diagnostics = {}
    frame_index = 0
    prompted_roi = False
    dirty = True
    dashboard = None
    last_age_redraw = 0.0

    def process_frame(reason, elapsed_capture=0.0):
        nonlocal results, rejected, record, diagnostics, dirty
        started = time.perf_counter()
        results, rejected = pipeline.process_all(frame)
        if reason.endswith('_same_frame') and pipeline.last_picking.get('enabled'):
            pipeline.last_picking.update(status='RECAPTURE_REQUIRED', target_instance_id=None, pose=None,
                reasons=['SETTINGS_CHANGED_CAPTURE_NEW_FRAME'])
            if 'tcp_preview' in pipeline.last_picking:
                pipeline.last_picking['tcp_preview'] = None
        diagnostics = frame_diagnostics(frame, measurement_cfg, postprocessor)
        record = {
            'frame_index': max(0, frame_index - 1) if reason.endswith('_same_frame') else frame_index,
            'frame_id': frame.frame_id,
            'timestamp_utc': frame.timestamp_utc,
            'capture_reason': reason,
            'scan_mode': schedule.mode,
            'request_id': active_request.request_id if active_request else None,
            'processing_ms': elapsed_capture + (time.perf_counter() - started) * 1000,
            'pipeline_timings_ms': pipeline.last_timings_ms,
            **result_counts(results, rejected),
            'diagnostics': diagnostics,
            'inference_settings': dict(yolo_cfg),
            'objects': [item.to_dict() for item in results],
            'rejected': [item.to_dict() for item in rejected],
            'picking': pipeline.last_picking,
            'robot_ready': False,
        }
        if runtime_cfg.get('save_jsonl', True):
            append_jsonl(output / 'objects.jsonl', record)
        print(
            f'Frame {frame_index:05d} ID={frame.frame_id} | 2D={len(results)} '
            f"3D-confirmed={record['verified_3d_count']} unknown={record['unverified_count']} "
            f"rejected={len(rejected)} | {record['processing_ms']:.1f} ms"
        )
        for item in results + rejected:
            post = item.postprocess
            height_text = (
                f'Hmax={post.surface_height_max_mm:.1f} mm '
                f'H99={post.surface_height_p99_mm:.1f} mm '
                if post.surface_height_max_mm is not None
                and post.surface_height_p99_mm is not None else ''
            )
            print(
                f'  #{item.detection.instance_id} {item.detection.class_name} '
                f'conf={item.detection.score:.2f} measure={item.measurement.point_count} pts '
                f'surface={post.surface_status} '
                f'raised={post.surface_raised_points}/{post.surface_mask_pixels} '
                f'{height_text}'
                f"reason={post.reason or 'OK'}"
            )
        print(f"  3D valid in ROI: {diagnostics['pointcloud_valid_ratio_roi']:.1%}")
        dirty = True

    try:
        if mode == 'trigger':
            start_trigger()
        with camera:
            print(f'Da ket noi PhoXi: {camera.device_id}; mode={schedule.mode}')
            while True:
                broker.set_enabled(schedule.mode == 'trigger' and not schedule.paused)
                active_request = broker.claim()
                reason = 'tcp_trigger' if active_request else schedule.take_due(time.monotonic())
                captured_now = reason is not None
                if captured_now:
                    if active_request is None:
                        broker.set_enabled(False)
                    started = time.perf_counter()
                    frame = camera.capture()
                    capture_ms = (time.perf_counter() - started) * 1000
                    if (cv2 is not None and schedule.mode != 'trigger'
                            and postprocessor.roi_enabled and postprocessor.roi is None
                            and not prompted_roi
                            and bool(postprocessor.roi_settings.get('prompt_if_missing', True))):
                        prompted_roi = True
                        roi = edit_polygon_roi(frame.texture_bgr)
                        if roi is not None:
                            postprocessor.set_roi(roi, save=True)
                    process_frame(reason, capture_ms)
                    frame_index += 1
                    schedule.completed(time.monotonic())
                    if active_request is not None:
                        broker.finish(active_request, {'status': 'done', **record})
                        active_request = None

                # Redraw the last completed frame: no capture or YOLO on pause,
                # debug toggles, fill toggles or Save.
                if (frame is not None and config.get('picking', {}).get('enabled')
                        and time.monotonic() - last_age_redraw > .5):
                    dirty = True
                    last_age_redraw = time.monotonic()
                if cv2 is not None and dirty:
                    if frame is None:
                        blank = np.zeros((500, 800, 3), dtype=np.uint8)
                        annotated, depth = blank, blank.copy()
                    else:
                        annotated = draw_picking(
                            frame, results, rejected, pipeline.last_picking,
                            alpha=float(runtime_cfg.get('mask_alpha', 0.22)),
                            fill_masks=fill_masks, debug=debug_rejected, roi=postprocessor.roi,
                            live=True, max_age_ms=float(config.get('picking', {}).get('max_frame_age_ms', 5000)),
                        )
                        depth = depth_preview(frame.depth_mm, frame.point_cloud_mm)
                    state = 'PAUSED' if schedule.paused else ('WAIT SIGNAL' if schedule.mode == 'trigger' else 'HOLD' if schedule.mode == 'manual' else 'RUN')
                    footer = [
                        f'MODE={schedule.mode.upper()} {state} | FRAME={None if frame is None else frame.frame_id} | '
                        f'2D={len(results)} 3D-confirmed={sum(x.verified_3d for x in results)} REJECT={len(rejected)}',
                        ' | '.join(postprocessor.status_lines()),
                    ]
                    if diagnostics:
                        bg = diagnostics['background_valid_ratio_roi']
                        footer.append(
                            f"ROI valid 3D={diagnostics['pointcloud_valid_ratio_roi']:.1%} "
                            f"in-range={diagnostics['pointcloud_in_range_ratio_roi']:.1%} "
                            f"background={'MISSING' if bg is None else format(bg, '.1%')}"
                        )
                    footer.extend([
                        '1 Manual | 2 Continuous | 3 Interval | 4 Trigger | Space/N Next | P Pause',
                        'R ROI | F Floor region | B Empty-bin background | D All objects | M Fill | S Save | Q Exit',
                    ])
                    dashboard = make_dashboard(annotated, depth, footer_lines=footer)
                    cv2.imshow('PhoXi YOLO 3D', dashboard)
                    dirty = False
                    if captured_now and save_images:
                        save_bgr_png(output / 'annotated' / f'frame_{frame_index - 1:06d}.png', dashboard)
                elif cv2 is None and captured_now and save_images:
                    save_bgr_png(output / 'annotated' / f'frame_{frame_index - 1:06d}.png', draw_picking(frame, results, rejected, pipeline.last_picking))
                if captured_now and args.count and frame_index >= args.count:
                    break
                if cv2 is None:
                    time.sleep(0.02)
                    continue
                key = cv2.waitKey(20) & 0xFF
                if key in (ord('q'), 27):
                    break
                if ord('1') <= key <= ord('4'):
                    new_mode = CAPTURE_MODES[key - ord('1')]
                    if new_mode == 'trigger':
                        try:
                            start_trigger()
                        except OSError as exc:
                            print(f'Khong mo duoc trigger TCP: {exc}')
                            continue
                    broker.set_enabled(False)
                    schedule.set_mode(new_mode, time.monotonic())
                    dirty = True
                elif key in (32, ord('n')):
                    schedule.request_once()
                elif key == ord('p'):
                    schedule.toggle_pause(time.monotonic())
                    broker.set_enabled(False)
                    dirty = True
                elif key == ord('d'):
                    debug_rejected = not debug_rejected
                    dirty = True
                elif key == ord('m'):
                    fill_masks = not fill_masks
                    dirty = True
                elif key == ord('s') and frame is not None:
                    save_inspection(output, frame, record, results, rejected, dashboard, config, postprocessor)
                elif key == ord('r') and frame is not None:
                    broker.set_enabled(False)
                    roi = edit_polygon_roi(frame.texture_bgr, initial_roi=postprocessor.roi)
                    if roi is not None:
                        postprocessor.set_roi(roi, save=True)
                        process_frame('roi_changed_same_frame')
                elif key == ord('f') and frame is not None:
                    broker.set_enabled(False)
                    print('Chon CHI vung day phang cua thung TRONG, tranh thanh thung. Sau do bam B.')
                    floor_path = project_path(postprocessor.surface_settings.get('floor_roi_path', 'config/bin_floor_roi.json'), config)
                    floor_roi = edit_polygon_roi(frame.texture_bgr)
                    if floor_roi is not None:
                        floor_roi.save(floor_path)
                        print(f'Da luu vung fit mat phang: {floor_path}; bam B de chup lai nen.')
                        # The old plane must not survive a changed calibration region.
                        if postprocessor.surface_reference is not None:
                            postprocessor.surface_reference.plane = None
                            postprocessor.surface_reference.plane_error = 'FLOOR_ROI_CHANGED_RECAPTURE_REQUIRED'
                            postprocessor.set_surface_reference(postprocessor.surface_reference, save=True)
                        process_frame('floor_roi_changed_same_frame')
                elif key == ord('b'):
                    broker.set_enabled(False)
                    print('Chup nen: thung phai HOAN TOAN TRONG, camera va thung dung yen.')
                    surface_cfg = postprocessor.surface_settings
                    floor_path = project_path(surface_cfg.get('floor_roi_path', 'config/bin_floor_roi.json'), config)
                    floor_roi = PolygonROI.load(floor_path) if floor_path.exists() else None
                    reference = capture_surface_reference(
                        camera, frame_count=int(surface_cfg.get('capture_frames', 15)),
                        min_z_mm=float(measurement_cfg.get('min_z_mm', 0.0)),
                        max_z_mm=float(measurement_cfg.get('max_z_mm', float('inf'))),
                        min_valid_fraction=float(surface_cfg.get('reference_min_valid_fraction', 0.50)),
                        progress=lambda done, total: print(f'  Nen {done}/{total}'),
                        floor_roi=floor_roi, plane_settings=surface_cfg,
                    )
                    postprocessor.set_surface_reference(reference, save=True)
                    print('Floor plane:', reference.plane or reference.plane_error or 'MISSING: ve vung day bang F roi bam B')
                    valid_bg = np.isfinite(reference.depth_mm) & (reference.depth_mm > 0)
                    region = np.ones(valid_bg.shape, dtype=bool) if postprocessor.roi is None else postprocessor.roi.mask(valid_bg.shape)
                    coverage = np.count_nonzero(valid_bg & region) / max(1, np.count_nonzero(region))
                    print(f'Da luu nen: {postprocessor.surface_path}; valid trong ROI={coverage:.1%}')
                    if coverage < 0.5:
                        print('Nen bi khuyet nhieu: cac mask o vung nay se CHUA XAC NHAN 3D.')
                    if frame is not None:
                        process_frame('background_changed_same_frame')
                    schedule.completed(time.monotonic())
                    dirty = True
    except KeyboardInterrupt:
        print('Dung boi nguoi dung.')
    except Exception as exc:
        if active_request is not None:
            broker.finish(active_request, {'status': 'error', 'reason': str(exc)})
            active_request = None
        raise
    finally:
        if active_request is not None:
            broker.finish(active_request, {'status': 'cancelled', 'reason': 'application_stopped'})
        if trigger_server is not None:
            trigger_server.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
    print(f'Ket qua: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
