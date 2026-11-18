#!/usr/bin/env python3
"""Export current source/config/log files for a new chat; no camera or robot access."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

EXCLUDED_DIRS = {
    '.git', '.venv', 'venv', 'env', '__pycache__', 'node_modules', 'build', 'dist',
    'data', 'dataset', 'datasets', 'models', 'weights', 'runs', 'results', 'generated',
}
SOURCE_SUFFIXES = {'.py', '.drl', '.yaml', '.yml', '.json', '.md', '.txt', '.toml', '.cfg', '.rst'}
MAX_FILE = 8 * 1024 * 1024
MAX_TOTAL = 60 * 1024 * 1024
JOURNAL_TAIL = 512 * 1024


def merge(base, overlay):
    result = deepcopy(base)
    for key, value in overlay.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def resolve_reference(value, root, config_path):
    value = str(value)
    if value.startswith('@config/'):
        return config_path.parent / value[len('@config/'):]
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def export(root, vision_path, continuous_path, output):
    root = Path(root).resolve()
    vision_path, continuous_path = Path(vision_path).resolve(), Path(continuous_path).resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite existing export: ' + str(output))
    if not (root / 'phoxi_yolo3d').is_dir():
        raise ValueError('Not a phoxi-single-target project root: ' + str(root))
    warnings, sources = [], {}

    def add(path, label=None, tail=False):
        path = Path(path)
        if not path.is_file() or path.is_symlink():
            return
        if label is None:
            try:
                label = path.resolve().relative_to(root).as_posix()
            except ValueError:
                raise ValueError('External file requires an explicit archive label')
        if '..' in Path(label).parts or Path(label).is_absolute():
            raise ValueError('Invalid archive label')
        sources[label] = (path, tail)

    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith('.')
                         and not (Path(directory) / d).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if name.startswith('.') or path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            if any(word in name.lower() for word in ('_backup_', 'credentials', 'secrets')):
                continue
            add(path)
    for path in sorted((root / 'generated').glob('*.drl')):
        add(path)

    yaml_ok = True
    try:
        import yaml
    except ImportError:
        yaml = None
        yaml_ok = False
        warnings.append('PyYAML unavailable: raw YAML included, effective values not computed')

    def read_yaml(path):
        nonlocal yaml_ok
        path = Path(path)
        if not path.is_file():
            yaml_ok = False
            warnings.append('Missing configuration: ' + str(path))
            return {}
        try:
            add(path) if path.is_relative_to(root) else add(path, 'external_references/' + path.name)
            if yaml is None:
                return {}
            value = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
            if not isinstance(value, dict):
                raise ValueError('YAML root is not a mapping')
            return value
        except (OSError, ValueError, yaml.YAMLError if yaml else ValueError) as exc:
            yaml_ok = False
            warnings.append('Cannot read configuration %s: %s' % (path, exc))
            return {}

    cfg = merge(read_yaml(root / 'config/project.yaml'), read_yaml(vision_path))
    policy = read_yaml(continuous_path).get('continuous', {})
    if not isinstance(policy, dict):
        warnings.append('continuous policy is not a mapping')
        policy = {}
        yaml_ok = False
    post = cfg.get('postprocess', {})
    robot = cfg.get('picking', {}).get('robot_target', {})
    references = {
        'calibration': robot.get('calibration_path'), 'points': policy.get('points_path'),
        'roi': post.get('roi', {}).get('path'), 'floor_roi': post.get('surface', {}).get('floor_roi_path'),
    }
    for label, value in references.items():
        if value:
            # Match core project_path(value, cfg): @config is the vision YAML directory.
            path = resolve_reference(value, root, vision_path).resolve()
            if path.is_file() and path.suffix.lower() in ('.json', '.yaml', '.yml'):
                add(path) if path.is_relative_to(root) else add(path, 'external_references/' + label + '_' + path.name)
            else:
                warnings.append('Missing reference %s: %s' % (label, path))

    snapshot = {
        'format': 'PHOXI_CHAT_HANDOVER_V1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'project_root': str(root), 'vision_config': str(vision_path), 'continuous_config': str(continuous_path),
        'configuration_reads_ok': yaml_ok,
        'evidence_scope': 'Files on disk only. No hardware validation or confirmation of code installed on pendant.',
        'python': sys.version, 'python_executable': sys.executable, 'platform': platform.platform(),
        'disk_settings': {
            'max_frame_age_ms': cfg.get('picking', {}).get('max_frame_age_ms'),
            'robot_target': robot, 'continuous': policy,
            'camera_yaml_labels': cfg.get('camera', {}), 'texture': cfg.get('texture', {}),
        },
        'packages': {}, 'git': {}, 'latest_continuous_run': None, 'latest_live_result': None,
    }
    for package in ('numpy', 'scipy', 'opencv-python', 'opencv-python-headless', 'ultralytics', 'PyYAML'):
        try:
            snapshot['packages'][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    for label, args in (
        ('head', ['rev-parse', 'HEAD']), ('status_short', ['status', '--short']),
        ('recent_commits', ['log', '-5', '--oneline']),
    ):
        try:
            result = subprocess.run(['git', '-C', str(root)] + args, capture_output=True, text=True,
                                    timeout=5, env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
            snapshot['git'][label] = result.stdout.strip() if result.returncode == 0 else 'unavailable'
        except (OSError, subprocess.TimeoutExpired):
            snapshot['git'][label] = 'unavailable'

    background = post.get('surface', {}).get('path')
    if background:
        path = resolve_reference(background, root, vision_path).resolve()
        snapshot['background'] = {'path': str(path), 'full_npz_included': False}
        try:
            import numpy as np
            with np.load(path, allow_pickle=False) as data:
                snapshot['background']['metadata'] = json.loads(str(data['metadata_json'].item()))
        except Exception as exc:
            warnings.append('Background metadata unavailable: ' + str(exc))

    output_dir = resolve_reference(policy.get('output_dir', 'results/continuous'), root, vision_path)
    runs = [p for p in output_dir.glob('run_*') if p.is_dir() and not p.is_symlink()]
    if runs:
        def run_time(path):
            journal = path / 'events.jsonl'
            return (journal if journal.exists() else path).stat().st_mtime_ns
        latest = max(runs, key=run_time)
        snapshot['latest_continuous_run'] = str(latest)
        for name in ('contract.json', 'configuration.json', 'events.jsonl'):
            path = latest / name
            label = None if path.is_relative_to(root) else 'latest_continuous/' + name
            add(path, label, tail=name == 'events.jsonl')
        scans = sorted(latest.glob('scan_*/result.json'), key=lambda p: p.stat().st_mtime_ns)[-2:]
        for scan in scans:
            for path in (scan, scan.parent / 'dashboard.png'):
                label = None if path.is_relative_to(root) else 'latest_continuous/' + scan.parent.name + '/' + path.name
                add(path, label)
    live_dir = resolve_reference(cfg.get('runtime', {}).get('output_dir', 'results'), root, vision_path)
    live_results = list(live_dir.glob('live_*/inspection_*/result.json'))
    if live_results:
        latest = max(live_results, key=lambda p: p.stat().st_mtime_ns)
        snapshot['latest_live_result'] = str(latest)
        add(latest)
    weights = cfg.get('yolo', {}).get('weights')
    if weights:
        path = resolve_reference(weights, root, vision_path)
        snapshot['model'] = {'path': str(path), 'included': False, 'exists': path.is_file(),
                             'size_bytes': path.stat().st_size if path.is_file() else None}

    manifest, total = [], 0
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is outside cleanup: a race must never delete someone else's file.
    owned_stream = output.open('xb')
    try:
        with owned_stream, zipfile.ZipFile(owned_stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for label, (path, tail) in sorted(sources.items()):
                try:
                    before = path.stat()
                    if before.st_size > MAX_FILE and not tail:
                        warnings.append('Skipped file over 8 MiB: ' + label)
                        continue
                    with path.open('rb') as stream:
                        if tail and before.st_size > JOURNAL_TAIL:
                            stream.seek(-JOURNAL_TAIL, os.SEEK_END)
                            stream.readline()
                            warnings.append('Only last 512 KiB of journal included: ' + label)
                        data = stream.read(MAX_FILE + 1)
                    if len(data) > MAX_FILE or total + len(data) > MAX_TOTAL:
                        warnings.append('Skipped due to file/total size limit: ' + label)
                        continue
                    if tail and data and not data.endswith(b'\n'):
                        end = data.rfind(b'\n')
                        data = data[:end + 1] if end >= 0 else b''
                        warnings.append('Incomplete final journal record omitted: ' + label)
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        warnings.append('Changed during export; export again after stopping writers: ' + label)
                    archive.writestr(root.name + '/' + label, data)
                    manifest.append({'path': label, 'size_bytes': len(data),
                                     'sha256': hashlib.sha256(data).hexdigest()})
                    total += len(data)
                except OSError as exc:
                    warnings.append('Cannot archive %s: %s' % (label, exc))
            snapshot['warnings'] = warnings
            snapshot['archived_source_files'] = len(manifest)
            archive.writestr(root.name + '/HANDOVER_SNAPSHOT.json', json.dumps(snapshot, ensure_ascii=False, indent=2))
            archive.writestr(root.name + '/HANDOVER_MANIFEST.json', json.dumps({'files': manifest, 'warnings': warnings}, ensure_ascii=False, indent=2))
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return output, manifest, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', default='config/picking_robot.yaml')
    parser.add_argument('--continuous-config', default='config/continuous.yaml')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    vision = resolve_reference(args.config, root, root / 'config/project.yaml')
    continuous = resolve_reference(args.continuous_config, root, root / 'config/project.yaml')
    output = args.output or root.parent / datetime.now().strftime('phoxi-chat-context_%Y%m%d_%H%M%S_%f.zip')
    output, files, warnings = export(root, vision, continuous, output)
    print('FILE TO ATTACH IN THE NEW CHAT:', output)
    print('Files:', len(files), '| ZIP bytes:', output.stat().st_size)
    print('No camera/robot connection. Existing source and configuration were not modified.')
    for warning in warnings:
        print('NOTE:', warning)


if __name__ == '__main__':
    main()
