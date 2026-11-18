#!/usr/bin/env python3
"""Read-only diagnostics: current background identity, YAML labels and last recorded scanner values."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from phoxi_yolo3d.config import load_config,project_path
    from phoxi_yolo3d.surface_reference import SurfaceReference
    import yaml
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'config/picking_robot.yaml'))
    p.add_argument('--continuous-config',default=str(ROOT/'config/continuous.yaml'))
    args=p.parse_args()
    cfg=load_config(args.config)
    path=project_path(cfg['postprocess']['surface']['path'],cfg)
    reference=SurfaceReference.load(path)
    print('REFERENCE_FILE:',path)
    print('SAVED_IDENTITY:',json.dumps(reference.identity,ensure_ascii=False))
    print('RUN_LIVE_YAML_LABELS:',json.dumps({k:cfg['camera'].get(k) for k in ('texture_source','led_power')},ensure_ascii=False))
    policy=yaml.safe_load(Path(args.continuous_config).read_text())['continuous']
    records=list(project_path(policy['output_dir'],cfg).glob('run_*/scan_*/result.json'))
    if not records:
        print('No recorded continuous scan; no camera or robot connection was made.')
        return
    latest=max(records,key=lambda f:f.stat().st_mtime_ns)
    record=json.loads(latest.read_text(encoding='utf-8'))
    values=(record.get('camera_settings') or {}).get('values') or {}
    print('LAST_RECORDED_SCAN:',latest)
    print('LAST_SCANNER_READBACK:',json.dumps(values,ensure_ascii=False))
    actual=record.get('acquisition_identity') or (record.get('reference_identity') or {}).get('frame_identity')
    if actual is None:
        actual={}
        for key,setting in [('texture_source','CapturingSettings/TextureSource'),('led_power','CapturingSettings/LEDPower')]:
            if setting in values: actual[key]=values[setting]
        print('Old result format: only recorded illumination fields can be compared; other fields are unknown.')
    differences={k:{'saved_background':reference.identity.get(k),'last_scan':v}
                 for k,v in actual.items() if reference.identity.get(k)!=v}
    print('RECORDED_DIFFERENCES:',json.dumps(differences,ensure_ascii=False))
    print('These are saved records, not a query of the current physical camera. No settings were changed.')


if __name__=='__main__':
    main()
