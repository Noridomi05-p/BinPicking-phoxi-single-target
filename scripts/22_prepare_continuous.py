#!/usr/bin/env python3
"""Generate matching Task Writer definitions from the user's current TCP/points/settings."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from phoxi_yolo3d.continuous.contract import read_configuration, digest
    from phoxi_yolo3d.continuous.controller import write_controller
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'config/picking_robot.yaml'))
    p.add_argument('--continuous-config', default=str(ROOT/'config/continuous.yaml'))
    p.add_argument('--output', default=str(ROOT/'generated'))
    args = p.parse_args()
    cfg, policy, contract, watched = read_configuration(args.config, args.continuous_config)
    for path in write_controller(contract, policy, args.output):
        print(path)
    print('CONTRACT_ID:', digest(contract))
    print('Before MainSub: task_writer_continuous_init.drl; inside MainSub: phoxi_run_once()')


if __name__ == '__main__':
    main()
