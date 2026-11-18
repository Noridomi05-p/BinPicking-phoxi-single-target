from __future__ import annotations

import math
from pathlib import Path
from pprint import pformat

from .contract import digest

CYCLE_CODE = '# Custom Code SCRIPT inside MainSub; set MainSub repeat to the desired count.\nphoxi_run_once()\n'


def render_controller(contract, policy):
    header = (
        '# Generated from the current picking_robot.yaml and continuous.yaml.\n'
        '# Paste this complete file into a Custom Code SCRIPT BEFORE MainSub.\n'
        '# Add phoxi_run_once() in a second Custom Code SCRIPT INSIDE MainSub.\n'
        '# Fixed commissioned route, no automatic obstacle avoidance.\n'
        'PX_PC_IP = '+repr(policy['listen'])+'\n'
        'PX_PC_PORT = '+str(policy['port'])+'\n'
        'PX_CONTRACT_ID = '+repr(digest(contract))+'\n'
        'PX_MIN_DOWN_COS = '+repr(math.cos(math.radians(contract['max_tilt_deg'])))+'\n'
        'PX_CONTRACT = '+pformat(contract, width=100, sort_dicts=True)+'\n\n'
    )
    return header+(Path(__file__).with_name('task_writer_body.drl')).read_text(encoding='utf-8')


def write_controller(contract, policy, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    init = output/'task_writer_continuous_init.drl'
    cycle = output/'task_writer_continuous_cycle.drl'
    init.write_text(render_controller(contract, policy), encoding='utf-8')
    cycle.write_text(CYCLE_CODE, encoding='utf-8')
    return init, cycle
