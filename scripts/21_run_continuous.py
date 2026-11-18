#!/usr/bin/env python3
"""Open disarmed GUI; no camera/robot connection until the operator presses Start."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'config/picking_robot.yaml'))
    parser.add_argument('--continuous-config', default=str(ROOT/'config/continuous.yaml'))
    args = parser.parse_args()
    import tkinter as tk
    from phoxi_yolo3d.continuous.gui import ContinuousGUI
    root = tk.Tk()
    ContinuousGUI(root, args.config, args.continuous_config)
    root.mainloop()


if __name__ == '__main__':
    main()
