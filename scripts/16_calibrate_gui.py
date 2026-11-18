#!/usr/bin/env python3
"""Calibrate a fixed PhoXi against BASE using a rigid robot-mounted checkerboard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/picking.yaml"), help="Existing camera configuration")
    parser.add_argument("--calibration-config", default=str(ROOT / "config/calibration.yaml"))
    parser.add_argument("--session", help="Resume a session directory or session.json")
    parser.add_argument("--demo", action="store_true", help="Synthetic camera; never connects to robot / scanner")
    parser.add_argument("--solve", metavar="SESSION", help="Solve an existing session without opening GUI")
    parser.add_argument("--export", metavar="JSON", help="Export a complete result after --solve")
    parser.add_argument("--diagnose-scan", metavar="NPZ", help="Check checkerboard 3D pose on a recorded frame")
    args = parser.parse_args()
    try:
        import yaml
        from phoxi_yolo3d.calibration.board import BoardSpec, estimate_board
        from phoxi_yolo3d.calibration.session import Session
        from phoxi_yolo3d.calibration.solver import solve
        options = yaml.safe_load(Path(args.calibration_config).read_text(encoding="utf-8")) or {}
        if args.solve:
            session = Session.load(args.solve)
            result = session.store_result(solve(session.doc["samples"], session.doc["solver_settings"]))
            if args.export:
                session.export(args.export)
            print(json.dumps({k: result[k] for k in ("status", "demo", "T_base_camera", "fit", "validation", "issues")}, ensure_ascii=False, indent=2))
            return 0
        if args.export:
            parser.error("--export đi cùng --solve SESSION")
        if args.diagnose_scan:
            from phoxi_yolo3d.storage import load_capture
            b = options.get("board", {})
            spec = BoardSpec(int(b.get("cols", 15)), int(b.get("rows", 10)), float(b.get("square_mm", 16)))
            result = estimate_board(load_capture(args.diagnose_scan), spec, settings=options.get("detection", {}))
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return 0
        from phoxi_yolo3d.config import load_config
        config = load_config(args.config)
        import tkinter as tk
        from phoxi_yolo3d.calibration.gui import CalibrationApp
        root = tk.Tk()
        CalibrationApp(root, config, options, args.demo, args.session)
        root.mainloop()
        return 0
    except ImportError as exc:
        print(f"Thiếu thư viện: {exc}\nKích hoạt .venv. Cài các dependency của dự án.\nNếu thiếu tkinter: sudo apt install python3-tk", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Calibration: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
