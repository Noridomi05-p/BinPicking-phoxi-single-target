#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from phoxi_yolo3d.config import load_config
from phoxi_yolo3d.pipeline import PhoXiYoloPipeline
from phoxi_yolo3d.postprocess import build_postprocessor
from phoxi_yolo3d.picking_visualization import draw_picking
from phoxi_yolo3d.replay import load_frozen_inspection
from phoxi_yolo3d.storage import save_bgr_png


def main():
    parser = argparse.ArgumentParser(description="Replay masks va 3D da luu; khong can camera/YOLO/GPU.")
    parser.add_argument("inspection", type=Path)
    parser.add_argument("--config", type=Path, help="Override de thu nguong; mac dinh replay.yaml cua scan.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.inspection.resolve()
    frame, segmenter = load_frozen_inspection(root)
    config = load_config(args.config or root / "replay.yaml")
    # Saved local references belong to this frame, even when testing new settings.
    for key, name in (("roi", "bin_roi.json"), ("surface", "bin_surface.npz")):
        config.setdefault("postprocess", {}).setdefault(key, {})["path"] = str(root / "reference" / name)
    config["picking"]["enabled"] = True
    post = build_postprocessor(config)
    pipeline = PhoXiYoloPipeline(segmenter, config["measurement"], post,
        config.get("postprocess", {}).get("duplicates", {}), config["picking"], config["texture"], offline=True)
    accepted, rejected = pipeline.process_all(frame)
    output = args.output or root / "picking_replay"
    output.mkdir(parents=True, exist_ok=True)
    report = {"frame_id": frame.frame_id, "timestamp_utc": frame.timestamp_utc,
              "objects": [r.to_dict() for r in accepted], "rejected": [r.to_dict() for r in rejected],
              "picking": pipeline.last_picking, "robot_ready": False}
    report['pipeline_timings_ms'] = pipeline.last_timings_ms
    (output / "picking.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    save_bgr_png(output / "target.png", draw_picking(frame, accepted, rejected, pipeline.last_picking,
                                                   debug=args.debug, roi=post.roi))
    print(json.dumps({"status": pipeline.last_picking["status"], "target": pipeline.last_picking["target_instance_id"],
                      "output": str(output), "robot_ready": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
