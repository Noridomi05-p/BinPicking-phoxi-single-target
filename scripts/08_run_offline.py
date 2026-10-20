#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from phoxi_yolo3d.config import load_config, project_path
from phoxi_yolo3d.detector import YOLOInstanceSegmenter
from phoxi_yolo3d.diagnostics import frame_diagnostics, result_counts
from phoxi_yolo3d.pipeline import PhoXiYoloPipeline
from phoxi_yolo3d.postprocess import build_postprocessor
from phoxi_yolo3d.storage import load_capture, save_bgr_png
from phoxi_yolo3d.visualization import depth_preview, draw_results, make_dashboard
from phoxi_yolo3d.picking_visualization import draw_picking


def main() -> int:
    parser = argparse.ArgumentParser(description="Nhan dien + do 3D tren cac file NPZ da luu")
    parser.add_argument("source", help="Mot file .npz hoac thu muc scans")
    parser.add_argument("--config", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    yolo_cfg = config["yolo"]
    if args.conf is not None:
        if not 0 <= args.conf <= 1:
            parser.error("--conf phai nam trong [0,1]")
        yolo_cfg["confidence"] = args.conf
    if args.imgsz is not None:
        if args.imgsz <= 0:
            parser.error("--imgsz phai > 0")
        yolo_cfg["image_size"] = args.imgsz
    weights = Path(args.weights).expanduser().resolve() if args.weights else project_path(yolo_cfg["weights"], config)
    source = Path(args.source).expanduser().resolve()
    scans = [source] if source.is_file() else sorted(source.glob("*.npz"))
    if not scans:
        raise SystemExit(f"Khong tim thay file .npz tai {source}")
    postprocessor = build_postprocessor(config)
    pipeline = PhoXiYoloPipeline(
        YOLOInstanceSegmenter(weights, yolo_cfg),
        config["measurement"],
        postprocessor=postprocessor,
        duplicate_settings=config.get("postprocess", {}).get("duplicates", {}),
        picking_settings=config.get("picking", {}), texture_settings=config.get("texture", {}), offline=True,
    )
    output = project_path(config["runtime"]["output_dir"], config) / "offline"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for scan in scans:
        started = time.perf_counter()
        frame = load_capture(scan)
        results, rejected = pipeline.process_all(frame)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        counts = result_counts(results, rejected)
        diagnostics = frame_diagnostics(frame, config["measurement"], postprocessor)
        save_bgr_png(
            output / f"{scan.stem}_annotated.png",
            make_dashboard(draw_picking(
                frame, results, rejected, pipeline.last_picking,
                alpha=float(config["runtime"].get("mask_alpha", 0.22)),
                fill_masks=bool(config["runtime"].get("mask_fill", True)),
                debug=args.debug or bool(config["runtime"].get("debug_rejected", False)),
                roi=postprocessor.roi,
            ), depth_preview(frame.depth_mm, frame.point_cloud_mm), footer_lines=[
                f"2D={len(results)} 3D-confirmed={counts['verified_3d_count']} REJECT={len(rejected)}",
                " | ".join(postprocessor.status_lines()),
                f"ROI valid 3D={diagnostics['pointcloud_valid_ratio_roi']:.1%}",
            ]),
        )
        records.append(
            {
                "scan": str(scan),
                "processing_ms": elapsed_ms,
                "pipeline_timings_ms": pipeline.last_timings_ms,
                **counts,
                "diagnostics": diagnostics,
                "inference_settings": dict(yolo_cfg),
                "objects": [item.to_dict() for item in results],
                "rejected": [item.to_dict() for item in rejected],
                "picking": pipeline.last_picking,
                "robot_ready": False,
            }
        )
        print(
            f"{scan.name}: {len(results)} objects, "
            f"{len(rejected)} rejected, {elapsed_ms:.1f} ms"
        )
    report = output / "offline_results.json"
    report.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Da luu: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
