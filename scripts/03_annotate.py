#!/usr/bin/env python3
from __future__ import annotations

import argparse

from phoxi_yolo3d.annotator import OpenCVPolygonAnnotator
from phoxi_yolo3d.config import load_class_names, load_config, project_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Gan polygon thu cong va luu YOLO segmentation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--review", action="store_true", help="Mo ca cac anh da co nhan")
    parser.add_argument(
        "--no-auto-fallback",
        action="store_true",
        help="Khong nap nhan goi y tu data/raw/auto_labels",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    annotator = OpenCVPolygonAnnotator(
        images_dir=project_path(dataset_cfg["raw_images"], config),
        labels_dir=project_path(dataset_cfg["raw_labels"], config),
        auto_labels_dir=(
            None
            if args.no_auto_fallback
            else project_path(dataset_cfg["auto_labels"], config)
        ),
        class_names=load_class_names(),
        review_all=args.review,
    )
    annotator.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

