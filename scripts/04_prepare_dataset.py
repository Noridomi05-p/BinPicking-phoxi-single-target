#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from phoxi_yolo3d.config import load_class_names, load_config, project_path
from phoxi_yolo3d.dataset import prepare_dataset, validate_raw_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiem tra nhan va chia train/val/test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--labelled-only",
        action="store_true",
        help=(
            "Chi kiem tra/chia cac anh da co file nhan; "
            "bo qua anh chua gan nhan de train model pilot"
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["dataset"]
    classes = load_class_names()
    images = project_path(cfg["raw_images"], config)
    labels = project_path(cfg["raw_labels"], config)
    if args.check_only:
        report = validate_raw_labels(
            images,
            labels,
            classes,
            allow_missing_labels=args.labelled_only,
        )
    else:
        report = prepare_dataset(
            images_dir=images,
            labels_dir=labels,
            output_dir=project_path(cfg["prepared_dir"], config),
            class_names=classes,
            ratios=(
                float(cfg["train_ratio"]),
                float(cfg["val_ratio"]),
                float(cfg["test_ratio"]),
            ),
            seed=int(cfg["seed"]),
            overwrite=args.overwrite,
            labelled_only=args.labelled_only,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.check_only and report["errors"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
