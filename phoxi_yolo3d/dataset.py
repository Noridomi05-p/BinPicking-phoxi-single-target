from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import yaml


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_yolo_segmentation(path: str | Path) -> list[tuple[int, list[float]]]:
    label_path = Path(path)
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    objects: list[tuple[int, list[float]]] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{label_path}:{line_number}: dong nhan khong hop le") from exc
        if len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError(
                f"{label_path}:{line_number}: polygon can it nhat 3 diem x,y"
            )
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(
                f"{label_path}:{line_number}: toa do YOLO phai nam trong [0,1]"
            )
        objects.append((class_id, coordinates))
    return objects


def write_yolo_segmentation(
    path: str | Path, objects: Iterable[tuple[int, Iterable[float]]]
) -> None:
    label_path = Path(path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for class_id, coordinates in objects:
        values = " ".join(f"{float(value):.6f}" for value in coordinates)
        lines.append(f"{int(class_id)} {values}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def validate_raw_labels(
    images_dir: str | Path,
    labels_dir: str | Path,
    class_names: list[str],
    allow_missing_labels: bool = False,
) -> dict:
    images = list_images(images_dir)
    labels_root = Path(labels_dir)
    errors: list[str] = []
    warnings: list[str] = []
    class_instances: Counter[int] = Counter()
    labelled_images = 0
    unlabelled_images = 0
    empty_images = 0

    for image_path in images:
        label_path = labels_root / f"{image_path.stem}.txt"
        if not label_path.exists():
            unlabelled_images += 1
            if not allow_missing_labels:
                errors.append(f"Thieu nhan: {label_path}")
            continue
        try:
            objects = parse_yolo_segmentation(label_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        labelled_images += 1
        if not objects:
            empty_images += 1
        for class_id, _ in objects:
            if class_id < 0 or class_id >= len(class_names):
                errors.append(
                    f"{label_path}: class_id={class_id} ngoai [0,{len(class_names)-1}]"
                )
            else:
                class_instances[class_id] += 1

    image_stems = {path.stem for path in images}
    for label_path in labels_root.glob("*.txt") if labels_root.exists() else []:
        if label_path.stem not in image_stems:
            warnings.append(f"Nhan khong co anh: {label_path}")
    for class_id, class_name in enumerate(class_names):
        if class_instances[class_id] == 0:
            warnings.append(f"Lop '{class_name}' chua co instance nao")
    if allow_missing_labels and unlabelled_images:
        warnings.append(
            f"Che do labelled-only: bo qua {unlabelled_images} anh chua co nhan"
        )

    return {
        "image_count": len(images),
        "labelled_image_count": labelled_images,
        "unlabelled_image_count": unlabelled_images,
        "empty_image_count": empty_images,
        "class_instances": {
            class_names[class_id]: class_instances[class_id]
            for class_id in range(len(class_names))
        },
        "errors": errors,
        "warnings": warnings,
    }


def _split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    train_ratio, val_ratio, test_ratio = ratios
    if total < 3:
        return total, 0, 0
    val_count = max(1, round(total * val_ratio)) if val_ratio > 0 else 0
    test_count = max(1, round(total * test_ratio)) if test_ratio > 0 else 0
    train_count = total - val_count - test_count
    if train_count < 1:
        train_count = 1
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
    return train_count, val_count, test_count


def prepare_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    class_names: list[str],
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42,
    overwrite: bool = False,
    labelled_only: bool = False,
) -> dict:
    ratio_sum = sum(ratios)
    if any(value < 0 for value in ratios) or abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Ti le train/val/test phai >=0 va tong bang 1, nhan {ratios}")

    report = validate_raw_labels(
        images_dir,
        labels_dir,
        class_names,
        allow_missing_labels=labelled_only,
    )
    if report["errors"]:
        preview = "\n".join(report["errors"][:20])
        raise ValueError(f"Bo nhan chua hop le:\n{preview}")

    all_images = list_images(images_dir)
    labels_root = Path(labels_dir)
    if labelled_only:
        images = [
            image_path
            for image_path in all_images
            if (labels_root / f"{image_path.stem}.txt").exists()
        ]
    else:
        images = all_images
    if not images:
        if labelled_only:
            raise ValueError("Khong co anh nao da gan nhan de chia dataset pilot.")
        raise ValueError("Khong co anh de chia dataset.")

    output = Path(output_dir).resolve()
    managed_children = [output / "images", output / "labels"]
    if any(path.exists() for path in managed_children) and not overwrite:
        raise FileExistsError(
            f"Dataset da ton tai tai {output}. Dung --overwrite neu muon tao lai."
        )
    if overwrite:
        for child in managed_children:
            if child.exists():
                shutil.rmtree(child)

    shuffled = images.copy()
    random.Random(seed).shuffle(shuffled)
    train_count, val_count, test_count = _split_counts(len(shuffled), ratios)
    split_images = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count : train_count + val_count + test_count],
    }
    split_stats: dict[str, dict] = {}
    for split_name, members in split_images.items():
        image_target = output / "images" / split_name
        label_target = output / "labels" / split_name
        image_target.mkdir(parents=True, exist_ok=True)
        label_target.mkdir(parents=True, exist_ok=True)
        counts: Counter[int] = Counter()
        for source_image in members:
            source_label = labels_root / f"{source_image.stem}.txt"
            shutil.copy2(source_image, image_target / source_image.name)
            shutil.copy2(source_label, label_target / source_label.name)
            for class_id, _ in parse_yolo_segmentation(source_label):
                counts[class_id] += 1
        split_stats[split_name] = {
            "images": len(members),
            "instances": {
                class_names[class_id]: counts[class_id]
                for class_id in range(len(class_names))
            },
        }

    dataset_yaml = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    yaml_path = output / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    final_report = {
        "dataset_yaml": str(yaml_path),
        "seed": seed,
        "labelled_only": labelled_only,
        "selected_image_count": len(images),
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "splits": split_stats,
        "raw_validation": report,
    }
    (output / "dataset_report.json").write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return final_report
