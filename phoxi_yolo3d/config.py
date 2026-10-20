from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .config_validation import validate_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "project.yaml"
DEFAULT_CLASSES_PATH = PROJECT_ROOT / "config" / "classes.txt"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    default = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(default, dict):
        raise ValueError("project.yaml phai la mapping YAML.")
    if path is None:
        config = default
    else:
        supplied_path = Path(path).expanduser().resolve()
        supplied = yaml.safe_load(supplied_path.read_text(encoding="utf-8")) or {}
        if not isinstance(supplied, dict):
            raise ValueError("File config phai la mapping YAML.")
        config = _deep_merge(default, supplied)
    config["_project_root"] = str(PROJECT_ROOT)
    config["_config_dir"] = str(DEFAULT_CONFIG_PATH.parent if path is None else supplied_path.parent)
    validate_config(config)
    return config


def validate_model_classes(model_names, class_names: list[str]) -> None:
    if len(model_names) != len(class_names):
        raise ValueError(f"So lop model ({len(model_names)}) khac classes.txt ({len(class_names)}).")
    if isinstance(model_names, dict) and set(model_names) != set(range(len(class_names))):
        raise ValueError("ID lop cua model phai lien tuc tu 0, giu dung thu tu classes.txt.")
    ordered = [str(model_names[i]) for i in range(len(class_names))]
    if ordered != class_names:
        raise ValueError(f"Thu tu class model {ordered} khac config {class_names}. Khong auto-label.")


def project_path(value: str | Path, config: dict[str, Any] | None = None) -> Path:
    if str(value).startswith("@config/"):
        return (Path((config or {}).get("_config_dir", DEFAULT_CONFIG_PATH.parent)) / str(value)[8:]).resolve()
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root = Path((config or {}).get("_project_root", PROJECT_ROOT))
    return (root / path).resolve()


def load_class_names(path: str | Path | None = None) -> list[str]:
    class_path = Path(path).expanduser().resolve() if path else DEFAULT_CLASSES_PATH
    names: list[str] = []
    for raw_line in class_path.read_text(encoding="utf-8").splitlines():
        name = raw_line.strip()
        if not name or name.startswith("#"):
            continue
        names.append(name)
    if not names:
        raise ValueError(
            "Chua co ten lop. Chay: python scripts/01_set_classes.py --classes \"lop_1,lop_2\""
        )
    if len(set(names)) != len(names):
        raise ValueError("Ten lop bi trung trong config/classes.txt")
    return names

