from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .types import Frame3D


def utc_stem(prefix: str = "phoxi") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"{prefix}_{stamp}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_bgr_png(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(image_bgr, dtype=np.uint8)[:, :, ::-1]
    Image.fromarray(rgb, mode="RGB").save(path)


def load_bgr_png(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def frame_statistics(frame: Frame3D) -> dict[str, Any]:
    z = frame.point_cloud_mm[:, :, 2]
    valid = np.isfinite(frame.point_cloud_mm).all(axis=2) & (z > 0)
    stats: dict[str, Any] = {
        "image_shape": list(frame.texture_bgr.shape),
        "point_cloud_shape": list(frame.point_cloud_mm.shape),
        "valid_3d_points": int(valid.sum()),
        "valid_3d_ratio": float(valid.mean()),
    }
    if valid.any():
        stats["z_mm_percentiles"] = {
            str(p): float(np.percentile(z[valid], p)) for p in (1, 5, 50, 95, 99)
        }
    if frame.confidence_map is not None:
        conf = frame.confidence_map[np.isfinite(frame.confidence_map)]
        if conf.size:
            stats["confidence_percentiles"] = {
                str(p): float(np.percentile(conf, p)) for p in (1, 5, 50, 95, 99)
            }
    return stats


def save_capture(
    frame: Frame3D,
    output_dir: str | Path,
    stem: str | None = None,
    save_3d: bool = False,
) -> dict[str, Path]:
    frame.validate()
    root = Path(output_dir)
    stem = stem or utc_stem()
    image_path = root / "images" / f"{stem}.png"
    metadata_path = root / "metadata" / f"{stem}.json"
    save_bgr_png(image_path, frame.texture_bgr)

    metadata = {
        "stem": stem,
        "frame_id": frame.frame_id,
        "timestamp_utc": frame.timestamp_utc,
        "coordinate_space": "camera",
        "distance_unit": "millimetre",
        "frame_info": _json_safe(frame.info),
        "statistics": frame_statistics(frame),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    paths = {"image": image_path, "metadata": metadata_path}
    if save_3d:
        scan_path = root / "scans" / f"{stem}.npz"
        scan_path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "texture_bgr": frame.texture_bgr,
            "point_cloud_mm": frame.point_cloud_mm.astype(np.float32, copy=False),
            "metadata_json": np.array(json.dumps(metadata, ensure_ascii=False)),
        }
        if frame.texture_raw is not None:
            arrays["texture_raw"] = frame.texture_raw
        if frame.depth_mm is not None:
            arrays["depth_mm"] = frame.depth_mm.astype(np.float32, copy=False)
        if frame.confidence_map is not None:
            arrays["confidence_map"] = frame.confidence_map.astype(np.float32, copy=False)
        if frame.normal_map is not None:
            arrays["normal_map"] = frame.normal_map.astype(np.float32, copy=False)
        np.savez_compressed(scan_path, **arrays)
        paths["scan"] = scan_path
    return paths


def load_capture(scan_path: str | Path) -> Frame3D:
    path = Path(scan_path)
    with np.load(path, allow_pickle=False) as data:
        metadata: dict[str, Any] = {}
        if "metadata_json" in data:
            metadata = json.loads(str(data["metadata_json"].item()))
        frame = Frame3D(
            texture_bgr=np.array(data["texture_bgr"], copy=True),
            texture_raw=(
                np.array(data["texture_raw"], copy=True) if "texture_raw" in data else None
            ),
            point_cloud_mm=np.array(data["point_cloud_mm"], dtype=np.float32, copy=True),
            depth_mm=(
                np.array(data["depth_mm"], dtype=np.float32, copy=True)
                if "depth_mm" in data
                else None
            ),
            confidence_map=(
                np.array(data["confidence_map"], dtype=np.float32, copy=True)
                if "confidence_map" in data
                else None
            ),
            normal_map=(
                np.array(data["normal_map"], dtype=np.float32, copy=True)
                if "normal_map" in data
                else None
            ),
            frame_id=metadata.get("frame_id"),
            timestamp_utc=metadata.get("timestamp_utc"),
            info=metadata.get("frame_info", {}),
        )
    frame.validate()
    return frame

