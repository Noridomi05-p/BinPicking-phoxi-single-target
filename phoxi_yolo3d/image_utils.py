from __future__ import annotations

import numpy as np


def texture_recipe(settings):
    cfg = settings.get("clahe", {})
    return {"lower_percentile": float(settings.get("lower_percentile", 1.)),
            "upper_percentile": float(settings.get("upper_percentile", 99.5)),
            "clahe": {"enabled": bool(cfg.get("enabled", False)),
                      "clip_limit": float(cfg.get("clip_limit", 1.5)),
                      "tile_grid_size": list(cfg.get("tile_grid_size", [8, 8]))}}


def prepare_frame_texture(frame, settings):
    """Idempotent recipe shared by capture, inference and replay; raw stays intact."""
    recipe = texture_recipe(settings)
    previous = frame.info.get("texture_recipe")
    if previous == recipe:
        return frame.texture_bgr
    if frame.texture_raw is not None:
        image = normalize_texture(frame.texture_raw, recipe["lower_percentile"], recipe["upper_percentile"])
    elif previous is not None:
        raise ValueError("RAW_TEXTURE_REQUIRED_FOR_PREPROCESSING_CHANGE")
    else:
        image = frame.texture_bgr.copy()
    cfg = recipe["clahe"]
    if cfg["enabled"]:
        import cv2
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=cfg["clip_limit"], tileGridSize=tuple(cfg["tile_grid_size"])).apply(lab[..., 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    frame.texture_bgr = image
    frame.info["texture_recipe"] = recipe
    return image


def _scale_channel(channel: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    source = np.asarray(channel, dtype=np.float32)
    finite = source[np.isfinite(source)]
    if finite.size == 0:
        return np.zeros(source.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [low_pct, high_pct])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros(source.shape, dtype=np.uint8)
    scaled = (np.nan_to_num(source, nan=low, posinf=high, neginf=low) - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def normalize_texture(
    texture: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """Convert PhoXi uint16/float/grayscale/RGB texture to uint8 BGR."""

    array = np.asarray(texture)
    array = np.squeeze(array)
    if array.ndim == 2:
        gray = (
            array.astype(np.uint8, copy=False)
            if array.dtype == np.uint8
            else _scale_channel(array, lower_percentile, upper_percentile)
        )
        return np.repeat(gray[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"Texture khong duoc ho tro: shape={array.shape}, dtype={array.dtype}")
    array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.stack(
            [
                _scale_channel(array[:, :, channel], lower_percentile, upper_percentile)
                for channel in range(3)
            ],
            axis=2,
        )
    # PhoXi color arrays are RGB; OpenCV and the rest of this project use BGR.
    return np.ascontiguousarray(array[:, :, ::-1])


def erode_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if radius_px <= 0:
        return binary.copy()
    try:
        import cv2

        size = radius_px * 2 + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        return cv2.erode(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
    except ImportError:
        padded = np.pad(binary, radius_px, mode="constant", constant_values=False)
        result = np.ones_like(binary)
        height, width = binary.shape
        for dy in range(2 * radius_px + 1):
            for dx in range(2 * radius_px + 1):
                result &= padded[dy : dy + height, dx : dx + width]
        return result


def resize_mask_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.shape == (height, width):
        return binary.astype(bool)
    try:
        import cv2

        resized = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
    except ImportError:
        y_idx = np.minimum((np.arange(height) * binary.shape[0] / height).astype(int), binary.shape[0] - 1)
        x_idx = np.minimum((np.arange(width) * binary.shape[1] / width).astype(int), binary.shape[1] - 1)
        resized = binary[y_idx[:, None], x_idx[None, :]]
    return resized.astype(bool)

