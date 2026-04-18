"""Image utility functions for loading, conversion and helpers."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import tifffile  # type: ignore
    HAS_TIFFFILE = True
except Exception:
    HAS_TIFFFILE = False


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 range [0, 255]."""
    if img is None:
        return img
    if img.dtype == np.uint8:
        return img
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert any image to single-channel grayscale uint8."""
    img8 = ensure_uint8(img)
    if img8.ndim == 2:
        return img8
    if img8.shape[2] == 4:
        return cv2.cvtColor(img8, cv2.COLOR_BGRA2GRAY)
    if img8.shape[2] == 3:
        return cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
    return img8[:, :, 0]


def cv_to_pil_rgb(img: np.ndarray) -> Image.Image:
    """Convert a BGR/BGRA/Gray OpenCV image to an RGB PIL Image."""
    img8 = ensure_uint8(img)
    if img8.ndim == 2:
        rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
    elif img8.shape[2] == 4:
        rgb = cv2.cvtColor(img8, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(img8, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def dedupe_xy(
    xs: List[float], ys: List[float], eps: float = 1e-6
) -> Tuple[List[float], List[float]]:
    """Remove consecutive duplicate (x, y) pairs."""
    if not xs:
        return xs, ys
    out_x = [xs[0]]
    out_y = [ys[0]]
    for x, y in zip(xs[1:], ys[1:]):
        if abs(x - out_x[-1]) > eps or abs(y - out_y[-1]) > eps:
            out_x.append(x)
            out_y.append(y)
    return out_x, out_y


def load_frames(path: str) -> List[np.ndarray]:
    """Load an image or multi-frame TIFF stack into a list of frames."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext in (".tif", ".tiff") and HAS_TIFFFILE:
        data = tifffile.imread(path)
        if data.ndim == 2:
            return [data]
        if data.ndim == 3:
            if data.shape[-1] in (1, 2, 3, 4) and data.shape[0] > 8 and data.shape[1] > 8:
                return [data]
            return [data[i] for i in range(data.shape[0])]
        if data.ndim == 4:
            return [data[i] for i in range(data.shape[0])]
        return [data]

    # Pillow fallback (multi-page TIFF too)
    try:
        pil = Image.open(path)
        frames: List[np.ndarray] = []
        idx = 0
        while True:
            try:
                pil.seek(idx)
                arr = np.array(pil)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                elif arr.ndim == 3 and arr.shape[2] == 4:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
                frames.append(arr)
                idx += 1
            except EOFError:
                break
        if frames:
            return frames
    except Exception:
        pass

    # OpenCV fallback (single image)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return [img]
