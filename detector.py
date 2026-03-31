"""Contour detection engine with preprocessing and caching."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .models import DetectParams, PreprocessParams
from .utils import ensure_uint8, load_frames, to_gray


class ContourDetector:
    """Loads images / TIFF stacks, applies preprocessing, and detects contours."""

    def __init__(self):
        self.frames: List[np.ndarray] = []
        self.current_frame_idx: int = 0
        self.image_path: Optional[str] = None
        self.original: Optional[np.ndarray] = None

        # pre_cache[frame_idx][pre_key] = gray_preprocessed_uint8
        self.pre_cache: Dict[int, Dict[int, np.ndarray]] = {}
        # detect cache per frame
        self.cache: Dict[int, Dict] = {}

    # ------------------------------------------------------------------
    # Frame management
    # ------------------------------------------------------------------

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def load(self, path: str):
        self.image_path = path
        self.frames = load_frames(path)
        if not self.frames:
            raise ValueError("No frames found.")
        self.pre_cache.clear()
        self.cache.clear()
        self.set_frame(0)

    def set_frame(self, idx: int):
        if not self.frames:
            return
        idx = max(0, min(idx, self.num_frames - 1))
        self.current_frame_idx = idx
        self.original = self.frames[idx]

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def get_preprocessed_gray(self, pre: PreprocessParams) -> np.ndarray:
        if self.original is None:
            raise ValueError("No image loaded.")
        pre_key = hash(pre)
        fi = self.current_frame_idx

        if fi in self.pre_cache and pre_key in self.pre_cache[fi]:
            return self.pre_cache[fi][pre_key]

        gray = to_gray(self.original)

        if pre.invert:
            gray = cv2.bitwise_not(gray)

        gray = cv2.convertScaleAbs(gray, alpha=float(pre.alpha), beta=int(pre.beta))

        k = int(pre.blur_ksize)
        if k % 2 == 0:
            k += 1
        k = max(1, k)
        if k > 1:
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        self.pre_cache.setdefault(fi, {})[pre_key] = gray
        return gray

    # ------------------------------------------------------------------
    # Contour properties
    # ------------------------------------------------------------------

    @staticmethod
    def _props(cnt: np.ndarray) -> Dict:
        area = float(cv2.contourArea(cnt))
        peri = float(cv2.arcLength(cnt, True))
        x, y, w, h = cv2.boundingRect(cnt)
        m = cv2.moments(cnt)
        if m["m00"] != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        circ = (4 * math.pi * area / (peri ** 2)) if peri > 0 else 0.0
        ar = (float(w) / h) if h > 0 else 0.0
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        sol = (area / hull_area) if hull_area > 0 else 0.0

        return {
            "area": round(area, 1),
            "perimeter": round(peri, 1),
            "bbox": (int(x), int(y), int(w), int(h)),
            "centroid": (int(cx), int(cy)),
            "circularity": round(circ, 3),
            "aspect_ratio": round(ar, 3),
            "solidity": round(sol, 3),
        }

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    @staticmethod
    def _draw(
        img: np.ndarray,
        contours: List[np.ndarray],
        props: List[Dict],
        contour_color: Tuple[int, int, int],
        bbox_color: Tuple[int, int, int],
        centroid_color: Tuple[int, int, int],
        thickness: int,
        draw_bbox: bool,
        draw_centroid: bool,
        draw_labels: bool,
    ) -> np.ndarray:
        if img.ndim == 2:
            out = cv2.cvtColor(ensure_uint8(img), cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            out = cv2.cvtColor(ensure_uint8(img), cv2.COLOR_BGRA2BGR)
        else:
            out = ensure_uint8(img).copy()

        cv2.drawContours(out, contours, -1, contour_color, thickness)

        for i, p in enumerate(props, start=1):
            if draw_bbox:
                x, y, w, h = p["bbox"]
                cv2.rectangle(out, (x, y), (x + w, y + h), bbox_color, thickness)
            if draw_centroid:
                cx, cy = p["centroid"]
                cv2.circle(out, (cx, cy), 5, centroid_color, -1)
            if draw_labels:
                cx, cy = p["centroid"]
                cv2.putText(out, f"#{i}", (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(out, f"#{i}", (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return out

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, params: DetectParams, pre: PreprocessParams) -> Dict:
        if self.original is None:
            raise ValueError("No image loaded.")

        combined_key = hash((params, pre))
        cached = self.cache.get(self.current_frame_idx)
        if cached and cached.get("param_key") == combined_key:
            return cached

        gray_pre = self.get_preprocessed_gray(pre)

        k = int(params.blur_ksize)
        if k % 2 == 0:
            k += 1
        k = max(1, k)
        blurred = cv2.GaussianBlur(gray_pre, (k, k), 0) if k > 1 else gray_pre

        method = params.method
        if method == "Otsu Threshold":
            _, binary = cv2.threshold(blurred, params.thresh_value, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "Binary Threshold":
            _, binary = cv2.threshold(blurred, params.thresh_value, 255,
                                      cv2.THRESH_BINARY)
        elif method == "Inv. Binary Threshold":
            _, binary = cv2.threshold(blurred, params.thresh_value, 255,
                                      cv2.THRESH_BINARY_INV)
        elif method == "Adaptive Mean":
            binary = cv2.adaptiveThreshold(blurred, 255,
                                           cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        elif method == "Adaptive Gaussian":
            binary = cv2.adaptiveThreshold(blurred, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        elif method == "Canny Edge":
            edges = cv2.Canny(blurred, params.canny_low, params.canny_high)
            binary = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        else:
            raise ValueError(f"Unknown method: {method}")

        if params.use_morphology and params.morph_ksize > 0:
            mk = int(params.morph_ksize)
            if mk % 2 == 0:
                mk += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        filtered: List[np.ndarray] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < params.min_area:
                continue
            if params.max_area is not None and area > params.max_area:
                continue
            filtered.append(cnt)

        props = [self._props(c) for c in filtered]

        gray_bgr = (cv2.cvtColor(gray_pre, cv2.COLOR_GRAY2BGR)
                     if gray_pre.ndim == 2 else gray_pre)
        annotated = self._draw(
            gray_bgr, filtered, props,
            params.contour_color, params.bbox_color, params.centroid_color,
            params.thickness, params.draw_bbox, params.draw_centroid,
            params.draw_labels,
        )

        res = {
            "param_key": combined_key,
            "binary": binary,
            "contours": filtered,
            "properties": props,
            "annotated": annotated,
            "pre_key": hash(pre),
        }
        self.cache[self.current_frame_idx] = res
        return res
