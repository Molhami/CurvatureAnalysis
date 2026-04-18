"""Contour detection engine with preprocessing and caching."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

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
        self.rotation_angle: float = 0.0

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
        self.rotation_angle = 0.0
        self.pre_cache.clear()
        self.cache.clear()
        self.set_frame(0)

    def load_paths(self, paths: Sequence[str]):
        if not paths:
            raise ValueError("No images selected.")
        frames: List[np.ndarray] = []
        used: List[str] = []
        for p in paths:
            cur = load_frames(str(p))
            if not cur:
                continue
            frames.extend(cur)
            used.append(str(p))
        if not frames:
            raise ValueError("No frames found.")
        if len(used) == 1:
            self.image_path = used[0]
        else:
            self.image_path = f"{used[0]} (+{len(used) - 1} files)"
        self.frames = frames
        self.rotation_angle = 0.0
        self.pre_cache.clear()
        self.cache.clear()
        self.set_frame(0)

    def set_frame(self, idx: int):
        if not self.frames:
            return
        idx = max(0, min(idx, self.num_frames - 1))
        self.current_frame_idx = idx
        raw_frame = self.frames[idx]
        if abs(self.rotation_angle) > 1e-7:
            self.original = self._rotate_image_any(raw_frame, self.rotation_angle)
        else:
            self.original = raw_frame

    def rotate_stack(self, quarter_turns: int):
        """Rotate all frames by 90-degree steps (positive = CCW, negative = CW)."""
        if not self.frames:
            return
        k = int(quarter_turns) % 4
        if k == 0:
            return
        self.frames = [np.ascontiguousarray(np.rot90(f, k=k)) for f in self.frames]
        self.pre_cache.clear()
        self.cache.clear()
        self.set_frame(self.current_frame_idx)

    @staticmethod
    def _rotate_image_any(img: np.ndarray, angle_deg: float) -> np.ndarray:
        h, w = img.shape[:2]
        cx, cy = w * 0.5, h * 0.5
        m = cv2.getRotationMatrix2D((cx, cy), float(angle_deg), 1.0)

        c = abs(m[0, 0])
        s = abs(m[0, 1])
        new_w = max(1, int(round(h * s + w * c)))
        new_h = max(1, int(round(h * c + w * s)))

        m[0, 2] += (new_w * 0.5) - cx
        m[1, 2] += (new_h * 0.5) - cy

        if img.ndim == 2:
            border = 0
        elif img.shape[2] == 4:
            border = (0, 0, 0, 0)
        else:
            border = (0, 0, 0)

        out = cv2.warpAffine(
            img,
            m,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )
        return np.ascontiguousarray(out)

    def rotate_stack_by_angle(self, angle_deg: float):
        if not self.frames:
            return
        a = float(angle_deg)
        if abs(a) < 1e-7:
            return
        self.rotation_angle += a
        self.rotation_angle = self.rotation_angle % 360.0
        if self.rotation_angle > 180.0:
            self.rotation_angle -= 360.0
        self.pre_cache.clear()
        self.cache.clear()
        self.set_frame(self.current_frame_idx)

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
        is_closed = ContourDetector._is_effectively_closed(cnt)
        area = float(cv2.contourArea(cnt)) if is_closed else 0.0
        peri = float(cv2.arcLength(cnt, is_closed))
        x, y, w, h = cv2.boundingRect(cnt)
        m = cv2.moments(cnt)
        if m["m00"] != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        circ = (4 * math.pi * area / (peri ** 2)) if (peri > 0 and is_closed) else 0.0
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

    @staticmethod
    def _is_effectively_closed(cnt: np.ndarray, tol_px: float = 2.5) -> bool:
        pts = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts) < 3:
            return False
        return float(np.linalg.norm(pts[0] - pts[-1])) <= float(tol_px)

    @staticmethod
    def _is_straight_line_contour(cnt: np.ndarray, tol: float) -> bool:
        """Detect near-linear contours using distance to best-fit line."""
        pts = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts) < 8:
            return False

        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = float(vx[0]), float(vy[0])
        x0, y0 = float(x0[0]), float(y0[0])
        norm = math.hypot(vx, vy)
        if norm < 1e-8:
            return True
        vx /= norm
        vy /= norm

        rel = pts - np.array([x0, y0], dtype=np.float32)
        proj = rel[:, 0] * vx + rel[:, 1] * vy
        span = float(np.max(proj) - np.min(proj))
        if span < 8.0:
            return False

        # 2D cross product magnitude against unit direction = perpendicular distance.
        perp = np.abs(rel[:, 0] * vy - rel[:, 1] * vx)
        mean_ratio = float(np.mean(perp) / span)
        max_ratio = float(np.max(perp) / span)

        cov = np.cov(pts.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        linearity = float(eigvals[0] / max(eigvals[1], 1e-8))

        seg = np.diff(pts, axis=0)
        path_len = float(np.sum(np.linalg.norm(seg, axis=1)))
        end_dist = float(np.linalg.norm(pts[-1] - pts[0])) + 1e-8
        tortuosity = path_len / end_dist

        _, (rw, rh), _ = cv2.minAreaRect(pts.reshape(-1, 1, 2))
        thinness = float(min(rw, rh))
        elongation = float(max(rw, rh))

        tol = max(1e-4, float(tol))
        return (
            (
                mean_ratio < (2.5 * tol)
                and max_ratio < (6.0 * tol)
                and linearity < 0.015
                and tortuosity < 1.15
            )
            or (elongation > 20.0 and thinness < 2.5)
        )

    @staticmethod
    def _split_contour_on_roi_border(
        cnt: np.ndarray, roi: Tuple[int, int, int, int]
    ) -> List[np.ndarray]:
        """Split a contour into open segments, dropping points on ROI border."""
        rx, ry, rw, rh = roi
        pts = cnt.reshape(-1, 2)
        if len(pts) < 2:
            return []

        x0, y0 = int(rx), int(ry)
        x1, y1 = int(rx + rw - 1), int(ry + rh - 1)
        on_border = (
            (pts[:, 0] <= x0) | (pts[:, 0] >= x1) |
            (pts[:, 1] <= y0) | (pts[:, 1] >= y1)
        )
        keep = ~on_border
        if int(np.count_nonzero(keep)) < 2:
            return []
        if not np.any(on_border):
            return [cnt.astype(np.int32)]

        n = len(pts)
        first_border = int(np.where(on_border)[0][0])
        pts_r = np.concatenate((pts[first_border + 1:], pts[:first_border + 1]), axis=0)
        keep_r = np.concatenate((keep[first_border + 1:], keep[:first_border + 1]), axis=0)

        out: List[np.ndarray] = []
        i = 0
        while i < n:
            if not bool(keep_r[i]):
                i += 1
                continue
            j = i
            while j < n and bool(keep_r[j]):
                j += 1
            seg = pts_r[i:j]
            if len(seg) >= 2:
                out.append(seg.reshape(-1, 1, 2).astype(np.int32))
            i = j
        return out

    @staticmethod
    def _extract_contours(
        binary: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        if roi is None:
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            return contours, binary

        rx, ry, rw, rh = roi
        view_binary = np.zeros_like(binary)
        roi_bin = binary[ry:ry + rh, rx:rx + rw].copy()
        view_binary[ry:ry + rh, rx:rx + rw] = roi_bin

        local_contours, _ = cv2.findContours(roi_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        out: List[np.ndarray] = []
        for cnt in local_contours:
            if cnt.size == 0:
                continue
            cnt_g = cnt.copy()
            cnt_g[:, :, 0] += rx
            cnt_g[:, :, 1] += ry
            out.extend(ContourDetector._split_contour_on_roi_border(cnt_g, roi))
        return out, view_binary

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

        for cnt in contours:
            pts = cnt.reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=False, color=contour_color,
                          thickness=thickness, lineType=cv2.LINE_AA)

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

    def detect(self, params: DetectParams, pre: PreprocessParams,
               roi: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        if self.original is None:
            raise ValueError("No image loaded.")

        combined_key = hash((params, pre, roi))
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

        roi_clamped: Optional[Tuple[int, int, int, int]] = None
        if roi is not None:
            rx, ry, rw, rh = roi
            ih, iw = binary.shape[:2]
            rx, ry = max(0, rx), max(0, ry)
            rw = max(1, min(rw, iw - rx))
            rh = max(1, min(rh, ih - ry))
            roi_clamped = (int(rx), int(ry), int(rw), int(rh))

        contours, binary_view = self._extract_contours(binary, roi_clamped)

        filtered: List[np.ndarray] = []
        for cnt in contours:
            is_closed = self._is_effectively_closed(cnt)
            area = float(cv2.contourArea(cnt)) if is_closed else 0.0
            length = float(cv2.arcLength(cnt, is_closed))

            if is_closed:
                if area < params.min_area:
                    continue
                if params.max_area is not None and area > params.max_area:
                    continue
            else:
                min_open_len = max(8.0, 0.25 * float(params.min_area))
                if length < min_open_len:
                    continue

            if params.min_rect_width > 0 and is_closed and len(cnt) >= 5:
                _, (rw, rh), _ = cv2.minAreaRect(cnt)
                if min(rw, rh) < params.min_rect_width:
                    continue
            if (
                params.reject_straight_lines
                and self._is_straight_line_contour(cnt, params.straight_line_tol)
            ):
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
            "binary": binary_view,
            "contours": filtered,
            "properties": props,
            "annotated": annotated,
            "pre_key": hash(pre),
        }
        self.cache[self.current_frame_idx] = res
        return res
