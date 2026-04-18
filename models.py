"""Data-class parameter definitions for preprocessing and detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PreprocessParams:
    """Parameters controlling image preprocessing (applied before detection)."""
    invert: bool
    alpha: float          # contrast multiplier
    beta: int             # brightness offset
    blur_ksize: int       # Gaussian blur kernel size (odd, >= 1)


@dataclass(frozen=True)
class DetectParams:
    """Parameters controlling contour detection and drawing."""
    method: str
    blur_ksize: int       # extra blur in detector (odd, >= 1)
    thresh_value: int
    canny_low: int
    canny_high: int
    use_morphology: bool
    morph_ksize: int
    min_area: int
    max_area: Optional[int]
    min_rect_width: int    # filter contours whose thinnest dimension < this (0 = off)
    contour_color: Tuple[int, int, int]
    bbox_color: Tuple[int, int, int]
    centroid_color: Tuple[int, int, int]
    thickness: int
    draw_bbox: bool
    draw_centroid: bool
    draw_labels: bool
    reject_straight_lines: bool = True
    straight_line_tol: float = 0.01  # normalized line-fit deviation (smaller = stricter)
