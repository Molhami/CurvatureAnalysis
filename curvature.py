"""Curvature computation engine via parametric spline fitting."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from scipy.interpolate import splprep, splev

from .utils import dedupe_xy


class CurvatureEngine:
    """Computes curvature, radius of curvature, tangents and normals
    along a set of 2-D points using B-spline interpolation."""

    def __init__(self):
        self.points_x: List[float] = []
        self.points_y: List[float] = []
        self.curve_data: Optional[Dict[str, np.ndarray]] = None

    def clear(self):
        self.points_x = []
        self.points_y = []
        self.curve_data = None

    def set_points(self, xs: List[float], ys: List[float]):
        xs, ys = dedupe_xy(xs, ys)
        self.points_x = xs
        self.points_y = ys
        self.curve_data = None

    def compute(
        self,
        smooth: float,
        res_mult: float,
        max_points: int = 2500,
    ) -> Dict[str, np.ndarray]:
        if len(self.points_x) < 6:
            raise ValueError("Not enough points.")

        xs, ys = dedupe_xy(self.points_x, self.points_y)

        closed = 0
        if len(xs) > 5 and math.hypot(xs[0] - xs[-1], ys[0] - ys[-1]) < 5.0:
            closed = 1

        tck, _u = splprep([xs, ys], s=float(smooth), per=closed)
        n = min(int(len(xs) * float(res_mult)), max_points)
        n = max(n, 50)
        u_new = np.linspace(0, 1, num=n)

        x_fit, y_fit = splev(u_new, tck)
        dx, dy = splev(u_new, tck, der=1)
        ddx, ddy = splev(u_new, tck, der=2)

        denom = np.power(dx * dx + dy * dy, 1.5)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.abs(dx * ddy - dy * ddx) / denom
            r = 1.0 / k

        norm = np.sqrt(dx * dx + dy * dy)
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = dx / norm
            ty = dy / norm
            nx = -dy / norm
            ny = dx / norm

        for arr in (k, r, tx, ty, nx, ny):
            arr[np.isnan(arr)] = 0.0
            arr[np.isinf(arr)] = 0.0

        self.curve_data = {
            "x": np.asarray(x_fit),
            "y": np.asarray(y_fit),
            "k": np.asarray(k),
            "r": np.asarray(r),
            "tx": np.asarray(tx),
            "ty": np.asarray(ty),
            "nx": np.asarray(nx),
            "ny": np.asarray(ny),
        }
        return self.curve_data
