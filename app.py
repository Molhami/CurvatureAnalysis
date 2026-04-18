"""Main application window - PyQt5 version, no Tkinter."""
from __future__ import annotations

import copy
import csv
import math
import re
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QScrollArea, QGroupBox, QLabel, QPushButton,
    QSlider, QCheckBox, QComboBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QProgressDialog,
    QMessageBox, QFileDialog, QAction, QFrame, QSizePolicy, QToolBar,
    QSpacerItem, QSpinBox,
)
from PyQt5.QtCore import Qt, QPoint, QPointF, QSize, pyqtSignal, QRect, QRectF, QTimer
from PyQt5.QtGui import (
    QPainter, QPixmap, QImage, QColor, QPen, QBrush, QFont,
    QPainterPath, QTransform,
)

from .detector import ContourDetector
from .curvature import CurvatureEngine
from .models import DetectParams, PreprocessParams

# -- Colour palette (Catppuccin Mocha-inspired) -------------------------------
_BG      = "#1e1e2e"
_BG_ALT  = "#181825"
_PANEL   = "#313244"
_WIDGET  = "#45475a"
_ACCENT  = "#89b4fa"
_ACCENT2 = "#7287fd"
_GREEN   = "#a6e3a1"
_GREEN2  = "#40a02b"
_TEXT    = "#cdd6f4"
_SUBTEXT = "#a6adc8"
_BORDER  = "#45475a"
_RED     = "#f38ba8"
_YELLOW  = "#f9e2af"

DARK_QSS = f"""
QMainWindow, QDialog {{ background: {_BG}; color: {_TEXT}; }}
QWidget {{ background: {_BG}; color: {_TEXT};
          font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; }}
QGroupBox {{ border: 1px solid {_BORDER}; border-radius: 6px;
            margin-top: 8px; padding: 8px 6px 6px 6px;
            font-weight: bold; color: {_ACCENT}; background: {_BG}; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left;
                    left: 10px; padding: 0 4px; color: {_ACCENT}; }}
QPushButton {{ background: {_PANEL}; color: {_TEXT}; border: 1px solid {_BORDER};
               border-radius: 4px; padding: 5px 12px; min-height: 24px; }}
QPushButton:hover {{ background: {_WIDGET}; border-color: {_ACCENT}; }}
QPushButton:pressed {{ background: {_ACCENT}; color: {_BG}; }}
QPushButton#accentBtn {{ background: {_ACCENT}; color: {_BG};
                         font-weight: bold; border: none; }}
QPushButton#accentBtn:hover {{ background: {_ACCENT2}; color: {_TEXT}; }}
QPushButton#greenBtn {{ background: {_GREEN}; color: {_BG};
                        font-weight: bold; border: none; }}
QPushButton#greenBtn:hover {{ background: {_GREEN2}; color: {_TEXT}; }}
QSlider::groove:horizontal {{ height: 4px; background: {_BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {_ACCENT}; border: none;
                               width: 14px; height: 14px;
                               margin: -5px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {_ACCENT}; border-radius: 2px; }}
QComboBox {{ background: {_PANEL}; color: {_TEXT}; border: 1px solid {_BORDER};
             border-radius: 4px; padding: 4px 8px; min-height: 24px; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{ background: {_PANEL}; color: {_TEXT};
    selection-background-color: {_ACCENT}; selection-color: {_BG};
    border: 1px solid {_BORDER}; }}
QCheckBox, QRadioButton {{ color: {_TEXT}; spacing: 6px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {_BORDER};
                        border-radius: 3px; background: {_PANEL}; }}
QCheckBox::indicator:checked {{ background: {_ACCENT}; border-color: {_ACCENT}; }}
QRadioButton::indicator {{ width: 16px; height: 16px; border: 1px solid {_BORDER};
                           border-radius: 8px; background: {_PANEL}; }}
QRadioButton::indicator:checked {{ background: {_ACCENT}; border-color: {_ACCENT}; }}
QTableWidget {{ background: {_BG}; alternate-background-color: #252536;
                color: {_TEXT}; gridline-color: {_PANEL};
                border: 1px solid {_BORDER}; }}
QTableWidget::item:selected {{ background: {_ACCENT}; color: {_BG}; }}
QHeaderView::section {{ background: {_PANEL}; color: {_ACCENT}; padding: 5px;
                        border: 1px solid {_BORDER}; font-weight: bold; }}
QTabWidget::pane {{ border: 1px solid {_BORDER}; border-radius: 4px; background: {_BG}; }}
QTabBar::tab {{ background: {_PANEL}; color: {_SUBTEXT}; padding: 8px 14px;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                margin-right: 2px; border: 1px solid {_BORDER}; border-bottom: none; }}
QTabBar::tab:selected {{ background: {_ACCENT}; color: {_BG}; font-weight: bold; }}
QTabBar::tab:hover:!selected {{ background: {_WIDGET}; color: {_TEXT}; }}
QScrollBar:vertical {{ background: {_BG}; width: 10px; border-radius: 5px; }}
QScrollBar::handle:vertical {{ background: {_WIDGET}; border-radius: 5px; min-height: 20px; }}
QScrollBar::handle:vertical:hover {{ background: {_ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: {_BG}; height: 10px; border-radius: 5px; }}
QScrollBar::handle:horizontal {{ background: {_WIDGET}; border-radius: 5px; min-width: 20px; }}
QScrollBar::handle:horizontal:hover {{ background: {_ACCENT}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QStatusBar {{ background: {_BG_ALT}; color: {_SUBTEXT}; border-top: 1px solid {_PANEL}; }}
QMenuBar {{ background: {_BG_ALT}; color: {_TEXT}; border-bottom: 1px solid {_PANEL}; }}
QMenuBar::item:selected {{ background: {_PANEL}; }}
QMenu {{ background: {_PANEL}; color: {_TEXT}; border: 1px solid {_BORDER}; }}
QMenu::item:selected {{ background: {_ACCENT}; color: {_BG}; }}
QMenu::separator {{ height: 1px; background: {_BORDER}; margin: 2px 4px; }}
QLineEdit {{ background: {_PANEL}; color: {_TEXT}; border: 1px solid {_BORDER};
             border-radius: 4px; padding: 4px 8px; min-height: 24px; }}
QLineEdit:focus {{ border-color: {_ACCENT}; }}
QSplitter::handle {{ background: {_BORDER}; }}
QSplitter::handle:horizontal {{ width: 3px; }}
QSplitter::handle:vertical {{ height: 3px; }}
QToolBar {{ background: {_BG_ALT}; border-bottom: 1px solid {_PANEL};
            spacing: 4px; padding: 4px; }}
QProgressBar {{ background: {_PANEL}; border: 1px solid {_BORDER}; border-radius: 4px;
                color: {_BG}; text-align: center; }}
QProgressBar::chunk {{ background: {_ACCENT}; border-radius: 3px; }}
"""


# -- Utility ------------------------------------------------------------------

def _cv_to_qpixmap(img: np.ndarray) -> QPixmap:
    """Convert an OpenCV BGR/gray/BGRA numpy array to a QPixmap."""
    from .utils import ensure_uint8
    img8 = ensure_uint8(img)
    if img8.ndim == 2:
        rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
    elif img8.shape[2] == 4:
        rgb = cv2.cvtColor(img8, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(img8, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# -- LabeledSlider -------------------------------------------------------------

class LabeledSlider(QWidget):
    """Slider with a label and a live value readout."""

    value_changed = pyqtSignal()

    def __init__(
        self,
        label: str,
        min_val,
        max_val,
        default,
        is_float: bool = False,
        float_scale: int = 10,
        parent=None,
    ):
        super().__init__(parent)
        self._is_float = is_float
        self._scale = float_scale if is_float else 1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {_SUBTEXT};")
        self._val_lbl = QLabel()
        self._val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._val_lbl.setFixedWidth(52)
        self._val_lbl.setStyleSheet(f"color: {_ACCENT}; font-weight: bold;")
        top.addWidget(lbl)
        top.addWidget(self._val_lbl)
        layout.addLayout(top)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(
            int(min_val * self._scale), int(max_val * self._scale)
        )
        self.slider.setValue(int(default * self._scale))
        self.slider.valueChanged.connect(self._on_change)
        layout.addWidget(self.slider)

        self._update_label(self.slider.value())

    def _update_label(self, int_val: int):
        if self._is_float:
            self._val_lbl.setText(f"{int_val / self._scale:.1f}")
        else:
            self._val_lbl.setText(str(int_val))

    def _on_change(self, int_val: int):
        self._update_label(int_val)
        self.value_changed.emit()

    def get_value(self):
        v = self.slider.value()
        return v / self._scale if self._is_float else v

    def set_value(self, val):
        self.slider.blockSignals(True)
        self.slider.setValue(int(val * self._scale))
        self._update_label(self.slider.value())
        self.slider.blockSignals(False)


# -- HistogramWidget -----------------------------------------------------------

class HistogramWidget(QWidget):
    """Custom-painted pixel histogram."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: Optional[Dict[str, np.ndarray]] = None
        self._range_max: float = 256.0
        self._log_scale: bool = False
        self.setMinimumWidth(250)
        self.setMinimumHeight(150)
        self.setAttribute(Qt.WA_OpaquePaintEvent)

    def set_data(
        self,
        data: Optional[Dict[str, np.ndarray]],
        range_max: float = 256.0,
    ):
        self._data = data
        self._range_max = float(range_max)
        self.update()

    def set_log(self, flag: bool):
        self._log_scale = flag
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(_BG))

        if not self._data:
            painter.setPen(QColor(_SUBTEXT))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image")
            return

        w = self.width()
        h = self.height()
        pl, pb, pr, pt = 46, 32, 12, 10
        aw = w - pl - pr
        ah = h - pt - pb
        if aw <= 0 or ah <= 0:
            return

        is_log = self._log_scale
        max_v = max(float(np.max(d)) for d in self._data.values())
        if max_v == 0:
            max_v = 1.0
        max_disp = math.log10(max_v + 1) if is_log else max_v

        # Axes
        ax_pen = QPen(QColor("#444455"), 1)
        painter.setPen(ax_pen)
        painter.drawLine(pl, h - pb, w - pr, h - pb)
        painter.drawLine(pl, pt, pl, h - pb)

        small_font = QFont("Segoe UI", 8)
        painter.setFont(small_font)
        painter.setPen(QColor(_SUBTEXT))

        # X-axis ticks
        for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = pl + int(ratio * aw)
            painter.drawLine(x, h - pb, x, h - pb + 4)
            val = int(ratio * (self._range_max - 1))
            lbl = f"{val//1000}k" if val >= 1000 else str(val)
            painter.drawText(x - 16, h - pb + 6, 32, 14, Qt.AlignCenter, lbl)

        painter.drawText(pl, h - 4, aw, 10, Qt.AlignCenter, "Pixel Intensity")

        # Y-axis ticks
        for ratio in (0.0, 0.5, 1.0):
            y = h - pb - int(ratio * ah)
            painter.drawLine(pl - 4, y, pl, y)
            if is_log:
                val = int(10 ** (ratio * max_disp) - 1)
            else:
                val = int(ratio * max_v)
            lbl = f"{val//1000}k" if val >= 1000 else str(val)
            painter.drawText(pl - 44, y - 7, 40, 14, Qt.AlignRight | Qt.AlignVCenter, lbl)

        # Channel colours (BGR -> display: R G B or gray)
        ch_colors = {
            "gray": QColor("#cccccc"),
            "b": QColor("#5B9BD5"),
            "g": QColor("#70AD47"),
            "r": QColor("#FF6B6B"),
        }
        order = ["gray"] if "gray" in self._data else ["r", "g", "b"]

        info_y = pt
        peak_font = QFont("Segoe UI", 8)
        peak_font.setBold(True)
        painter.setFont(peak_font)
        painter.setPen(QColor(_SUBTEXT))
        painter.drawText(w - pr - 140, info_y, 140, 12, Qt.AlignRight, "Peak intensity:")
        info_y += 14

        for key in order:
            if key not in self._data:
                continue
            h_data = self._data[key]
            color = ch_colors.get(key, QColor("white"))
            pen = QPen(color, 1.5)
            pen.setCosmetic(True)
            painter.setPen(pen)

            path = QPainterPath()
            first = True
            for i in range(256):
                x = pl + int(i * aw / 255.0)
                v = float(h_data[i])
                v_d = math.log10(v + 1) if is_log else v
                y = h - pb - int((v_d / max_disp) * ah)
                y = max(pt, min(h - pb, y))
                if first:
                    path.moveTo(x, y)
                    first = False
                else:
                    path.lineTo(x, y)
            painter.drawPath(path)

            peak_bin = int(np.argmax(h_data))
            peak_intensity = int(peak_bin * (self._range_max / 256.0))
            peak_count = int(h_data[peak_bin])
            pi_s = f"{peak_intensity//1000}k" if (self._range_max > 256 and peak_intensity >= 1000) else str(peak_intensity)
            pc_s = f"{peak_count//1000}k" if peak_count >= 1000 else str(peak_count)
            ch_name = key.upper() if key != "gray" else "Gray"
            painter.setPen(color)
            painter.setFont(small_font)
            painter.drawText(w - pr - 140, info_y, 140, 12,
                             Qt.AlignRight, f"{ch_name}: {pi_s} ({pc_s} px)")
            info_y += 12

        painter.end()


# -- ImageCanvas --------------------------------------------------------------

class ImageCanvas(QWidget):
    """Zoomable/pannable image canvas with overlay drawing."""

    zoom_changed = pyqtSignal(float)
    points_captured = pyqtSignal(list, list)   # xs, ys
    status_message = pyqtSignal(str)
    roi_changed = pyqtSignal(object)           # QRect in image coords, or None
    align_line_drawn = pyqtSignal(object)      # (x1, y1, x2, y2) in image coords, or None
    _ROI_HANDLE_RADIUS_PX = 6
    _ROI_MIN_SIZE_PX = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent)

        # Viewport
        self.zoom: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self._panning = False
        self._last_mouse = QPoint()

        # Image cache
        self._cv_img: Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap] = None

        # Overlays
        self.contours: List[np.ndarray] = []
        self.selected_contour_idx: Optional[int] = None
        self.points_x: List[float] = []
        self.points_y: List[float] = []
        self.curve_data: Optional[Dict] = None
        self.show_vectors: bool = False
        self.vec_density: int = 15
        self.vec_len: float = 50.0

        # Drawing
        self.draw_enabled: bool = False
        self.draw_mode: str = "freehand"
        self._is_drawing: bool = False
        self._circle_start: Tuple[float, float] = (0.0, 0.0)
        self._circle_r: float = 0.0

        # Hover
        self._hover_data: Optional[Tuple] = None  # (sx, sy, k, r, nx, ny, tx, ty)

        # ROI
        self.roi_rect: Optional[QRect] = None
        self.roi_mode: bool = False
        self._roi_start: Optional[Tuple[float, float]] = None
        self._roi_temp: Optional[QRect] = None   # in image coords while dragging
        self._roi_drag_mode: Optional[str] = None
        self._roi_drag_origin: Optional[Tuple[float, float]] = None
        self._roi_drag_start_rect: Optional[QRect] = None
        self.align_mode: bool = False
        self._align_start: Optional[Tuple[float, float]] = None
        self._align_end: Optional[Tuple[float, float]] = None

        # Pixel info
        self._hover_img_pos: Optional[Tuple[int, int]] = None

        self.setStyleSheet(f"background: #1a1a2e;")
        self.setMinimumSize(300, 200)

    # -- Coordinate helpers ------------------------------------------------

    def to_screen(self, wx: float, wy: float) -> Tuple[float, float]:
        return wx * self.zoom + self.pan_x, wy * self.zoom + self.pan_y

    def to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        return (sx - self.pan_x) / self.zoom, (sy - self.pan_y) / self.zoom

    # -- Image update -------------------------------------------------------

    def set_image(self, cv_img: Optional[np.ndarray]):
        self._cv_img = cv_img
        self._pixmap = _cv_to_qpixmap(cv_img) if cv_img is not None else None
        self.update()

    def fit_to_view(self):
        if self._cv_img is None:
            return
        ih, iw = self._cv_img.shape[:2]
        if iw < 1 or ih < 1:
            return
        cw, ch = self.width(), self.height()
        if cw < 2 or ch < 2:
            return
        self.zoom = float(np.clip(min(cw / iw, ch / ih), 0.05, 30.0))
        self.pan_x = (cw - iw * self.zoom) / 2.0
        self.pan_y = (ch - ih * self.zoom) / 2.0
        self.zoom_changed.emit(self.zoom)
        self.update()

    # -- Paint --------------------------------------------------------------

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1a1a2e"))

        if self._pixmap is None:
            painter.setPen(QColor("#444466"))
            painter.setFont(QFont("Segoe UI", 16))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Open an image  (Ctrl+O)")
            return

        # Draw the base image via transform
        transform = QTransform()
        transform.translate(self.pan_x, self.pan_y)
        transform.scale(self.zoom, self.zoom)
        painter.setTransform(transform)

        if self._pixmap:
            painter.drawPixmap(0, 0, self._pixmap)

        # Back to identity for screen-space overlays
        painter.resetTransform()

        self._paint_selected_contour(painter)
        self._paint_roi(painter)
        self._paint_align_line(painter)
        self._paint_points(painter)
        self._paint_curve(painter)
        if self.show_vectors and self.curve_data:
            self._paint_vectors(painter)
        if self._is_drawing and self.draw_mode == "circle" and self._circle_r > 1e-6:
            self._paint_circle_preview(painter)
        if self._hover_data:
            self._paint_hover(painter)
        self._paint_pixel_info(painter)

        painter.end()

    def _paint_selected_contour(self, painter: QPainter):
        if self.selected_contour_idx is None or not self.contours:
            return
        idx = self.selected_contour_idx
        if idx < 0 or idx >= len(self.contours):
            return
        pts = self.contours[idx].reshape(-1, 2).astype(float)
        pen = QPen(QColor("#ff9f43"), 2)
        painter.setPen(pen)
        path = QPainterPath()
        sx, sy = self.to_screen(float(pts[0, 0]), float(pts[0, 1]))
        path.moveTo(sx, sy)
        for x, y in pts[1:]:
            sx, sy = self.to_screen(float(x), float(y))
            path.lineTo(sx, sy)
        painter.drawPath(path)

    def _paint_roi(self, painter: QPainter):
        rect_src = self._roi_temp if self._roi_temp is not None else self.roi_rect
        if rect_src is None:
            return
        screen_rect = self._roi_to_screen_rect(rect_src)
        painter.fillRect(screen_rect, QColor(253, 127, 40, 38))   # orange, 15% opacity
        pen = QPen(QColor("#fd7f28"), 2, Qt.DashLine)
        pen.setDashPattern([6, 3])
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(screen_rect)
        if self._roi_temp is None:
            handle_r = float(self._ROI_HANDLE_RADIUS_PX)
            painter.setPen(QPen(QColor("#1a1a2e"), 1))
            painter.setBrush(QBrush(QColor("#fd7f28")))
            for pt in self._roi_handle_points_screen(rect_src).values():
                painter.drawRect(QRectF(pt.x() - handle_r, pt.y() - handle_r,
                                        2 * handle_r, 2 * handle_r))

    def _paint_align_line(self, painter: QPainter):
        if self._align_start is None or self._align_end is None:
            return
        x1, y1 = self._align_start
        x2, y2 = self._align_end
        sx1, sy1 = self.to_screen(float(x1), float(y1))
        sx2, sy2 = self.to_screen(float(x2), float(y2))
        pen = QPen(QColor("#89b4fa"), 2, Qt.DashLine)
        pen.setDashPattern([5, 3])
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))

    def _roi_to_screen_rect(self, rect: QRect) -> QRectF:
        sx1, sy1 = self.to_screen(float(rect.x()), float(rect.y()))
        sx2, sy2 = self.to_screen(float(rect.x() + rect.width()),
                                   float(rect.y() + rect.height()))
        left, right = sorted((sx1, sx2))
        top, bottom = sorted((sy1, sy2))
        return QRectF(left, top, right - left, bottom - top)

    def _roi_handle_points_screen(self, rect: QRect) -> Dict[str, QPointF]:
        sr = self._roi_to_screen_rect(rect)
        x1, y1 = sr.left(), sr.top()
        x2, y2 = sr.right(), sr.bottom()
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        return {
            "tl": QPointF(x1, y1),
            "t": QPointF(cx, y1),
            "tr": QPointF(x2, y1),
            "r": QPointF(x2, cy),
            "br": QPointF(x2, y2),
            "b": QPointF(cx, y2),
            "bl": QPointF(x1, y2),
            "l": QPointF(x1, cy),
        }

    def _roi_hit_test(self, sx: float, sy: float) -> Optional[str]:
        if self.roi_rect is None:
            return None
        hit_pad = float(self._ROI_HANDLE_RADIUS_PX) + 2.0
        handles = self._roi_handle_points_screen(self.roi_rect)
        for key in ("tl", "tr", "br", "bl", "l", "r", "t", "b"):
            p = handles[key]
            if abs(sx - p.x()) <= hit_pad and abs(sy - p.y()) <= hit_pad:
                return key
        if self._roi_to_screen_rect(self.roi_rect).contains(QPointF(sx, sy)):
            return "move"
        return None

    def _cursor_for_roi_mode(self, mode: Optional[str]):
        if mode in ("l", "r"):
            return Qt.SizeHorCursor
        if mode in ("t", "b"):
            return Qt.SizeVerCursor
        if mode in ("tl", "br"):
            return Qt.SizeFDiagCursor
        if mode in ("tr", "bl"):
            return Qt.SizeBDiagCursor
        if mode == "move":
            return Qt.SizeAllCursor
        return Qt.CrossCursor

    def _clamp_roi_rect(self, rect: QRect) -> QRect:
        if self._cv_img is None:
            return rect
        ih, iw = self._cv_img.shape[:2]
        if iw < 1 or ih < 1:
            return rect

        x1 = int(np.clip(rect.x(), 0, max(0, iw - 1)))
        y1 = int(np.clip(rect.y(), 0, max(0, ih - 1)))
        x2 = int(np.clip(rect.x() + rect.width(), x1 + 1, iw))
        y2 = int(np.clip(rect.y() + rect.height(), y1 + 1, ih))
        return QRect(x1, y1, x2 - x1, y2 - y1)

    def _clamp_roi_move(self, x: int, y: int, w: int, h: int) -> QRect:
        if self._cv_img is None:
            return QRect(x, y, w, h)
        ih, iw = self._cv_img.shape[:2]
        if iw < 1 or ih < 1:
            return QRect(x, y, w, h)
        w = max(1, min(int(w), iw))
        h = max(1, min(int(h), ih))
        x = int(np.clip(x, 0, max(0, iw - w)))
        y = int(np.clip(y, 0, max(0, ih - h)))
        return QRect(x, y, w, h)

    def _roi_drag_rect(self, wx: float, wy: float) -> Optional[QRect]:
        mode = self._roi_drag_mode
        start = self._roi_drag_start_rect
        if mode is None or start is None:
            return None

        x1, y1 = start.x(), start.y()
        x2, y2 = x1 + start.width(), y1 + start.height()
        min_size = int(self._ROI_MIN_SIZE_PX)
        cx, cy = int(round(wx)), int(round(wy))

        if mode == "move":
            if self._roi_drag_origin is None:
                return None
            ox, oy = self._roi_drag_origin
            dx, dy = int(round(wx - ox)), int(round(wy - oy))
            return self._clamp_roi_move(x1 + dx, y1 + dy, start.width(), start.height())

        if "l" in mode:
            x1 = min(cx, x2 - min_size)
        if "r" in mode:
            x2 = max(cx, x1 + min_size)
        if "t" in mode:
            y1 = min(cy, y2 - min_size)
        if "b" in mode:
            y2 = max(cy, y1 + min_size)
        return self._clamp_roi_rect(QRect(x1, y1, x2 - x1, y2 - y1))

    def _refresh_pixel_info(self, sx: int, sy: int):
        if self._cv_img is None:
            if self._hover_img_pos is not None:
                self._hover_img_pos = None
                self.update()
            return
        wx, wy = self.to_world(float(sx), float(sy))
        ih, iw = self._cv_img.shape[:2]
        ix, iy = int(wx), int(wy)
        new_pos = (ix, iy) if (0 <= ix < iw and 0 <= iy < ih) else None
        if new_pos != self._hover_img_pos:
            self._hover_img_pos = new_pos
            self.update()

    def _paint_pixel_info(self, painter: QPainter):
        if self._cv_img is None or self._hover_img_pos is None:
            return
        ix, iy = self._hover_img_pos
        ih, iw = self._cv_img.shape[:2]
        if not (0 <= ix < iw and 0 <= iy < ih):
            return
        img = self._cv_img
        if img.ndim == 2:
            text = f"x: {ix}   y: {iy}   |   I: {int(img[iy, ix])}"
        elif img.ndim >= 3 and img.shape[2] >= 3:
            b, g, r = int(img[iy, ix, 0]), int(img[iy, ix, 1]), int(img[iy, ix, 2])
            text = f"x: {ix}   y: {iy}   |   R:{r}  G:{g}  B:{b}"
        else:
            return
        font = QFont("Courier New", 9)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text) + 16
        th = fm.height() + 10
        rx, ry = 6, self.height() - th - 6
        painter.fillRect(rx, ry, tw, th, QColor(30, 30, 46, 217))
        painter.setPen(QColor("#cdd6f4"))
        painter.drawText(rx + 8, ry + th - 6, text)

    def _paint_points(self, painter: QPainter):
        if len(self.points_x) < 2:
            return
        pen = QPen(QColor("#f9e2af"), 1)
        painter.setPen(pen)
        path = QPainterPath()
        sx, sy = self.to_screen(self.points_x[0], self.points_y[0])
        path.moveTo(sx, sy)
        for wx, wy in zip(self.points_x[1:], self.points_y[1:]):
            sx, sy = self.to_screen(wx, wy)
            path.lineTo(sx, sy)
        painter.drawPath(path)

    def _paint_curve(self, painter: QPainter):
        data = self.curve_data
        if data is None:
            return
        xs, ys = data["x"], data["y"]
        pen = QPen(QColor("#89dceb"), 2)
        painter.setPen(pen)
        path = QPainterPath()
        sx, sy = self.to_screen(float(xs[0]), float(ys[0]))
        path.moveTo(sx, sy)
        for i in range(1, len(xs)):
            sx, sy = self.to_screen(float(xs[i]), float(ys[i]))
            path.lineTo(sx, sy)
        painter.drawPath(path)

    def _paint_vectors(self, painter: QPainter):
        data = self.curve_data
        if data is None:
            return
        xs, ys = data["x"], data["y"]
        tx, ty = data["tx"], data["ty"]
        nx, ny = data["nx"], data["ny"]
        step = max(1, int(self.vec_density))
        vlen = float(self.vec_len)

        tang_pen = QPen(QColor("#cba6f7"), 1)
        norm_pen = QPen(QColor("#f9e2af"), 1)

        for i in range(0, len(xs), step):
            sx, sy = self.to_screen(float(xs[i]), float(ys[i]))

            painter.setPen(tang_pen)
            painter.drawLine(
                QPointF(sx - float(tx[i]) * vlen, sy - float(ty[i]) * vlen),
                QPointF(sx + float(tx[i]) * vlen, sy + float(ty[i]) * vlen),
            )
            painter.setPen(norm_pen)
            painter.drawLine(
                QPointF(sx, sy),
                QPointF(sx + float(nx[i]) * vlen, sy + float(ny[i]) * vlen),
            )

    def _paint_circle_preview(self, painter: QPainter):
        cx, cy = self.to_screen(*self._circle_start)
        r = self._circle_r * self.zoom
        pen = QPen(QColor("#f9e2af"), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), r, r)

    def _paint_hover(self, painter: QPainter):
        sx, sy, k, r, nx_, ny_, tx_, ty_ = self._hover_data  # type: ignore

        # Tangent line
        tlen = 100.0
        pen = QPen(QColor("#cba6f7"), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(
            QPointF(sx - tx_ * tlen, sy - ty_ * tlen),
            QPointF(sx + tx_ * tlen, sy + ty_ * tlen),
        )

        # Osculating circle
        r_screen = r * self.zoom
        r_vis = min(max(r_screen, 0.0), 2000.0)
        csx, csy = sx + nx_ * r_vis, sy + ny_ * r_vis

        # Point marker
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#f38ba8")))
        painter.drawEllipse(QPointF(sx, sy), 4, 4)

        # Radius line
        pen2 = QPen(QColor("#f38ba8"), 1, Qt.DashLine)
        painter.setPen(pen2)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(QPointF(sx, sy), QPointF(csx, csy))

        if r_vis < 1200:
            pen3 = QPen(QColor("#a6e3a1"), 2)
            painter.setPen(pen3)
            painter.drawEllipse(QPointF(csx, csy), r_vis, r_vis)

        # Text label
        txt = f"R: {r:.1f}\nk: {k:.4f}"
        painter.setPen(QColor(_TEXT))
        font = QFont("Segoe UI", 10)
        font.setBold(True)
        painter.setFont(font)
        # shadow
        painter.setPen(QColor("#000000"))
        painter.drawText(QRect(int(sx) + 13, int(sy) - 38, 120, 36), 0, txt)
        painter.setPen(QColor(_TEXT))
        painter.drawText(QRect(int(sx) + 12, int(sy) - 39, 120, 36), 0, txt)

    # -- Wheel (zoom) --------------------------------------------------------

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        scale = 1.1 if delta > 0 else 0.9
        mx, my = event.x(), event.y()
        mwx = (mx - self.pan_x) / self.zoom
        mwy = (my - self.pan_y) / self.zoom
        self.zoom = float(np.clip(self.zoom * scale, 0.05, 30.0))
        self.pan_x = mx - mwx * self.zoom
        self.pan_y = my - mwy * self.zoom
        self.zoom_changed.emit(self.zoom)
        self.update()

    # -- Mouse events (pan + draw + hover) -----------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._panning = True
            self._last_mouse = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton and self.align_mode:
            wx, wy = self.to_world(event.x(), event.y())
            self._align_start = (wx, wy)
            self._align_end = (wx, wy)
            self.update()
        elif event.button() == Qt.LeftButton and self.roi_mode:
            wx, wy = self.to_world(event.x(), event.y())
            self._roi_start = (wx, wy)
            self._roi_temp = None
            self.update()
        elif event.button() == Qt.LeftButton and self.roi_rect is not None and not self.draw_enabled:
            hit = self._roi_hit_test(float(event.x()), float(event.y()))
            if hit is not None:
                wx, wy = self.to_world(event.x(), event.y())
                self._roi_drag_mode = hit
                self._roi_drag_origin = (wx, wy)
                self._roi_drag_start_rect = QRect(self.roi_rect)
                self.setCursor(self._cursor_for_roi_mode(hit))
                self.update()
        elif event.button() == Qt.LeftButton and self.draw_enabled:
            self._is_drawing = True
            wx, wy = self.to_world(event.x(), event.y())
            self._circle_start = (wx, wy)
            self._circle_r = 0.0
            if self.draw_mode == "freehand":
                self.points_x = [wx]
                self.points_y = [wy]
            self.curve_data = None
            self.update()

    def mouseMoveEvent(self, event):
        if self._panning:
            dx = event.x() - self._last_mouse.x()
            dy = event.y() - self._last_mouse.y()
            self.pan_x += dx
            self.pan_y += dy
            self._last_mouse = event.pos()
            self.update()
            return

        if self.roi_mode and self._roi_start is not None:
            wx, wy = self.to_world(event.x(), event.y())
            x0, y0 = self._roi_start
            ix, iy = int(min(x0, wx)), int(min(y0, wy))
            iw, ih = int(abs(wx - x0)), int(abs(wy - y0))
            self._roi_temp = QRect(ix, iy, iw, ih)
            self.update()
            return

        if self.align_mode and self._align_start is not None:
            wx, wy = self.to_world(event.x(), event.y())
            self._align_end = (wx, wy)
            self.setCursor(Qt.CrossCursor)
            self.update()
            return

        if self._roi_drag_mode is not None:
            wx, wy = self.to_world(event.x(), event.y())
            new_rect = self._roi_drag_rect(wx, wy)
            if new_rect is not None:
                self.roi_rect = new_rect
                self.update()
            return

        if self.roi_rect is not None and not self.draw_enabled:
            self.setCursor(self._cursor_for_roi_mode(
                self._roi_hit_test(float(event.x()), float(event.y()))
            ))
        elif not self._is_drawing:
            self.setCursor(Qt.CrossCursor)

        if self._is_drawing and self.draw_enabled:
            wx, wy = self.to_world(event.x(), event.y())
            if self.draw_mode == "freehand":
                self.points_x.append(wx)
                self.points_y.append(wy)
            else:  # circle preview
                self._circle_r = math.hypot(
                    wx - self._circle_start[0], wy - self._circle_start[1]
                )
            self.update()
            return

        # Hover inspection
        data = self.curve_data
        if data is not None:
            mx, my = self.to_world(event.x(), event.y())
            xs, ys = data["x"], data["y"]
            d2 = (xs - mx) ** 2 + (ys - my) ** 2
            idx = int(np.argmin(d2))
            sx, sy = self.to_screen(float(xs[idx]), float(ys[idx]))
            if math.hypot(sx - event.x(), sy - event.y()) < 35:
                self._hover_data = (
                    sx, sy,
                    float(data["k"][idx]), float(data["r"][idx]),
                    float(data["nx"][idx]), float(data["ny"][idx]),
                    float(data["tx"][idx]), float(data["ty"][idx]),
                )
                self.update()
                return
        if self._hover_data is not None:
            self._hover_data = None
            self.update()
        self._refresh_pixel_info(event.x(), event.y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._panning = False
            self.setCursor(Qt.CrossCursor)
        elif event.button() == Qt.LeftButton and self.align_mode and self._align_start is not None:
            wx, wy = self.to_world(event.x(), event.y())
            self._align_end = (wx, wy)
            x0, y0 = self._align_start
            x1, y1 = self._align_end
            self.align_mode = False
            self._align_start = None
            self._align_end = None
            self.setCursor(Qt.CrossCursor)
            if math.hypot(x1 - x0, y1 - y0) > 2.0:
                self.align_line_drawn.emit((x0, y0, x1, y1))
            else:
                self.align_line_drawn.emit(None)
            self.update()
        elif event.button() == Qt.LeftButton and self.roi_mode and self._roi_start is not None:
            wx, wy = self.to_world(event.x(), event.y())
            x0, y0 = self._roi_start
            ix, iy = int(min(x0, wx)), int(min(y0, wy))
            iw, ih = int(abs(wx - x0)), int(abs(wy - y0))
            if iw > 2 and ih > 2:
                self.roi_rect = self._clamp_roi_rect(QRect(ix, iy, iw, ih))
            self._roi_start = None
            self._roi_temp = None
            self.roi_mode = False
            self.setCursor(Qt.CrossCursor)
            self.roi_changed.emit(self.roi_rect)
            self.update()
        elif event.button() == Qt.LeftButton and self._roi_drag_mode is not None:
            self._roi_drag_mode = None
            self._roi_drag_origin = None
            self._roi_drag_start_rect = None
            self.setCursor(self._cursor_for_roi_mode(
                self._roi_hit_test(float(event.x()), float(event.y()))
            ))
            self.roi_changed.emit(self.roi_rect)
            self.update()
        elif event.button() == Qt.LeftButton and self._is_drawing:
            self._is_drawing = False
            if self.draw_mode == "circle":
                wx, wy = self.to_world(event.x(), event.y())
                r = math.hypot(wx - self._circle_start[0],
                               wy - self._circle_start[1])
                if r > 1e-6:
                    angles = np.linspace(0, 2 * math.pi, 200)
                    cx0, cy0 = self._circle_start
                    self.points_x = (cx0 + r * np.cos(angles)).tolist()
                    self.points_y = (cy0 + r * np.sin(angles)).tolist()
            n = len(self.points_x)
            self.status_message.emit(
                f"Points captured: {n}. Click 'Compute curvature'."
            )
            self.points_captured.emit(self.points_x[:], self.points_y[:])
            self.update()


# -- MainWindow ---------------------------------------------------------------

class MainWindow(QMainWindow):
    """Top-level PyQt5 window - replaces Tkinter CombinedApp."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tip Curvature Analyzer  v2")
        self.resize(1520, 960)
        self.setMinimumSize(1100, 700)

        self.detector = ContourDetector()
        self.curv = CurvatureEngine()

        self.current_view: str = "preprocessed"
        self.selected_contour_index: Optional[int] = None
        self.frame_pre_params: Dict[int, Tuple[float, int]] = {}
        self._active_pre_key: Optional[int] = None

        self._undo_stack: deque = deque(maxlen=20)

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)

        self._root_splitter = None  # set by _build_central
        self._splitter_sized = False

        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self.statusBar().showMessage("Ready - Ctrl+O to open an image.")

    def showEvent(self, event):
        super().showEvent(event)
        if not self._splitter_sized and self._root_splitter is not None:
            self._splitter_sized = True
            total = self._root_splitter.width()
            self._root_splitter.setSizes([
                total * 1 // 6,
                total * 3 // 6,
                total * 2 // 6,
            ])

    # -- Menu -----------------------------------------------------------------

    def _build_menu(self):
        mb = self.menuBar()

        file_m = mb.addMenu("File")
        file_m.addAction(self._act("Open Image(s)  (Ctrl+O)", self.open_image, "Ctrl+O"))
        file_m.addSeparator()
        file_m.addAction(self._act("Save Annotated  (Ctrl+S)", self.save_annotated, "Ctrl+S"))
        file_m.addAction(self._act("Save Binary Mask", self.save_binary))
        file_m.addAction(self._act("Save Curvature Image", self.save_curvature_image))
        file_m.addSeparator()
        file_m.addAction(self._act("Export Properties CSV (Current Frame)", self.export_csv_current))
        file_m.addAction(self._act("Export Properties CSV (All Detected Frames)", self.export_csv_all))
        file_m.addSeparator()
        file_m.addAction(self._act("Export Contour Points (Current Frame)", self.export_contours_current))
        file_m.addAction(self._act("Export Contour Points (All Frames)", self.export_contours_all))
        file_m.addSeparator()
        file_m.addAction(self._act("Export Curvature (Current Frame)", self.export_curvature_current))
        file_m.addAction(self._act("Export Curvature (All Frames)", self.export_curvature_all))
        file_m.addSeparator()
        file_m.addAction(self._act("Exit", self.close))

        edit_m = mb.addMenu("Edit")
        edit_m.addAction(self._act("Undo  (Ctrl+Z)", self._undo, "Ctrl+Z"))

        view_m = mb.addMenu("View")
        view_m.addAction(self._act("Preprocessed", lambda: self.set_view("preprocessed")))
        view_m.addAction(self._act("Binary Mask", lambda: self.set_view("binary")))
        view_m.addAction(self._act("Contours", lambda: self.set_view("contours")))
        view_m.addSeparator()
        view_m.addAction(self._act("Align by Line", self._start_align_mode))
        view_m.addSeparator()
        view_m.addAction(self._act("Fit to View  (F)", self.fit_to_view, "F"))

        help_m = mb.addMenu("Help")
        help_m.addAction(self._act("About", self._about))

    def _act(self, label: str, slot, shortcut: str = "") -> QAction:
        a = QAction(label, self)
        if shortcut:
            a.setShortcut(shortcut)
        a.triggered.connect(slot)
        return a

    @staticmethod
    def _series_sort_key(path: str):
        name = Path(path).name.lower()
        return [
            int(tok) if tok.isdigit() else tok
            for tok in re.split(r"(\d+)", name)
        ]

    # -- Toolbar ---------------------------------------------------------------

    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))

        btn_open = QPushButton("  Open")
        btn_open.setToolTip("Open one/multiple images or TIFF stack (Ctrl+O)")
        btn_open.clicked.connect(self.open_image)
        tb.addWidget(btn_open)

        tb.addSeparator()

        btn_detect = QPushButton("  Detect Frame")
        btn_detect.setObjectName("accentBtn")
        btn_detect.setToolTip("Run contour detection on the current frame")
        btn_detect.clicked.connect(self.detect_current)
        tb.addWidget(btn_detect)

        btn_all = QPushButton("  Detect All")
        btn_all.setObjectName("greenBtn")
        btn_all.setToolTip("Run contour detection on all frames in the stack")
        btn_all.clicked.connect(self.detect_all)
        tb.addWidget(btn_all)

        tb.addSeparator()

        self._btn_roi = QPushButton("  ROI")
        self._btn_roi.setCheckable(True)
        self._btn_roi.setToolTip(
            "Draw ROI with left-drag. Then drag inside/handles to move or resize."
        )
        self._btn_roi.clicked.connect(self._on_toolbar_roi_toggle)
        tb.addWidget(self._btn_roi)

        self._align_target_cb = QComboBox()
        self._align_target_cb.addItems(["Vertical", "Horizontal"])
        self._align_target_cb.setToolTip("Target orientation for line-based alignment")
        self._align_target_cb.setFixedWidth(110)
        tb.addWidget(self._align_target_cb)

        self._btn_align = QPushButton("  Align")
        self._btn_align.setCheckable(True)
        self._btn_align.setToolTip("Draw a reference line on the image to align stack")
        self._btn_align.clicked.connect(self._on_align_toggle)
        tb.addWidget(self._btn_align)

        tb.addSeparator()

        btn_fit = QPushButton("Fit")
        btn_fit.setToolTip("Fit image to window (F)")
        btn_fit.clicked.connect(self.fit_to_view)
        tb.addWidget(btn_fit)

        # Zoom label (right-aligned via spacer)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)

        self.zoom_label = QLabel("Zoom: 1.00x")
        self.zoom_label.setStyleSheet(f"color: {_SUBTEXT}; padding-right: 8px;")
        tb.addWidget(self.zoom_label)

    # -- Central layout --------------------------------------------------------

    def _build_central(self):
        root_splitter = QSplitter(Qt.Horizontal)
        root_splitter.setHandleWidth(4)
        self.setCentralWidget(root_splitter)

        # ── Panel 1 (1/6): tabs ──────────────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(200)
        left_scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {_BG}; }}"
        )

        left_container = QWidget()
        left_vbox = QVBoxLayout(left_container)
        left_vbox.setContentsMargins(6, 6, 6, 6)
        left_vbox.setSpacing(0)

        self.nb = QTabWidget()
        self.nb.currentChanged.connect(self._on_tab_changed)
        left_vbox.addWidget(self.nb)

        tab_pre = QWidget()
        tab_cnt = QWidget()
        tab_curv = QWidget()
        self.nb.addTab(tab_pre, "Preprocessing")
        self.nb.addTab(tab_cnt, "Contour")
        self.nb.addTab(tab_curv, "Curvature")

        self._build_preprocessing_tab(tab_pre)
        self._build_contour_tab(tab_cnt)
        self._build_curvature_tab(tab_curv)

        left_scroll.setWidget(left_container)
        root_splitter.addWidget(left_scroll)

        # ── Panel 2 (3/6): image canvas + frame nav ───────────────────────────
        mid_w = QWidget()
        mid_vbox = QVBoxLayout(mid_w)
        mid_vbox.setContentsMargins(4, 4, 4, 4)
        mid_vbox.setSpacing(4)

        view_row = QHBoxLayout()
        self.view_label = QLabel("View: (none)")
        self.view_label.setStyleSheet(
            f"color: {_ACCENT}; font-weight: bold; font-size: 11pt;"
        )
        view_row.addWidget(self.view_label)
        view_row.addStretch()
        mid_vbox.addLayout(view_row)

        self.canvas = ImageCanvas()
        self.canvas.zoom_changed.connect(
            lambda z: self.zoom_label.setText(f"Zoom: {z:.2f}x")
        )
        self.canvas.status_message.connect(self.statusBar().showMessage)
        self.canvas.points_captured.connect(self._on_canvas_points)
        self.canvas.roi_changed.connect(self._on_roi_changed)
        self.canvas.align_line_drawn.connect(self._on_align_line_drawn)
        mid_vbox.addWidget(self.canvas, stretch=1)

        self._build_frame_nav(mid_vbox)

        root_splitter.addWidget(mid_w)

        # ── Panel 3 (2/6): histogram (top) + contour properties (bottom) ──────
        right_col_split = QSplitter(Qt.Vertical)
        right_col_split.setHandleWidth(4)

        hist_w = QWidget()
        hist_vbox = QVBoxLayout(hist_w)
        hist_vbox.setContentsMargins(4, 4, 4, 4)
        hist_vbox.setSpacing(4)

        hist_header = QHBoxLayout()
        lbl_h = QLabel("Pixel Histogram")
        lbl_h.setStyleSheet(f"color: {_ACCENT}; font-weight: bold;")
        hist_header.addWidget(lbl_h)
        hist_header.addStretch()
        self._log_cb = QCheckBox("Log scale")
        self._log_cb.stateChanged.connect(
            lambda s: (self.histogram.set_log(bool(s)), None)
        )
        hist_header.addWidget(self._log_cb)
        hist_vbox.addLayout(hist_header)

        self.histogram = HistogramWidget()
        hist_vbox.addWidget(self.histogram)
        right_col_split.addWidget(hist_w)

        table_w = QWidget()
        self._build_table(table_w)
        right_col_split.addWidget(table_w)
        right_col_split.setStretchFactor(0, 1)
        right_col_split.setStretchFactor(1, 1)
        right_col_split.setCollapsible(0, False)
        right_col_split.setCollapsible(1, False)

        root_splitter.addWidget(right_col_split)

        # Proportions: 1 : 3 : 2
        root_splitter.setStretchFactor(0, 1)
        root_splitter.setStretchFactor(1, 3)
        root_splitter.setStretchFactor(2, 2)
        root_splitter.setCollapsible(0, False)
        root_splitter.setCollapsible(1, False)
        root_splitter.setCollapsible(2, False)

        self._root_splitter = root_splitter

    # -- Frame navigation ------------------------------------------------------

    def _build_frame_nav(self, layout):
        box = QGroupBox()
        row = QHBoxLayout(box)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(4)

        _btn_style = (
            f"QPushButton {{ background: {_PANEL}; color: {_TEXT}; border: 1px solid {_BORDER};"
            f"  border-radius: 6px; padding: 4px 8px; font-size: 16pt;"
            f"  min-width: 36px; min-height: 36px; max-width: 44px; max-height: 44px; }}"
            f"QPushButton:hover {{ background: {_WIDGET}; border-color: {_ACCENT}; }}"
            f"QPushButton:pressed {{ background: {_ACCENT}; color: {_BG}; }}"
        )

        btn_first = QPushButton("\u23ee")
        btn_first.setToolTip("First frame")
        btn_first.setStyleSheet(_btn_style)
        btn_first.clicked.connect(lambda: self.go_to_frame(0))

        btn_prev = QPushButton("\u23ea")
        btn_prev.setToolTip("Previous frame")
        btn_prev.setStyleSheet(_btn_style)
        btn_prev.clicked.connect(self.prev_frame)

        self._btn_play = QPushButton("\u25b6")
        self._btn_play.setToolTip("Play / Stop")
        self._btn_play.setCheckable(True)
        self._btn_play.setStyleSheet(
            _btn_style
            + f"QPushButton:checked {{ background: {_ACCENT}; color: {_BG}; border-color: {_ACCENT}; }}"
        )
        self._btn_play.clicked.connect(self._on_play_toggle)

        btn_next = QPushButton("\u23e9")
        btn_next.setToolTip("Next frame")
        btn_next.setStyleSheet(_btn_style)
        btn_next.clicked.connect(self.next_frame)

        btn_last = QPushButton("\u23ed")
        btn_last.setToolTip("Last frame")
        btn_last.setStyleSheet(_btn_style)
        btn_last.clicked.connect(lambda: self.go_to_frame(
            max(0, self.detector.num_frames - 1)
        ))

        for b in (btn_first, btn_prev, self._btn_play, btn_next, btn_last):
            row.addWidget(b)

        row.addSpacing(6)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_frame_slider)
        row.addWidget(self.frame_slider, stretch=1)

        row.addSpacing(6)

        self.frame_info_lbl = QLabel("No image loaded")
        self.frame_info_lbl.setStyleSheet(f"color: {_ACCENT}; font-weight: bold;")
        self.frame_info_lbl.setFixedWidth(110)
        self.frame_info_lbl.setAlignment(Qt.AlignCenter)
        row.addWidget(self.frame_info_lbl)

        self.detect_info_lbl = QLabel("")
        self.detect_info_lbl.setStyleSheet(f"color: {_SUBTEXT};")
        row.addWidget(self.detect_info_lbl)

        layout.addWidget(box)

    def _on_play_toggle(self, checked: bool):
        if checked:
            if self.detector.num_frames < 2:
                self._btn_play.setChecked(False)
                return
            self._btn_play.setText("\u23f9")   # ⏹ stop
            self._btn_play.setToolTip("Stop")
            self._play_timer.start(100)  # 10 fps
        else:
            self._play_timer.stop()
            self._btn_play.setText("\u25b6")   # ▶ play
            self._btn_play.setToolTip("Play / Stop")

    def _play_tick(self):
        if not self.detector.frames:
            self._play_timer.stop()
            self._btn_play.setChecked(False)
            self._btn_play.setText("\u25b6")
            return
        next_idx = self.detector.current_frame_idx + 1
        if next_idx >= self.detector.num_frames:
            next_idx = 0   # loop
        self.go_to_frame(next_idx)

    # -- Contour table ---------------------------------------------------------

    def _build_table(self, parent: QWidget):
        vbox = QVBoxLayout(parent)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(4)

        hdr = QLabel("Contour Properties  (select a row)")
        hdr.setStyleSheet(f"color: {_ACCENT}; font-weight: bold; font-size: 11pt; padding: 4px 6px;")
        vbox.addWidget(hdr)

        cols = ["#", "Area", "Perimeter", "Centroid", "BBox",
                "Circularity", "Aspect Ratio", "Solidity"]
        self.table = QTableWidget(0, len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)

        col_widths = [40, 80, 80, 110, 150, 80, 90, 70]
        for i, w in enumerate(col_widths):
            self.table.setColumnWidth(i, w)

        self.table.itemSelectionChanged.connect(self._on_table_select)
        vbox.addWidget(self.table)

    # -- Preprocessing tab -----------------------------------------------------

    def _build_preprocessing_tab(self, parent: QWidget):
        vbox = QVBoxLayout(parent)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        # Preprocessing params
        grp = QGroupBox("Preprocessing")
        g_lay = QVBoxLayout(grp)
        g_lay.setSpacing(6)

        self.pre_invert_cb = QCheckBox("Invert image")
        self.pre_invert_cb.stateChanged.connect(self._on_pre_change)
        g_lay.addWidget(self.pre_invert_cb)

        self.pre_alpha_sl = LabeledSlider(
            "Contrast (alpha):", 0.1, 20.0, 1.0, is_float=True, float_scale=10
        )
        self.pre_alpha_sl.setToolTip("Contrast multiplier - 1.0 = no change, >1 increases contrast")
        self.pre_alpha_sl.value_changed.connect(self._on_pre_change)
        g_lay.addWidget(self.pre_alpha_sl)

        self.pre_beta_sl = LabeledSlider("Brightness (beta):", -100, 100, 0)
        self.pre_beta_sl.setToolTip("Brightness offset - positive = brighter, negative = darker")
        self.pre_beta_sl.value_changed.connect(self._on_pre_change)
        g_lay.addWidget(self.pre_beta_sl)

        vbox.addWidget(grp)

        # Blur
        grp_blur = QGroupBox("Gaussian Blur")
        gb_lay = QVBoxLayout(grp_blur)
        self.pre_blur_sl = LabeledSlider("Blur ksize (odd):", 1, 31, 5)
        self.pre_blur_sl.setToolTip("Gaussian blur kernel size (odd) - larger = more blur")
        self.pre_blur_sl.value_changed.connect(self._on_pre_change)
        gb_lay.addWidget(self.pre_blur_sl)
        vbox.addWidget(grp_blur)

        # Peak normalization
        grp_norm = QGroupBox("Peak Normalization")
        gn_lay = QVBoxLayout(grp_norm)
        gn_lay.setSpacing(6)

        lbl_target = QLabel("Target peak intensity (0-255):")
        lbl_target.setStyleSheet(f"color: {_SUBTEXT};")
        gn_lay.addWidget(lbl_target)

        self.peak_target_sb = QSpinBox()
        self.peak_target_sb.setRange(1, 255)
        self.peak_target_sb.setValue(200)
        self.peak_target_sb.setSingleStep(1)
        self.peak_target_sb.setStyleSheet(
            f"background: {_PANEL}; color: {_TEXT}; border: 1px solid {_BORDER};"
            f" border-radius: 4px; padding: 2px 6px;"
        )
        gn_lay.addWidget(self.peak_target_sb)

        btn_norm = QPushButton("Apply to Stack")
        btn_norm.setObjectName("accentBtn")
        btn_norm.clicked.connect(self._apply_peak_normalization)
        gn_lay.addWidget(btn_norm)

        self.peak_norm_lbl = QLabel("")
        self.peak_norm_lbl.setWordWrap(True)
        self.peak_norm_lbl.setStyleSheet(f"color: {_SUBTEXT}; font-size: 9pt;")
        gn_lay.addWidget(self.peak_norm_lbl)

        vbox.addWidget(grp_norm)

        # Reset
        grp_act = QGroupBox("Actions")
        ga_lay = QVBoxLayout(grp_act)
        btn_rst = QPushButton("Reset defaults")
        btn_rst.clicked.connect(self._reset_pre_defaults)
        ga_lay.addWidget(btn_rst)
        vbox.addWidget(grp_act)

        vbox.addStretch()

    # -- Contour tab -----------------------------------------------------------

    def _build_contour_tab(self, parent: QWidget):
        vbox = QVBoxLayout(parent)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        # Method
        grp_m = QGroupBox("Detection Method")
        gm_lay = QVBoxLayout(grp_m)
        gm_lay.addWidget(QLabel("Method:"))
        self.method_cb = QComboBox()
        methods = ["Binary Threshold", "Canny Edge"]
        self.method_cb.addItems(methods)
        self.method_cb.currentTextChanged.connect(self._update_method_frames)
        gm_lay.addWidget(self.method_cb)
        vbox.addWidget(grp_m)

        # Threshold / Canny (toggleable)
        self.grp_thresh = QGroupBox("Threshold")
        gt = QVBoxLayout(self.grp_thresh)
        self.thresh_sl = LabeledSlider("Threshold value:", 0, 255, 127)
        self.thresh_sl.setToolTip("Pixel threshold (0-255) - pixels above become white in binary mask")
        gt.addWidget(self.thresh_sl)
        vbox.addWidget(self.grp_thresh)

        self.grp_canny = QGroupBox("Canny")
        gc = QVBoxLayout(self.grp_canny)
        self.canny_low_sl = LabeledSlider("Low:", 0, 500, 50)
        self.canny_low_sl.setToolTip("Canny lower hysteresis threshold - lower = more edge detail")
        self.canny_high_sl = LabeledSlider("High:", 0, 500, 150)
        self.canny_high_sl.setToolTip("Canny upper hysteresis threshold - higher = stricter edges")
        gc.addWidget(self.canny_low_sl)
        gc.addWidget(self.canny_high_sl)
        vbox.addWidget(self.grp_canny)

        self._update_method_frames()

        # Filtering
        grp_f = QGroupBox("Filtering")
        gf = QVBoxLayout(grp_f)
        self.min_area_sl = LabeledSlider("Min area:", 0, 20000, 100)
        self.min_area_sl.setToolTip("Minimum contour area in pixels - smaller shapes are discarded")
        self.max_area_sl = LabeledSlider("Max area (0=off):", 0, 500000, 0)
        self.max_area_sl.setToolTip("Maximum contour area (0 = no limit)")
        self.min_rect_width_sl = LabeledSlider("Min width (px, 0=off):", 0, 50, 5)
        self.min_rect_width_sl.setToolTip(
            "Reject contours whose thinnest dimension is below this value - "
            "removes straight-line artefacts"
        )
        gf.addWidget(self.min_area_sl)
        gf.addWidget(self.max_area_sl)
        gf.addWidget(self.min_rect_width_sl)
        vbox.addWidget(grp_f)

        # Detect buttons
        btn_det = QPushButton("Detect Current Frame")
        btn_det.setObjectName("accentBtn")
        btn_det.clicked.connect(self.detect_current)
        vbox.addWidget(btn_det)

        btn_all = QPushButton("Detect All Frames")
        btn_all.setObjectName("greenBtn")
        btn_all.clicked.connect(self.detect_all)
        vbox.addWidget(btn_all)

        grp_roi = QGroupBox("Region of Interest (ROI)")
        gr = QVBoxLayout(grp_roi)
        roi_btn_row = QHBoxLayout()
        btn_draw_roi = QPushButton("Draw ROI")
        btn_draw_roi.setToolTip(
            "Activate ROI mode - left-drag to create; drag ROI/handles to adjust."
        )
        btn_draw_roi.clicked.connect(self._on_roi_draw)
        btn_clear_roi = QPushButton("Clear ROI")
        btn_clear_roi.setToolTip("Remove ROI - detection uses the full image")
        btn_clear_roi.clicked.connect(self._on_roi_clear)
        roi_btn_row.addWidget(btn_draw_roi)
        roi_btn_row.addWidget(btn_clear_roi)
        gr.addLayout(roi_btn_row)
        self._roi_status_lbl = QLabel("No ROI set")
        self._roi_status_lbl.setStyleSheet(f"color: {_SUBTEXT}; font-size: 9pt;")
        gr.addWidget(self._roi_status_lbl)
        vbox.addWidget(grp_roi)
        vbox.addStretch()

    @staticmethod
    def _color_row(layout, label: str, default: str) -> QLineEdit:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {_SUBTEXT};")
        row.addWidget(lbl)
        edit = QLineEdit(default)
        edit.setFixedWidth(90)
        row.addWidget(edit)
        layout.addLayout(row)
        return edit

    # -- Curvature tab ---------------------------------------------------------

    def _build_curvature_tab(self, parent: QWidget):
        vbox = QVBoxLayout(parent)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        grp_src = QGroupBox("Source of Points")
        gs = QVBoxLayout(grp_src)

        btn_row = QHBoxLayout()
        btn_sel = QPushButton("Select contour")
        btn_sel.clicked.connect(self.use_selected_contour)
        btn_lg = QPushButton("Largest contour")
        btn_lg.clicked.connect(self.use_largest_contour)
        btn_clr = QPushButton("Clear")
        btn_clr.clicked.connect(self.clear_curvature)
        btn_row.addWidget(btn_sel)
        btn_row.addWidget(btn_lg)
        btn_row.addWidget(btn_clr)
        gs.addLayout(btn_row)
        vbox.addWidget(grp_src)

        grp_calc = QGroupBox("Spline & Curvature")
        gc = QVBoxLayout(grp_calc)
        self.smooth_sl = LabeledSlider("Smoothing factor (s):", 0, 2000, 300)
        self.smooth_sl.setToolTip("B-spline smoothing (s=0 exact fit, higher = smoother)")
        self.res_sl = LabeledSlider(
            "Resolution multiplier:", 1.0, 5.0, 2.0, is_float=True, float_scale=10
        )
        self.res_sl.setToolTip("Output spline point density multiplier")
        gc.addWidget(self.smooth_sl)
        gc.addWidget(self.res_sl)
        btn_compute_all = QPushButton("Compute Curvature")
        btn_compute_all.setObjectName("accentBtn")
        btn_compute_all.setToolTip("Batch compute curvature on largest contour of every frame")
        btn_compute_all.clicked.connect(self.compute_curvature_all)
        gc.addWidget(btn_compute_all)
        vbox.addWidget(grp_calc)

        grp_vis = QGroupBox("Visualization")
        gv = QVBoxLayout(grp_vis)
        self.vec_density_sl = LabeledSlider("Vector step (density):", 5, 50, 15)
        self.vec_density_sl.setToolTip("Draw a tangent/normal vector every N spline points")
        self.vec_len_sl = LabeledSlider("Vector length (px):", 10, 500, 50)
        self.vec_len_sl.setToolTip("Visual length of tangent/normal vectors in screen pixels")
        self.vec_density_sl.value_changed.connect(self._on_vec_change)
        self.vec_len_sl.value_changed.connect(self._on_vec_change)
        gv.addWidget(self.vec_density_sl)
        gv.addWidget(self.vec_len_sl)
        self.show_vec_cb = QCheckBox("Show tangents / normals")
        self.show_vec_cb.stateChanged.connect(self._on_vec_change)
        gv.addWidget(self.show_vec_cb)
        vbox.addWidget(grp_vis)

        self.curv_msg_lbl = QLabel(
            "\u2139  Tip: select a contour row in the table, then click 'Select contour'."
        )
        self.curv_msg_lbl.setWordWrap(True)
        self.curv_msg_lbl.setStyleSheet(
            f"color: {_SUBTEXT}; font-size: 9pt; "
            f"background: {_PANEL}; border-radius: 4px; padding: 4px 6px;"
        )
        vbox.addWidget(self.curv_msg_lbl)
        vbox.addStretch()

    # -- Method visibility -----------------------------------------------------

    def _update_method_frames(self):
        m = self.method_cb.currentText()
        self.grp_thresh.setVisible(m == "Binary Threshold")
        self.grp_canny.setVisible(m == "Canny Edge")

    # -- Preprocessing helpers -------------------------------------------------

    def _current_pre_params(self, frame_idx: Optional[int] = None) -> PreprocessParams:
        if frame_idx is None:
            frame_idx = self.detector.current_frame_idx
        alpha = self.pre_alpha_sl.get_value()
        beta = int(self.pre_beta_sl.get_value())
        if frame_idx in self.frame_pre_params:
            alpha, beta = self.frame_pre_params[frame_idx]
        k = int(self.pre_blur_sl.get_value())
        if k % 2 == 0:
            k += 1
        k = max(1, k)
        return PreprocessParams(
            invert=self.pre_invert_cb.isChecked(),
            alpha=float(alpha),
            beta=int(beta),
            blur_ksize=k,
        )

    def _get_detect_params(self) -> DetectParams:
        max_area = int(self.max_area_sl.get_value())
        max_area_val = max_area if max_area > 0 else None
        return DetectParams(
            method=self.method_cb.currentText(),
            blur_ksize=1,
            thresh_value=int(self.thresh_sl.get_value()),
            canny_low=int(self.canny_low_sl.get_value()),
            canny_high=int(self.canny_high_sl.get_value()),
            use_morphology=True,
            morph_ksize=5,
            min_area=int(self.min_area_sl.get_value()),
            max_area=max_area_val,
            min_rect_width=int(self.min_rect_width_sl.get_value()),
            contour_color=(0, 255, 0),
            bbox_color=(255, 0, 0),
            centroid_color=(0, 0, 255),
            thickness=5,
            draw_bbox=True,
            draw_centroid=True,
            draw_labels=True,
            reject_straight_lines=True,
            straight_line_tol=0.006,
        )

    def _reset_pre_defaults(self):
        self.pre_invert_cb.setChecked(False)
        self.pre_alpha_sl.set_value(1.0)
        self.pre_beta_sl.set_value(0)
        self.pre_blur_sl.set_value(5)
        self._on_pre_change()

    def _apply_peak_normalization(self):
        if not self.detector.frames:
            QMessageBox.warning(self, "No image", "Open an image first.")
            return
        self._save_undo("Peak normalization")
        target = self.peak_target_sb.value()
        n = self.detector.num_frames
        k = int(self.pre_blur_sl.get_value())
        if k % 2 == 0:
            k += 1
        k = max(1, k)
        invert = self.pre_invert_cb.isChecked()
        beta = int(self.pre_beta_sl.get_value())
        # Base preprocessing: invert + blur only (alpha=1, beta=0) to find raw peak
        base_pre = PreprocessParams(invert=invert, alpha=1, beta=0, blur_ksize=k)
        orig_idx = self.detector.current_frame_idx
        applied = 0
        for i in range(n):
            self.detector.set_frame(i)
            gray = self.detector.get_preprocessed_gray(base_pre)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist[0] = 0  # ignore background
            peak = int(np.argmax(hist))
            if peak <= 0:
                continue
            alpha = float(target) / float(peak)
            self.frame_pre_params[i] = (alpha, beta)
            self.detector.cache.pop(i, None)
            applied += 1
        self.detector.set_frame(orig_idx)
        self._active_pre_key = None
        self.peak_norm_lbl.setText(
            f"Applied to {applied}/{n} frames - target peak: {target}"
        )
        if self.detector.original is not None:
            pre_gray = self.detector.get_preprocessed_gray(self._current_pre_params())
            self._update_histogram(pre_gray)
        self.statusBar().showMessage(
            f"Peak normalization done - target {target} - {applied}/{n} frame(s) updated."
        )

    def _on_pre_change(self):
        if self.detector.frames:
            idx = self.detector.current_frame_idx
            self.frame_pre_params[idx] = (
                self.pre_alpha_sl.get_value(), int(self.pre_beta_sl.get_value())
            )
        self.detector.cache.clear()
        self.curv.clear()
        self.canvas.curve_data = None
        self.canvas.points_x = []
        self.canvas.points_y = []
        self.selected_contour_index = None
        self._populate_table([])
        self._update_frame_ui()
        self._active_pre_key = None
        if self.detector.original is not None:
            pre_gray = self.detector.get_preprocessed_gray(self._current_pre_params())
            self._update_histogram(pre_gray)

    def _apply_preprocessing_all_frames(self):
        n = self.detector.num_frames
        if n <= 0:
            return
        pre = self._current_pre_params()
        if n >= 15:
            prog = QProgressDialog(
                "Preprocessing all frames...", None, 0, n, self
            )
            prog.setWindowTitle("Preprocessing")
            prog.setWindowModality(Qt.WindowModal)
            prog.setValue(0)
            prog.show()
        else:
            prog = None

        current_idx = self.detector.current_frame_idx
        try:
            for i in range(n):
                self.detector.set_frame(i)
                self.detector.get_preprocessed_gray(self._current_pre_params(i))
                if prog:
                    prog.setValue(i + 1)
                    QApplication.processEvents()
        finally:
            self.detector.set_frame(current_idx)
            if prog:
                prog.close()

        self.statusBar().showMessage(
            f"Preprocessing cached for all {n} frames."
        )

    # -- Tab changed -----------------------------------------------------------

    def _on_tab_changed(self, idx: int):
        tab_text = self.nb.tabText(idx)
        if tab_text != "Contour":
            return
        if self.detector.num_frames == 0:
            return
        pre = self._current_pre_params()
        pre_key = hash(pre)
        if self._active_pre_key != pre_key:
            self.detector.cache.clear()
            self.selected_contour_index = None
            self._populate_table([])
            self._update_frame_ui()
            self.statusBar().showMessage(
                "Preprocessing changed - detections cleared."
            )
        self._apply_preprocessing_all_frames()
        self._active_pre_key = pre_key
        self._refresh_view()

    # -- View helpers ----------------------------------------------------------

    def _get_base_cv(self) -> Optional[np.ndarray]:
        if self.detector.original is None:
            return None
        if self.current_view == "preprocessed":
            try:
                return self.detector.get_preprocessed_gray(self._current_pre_params())
            except Exception:
                pass
        cached = self.detector.cache.get(self.detector.current_frame_idx)
        if self.current_view == "binary":
            return cached["binary"] if cached else None
        if self.current_view == "contours":
            return cached["annotated"] if cached else None
        # default: preprocessed
        try:
            return self.detector.get_preprocessed_gray(self._current_pre_params())
        except Exception:
            return None

    def _refresh_view(self):
        cv_img = self._get_base_cv()
        self.canvas.set_image(cv_img)

        # Contour overlays
        cached = self.detector.cache.get(self.detector.current_frame_idx)
        self.canvas.contours = cached["contours"] if cached else []
        self.canvas.selected_contour_idx = self.selected_contour_index

        # Histogram
        self._update_histogram(cv_img)

    def _update_histogram(self, cv_img: Optional[np.ndarray]):
        if cv_img is None:
            self.histogram.set_data(None)
            return
        if cv_img.dtype == np.uint16:
            hr = [0, 65536]
            rmax = 65536.0
        elif cv_img.dtype == np.uint8:
            hr = [0, 256]
            rmax = 256.0
        else:
            mx = float(cv_img.max()) + 1
            hr = [0, max(256.0, mx)]
            rmax = hr[1]
            cv_img = cv_img.astype(np.float32)

        if cv_img.ndim == 2:
            hist = cv2.calcHist([cv_img], [0], None, [256], hr)
            data = {"gray": hist.flatten()}
        else:
            gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray_img], [0], None, [256], hr)
            data = {"gray": hist.flatten()}
        self.histogram.set_data(data, rmax)

    def set_view(self, view: str):
        if self.detector.original is None:
            return
        if view in ("binary", "contours"):
            if not self.detector.cache.get(self.detector.current_frame_idx):
                QMessageBox.information(
                    self, "Info", "Run detection first to generate this view."
                )
                return
        self.current_view = view
        labels = {
            "preprocessed": "View: Preprocessed",
            "binary": "View: Binary Mask",
        }
        if view == "contours":
            c = len(self.detector.cache[
                self.detector.current_frame_idx]["contours"])
            self.view_label.setText(f"View: Contours  ({c})")
        else:
            self.view_label.setText(labels.get(view, "View: ?"))
        self._refresh_view()

    def fit_to_view(self):
        self.canvas.fit_to_view()

    # -- ROI -------------------------------------------------------------------

    def _get_roi_tuple(self):
        r = self.canvas.roi_rect
        if r is None or r.width() <= 0 or r.height() <= 0:
            return None
        return (r.x(), r.y(), r.width(), r.height())

    def _on_toolbar_roi_toggle(self, checked: bool):
        if checked:
            self.canvas.roi_mode = True
            self.canvas.align_mode = False
            self._btn_align.setChecked(False)
            self.canvas.draw_enabled = False
            self.statusBar().showMessage("ROI mode: left-drag on canvas to draw the region.")
        else:
            self.canvas.roi_mode = False

    def _on_roi_draw(self):
        self.canvas.roi_mode = True
        self.canvas.align_mode = False
        self._btn_align.setChecked(False)
        self.canvas.draw_enabled = False
        self._btn_roi.setChecked(True)
        self.statusBar().showMessage("ROI mode: left-drag on canvas to draw the region.")

    def _on_roi_clear(self):
        self._save_undo("Clear ROI")
        self.canvas.roi_rect = None
        self.canvas.roi_mode = False
        self._btn_roi.setChecked(False)
        self._roi_status_lbl.setText("No ROI set")
        self.canvas.update()
        self.statusBar().showMessage("ROI cleared - detection uses full image.")

    def _on_roi_changed(self, roi_rect):
        self._btn_roi.setChecked(False)
        if roi_rect is not None:
            x, y, w, h = roi_rect.x(), roi_rect.y(), roi_rect.width(), roi_rect.height()
            self._roi_status_lbl.setText(f"ROI: ({x}, {y})  {w}x{h} px")
            self.statusBar().showMessage("ROI set - run detection to apply.")
        else:
            self._roi_status_lbl.setText("No ROI set")

    # -- Open / Detect / Save --------------------------------------------------

    def open_image(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open image(s)", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*.*)",
        )
        if not paths:
            return
        paths = sorted(paths, key=self._series_sort_key)
        try:
            if len(paths) == 1:
                self.detector.load(paths[0])
            else:
                self.detector.load_paths(paths)
            self.frame_pre_params.clear()
            self._undo_stack.clear()
            self._reset_after_stack_transform()

            orig = self.detector.original
            shape = orig.shape
            ch = shape[2] if orig.ndim == 3 else 1
            self.statusBar().showMessage(
                f"Loaded: {(Path(paths[0]).name if len(paths) == 1 else str(len(paths)) + ' files as series')} | Frames: {self.detector.num_frames} | "
                f"Size: {shape[1]}x{shape[0]} | Dtype: {orig.dtype} | Channels: {ch}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Open error", str(e))

    def _reset_after_stack_transform(self):
        self.current_view = "preprocessed"
        self.view_label.setText("View: Preprocessed")
        self.selected_contour_index = None
        self.curv.clear()
        self._active_pre_key = None
        self.canvas.curve_data = None
        self.canvas.points_x = []
        self.canvas.points_y = []
        self.canvas.contours = []
        self.canvas.selected_contour_idx = None
        self.canvas.roi_rect = None
        self.canvas.roi_mode = False
        self._btn_roi.setChecked(False)
        self.canvas.align_mode = False
        self.canvas._align_start = None
        self.canvas._align_end = None
        self._btn_align.setChecked(False)
        self._roi_status_lbl.setText("No ROI set")
        self._populate_table([])

        pre_gray = self.detector.get_preprocessed_gray(self._current_pre_params())
        self.canvas.set_image(pre_gray)
        self.canvas.fit_to_view()
        self._update_histogram(pre_gray)
        self._update_frame_ui()

    @staticmethod
    def _norm_angle_deg(a: float) -> float:
        return ((float(a) + 180.0) % 360.0) - 180.0

    # -- Undo ------------------------------------------------------------------

    def _save_undo(self, description: str):
        """Snapshot current state onto the undo stack before a destructive action."""
        snap = {
            "desc": description,
            "rotation_angle": self.detector.rotation_angle,
            "current_frame_idx": self.detector.current_frame_idx,
            "cache": copy.deepcopy(self.detector.cache),
            "pre_cache": {},  # lightweight: let it recompute
            "frame_pre_params": dict(self.frame_pre_params),
            "current_view": self.current_view,
            "selected_contour_index": self.selected_contour_index,
            "roi_rect": QRect(self.canvas.roi_rect) if self.canvas.roi_rect is not None else None,
            "canvas_points_x": self.canvas.points_x[:],
            "canvas_points_y": self.canvas.points_y[:],
            "curve_data": copy.deepcopy(self.curv.curve_data),
            "curv_points_x": self.curv.points_x[:],
            "curv_points_y": self.curv.points_y[:],
        }
        self._undo_stack.append(snap)

    def _undo(self):
        if not self._undo_stack:
            self.statusBar().showMessage("Nothing to undo.")
            return
        snap = self._undo_stack.pop()

        # Restore detector state
        self.detector.rotation_angle = snap["rotation_angle"]
        self.detector.cache = snap["cache"]
        self.detector.pre_cache.clear()
        self.detector.set_frame(snap["current_frame_idx"])

        # Restore app state
        self.frame_pre_params = snap["frame_pre_params"]
        self.selected_contour_index = snap["selected_contour_index"]
        self._active_pre_key = None

        # Restore canvas overlay state
        self.canvas.points_x = snap["canvas_points_x"]
        self.canvas.points_y = snap["canvas_points_y"]
        self.canvas.roi_rect = snap["roi_rect"]
        self._roi_status_lbl.setText(
            "No ROI set" if snap["roi_rect"] is None
            else f"ROI: {snap['roi_rect'].width()}x{snap['roi_rect'].height()}"
        )

        # Restore curvature state
        self.curv.points_x = snap["curv_points_x"]
        self.curv.points_y = snap["curv_points_y"]
        self.curv.curve_data = snap["curve_data"]
        self.canvas.curve_data = snap["curve_data"]

        # Refresh UI
        self.set_view(snap["current_view"])
        self._update_frame_ui()
        if self.detector.original is not None:
            pre_gray = self.detector.get_preprocessed_gray(self._current_pre_params())
            self._update_histogram(pre_gray)

        self.statusBar().showMessage(f"Undone: {snap['desc']}")

    def _on_align_toggle(self, checked: bool):
        if checked:
            self._start_align_mode()
        else:
            self.canvas.align_mode = False
            self.canvas._align_start = None
            self.canvas._align_end = None
            self.canvas.update()

    def _start_align_mode(self):
        if self.detector.original is None:
            QMessageBox.warning(self, "No image", "Open an image first.")
            self._btn_align.setChecked(False)
            return
        self.canvas.align_mode = True
        self.canvas.draw_enabled = False
        self.canvas.roi_mode = False
        self._btn_roi.setChecked(False)
        self._btn_align.setChecked(True)
        target = self._align_target_cb.currentText().lower()
        self.statusBar().showMessage(
            f"Alignment mode: draw a line to align it {target}."
        )

    def _on_align_line_drawn(self, line):
        self._btn_align.setChecked(False)
        if line is None:
            self.statusBar().showMessage("Alignment cancelled.")
            return
        if self.detector.original is None or not self.detector.frames:
            return
        x0, y0, x1, y1 = map(float, line)
        dx, dy = (x1 - x0), (y1 - y0)
        if math.hypot(dx, dy) < 1e-6:
            self.statusBar().showMessage("Alignment line too short.")
            return

        line_angle = math.degrees(math.atan2(dy, dx))
        target = self._align_target_cb.currentText().lower()
        # The draw direction determines the target orientation:
        #   Vertical:   draw upward (dy<0) → tip faces up (-90°)
        #               draw downward (dy>0) → tip faces down (+90°)
        #   Horizontal: draw rightward (dx>0) → tip faces right (0°)
        #               draw leftward (dx<0) → tip faces left (180°)
        if target == "horizontal":
            target_angle = 0.0 if dx >= 0 else 180.0
        else:
            target_angle = -90.0 if dy <= 0 else 90.0
        rot = self._norm_angle_deg(line_angle - target_angle)
        if abs(rot) < 0.05:
            self.statusBar().showMessage("Image already aligned.")
            return
        self._save_undo("Align by line")
        try:
            self.detector.rotate_stack_by_angle(rot)
        except Exception as e:
            self._undo_stack.pop()  # discard snapshot on failure
            QMessageBox.critical(self, "Alignment error", str(e))
            return

        self._reset_after_stack_transform()
        direction = "CCW" if rot > 0 else "CW"
        self.statusBar().showMessage(
            f"Aligned by rotating {abs(rot):.2f} deg {direction}. Detections were cleared."
        )

    def detect_current(self):
        if self.detector.original is None:
            QMessageBox.warning(self, "No image", "Open an image first.")
            return
        self._save_undo("Detect current frame")
        try:
            params = self._get_detect_params()
            pre = self._current_pre_params()
            res = self.detector.detect(params, pre, self._get_roi_tuple())
            self._populate_table(res["properties"])
            self.set_view("contours")
            self._update_frame_ui()
            self.statusBar().showMessage(
                f"Frame {self.detector.current_frame_idx + 1}: "
                f"{len(res['contours'])} contour(s) detected."
            )
        except Exception as e:
            QMessageBox.critical(self, "Detection error", str(e))

    def detect_all(self):
        if self.detector.original is None:
            QMessageBox.warning(self, "No image", "Open an image first.")
            return
        n = self.detector.num_frames
        if n > 20:
            ans = QMessageBox.question(
                self, "Confirm", f"Run detection on all {n} frames?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return

        self._save_undo("Detect all frames")
        params = self._get_detect_params()
        roi = self._get_roi_tuple()
        orig_idx = self.detector.current_frame_idx

        prog = QProgressDialog(
            "Detecting contours on all frames...", "Cancel", 0, n, self
        )
        prog.setWindowTitle("Processing")
        prog.setWindowModality(Qt.WindowModal)

        total = 0
        try:
            for i in range(n):
                if prog.wasCanceled():
                    break
                self.detector.set_frame(i)
                pre = self._current_pre_params(i)
                r = self.detector.detect(params, pre, roi)
                total += len(r["contours"])
                prog.setValue(i + 1)
                prog.setLabelText(
                    f"Frame {i + 1} / {n} - {len(r['contours'])} contour(s)"
                )
                QApplication.processEvents()
        except Exception as e:
            prog.close()
            QMessageBox.critical(self, "Detection error", str(e))
            self.detector.set_frame(orig_idx)
            self._update_frame_ui()
            return

        prog.close()
        self.detector.set_frame(orig_idx)
        self._update_frame_ui()
        self.set_view("contours")
        QMessageBox.information(
            self, "Done",
            f"Detection complete on {n} frames.\nTotal contours: {total}",
        )
        self.statusBar().showMessage(f"All frames processed - total contours: {total}")

    # -- Frame navigation ------------------------------------------------------

    def _update_frame_ui(self):
        n = self.detector.num_frames
        idx = self.detector.current_frame_idx
        if n <= 0:
            self.frame_slider.setRange(0, 0)
            self.frame_slider.setValue(0)
            self.frame_info_lbl.setText("No image loaded")
            self.detect_info_lbl.setText("")
            return
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, n - 1))
        self.frame_slider.setValue(idx)
        self.frame_slider.blockSignals(False)
        self.frame_info_lbl.setText(f"Frame {idx + 1} / {n}")
        cached = self.detector.cache.get(idx)
        if cached:
            self.detect_info_lbl.setText(f"  {len(cached['contours'])} contour(s)")
            self.detect_info_lbl.setStyleSheet(f"color: {_GREEN}; font-weight: bold;")
        else:
            self.detect_info_lbl.setText("Not detected")
            self.detect_info_lbl.setStyleSheet(f"color: {_YELLOW};")

    def _on_frame_slider(self, val: int):
        if not self.detector.frames:
            return
        if val != self.detector.current_frame_idx:
            self.go_to_frame(val)

    def go_to_frame(self, idx: int):
        if not self.detector.frames:
            return
        idx = max(0, min(idx, self.detector.num_frames - 1))
        self.detector.set_frame(idx)

        # Restore per-frame pre params
        if idx in self.frame_pre_params:
            alpha, beta = self.frame_pre_params[idx]
            self.pre_alpha_sl.set_value(alpha)
            self.pre_beta_sl.set_value(beta)

        self.curv.clear()
        self.canvas.curve_data = None
        self.canvas.points_x = []
        self.canvas.points_y = []
        self.selected_contour_index = None

        cached = self.detector.cache.get(idx)
        if cached is None and self.current_view in ("contours", "binary"):
            self.current_view = "preprocessed"
            self.view_label.setText("View: Preprocessed")
            
        if cached is not None and "curve_data" in cached:
            self.canvas.curve_data = cached["curve_data"]
            self.canvas.points_x = cached.get("curve_points_x", [])
            self.canvas.points_y = cached.get("curve_points_y", [])
            self.curv.set_points(self.canvas.points_x, self.canvas.points_y)
            self.curv.curve_data = self.canvas.curve_data
            self.selected_contour_index = cached.get("selected_contour_index", None)

        self._update_frame_ui()
        self._populate_table([] if cached is None else cached["properties"])
        self._refresh_view()

    def prev_frame(self):
        self.go_to_frame(self.detector.current_frame_idx - 1)

    def next_frame(self):
        self.go_to_frame(self.detector.current_frame_idx + 1)

    # -- Table -----------------------------------------------------------------

    def _populate_table(self, props: List[Dict]):
        self.table.setRowCount(0)
        for i, p in enumerate(props):
            self.table.insertRow(i)
            cx, cy = p["centroid"]
            x, y, w, h = p["bbox"]
            vals = [
                str(i + 1), str(p["area"]), str(p["perimeter"]),
                f"({cx},{cy})", f"({x},{y},{w},{h})",
                str(p["circularity"]), str(p["aspect_ratio"]), str(p["solidity"]),
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j, item)

    def _on_table_select(self):
        rows = self.table.selectedItems()
        if not rows:
            self.selected_contour_index = None
        else:
            self.selected_contour_index = self.table.currentRow()
        self.canvas.selected_contour_idx = self.selected_contour_index
        self.canvas.update()

    # -- Curvature actions -----------------------------------------------------

    def _on_canvas_points(self, xs: list, ys: list):
        self.curv.set_points(xs, ys)

    def _on_vec_change(self):
        self.canvas.show_vectors = self.show_vec_cb.isChecked()
        self.canvas.vec_density = int(self.vec_density_sl.get_value())
        self.canvas.vec_len = float(self.vec_len_sl.get_value())
        self.canvas.update()

    def use_selected_contour(self):
        if self.selected_contour_index is None:
            QMessageBox.information(
                self, "Select a contour",
                "Select a contour row in the table first."
            )
            return
        cached = self.detector.cache.get(self.detector.current_frame_idx)
        if not cached:
            QMessageBox.information(self, "No contours", "Run detection first.")
            return
        contours = cached["contours"]
        idx = self.selected_contour_index
        if idx < 0 or idx >= len(contours):
            return
        pts = contours[idx].reshape(-1, 2)
        if len(pts) < 10:
            QMessageBox.information(
                self, "Too small",
                "Selected contour has too few points."
            )
            return
        xs = pts[:, 0].astype(float).tolist()
        ys = pts[:, 1].astype(float).tolist()
        self.curv.set_points(xs, ys)
        self.canvas.points_x = xs
        self.canvas.points_y = ys
        self.curv_msg_lbl.setText(
            f"Loaded contour #{idx + 1}  ({len(pts)} pts). "
            "Click 'Compute Curvature'."
        )
        self.canvas.update()

    def use_largest_contour(self):
        cached = self.detector.cache.get(self.detector.current_frame_idx)
        if not cached or not cached["contours"]:
            QMessageBox.information(self, "No contours", "Run detection first.")
            return
        contours = cached["contours"]
        idx = int(np.argmax([cv2.contourArea(c) for c in contours]))
        self.selected_contour_index = idx
        self.table.selectRow(idx)
        self.use_selected_contour()

    def clear_curvature(self):
        self._save_undo("Clear curvature")
        self.curv.clear()
        self.canvas.curve_data = None
        self.canvas.points_x = []
        self.canvas.points_y = []
        self.show_vec_cb.setChecked(False)
        self.canvas.show_vectors = False
        self.curv_msg_lbl.setText("Cleared. Select a contour or draw manually.")
        self.canvas.update()

    def compute_curvature(self):
        self._save_undo("Compute curvature")
        try:
            data = self.curv.compute(
                smooth=float(self.smooth_sl.get_value()),
                res_mult=float(self.res_sl.get_value()),
            )
            self.canvas.curve_data = data
            
            # Cache it for the current frame
            idx = self.detector.current_frame_idx
            cached = self.detector.cache.get(idx)
            if cached is not None:
                cached["curve_data"] = data
                cached["curve_points_x"] = list(self.canvas.points_x)
                cached["curve_points_y"] = list(self.canvas.points_y)
                cached["selected_contour_index"] = self.selected_contour_index

            self.curv_msg_lbl.setText(
                "Curvature computed. Hover near the curve to inspect R and k."
            )
            self.canvas.update()
        except Exception as e:
            QMessageBox.critical(self, "Curvature error", str(e))

    def compute_curvature_all(self):
        n = self.detector.num_frames
        if n == 0:
            return

        smooth_val = float(self.smooth_sl.get_value())
        res_mult_val = float(self.res_sl.get_value())

        prog = QProgressDialog("Computing curvature for all frames...", "Cancel", 0, n, self)
        prog.setWindowTitle("Processing")
        prog.setWindowModality(Qt.WindowModal)

        orig_idx = self.detector.current_frame_idx
        success_count = 0

        for i in range(n):
            if prog.wasCanceled():
                break
            cached = self.detector.cache.get(i)
            if not cached or not cached.get("contours"):
                # You might auto-detect here if desired, but for now we skip frames without contours
                continue

            contours = cached["contours"]
            # Find largest contour
            largest_idx = int(np.argmax([cv2.contourArea(c) for c in contours]))
            pts = contours[largest_idx].reshape(-1, 2)
            if len(pts) < 10:
                continue

            xs = pts[:, 0].astype(float).tolist()
            ys = pts[:, 1].astype(float).tolist()
            
            # Temporary curvature engine to not mess up current UI state until end
            from .curvature import CurvatureEngine
            temp_curv = CurvatureEngine()
            temp_curv.set_points(xs, ys)
            try:
                data = temp_curv.compute(smooth=smooth_val, res_mult=res_mult_val)
                cached["curve_data"] = data
                cached["curve_points_x"] = xs
                cached["curve_points_y"] = ys
                cached["selected_contour_index"] = largest_idx
                success_count += 1
            except Exception:
                pass

            prog.setValue(i + 1)
            QApplication.processEvents()

        prog.close()
        
        # Refresh current frame
        self.go_to_frame(orig_idx)
        QMessageBox.information(
            self, "Done", f"Curvature computed for {success_count} frames."
        )

    # -- Save / Export ---------------------------------------------------------

    def save_annotated(self):
        cached = self.detector.cache.get(self.detector.current_frame_idx)
        if not cached:
            QMessageBox.warning(self, "Nothing to save", "Run detection first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save annotated image", "",
            "PNG (*.png);;TIFF (*.tiff);;JPEG (*.jpg)"
        )
        if path:
            cv2.imwrite(path, cached["annotated"])
            self.statusBar().showMessage(f"Saved: {path}")

    def save_binary(self):
        cached = self.detector.cache.get(self.detector.current_frame_idx)
        if not cached:
            QMessageBox.warning(self, "Nothing to save", "Run detection first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save binary mask", "", "PNG (*.png);;TIFF (*.tiff)"
        )
        if path:
            cv2.imwrite(path, cached["binary"])
            self.statusBar().showMessage(f"Saved: {path}")

    def save_curvature_image(self):
        data = self.curv.curve_data
        if data is None:
            QMessageBox.warning(self, "Nothing to save", "Compute curvature first.")
            return
        base_cv = self._get_base_cv()
        if base_cv is None:
            QMessageBox.warning(self, "No image", "No image is currently loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save curvature image", "",
            "PNG (*.png);;TIFF (*.tiff);;JPEG (*.jpg)"
        )
        if not path:
            return
        img = cv2.cvtColor(base_cv, cv2.COLOR_GRAY2BGR) if base_cv.ndim == 2 else base_cv.copy()
        xs, ys = data["x"], data["y"]
        pts = np.vstack((xs, ys)).T.astype(np.int32)
        cv2.polylines(img, [pts], False, (255, 255, 0), 2)
        if self.show_vec_cb.isChecked():
            step = int(self.vec_density_sl.get_value())
            vlen = float(self.vec_len_sl.get_value())
            tx, ty = data["tx"], data["ty"]
            nx_a, ny_a = data["nx"], data["ny"]
            for i in range(0, len(xs), step):
                vx, vy = float(xs[i]), float(ys[i])
                t1 = (int(vx - tx[i] * vlen), int(vy - ty[i] * vlen))
                t2 = (int(vx + tx[i] * vlen), int(vy + ty[i] * vlen))
                cv2.line(img, t1, t2, (255, 0, 255), 1, cv2.LINE_AA)
                n2 = (int(vx + nx_a[i] * vlen), int(vy + ny_a[i] * vlen))
                cv2.arrowedLine(img, (int(vx), int(vy)), n2,
                                (0, 255, 255), 1, cv2.LINE_AA, 0, 0.2)
        cv2.imwrite(path, img)
        self.statusBar().showMessage(f"Saved: {path}")

    def export_csv_current(self):
        cached = self.detector.cache.get(self.detector.current_frame_idx)
        if not cached or not cached["properties"]:
            QMessageBox.warning(self, "No data", "No properties to export for this frame.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV (current frame)", "", "CSV (*.csv)"
        )
        if path:
            self._write_csv(path, {self.detector.current_frame_idx: cached["properties"]})
            self.statusBar().showMessage(f"Exported CSV: {path}")

    def export_csv_all(self):
        if not self.detector.cache:
            QMessageBox.warning(self, "No data", "No detection results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV (all frames)", "", "CSV (*.csv)"
        )
        if not path:
            return
        frame_props = {fi: r["properties"] for fi, r in self.detector.cache.items()
                       if r.get("properties")}
        if not frame_props:
            QMessageBox.warning(self, "No data", "No frames have contour properties.")
            return
        self._write_csv(path, frame_props)
        total = sum(len(v) for v in frame_props.values())
        self.statusBar().showMessage(f"Exported {total} contours to CSV: {path}")

    @staticmethod
    def _write_csv(path: str, frame_props: Dict[int, List[Dict]]):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Frame", "Contour_ID", "Area", "Perimeter",
                        "Centroid_X", "Centroid_Y",
                        "BBox_X", "BBox_Y", "BBox_W", "BBox_H",
                        "Circularity", "Aspect_Ratio", "Solidity"])
            for fi in sorted(frame_props):
                for i, p in enumerate(frame_props[fi], 1):
                    cx, cy = p["centroid"]
                    bx, by, bw, bh = p["bbox"]
                    w.writerow([fi + 1, i, p["area"], p["perimeter"],
                                cx, cy, bx, by, bw, bh,
                                p["circularity"], p["aspect_ratio"], p["solidity"]])

    def export_contours_current(self):
        cached = self.detector.cache.get(self.detector.current_frame_idx)
        if not cached or not cached.get("contours"):
            QMessageBox.warning(self, "No data", "No contours to export for this frame.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Contours (current frame)", "", "CSV (*.csv)"
        )
        if path:
            self._write_contours_csv(
                path, {self.detector.current_frame_idx: cached["contours"]}
            )
            self.statusBar().showMessage(f"Exported contours: {path}")

    def export_contours_all(self):
        if not self.detector.cache:
            QMessageBox.warning(self, "No data", "No detection results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Contours (all frames)", "", "CSV (*.csv)"
        )
        if not path:
            return
        frame_contours = {fi: r["contours"] for fi, r in self.detector.cache.items()
                          if r.get("contours")}
        if not frame_contours:
            QMessageBox.warning(self, "No data", "No frames have contours.")
            return
        self._write_contours_csv(path, frame_contours)
        self.statusBar().showMessage(f"Exported contours to CSV: {path}")

    @staticmethod
    def _write_contours_csv(path: str, frame_contours: Dict[int, List[np.ndarray]]):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Frame", "Contour_ID", "Point_Index", "X", "Y"])
            for fi in sorted(frame_contours):
                for i, cnt in enumerate(frame_contours[fi], 1):
                    pts = cnt.reshape(-1, 2)
                    for j, (x, y) in enumerate(pts):
                        w.writerow([fi + 1, i, j, x, y])

    def export_curvature_current(self):
        cached = self.detector.cache.get(self.detector.current_frame_idx)
        if not cached or "curve_data" not in cached:
            QMessageBox.warning(self, "No data", "No curvature data to export for this frame.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Curvature (current frame)", "", "CSV (*.csv)"
        )
        if path:
            self._write_curvature_csv(
                path, {self.detector.current_frame_idx: cached["curve_data"]}
            )
            self.statusBar().showMessage(f"Exported curvature: {path}")

    def export_curvature_all(self):
        if not self.detector.cache:
            QMessageBox.warning(self, "No data", "No detection results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Curvature (all frames)", "", "CSV (*.csv)"
        )
        if not path:
            return
        frame_curves = {fi: r["curve_data"] for fi, r in self.detector.cache.items()
                        if "curve_data" in r}
        if not frame_curves:
            QMessageBox.warning(self, "No data", "No frames have curvature data.")
            return
        self._write_curvature_csv(path, frame_curves)
        self.statusBar().showMessage(f"Exported curvature to CSV: {path}")

    @staticmethod
    def _write_curvature_csv(path: str, frame_curves: Dict[int, Dict[str, np.ndarray]]):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Frame", "Point_Index", "X", "Y", "Curvature_k", "Radius_R", "Tangent_X", "Tangent_Y", "Normal_X", "Normal_Y"])
            for fi in sorted(frame_curves):
                data = frame_curves[fi]
                for j in range(len(data["x"])):
                    w.writerow([
                        fi + 1, j, 
                        data["x"][j], data["y"][j],
                        data["k"][j], data["r"][j],
                        data["tx"][j], data["ty"][j],
                        data["nx"][j], data["ny"][j]
                    ])

    def _about(self):
        QMessageBox.information(
            self, "About - Tip Curvature Analyzer v2",
            "TIFF Contour + Curvature Analyzer\n\n"
            "-¢ Preprocessing tab - live contrast/blur preview\n"
            "-¢ Contour tab - multiple threshold methods, CSV export\n"
            "-¢ Curvature tab - spline fitting, hover inspection\n\n"
            "Keyboard shortcuts:\n"
            "  Ctrl+O  Open image\n"
            "  Ctrl+S  Save annotated\n"
            "  F       Fit to view\n"
            "Canvas controls:\n"
            "  Scroll wheel  Zoom\n"
            "  Right-drag    Pan\n"
            "  Left-drag     Draw (when enabled)\n"
        )


def run():
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

