# TipCurvature

GUI for measuring the radius of curvature of nanotips from TEM images (single frames or TIFF stacks).

The tip contour is detected via thresholding (Otsu, binary, adaptive, or Canny edge), with optional Gaussian blur and morphological closing. The curvature is then computed along the detected contour using B-spline interpolation, giving the local curvature κ and radius R = 1/κ at each point.

## Requirements

- Python 3.8+
- numpy, scipy, opencv-python, Pillow, tifffile, PyQt5

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```
