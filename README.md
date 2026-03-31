# TipCurvatureAppV2

A PyQt5-based GUI application for analyzing the curvature of TEM (Transmission Electron Microscopy) tip images. This is a rewrite of the original Tkinter-based TipCurvatureApp, featuring a dark theme, improved canvas controls, and live parameter updates.

## Features

- Load and navigate multi-frame TIFF stacks
- Interactive image canvas with zoom (mouse wheel) and pan (right-drag)
- Histogram widget with linear/log scale toggle
- Live tip detection and curvature analysis with adjustable parameters
- Dark theme (Catppuccin Mocha)

## Requirements

- Python 3.8+
- numpy
- opencv-python
- Pillow
- scipy
- tifffile
- PyQt5

Install dependencies:

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda create -n tipcurvature python=3.10
conda activate tipcurvature
pip install -r requirements.txt
```

## Usage

From inside the `TipCurvatureAppV2` directory:

```bash
python main.py
```

Or from the parent directory:

```bash
python -m TipCurvatureAppV2
```
