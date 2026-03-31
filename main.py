#!/usr/bin/env python3
"""Standalone entry point: python main.py (from inside the package dir)."""
import sys
from pathlib import Path

_pkg = Path(__file__).resolve().parent
if str(_pkg.parent) not in sys.path:
    sys.path.insert(0, str(_pkg.parent))

from TipCurvatureAppV2.app import run  # noqa: E402

if __name__ == "__main__":
    run()
