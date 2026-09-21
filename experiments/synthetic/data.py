"""Compatibility shim for historical tests; not part of the canonical study."""

import runpy
from pathlib import Path

_LEGACY = Path(__file__).resolve().parents[1] / "experiments/synthetic/data.py"
make_split = runpy.run_path(str(_LEGACY))["make_split"]
