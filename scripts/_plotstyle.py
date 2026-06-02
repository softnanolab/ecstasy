"""Shared matplotlib style for the benchmark figures: CMU Concrete typeface.

The CMU Concrete OTFs live in ~/.fonts but aren't in matplotlib's default scan,
so we register them explicitly and set them as the default family. Math text is
switched to Computer Modern so equations/sub-/superscripts match the body.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm

# CMU Concrete OTFs (cmunorm/cmunobx/cmunoti/cmunobi). Looked up in $CMU_FONT_DIR if
# set, else the running user's ~/.fonts — no hardcoded home. Degrades gracefully
# (returns False) if the font isn't installed.
_FONT_DIR = os.environ.get("CMU_FONT_DIR", str(Path.home() / ".fonts"))


def use_cmu_concrete() -> bool:
    """Register CMU Concrete and make it the default font. Returns True on success."""
    files = glob.glob(os.path.join(_FONT_DIR, "cmuno*.otf"))
    for f in files:
        try:
            fm.fontManager.addfont(f)
        except Exception:        # noqa: BLE001
            pass
    if not any("Concrete" in f.name for f in fm.fontManager.ttflist):
        return False
    matplotlib.rcParams["font.family"] = "CMU Concrete"
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["axes.unicode_minus"] = False
    return True
