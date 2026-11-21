"""Shared style for the v7 figure set: the v6.x look (figurev6 palette and cancer order, DejaVu Sans bold,
grey tile spines, faint cancer-colour tile backgrounds, panel letters only, one consolidated legend).
Imports the palette from figurev6/scripts/fig89_common.py so colours stay identical to v6.x.
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[3]))
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

ROOT = PROJECT_ROOT
V7 = ROOT / "v7_plan"
RES = V7 / "results"
OUT = V7 / "figures/output"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "figurev6/scripts"))

# v7 colour rule: purple belongs to the 26-protein list only, in one shade. The v6 builders hard-code CLL as purple
# (#7a3e9d) and use several other purples, so their source is recoloured as it is loaded (import, inspect.getsource
# and Path.read_text, which the v7 builders use to patch v6 code in memory). The v6 files on disk are not changed.
PURPLE = "#6a3d9a"
CLL_COLOR = "#8a9a1e"      # olive, in place of the v6 purple
GO_COLOR = "#3f6fb0"       # GO database colour in the annotation panels (v6: purple)
BRC_COLOR, CVX_COLOR = "#c13d86", "#c97b63"   # swapped (author, 17 Oct): v6 BRC salmon sat too close to CRC amber
RECOLOR = {"#7a3e9d": CLL_COLOR, "#a06cd5": PURPLE, "#9467bd": PURPLE, "#7b3f9e": GO_COLOR,
           "#c97b63": BRC_COLOR, "#c13d86": CVX_COLOR}
_RX = re.compile("|".join(RECOLOR), re.IGNORECASE)
V6_DIRS = [ROOT / "figurev6/scripts", ROOT / "overleaf-tools"]


def recolor(src):
    return _RX.sub(lambda m: RECOLOR[m.group(0).lower()], src)


def _in_v6(p):
    try:
        p = Path(p).resolve()
    except OSError:
        return False
    return p.suffix == ".py" and any(d.resolve() in p.parents for d in V6_DIRS)


class _V6Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):   # always from recoloured source, never from a cached .pyc
        with open(self.path, encoding="utf-8") as fh:   # not Path.read_text: that is patched below and would
            src = fh.read()                              # recolour twice, which undoes the BRC/CVX swap
        return compile(recolor(src), self.path, "exec", dont_inherit=True)


class _V6Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if "." in name:
            return None
        for d in V6_DIRS:
            p = d / f"{name}.py"
            if p.exists():
                return importlib.util.spec_from_file_location(name, p, loader=_V6Loader(name, str(p)))
        return None


if not any(isinstance(f, _V6Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _V6Finder())
    _read_text, _getsource = Path.read_text, inspect.getsource
    Path.read_text = lambda self, *a, **k: (recolor(_read_text(self, *a, **k)) if _in_v6(self)
                                            else _read_text(self, *a, **k))
    inspect.getsource = lambda obj: (recolor(_getsource(obj)) if _in_v6(inspect.getsourcefile(obj) or "")
                                     else _getsource(obj))

import fig89_common as FC  # noqa: E402

ORDER = list(FC.CANCER_ORDER)                       # AML ... PRC, with LYMPH for DLBCL
NAME = {c: ("DLBCL" if c == "LYMPH" else c) for c in ORDER}
COL = dict(FC.PLOT_COLOR)
COL["DLBCL"] = COL["LYMPH"]
ORDER_D = [NAME[c] for c in ORDER]                  # display order with DLBCL

PURPLE_DARK, PURPLE_LIGHT = PURPLE, "#cbb6e3"   # one purple for the 26-protein list; the light one is for tints only
MACRO = PURPLE
assert (FC.PLOT_COLOR["CLL"], FC.PLOT_COLOR["BRC"], FC.PLOT_COLOR["CVX"], FC.MACRO_COLOR) == \
    (CLL_COLOR, BRC_COLOR, CVX_COLOR, PURPLE), "v6 recolouring hook not active"
GREY, INK, SPINE = "#6b6b6b", "#3a3a3a", "#666666"
GAIN, LOSS = "#2e7d4f", "#b0413e"
FS = dict(title=26, tick=17, label=17, letter=30, group=23, legend=18, annot=16)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.weight": "bold", "axes.labelweight": "bold",
                     "axes.titleweight": "bold"})


def frame(ax, fs=None):
    for sp in ax.spines.values():
        sp.set_color(SPINE); sp.set_linewidth(0.9)
    ax.tick_params(labelsize=fs or FS["tick"], length=4, pad=3)


def tint(c, a=0.10):
    r, g, b = mcolors.to_rgb(COL.get(c, c))
    return (1 - a) + a * r, (1 - a) + a * g, (1 - a) + a * b


def letter(fig, ax, s, dx=-0.012, dy=0.012):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, s, fontsize=FS["letter"], fontweight="bold", ha="right", va="bottom")


def save(fig, name, dpi=300):
    p = OUT / name
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)
    return p
