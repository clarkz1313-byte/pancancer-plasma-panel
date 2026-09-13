#!/usr/bin/env python3
"""
fig_candidate_common.py — shared pieces for the candidate figures 3-5, 3-6,
4-5 and 4-6, so the four read as one family and a change lands on all of them.

Three things live here.

1. A lifted expression colormap. fig4_panels runs its diverging scale from
   #4d4d4d, close to black, which sits heavily in a manuscript otherwise built
   from bright saturated hues on white. The candidates fade the depleted end
   and strengthen the elevated end instead, widening the asymmetry so that
   "above average in this cancer" is the thing the eye lands on.

2. Save-with-measurement. A figure saved with bbox_inches="tight" has an
   aspect ratio that cannot be predicted from figsize, and the mosaic builder
   scales panels by width, so two panels in one row end up neither the same
   height nor aligned on their axes. `save_measured` records where the
   reference axes' baseline sits inside the saved image; `align_images` then
   pads each image with white so that every panel shares one aspect ratio AND
   one baseline position. After compositing, the x axes line up.

3. The per-cancer expression tile, factored out of fig4_6 so Figure 3-6 can
   draw the same tile in its own layout.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
MAIN_DATA = ROOT / "data" / "processed" / "filtered_pancancer_data.csv"
PANELS_CSV = (ROOT / "revision_package_2026-09-04" / "05_data_audit"
              / "ora_assay_background_2026-09-05" / "panel_memberships.csv")
RANK_ROOT = ROOT / "revise_plan" / "part_a_single"

CANCER_ORDER = ["AML", "BRC", "CLL", "CRC", "CVX", "ENDC",
                "GLIOM", "LUNGC", "LYMPH", "MYEL", "OVC", "PRC"]
PLOT_COLOR = {
    "AML": "#b22222", "BRC": "#c97b63", "CLL": "#7a3e9d",
    "CRC": "#d68600", "CVX": "#c13d86", "ENDC": "#8e5d2c",
    "GLIOM": "#1a8db8", "LUNGC": "#2e8b57", "LYMPH": "#3856a6",
    "MYEL": "#8c564b", "OVC": "#d1495b", "PRC": "#008b8b",
}
SLUG_KEY = {"LYMPH": "single_DLBCL"}
VMAX = 2.0

# ── 1. lifted expression scale ───────────────────────────────────────────────
# Depleted end, shared across all twelve tiles so that "below average" always
# reads the same. Faded from fig4_panels' #4d4d4d / #b0b0b0.
LOW_DARK = "#979a9e"
LOW_MID = "#cfd1d4"
NEUTRAL = "#efeef2"
# Multiclass purple, deepened from #6a3d9a so the Part B tile carries the same
# asymmetry as the per-cancer tiles.
PARTB_HIGH = "#5b2d8e"
PARTB_MID = "#a97fd0"


def _bold(colour: str, frac: float = 1.0) -> tuple:
    """Blend toward white; frac = 1.0 leaves the hue at full strength."""
    rgba = np.array(mcolors.to_rgba(colour))
    return tuple((rgba[:3] * frac + (1 - frac)).clip(0, 1))


def cancer_cmap(colour: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f"cand_{colour}",
        [(0.00, LOW_DARK), (0.28, LOW_MID), (0.50, NEUTRAL),
         (0.72, _bold(colour, 0.62)), (1.00, _bold(colour, 1.0))],
        N=256,
    )


PARTB_CMAP = LinearSegmentedColormap.from_list(
    "cand_partb",
    [(0.00, LOW_DARK), (0.28, LOW_MID), (0.50, NEUTRAL),
     (0.72, PARTB_MID), (1.00, PARTB_HIGH)],
    N=256,
)


# ── 2. save with a measured baseline, then align ─────────────────────────────
def save_measured(fig, path: Path, ref_ax, dpi: int = 400,
                  pad_inches: float = 0.03) -> None:
    """Save `fig`, and record where `ref_ax`'s bottom sits in the saved image."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    axes_box = (ref_ax.get_window_extent(renderer)
                .transformed(fig.dpi_scale_trans.inverted()))
    height = tight.height + 2 * pad_inches
    frac = ((axes_box.y0 - tight.y0) + pad_inches) / height
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight",
                pad_inches=pad_inches)
    path.with_suffix(".axis.json").write_text(
        json.dumps({"baseline_frac_from_bottom": float(frac)}), encoding="utf-8")


def measure_baseline(path: Path, dark: int = 150, min_frac: float = 0.40,
                     search_from: float = 0.30) -> float | None:
    """Find the axes baseline in a saved panel, in pixels, by looking at it.

    The analytic estimate from `save_measured` is computed before savefig
    recomputes its own tight bbox, and a figure-level legend placed outside the
    axes is not always inside the bbox matplotlib reports, so the two can
    disagree by a few per cent -- enough to leave a visible step between two
    panels in a mosaic row. Scanning the rendered pixels for the lowest long
    dark horizontal run finds the bottom spine directly, which is the line a
    reader actually sees.
    """
    img = np.asarray(Image.open(path).convert("L"))
    height, width = img.shape
    start = int(height * search_from)
    best_row = None
    for offset in range(img.shape[0] - start):
        mask = img[start + offset] < dark
        if not mask.any():
            continue
        # longest CONTIGUOUS dark run. A count of dark pixels is not enough:
        # a legend row of short line handles plus glyph strokes can total a
        # third of the width while containing no line at all.
        longest = best = 0
        for value in mask:
            best = best + 1 if value else 0
            longest = max(longest, best)
        if longest > min_frac * width:
            best_row = start + offset
    if best_row is None:
        return None
    return 1.0 - best_row / height                       # fraction from bottom


def align_images(paths: list[Path], out_paths: list[Path]) -> float:
    """Pad each image with white to a common aspect AND baseline fraction.

    The mosaic compositor scales a panel to its cell width, so two panels in
    one row share a displayed height only if they share an aspect ratio; and
    they share an axis line only if the baseline sits at the same fraction of
    that height. Padding solves both without cropping anything.
    """
    imgs = [Image.open(p).convert("RGB") for p in paths]
    fracs = []
    for p in paths:
        measured = measure_baseline(p)
        if measured is None:
            measured = json.loads(
                p.with_suffix(".axis.json").read_text(encoding="utf-8")
            )["baseline_frac_from_bottom"]
            print(f"  {p.name}: baseline from metadata {measured:.4f}")
        else:
            print(f"  {p.name}: baseline measured {measured:.4f}")
        fracs.append(measured)

    aspect = min(im.width / im.height for im in imgs)
    for _ in range(60):
        heights = [im.width / aspect for im in imgs]
        lows = [f * im.height / H for f, im, H in zip(fracs, imgs, heights)]
        highs = [(H - im.height + f * im.height) / H
                 for f, im, H in zip(fracs, imgs, heights)]
        target = max(lows)
        if target <= min(highs) + 1e-9:
            break
        aspect *= 0.97          # make every panel taller, then retry
    else:
        raise RuntimeError("could not find a common baseline fraction")

    for im, frac, height, out in zip(imgs, fracs, heights, out_paths):
        H = int(round(height))
        pad_bottom = int(round(target * H - frac * im.height))
        pad_bottom = max(0, min(pad_bottom, H - im.height))
        canvas = Image.new("RGB", (im.width, H), (255, 255, 255))
        canvas.paste(im, (0, H - im.height - pad_bottom))
        canvas.save(out)
        print(f"  aligned {out.name}: {im.width}x{im.height} -> "
              f"{im.width}x{H} (aspect {im.width / H:.4f}, "
              f"baseline {target:.4f})")
    return target


# ── 3. the per-cancer expression tile ────────────────────────────────────────
def deployed_members() -> dict[str, list[str]]:
    """Frozen panel membership, ordered by that cancer's own L1 importance."""
    raw: dict[str, list[str]] = {}
    for row in csv.DictReader(open(PANELS_CSV, encoding="utf-8-sig")):
        values = list(row.values())
        raw.setdefault(values[0].strip(), []).append(values[1].strip())

    out = {}
    for cancer in CANCER_ORDER:
        prots = raw[SLUG_KEY.get(cancer, f"single_{cancer}")]
        rk = pd.read_csv(RANK_ROOT / cancer.lower() / "tables"
                         / f"{cancer.lower()}_top_feature_ranking.csv"
                         ).set_index("protein")
        imp = {p: float(rk.loc[p, "importance"]) if p in rk.index else 0.0
               for p in prots}
        out[cancer] = sorted(prots, key=lambda p: -imp[p])
    return out


def zscore_matrix(members: dict[str, list[str]]) -> pd.DataFrame:
    uniq, seen = [], set()
    for cancer in CANCER_ORDER:
        for p in members[cancer]:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
    raw = pd.read_csv(MAIN_DATA, usecols=["Cancer"] + uniq)
    mean_npx = raw.groupby("Cancer")[uniq].mean().loc[CANCER_ORDER]
    return (mean_npx - mean_npx.mean()) / mean_npx.std().replace(0, 1)


def cell_gridlines(ax, n_rows: int, n_cols: int) -> None:
    for x in range(n_cols + 1):
        ax.axvline(x - 0.5, color="black", lw=0.4, alpha=0.10, zorder=3)
    for y in range(n_rows + 1):
        ax.axhline(y - 0.5, color="black", lw=0.4, alpha=0.10, zorder=3)


def draw_expression_tile(ax, fig, cancer, prots, z, display_label,
                         label_size=8.0, tick_size=8.0, cbar=True,
                         cbar_label=True):
    """One cancer's Figure 4C tile, on the lifted colour scale."""
    colour = PLOT_COLOR[cancer]
    n = len(prots)
    z_mat = z.loc[CANCER_ORDER, prots].T.values          # (n_prots, 12)

    im = ax.imshow(z_mat, aspect="auto", cmap=cancer_cmap(colour),
                   vmin=-VMAX, vmax=VMAX, interpolation="nearest")
    cell_gridlines(ax, n, 12)

    ax.set_yticks(range(n))
    ax.set_yticklabels(prots, fontsize=label_size, fontweight="bold")
    ax.set_xticks(range(12))
    ax.set_xticklabels([display_label(c) for c in CANCER_ORDER],
                       rotation=62, ha="right", fontsize=tick_size)
    target = CANCER_ORDER.index(cancer)
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(PLOT_COLOR[CANCER_ORDER[i]])
        if i == target:
            tick.set_fontweight("bold")
            tick.set_fontsize(tick_size * 1.25)
    ax.add_patch(Rectangle((target - 0.5, -0.5), 1, n, fill=False,
                           edgecolor=colour, lw=1.8, zorder=5))

    if cbar:
        cax = ax.inset_axes([0.0, 1.06, 0.46, 0.055])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks([-2.0, 0.0, 2.0])
        cb.ax.set_xticklabels(["−2", "0", "+2"], fontsize=tick_size,
                              fontweight="bold")
        cb.ax.tick_params(labelsize=tick_size, length=2.0, pad=1.5)
        cb.ax.xaxis.set_ticks_position("top")
        if cbar_label:
            ax.text(0.50, 1.11, "mean NPX $z$", color="#333333",
                    fontsize=tick_size, transform=ax.transAxes, ha="left",
                    va="center")

    ax.tick_params(length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(0.8)
    return im
