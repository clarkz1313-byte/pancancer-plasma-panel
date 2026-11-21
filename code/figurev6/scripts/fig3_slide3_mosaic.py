#!/usr/bin/env python3
"""
fig3_slide3_mosaic.py — slide 3 as a SINGLE deck-ready mosaic.

Slide 3 shows the locked-25 derivation as five columns that were previously
rendered as five separate PNGs and placed by hand in Canva:

  a  per-cancer volcano plots            (12 stacked tiles)
  b  Bonferroni fold-change box plots    (12 stacked tiles)
  c  Part A LASSO rank profiles          (12 stacked tiles)
  d  Part B LR-L2 selection curve        (one tall panel, 180 candidates)
  e  locked-25 x cancer membership grid  (one tall panel)

The alignment that matters on this slide is HORIZONTAL: columns a, b and c
are each a 12-row stack in the same CANCER_ORDER, so a reader should be able
to track one cancer straight across -- its volcano, its fold-change spread,
its LASSO profile. Rendering them as three independent figures cannot
guarantee that: each sets its own margins (a and b use left=0.12, c uses
left=0.14) and its own internal hspace, so the AML row in column c does not
land at the same height as the AML row in column a.

This builder puts a, b and c inside ONE outer GridSpec column each, every
one subdivided by the SAME 12-row subgridspec, so cancer rows are aligned by
construction. Columns d and e are full-height single axes beside them. Only
column a carries the per-cancer name (the row is then labelled once, not
three times), and the five column headings replace five separate figure
titles.

Tile drawing is delegated to `_volcano_ax` / `_bonf_ax` / `_lasso_ax` and to
`panel_d` / `panel_e`'s own logic in fig3_panels.py -- nothing is
re-implemented here, so a style fix to the standalone panels also lands on
this mosaic.

Output: figurev5/output/fig3_slide3_derivation_mosaic.pdf/.png
Run:    python fig3_slide3_mosaic.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

import fig3_panels as F
from fig3_panels import (
    CANCER_ORDER, PLOT_COLOR, OUT, DPI,
    VOLCANO_SRC, PART_B_DEA, SUPP_S1,
    _volcano_ax, _bonf_ax, _lasso_ax,
)
from figure_labels import display_label

# Column widths in inches. a/b/c are the 12-row stacks; d and e are the two
# single-axes selection panels and need more room for their protein labels.
COL_W = {"a": 2.40, "b": 2.00, "c": 2.20, "d": 3.95, "e": 3.45}
COL_ORDER = ["a", "b", "c", "d", "e"]

GRID_H      = 15.6   # height of the tile grid itself
LEFT_PAD    = 0.30
RIGHT_PAD   = 0.20
TOP_PAD     = 0.42   # panel letters only; detail belongs in the caption
BOTTOM_PAD  = 1.22

WSPACE_OUTER = 0.18   # between the five columns
HSPACE_INNER = 0.42   # between the 12 cancer rows (same for a, b and c)

def build_slide3_mosaic(stem="fig3_slide3_derivation_mosaic"):
    fig_w = LEFT_PAD + sum(COL_W[k] for k in COL_ORDER) \
            + WSPACE_OUTER * np.mean(list(COL_W.values())) * (len(COL_ORDER) - 1) \
            + RIGHT_PAD
    fig_h = TOP_PAD + GRID_H + BOTTOM_PAD

    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(
        1, len(COL_ORDER),
        left=LEFT_PAD / fig_w, right=1 - RIGHT_PAD / fig_w,
        top=1 - TOP_PAD / fig_h, bottom=BOTTOM_PAD / fig_h,
        wspace=WSPACE_OUTER,
        width_ratios=[COL_W[k] for k in COL_ORDER],
    )

    # One shared 12-row subdivision spec, applied identically to a, b and c
    # -- this is what makes the cancer rows line up across the three columns.
    def _stack(col_idx):
        return outer[0, col_idx].subgridspec(12, 1, hspace=HSPACE_INNER)

    src = pd.read_csv(VOLCANO_SRC)
    dea = pd.read_csv(PART_B_DEA,
                      usecols=["target_class", "significant_bonf", "fold_change"])
    s1  = pd.read_csv(SUPP_S1)

    gs_a, gs_b, gs_c = _stack(0), _stack(1), _stack(2)

    for r, cancer in enumerate(CANCER_ORDER):
        is_bottom = (r == 11)
        is_top    = (r == 0)

        # column a keeps the cancer name; b and c inherit it by row position
        ax = fig.add_subplot(gs_a[r, 0])
        _volcano_ax(ax, src, cancer, is_bottom, is_top, show_label=True,
                    show_protein_labels=False)

        ax = fig.add_subplot(gs_b[r, 0])
        _bonf_ax(ax, dea, cancer, is_bottom, show_label=False)

        ax = fig.add_subplot(gs_c[r, 0])
        _lasso_ax(ax, s1, cancer, is_bottom, is_top, show_label=False)

    # ---- columns d and e: full-height single axes ----
    # Order verified against Proteomics-canva14.pdf p.3: the membership dot
    # matrix is the FOURTH column and the Part B ranking curve is the FIFTH.
    # (fig_interpretation.md's letters are right; the file suffixes are not --
    # fig3e_* is the membership matrix, fig3d_* is the ranking curve.)
    ax_d = fig.add_subplot(outer[0, 3])
    F.draw_locked25_membership(ax_d)

    ax_e = fig.add_subplot(outer[0, 4])
    F.draw_partb_rank_profile(ax_e)
    # ---- concise panel identifiers; detail remains in the caption ----
    for i, key in enumerate(COL_ORDER):
        pos = outer[0, i].get_position(fig)
        fig.text((pos.x0 + pos.x1) / 2, 1 - 0.07 / fig_h,
                 key.upper(),
                 ha="center", va="top", fontsize=13.5, fontweight="bold",
                 color="#222222")

    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI, facecolor="white")
        print(f"  saved -> {p}  ({fig_w:.1f} x {fig_h:.1f} in)")
    plt.close(fig)


if __name__ == "__main__":
    print("Building slide 3 — locked-25 derivation mosaic ...")
    build_slide3_mosaic()
    print("\nDone.")
