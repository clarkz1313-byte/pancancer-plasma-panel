#!/usr/bin/env python3
"""
fig3_6_structure_with_sweep.py

Candidate Figure 3-6: Figure 4's layout with its concordance scatter replaced.

  A  the multiclass size sweep (Figure 3-5A)
  B  Figure 4's grouped expression heatmap of the 25 locked markers
  C  the twelve disease-specific expression tiles

Sweep first, so the figure reads in the order the work happened: why there are
25 markers, then what those 25 markers look like, then the twelve
disease-specific panels. What it gives up is Figure 4's Spearman concordance
scatter, the paper's evidence that multiclass importance and single-cancer
importance are different quantities; that would have to move to the text or to
a supplement.

Two things this builder does beyond compositing
-----------------------------------------------
1. Panels B and C are re-rendered on the candidate colour scale (faded
   depleted end, full-strength cancer and multiclass hues) rather than reused
   from fig4_panels' output, so the whole figure shares one colour language.
   fig4_panels itself is NOT modified: its PARTB_CMAP is swapped in memory for
   the duration of the call, and the result is written under a candidate name,
   so Figure 4 is untouched and the change stays reversible.

2. The top-row x axes are aligned. build_slide_mosaics scales each panel to its
   cell width and preserves aspect, so two panels share a height only if their
   saved images share an aspect ratio, and share an axis line only if the
   baseline sits at the same fraction of that height. Both panels record where
   their axes' baseline landed, and fig_candidate_common.align_images pads each
   with white until both agree. Nothing is cropped.

Run:    python fig3_5_and_fig4_5_panel_size_selection.py   (writes fig3_6a)
        python fig3_6_structure_with_sweep.py
Output: figurev6/output/ and overleaf-source-git/figures/
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fig4_panels as f4
from build_slide_mosaics import OUT, build
from fig_candidate_common import (
    CANCER_ORDER, PARTB_CMAP, align_images, cancer_cmap, deployed_members,
    draw_expression_tile, save_measured, zscore_matrix,
)
from figure_labels import display_label

ROOT = Path(__file__).resolve().parents[2]
OVERLEAF = ROOT / "overleaf-source-git" / "figures"

STEM = "fig3_6_structure_with_size_sweep"
SWEEP = "fig3_6a_multiclass_size_sweep"
PARTB = "fig3_6b_partb_expression_candidate"
TILES = "fig3_6c_parta_expression_candidate"


def build_partb_panel() -> None:
    """fig4_panels.panel_a on the candidate colour scale, under a new name.

    panel_a draws a four-row GridSpec (colorbar, importance bars, cancer
    stripe, dendrogram + heatmap) that is not worth duplicating. Swapping the
    module-level colormap and intercepting its save call reuses all of it
    without editing the file.
    """
    original_cmap, original_save = f4.PARTB_CMAP, f4.save
    captured = {}

    def capture(fig, stem):
        # the heatmap is the largest axes in the figure
        ax_hm = max(fig.axes, key=lambda a: a.get_position().size.prod())
        captured["fig"], captured["ax"] = fig, ax_hm

    try:
        f4.PARTB_CMAP = PARTB_CMAP
        f4.save = capture
        f4.panel_a()
    finally:
        f4.PARTB_CMAP, f4.save = original_cmap, original_save

    save_measured(captured["fig"], OUT / f"{PARTB}.png", captured["ax"], dpi=400)
    plt.close(captured["fig"])


def build_tiles_panel() -> None:
    """The twelve disease-specific tiles as one 2x6 strip, candidate colours."""
    members = deployed_members()
    z = zscore_matrix(members)

    fig, axes = plt.subplots(2, 6, figsize=(24.0, 13.4), squeeze=False)
    fig.subplots_adjust(hspace=0.70, wspace=0.34,
                        left=0.035, right=0.99, top=0.94, bottom=0.14)
    for ax, cancer in zip(axes.ravel(), CANCER_ORDER):
        draw_expression_tile(ax, fig, cancer, members[cancer], z,
                             display_label, label_size=11.0, tick_size=11.0,
                             cbar_label=False)
        ax.text(0.52, 1.105, f"{display_label(cancer)}  n={len(members[cancer])}",
                color=f4.PLOT_COLOR[cancer], fontweight="bold", fontsize=15,
                transform=ax.transAxes, ha="left", va="center")
    fig.savefig(OUT / f"{TILES}.png", dpi=400, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not (OUT / f"{SWEEP}.png").exists():
        raise SystemExit("run fig3_5_and_fig4_5_panel_size_selection.py first")

    build_partb_panel()
    build_tiles_panel()

    aligned = [f"{SWEEP}_aligned", f"{PARTB}_aligned"]
    align_images([OUT / f"{SWEEP}.png", OUT / f"{PARTB}.png"],
                 [OUT / f"{aligned[0]}.png", OUT / f"{aligned[1]}.png"])

    spec = dict(
        title="Panel size and selected panel structure",
        grid=(2, 2),
        panels=[
            (aligned[0], 0, 0, 1, 1),   # A, the sweep
            (aligned[1], 0, 1, 1, 1),   # B, the 25-marker heatmap
            (TILES, 1, 0, 1, 2),        # C, full-width strip
        ],
    )
    build(STEM, spec)
    shutil.copyfile(OUT / f"{STEM}.png", OVERLEAF / f"{STEM}.png")
    print(f"  copied -> {OVERLEAF / f'{STEM}.png'}")


if __name__ == "__main__":
    main()
