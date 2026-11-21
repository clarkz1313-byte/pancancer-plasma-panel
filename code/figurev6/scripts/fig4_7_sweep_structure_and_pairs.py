#!/usr/bin/env python3
"""
fig4_7_sweep_structure_and_pairs.py

Builds the manuscript's Figure 4 (fig:structure), replacing the original
three-panel version (expression heatmap / concordance scatter / twelve static
heatmaps). This was the last of a chain of candidates -- 3-5, 3-6, 4-5, 4-6 --
and is the one that was adopted; its output is written under the figure's
permanent name, fig4_slide4_structure_mosaic.png, so sections/03_results.tex
needs no filename change, only the caption rewrite that goes with it.

  A  the multiclass size sweep                      (as Figure 3-6A)
  B  grouped expression heatmap of the 25 markers   (as Figure 3-6B)
  C  the twelve disease-specific panels, each as a size sweep beside its
     expression tile                                (as Figure 4-6)

So the figure carries both size arguments and both structure views in one
place: why 25 markers and what they look like on top, then the same two
questions answered cancer by cancer underneath.

Layout note. Figure 4-6 stands alone at 4 rows x 3 cancers, which is close to
square and would make this composite about 6,700 px tall with every element
downscaled twice over. Panel C is therefore re-rendered here at 3 rows x 4
cancers and a wider canvas: the strip lands at roughly 1.8:1, the composite
stays near square, and each pair keeps about 1,200 px of final width instead
of 790. Same drawing code, same data, different grid.

Panels A and B are reused from the Figure 3-6 builder, including its measured
baseline alignment, so the top-row x axes line up here for the same reason
they line up there.

Run:    python fig3_5_and_fig4_5_panel_size_selection.py
        python fig4_7_sweep_structure_and_pairs.py
Output: figurev6/output/ and overleaf-source-git/figures/
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_slide_mosaics import OUT, build
from fig3_5_and_fig4_5_panel_size_selection import (
    draw_single_sweep, load_single, sweep_legend_handles,
)
from fig3_6_structure_with_sweep import PARTB, SWEEP, build_partb_panel
from fig_candidate_common import (
    CANCER_ORDER, align_images, deployed_members, draw_expression_tile,
    zscore_matrix,
)
from figure_labels import display_label

ROOT = Path(__file__).resolve().parents[2]
OVERLEAF = ROOT / "overleaf-source-git" / "figures"

STEM = "fig4_slide4_structure_mosaic"   # the manuscript's permanent Figure 4 name
PAIRS = "fig4_7c_pairs_wide"

NROW, NCANC = 3, 4


def build_pairs_panel() -> None:
    """The twelve sweep-and-tile pairs, laid out wide for a full-width cell."""
    members = deployed_members()
    z = zscore_matrix(members)

    fig = plt.figure(figsize=(30.0, 17.0))
    outer = fig.add_gridspec(NROW, NCANC, hspace=0.52, wspace=0.26,
                             left=0.035, right=0.99, top=0.97, bottom=0.10)

    for idx, cancer in enumerate(CANCER_ORDER):
        r, c = divmod(idx, NCANC)
        # Wide gutter inside each pair: the longest marker names (TNFRSF13C,
        # TNFRSF10A, ADAMTS13) hang left of their heatmap and would otherwise
        # reach into the sweep's axis.
        cell = outer[r, c].subgridspec(1, 2, width_ratios=[1.0, 1.10],
                                       wspace=0.46)
        ax_sweep = fig.add_subplot(cell[0, 0])
        ax_tile = fig.add_subplot(cell[0, 1])

        draw_single_sweep(ax_sweep, cancer, load_single(cancer.lower()),
                          show_ylabel=(c == 0), show_xlabel=(r == NROW - 1),
                          tick_size=12.0, axis_label_size=14.0)
        draw_expression_tile(ax_tile, fig, cancer, members[cancer], z,
                             display_label, label_size=12.0, tick_size=12.0)

    fig.legend(handles=sweep_legend_handles(), loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.012), handlelength=2.4,
               fontsize=16)
    fig.savefig(OUT / f"{PAIRS}.png", dpi=400, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  built {PAIRS}")


def main() -> None:
    if not (OUT / f"{SWEEP}.png").exists():
        raise SystemExit("run fig3_5_and_fig4_5_panel_size_selection.py first")

    build_partb_panel()
    build_pairs_panel()

    aligned = [f"{SWEEP}_aligned", f"{PARTB}_aligned"]
    align_images([OUT / f"{SWEEP}.png", OUT / f"{PARTB}.png"],
                 [OUT / f"{aligned[0]}.png", OUT / f"{aligned[1]}.png"])

    spec = dict(
        title="Panel size, selected structure, and the twelve panels",
        grid=(2, 2),
        # A slim blank band for the panel letters: without it "A" lands on top
        # of the sweep's own "1.0" y-tick, since both sit at the image's
        # top-left corner. 104 matches Figure 2's mosaic for consistency.
        label_band=104,
        panels=[
            (aligned[0], 0, 0, 1, 1),   # A, the multiclass sweep
            (aligned[1], 0, 1, 1, 1),   # B, the 25-marker heatmap
            (PAIRS, 1, 0, 1, 2),        # C, full-width pairs strip
        ],
    )
    build(STEM, spec)
    shutil.copyfile(OUT / f"{STEM}.png", OVERLEAF / f"{STEM}.png")
    print(f"  copied -> {OVERLEAF / f'{STEM}.png'}")


if __name__ == "__main__":
    main()
