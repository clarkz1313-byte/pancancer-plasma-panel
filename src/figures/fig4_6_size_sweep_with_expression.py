#!/usr/bin/env python3
"""
fig4_6_size_sweep_with_expression.py

Candidate Figure 4-6: for each of the 12 cancers, its panel-size sweep and its
panel expression heatmap side by side, so that "how big is this panel and why"
and "what does this panel measure" are read together rather than from two
separate figures.

  left  of each pair -- the size sweep drawn by fig4_5 (held-out AUC, balanced
                        accuracy and F1 against sizes 1-18, with the 0.02 AUC
                        tolerance band and the selected size marked three ways)
  right of each pair -- the same cancer's tile from Figure 4 panel C: mean NPX
                        z-score for its deployed markers across all 12 cancer
                        groups, its own column outlined

Both halves are reproduced rather than re-invented. The sweep comes from
fig3_5_and_fig4_5_panel_size_selection.draw_single_sweep and the tile from
fig_candidate_common.draw_expression_tile, so a change to either lands on
Figures 4-5, 4-6 and 3-6 together.

Deployed panel membership is the frozen list the manuscript reports
(panel_memberships.csv), ordered by that cancer's own L1 importance, which
reproduces Figure 4 panel C's row order exactly. The sweep curves come from the
seed-52 derivation rerun.

Output: figurev6/output/ and overleaf-source-git/figures/ at 600 dpi.
Run:    python fig4_6_size_sweep_with_expression.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig3_5_and_fig4_5_panel_size_selection import (
    CANCER_ORDER, DEPLOYED, draw_single_sweep, load_single,
    sweep_legend_handles,
)
from fig_candidate_common import (
    deployed_members, draw_expression_tile, zscore_matrix,
)
from figure_labels import display_label

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figurev6" / "output"
OVERLEAF = ROOT / "overleaf-source-git" / "figures"
DPI = 600


def main():
    members = deployed_members()
    z = zscore_matrix(members)

    NROW, NCANC = 4, 3
    fig = plt.figure(figsize=(16.0, 15.4))
    outer = fig.add_gridspec(NROW, NCANC, hspace=0.62, wspace=0.30,
                             left=0.05, right=0.985, top=0.975, bottom=0.075)

    for idx, cancer in enumerate(CANCER_ORDER):
        r, c = divmod(idx, NCANC)
        cell = outer[r, c].subgridspec(1, 2, width_ratios=[1.0, 1.22],
                                       wspace=0.42)
        ax_sweep = fig.add_subplot(cell[0, 0])
        ax_heat = fig.add_subplot(cell[0, 1])

        draw_single_sweep(ax_sweep, cancer, load_single(cancer.lower()),
                          show_ylabel=(c == 0), show_xlabel=(r == NROW - 1))
        draw_expression_tile(ax_heat, fig, cancer, members[cancer], z,
                             display_label, label_size=8.0, tick_size=8.0)

    fig.legend(handles=sweep_legend_handles(), loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.014), handlelength=2.2,
               fontsize=11)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig4_6_panel_size_and_expression.{ext}",
                    dpi=DPI, facecolor="white", bbox_inches="tight")
    fig.savefig(OVERLEAF / "fig4_6_panel_size_and_expression.png",
                dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {OUT / 'fig4_6_panel_size_and_expression.png'}")
    for cancer in CANCER_ORDER:
        print(f"  {display_label(cancer):<6} n={DEPLOYED[cancer]:>2}  "
              f"{';'.join(members[cancer])}")


if __name__ == "__main__":
    main()
