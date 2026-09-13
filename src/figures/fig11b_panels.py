#!/usr/bin/env python3
"""
Fig 11b — SLIDE 11 companion: EXTENDED PPI networks, the 12 disease-specific
panels (Part A), with STRING-added first-shell neighbours.

New 2026-08-24, added after a collaborator's request (paraphrased): "keep
one raw image (just our own protein set) and one extended image -- the
extended one expands the STRING network further, so we know which
pathways our proteins are involved in." fig11_panels.py is the raw view
(query_only, each panel's own proteins only). This script is the extended
companion: first_shell_max5 mode (add_nodes=5), where STRING adds up to
5 extra neighbouring proteins per query protein per panel, showing what
pathways/neighbours each panel connects to that its own proteins alone
don't reveal.

WHY THIS IS A SEPARATE FIGURE, NOT A REPLACEMENT for fig11: query_only
answers "does this panel's OWN protein set interact more than chance" --
the statistically honest test, since STRING is not allowed to pick
favourable neighbours there. first_shell_max5 answers a different
question -- "what pathways does this panel connect to" -- but STRING
chooses the added neighbours to maximise connectivity, so its p-values
here are not a fair enrichment test of the original panels and should
not be quoted as such (see fig11_panels.py's docstring for the full
reasoning on why query_only, not this mode, carries the enrichment
claim). Both questions are legitimate and answer different things, so
this is a second figure, not a replacement.

Result (first_shell_max5, live STRING v12.0, fetched by the collaborator's
original batch): every panel gains substantially more edges than in the
raw view once neighbours are added, e.g.
  LYMPH/DLBCL p=5.14e-07, BRC p=1.04e-05, CLL p=0.00027, PRC p=0.00048,
  AML p=0.0086, GLIOM p=0.0073, CRC p=0.0166
  borderline/non-sig: CVX 0.1756, OVC 0.2715, ENDC 0.2887, MYEL 0.4113,
  LUNGC 0.4293
Note the panels that showed literally zero interaction in the raw view
(ENDC, LUNGC, MYEL) still do not reach significance once neighbours are
added -- their own proteins are not just under-connected to each other,
they sit in different neighbourhoods.

NODE IDENTITY CAVEAT: STRING's image does not visually distinguish each
panel's own proteins from the added neighbours by colour -- per
STRING_evidence_view_legend.md, that distinction requires the separate
added-interactors table, not the node colour. Stated in the header
caption so a reader doesn't assume every labelled node in a box is one
of that panel's original screening proteins.

Layout mirrors fig11_panels.py's current (round-6) design: uniform grid
via fig.add_axes() with adjustable="datalim" (fixed-size boxes regardless
of each image's native aspect), single-line titles (cancer code + stars),
p-value in the box's own corner, one unified vertical... actually a single
horizontal legend row as in fig11 (12 boxes need the width back for the
image grid; unlike fig10/10b's single wide network, there's no spare side
column here to move the legend into). See fig11_panels.py and
fig_v5_changelog.md for the full layout history.

Source: revise_plan/ppi_string/
    string_outputs_disease_specific/official_figures/first_shell_max5/<PANEL>/<PANEL>_first_shell_max5_official_STRING_v12.png
    STRING_analysis_official_12_panels/00_common/PPI_statistics_all_12_panels.csv
Output: figurev5/output/fig11b_disease_panels_ppi_mosaic_extended.pdf/.png
Run:    python fig11b_panels.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_labels import display_label

ROOT = Path("e:/Proteomics")
PPI = ROOT / "revise_plan/ppi_string"
STATS_CSV = PPI / "STRING_analysis_official_12_panels/00_common/PPI_statistics_all_12_panels.csv"
IMG_DIR = PPI / "string_outputs_disease_specific/official_figures/first_shell_max5"
OUT = ROOT / "figurev5/output"
OUT.mkdir(parents=True, exist_ok=True)

ROWS = [
    ["AML", "BRC", "CLL", "CRC"],
    ["CVX", "ENDC", "GLIOM", "LUNGC"],
    ["LYMPH", "MYEL", "OVC", "PRC"],
]
N_COLS, N_ROWS = 4, 3

CANCER_COLOR = {
    "AML": "#b22222", "BRC": "#c97b63", "CLL": "#7a3e9d", "CRC": "#d68600",
    "CVX": "#c13d86", "ENDC": "#8e5d2c", "GLIOM": "#1a8db8", "LUNGC": "#2e8b57",
    "LYMPH": "#3856a6", "MYEL": "#8c564b", "OVC": "#d1495b", "PRC": "#008b8b",
}

# Same fixed STRING evidence-view colours as fig10/fig10b/fig11.
EVIDENCE_CHANNELS = [
    ("#e6194b", "fused gene (other species)"),      # Gene fusion
    ("#3cb44b", "neighboring genes"),                # Gene neighborhood
    ("#4363d8", "co-occurs across species"),         # Gene co-occurrence
    ("#a020f0", "lab-tested"),                       # Experiments
    ("#d4ac0d", "mentioned together in papers"),     # Text mining
    ("#42d4f4", "curated database"),                 # Curated databases
    ("#222222", "co-expressed"),                     # Co-expression
]

DPI = 400
PAD_PX = 15
CELL_W_IN = 4.5
CELL_H_IN = 3.15
GAP_IN = 0.22
TITLE_SPACE_IN = 0.34
MARGIN_IN = 0.18
HEADER_H_IN = 1.65       # suptitle + 3 short lines (extra line = node-identity caveat)
LEGEND_H_IN = 0.55

plt.rcParams.update({
    "font.family": "Arial", "font.size": 11, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
})


def _autocrop(img: np.ndarray) -> np.ndarray:
    """Trim to the non-white bounding box. Display transform only.
    These PNGs are RGBA with a transparent background stored as RGB
    (0,0,0) at alpha=0 -- composite against white before testing for
    near-white, or the whole canvas reads as "content"."""
    if img.shape[2] == 4:
        rgb, alpha = img[..., :3], img[..., 3:4]
        composited = rgb * alpha + 1.0 * (1 - alpha)
    else:
        composited = img[..., :3]
    non_white = (composited < 0.95).any(axis=2)
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    r0, r1 = max(rows[0] - PAD_PX, 0), min(rows[-1] + PAD_PX, img.shape[0])
    c0, c1 = max(cols[0] - PAD_PX, 0), min(cols[-1] + PAD_PX, img.shape[1])
    return img[r0:r1, c0:c1]


def load_stats() -> pd.DataFrame:
    d = pd.read_csv(STATS_CSV)
    return d[d["mode"] == "first_shell_max5"].set_index("panel")


def sig_marker(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "~"
    return ""


def _legend(fig, y_center_in: float, fig_h_in: float) -> None:
    handles = [
        plt.Line2D([0], [0], marker="o", ls="", ms=12, mfc="#8ab4d6",
                   mec="#222222", mew=1.2, label="protein (colour decorative)"),
    ] + [
        plt.Line2D([0], [0], color=c, lw=5.5, label=gloss)
        for c, gloss in EVIDENCE_CHANNELS
    ]
    fig.legend(handles=handles, loc="center", ncol=8, frameon=False,
               bbox_to_anchor=(0.5, y_center_in / fig_h_in),
               fontsize=9.5, handlelength=1.6, handletextpad=0.5,
               columnspacing=1.3)


def main() -> None:
    stats = load_stats()

    all_codes = [c for row in ROWS for c in row]
    imgs: dict[str, np.ndarray] = {}
    for code in all_codes:
        png = IMG_DIR / code / f"{code}_first_shell_max5_official_STRING_v12.png"
        imgs[code] = _autocrop(plt.imread(png))

    row_pitch_in = CELL_H_IN + TITLE_SPACE_IN + GAP_IN
    fig_w_in = N_COLS * CELL_W_IN + (N_COLS - 1) * GAP_IN + 2 * MARGIN_IN
    fig_h_in = HEADER_H_IN + N_ROWS * row_pitch_in - GAP_IN + LEGEND_H_IN + MARGIN_IN
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    grid_top_in = fig_h_in - HEADER_H_IN

    for r, row in enumerate(ROWS):
        for c, code in enumerate(row):
            x0_in = MARGIN_IN + c * (CELL_W_IN + GAP_IN)
            y1_in = grid_top_in - TITLE_SPACE_IN - r * row_pitch_in
            y0_in = y1_in - CELL_H_IN
            ax = fig.add_axes([x0_in / fig_w_in, y0_in / fig_h_in,
                                CELL_W_IN / fig_w_in, CELL_H_IN / fig_h_in])
            ax.imshow(imgs[code])
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([])
            ax.set_yticks([])
            col = CANCER_COLOR[code]
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(col)
                spine.set_linewidth(2.2)

            s = stats.loc[code]
            p = float(s["ppi_enrichment_p_value"])
            label = display_label(code)
            star = sig_marker(p)
            ax.set_title(f"{label}{star}", fontsize=13.5, fontweight="bold",
                         color=col, pad=5)
            ax.text(0.03, 0.97, f"$p$ = {p:.2e}", transform=ax.transAxes,
                     fontsize=9.5, fontweight="bold", color=col,
                     ha="left", va="top",
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                               edgecolor=col, linewidth=1.0, alpha=0.92))

    fig.suptitle(
        "Protein-protein association networks (extended) — 12 disease-specific panels (Part A)",
        fontsize=18, fontweight="bold", x=MARGIN_IN / fig_w_in, ha="left",
        y=(fig_h_in - 0.32) / fig_h_in)
    fig.text(
        MARGIN_IN / fig_w_in, (fig_h_in - 0.65) / fig_h_in,
        "Each box: this panel's proteins plus up to 5 STRING-added neighbours per protein (STRING v12.0, "
        r"confidence $\geq$0.400). Shows pathway context, not panel" "\n"
        "enrichment -- STRING chose the added neighbours to maximise connections; see fig11 (raw, "
        r"query-only) for the enrichment test.  **$p$<0.01   *$p$<0.05   ~$p$<0.10",
        fontsize=9.5, fontweight="bold", color="#333333", ha="left", va="top", linespacing=1.35)
    fig.text(
        MARGIN_IN / fig_w_in, (fig_h_in - 1.20) / fig_h_in,
        "Added neighbour proteins are not colour-coded apart from each panel's own proteins in these "
        "images -- see the added-interactors table on disk to tell them apart.",
        fontsize=9.5, fontweight="bold", color="#666666", ha="left", va="top", linespacing=1.35)

    _legend(fig, MARGIN_IN + LEGEND_H_IN * 0.5, fig_h_in)

    for ext in ("pdf", "png"):
        p = OUT / f"fig11b_disease_panels_ppi_mosaic_extended.{ext}"
        fig.savefig(p, dpi=DPI)
        print(f"  saved -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    print("Building fig11b (slide 11 companion — extended 12-panel PPI mosaic) ...")
    main()
    print("Done.")
