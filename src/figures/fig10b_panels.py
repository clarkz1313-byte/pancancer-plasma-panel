#!/usr/bin/env python3
"""
Fig 10b — SLIDE 10 companion: EXTENDED PPI network, locked 25-protein
multiclass panel (Part B), with STRING-added first-shell neighbours.

New 2026-08-24, added after a collaborator's request (paraphrased):
"keep one raw image (just our own protein set) and one extended image --
the extended one expands the STRING network further, so we know which
pathways our proteins are involved in." fig10_panels.py is the raw view
(query_only, this panel's 25 proteins only). This script is the extended
companion: first_shell_max5 mode (add_nodes=5), where STRING adds up to
5 extra neighbouring proteins per query protein, pulling in pathway
context the panel's own 25 proteins don't show on their own.

WHY THIS IS A SEPARATE FIGURE, NOT A REPLACEMENT for fig10: query_only
answers "does this panel's OWN protein set interact more than chance" --
the statistically honest test, since STRING is not allowed to pick
favourable neighbours. first_shell_max5 answers a different question --
"what pathways/neighbours does this panel connect to" -- but STRING
chooses the added neighbours to maximise connectivity, so its p-value
here is not a fair enrichment test of the original panel and should not
be quoted as such. Both questions are legitimate; they are not the same
question, so this stays a second figure rather than replacing fig10.

Result (first_shell_max5, live STRING v12.0, fetched 2026-08-23):
  30 nodes (25 original + up to 5 STRING-added neighbours), 25 edges,
  expected 10, PPI enrichment p = 5.47e-05.

NODE IDENTITY CAVEAT: STRING's image does not visually distinguish the
panel's own 25 proteins from the added neighbours by colour -- per
STRING_evidence_view_legend.md, that distinction requires the separate
added-interactors table, not the node colour. Stated explicitly in the
caption below so a reader doesn't assume every labelled node is one of
the panel's original screening proteins.

Layout mirrors fig10_panels.py's current (round-8) design exactly: image
at a fixed position with a dedicated side column for the p-value and a
vertical evidence-colour legend below it, sized from actual content with
no arbitrary width floor. See fig10_panels.py and fig_v5_changelog.md for
the layout history that led to that design.

Source:
    revise_plan/ppi_string/STRING_analysis_locked25/first_shell_max5/
        LOCKED25_first_shell_max5_official_STRING_v12.png
        ppi_enrichment.tsv
Output: figurev5/output/fig10b_locked25_ppi_network_extended.pdf/.png
Run:    python fig10b_panels.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("e:/Proteomics")
SRC = ROOT / "revise_plan/ppi_string/STRING_analysis_locked25/first_shell_max5"
OUT = ROOT / "figurev5/output"
OUT.mkdir(parents=True, exist_ok=True)

PNG = SRC / "LOCKED25_first_shell_max5_official_STRING_v12.png"
PPI_TSV = SRC / "ppi_enrichment.tsv"

# Same fixed STRING evidence-view colours as fig10_panels.py / fig11_panels.py.
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

plt.rcParams.update({
    "font.family": "Arial", "font.size": 11, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
})


def load_stats():
    d = pd.read_csv(PPI_TSV, sep="\t")
    r = d.iloc[0]
    return {
        "nodes": int(r["number_of_nodes"]),
        "edges": int(r["number_of_edges"]),
        "expected": float(r["expected_number_of_edges"]),
        "p": float(r["p_value"]),
    }


def _autocrop(img, pad_px: int = 20):
    """Trim to the non-white bounding box. This PNG's transparent
    background is RGB (0,0,0) at alpha=0 -- composite against white
    before testing for near-white, or the whole canvas reads as
    "content"."""
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
    r0, r1 = max(rows[0] - pad_px, 0), min(rows[-1] + pad_px, img.shape[0])
    c0, c1 = max(cols[0] - pad_px, 0), min(cols[-1] + pad_px, img.shape[1])
    return img[r0:r1, c0:c1]


def main() -> None:
    img = _autocrop(plt.imread(PNG))
    stats = load_stats()

    aspect = img.shape[1] / img.shape[0]
    img_h_in = 6.6
    img_w_in = img_h_in * aspect
    stats_col_in = 3.0
    margin_in = 0.20
    fig_w_in = img_w_in + stats_col_in + 2 * margin_in
    header_h_in = 1.40   # suptitle + 3 short caption lines (extra line = node-identity caveat)
    fig_h_in = img_h_in + header_h_in + margin_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    img_bottom_in = margin_in
    ax = fig.add_axes([margin_in / fig_w_in, img_bottom_in / fig_h_in,
                        img_w_in / fig_w_in, img_h_in / fig_h_in])
    ax.imshow(img)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    stats_x_in = margin_in + img_w_in + 0.25
    p_value_y_in = img_bottom_in + img_h_in - 0.55
    fig.text(stats_x_in / fig_w_in, p_value_y_in / fig_h_in,
              f"$p$ = {stats['p']:.2e}",
              fontsize=17, fontweight="bold", color="#7d1128",
              ha="left", va="top")

    fig.suptitle(
        "Protein-protein association network (extended) — locked 25-protein multiclass panel (Part B)",
        fontsize=16, fontweight="bold", x=margin_in / fig_w_in, ha="left",
        y=(fig_h_in - 0.30) / fig_h_in)
    fig.text(
        margin_in / fig_w_in, (fig_h_in - 0.62) / fig_h_in,
        f"STRING v12.0 · {stats['nodes']} proteins, {stats['edges']} edges "
        f"(expected {stats['expected']:.0f}) · this panel's 25 proteins plus up to 5 "
        "STRING-added neighbours per protein.",
        fontsize=10.5, fontweight="bold", color="#666666", ha="left", va="top")
    fig.text(
        margin_in / fig_w_in, (fig_h_in - 0.90) / fig_h_in,
        "Shows pathway context, not panel enrichment -- STRING chose the added neighbours "
        "to maximise connections; see the raw (query-only) figure for the enrichment test.",
        fontsize=10.5, fontweight="bold", color="#666666", ha="left", va="top")
    fig.text(
        margin_in / fig_w_in, (fig_h_in - 1.18) / fig_h_in,
        "Added neighbour proteins are not colour-coded apart from this panel's own proteins in "
        "this image -- see the added-interactors table on disk to tell them apart.",
        fontsize=10.5, fontweight="bold", color="#666666", ha="left", va="top")

    handles = [
        plt.Line2D([0], [0], marker="o", ls="", ms=12, mfc="#8ab4d6",
                   mec="#222222", mew=1.2, label="protein (colour decorative)"),
    ] + [
        plt.Line2D([0], [0], color=c, lw=5.5, label=gloss)
        for c, gloss in EVIDENCE_CHANNELS
    ]
    legend_top_in = p_value_y_in - 0.55
    fig.legend(handles=handles, loc="upper left", ncol=1, frameon=False,
               bbox_to_anchor=(stats_x_in / fig_w_in, legend_top_in / fig_h_in),
               fontsize=10, handlelength=1.8, handletextpad=0.6,
               labelspacing=0.85)

    for ext in ("pdf", "png"):
        p = OUT / f"fig10b_locked25_ppi_network_extended.{ext}"
        fig.savefig(p, dpi=DPI)
        print(f"  saved -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    print("Building fig10b (slide 10 companion — extended locked-25 PPI network) ...")
    main()
    print("Done.")
