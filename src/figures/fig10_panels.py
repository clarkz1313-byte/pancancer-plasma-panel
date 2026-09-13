#!/usr/bin/env python3
"""
Fig 10 — SLIDE 10: PPI network, the locked 25-protein multiclass panel (Part B).

New 2026-08-23. Replaces a manual STRING-website screenshot with a
reproducible pipeline: `revise_plan/ppi_string/fetch_locked25_string.py`
calls the official STRING v12.0 API directly (same settings as the sibling
12-disease-panel batch -- species 9606, functional network, medium
confidence 0.400) and writes the network image plus real PPI-enrichment
statistics to `revise_plan/ppi_string/STRING_analysis_locked25/`. This
script only composes; it does not redraw the network -- the PNG embedded
below is STRING's own official image, not a locally-recomputed layout.

Why this panel needed its own fetch: the original STRING batch handed off
by the collaborator (`revise_plan/ppi_string/`) covered only the 12
disease-specific panels (`disease_specific_panels.csv`); the locked-25
panel (`locked_25_multiclass_panel.csv`) was present in that folder but
was never run through the pipeline. See figurev5/fig_v5_changelog.md for
the full gap-and-fix record.

Mode shown: query_only (add_nodes=0) -- the panel's own 25 proteins only,
no STRING-added interactors. This is the statistically honest mode: STRING
picks first-shell neighbours to maximise connectivity, which inflates
apparent structure and is not a fair test of "does this panel show more
interaction than chance." Same convention as fig11.

Current result (official assay background, 2026-09-04):
  25 nodes, 7 edges, 8.0 expected, p = 0.7136, BH q = 0.9808.
The 10,000-draw mapped-network sensitivity p is 0.5673. No PPI enrichment
claim is supported.

TEXT/LAYOUT PASS 2026-08-23 (matches fig11_panels.py's same-day rounds,
see fig_v5_changelog.md for the full account): stats (nodes/edges/p) moved
out of a stacked header line into a small annotation next to the network
itself; legend unified into one row (node marker + 7 edge colours, short
single-phrase labels, formal STRING channel names kept only as a code
comment for provenance -- showing "Name (gloss)" for all 7 was the
cluttered, parenthesis-heavy legend a reader flagged as messy).

LAYOUT PASS 2 2026-08-23: the fixed 16.5in width floor (added earlier so
a long single-line caption wouldn't get cut off) left a wide dead strip
of white space to the right of the stats column once the caption was
already short -- removed; figure width is now sized from actual content
(image + stats column + margins) only. The 2026-09-04 audit restored the
official expected-edge value and BH q-value to the side annotation. The evidence-colour
legend changed from one horizontal row to a vertical column, moved into
the freed space beside the image -- this is what actually ate the old
whitespace, not a wider canvas. Legend line swatches drawn thicker
(lw 3.0 -> 5.5) so the 7 evidence colours read apart more easily; the
network's own edges are pixels inside STRING's official PNG and are not
redrawn by this script, so their thickness is set by STRING, not here --
this bolding only affects the legend key.

WHY NODE FILL COLOUR IS NOT REMOVED: see fig11_panels.py's docstring for
the full reasoning -- STRING's image API offers no "hollow, border only"
node style, and stripping fill via pixel-editing would risk damaging the
3D-structure cartoon drawn on top of it, which IS meaningful (unlike the
flat background colour behind it).

Source:
    revise_plan/ppi_string/STRING_analysis_locked25/query_only/
        LOCKED25_query_only_official_STRING_v12.png
        ppi_enrichment.tsv
Output: figurev5/output/fig10_locked25_ppi_network.pdf/.png
Run:    python fig10_panels.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from publication_style import enable_text_scaling

enable_text_scaling(factor=1.25, minimum=10.5)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revise_plan/ppi_string/STRING_analysis_locked25/query_only"
OUT = ROOT / "figurev6/output"
OUT.mkdir(parents=True, exist_ok=True)

PNG = SRC / "LOCKED25_query_only_official_STRING_v12.png"
PPI_TSV = SRC / "ppi_enrichment.tsv"
# Reaudited assay-background statistics. STRING's official background-aware
# endpoint is primary; the local 10,000-draw result is a sensitivity analysis.
REAUDIT_CSV = ROOT / "revise_plan/ppi_string/ppi_reaudit_combined_results_v2.csv"

# STRING's fixed evidence-view edge colours. Formal STRING names (Gene
# fusion, Gene neighborhood, Gene co-occurrence, Experiments, Text mining,
# Curated databases, Co-expression) kept as a comment for provenance;
# the figure itself shows only the plain-language phrase (see fig11's
# docstring for why the "Name (gloss)" double-labelling was dropped).
EVIDENCE_CHANNELS = [
    ("#e6194b", "fused gene (other species)"),      # Gene fusion
    ("#3cb44b", "neighboring genes"),                # Gene neighborhood
    ("#4363d8", "co-occurs across species"),         # Gene co-occurrence
    ("#a020f0", "lab-tested"),                       # Experiments
    ("#d4ac0d", "mentioned together in papers"),     # Text mining
    ("#42d4f4", "curated database"),                 # Curated databases
    ("#222222", "co-expressed"),                     # Co-expression
]

# Same purple used deck-wide (fig89_common.py's MACRO_COLOR) to mark Part B
# / the locked 25-protein multiclass panel -- e.g. the 13th "macro mean"
# tile in fig8/fig9. Reused here as this figure's border so it reads as
# "the multiclass panel" the same way it does everywhere else in the deck,
# rather than inventing a new colour just for this figure.
PART_B_COLOR = "#A06CD5"

DPI = 400

plt.rcParams.update({
    "font.family": "Arial", "font.size": 11, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
})


def load_stats():
    d = pd.read_csv(PPI_TSV, sep="\t")
    r = d.iloc[0]
    audit = pd.read_csv(REAUDIT_CSV).set_index("panel").loc["LOCKED25"]
    n, e = int(r["number_of_nodes"]), int(r["number_of_edges"])
    return {
        "nodes": n,
        "edges": e,
        "expected": float(audit["official_expected_number_of_edges"]),
        "p": float(audit["official_p"]),
        "q": float(audit["official_bh_q_14"]),
        "custom_p": float(audit["custom_empirical_p"]),
        # % of all possible pairs that are connected -- more intuitive than a
        # permutation p for a reader looking at a network picture, and it is
        # what makes fig10 and fig11c/d directly comparable (2.3% vs 3.0%,
        # i.e. the pooled network is NOT denser, it just has more nodes).
        "density": 100.0 * e / (n * (n - 1) / 2),
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


def main(lean: bool = False) -> None:
    """lean=True keeps this panel's own stats block (7 edges, p = 0.714, a
    number unique to the locked-25 network) but drops its STRING evidence
    legend, which in the merged slide-10.5 mosaic is a duplicate of the one
    shared legend drawn once for all four panels."""
    img = _autocrop(plt.imread(PNG))
    stats = load_stats()

    aspect = img.shape[1] / img.shape[0]
    img_h_in = 5.4   # tightened from 6.6 -- still well above native screen
                      # resolution at 400dpi, shrinks the dead space below
                      # the legend that a taller column left unused
    img_w_in = img_h_in * aspect
    # Dedicated space to the RIGHT of the image for the p-value box and
    # the (now vertical) legend -- guarantees no overlap with network
    # content by construction (an earlier version overlaid the box on the
    # image's own upper-left corner, which collided with real nodes there;
    # STRING's node layout is not known in advance, so any overlay
    # position can collide). Width MEASURED (scratch-figure text extent,
    # not guessed) from the longest legend label ("mentioned together in
    # papers") at the legend fontsize -- ~2.29in incl. handle -- plus a
    # small margin; was a flat 3.0in guess that left visible dead width.
    stats_col_in = 0.10 if lean else 2.5
    margin_in = 0.20
    # No fixed width floor: sizing from actual content (image + stats
    # column + margins) only. An earlier version forced a 16.5in floor to
    # fit a wider single-line caption -- once the caption moved to two
    # short lines that floor just left dead white space to the right of
    # the stats column.
    fig_w_in = img_w_in + stats_col_in + 2 * margin_in
    header_h_in = 0.20 if lean else 1.15
    footer_h_in = 0.20 if lean else margin_in
    fig_h_in = img_h_in + header_h_in + footer_h_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    img_bottom_in = footer_h_in
    ax = fig.add_axes([margin_in / fig_w_in, img_bottom_in / fig_h_in,
                        img_w_in / fig_w_in, img_h_in / fig_h_in])
    ax.imshow(img)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PART_B_COLOR)
        spine.set_linewidth(2.2)

    # Side column: observed edge count large, its random-draw comparator and
    # density immediately under it. The observed-vs-random pair IS the
    # argument -- no prose needed to say "this is unremarkable", the two
    # numbers sitting together show it.
    stats_x_in = margin_in + img_w_in + 0.15
    stats_y_in = img_bottom_in + img_h_in - 0.45
    if lean:
        pass  # merged mosaic adds the same aligned footer used by C and D
    else:
        fig.text(stats_x_in / fig_w_in, stats_y_in / fig_h_in,
                 f"{stats['edges']}/{stats['expected']:.0f}\nedges",
                 fontsize=14, fontweight="bold", color="#7d1128",
                 ha="left", va="top")

    # Vertical legend column, stacked below the p-value in the same
    # reserved side column -- this freed-up space (previously a full-width
    # horizontal row along the bottom) is what removes the dead white
    # space to the right of the plot, not a wider canvas.
    if not lean:
        handles = [
            plt.Line2D([0], [0], marker="o", ls="", ms=12, mfc="#8ab4d6",
                       mec="#222222", mew=1.2, label="protein (color decorative)"),
        ] + [
            plt.Line2D([0], [0], color=c, lw=5.5, label=gloss)
            for c, gloss in EVIDENCE_CHANNELS
        ]
        legend_top_in = stats_y_in - 1.35   # clears the stats block (headline
                                              # count + 3 comparator lines)
        fig.legend(handles=handles, loc="upper left", ncol=1, frameon=False,
                   bbox_to_anchor=(stats_x_in / fig_w_in, legend_top_in / fig_h_in),
                   fontsize=10, handlelength=1.8, handletextpad=0.6,
                   labelspacing=0.7)

    stem = "fig10_locked25_ppi_network" + ("_lean" if lean else "")
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI)
        print(f"  saved -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    print("Building fig10 (slide 10 — locked-25 PPI network) ...")
    main()
    print("Done.")
