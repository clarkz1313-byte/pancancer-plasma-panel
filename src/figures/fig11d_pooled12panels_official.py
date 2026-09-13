#!/usr/bin/env python3
"""
Fig 11d — POOLED network, STRING's own official rendering (no local
relayout, no panel colour-coding).

New 2026-08-24. fig11c (`fig11c_pooled12panels_network.py`) redraws the
same pooled STRING data locally so nodes can be coloured by which of the
12 disease panels each protein belongs to -- STRING's own renderer has no
concept of that grouping. The user pointed out these are two different,
both-legitimate questions: fig11c answers "how do the 12 cancers connect
to each other" (needs panel colour, needs a local layout); THIS figure
answers the more basic "do these 77 proteins interact at all" using
STRING's own native rendering untouched -- pure protein interaction, no
story about cancer grouping laid on top. Both are kept in the deck rather
than treating one as a replacement for the other.

This script only composites -- exactly like fig10/fig11's relationship
to their source PNGs. It does not relayout, recolour, or separate out
small clusters / isolated proteins the way fig11c deliberately does;
STRING's own image already places every node (connected or not) in one
canvas via its own layout, and that native placement is preserved as-is
here, per the user's explicit ask.

AUTOCROP NEEDED A DIFFERENT TEST THAN fig10/fig11's: this PNG's text
labels are drawn fully-opaque WHITE (confirmed by sampling ~206k opaque
near-white pixels), unlike the smaller per-panel/locked-25 images, whose
labels are dark. fig10/fig11's autocrop composites the transparent
background against WHITE and treats near-white as "background" -- against
THIS image that test would misclassify the white text itself as
background. Fixed by cropping on the alpha channel directly (content =
alpha > threshold), which is correct regardless of what colour the
foreground content happens to be.

WHITE-PAGE PASS 2026-08-24: the page itself now matches fig10/fig11's
white house style (title, caption, legend all dark-on-white), but the
IMAGE ITSELF still cannot sit on white -- its text is baked into the PNG
as opaque white pixels, a property of STRING's own render, not something
this script draws or can recolour. Putting that image on a white page
would make its own labels vanish. Fixed by giving just the image's own
axes a black facecolor (a "photo mat" behind the STRING image only) while
the surrounding figure stays white -- the image reads exactly as STRING
rendered it, and everything this script actually controls (title,
caption, legend, stats) follows the shared white house style.

Source:
    revise_plan/ppi_string/STRING_analysis_pooled12panels/
        POOLED12_query_only_official_STRING_v12.png
        ppi_enrichment.tsv
Output: figurev5/output/fig11d_pooled12panels_official.pdf/.png
Run:    python fig11d_pooled12panels_official.py
"""
from __future__ import annotations

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
from publication_style import enable_text_scaling

enable_text_scaling(factor=1.25, minimum=10.5)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revise_plan/ppi_string/STRING_analysis_pooled12panels"
OUT = ROOT / "figurev6/output"
OUT.mkdir(parents=True, exist_ok=True)

PNG = SRC / "POOLED12_query_only_official_STRING_v12_readable.png"
SVG = SRC / "POOLED12_query_only_official_STRING_v12.svg"
NETWORK = SRC / "network.tsv"
PPI_TSV = SRC / "ppi_enrichment.tsv"
REAUDIT_CSV = ROOT / "revise_plan/ppi_string/ppi_reaudit_combined_results_v2.csv"

# Same fixed STRING evidence-view colours as fig10/fig10b/fig11/fig11b --
# this PNG was fetched with network_flavor="evidence" too, so the same
# 7-channel edge colouring applies and the same legend is needed.
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
BG = "#ffffff"
IMG_BG = "#0a0a0a"   # "photo mat" behind the STRING image only -- its own
                     # text is baked-in opaque white and needs a dark
                     # backing regardless of the page's own theme
FG = "#1a1a1a"
MUTED = "#555555"
ACCENT = "#7d1128"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 11, "font.weight": "bold",
})


def load_stats():
    d = pd.read_csv(PPI_TSV, sep="\t")
    r = d.iloc[0]
    audit = pd.read_csv(REAUDIT_CSV).set_index("panel").loc["POOLED12"]
    n, e = int(r["number_of_nodes"]), int(r["number_of_edges"])
    return {
        "nodes": n,
        "edges": e,
        "expected": float(audit["official_expected_number_of_edges"]),
        "p": float(audit["official_p"]),
        "q": float(audit["official_bh_q_14"]),
        "density": 100.0 * e / (n * (n - 1) / 2),   # 3.0% -- cf. fig10's 2.3%
    }


def load_svg_positions():
    """Return STRING-native coordinates and canvas dimensions."""
    raw = SVG.read_text(encoding="utf-8")
    header = raw[:300]
    svg_w = float(re.search(r"width='(\d+)'", header).group(1))
    svg_h = float(re.search(r"height='(\d+)'", header).group(1))
    pos = {}
    for chunk in raw.split("<g class='nwnodecontainer'")[1:]:
        label = re.search(r"data-safe_div_label='([^']+)'", chunk)
        x = re.search(r"data-x_pos='(-?\d+)'", chunk)
        y = re.search(r"data-y_pos='(-?\d+)'", chunk)
        if label and x and y:
            pos[label.group(1)] = (float(x.group(1)), float(y.group(1)))
    return pos, svg_w, svg_h


def _autocrop_alpha(img, alpha_threshold: float = 0.04, pad_px: int = 25):
    """Crop to where alpha > threshold. Unlike fig10/fig11's autocrop
    (which composites onto white and tests for near-white background),
    this tests the alpha channel directly -- correct regardless of
    whether the foreground content (text, node fills) happens to be
    light or dark, which matters here because this PNG's text is white."""
    if img.shape[2] != 4:
        return img
    alpha = img[..., 3]
    non_bg = alpha > alpha_threshold
    rows = np.where(non_bg.any(axis=1))[0]
    cols = np.where(non_bg.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    r0, r1 = max(rows[0] - pad_px, 0), min(rows[-1] + pad_px, img.shape[0])
    c0, c1 = max(cols[0] - pad_px, 0), min(cols[-1] + pad_px, img.shape[1])
    return img[r0:r1, c0:c1]


def main(lean: bool = False) -> None:
    """lean=True drops this figure's ENTIRE right sidebar — the stats block,
    the STRING evidence legend, and the repeated methods line.

    Added 2026-08-31 for the merged slide-10.5 mosaic. Every element in that
    sidebar is a verbatim duplicate of fig11c's: same network, same 88 edges,
    same 80.0 expected, official p = 0.215, same 8-row evidence legend. Printed
    side by side they read as two different results rather than one result
    drawn two ways. In the merged figure fig11c keeps the stats and one
    shared legend is drawn once for all four panels.

    The standalone build (lean=False) now prints the audited official
    background result.
    """
    img = _autocrop_alpha(plt.imread(PNG))
    stats = load_stats()

    aspect = img.shape[1] / img.shape[0]
    img_h_in = 7.5   # tightened from 9.2 -- shrinks dead space below the
                      # legend while staying well above native screen res
    img_w_in = img_h_in * aspect
    stats_col_in = 0.10 if lean else 2.5   # measured (scratch-figure text
                          # extent) from the longest legend label, was a flat
                          # 3.0in guess; collapsed entirely in lean mode
    margin_in = 0.20
    fig_w_in = img_w_in + stats_col_in + 2 * margin_in
    header_h_in = 1.35
    fig_h_in = img_h_in + header_h_in + margin_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    fig.patch.set_facecolor(BG)
    img_bottom_in = margin_in
    ax = fig.add_axes([margin_in / fig_w_in, img_bottom_in / fig_h_in,
                        img_w_in / fig_w_in, img_h_in / fig_h_in])
    ax.set_facecolor(IMG_BG)
    ax.imshow(img)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    if lean:
        # The native STRING raster retains the complete topology, but 77
        # baked labels are too small at journal width. Overlay only the ten
        # highest-degree hubs at readable size; all native nodes remain.
        net = pd.read_csv(NETWORK, sep="\t")
        degree = {}
        for a, b in zip(net["preferredName_A"], net["preferredName_B"]):
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        hubs = {g for g, _ in sorted(degree.items(),
                                     key=lambda kv: (-kv[1], kv[0]))[:8]}
        pos, svg_w, svg_h = load_svg_positions()
        sx, sy = img.shape[1] / svg_w, img.shape[0] / svg_h
        mapped = {g: (pos[g][0] * sx, pos[g][1] * sy)
                  for g in hubs if g in pos}
        cx = img.shape[1] / 2
        left = sorted((g for g in mapped if mapped[g][0] < cx),
                      key=lambda g: mapped[g][1])
        right = sorted((g for g in mapped if mapped[g][0] >= cx),
                       key=lambda g: mapped[g][1])
        targets = {}
        for genes, tx, ha in ((left, img.shape[1] * 0.03, "left"),
                              (right, img.shape[1] * 0.97, "right")):
            ys = np.linspace(img.shape[0] * 0.20, img.shape[0] * 0.80,
                             max(len(genes), 1))
            for gene, ty in zip(genes, ys):
                targets[gene] = (tx, ty, ha)
        for gene, (x, y) in mapped.items():
            tx, ty, ha = targets[gene]
            ann = ax.annotate(
                gene, xy=(x, y), xytext=(tx, ty), textcoords="data",
                ha=ha, va="center", fontsize=13.5, fontweight="bold",
                color="#111111", zorder=8,
                arrowprops=dict(arrowstyle="-", color="#777777", lw=0.8),
            )
            ann.set_path_effects([
                patheffects.withStroke(linewidth=3.0, foreground="white")
            ])

    # Same stats block shape as fig10/fig11c so all three read as one system.
    stats_x_in = margin_in + img_w_in + 0.15
    p_value_y_in = img_bottom_in + img_h_in - 0.45
    if not lean:
        fig.text(stats_x_in / fig_w_in, p_value_y_in / fig_h_in,
                  f"{stats['edges']} edges",
                  fontsize=19, fontweight="bold", color=ACCENT,
                  ha="left", va="top")
        fig.text(stats_x_in / fig_w_in, (p_value_y_in - 0.34) / fig_h_in,
                  f"{stats['expected']:.1f} expected\n"
                  f"{stats['density']:.1f}% of pairs\n"
                  f"STRING $p$ = {stats['p']:.3f}\n"
                  f"BH $q$ = {stats['q']:.3f}",
                  fontsize=11.5, fontweight="bold", color=MUTED,
                  ha="left", va="top", linespacing=1.55)

    if lean:
        # one line only: the network, the stats and the methods are all
        # established by the panel next to it in the merged figure.
        # SHORTENED 2026-08-31 pass 2 — the previous wording was measured at
        # ~7.6in against a lean canvas of ~6.6in of usable width, so it ran off
        # the right edge and the merged figure showed "...not grouped by cance".
        pass
    else:
        fig.text(
            margin_in / fig_w_in, (fig_h_in - 0.62) / fig_h_in,
            f"STRING v12.0 · {stats['nodes']} proteins pooled from all 12 panels (query-only, no added neighbours).",
            fontsize=10.5, fontweight="bold", color=MUTED, ha="left", va="top")
        fig.text(
            margin_in / fig_w_in, (fig_h_in - 0.90) / fig_h_in,
            "STRING's native layout, unedited | protein associations are not grouped by cancer\n"
            "(see panel c for the same edges colored by cancer-specific panel).",
            fontsize=10.5, fontweight="bold", color=MUTED, ha="left", va="top", linespacing=1.4)

    if not lean:
        handles = [
            plt.Line2D([0], [0], marker="o", ls="", ms=12, mfc="#8ab4d6",
                       mec=FG, mew=1.0, label="protein (color decorative)"),
        ] + [
            plt.Line2D([0], [0], color=c, lw=5.5, label=gloss)
            for c, gloss in EVIDENCE_CHANNELS
        ]
        legend_top_in = p_value_y_in - 1.45   # clears the stats block
        leg = fig.legend(handles=handles, loc="upper left", ncol=1, frameon=False,
                          bbox_to_anchor=(stats_x_in / fig_w_in, legend_top_in / fig_h_in),
                          fontsize=10, handlelength=1.8, handletextpad=0.6,
                          labelspacing=0.7)
        for text in leg.get_texts():
            text.set_color(FG)

    stem = "fig11d_pooled12panels_official" + ("_lean" if lean else "")
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI, facecolor=BG)
        print(f"  saved -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    print("Building fig11d (pooled 12-panel network, official STRING render) ...")
    main()
    print("Done.")
