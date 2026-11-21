#!/usr/bin/env python3
"""
Fig 11c — POOLED network: all 12 disease-specific panels' own proteins
combined into ONE STRING query (query_only, add_nodes=0 -- no STRING-added
neighbours), coloured by which panel(s) each protein belongs to.

New 2026-08-24, in response to: "can you make a pool graph of all 12
cancers together (no extended) to see how they totally connect to each
other?" -- see fig11d_pooled12panels_official.py's docstring for the full
background on why pooling (not first_shell_max5 extension) is the right
tool for that question. fig11d shows STRING's own official rendering of
this exact same query, untouched; THIS figure re-colours the same edges
by panel membership so cross-cancer connectivity is visible at a glance.

LAYOUT PASS 2026-08-24: the first version of this figure used an
independent local layout (kamada_kawai for the 50-node giant component,
spring_layout for the 3 small components, isolated proteins listed in a
separate strip below). The user asked for something different: fig11c and
fig11d should place each protein in NEARLY THE SAME position, so they read
as "the same picture, two colourings" rather than as two unrelated
diagrams -- and specifically objected to isolated proteins and small
components being pulled out into their own areas rather than left where
they actually sit.

FIXED by reusing STRING's OWN node coordinates instead of computing a
layout at all: `fetch_pooled12panels_string.py` now also fetches the
network as SVG (`/api/svg/network`), and STRING embeds each node's exact
plotted position as `data-x_pos` / `data-y_pos` attributes on that node's
`<g class="nwnodecontainer">` element, keyed by `data-safe_div_label`
(the gene symbol) -- all 77 pooled proteins matched with none missing.
Confirmed the PNG is this SAME layout at a fixed scale-up (PNG size /
SVG size = 3.333 on both axes, i.e. the highres PNG is just the SVG
rendered at higher DPI, not a different layout run). Positions are
converted from SVG space to the CROPPED PNG's pixel space (same scale
factor, same alpha-based crop offset as fig11d's autocrop) so a node's
pixel position here lines up with where that same protein sits in
fig11d -- overlaying the two is now meaningful, which it was not before.

No more isolated-protein strip, no more small-component slots: every
protein is drawn exactly where STRING itself placed it, whether that
happens to be in the dense central hub or off on its own -- matching
fig11d's presentation, per the user's request, rather than editorialising
the layout to group disconnected pieces together.

Multi-panel proteins (15 of 77) still drawn as pie-sliced nodes (one
wedge per panel) rather than a single arbitrary colour. Edges still
coloured same-panel (grey) vs cross-panel (dark red) -- not by STRING
evidence channel, since this figure's claim is about network topology
across panels, not evidence type (see fig11d for the evidence-colour
version of these same edges).

Source:
    revise_plan/ppi_string/STRING_analysis_pooled12panels/
        POOLED12_query_only_official_STRING_v12.{png,svg}
        network.tsv, ppi_enrichment.tsv, gene_panel_membership.csv
Output: figurev5/output/fig11c_pooled12panels_network.pdf/.png
Run:    python fig11c_pooled12panels_network.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_labels import display_label
from publication_style import enable_text_scaling

enable_text_scaling(factor=1.25, minimum=10.5)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revise_plan/ppi_string/STRING_analysis_pooled12panels"
OUT = ROOT / "figurev6/output"
OUT.mkdir(parents=True, exist_ok=True)

PNG = SRC / "POOLED12_query_only_official_STRING_v12.png"
SVG = SRC / "POOLED12_query_only_official_STRING_v12.svg"
REAUDIT_CSV = ROOT / "revise_plan/ppi_string/ppi_reaudit_combined_results_v2.csv"

CANCER_COLOR = {
    "AML": "#b22222", "BRC": "#c97b63", "CLL": "#7a3e9d", "CRC": "#d68600",
    "CVX": "#c13d86", "ENDC": "#8e5d2c", "GLIOM": "#1a8db8", "LUNGC": "#2e8b57",
    "LYMPH": "#3856a6", "MYEL": "#8c564b", "OVC": "#d1495b", "PRC": "#008b8b",
}
PANEL_ORDER = list(CANCER_COLOR)

DPI = 400
SAME_PANEL_EDGE_COLOR = "#999999"
CROSS_PANEL_EDGE_COLOR = "#8b1a1a"
BG = "#ffffff"
FG = "#1a1a1a"
MUTED = "#555555"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 11, "font.weight": "bold",
})


def load_membership() -> dict[str, list[str]]:
    m = {}
    with (SRC / "gene_panel_membership.csv").open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            m[row["gene"]] = sorted(row["panels"].split(";"))
    return m


def load_stats() -> dict:
    d = pd.read_csv(SRC / "ppi_enrichment.tsv", sep="\t")
    r = d.iloc[0]
    audit = pd.read_csv(REAUDIT_CSV).set_index("panel").loc["POOLED12"]
    n, e = int(r["number_of_nodes"]), int(r["number_of_edges"])
    return {
        "nodes": n,
        "edges": e,
        "expected": float(audit["official_expected_number_of_edges"]),
        "p": float(audit["official_p"]),
        "q": float(audit["official_bh_q_14"]),
        "custom_p": float(audit["custom_empirical_p"]),
        # 3.0% here vs 2.3% in fig10 -- the pooled network is NOT denser than
        # the locked-25 panel, it just has 3x the nodes, which makes any
        # force-directed layout look like a web. Printing density is what
        # stops a reader reading slide 11 as contradicting slide 10.
        "density": 100.0 * e / (n * (n - 1) / 2),
    }


def load_svg_positions() -> tuple[dict[str, tuple[float, float]], float, float]:
    """STRING's own node positions, straight from its SVG export --
    data-x_pos/data-y_pos on each node's <g class="nwnodecontainer">,
    keyed by data-safe_div_label (the gene symbol). Returns positions plus
    the SVG canvas width/height for scaling into PNG pixel space."""
    text = SVG.read_text(encoding="utf-8")
    header = text[:300]
    svg_w = float(re.search(r"width='(\d+)'", header).group(1))
    svg_h = float(re.search(r"height='(\d+)'", header).group(1))
    pos = {}
    for chunk in text.split("<g class='nwnodecontainer'")[1:]:
        label = re.search(r"data-safe_div_label='([^']+)'", chunk)
        x = re.search(r"data-x_pos='(-?\d+)'", chunk)
        y = re.search(r"data-y_pos='(-?\d+)'", chunk)
        if label and x and y:
            pos[label.group(1)] = (float(x.group(1)), float(y.group(1)))
    return pos, svg_w, svg_h


def _autocrop_bounds(img, alpha_threshold: float = 0.04, pad_px: int = 25):
    """Same alpha-based crop as fig11d_pooled12panels_official.py -- returns
    the (r0, r1, c0, c1) bounds instead of the cropped array, so node
    positions here can be shifted by the identical offset fig11d used."""
    alpha = img[..., 3]
    non_bg = alpha > alpha_threshold
    rows = np.where(non_bg.any(axis=1))[0]
    cols = np.where(non_bg.any(axis=0))[0]
    r0, r1 = max(rows[0] - pad_px, 0), min(rows[-1] + pad_px, img.shape[0])
    c0, c1 = max(cols[0] - pad_px, 0), min(cols[-1] + pad_px, img.shape[1])
    return r0, r1, c0, c1


def draw_node(ax, xy, panels: list[str], r: float):
    if len(panels) == 1:
        ax.add_patch(plt.Circle(xy, r, facecolor=CANCER_COLOR[panels[0]],
                                 edgecolor=FG, linewidth=0.8, zorder=3))
        return
    n = len(panels)
    step = 360.0 / n
    for i, p in enumerate(panels):
        ax.add_patch(mpatches.Wedge(xy, r, i * step, (i + 1) * step,
                                     facecolor=CANCER_COLOR[p], edgecolor=FG,
                                     linewidth=0.6, zorder=3))
    ax.add_patch(plt.Circle(xy, r, facecolor="none", edgecolor=FG,
                             linewidth=0.9, zorder=4))


def main(lean: bool = False) -> None:
    """lean=True collapses this figure's right sidebar entirely — the stats
    block, the 12-cancer colour key and the edge-type key all come off.

    Added 2026-08-31 (pass 2) for the merged slide-10.5 mosaic. The sidebar
    was ~2.2in of a ~10in canvas, so in the merge this panel's NETWORK was
    drawn about 20% smaller than the identically-positioned network in the
    panel beside it (fig11d, already lean) — two renders of the same layout
    at visibly different scale, which reads as two different networks.

    Nothing is lost: the stats fold into a single header line, and the
    cancer/edge keys move into the merged figure's shared legend band, where
    they are correctly attributed to this panel alone (the evidence-channel
    key does NOT describe this panel's edges — see that script).

    The standalone build (lean=False) is unchanged.
    """
    membership = load_membership()
    stats = load_stats()
    net = pd.read_csv(SRC / "network.tsv", sep="\t")
    edges = list(zip(net["preferredName_A"], net["preferredName_B"]))

    # Seventy-seven simultaneous labels cannot survive journal-width
    # reduction. Keep every node and edge, but label data-defined hubs plus
    # key locked candidates at a genuinely readable size.
    degree = {}
    for a, b in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    label_genes = {
        g for g, _ in sorted(degree.items(),
                             key=lambda kv: (-kv[1], kv[0]))[:8]
    }
    label_genes.update({"BMP4", "CXCL13"})

    svg_pos, svg_w, svg_h = load_svg_positions()
    png_img = plt.imread(PNG)
    png_h, png_w = png_img.shape[:2]
    r0, r1, c0, c1 = _autocrop_bounds(png_img)
    scale_x, scale_y = png_w / svg_w, png_h / svg_h

    # Node position in the SAME cropped-pixel space fig11d's image occupies,
    # so the two figures place each protein in matching spots.
    pos = {g: (x * scale_x - c0, y * scale_y - r0) for g, (x, y) in svg_pos.items()}
    node_r_px = 20 * scale_x  # STRING draws every node at SVG radius 20
    label_genes.intersection_update(pos)

    same_panel = sum(1 for a, b in edges if set(membership[a]) & set(membership[b]))
    cross_panel = len(edges) - same_panel

    crop_w, crop_h = c1 - c0, r1 - r0
    cx = np.mean([p[0] for p in pos.values()])
    left = sorted((g for g in label_genes if pos[g][0] < cx),
                  key=lambda g: pos[g][1])
    right = sorted((g for g in label_genes if pos[g][0] >= cx),
                   key=lambda g: pos[g][1])
    callout_targets = {}
    for genes, tx, ha in ((left, crop_w * 0.025, "left"),
                          (right, crop_w * 0.975, "right")):
        ys = np.linspace(crop_h * 0.18, crop_h * 0.82, max(len(genes), 1))
        for gene, ty in zip(genes, ys):
            callout_targets[gene] = (tx, ty, ha)
    img_h_in = 7.5   # tightened from 9.2, matches fig11d's tightened height
    img_w_in = img_h_in * (crop_w / crop_h)
    stats_col_in = 0.10 if lean else 2.2   # measured (scratch-figure text
                          # extent) from the longest legend label, was a flat
                          # 3.0in guess; collapsed entirely in lean mode
    margin_in = 0.20
    fig_w_in = img_w_in + stats_col_in + 2 * margin_in
    header_h_in = 0.20 if lean else 1.95
    fig_h_in = img_h_in + header_h_in + margin_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    fig.patch.set_facecolor(BG)
    img_bottom_in = margin_in
    ax = fig.add_axes([margin_in / fig_w_in, img_bottom_in / fig_h_in,
                        img_w_in / fig_w_in, img_h_in / fig_h_in])
    ax.set_facecolor(BG)
    ax.set_xlim(0, crop_w)
    ax.set_ylim(crop_h, 0)   # inverted: row 0 (SVG/PNG top) is the top of the axes
    ax.set_aspect("equal")
    ax.axis("off")

    for a, b in edges:
        cross = not (set(membership[a]) & set(membership[b]))
        x = [pos[a][0], pos[b][0]]
        y = [pos[a][1], pos[b][1]]
        if cross:
            ax.plot(x, y, color=CROSS_PANEL_EDGE_COLOR, linewidth=1.8, alpha=0.85, zorder=1)
        else:
            ax.plot(x, y, color=SAME_PANEL_EDGE_COLOR, linewidth=1.3, alpha=0.9, zorder=1)

    for gene, xy in pos.items():
        draw_node(ax, xy, membership[gene], r=node_r_px)
        if gene in label_genes:
            tx, ty, ha = callout_targets[gene]
            t = ax.annotate(
                gene, xy=xy, xytext=(tx, ty), textcoords="data",
                fontsize=14.5, fontweight="bold", ha=ha, va="center",
                color=FG, zorder=5,
                arrowprops=dict(arrowstyle="-", color="#7a7a7a", lw=0.7),
            )
            t.set_path_effects([
                patheffects.withStroke(linewidth=3.0, foreground="#ffffff")
            ])

    stats_x_in = margin_in + img_w_in + 0.15
    if lean:
        # Counts are added by the merged mosaic in a uniform footer shared
        # with panels B and D; keep this pre-render to the network itself.
        pass
    else:
        fig.text(
            margin_in / fig_w_in, (fig_h_in - 0.62) / fig_h_in,
            f"STRING v12.0 · {stats['nodes']} proteins pooled from all 12 panels (query-only, no added neighbours).",
            fontsize=10.5, fontweight="bold", color=MUTED, ha="left", va="top")
        fig.text(
            margin_in / fig_w_in, (fig_h_in - 0.90) / fig_h_in,
            f"{cross_panel} of {len(edges)} edges ({cross_panel/len(edges)*100:.0f}%) link proteins from TWO "
            "DIFFERENT panels (dark red) | the marker space is not 12 isolated silos.",
            fontsize=10.5, fontweight="bold", color=CROSS_PANEL_EDGE_COLOR, ha="left", va="top")
        fig.text(
            margin_in / fig_w_in, (fig_h_in - 1.18) / fig_h_in,
            "Pie-sliced nodes = gene shared by name across >1 panel (e.g. LEP in BRC/ENDC/PRC). "
            "Same node positions as fig11d.",
            fontsize=10, fontweight="bold", color=MUTED, ha="left", va="top")

        # Same stats block shape as fig10, so the two slides read as one system:
        # observed count large, random comparator + density + p beneath it.
        stats_y_in = img_bottom_in + img_h_in - 0.45
        fig.text(stats_x_in / fig_w_in, stats_y_in / fig_h_in,
                  f"{stats['edges']} edges",
                  fontsize=19, fontweight="bold", color=CROSS_PANEL_EDGE_COLOR,
                  ha="left", va="top")
        fig.text(stats_x_in / fig_w_in, (stats_y_in - 0.34) / fig_h_in,
                  f"{stats['expected']:.1f} expected\n"
                  f"{stats['density']:.1f}% of pairs\n"
                  f"STRING $p$ = {stats['p']:.3f}\n"
                  f"BH $q$ = {stats['q']:.3f}",
                  fontsize=11.5, fontweight="bold", color=MUTED,
                  ha="left", va="top", linespacing=1.55)

        handles = [mpatches.Patch(facecolor=CANCER_COLOR[p], edgecolor=FG, label=display_label(p))
                   for p in PANEL_ORDER]
        handles += [
            plt.Line2D([0], [0], color=CROSS_PANEL_EDGE_COLOR, lw=2.2, label="edge: different panels"),
            plt.Line2D([0], [0], color=SAME_PANEL_EDGE_COLOR, lw=2.2, label="edge: same panel"),
        ]
        leg = fig.legend(handles=handles, loc="upper left", ncol=1, frameon=False,
                          bbox_to_anchor=(stats_x_in / fig_w_in, (stats_y_in - 1.45) / fig_h_in),
                          fontsize=10, handlelength=1.6, handletextpad=0.5, labelspacing=0.55)
        for text in leg.get_texts():
            text.set_color(FG)

    stem = "fig11c_pooled12panels_network" + ("_lean" if lean else "")
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI, facecolor=BG)
        print(f"  saved -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    print("Building fig11c (pooled 12-panel PPI network, coloured by panel) ...")
    main()
    print("Done.")
