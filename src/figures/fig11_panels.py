#!/usr/bin/env python3
"""
Fig 11 — SLIDE 11: PPI networks, the 12 disease-specific panels (Part A).

New 2026-08-23. Replaces manual STRING-website screenshots with the
official STRING v12.0 API export a collaborator produced in
`revise_plan/ppi_string/` (scripts `batch_string_export.py` and
`download_STRING_analysis_12_panels.py`) -- reproducible, not redrawn
locally. This script only composes the 12 panels into one deck-ready
mosaic with real statistics annotated on every tile; it does not recompute
any network.

Mode shown: query_only (add_nodes=0) for every panel -- each panel's own
proteins only, no STRING-added neighbours. First_shell_max5 mode lets
STRING pick the highest-scoring neighbours to add, which inflates
apparent connectivity and is not a fair test of whether the SELECTED
panel shows more interaction than chance -- kept on disk for exploratory
reference, not part of this figure's claim.

CURRENT RESULT, 2026-09-04: statistics come from STRING's official
background-aware endpoint with 1,452 mapped assay proteins. CLL has the
smallest unadjusted p = 0.0124, but no panel survives BH correction across
14 tested sets. The figure supports a sparse-network description only.

LAYOUT HISTORY (all 2026-08-23) -- six rounds of user feedback, each a
real problem, not a nitpick; full account in fig_v5_changelog.md:
  v1  wasted ~half the figure as white space (autocrop bug: RGBA
      transparent background stored as black at alpha=0, read as "content").
  v2  fixed crop, then solved row-height-to-fill-width -> node circles came
      out wildly different sizes across panels (an artifact of row
      membership, not the data).
  v3  one shared px->inch scale for consistent node size -> mosaic looked
      uneven (variable tile sizes); user asked for the fig2f_umap.py look
      (identical-size boxes) instead.
  v4  uniform grid via fig.add_axes(), but imshow's default aspect="equal"
      with adjustable="box" (matplotlib default) resizes the AXES itself
      to match each image's aspect -- identical rects still rendered as
      different box sizes. Fixed with adjustable="datalim" (box stays
      fixed, image is letterboxed inside it).
  v5  legend/description pass: added plain-language glosses, cut a
      misleading "empty node" legend example (every node in this dataset
      is filled -- AlphaFold covers virtually the whole human proteome --
      so an example that never appears in the figure only confuses).
  v6 (this version) -- per-tile text simplified (title = cancer code +
      significance stars only; protein/edge counts dropped -- countable
      by eye -- and p-value moved from the title block into a small
      in-box annotation, upper-left corner, next to the network it
      describes rather than stacked in a header above everything). Legend
      unified into ONE row (marker + 7 edge-colour lines together, single
      short label each) instead of two separately-titled blocks.

WHY NODE FILL COLOUR IS NOT REMOVED: asked whether the decorative fill
could be dropped so only the border is coloured (a plain white node with
a coloured ring). Checked STRING's own image API for a parameter that
does this -- it offers only "default" (structure-preview cartoon on a
coloured disc) and "flat_node_design" (plain single-colour disc, no
cartoon), neither of which is "hollow, border only." Achieving that would
mean pixel-editing STRING's own rendered PNG -- detecting each node's
circular region and stripping its fill while preserving the 3D-structure
cartoon drawn on top of it, which is NOT decorative (STRING documents it
as meaningful: filled = structure available). That is a fragile,
error-prone image-processing task for a cosmetic change, and risks
damaging the one part of node appearance that actually carries
information. Left as-is; the legend already discloses that fill colour
itself (not the cartoon) is decorative.

Source: revise_plan/ppi_string/
    string_outputs_disease_specific/official_figures/query_only/<PANEL>/<PANEL>_query_only_official_STRING_v12.png
    STRING_analysis_official_12_panels/00_common/PPI_statistics_all_12_panels.csv
Output: figurev5/output/fig11_disease_panels_ppi_mosaic.pdf/.png
Run:    python fig11_panels.py
"""
from __future__ import annotations

from pathlib import Path
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_labels import display_label
from publication_style import enable_text_scaling

enable_text_scaling(factor=1.25, minimum=10.5)

ROOT = Path(__file__).resolve().parents[2]
PPI = ROOT / "revise_plan/ppi_string"
STATS_CSV = PPI / "STRING_analysis_official_12_panels/00_common/PPI_statistics_all_12_panels.csv"
# Reaudited official assay-background statistics and local sensitivity test.
REAUDIT_CSV = PPI / "ppi_reaudit_combined_results_v2.csv"
IMG_DIR = PPI / "string_outputs_disease_specific/official_figures/query_only"
OUT = ROOT / "figurev6/output"
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

# STRING's fixed evidence-view edge colours. Formal STRING names (Gene
# fusion, Gene neighborhood, Gene co-occurrence, Experiments, Text mining,
# Curated databases, Co-expression -- see STRING_evidence_view_legend.md)
# are kept here as a code comment for provenance/traceability, but the
# FIGURE shows only the plain-language phrase: showing both together
# ("Name (gloss)") for all 7 was the "messy, parenthesised" clutter this
# version removes.
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
TITLE_SPACE_IN = 0.45   # panel A keeps only the cancer name above each tile
                         # shrunk from 0.55in now that protein/edge counts
                         # no longer live in the title block
MARGIN_IN = 0.18
HEADER_H_IN = 0.25       # figure-level heading removed in Figure v6
                          # per-tile p-values moved out of the header)
LEGEND_H_IN = 0.55        # ONE unified row: node marker + 7 edge colours

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

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


def _readable_string_render(code: str) -> Path:
    """Render a derived copy of STRING's SVG with labels clear of nodes.

    STRING places labels only 18 px diagonally from a 20 px-radius node, so
    letters at the lower-left of a label can be covered by the circle.  The
    derived SVG moves labels slightly farther away and enlarges them; the
    source API exports remain untouched.
    """
    src = IMG_DIR / code / f"{code}_query_only_official_STRING_v12.svg"
    derived = OUT / "fig10_panelA_readable_sources"
    derived.mkdir(parents=True, exist_ok=True)
    svg_out = derived / f"{code}_readable.svg"
    png_out = derived / f"{code}_readable.png"

    text = src.read_text(encoding="utf-8")
    text = text.replace(
        "text { font-family: Arial; font-size: 12px; }",
        "text { font-family: Arial; font-size: 19px; font-weight: 700; }",
    )
    header = text[:500]
    width = int(re.search(r"width='(\d+)'", header).group(1))
    height = int(re.search(r"height='(\d+)'", header).group(1))

    nodes = [(m.group(1), int(m.group(2)), int(m.group(3))) for m in re.finditer(
        r"data-safe_div_label='([^']+)'.*?data-x_pos='(-?\d+)'.*?data-y_pos='(-?\d+)'",
        text, flags=re.S)]
    node_xy = [(x, y) for _, x, y in nodes]
    placed_boxes: list[tuple[float, float, float, float]] = []

    def overlap_area(a, b) -> float:
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * \
               max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

    def move_labels(match: re.Match) -> str:
        chunk = match.group(0)
        gene = re.search(r"data-safe_div_label='([^']+)'", chunk).group(1)
        x = int(re.search(r"data-x_pos='(-?\d+)'", chunk).group(1))
        y = int(re.search(r"data-y_pos='(-?\d+)'", chunk).group(1))
        font_h = 20.0
        label_w = max(30.0, len(gene) * 10.5)
        gap = 10.0
        radius = 20.0
        # baseline x/y, anchor. Diagonals first retain STRING's familiar look;
        # the alternatives let crowded panels route labels into open space.
        candidates = [
            (x + radius + gap, y - radius - gap, "start"),
            (x - radius - gap, y - radius - gap, "end"),
            (x + radius + gap, y + radius + gap + font_h, "start"),
            (x - radius - gap, y + radius + gap + font_h, "end"),
            (x, y - radius - gap, "middle"),
            (x, y + radius + gap + font_h, "middle"),
            (x + radius + gap, y + font_h * 0.35, "start"),
            (x - radius - gap, y + font_h * 0.35, "end"),
        ]

        best = None
        for rank, (tx, ty, anchor) in enumerate(candidates):
            if anchor == "start":
                x0, x1 = tx, tx + label_w
            elif anchor == "end":
                x0, x1 = tx - label_w, tx
            else:
                x0, x1 = tx - label_w / 2, tx + label_w / 2
            box = (x0, ty - font_h, x1, ty + 2)
            outside = (max(0, 4 - x0) + max(0, x1 - width + 4) +
                       max(0, 4 - box[1]) + max(0, box[3] - height + 4))
            node_penalty = 0.0
            for nx, ny in node_xy:
                node_box = (nx - radius - 2, ny - radius - 2,
                            nx + radius + 2, ny + radius + 2)
                node_penalty += overlap_area(box, node_box)
            label_penalty = sum(overlap_area(box, prev) for prev in placed_boxes)
            score = outside * 10000 + node_penalty * 100 + label_penalty * 500 + rank
            if best is None or score < best[0]:
                best = (score, tx, ty, anchor, box)

        _, tx, ty, anchor, box = best
        placed_boxes.append(box)
        chunk = re.sub(r"text-anchor='start' x='-?\d+' y='-?\d+'",
                       f"text-anchor='{anchor}' x='{tx:.1f}' y='{ty:.1f}'", chunk)
        return chunk

    text = re.sub(r"<g class='nwnodecontainer'.*?(?=<g class='nwnodecontainer'|</svg>)", move_labels,
                  text, flags=re.S)
    svg_out.write_text(text, encoding="utf-8")

    if not CHROME.exists():
        raise FileNotFoundError(f"Chrome is required to render adjusted SVG: {CHROME}")
    subprocess.run([
        str(CHROME), "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--force-device-scale-factor=2", f"--window-size={width},{height}",
        f"--screenshot={png_out}", svg_out.resolve().as_uri(),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return png_out


def load_stats() -> pd.DataFrame:
    d = pd.read_csv(STATS_CSV)
    return d[d["mode"] == "query_only"].set_index("panel")


def load_reaudit() -> pd.DataFrame:
    return pd.read_csv(REAUDIT_CSV).set_index("panel")


# Significance stars removed 2026-08-24. Against the background-matched null
# only DLBCL and CLL would carry one, and CLL's "network" is a single edge
# between two proteins -- a star there implies a finding the data does not
# support. The observed-vs-random pair printed on each tile shows the reader
# the comparison directly instead of encoding it as a symbol.
#
# NOTE: density (% of possible pairs) is deliberately NOT shown per tile --
# it is meaningless at these sizes (CLL's 2 proteins with 1 edge is "100%
# dense"). Density appears only on the large networks, fig10 and fig11c/d.


def _legend(fig, y_center_in: float, fig_h_in: float) -> None:
    """One unified row -- node marker first, then the 7 edge colours --
    instead of two separately-titled legend blocks. Short single-phrase
    labels only (see EVIDENCE_CHANNELS comment for the formal STRING
    names this intentionally omits from the figure itself)."""
    handles = [
        plt.Line2D([0], [0], marker="o", ls="", ms=12, mfc="#8ab4d6",
                   mec="#222222", mew=1.2, label="protein (color decorative)"),
    ] + [
        plt.Line2D([0], [0], color=c, lw=3.0, label=gloss)
        for c, gloss in EVIDENCE_CHANNELS
    ]
    fig.legend(handles=handles, loc="center", ncol=8, frameon=False,
               bbox_to_anchor=(0.5, y_center_in / fig_h_in),
               fontsize=9.5, handlelength=1.6, handletextpad=0.5,
               columnspacing=1.3)


def main(lean: bool = False) -> None:
    """lean=True drops this figure's own STRING evidence legend and the
    header line that points at it.

    Added 2026-08-31 (pass 2) for the merged slide-10.5 mosaic. That figure
    draws ONE evidence key for all the panels that use evidence colouring;
    this figure's own copy was the third printing of the same eight rows.
    The standalone build (lean=False) is unchanged.
    """
    stats = load_stats()
    reaudit = load_reaudit()

    rows = ([sum(ROWS, [])[:6], sum(ROWS, [])[6:]] if lean else ROWS)
    n_rows, n_cols = len(rows), len(rows[0])
    all_codes = [c for row in rows for c in row]
    imgs: dict[str, np.ndarray] = {}
    for code in all_codes:
        png = _readable_string_render(code)
        imgs[code] = _autocrop(plt.imread(png))

    legend_h_in = 0.0 if lean else LEGEND_H_IN
    row_pitch_in = CELL_H_IN + TITLE_SPACE_IN + GAP_IN
    fig_w_in = n_cols * CELL_W_IN + (n_cols - 1) * GAP_IN + 2 * MARGIN_IN
    fig_h_in = HEADER_H_IN + n_rows * row_pitch_in - GAP_IN + legend_h_in + MARGIN_IN
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    grid_top_in = fig_h_in - HEADER_H_IN

    for r, row in enumerate(rows):
        for c, code in enumerate(row):
            x0_in = MARGIN_IN + c * (CELL_W_IN + GAP_IN)
            y1_in = grid_top_in - TITLE_SPACE_IN - r * row_pitch_in
            y0_in = y1_in - CELL_H_IN
            ax = fig.add_axes([x0_in / fig_w_in, y0_in / fig_h_in,
                                CELL_W_IN / fig_w_in, CELL_H_IN / fig_h_in])
            ax.imshow(imgs[code])
            ax.set_aspect("equal", adjustable="datalim")  # keep box fixed
            ax.set_xticks([])
            ax.set_yticks([])
            col = CANCER_COLOR[code]
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(col)
                spine.set_linewidth(2.2)

            label = display_label(code)
            ax.text(0.5, 1.04, label, transform=ax.transAxes,
                    fontsize=17, fontweight="bold", color=col,
                    ha="center", va="bottom", clip_on=False)

    if not lean:
        _legend(fig, MARGIN_IN + LEGEND_H_IN * 0.5, fig_h_in)

    stem = "fig11_disease_panels_ppi_mosaic" + ("_lean" if lean else "")
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI)
        print(f"  saved -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    print("Building fig11 (slide 11 — 12 disease-panel PPI mosaic) ...")
    main()
    print("Done.")
