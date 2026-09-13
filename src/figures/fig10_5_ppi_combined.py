#!/usr/bin/env python3
"""
fig10_5_ppi_combined.py — slides 10 and 11 merged into ONE 4-panel figure.

WHY
---
Reviewed 2026-08-31. Slides 10 and 11 were two separate side-by-side
composites (`build_ppi_slides.py`), and between them they printed the same
things over and over:

  * the 8-row STRING evidence-channel legend appeared THREE times —
    fig11 (bottom strip), fig10 (right sidebar), fig11d (right sidebar);
  * the pooled-network statistics appeared twice,
    on fig11c and fig11d, which are the SAME network drawn two ways;
  * "STRING v12.0 · 77 proteins pooled from all 12 panels (query-only, no
    added neighbours)" appeared twice, for the same reason;
  * neither slide carried panel letters at all.

Printed side by side with duplicated stat blocks, fig11c and fig11d read as
two different results rather than one result rendered two ways — the exact
confusion the merge is meant to remove.

GEOMETRY — CHECKED BEFORE COMMITTING TO THIS
--------------------------------------------
The worry with a 2x2 merge was the 12-panel mosaic (panel a), whose native
width is 7608 px and whose per-protein labels are already small. Measured:

  on the old slide 10, it was displayed at ~1860 px  = 24% of native
  in this merged 2x2 at 3600 px wide, it displays at 1800 px = 24% of native

i.e. the merge costs panel a essentially nothing, because slide 10 was
already scaling it down by the same factor to sit beside fig10. Verified by
rendering the composite and reading the labels before this script was
written — not assumed.

WHAT THIS SAVES
---------------
Two of the three evidence legends, one full stats block, one methods line,
and one whole slide. fig11d is rendered in `lean` mode (see its own
docstring) which drops its entire right sidebar; the one surviving evidence
legend is drawn here, once, for all four panels.

PANELS
------
  a  12 disease-specific panels, one small network each      (Part A)
  b  locked-25 panel as one network                          (Part B)
  c  all 12 panels pooled, coloured by which panel           (Part A pooled)
  d  the same pooled network, STRING's own native render

Panel c keeps the pooled-network statistics and the 12-cancer color key.
Panel b keeps the locked-25 statistics. Both use the reaudited official
assay-background result and its 14-test BH correction.

Source: figurev5/output/{fig11_disease_panels_ppi_mosaic,
    fig10_locked25_ppi_network, fig11c_pooled12panels_network,
    fig11d_pooled12panels_official_lean}.png
Output: figurev5/output/fig10_5_ppi_combined.png (+ .pdf)
Run:    python fig11d_pooled12panels_official.py   # then, for the lean one:
        python -c "import fig11d_pooled12panels_official as m; m.main(lean=True)"
        python fig10_5_ppi_combined.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from publication_style import enable_text_scaling

enable_text_scaling(factor=1.25, minimum=11.5)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figurev6" / "output"

DPI = 400
FG = "#111111"
MUTED = "#5d6b78"

# The single surviving copy of STRING's evidence-channel key. Values copied
# from fig11d_pooled12panels_official.EVIDENCE_CHANNELS so the one legend
# drawn here matches what the panels above it actually show.
EVIDENCE_CHANNELS = [
    ("#e8384f", "fusion"),
    ("#4a9c3f", "neighborhood"),
    ("#4f5fd0", "co-occurrence"),
    ("#a12ec4", "experiments"),
    ("#d9a520", "text mining"),
    ("#3fc4e8", "databases"),
    ("#111111", "co-expression"),
]

PANELS = [
    ("fig11_disease_panels_ppi_mosaic_lean",    "A"),
    ("fig10_locked25_ppi_network_lean",         "B"),
    ("fig11c_pooled12panels_network_lean",      "C"),
    ("fig11d_pooled12panels_official_lean",     "D"),
]

PANEL_NAMES = {
    "A": "Cancer-specific networks",
    "B": "Locked 25",
    "C": "Pooled · panel colors",
    "D": "Pooled · STRING evidence",
}
PANEL_STATS = {
    "B": "7 observed · 8 expected edges",
    "C": "88 observed · 80 expected edges",
    "D": "88 observed · 80 expected edges",
}
# The composite itself carries only panel letters.  Detailed descriptions
# belong in the manuscript caption rather than in a repeated title band.
PANEL_NAMES = {k: "" for k in ("A", "B", "C", "D")}
# Edge counts are compact, local panel annotations. They belong beneath the
# corresponding networks, rather than as one detached sentence in the footer.

# Panel c's edges are NOT coloured by STRING evidence channel — they are grey
# / dark-red by whether the two proteins come from the same disease panel. The
# first version of this figure printed "one key for all four panels" over the
# evidence legend, which was simply wrong for panel c. The legend band now has
# two correctly-attributed halves; this is panel c's half.
CANCER_COLOR = {
    "AML": "#b22222", "BRC": "#c97b63", "CLL": "#7a3e9d", "CRC": "#d68600",
    "CVX": "#c13d86", "ENDC": "#8e5d2c", "GLIOM": "#1a8db8", "LUNGC": "#2e8b57",
    "DLBCL": "#3856a6", "MYEL": "#8c564b", "OVC": "#d1495b", "PRC": "#008b8b",
}
CROSS_PANEL_EDGE_COLOR = "#8b1a1a"
SAME_PANEL_EDGE_COLOR = "#999999"


def _crop_white_margin(img, pad=12):
    """Remove only the uniform outer whitespace left by deleted headings."""
    rgb = img[..., :3]
    mask = np.any(rgb < 0.985, axis=2)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(img.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(img.shape[1], int(xs.max()) + pad + 1)
    return img[y0:y1, x0:x1]


def main() -> None:
    imgs = []
    missing = []
    for stem, letter in PANELS:
        p = OUT / f"{stem}.png"
        if not p.exists():
            missing.append(stem)
            continue
        imgs.append((_crop_white_margin(plt.imread(p)), letter))
    if missing:
        raise SystemExit(
            "missing lean pre-renders: " + ", ".join(missing) + "\n"
            "build them first:\n"
            '  python -c "import fig11_panels as m; m.main(lean=True)"\n'
            '  python -c "import fig10_panels as m; m.main(lean=True)"\n'
            '  python -c "import fig11c_pooled12panels_network as m; m.main(lean=True)"\n'
            '  python -c "import fig11d_pooled12panels_official as m; m.main(lean=True)"')

    # Panel D is intentionally retained. Panel C is a local redraw coloured
    # by cancer-panel membership, whereas D is STRING's native pooled render;
    # they answer different visual questions and are not interchangeable.
    # A is a wide 6×2 strip and uses roughly half the figure height.  B gets
    # the largest lower-row cell; C and D use equal cells because they are
    # two views of the same pooled network.
    # Compact construction raises vector text at final journal width. Lower
    # widths follow native aspects so B-D have exactly equal displayed height.
    fig_w_in = 18.0
    # A larger inter-row gutter gives B--D their own letter space rather than
    # allowing their labels to sit on Panel A's bottom border.
    GUT_IN, EDGE_IN, TOP_IN = 0.70, 0.14, 0.23
    avail_w_in = fig_w_in - 2 * EDGE_IN - GUT_IN

    def aspect(img):
        return img.shape[1] / img.shape[0]

    (img_a, letter_a), (img_b, letter_b), (img_c, letter_c), \
        (img_d, letter_d) = imgs
    top_w = fig_w_in - 2 * EDGE_IN
    top_h = top_w / aspect(img_a)
    lower_aspects = [aspect(img) for img in (img_b, img_c, img_d)]
    aspect_total = sum(lower_aspects)
    bottom_widths = [avail_w_in * a / aspect_total for a in lower_aspects]
    bottom_h = max(w / aspect(img) for w, img in
                   zip(bottom_widths, (img_b, img_c, img_d)))
    bottom_items = [(bottom_widths[0], img_b, letter_b),
                    (bottom_widths[1], img_c, letter_c),
                    (bottom_widths[2], img_d, letter_d)]

    legend_h_in = 1.62
    fig_h_in = TOP_IN + top_h + GUT_IN + bottom_h + legend_h_in + 0.14

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    fig.patch.set_facecolor("white")

    y_top_in = fig_h_in - TOP_IN

    def place(x_in, y_top, w_in, img, letter):
        own_h = w_in / aspect(img)
        ax = fig.add_axes([x_in / fig_w_in, (y_top - own_h) / fig_h_in,
                           w_in / fig_w_in, own_h / fig_h_in])
        ax.imshow(img)
        ax.axis("off")
        ax.text(0.002, 1.008, f"{letter}  {PANEL_NAMES[letter]}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=25, fontweight="bold",
                color=FG,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.0))
        if ax.texts:
            ax.texts[-1].set_visible(False)
        ax.text(0.002, 1.008, letter, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=25, fontweight="bold",
                color=FG, clip_on=False)
        if letter in PANEL_STATS:
            ax.text(0.50, -0.030, PANEL_STATS[letter], transform=ax.transAxes,
                    ha="center", va="top", fontsize=10.5, fontweight="bold",
                    color="black", clip_on=False,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.45))

    place(EDGE_IN, y_top_in, top_w, img_a, letter_a)
    y_bottom_top = y_top_in - top_h - GUT_IN
    x_in = EDGE_IN
    for w_in, img, letter in bottom_items:
        place(x_in, y_bottom_top, w_in, img, letter)
        x_in += w_in + GUT_IN

    cd_center = EDGE_IN + bottom_widths[0] + GUT_IN + \
        (bottom_widths[1] + GUT_IN + bottom_widths[2]) / 2
    if False: fig.text(0.5,
             (legend_h_in - 0.03) / fig_h_in,
             "B: 7 observed · 8 expected edges   |   C/D: 88 observed · 80 expected edges",
             ha="center", va="top", fontsize=16, fontweight="bold",
             color="black")

    # ── the shared legend band: TWO keys, each attributed ────────────────
    # Pass 1 drew only the evidence key and captioned it "one key for all four
    # panels". That was wrong: panel c's edges are coloured by whether the two
    # proteins come from the same disease panel, not by evidence channel. A
    # reader matching panel c's dark-red edges against "fused gene (other
    # species)" would have read a real result backwards.
    ev_handles = [
        plt.Line2D([0], [0], marker="o", ls="", ms=11, mfc="#8ab4d6",
                   mec=FG, mew=1.0, label="protein"),
    ] + [
        plt.Line2D([0], [0], color=c, lw=5.0, label=gloss)
        for c, gloss in EVIDENCE_CHANNELS
    ]
    leg_ev = fig.legend(handles=ev_handles, loc="upper center", ncol=8,
                        frameon=False, fontsize=13, handlelength=1.55,
                        handletextpad=0.4, columnspacing=1.25,
                        bbox_to_anchor=(0.5, (legend_h_in - 0.45) / fig_h_in))
    for t in leg_ev.get_texts():
        t.set_color(FG)
    fig.add_artist(leg_ev)

    panel_handles = [
        plt.Line2D([0], [0], marker="o", ls="", ms=11, mfc=col, mec=FG,
                   mew=0.8, label=name)
        for name, col in CANCER_COLOR.items()
    ] + [
        plt.Line2D([0], [0], color=CROSS_PANEL_EDGE_COLOR, lw=5.0,
                   label="cross-panel"),
        plt.Line2D([0], [0], color=SAME_PANEL_EDGE_COLOR, lw=5.0,
                   label="within-panel"),
    ]
    leg_pn = fig.legend(handles=panel_handles, loc="upper center", ncol=8,
                        frameon=False, fontsize=12.5, handlelength=1.35,
                        handletextpad=0.35, columnspacing=0.85,
                        bbox_to_anchor=(0.5, (legend_h_in - 0.88) / fig_h_in))
    for t in leg_pn.get_texts():
        t.set_color(FG)

    for ext in ("png", "pdf"):
        p = OUT / f"fig10_5_ppi_combined.{ext}"
        fig.savefig(p, dpi=DPI, facecolor="white", bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    print("Building fig10.5 — slides 10 + 11 merged, one shared legend ...")
    main()
    print("Done.")
