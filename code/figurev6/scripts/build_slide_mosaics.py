#!/usr/bin/env python3
"""
build_slide_mosaics.py — deck-ready mosaics for slides 2, 4, 5, 6 and 7.

Two different problems, two different tools
-------------------------------------------
Slides 3, 8 and 9 are filmstrip slides: several rows/columns that all share
the SAME 12-cancer structure, so a reader must be able to track one cancer
straight across or down. Those cannot be built by pasting finished PNGs --
each source figure sets its own margins, so the columns drift. They are
rebuilt from tile functions into one shared GridSpec by
`fig3_slide3_mosaic.py` and `fig89_stacked_mosaic.py`.

Slides 2, 4, 5, 6 and 7 are different: each is a handful of INDEPENDENT
figures (a cohort bar chart, a UMAP, a chord diagram, a Sankey, a
calibration plot) that share no common axis and need no cross-panel
alignment. For those, compositing the already-final rendered PNGs is the
correct and lowest-risk tool -- it reuses the exact images the deck already
contains, and no plotting code is touched or duplicated. Same approach
`build_ppi_slides.py` uses for slides 10 and 11.

Layouts are declarative (see SLIDES below). Each panel is placed by a
row/column span on a simple grid; its own aspect ratio is preserved and the
cell is scaled to fit. Panel letters (a, b, c...) are drawn automatically in
the order the panels are listed, so the caption can refer to them.

Output: figurev5/output/fig{2,4,5,6,7}_slide{N}_*_mosaic.png
Run:    python build_slide_mosaics.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figurev6" / "output"

TARGET_W = 4800      # 12-inch composite width at 400 dpi
MARGIN = 35          # gutter between panels, scaled from the 300-dpi layout
EDGE = 27            # outer border, scaled from the 300-dpi layout
TITLE_H = 0          # manuscript caption supplies the figure title
BG = (255, 255, 255)

LABEL_SIZE = 96      # panel-letter size, scaled from the 300-dpi layout
LABEL_PAD = 13       # inset of the letter from the panel's top-left corner
NAME_SIZE = 48       # short factual panel name; interpretation stays in caption
PANEL_HEADER_H = 112 # reserved only on mosaics that request panel names
HEADER_LABEL_SIZE = 82

FIG7_SHARED_LEGEND = [
    ("AML", "#b22222"), ("BRC", "#c97b63"), ("CLL", "#7a3e9d"),
    ("CRC", "#d68600"), ("CVX", "#c13d86"), ("ENDC", "#8e5d2c"),
    ("GLIOM", "#1a8db8"), ("LUNGC", "#2e8b57"),
    ("DLBCL", "#3856a6"), ("MYEL", "#8c564b"),
    ("OVC", "#d1495b"), ("PRC", "#008b8b"),
    ("Treat all", "#bbbbbb"), ("Treat none", "#555555"),
]

FIG5_SHARED_LEGEND = [
    ("AML", "#b22222"), ("BRC", "#c97b63"), ("CLL", "#7a3e9d"),
    ("CRC", "#d68600"), ("CVX", "#c13d86"), ("ENDC", "#8e5d2c"),
    ("GLIOM", "#1a8db8"), ("LUNGC", "#2e8b57"),
    ("DLBCL", "#3856a6"), ("MYEL", "#8c564b"),
    ("OVC", "#d1495b"), ("PRC", "#008b8b"),
    ("Macro", "#A06CD5"), ("Treat all", "#bbbbbb"),
    ("Treat none", "#555555"),
]


def _font(size: int):
    for name in ("arialbd.ttf", "Arial_Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Layout specs.  panels: (stem, row, col, rowspan, colspan)
#   grid         (n_rows, n_cols)
#   row_weights  scale a row's natural height
#   col_weights  scale a column's width (e.g. give a heatmap more room than
#                the bar chart beside it)
# Panel letters are assigned in listing order, so list panels in the reading
# order you want them lettered.
# ---------------------------------------------------------------------------
SLIDES = {
    "fig2_slide2_cohort_mosaic": dict(
        title="Cohort and data distribution   ·   1,375 plasma samples   ·   "
              "12 cancer types   ·   1,463 Olink proteins",
        grid=(2, 2),
        # Figure 2 is sourced from the approved v5 panel renders.  The
        # slightly heavier bottom row gives the pooled-distribution panel C
        # the extra reading area requested for the final mosaic.
        source_out=ROOT / "figurev5" / "output",
        source_paths={
            "fig2a_cohort": OUT / "fig2a_cohort.png",
            "fig2b_npx_heatmap": OUT / "fig2b_npx_heatmap.png",
            "fig2b_pooled_distributions": OUT / "fig2b_pooled_distributions.png",
            "fig2f_umap": OUT / "fig2f_umap.png",
        },
        # the NPX heatmap and the UMAP carry far more detail than the cohort
        # bar chart, so the right column gets the extra width
        col_weights=[0.94, 1.06],
        row_weights=[1.00, 1.10],
        # A slim, blank letter band keeps A--D out of the data region.  It is
        # deliberately separate from a panel title: the manuscript caption
        # supplies interpretation, while each plot retains its own axes.
        label_band=104,
        panels=[
            ("fig2a_cohort",               0, 0, 1, 1),   # a
            ("fig2b_npx_heatmap",          0, 1, 1, 1),   # b
            ("fig2b_pooled_distributions", 1, 0, 1, 1),   # c
            ("fig2f_umap",                 1, 1, 1, 1),   # d
        ],
    ),
    # fig4_slide4_structure_mosaic is no longer built here. Figure 4 was
    # rebuilt around the panel-size evidence (why 25 markers, and the size
    # rule for each disease-specific panel); the concordance scatter it used
    # to show is now text-only. Its dedicated builder is
    # fig4_7_sweep_structure_and_pairs.py -- run that instead of adding this
    # slide back, or a full rerun of this file will silently regress Figure 4
    # to the pre-2026-09-12 three-panel version.
    "fig5_slide5_multiclass_mosaic": dict(
        title="Internal evaluation: 25-protein multiclass panel | "
              "413 held-out samples, 75.3% correctly classified | "
              "1,000 bootstrap resamples",
        grid=(2, 2),
        # REBUILT 2026-08-31. Two changes, both to buy the radar its space:
        #
        # 1. THE SANKEY IS GONE. fig5a_confusion_sankey and fig5d_score_matrix
        #    answer the same question — which cancers get confused with which.
        #    The Sankey spent two full cells (~40% of the slide) doing it in
        #    ribbons nobody can measure; the score matrix does it in exact
        #    per-cell numbers in one cell. The Sankey's one unique fact, the
        #    overall 311/413 correct split, is now in the slide title above.
        # 2. fig5c IS SPLIT. It stacked the radar over the per-cancer strip in
        #    one portrait image, giving the radar 0.270 of the height against
        #    the strip's 0.465 — backwards, and the reason the radar's CI
        #    numbers were unreadable. Now fig5f (radar, square, fonts x1.55)
        #    and fig5g (strip, 16in wide) are separate panels, each shaped the
        #    way its own content wants.
        # PASS 2, 2026-08-31: equal columns. The three top panels were
        # re-cut to near-identical saved aspect ratios (1.102 / 1.079 /
        # 1.090), so equal cell widths now render them at genuinely
        # matching size — previously b and c were both smaller than a AND
        # different sizes from each other, because unequal weights were
        # compensating for unequal source aspects.
        col_weights=[1.00, 1.00],
        shared_legend="fig5",
        panels=[
            ("fig5f_macro_radar",      0, 0, 1, 1),   # a, the headline panel
            ("fig5d_score_matrix",     0, 1, 1, 1),   # b
            ("fig5e_decision_curve",   1, 0, 1, 1),   # c
            ("fig5g_percancer_violin", 1, 1, 1, 1),   # d, beside c
        ],
    ),
    "fig6_slide6_pathway_mosaic": dict(
        title="Pathway interpretation   ·   over-representation of the 25-protein "
              "panel against the measured 1,463-protein assay",
        grid=(2, 2),
        row_weights=[1.00, 1.08],
        panels=[
            ("fig6_pathway_enrichment", 0, 0, 1, 2),   # a, full-width GO/KEGG/Reactome
            ("fig6_chord_network",      1, 0, 1, 1),   # b
            ("fig6_parta_sankey",       1, 1, 1, 1),   # c
        ],
    ),
    "fig7_slide7_singleclass_mosaic": dict(
        title="Internal evaluation: cancer-specific panels | "
              "fixed threshold 0.50",
        grid=(2, 2),
        # Equal columns keep B/C axis titles, ticks, and legends at the same
        # apparent size in the final composite.
        col_weights=[1.00, 1.00],
        shared_legend="fig7",
        panels=[
            ("fig7d_confusion_grid", 0, 0, 1, 2),   # A, full-width 12-cohort strip
            ("fig7h_calibration",    1, 0, 1, 1),   # B
            ("fig7i_decision_curve", 1, 1, 1, 1),   # C
        ],
    ),
}


def build(stem: str, spec: dict) -> None:
    n_rows, n_cols = spec["grid"]
    row_w = spec.get("row_weights", [1.0] * n_rows)
    col_w = spec.get("col_weights", [1.0] * n_cols)

    source_out = spec.get("source_out", OUT)
    imgs = {}
    for name, *_ in spec["panels"]:
        p = spec.get("source_paths", {}).get(name, source_out / f"{name}.png")
        if not p.exists():
            raise FileNotFoundError(f"missing source panel: {p}")
        imgs[name] = Image.open(p).convert("RGB")

    inner_w = TARGET_W - 2 * EDGE - MARGIN * (n_cols - 1)
    tot_cw = sum(col_w)
    cell_w = [inner_w * w / tot_cw for w in col_w]

    def span_width(c, cs):
        return sum(cell_w[c:c + cs]) + MARGIN * (cs - 1)

    def col_left(c):
        return EDGE + sum(cell_w[:c]) + MARGIN * c

    # Row heights: each row is as tall as the tallest panel it must hold,
    # after that panel has been scaled to its column span, then weighted.
    row_h = [0.0] * n_rows
    for name, r, c, rs, cs in spec["panels"]:
        im = imgs[name]
        h = span_width(c, cs) * im.height / im.width
        for rr in range(r, r + rs):
            row_h[rr] = max(row_h[rr], h / rs)
    row_h = [h * w for h, w in zip(row_h, row_w)]
    header_h = PANEL_HEADER_H if spec.get("panel_names") else 0
    label_band = int(spec.get("label_band", 0))
    reserved_h = header_h + label_band
    if reserved_h:
        row_h = [h + reserved_h for h in row_h]

    shared_legend_h = 230 if spec.get("shared_legend") else 0
    total_h = int(TITLE_H + sum(row_h) + MARGIN * (n_rows - 1)
                   + 2 * EDGE + shared_legend_h)
    canvas = Image.new("RGB", (TARGET_W, total_h), BG)
    draw = ImageDraw.Draw(canvas)

    def row_top(r):
        return EDGE + TITLE_H + sum(row_h[:r]) + MARGIN * r

    lab_font = _font(LABEL_SIZE)
    header_lab_font = _font(HEADER_LABEL_SIZE)
    name_font = _font(NAME_SIZE)
    for idx, (name, r, c, rs, cs) in enumerate(spec["panels"]):
        im = imgs[name]
        sw = span_width(c, cs)
        sh = sum(row_h[r:r + rs]) + MARGIN * (rs - 1)
        content_y = row_top(r) + reserved_h
        content_h = sh - reserved_h
        scale = min(sw / im.width, content_h / im.height)
        new = im.resize((max(1, int(im.width * scale)),
                         max(1, int(im.height * scale))), Image.LANCZOS)
        x = col_left(c) + (sw - new.width) / 2
        y = content_y + (content_h - new.height) / 2
        canvas.paste(new, (int(x), int(y)))

        # A requested letter band is intentionally plot-free, so letters
        # cannot cover a y label, heatmap row, or annotation.  Other mosaics
        # retain their legacy placed-image behavior.
        panel_name = spec.get("panel_names", {}).get(name)
        if panel_name:
            hx = int(col_left(c)) + LABEL_PAD
            hy = int(row_top(r)) + 4
            draw.text((hx, hy), chr(ord("A") + idx),
                      fill=(17, 17, 17), font=header_lab_font)
            draw.text((hx + int(HEADER_LABEL_SIZE * 0.82),
                       hy + int(HEADER_LABEL_SIZE * 0.25)),
                      panel_name, fill=(17, 17, 17), font=name_font,
                      stroke_width=3, stroke_fill=BG)
        elif label_band:
            lx = int(col_left(c)) + LABEL_PAD
            ly = int(row_top(r)) + max(2, (label_band - LABEL_SIZE) // 2)
            draw.text((lx, ly), chr(ord("A") + idx),
                      fill=(17, 17, 17), font=lab_font)
        else:
            dx, dy = spec.get("label_xy", {}).get(name, (LABEL_PAD, LABEL_PAD))
            draw.text((int(x) + dx, int(y) + dy),
                      chr(ord("A") + idx), fill=(17, 17, 17), font=lab_font)

    if spec.get("shared_legend"):
        # Center the publication-wide key beneath the complete mosaic.  Rows
        # are measured before drawing, so neither Figure 5 nor Figure 7 starts
        # from the left margin or appears attached to only one lower panel.
        items = (FIG5_SHARED_LEGEND if spec["shared_legend"] == "fig5"
                 else FIG7_SHARED_LEGEND)
        legend_font = _font(48 if spec["shared_legend"] == "fig7" else 42)
        max_row_w = TARGET_W - 2 * EDGE
        measured = []
        for label, color in items:
            box = draw.textbbox((0, 0), label, font=legend_font)
            measured.append((label, color, 56 + (box[2] - box[0]) + 34))
        rows, row, used = [], [], 0
        for item in measured:
            if row and used + item[2] > max_row_w:
                rows.append(row)
                row, used = [], 0
            row.append(item)
            used += item[2]
        if row:
            rows.append(row)
        row_step = 76
        y0 = total_h - shared_legend_h + max(20, (shared_legend_h - len(rows) * row_step) // 2)
        for ridx, row in enumerate(rows):
            row_w = sum(item[2] for item in row)
            x = (TARGET_W - row_w) / 2
            y = y0 + ridx * row_step
            for label, color, item_w in row:
                draw.line((x, y + 26, x + 42, y + 26), fill=color, width=16)
                draw.text((x + 52, y), label, fill=(25, 25, 25), font=legend_font)
                x += item_w

    out = OUT / f"{stem}.png"
    canvas.save(out, optimize=True, dpi=(400, 400))
    mb = out.stat().st_size / 1e6
    letters = ", ".join(f"{chr(ord('A')+i)}={n}" for i, (n, *_) in enumerate(spec["panels"]))
    print(f"  saved -> {out}  ({canvas.width} x {canvas.height}, {mb:.2f} MB)")
    print(f"           {letters}")


if __name__ == "__main__":
    for stem, spec in SLIDES.items():
        print(f"Building {stem} ...")
        build(stem, spec)
    print("\nDone.")
