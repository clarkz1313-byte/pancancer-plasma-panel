#!/usr/bin/env python3
"""
fig89_stacked_mosaic.py — slides 8 and 9 as SINGLE deck-ready mosaics.

Why this exists
---------------
Slides 8 and 9 were previously assembled by hand in Canva from seven
separately-rendered filmstrip PNGs (fig8a..fig8g / fig9a..fig9g). That
approach cannot produce a clean per-cancer read-down, for two reasons:

  1. Each row builder in fig89_common.py sets its OWN figure margins --
     left=0.05/right=0.997 on rows 1-3, but left=0.01..0.02 on rows 4-7.
     At ~9,000 px wide that is ~300 px of horizontal drift, so AML's column
     in row 4 does not sit above AML's column in row 1. Stacking the PNGs
     bakes that misalignment in.
  2. Every row carries its own fig.suptitle(), and four of the seven carry
     their own fig.legend(). Stacked, that is seven title bands and four
     scattered legend boxes competing with the data.

This builder instead opens ONE figure with ONE GridSpec (7 rows x 12 or 13
columns). Column boundaries are therefore defined once and every cancer's
column is pixel-aligned from the top row to the bottom by construction, not
by careful hand-nudging. Per-tile cancer titles are suppressed on rows 2-7
(only the top row names the cancers), the seven row titles collapse into one
left-margin row label each, and the four legends collapse into one.

It calls the SAME tile renderers the filmstrip builders call
(`_box_ax`, `_roc_ax`, `_cm_ax`, `_ring_gauge_ax`, `_icon_array_ax`,
`_dca_ax`, `_lr_tile_ax`, `_calib_tile_ax`) via their new `show_label`
argument -- no drawing logic is duplicated here, so a style fix in
fig89_common.py still lands on both the filmstrips and these mosaics.

Numbers are identical to the filmstrips and to
`paper/tables/external_validation_canonical_2026-08-17.csv` -- same
`load_preds()` per-participant averaging, same thresholds, same Wilson CIs.

Output: figurev5/output/fig8_slide8_confirmation_mosaic.pdf/.png
        figurev5/output/fig9_slide9_screening_mosaic.pdf/.png
Run:    python fig89_stacked_mosaic.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from publication_style import enable_text_scaling

from fig89_common import (
    CANCER_ORDER, PLOT_COLOR, MACRO_COLOR, TILE_LABEL, DPI, OUT,
    PARTA_PRED, PARTB_PRED,
    load_preds, _wilson_ci, _lr_value_ci, _dca_curve, _calib_curve,
    _box_ax, _roc_ax, _cm_ax, _cm_macro_ax, _icon_array_ax, _ring_gauge_ax,
    _dca_ax, _lr_tile_ax, _calib_tile_ax, _SP,
)

# Row heights in inches. Row 1 (box plots) needs the most vertical room --
# it is the only row showing a full distribution rather than a summary.
# Rows 4 (gauge) and 6 (LR) need a little extra because their tiles are
# aspect-locked / log-scaled respectively.
# Tuned so the finished mosaic lands near 1.6:1 -- wide enough to drop onto a
# 16:9 slide with a title strip above it, rather than the portrait shape a
# naive stack of seven full-height filmstrips produces.
ROW_H = {
    "box":   2.75,
    "roc":   1.90,
    "cm":    2.00,
    "gauge": 2.15,
    "dca":   1.90,
    "lr":    2.00,
    "calib": 1.90,
}
COL_W       = 2.40   # inches per cancer column
LEFT_PAD    = 1.28   # room for panel letters + first-tile y-axis labels
RIGHT_PAD   = 0.18
TOP_PAD     = 0.62   # keep the first-row cancer labels inside the canvas
BOTTOM_PAD  = 1.10   # single consolidated legend

WSPACE = 0.07   # identical for every row -- this is what guarantees alignment
HSPACE = 0.42

enable_text_scaling(factor=1.25, minimum=11.0, axis_label_scale=0.75)


def _row_label(fig, y_top, letter, unit=None, y_bottom=None):
    """Draw only the panel identifier; caption carries the description."""
    fig.text(0.008, y_top, letter, ha="left", va="top",
             fontsize=21, fontweight="bold", color="#222222",
             transform=fig.transFigure)
    if unit and y_bottom is not None:
        fig.text(0.026, (y_top + y_bottom) / 2.0, unit,
                 ha="center", va="center", rotation=90,
                 fontsize=14, fontweight="bold", color="#222222",
                 transform=fig.transFigure)


def build_stacked_mosaic(pred_dict, thr, part, add_macro, gauge_metric,
                         gauge_target, dca_style, lr_kind, suptitle, stem):
    """part: 'A' (confirmation, slide 8) or 'B' (screening, slide 9)."""
    n_tiles = 13 if add_macro else 12
    row_keys = ["box", "roc", "cm", "gauge", "dca", "lr", "calib"]
    heights = [ROW_H[k] for k in row_keys]
    if part == "B":
        heights[row_keys.index("gauge")] = 2.38

    fig_w = LEFT_PAD + n_tiles * COL_W + RIGHT_PAD
    grid_h = sum(heights) + HSPACE * np.mean(heights) * (len(heights) - 1)
    fig_h = TOP_PAD + grid_h + BOTTOM_PAD

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        len(row_keys), n_tiles,
        left=LEFT_PAD / fig_w, right=1 - RIGHT_PAD / fig_w,
        top=1 - TOP_PAD / fig_h, bottom=BOTTOM_PAD / fig_h,
        wspace=WSPACE, hspace=HSPACE,
        height_ratios=heights,
    )

    # ---- pre-compute everything that needs a shared scale across tiles ----
    per = {}
    for c in CANCER_ORDER:
        pdata = load_preds(pred_dict[c])
        y = pdata["true_label"].values
        yp = (pdata["prob"].values >= thr).astype(int)
        tp = int(((y == 1) & (yp == 1)).sum()); fn = int(((y == 1) & (yp == 0)).sum())
        tn = int(((y == 0) & (yp == 0)).sum()); fp = int(((y == 0) & (yp == 1)).sum())
        per[c] = dict(pdata=pdata, tp=tp, fn=fn, tn=tn, fp=fp,
                      prev=float(y.mean()))

    pt_grid = np.linspace(0.05, 0.85, 150)
    nb_all = {c: _dca_curve(per[c]["pdata"], pt_grid) for c in CANCER_ORDER}
    # Shared y-limits so the DCA row is comparable left-to-right (same rule
    # the filmstrip builder uses).
    ylim_dca = (-0.3, max(nb.max() for nb in nb_all.values()) * 1.15)

    # ================= ROW 1 — probability box plots =================
    for i, c in enumerate(CANCER_ORDER):
        ax = fig.add_subplot(gs[0, i])
        _box_ax(ax, per[c]["pdata"], c, PLOT_COLOR[c], first=(i == 0),
                show_label=True, compact=True)
    if add_macro:
        ax = fig.add_subplot(gs[0, -1])
        pooled = pd.concat([per[c]["pdata"] for c in CANCER_ORDER])
        _box_ax(ax, pooled, "Macro\nmean", MACRO_COLOR, first=False,
                show_label=True, compact=True)

    # ================= ROW 2 — ROC =================
    for i, c in enumerate(CANCER_ORDER):
        ax = fig.add_subplot(gs[1, i])
        _roc_ax(ax, per[c]["pdata"], c, PLOT_COLOR[c], first=(i == 0),
                show_label=False)
        if i > 0:
            # "1-Spec" repeated 12x is noise, and at this row height it
            # collides with the confusion row's top-mounted header below
            ax.set_xlabel(""); ax.set_xticklabels([])
    if add_macro:
        ax = fig.add_subplot(gs[1, -1])
        fpr_grid = np.linspace(0, 1, 200)
        tprs = []
        for c in CANCER_ORDER:
            fpr, tpr, _ = roc_curve(per[c]["pdata"]["true_label"],
                                    per[c]["pdata"]["prob"])
            tprs.append(np.interp(fpr_grid, fpr, tpr))
        macro_tpr = np.mean(tprs, axis=0)
        ax.fill_between(fpr_grid, macro_tpr, alpha=0.15, color=MACRO_COLOR)
        ax.plot(fpr_grid, macro_tpr, color=MACRO_COLOR, lw=3.2)
        ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1.1, ls="--")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels([])
        ax.tick_params(labelsize=14, length=3, pad=2)
        ax.set_xticklabels([])   # axis labelled once, on the first tile
        ax.text(0.97, 0.05, f"AUC: {auc(fpr_grid, macro_tpr):.2f}",
                ha="right", va="bottom", transform=ax.transAxes,
                fontsize=18, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.38", facecolor="white",
                          edgecolor="black", linewidth=1.5, alpha=0.92))
        for sp in ax.spines.values():
            sp.set_color(_SP["color"]); sp.set_linewidth(_SP["lw"])

    # ================= ROW 3 — confusion matrices =================
    for i, c in enumerate(CANCER_ORDER):
        ax = fig.add_subplot(gs[2, i])
        _cm_ax(ax, per[c]["pdata"], c, PLOT_COLOR[c], thr, first=(i == 0),
               show_label=False)
    if add_macro:
        ax = fig.add_subplot(gs[2, -1])
        _cm_macro_ax(ax,
                     sum(per[c]["tp"] for c in CANCER_ORDER),
                     sum(per[c]["fn"] for c in CANCER_ORDER),
                     sum(per[c]["tn"] for c in CANCER_ORDER),
                     sum(per[c]["fp"] for c in CANCER_ORDER))
        ax.set_title("")

    # ================= ROW 4 — gauge (spec ring / sens icon array) =========
    gvals = []
    for i, c in enumerate(CANCER_ORDER):
        ax = fig.add_subplot(gs[3, i])
        d = per[c]
        if gauge_metric == "sens":
            val, lo, hi = _wilson_ci(d["tp"], d["tp"] + d["fn"]); n = d["tp"] + d["fn"]
        else:
            val, lo, hi = _wilson_ci(d["tn"], d["tn"] + d["fp"]); n = d["tn"] + d["fp"]
        gvals.append(val)
        if part == "A":
            _ring_gauge_ax(ax, c, PLOT_COLOR[c], val, lo, hi, n, gauge_target,
                           show_label=False)
        else:
            _icon_array_ax(ax, c, PLOT_COLOR[c], val, lo, hi, n, gauge_target,
                           show_label=False)
    if add_macro:
        ax = fig.add_subplot(gs[3, -1])
        mv = float(np.mean(gvals))
        _icon_array_ax(ax, "Macro mean", MACRO_COLOR, mv, mv, mv,
                       len(CANCER_ORDER), gauge_target, show_label=False)

    # ================= ROW 5 — decision curves =================
    for i, c in enumerate(CANCER_ORDER):
        ax = fig.add_subplot(gs[4, i])
        _dca_ax(ax, c, PLOT_COLOR[c], pt_grid, nb_all[c], per[c]["prev"],
                dca_style, ylim_dca, first=(i == 0), show_label=False)
        if i > 0:
            ax.set_xlabel("")
    if add_macro:
        ax = fig.add_subplot(gs[4, -1])
        _dca_ax(ax, "Macro", MACRO_COLOR, pt_grid,
                np.mean(np.array([nb_all[c] for c in CANCER_ORDER]), axis=0),
                float(np.mean([per[c]["prev"] for c in CANCER_ORDER])),
                dca_style, ylim_dca, first=False, show_label=False)
        ax.set_xlabel("")

    # ================= ROW 6 — likelihood ratios =================
    lrvals = []
    for i, c in enumerate(CANCER_ORDER):
        ax = fig.add_subplot(gs[5, i])
        d = per[c]
        val, lo, hi = _lr_value_ci(d["tp"], d["fn"], d["tn"], d["fp"], lr_kind)
        lrvals.append(val)
        _lr_tile_ax(ax, c, PLOT_COLOR[c], val, lo, hi, lr_kind,
                    first=(i == 0), show_label=False)
    if add_macro:
        ax = fig.add_subplot(gs[5, -1])
        gm = float(np.exp(np.mean(np.log(lrvals))))
        _lr_tile_ax(ax, "Macro", MACRO_COLOR, gm, gm, gm, lr_kind,
                    first=False, show_label=False)

    # ================= ROW 7 — calibration =================
    for i, c in enumerate(CANCER_ORDER):
        ax = fig.add_subplot(gs[6, i])
        grp, ece, brier = _calib_curve(per[c]["pdata"])
        _calib_tile_ax(ax, c, PLOT_COLOR[c], grp, ece, brier,
                       first=(i == 0), show_label=False)
        if i > 0:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        else:
            # Avoid the duplicated 0.0 labels at the bottom-left corner.
            ax.set_xticks([0.5, 1.0])
            ax.set_xticklabels(["0.5", "1.0"])
    if add_macro:
        ax = fig.add_subplot(gs[6, -1])
        pooled = pd.concat([per[c]["pdata"] for c in CANCER_ORDER])
        grp, ece, brier = _calib_curve(pooled, n_bins=8)
        _calib_tile_ax(ax, "Macro", MACRO_COLOR, grp, ece, brier,
                       first=False, show_label=False)
        ax.set_xlabel("")
        ax.set_xticklabels([])

    # ---------------- row labels in the left margin ----------------
    # Convert each row's GridSpec extent into figure coords for the letter.
    for r, letter in enumerate("ABCDEFG"):
        pos = gs[r, 0].get_position(fig)
        unit = None
        if r == 3:
            unit = "specificity" if part == "A" else "sensitivity"
        _row_label(fig, pos.y1, letter, unit=unit, y_bottom=pos.y0)

    # ---------------- one consolidated legend ----------------
    gauge_correct = ("correctly caught" if gauge_metric == "sens"
                     else "correctly cleared")
    gauge_miss = "missed case" if gauge_metric == "sens" else "false alarm"
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#888888", ec="white",
                      label=f"gauge: {gauge_correct}"),
        plt.Rectangle((0, 0), 1, 1, fc="#c9c9c9", ec="white",
                      label=f"gauge: {gauge_miss}"),
        plt.Line2D([0], [0], color="black", lw=2.6, label="95% CI"),
        plt.Line2D([0], [0], color="#ff006e", lw=2.6, ls="--",
                   label=f"{gauge_target:.2f} target"),
        plt.Line2D([0], [0], color="#999999", lw=1.4, ls=":", label="DCA: treat all"),
        plt.Line2D([0], [0], color="#555555", lw=1.4, ls="--", label="DCA: treat none"),
        plt.Rectangle((0, 0), 1, 1, fc="#2a9d5c", alpha=0.25, label=f"{lr_kind}: strong"),
        plt.Rectangle((0, 0), 1, 1, fc="#e9c46a", alpha=0.30, label=f"{lr_kind}: moderate"),
        plt.Rectangle((0, 0), 1, 1, fc="#e76f51", alpha=0.25, label=f"{lr_kind}: weak/none"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.004), fontsize=12, framealpha=0.94,
               columnspacing=1.6, handletextpad=0.6)

    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI, facecolor="white")
        print(f"  saved -> {p}  ({fig_w:.1f} x {fig_h:.1f} in)")
    plt.close(fig)


if __name__ == "__main__":
    print("Building slide 8 — Part A confirmation mosaic ...")
    build_stacked_mosaic(
        PARTA_PRED, thr=0.80, part="A", add_macro=False,
        gauge_metric="spec", gauge_target=0.90,
        dca_style="area", lr_kind="LR+",
        suptitle="External marker-set evaluation: cancer-specific panels | "
                 "12 cancer endpoints in external datasets | fixed threshold 0.80",
        stem="fig8_slide8_confirmation_mosaic")

    print("\nBuilding slide 9 — Part B screening mosaic ...")
    build_stacked_mosaic(
        PARTB_PRED, thr=0.50, part="B", add_macro=True,
        gauge_metric="sens", gauge_target=0.90,
        dca_style="area", lr_kind="LR-",
        suptitle="External marker-set evaluation: 25-protein set | "
                 "12 cancer endpoints in external datasets | fixed threshold 0.50",
        stem="fig9_slide9_screening_mosaic")

    print("\nDone.")
