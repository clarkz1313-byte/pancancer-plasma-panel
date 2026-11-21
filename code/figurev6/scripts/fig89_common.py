#!/usr/bin/env python3
"""
fig89_common.py — shared machinery for the external-validation mosaics.

Slide 8 (fig8_panels.py) and slide 9 (fig9_panels.py) are the same seven
row types applied to two different models, so every tile renderer and row
builder lives here and is imported by both. Do NOT duplicate a builder into
either script -- a style fix applied to one row must land on both slides.

  Part A = single-class confirmation panels -> fig8_panels.py, slide 8.
           12 INDEPENDENT binary panels, so NO macro/aggregate cells
           anywhere (there is no single model to average).
  Part B = multiclass locked 25-protein panel -> fig9_panels.py, slide 9.
           ONE model evaluated on 12 cohorts, so a 13th purple macro tile
           IS statistically defensible on most rows (see MACRO_COLOR).

Row builders (each renders one full-width 12- or 13-tile filmstrip):
  build_boxplots            predicted-probability box plots
  build_roc                 ROC curves (macro ROC interpolated on a shared
                            FPR grid when add_macro=True)
  build_cm                  confusion matrices (macro tile = pooled raw
                            counts, labelled "Pooled total", not a rate)
  build_metric_mosaic       sensitivity as a 100-square ICON ARRAY
  build_ring_mosaic         sensitivity or specificity as a RING/DONUT gauge
  build_dca_mosaic          decision curve analysis (style="area" or "line")
  build_lr_mosaic           likelihood ratios, log scale (LR+ bar / LR- dot)
  build_calibration_mosaic  reliability diagram w/ per-tile ECE + Brier

Icon array vs. ring gauge is a deliberate geometry split: the screening
(sensitivity) and confirmation (specificity) rows must not be mistakable
for each other when the two slides sit side by side.

`panel_auc_slope()` at the bottom is DEPRECATED (kept for provenance only,
not called by either script).
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Wedge
import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix

np.random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
EV   = ROOT / "revise_plan" / "external_validation_roadmap"
OUT  = ROOT / "figurev6" / "output"
OUT.mkdir(parents=True, exist_ok=True)

CANCER_ORDER = ["AML","BRC","CLL","CRC","CVX","ENDC","GLIOM","LUNGC","LYMPH","MYEL","OVC","PRC"]
CANCER_COLOR = {
    "AML":"#b22222","BRC":"#c97b63","CLL":"#7a3e9d",
    "CRC":"#d68600","CVX":"#c13d86","ENDC":"#8e5d2c",
    "GLIOM":"#7ec8e3","LUNGC":"#2e8b57","LYMPH":"#3856a6",
    "MYEL":"#8c564b","OVC":"#d1495b","PRC":"#008b8b",
}
PLOT_COLOR = {**CANCER_COLOR, "GLIOM": "#1a8db8"}

CANCER_FULL = {
    "AML":   "AML",
    "BRC":   "Breast Cancer",
    "CLL":   "CLL",
    "CRC":   "Colorectal Cancer",
    "CVX":   "Cervical Cancer",
    "ENDC":  "Endometrial Cancer",
    "GLIOM": "Glioma",
    "LUNGC": "Lung Cancer",
    "LYMPH": "DLBCL",
    "MYEL":  "Myeloma",
    "OVC":   "Ovarian Cancer",
    "PRC":   "Prostate Cancer",
}

# Short code shown on each tile. Defaults to the internal class code, which is
# already a real disease abbreviation for every class except LYMPH -- that one
# is DLBCL specifically (Alves supplementary sheet 1a: class "DLBCL", n=55),
# and the external cohort is now the DLBCL-only subset, so rendering "LYMPH"
# would assert a broader disease than either the model or the data supports.
TILE_LABEL = {c: c for c in CANCER_ORDER}
TILE_LABEL["LYMPH"] = "DLBCL"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "font.weight": "bold",
    "axes.titlesize": 14, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
})
DPI = 400


def save(fig, stem):
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


def load_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    prob  = df.groupby("sample_id")["predicted_probability_tumor"].mean()
    label = df.groupby("sample_id")["true_label"].first()
    return pd.DataFrame({"prob": prob, "true_label": label})


def youden_thr(fpr, tpr, thrs):
    return float(thrs[np.argmax(tpr - fpr)])


def _blend(col: str, frac: float) -> tuple:
    rgb = np.array(mcolors.to_rgba(col))[:3]
    return tuple((rgb * frac + (1 - frac)).clip(0, 1))


def _cm_palette(col: str):
    return (
        _blend(col, 1.00),   # TP  full color
        _blend(col, 0.62),   # TN  medium shade
        _blend(col, 0.15),   # FN  very light
        _blend(col, 0.09),   # FP  lightest
    )


def _wilson_ci(x, n, z=1.959964):
    """95% Wilson score interval for a binomial proportion x/n."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    phat = x / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


# ─── cohort maps ─────────────────────────────────────────────────────────────
# BRC: GSE42568 for both Part B and Part A (higher AUC, full 13-protein coverage)
PARTB_PRED = {
    "AML":   EV/"gse13159_aml/results/aml_vs_control/locked25_grouped_predictions.csv",
    "BRC":   EV/"gse42568_brc_gene/results/brc_vs_control/locked25_grouped_predictions.csv",
    "CLL":   EV/"gse13159_aml/results/cll_vs_control/locked25_grouped_predictions.csv",
    "CRC":   EV/"geo_crc_GSE41258/results/crc_vs_control/locked25_grouped_predictions.csv",
    "CVX":   EV/"candidate_cvx_gse9750/results/cvx_vs_control/locked25_grouped_predictions.csv",  # promoted 2026-08-17, was geo_cvx_GSE63514 (now supplementary only)
    "ENDC":  EV/"geo_endc_GSE17025/results/endc_vs_control/locked25_grouped_predictions.csv",
    "GLIOM": EV/"gse4290/results/glioma_vs_nontumor/locked25_grouped_predictions.csv",
    "LUNGC": EV/"cptac_icpc_luad_protein/results/lungc_vs_control_patient_grouped_all_present/locked25_grouped_predictions.csv",
    "LYMPH": EV/"gse32018_lymph_gene/results/dlbcl_vs_reactive_control_production/locked25_grouped_predictions.csv",
    "MYEL":  EV/"gse6477_myel_gene/results/myel_vs_control/locked25_grouped_predictions.csv",
    "OVC":   EV/"geo_ovc_GSE18520/results/ovc_vs_control/locked25_grouped_predictions.csv",
    "PRC":   EV/"geo_prc_GSE17951/results/prc_vs_control/locked25_grouped_predictions.csv",
}
PARTA_PRED = {
    "AML":   EV/"gse13159_aml/results_single_panel/aml_vs_control/locked25_grouped_predictions.csv",
    "BRC":   EV/"gse42568_brc_gene/results_single_panel/brc_vs_control/locked25_grouped_predictions.csv",
    "CLL":   EV/"gse13159_aml/results_single_panel/cll_vs_control/locked25_grouped_predictions.csv",
    "CRC":   EV/"geo_crc_GSE41258/results_single_panel/crc_vs_control/locked25_grouped_predictions.csv",
    "CVX":   EV/"candidate_cvx_gse9750/results_single_panel/cvx_vs_control/locked25_grouped_predictions.csv",  # promoted 2026-08-17 alongside Part B, so CVX uses ONE cohort across both stages (was geo_cvx_GSE63514)
    "ENDC":  EV/"geo_endc_GSE17025/results_single_panel/endc_vs_control/locked25_grouped_predictions.csv",
    "GLIOM": EV/"gse4290/results_single_panel/glioma_vs_nontumor/locked25_grouped_predictions.csv",
    "LUNGC": EV/"cptac_icpc_luad_protein/results_single_panel/lungc_vs_control_patient_grouped/locked25_grouped_predictions.csv",
    "LYMPH": EV/"gse32018_lymph_gene/results_single_panel/dlbcl_vs_reactive_control_production/locked25_grouped_predictions.csv",
    "MYEL":  EV/"gse6477_myel_gene/results_single_panel/myel_vs_control/locked25_grouped_predictions.csv",
    "OVC":   EV/"geo_ovc_GSE18520/results_single_panel/ovc_vs_control/locked25_grouped_predictions.csv",
    "PRC":   EV/"geo_prc_GSE17951/results_single_panel/prc_vs_control/locked25_grouped_predictions.csv",
}

_SP = dict(color="#666666", lw=0.9)   # tile spine style


# ─── box plot ────────────────────────────────────────────────────────────────
def _box_ax(ax, pdata, cancer, col, first, show_label=True, compact=False):
    """show_label=False suppresses the per-tile cancer title (used by the
    stacked mosaic builder, where only the top row carries cancer names).
    compact=True swaps the rotated full-disease x-tick labels for short
    horizontal 'Case'/'Ctrl', which is what fits when this row sits inside a
    7-row stack rather than alone."""
    c_probs = pdata[pdata["true_label"] == 1]["prob"].values
    n_probs = pdata[pdata["true_label"] == 0]["prob"].values
    ctrl_col = _blend(col, 0.10)   # very faint control

    bp = ax.boxplot(
        [c_probs, n_probs], positions=[0, 1], widths=0.56,
        patch_artist=True, showfliers=False,
        whiskerprops=dict(color="#222222", lw=1.8),
        capprops=dict(color="#222222", lw=1.8),
        medianprops=dict(color="black", lw=3.0, solid_capstyle="round"),
        boxprops=dict(linewidth=1.4),
    )
    bp["boxes"][0].set_facecolor(col)
    bp["boxes"][1].set_facecolor(ctrl_col)

    jw = 0.13
    for xi, (vals, fc, al) in enumerate([
            (c_probs, col, 0.45),
            (n_probs, _blend(col, 0.55), 0.55)]):
        jx = np.random.uniform(-jw, jw, len(vals))
        ax.scatter(xi + jx, vals, s=8, color=fc, alpha=al,
                   linewidths=0, edgecolors="none", zorder=4)

    ax.axhline(0.5, color="#555555", lw=2.2, ls="--", zorder=1)

    ax.set_xlim(-0.68, 1.68); ax.set_ylim(-0.04, 1.14)
    ax.set_xticks([0, 1])
    if compact:
        ax.set_xticklabels(["Case", "Ctrl"], fontsize=12, fontweight="bold")
    else:
        ax.set_xticklabels([CANCER_FULL.get(cancer, cancer), "Control"],
                           fontsize=11, fontweight="bold",
                           rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"], fontsize=13, fontweight="bold")
    ax.tick_params(length=3, pad=2)
    if first:
        ax.set_ylabel("P(cancer)", fontsize=13, fontweight="bold")

    for sp in ax.spines.values():
        sp.set_color(_SP["color"]); sp.set_linewidth(_SP["lw"])
    if show_label:
        ax.set_title(TILE_LABEL.get(cancer, cancer), color=col, fontsize=18, fontweight="bold", pad=4)


# ─── ROC curve ───────────────────────────────────────────────────────────────
def _roc_ax(ax, pdata, cancer, col, first, show_label=True):
    fpr, tpr, thrs = roc_curve(pdata["true_label"], pdata["prob"])
    roc_auc = auc(fpr, tpr)

    ax.fill_between(fpr, tpr, alpha=0.15, color=col)
    ax.plot(fpr, tpr, color=col, lw=3.2)
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1.1, ls="--")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=14, length=3, pad=2)
    ax.set_xlabel("1−Spec", fontsize=13, fontweight="bold")
    if first:
        ax.set_ylabel("Sensitivity", fontsize=14, fontweight="bold")
        ax.set_yticklabels(["0", "0.5", "1"], fontsize=14, fontweight="bold")
    else:
        ax.set_yticklabels([])

    if show_label:
        ax.set_title(TILE_LABEL.get(cancer, cancer), color=col, fontsize=18, fontweight="bold", pad=4)
    ax.text(0.97, 0.05, f"AUC: {roc_auc:.2f}", ha="right", va="bottom",
            transform=ax.transAxes, fontsize=18, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.38", facecolor="white",
                      edgecolor="black", linewidth=1.5, alpha=0.92))
    for sp in ax.spines.values():
        sp.set_color(_SP["color"]); sp.set_linewidth(_SP["lw"])


# ─── confusion matrix ─────────────────────────────────────────────────────────
def _cm_ax(ax, pdata, cancer, col, thr, first, show_label=True):
    y_true = pdata["true_label"].values
    y_pred = (pdata["prob"].values >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    tp_c, tn_c, fn_c, fp_c = _cm_palette(col)
    fills = [[tp_c, fn_c], [fp_c, tn_c]]
    # TP (full color, dark) → white text; all others → black
    txt_cols = [["white", "black"], ["black", "black"]]

    for r in range(2):
        for c in range(2):
            ax.add_patch(plt.Rectangle(
                (c, 1 - r), 1, 1,
                facecolor=fills[r][c],
                edgecolor="white", linewidth=2.5, zorder=2))
            ax.text(c + 0.5, 1.5 - r, str(cm[r, c]),
                    ha="center", va="center",
                    fontsize=28, fontweight="bold",
                    color=txt_cols[r][c], zorder=3)

    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Pred +", "Pred −"], fontsize=14, fontweight="bold")
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_yticks([0.5, 1.5])
    if first:
        ax.set_yticklabels(["Control", "Cancer"], fontsize=14, fontweight="bold")
    else:
        ax.set_yticklabels(["", ""])
    ax.tick_params(length=0, pad=4)

    if show_label:
        ax.set_title(TILE_LABEL.get(cancer, cancer), color=col, fontsize=18, fontweight="bold", pad=22)
    for sp in ax.spines.values():
        sp.set_color(_SP["color"]); sp.set_linewidth(_SP["lw"])


# ─── sensitivity / specificity icon-array tile ───────────────────────────────
# Screening (Part B) is judged on sensitivity, confirmation (Part A) on
# specificity -- everything else in the mosaic (boxplots, ROC, CM) is shared
# machinery that looks near-identical between the two rows. This panel is
# deliberately a DIFFERENT chart family, not just a different metric in the
# same bar/box shape: a 100-square icon array ("X out of 100 patients"),
# which also reads as the more clinically intuitive framing for a
# screening/confirmation story than a bare proportion. Threshold matches
# fig8c/8f exactly (fixed 0.50 for Part B, fixed 0.80 for Part A).
def _icon_array_ax(ax, cancer, col, val, lo, hi, n, target, show_label=True):
    k, lo_k, hi_k = (int(round(x * 100)) for x in (val, lo, hi))
    target_k = int(round(target * 100))

    size, gap = 1.0, 0.18
    step = size + gap
    # three-tier spectrum needs to read at a glance: solid = full color,
    # band = clearly-tinted mid shade (bumped 0.55->0.70 -- still too close
    # to the grey tier to distinguish at a glance for lighter base hues),
    # grey = neutral, darkened slightly (#ececec->#d9d9d9) for more
    # separation from both the band tier and the white tile background.
    solid, band, grey = col, _blend(col, 0.70), "#d9d9d9"

    idx = 0
    for r in range(10):
        for c in range(10):
            fc = solid if idx < lo_k else band if idx < hi_k else grey
            ax.add_patch(plt.Rectangle((c * step, r * step), size, size,
                                       facecolor=fc, edgecolor="white",
                                       linewidth=0.9, zorder=2))
            idx += 1

    # 0.90 target lands exactly on a row boundary (9 full rows = 90 squares)
    # -- thicker/darker per feedback ("dash cut off line must be thicker, bolder")
    y_line = (target_k // 10) * step - gap / 2
    ax.axhline(y_line, color="#111111", lw=3.4, ls="--", dashes=(5, 2.5), zorder=3)

    grid_w = grid_h = 10 * step - gap

    # cancer name + value are drawn as data-coordinate text (not ax.set_title,
    # which anchors to the axes bounding box -- with set_aspect("equal") on a
    # very wide/short subplot that box is much taller than the square grid
    # itself, so a title-pad offset and a data-coordinate value text ended up
    # landing on top of each other). Stacking both in data coordinates with a
    # fixed vertical gap keeps them apart regardless of the box's aspect padding.
    below_target = val < target
    if show_label:
        ax.text(grid_w / 2, grid_h + 1.55, TILE_LABEL.get(cancer, cancer), ha="center", va="bottom",
                fontsize=17, fontweight="bold", color=col)
    # Keep the value in a true header band above the icon grid. At mosaic
    # scale the previous +0.35 offset touched the top row of squares.
    ax.text(grid_w / 2, grid_h + 0.78, f"{val:.2f}", ha="center", va="bottom",
            fontsize=21, fontweight="bold",
            color="#b22222" if below_target else "black")
    ax.text(grid_w / 2, -0.76, f"n={n}", ha="center", va="top",
            fontsize=18, fontweight="bold", color="#333333")

    ax.set_xlim(-gap, grid_w + gap)
    ax.set_ylim(-1.34, grid_h + (1.95 if show_label else 1.82))
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    for sp in ax.spines.values():
        sp.set_color(_SP["color"]); sp.set_linewidth(_SP["lw"])


def build_metric_mosaic(pred_dict, fixed_thr, metric, target, suptitle, stem, add_macro=False):
    """metric: 'sens' (Part B screening row) or 'spec' (Part A confirmation row)."""
    n_tiles = 13 if add_macro else 12
    fig, axes = plt.subplots(1, n_tiles, figsize=(30 * n_tiles / 12, 4.0), squeeze=True)
    fig.subplots_adjust(wspace=0.04, left=0.01, right=0.999,
                        top=0.80, bottom=0.15)

    per_cancer_vals = []
    for i, cancer in enumerate(CANCER_ORDER):
        pdata = load_preds(pred_dict[cancer])
        y_true = pdata["true_label"].values
        if fixed_thr is not None:
            thr = fixed_thr
        else:
            fpr, tpr, thrs = roc_curve(y_true, pdata["prob"].values)
            thr = youden_thr(fpr, tpr, thrs)
        y_pred = (pdata["prob"].values >= thr).astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())

        if metric == "sens":
            val, lo, hi = _wilson_ci(tp, tp + fn); n = tp + fn
        else:
            val, lo, hi = _wilson_ci(tn, tn + fp); n = tn + fp
        per_cancer_vals.append(val)

        _icon_array_ax(axes[i], cancer, PLOT_COLOR[cancer], val, lo, hi, n, target)

    if add_macro:
        # macro mean of the 12 point estimates -- no CI banding here (unlike
        # the individual tiles): aggregating uncertainty across 12
        # heterogeneous external cohorts isn't statistically well-defined
        # the same way a single cohort's Wilson CI is, so the macro tile is
        # drawn as a plain solid fill to avoid implying a rigor it doesn't have
        macro_val = float(np.mean(per_cancer_vals))
        _icon_array_ax(axes[-1], "Macro mean", MACRO_COLOR, macro_val,
                       macro_val, macro_val, len(CANCER_ORDER), target)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#444444", ec="white",
                      label="below 95% CI lower bound"),
        plt.Rectangle((0, 0), 1, 1, fc="#aaaaaa", ec="white",
                      label="within 95% CI"),
        plt.Rectangle((0, 0), 1, 1, fc="#e3e3e3", ec="white",
                      label="above estimate"),
        plt.Line2D([0], [0], color="#333333", lw=2.0, ls="--",
                   label=f"{target:.2f} target"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
              bbox_to_anchor=(0.5, 0.005), fontsize=10, framealpha=0.92)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    save(fig, stem)


# ─── shared macro-cell helper ────────────────────────────────────────────────
# Row 1/2/4/5/6 on the multiclass (Part B) side get a 13th "macro mean" tile
# in purple, matching the macro-aggregate color used elsewhere in this deck
# (fig5c radar, fig5e/7i decision curves). Part B is genuinely ONE model
# evaluated on 12 cohorts, so a macro summary is defensible; Part A is 12
# INDEPENDENT binary panels, so it never gets one (adding it there would
# imply a pooled analysis that was never actually done). Row 3 (confusion
# matrix) also never gets one -- raw counts summed across cohorts ranging
# ~52 to ~616 samples would just be the biggest cohort wearing a "macro"
# label, not a real aggregate.
MACRO_COLOR = "#A06CD5"


# ─── figure builders ─────────────────────────────────────────────────────────
def build_boxplots(pred_dict, suptitle, stem, add_macro=False):
    n_tiles = 13 if add_macro else 12
    fig, axes = plt.subplots(1, n_tiles, figsize=(30 * n_tiles / 12, 6.2), squeeze=True)
    fig.subplots_adjust(wspace=0.07, left=0.05, right=0.997,
                        top=0.90, bottom=0.22)
    for i, cancer in enumerate(CANCER_ORDER):
        _box_ax(axes[i], load_preds(pred_dict[cancer]),
                cancer, PLOT_COLOR[cancer], first=(i == 0))
    if add_macro:
        pooled = pd.concat([load_preds(pred_dict[c]) for c in CANCER_ORDER])
        macro_data = pd.DataFrame({
            "prob": pooled["prob"].values, "true_label": pooled["true_label"].values,
        })
        _box_ax(axes[-1], macro_data, "Macro\nmean", MACRO_COLOR, first=False)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    save(fig, stem)


def build_roc(pred_dict, suptitle, stem, add_macro=False):
    n_tiles = 13 if add_macro else 12
    fig, axes = plt.subplots(1, n_tiles, figsize=(30 * n_tiles / 12, 3.8), squeeze=True)
    fig.subplots_adjust(wspace=0.05, left=0.05, right=0.997,
                        top=0.88, bottom=0.14)
    for i, cancer in enumerate(CANCER_ORDER):
        _roc_ax(axes[i], load_preds(pred_dict[cancer]),
                cancer, PLOT_COLOR[cancer], first=(i == 0))
    if add_macro:
        # macro-averaged ROC: interpolate each cancer's TPR onto a shared
        # FPR grid, then average pointwise -- same technique already used
        # for the macro-AUPRC curve in fig7e
        fpr_grid = np.linspace(0, 1, 200)
        tprs = []
        for c in CANCER_ORDER:
            pdata = load_preds(pred_dict[c])
            fpr, tpr, _ = roc_curve(pdata["true_label"], pdata["prob"])
            tprs.append(np.interp(fpr_grid, fpr, tpr))
        macro_tpr = np.mean(tprs, axis=0)
        macro_auc = auc(fpr_grid, macro_tpr)
        ax = axes[-1]
        ax.fill_between(fpr_grid, macro_tpr, alpha=0.15, color=MACRO_COLOR)
        ax.plot(fpr_grid, macro_tpr, color=MACRO_COLOR, lw=3.2)
        ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1.1, ls="--")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=14, length=3, pad=2)
        ax.set_xlabel("1−Spec", fontsize=13, fontweight="bold")
        ax.set_yticklabels([])
        ax.set_title("Macro\nmean", color=MACRO_COLOR, fontsize=18, fontweight="bold", pad=4)
        ax.text(0.97, 0.05, f"AUC: {macro_auc:.2f}", ha="right", va="bottom",
                transform=ax.transAxes, fontsize=18, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.38", facecolor="white",
                          edgecolor="black", linewidth=1.5, alpha=0.92))
        for sp in ax.spines.values():
            sp.set_color(_SP["color"]); sp.set_linewidth(_SP["lw"])
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    save(fig, stem)


def _cm_macro_ax(ax, tp, fn, tn, fp):
    """Macro tile as a literal POOLED TOTAL (sum of raw counts across all 12
    cohorts) -- same integer-count visual style as the other 12 tiles for
    consistency, labeled "Pooled total" (not "Macro mean") to be honest that
    this is a sum, not a fair per-cohort-equal-weighted average: it's
    dominated by whichever cohorts have the most samples (LUNGC n=794 vs
    CVX n=52), same as any pooled/combined external validation total."""
    col = MACRO_COLOR
    tp_c, tn_c, fn_c, fp_c = _cm_palette(col)
    fills = [[tp_c, fn_c], [fp_c, tn_c]]
    counts = [[tp, fn], [fp, tn]]
    txt_cols = [["white", "black"], ["black", "black"]]

    for r in range(2):
        for c in range(2):
            ax.add_patch(plt.Rectangle(
                (c, 1 - r), 1, 1,
                facecolor=fills[r][c],
                edgecolor="white", linewidth=2.5, zorder=2))
            ax.text(c + 0.5, 1.5 - r, str(counts[r][c]),
                    ha="center", va="center",
                    fontsize=28, fontweight="bold",
                    color=txt_cols[r][c], zorder=3)

    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Pred +", "Pred -"], fontsize=14, fontweight="bold")
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["", ""])
    ax.tick_params(length=0, pad=4)
    ax.set_title("Pooled\ntotal", color=col, fontsize=18, fontweight="bold", pad=22)
    for sp in ax.spines.values():
        sp.set_color(_SP["color"]); sp.set_linewidth(_SP["lw"])


def build_cm(pred_dict, fixed_thr, suptitle, stem, add_macro=False):
    thresholds = {}
    for cancer in CANCER_ORDER:
        if fixed_thr is not None:
            thresholds[cancer] = fixed_thr
        else:
            pdata = load_preds(pred_dict[cancer])
            fpr, tpr, thrs = roc_curve(pdata["true_label"], pdata["prob"])
            thresholds[cancer] = youden_thr(fpr, tpr, thrs)

    n_tiles = 13 if add_macro else 12
    fig, axes = plt.subplots(1, n_tiles, figsize=(30 * n_tiles / 12, 4.0), squeeze=True)
    fig.subplots_adjust(wspace=0.04, left=0.05, right=0.997,
                        top=0.78, bottom=0.04)

    tp_tot = fn_tot = tn_tot = fp_tot = 0
    for i, cancer in enumerate(CANCER_ORDER):
        pdata = load_preds(pred_dict[cancer])
        _cm_ax(axes[i], pdata, cancer, PLOT_COLOR[cancer], thresholds[cancer], first=(i == 0))
        y_true = pdata["true_label"].values
        y_pred = (pdata["prob"].values >= thresholds[cancer]).astype(int)
        tp_tot += int(((y_true == 1) & (y_pred == 1)).sum())
        fn_tot += int(((y_true == 1) & (y_pred == 0)).sum())
        tn_tot += int(((y_true == 0) & (y_pred == 0)).sum())
        fp_tot += int(((y_true == 0) & (y_pred == 1)).sum())

    if add_macro:
        _cm_macro_ax(axes[-1], tp_tot, fn_tot, tn_tot, fp_tot)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    save(fig, stem)


# ─── specificity ring/donut gauge (Part A row 4, replaces icon array) ────────
# Row 4 originally used the SAME icon-array geometry for both sensitivity
# (Part B) and specificity (Part A) -- same shape, same color scheme, which
# reads as "the same measurement" even though they're opposite halves of the
# screening/confirmation story. The donut is deliberately a different chart
# family (circular, not a grid) so the two rows can never be confused for
# each other, even skimmed at mosaic scale. Correctly-cleared arc = a fading
# spectrum of the cancer's own color (light near the inner edge -> full
# saturation at the rim, strict per-cancer identity, no shared foreign hue);
# false-alarm arc = neutral grey, not red -- this row isn't meant to visually
# alarm the reader about weak cancers (that's what row 6/7 already do
# analytically), so the below-target value text also no longer switches to
# red for a low result.
FALSE_ALARM_COLOR = "#c9c9c9"


def _ring_gauge_ax(ax, cancer, col, val, lo, hi, n, target, show_label=True):
    # ring enlarged again (r_out 1.15->1.28) now that the tile's own axis
    # frame is gone -- the freed space goes to the ring itself and tighter
    # inter-tile spacing (see wspace in build_ring_mosaic)
    r_out, r_in = 1.21, 0.76

    theta_correct = 360 * val
    # "fading spectrum" of the cancer color -- several thin concentric bands
    # interpolating from a light tint near the inner edge to full saturation
    # at the outer rim, instead of one flat fill
    N_BANDS = 6
    for i in range(N_BANDS):
        r0 = r_in + (r_out - r_in) * i / N_BANDS
        r1 = r_in + (r_out - r_in) * (i + 1) / N_BANDS
        shade_frac = 0.45 + 0.55 * (i / (N_BANDS - 1))
        ax.add_patch(Wedge((0, 0), r1, 90 - theta_correct, 90,
                           width=r1 - r0, facecolor=_blend(col, shade_frac),
                           edgecolor="none", zorder=2))
    w_false = Wedge((0, 0), r_out, 90, 450 - theta_correct,
                    width=r_out - r_in, facecolor=FALSE_ALARM_COLOR,
                    edgecolor="none", zorder=2)
    ax.add_patch(w_false)
    # single outline over both wedges for a clean edge (the per-band fades
    # have no edge of their own, so without this the correct-side looked
    # seamed between bands)
    ax.add_patch(Wedge((0, 0), r_out, 0, 360, width=r_out - r_in,
                       facecolor="none", edgecolor="white", linewidth=1.6, zorder=3))

    # 95% CI: TWO tick marks bounding the interval (lower bound AND upper
    # bound -- that's what a CI is, not a display bug), confined strictly
    # INSIDE the ring band [r_in, r_out], never past the outer rim or in
    # toward the center
    for frac in (lo, hi):
        theta = math.radians(90 - 360 * frac)
        x1, y1 = (r_in + 0.03) * math.cos(theta), (r_in + 0.03) * math.sin(theta)
        x2, y2 = (r_out - 0.03) * math.cos(theta), (r_out - 0.03) * math.sin(theta)
        ax.plot([x1, x2], [y1, y2], color="black", lw=2.6, zorder=4)

    # target: a radial spoke that actually crosses the ring band (both its
    # inner AND outer edge), not a tick sitting only outside it -- a mark
    # that stops at the rim reads as decoration, not as "here is the bar the
    # ring has to clear." Runs from just inside r_in out to just past r_out,
    # stopping well short of the center core (where the value number sits)
    # so it never crosses into the text. Distinct magenta so it can't be
    # mistaken for the black CI ticks or any ring's own fill color.
    theta_t = math.radians(90 - 360 * target)
    r_t_in, r_t_out = r_in - 0.20, r_out + 0.16
    ax.plot([r_t_in * math.cos(theta_t), r_t_out * math.cos(theta_t)],
            [r_t_in * math.sin(theta_t), r_t_out * math.sin(theta_t)],
            color="#ff006e", lw=2.6, ls="--", dashes=(4.5, 2.2), zorder=6,
            solid_capstyle="butt")

    ax.text(0, 0.12, f"{val:.2f}", ha="center", va="center", fontsize=22,
            fontweight="bold", color="black", zorder=5)
    ax.text(0, -0.22, f"n={n}", ha="center", va="center", fontsize=15,
            fontweight="bold", color="#555555", zorder=5)
    if show_label:
        ax.text(0, 1.50, TILE_LABEL.get(cancer, cancer), ha="center", va="bottom",
                fontsize=19, fontweight="bold", color=col)

    ax.set_xlim(-1.42, 1.42)
    # headroom above the ring exists only to seat the cancer label; with the
    # label suppressed (stacked mosaic) reclaim it so the ring fills its cell
    ax.set_ylim(-1.20, 1.70 if show_label else 1.34)
    ax.set_aspect("equal")
    ax.axis("off")


def build_ring_mosaic(pred_dict, fixed_thr, metric, target, suptitle, stem):
    """metric: 'spec' (Part A confirmation, correctly-cleared framing) or
    'sens' (Part B screening, correctly-caught framing) -- 'multipanel'
    version, reusing the same ring geometry for the sensitivity story."""
    fig, axes = plt.subplots(1, 12, figsize=(30, 3.9), squeeze=True)
    fig.subplots_adjust(wspace=0.006, left=0.004, right=0.999,
                        top=0.80, bottom=0.15)

    for i, cancer in enumerate(CANCER_ORDER):
        pdata = load_preds(pred_dict[cancer])
        y_true = pdata["true_label"].values
        if fixed_thr is not None:
            thr = fixed_thr
        else:
            fpr, tpr, thrs = roc_curve(y_true, pdata["prob"].values)
            thr = youden_thr(fpr, tpr, thrs)
        y_pred = (pdata["prob"].values >= thr).astype(int)

        if metric == "sens":
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            val, lo, hi = _wilson_ci(tp, tp + fn); nn = tp + fn
        else:
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            val, lo, hi = _wilson_ci(tn, tn + fp); nn = tn + fp

        _ring_gauge_ax(axes[i], cancer, PLOT_COLOR[cancer], val, lo, hi, nn, target)

    # "correct" swatch is deliberately neutral grey, not any one cancer's
    # color -- the ring itself uses each cancer's OWN color (that was the
    # whole point), so a legend swatch borrowed from one specific cancer
    # (e.g. AML's red) would misleadingly suggest red = correct, when red
    # never actually appears as "correct" in most of the other rings
    correct_label = "correctly caught (each ring's own color)" if metric == "sens" \
        else "correctly cleared (each ring's own color)"
    miss_label = "missed case" if metric == "sens" else "false alarm"
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#888888", ec="white", label=correct_label),
        plt.Rectangle((0, 0), 1, 1, fc=FALSE_ALARM_COLOR, ec="white", label=miss_label),
        plt.Line2D([0], [0], color="black", lw=2.6, label="95% CI bounds (2 ticks, inside rim)"),
        plt.Line2D([0], [0], color="#ff006e", lw=2.6, ls="--", label=f"{target:.2f} target (crosses the ring)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
              bbox_to_anchor=(0.5, 0.005), fontsize=10, framealpha=0.92)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    save(fig, stem)


# ─── decision curve analysis row ──────────────────────────────────────────────
# Part B tile style = shaded area under the curve (screening "yield");
# Both external-validation mosaics use the same shaded net-benefit convention
# so readers can compare the decision-curve row directly.  The reference is
# the better of treat-all and treat-none at each threshold.
def _dca_curve(pdata, pt_grid):
    y_true = pdata["true_label"].values
    y_score = pdata["prob"].values
    n = len(y_true)
    nb = np.empty_like(pt_grid)
    for i, pt in enumerate(pt_grid):
        pred_pos = y_score >= pt
        tp = np.sum(pred_pos & (y_true == 1))
        fp = np.sum(pred_pos & (y_true == 0))
        nb[i] = tp / n - fp / n * (pt / (1 - pt))
    return nb


def _dca_ax(ax, cancer, col, pt_grid, nb, prev, style, ylim, first, show_label=True):
    treat_all = prev - (1 - prev) * (pt_grid / (1 - pt_grid))

    if style == "area":
        # fill ONLY the model's advantage over the BETTER of the two naive
        # strategies at each threshold -- treat_all goes negative at high pt
        # for any moderate prevalence, so filling down to treat_all alone
        # let the shaded region cross below treat-none (y=0) too. The
        # correct DCA reference at any given pt is whichever baseline
        # (treat-all or treat-none) is higher; the model only adds real
        # value where it beats BOTH.
        reference = np.maximum(treat_all, 0)
        ax.fill_between(pt_grid, reference, nb, where=(nb >= reference),
                        color=col, alpha=0.25, zorder=2, interpolate=True)
        ax.plot(pt_grid, nb, color=col, lw=2.2, zorder=3)
    else:
        ax.plot(pt_grid, nb, color=col, lw=2.4, marker="o", markevery=25,
                ms=5, zorder=3, mfc=col, mec="white", mew=0.8)

    ax.plot(pt_grid, treat_all, color="#999999", lw=1.2, ls=":", zorder=1)
    ax.axhline(0, color="#555555", lw=1.2, ls="--", zorder=1)

    ax.set_xlim(0.05, 0.85)
    ax.set_ylim(*ylim)
    ax.set_xticks([0.2, 0.5, 0.8])
    ax.tick_params(labelsize=15, length=3, pad=2)
    ax.set_xlabel("Threshold (pt)", fontsize=13.5, fontweight="bold")
    if first:
        ax.set_ylabel("Net benefit", fontsize=14, fontweight="bold")
    else:
        ax.set_yticklabels([])
    if show_label:
        ax.set_title(TILE_LABEL.get(cancer, cancer), color=col, fontsize=18, fontweight="bold", pad=4)
    for sp in ax.spines.values():
        sp.set_color(_SP["color"]); sp.set_linewidth(_SP["lw"])


def build_dca_mosaic(pred_dict, fixed_thr, style, suptitle, stem, add_macro=False):
    """fixed_thr unused (DCA sweeps threshold itself) -- kept for call-signature
    symmetry with the other row builders. style: 'area' (Part B) or 'line' (Part A)."""
    pt_grid = np.linspace(0.05, 0.85, 150)
    n_tiles = 13 if add_macro else 12
    fig, axes = plt.subplots(1, n_tiles, figsize=(30 * n_tiles / 12, 4.2), squeeze=True)
    fig.subplots_adjust(wspace=0.06, left=0.02, right=0.998,
                        top=0.80, bottom=0.18)

    nb_curves, ymaxs = [], []
    per_cancer = {}
    for c in CANCER_ORDER:
        pdata = load_preds(pred_dict[c])
        nb = _dca_curve(pdata, pt_grid)
        prev = float(pdata["true_label"].mean())
        per_cancer[c] = (pdata, nb, prev)
        nb_curves.append(nb)
        ymaxs.append(nb.max())

    # ylim deliberately excludes treat-all's own range: treat-all's
    # prev - (1-prev)*pt/(1-pt) blows up toward -2/-3 as pt->0.85 for any
    # moderate prevalence (the FP penalty term is unbounded, not a data
    # issue), which was compressing the actually-informative model curves
    # into the top third of every tile. Fixed floor of -0.3 still shows
    # treat-all diving off the bottom edge as a visual cue without
    # sacrificing the vertical space that matters.
    ylim = (-0.3, max(ymaxs) * 1.15)

    for i, c in enumerate(CANCER_ORDER):
        pdata, nb, prev = per_cancer[c]
        _dca_ax(axes[i], c, PLOT_COLOR[c], pt_grid, nb, prev, style, ylim, first=(i == 0))

    if add_macro:
        nb_macro = np.mean(np.array(nb_curves), axis=0)
        prev_macro = float(np.mean([per_cancer[c][2] for c in CANCER_ORDER]))
        _dca_ax(axes[-1], "Macro\nmean", MACRO_COLOR, pt_grid, nb_macro, prev_macro,
               style, ylim, first=False)

    legend_handles = [
        plt.Line2D([0], [0], color="#999999", lw=1.2, ls=":", label="Treat all"),
        plt.Line2D([0], [0], color="#555555", lw=1.2, ls="--", label="Treat none"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, 0.005), fontsize=10, framealpha=0.92)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    save(fig, stem)


# ─── likelihood-ratio row (LR-, LR+) ──────────────────────────────────────────
# Prevalence-independent by construction -- the reason this row exists at
# all is that PPV was flattened to ~0.91-1.00 across every cancer by the
# external cohorts' case enrichment (prevalence 0.51-0.90), telling the
# reader almost nothing. LR doesn't have that problem. Part B tile = LR-
# ("how much does a negative result rule OUT the cancer" -- screening's job);
# Part A tile = LR+ ("how much does a positive result rule IN the cancer" --
# confirmation's job). Log-scale axis with shaded EBM interpretation bands
# (Deeks & Altman 2004 conventions) so each tile carries its own reference
# without needing extra caption text.
def _haldane_anscombe(tp, fn, tn, fp):
    """+0.5 continuity correction applied only when a cell is zero (avoids
    LR = 0 or infinity, which can't be plotted/CI'd on a log axis)."""
    if 0 in (tp, fn, tn, fp):
        return tp + 0.5, fn + 0.5, tn + 0.5, fp + 0.5
    return tp, fn, tn, fp


def _lr_value_ci(tp, fn, tn, fp, kind):
    tp, fn, tn, fp = _haldane_anscombe(tp, fn, tn, fp)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    if kind == "LR-":
        val = (1 - sens) / spec
        se_ln = np.sqrt((1 - sens) / (sens * (tp + fn)) + spec / ((1 - spec) * (tn + fp)))
    else:
        val = sens / (1 - spec)
        se_ln = np.sqrt((1 - sens) / (sens * (tp + fn)) + spec / ((1 - spec) * (tn + fp)))
    ln_val = np.log(val)
    lo, hi = np.exp(ln_val - 1.96 * se_ln), np.exp(ln_val + 1.96 * se_ln)
    return val, lo, hi


def _lr_tile_ax(ax, cancer, col, val, lo, hi, kind, first, show_label=True):
    """LR+ (Part A) and LR- (Part B) are now genuinely different chart
    types, not just different color bands on the same lollipop: LR+ is a
    BAR growing up from the LR=1 baseline (area-based -- "taller bar = more
    rule-in strength"), LR- stays a plain DOT + whisker (point-based, no
    fill). The two rows can't be visually confused even skimmed at mosaic
    scale. Whisker/cap/marker color is now the cancer's own color throughout
    (was black before -- strict per-cancer color identity). The value label
    is offset HORIZONTALLY from the whisker instead of stacked above it, so
    it never overlaps the CI bar regardless of how wide Haldane-Anscombe
    correction makes it."""
    ax.set_yscale("log")

    # Axis range must contain every point estimate AND its CI. The earlier
    # limits (LR- floor 0.01, LR+ ceiling 200) clipped real values off the
    # plot: Part B CLL's LR- point estimate is 0.0093, so its marker was not
    # drawn at all, and the Part A upper CI bounds for AML (472), CLL (500)
    # and ENDC (358) ran past the ceiling. Widened so nothing is silently
    # dropped -- a clipped point estimate reads as missing data.
    if kind == "LR-":
        # EBM bands: <0.1 strong rule-out, 0.1-0.2 moderate, >0.2 weak
        ax.axhspan(0.002, 0.1, color="#2a9d5c", alpha=0.10, zorder=0)
        ax.axhspan(0.1, 0.2, color="#e9c46a", alpha=0.14, zorder=0)
        ax.axhspan(0.2, 1.3, color="#e76f51", alpha=0.10, zorder=0)
        ax.set_ylim(0.002, 1.3)
    else:
        # EBM bands: >10 strong rule-in, 5-10 moderate, 1-5 weak
        ax.axhspan(10, 600, color="#2a9d5c", alpha=0.10, zorder=0)
        ax.axhspan(5, 10, color="#e9c46a", alpha=0.14, zorder=0)
        ax.axhspan(1, 5, color="#e76f51", alpha=0.10, zorder=0)
        ax.set_ylim(0.8, 600)

    ax.axhline(1.0, color="#888888", lw=1.3, ls="--", zorder=1)
    CAP = 0.12

    if kind == "LR+":
        # Shift the bar and CI left to reserve a dedicated text lane on the
        # right.  This keeps the estimate clear of both the vertical whisker
        # and its caps in the narrow 12-tile Figure 8 strip.
        lr_x = 0.36
        ax.bar([lr_x], [val - 1], bottom=1, width=0.42, color=col, alpha=0.80,
              edgecolor=col, linewidth=1.4, zorder=2)
        ax.plot([lr_x, lr_x], [lo, hi], color=col, lw=2.6, zorder=4, solid_capstyle="butt")
        for r in (lo, hi):
            ax.plot([lr_x - CAP, lr_x + CAP], [r, r], color=col, lw=2.2, zorder=4)
        ax.scatter([lr_x], [val], s=70, color="white", ec=col, lw=2.0, zorder=5)
    else:
        ax.plot([0.5, 0.5], [lo, hi], color=col, lw=2.6, zorder=3, solid_capstyle="butt")
        for r in (lo, hi):
            ax.plot([0.5 - CAP, 0.5 + CAP], [r, r], color=col, lw=2.2, zorder=3)
        ax.scatter([0.5], [val], s=120, color=col, ec="white", lw=1.6, zorder=4)

    value_text = f"{val:.1f}" if val >= 10 else f"{val:.2f}"
    if kind == "LR+":
        ax.text(0.63, val, value_text, ha="left", va="center",
                fontsize=14, fontweight="bold", color=col, clip_on=True)
    else:
        ax.text(0.74, val, value_text, ha="left", va="center",
                fontsize=16, fontweight="bold", color=col, clip_on=True)

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=10, length=3)
    if not first:
        ax.set_yticklabels([])
    if show_label:
        ax.set_title(TILE_LABEL.get(cancer, cancer), color=col, fontsize=18, fontweight="bold", pad=4)
    for sp in ax.spines.values():
        sp.set_color(_SP["color"]); sp.set_linewidth(_SP["lw"])


def build_lr_mosaic(pred_dict, fixed_thr, kind, suptitle, stem, add_macro=False):
    """kind: 'LR-' (Part B, fixed 0.50) or 'LR+' (Part A, fixed 0.80).

    Both are passed in via fixed_thr by the calling script. The
    youden_thr() fallback below is retained for ad-hoc use only -- no live
    fig8/fig9 call reaches it as of 2026-08-16.
    """
    n_tiles = 13 if add_macro else 12
    fig, axes = plt.subplots(1, n_tiles, figsize=(30 * n_tiles / 12, 4.6), squeeze=True)
    fig.subplots_adjust(wspace=0.10, left=0.02, right=0.998,
                        top=0.80, bottom=0.10)

    vals = []
    for i, cancer in enumerate(CANCER_ORDER):
        pdata = load_preds(pred_dict[cancer])
        y_true = pdata["true_label"].values
        if fixed_thr is not None:
            thr = fixed_thr
        else:
            fpr, tpr, thrs = roc_curve(y_true, pdata["prob"].values)
            thr = youden_thr(fpr, tpr, thrs)
        y_pred = (pdata["prob"].values >= thr).astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())

        val, lo, hi = _lr_value_ci(tp, fn, tn, fp, kind)
        vals.append(val)
        _lr_tile_ax(axes[i], cancer, PLOT_COLOR[cancer], val, lo, hi, kind, first=(i == 0))

    if add_macro:
        # geometric mean -- LR is a ratio, so the arithmetic mean of 12
        # ratios is not the statistically correct "average LR"; geometric
        # mean is, and it happens to equal the arithmetic mean on the log
        # scale this row is already plotted on
        macro_val = float(np.exp(np.mean(np.log(vals))))
        _lr_tile_ax(axes[-1], "Macro\nmean", MACRO_COLOR, macro_val, macro_val, macro_val,
                   kind, first=False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#2a9d5c", alpha=0.25, label="strong"),
        plt.Rectangle((0, 0), 1, 1, fc="#e9c46a", alpha=0.30, label="moderate"),
        plt.Rectangle((0, 0), 1, 1, fc="#e76f51", alpha=0.25, label="weak/none"),
        plt.Line2D([0], [0], color="#888888", lw=1.3, ls="--", label="LR = 1 (no info)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
              bbox_to_anchor=(0.5, 0.005), fontsize=10, framealpha=0.92)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    save(fig, stem)


# ─── external calibration row (reliability curves, Phase 2 row 7) ────────────
# Same binning/ECE logic as fig7h (internal calibration), pointed at the
# EXTERNAL cohorts instead. Built as filmstrip tiles (not one overlaid plot
# like fig7h) to match every other row's mosaic format. Part B macro tile =
# pooled data (same convention as fig8a's macro boxplot -- Part B is one
# model, so pooling its scores across cohorts is defensible); Part A stays
# per-cancer only, no macro cell, same reasoning as every other Part-A row.
def _calib_curve(pdata, n_bins=4):
    y_true = pdata["true_label"].values
    y_score = pdata["prob"].values
    try:
        bins = pd.qcut(y_score, n_bins, duplicates="drop")
    except ValueError:
        bins = pd.qcut(y_score, max(2, min(3, len(np.unique(y_score)))), duplicates="drop")
    bdf = pd.DataFrame({"bin": bins, "y_true": y_true, "y_score": y_score})
    grp = (bdf.groupby("bin", observed=True)
           .agg(mean_pred=("y_score", "mean"), obs_rate=("y_true", "mean"),
                n=("y_true", "size"))
           .sort_values("mean_pred"))
    ece = float((grp["n"] / len(y_true) * (grp["mean_pred"] - grp["obs_rate"]).abs()).sum())
    # Brier score: mean squared error between raw score and outcome, on the
    # UNBINNED data -- same convention as fig7h. Sensitive to calibration
    # AND sharpness together, unlike ECE which is binning-dependent.
    brier = float(np.mean((y_score - y_true) ** 2))
    return grp, ece, brier


def _calib_tile_ax(ax, cancer, col, grp, ece, brier, first, show_label=True):
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1.4, ls="--", zorder=1)
    ax.plot(grp["mean_pred"], grp["obs_rate"], color=col, lw=2.2, marker="o",
            ms=6, alpha=0.90, zorder=3)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=13, length=3, pad=2)
    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    if first:
        ax.set_ylabel("Observed rate", fontsize=13, fontweight="bold")
    else:
        ax.set_yticklabels([])
    # ECE top-left, Brier bottom-right (empty corner in a calibration plot,
    # since data always runs bottom-left -> top-right) -- stacking both in
    # one spot at this font size was cramped
    ax.text(0.05, 0.95, f"ECE={ece:.2f}", ha="left", va="top",
            fontsize=20, fontweight="bold", color=col, transform=ax.transAxes)
    ax.text(0.95, 0.05, f"Brier={brier:.2f}", ha="right", va="bottom",
            fontsize=20, fontweight="bold", color=col, transform=ax.transAxes)
    if show_label:
        ax.set_title(TILE_LABEL.get(cancer, cancer), color=col, fontsize=18, fontweight="bold", pad=4)
    for sp in ax.spines.values():
        sp.set_color(_SP["color"]); sp.set_linewidth(_SP["lw"])


def build_calibration_mosaic(pred_dict, suptitle, stem, add_macro=False):
    n_tiles = 13 if add_macro else 12
    fig, axes = plt.subplots(1, n_tiles, figsize=(30 * n_tiles / 12, 4.0), squeeze=True)
    # no fig.legend() in this builder -- bottom=0.18 was leaving dead space
    # with nothing to fill it (that margin only makes sense when a legend
    # sits below the row, e.g. build_dca_mosaic/build_lr_mosaic)
    fig.subplots_adjust(wspace=0.06, left=0.02, right=0.998,
                        top=0.82, bottom=0.13)

    for i, cancer in enumerate(CANCER_ORDER):
        pdata = load_preds(pred_dict[cancer])
        grp, ece, brier = _calib_curve(pdata)
        _calib_tile_ax(axes[i], cancer, PLOT_COLOR[cancer], grp, ece, brier, first=(i == 0))

    if add_macro:
        pooled = pd.concat([load_preds(pred_dict[c]) for c in CANCER_ORDER])
        grp, ece, brier = _calib_curve(pooled, n_bins=8)
        _calib_tile_ax(axes[-1], "Macro\nmean", MACRO_COLOR, grp, ece, brier, first=False)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    save(fig, stem)


# =============================================================================
# Panel 8g – Slope plot: internal OvR AUC  vs  external validation AUC
# =============================================================================
def panel_auc_slope():
    INTERNAL_CSV = (ROOT / "revise_plan/part_b_multiclass"
                    / "vRSX_v11_locked_reproducer/tables/locked_per_class_recall_auc.csv")
    EXTERNAL_CSV = (EV / "tables/locked25_external_validation_final_summary.csv")

    internal = pd.read_csv(INTERNAL_CSV).set_index("label")
    external = pd.read_csv(EXTERNAL_CSV)
    # Keep "main" entries only; for LUNGC two main entries exist → average
    ext_main = (external[external["status"] == "main"]
                .groupby("endpoint")["auc_mean"].mean())

    # Build per-cancer paired (internal, external) AUC
    records = []
    for c in CANCER_ORDER:
        int_auc = float(internal.loc[c, "ovr_auc"])
        ext_key = c.upper()
        if ext_key not in ext_main.index:
            continue
        ext_auc = float(ext_main[ext_key])
        records.append({"cancer": c, "internal": int_auc, "external": ext_auc})
    df = pd.DataFrame(records).sort_values("internal", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(left=0.16, right=0.84, top=0.88, bottom=0.10)

    X_INT, X_EXT = 0.0, 1.0
    y_lo, y_hi = 0.82, 1.01

    ax.axvline(X_INT, color="#cccccc", lw=1.5, zorder=1)
    ax.axvline(X_EXT, color="#cccccc", lw=1.5, zorder=1)

    for _, row in df.iterrows():
        c   = row["cancer"]
        col = PLOT_COLOR[c]
        y_i = row["internal"]
        y_e = row["external"]
        delta = y_e - y_i
        lw = 2.2

        ax.plot([X_INT, X_EXT], [y_i, y_e],
                color=col, lw=lw, zorder=2, solid_capstyle="round")
        ax.scatter([X_INT, X_EXT], [y_i, y_e],
                   s=55, color=col, zorder=3, ec="white", lw=0.8)

        full = CANCER_FULL[c]
        label_left = f"{full} ({c})" if full != c else c

        ax.text(X_INT - 0.025, y_i, label_left,
                ha="right", va="center",
                fontsize=9.5, fontweight="bold", color=col)
        ax.text(X_INT + 0.010, y_i,
                f"{y_i:.3f}",
                ha="left", va="center",
                fontsize=8.5, color=col)
        ax.text(X_EXT + 0.025, y_e,
                f"{y_e:.3f}",
                ha="left", va="center",
                fontsize=8.5, color=col)
        sign = "▲" if delta >= 0 else "▼"
        ax.text((X_INT + X_EXT) / 2, (y_i + y_e) / 2,
                f"{sign}{abs(delta):.3f}",
                ha="center", va="bottom",
                fontsize=7.5, color=col, alpha=0.85)

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks([X_INT, X_EXT])
    ax.set_xticklabels(["Internal\n(held-out test)", "External\n(independent cohorts)"],
                        fontsize=11, fontweight="bold")
    ax.set_ylabel("OvR AUC", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", lw=0.8, zorder=0)

    ax.set_title(
        "Part B locked 25-protein panel  ·  generalization to external datasets\n"
        "One line per cancer type  ·  ▲ = AUC gain  ·  ▼ = AUC drop",
        fontsize=12, fontweight="bold", pad=8,
    )

    save(fig, "fig8g_auc_slope")
