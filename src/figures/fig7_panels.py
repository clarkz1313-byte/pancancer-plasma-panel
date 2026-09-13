#!/usr/bin/env python3
"""
fig7_panels.py -- Slide 7: Two-stage workflow visualizations
Part A = single-cancer confirmation panels (Part B screening -> Part A confirmation)

Panels:
  7a  panel_efficiency_scatter  -- [DEPRECATED, trivial] Part A: panel size vs AUC
  7b  panel_ridgeline_s7        -- [DEPRECATED, superseded by 7h] score distribution among true positives only
  7c  panel_sens_spec_bars      -- Part B screening sensitivity vs Part A confirmation specificity
  7d  panel_confusion_grid      -- Part A internal confusion matrices (12 cancers, fixed 0.50
                                    confirmation threshold); radar includes 95% CI whiskers
                                    (Wilson for Sens/Spec, bootstrap for AUC)
  7e  panel_pr_overlay          -- Part A: overlaid precision-recall curves, all 12 cancers + macro AUPRC
  7f  panel_sens_spec_ci        -- [DEPRECATED, superseded] CI now lives in 7d's radar whiskers instead
  7g  panel_ppv_npv_prevalence  -- [DEPRECATED, superseded by 7i] PPV vs. ASSUMED prevalence (hypothetical sweep)
  7h  panel_calibration         -- Part A: calibration/reliability diagram, full test set (cases + controls)
  7i  panel_decision_curve      -- Part A: decision curve analysis, net benefit from OBSERVED test data

7b and 7g replaced: both only re-sliced information fig7d's radar/confusion
matrix already show (7b = true-positive scores only; 7g = a hypothetical,
unsourced prevalence sweep that made several cancers look uniformly bad
without reflecting this cohort's actual data). 7h and 7i use the FULL
observed test set directly and answer genuinely different questions --
"are the raw scores trustworthy as probabilities?" (7h) and "does using this
panel actually beat the naive treat-all/treat-none baselines?" (7i).

All panels use CANCER_COLOR consistently across all 12 cancer types.

Output: figurev5/output/fig7{a,b,c,d,e,f,g,h,i}_*.pdf/.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import pandas as pd
from figure_labels import display_label
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from publication_style import enable_text_scaling

enable_text_scaling(factor=1.50, minimum=11.5, axis_label_scale=0.75)

ROOT     = Path(__file__).resolve().parents[2]
TABLES   = ROOT / "revise_plan/part_b_multiclass/vRSX_v11_locked_reproducer/tables"
SUPP     = ROOT / "supplementary_tables/Supplementary_Table_S1_single_panel_protein_summary.csv"
PARTA_DIR = ROOT / "revise_plan/part_a_single_seed52_aligned"
PARTA_SEED52_PERF = PARTA_DIR / "single_panel_seed52_aligned_performance.csv"
OUT      = ROOT / "figurev6/output"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 400

CANCERS = ["AML","BRC","CLL","CRC","CVX","ENDC","GLIOM","LUNGC","LYMPH","MYEL","OVC","PRC"]
CANCER_FULL = {
    "AML":"AML","BRC":"Breast Cancer","CLL":"CLL",
    "CRC":"Colorectal Cancer","CVX":"Cervical Cancer",
    "ENDC":"Endometrial Cancer","GLIOM":"Glioma",
    "LUNGC":"Lung Cancer","LYMPH":"DLBCL",
    "MYEL":"Myeloma","OVC":"Ovarian Cancer","PRC":"Prostate Cancer",
}
CANCER_COLOR = {
    "AML":"#b22222","BRC":"#c97b63","CLL":"#7a3e9d","CRC":"#d68600","CVX":"#c13d86",
    "ENDC":"#8e5d2c","GLIOM":"#1a8db8","LUNGC":"#2e8b57","LYMPH":"#3856a6",
    "MYEL":"#8c564b","OVC":"#d1495b","PRC":"#008b8b",
}

def cancer_label(code):
    full = CANCER_FULL[code]
    if code == "LYMPH":
        return display_label(code)
    return full if full == code else f"{full} ({code})"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 12, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.linewidth": 1.2,
})


# =============================================================================
# Panel 7a -- Part A panel efficiency scatter
# x = deploy panel size (proteins)  |  y = single-panel internal deploy AUC
# =============================================================================
def panel_efficiency_scatter():
    supp = pd.read_csv(SUPP).set_index("target_class")

    # Manual label nudges to avoid overlap (dx, dy, ha, va)
    NUDGE = {
        "AML":   (-0.25,  0.004, "right", "bottom"),  # x=3,  y=1.000
        "CLL":   (-0.25,  0.004, "right", "bottom"),  # x=2,  y=0.997
        "GLIOM": ( 0.25, -0.011, "left",  "top"),     # x=3,  y=0.937 (below AML)
        "LYMPH": ( 0.25,  0.004, "left",  "bottom"),  # x=5,  y=0.949
        "LUNGC": ( 0.25, -0.011, "left",  "top"),     # x=6,  y=0.902
        "MYEL":  (-0.25,  0.004, "right", "bottom"),  # x=8,  y=1.000
        "OVC":   (-0.25,  0.004, "right", "bottom"),  # x=9,  y=0.947
        "CRC":   ( 0.25,  0.004, "left",  "bottom"),  # x=10, y=0.945
        "ENDC":  ( 0.25, -0.011, "left",  "top"),     # x=10, y=0.838
        "CVX":   (-0.25,  0.004, "right", "bottom"),  # x=11, y=0.921
        "PRC":   ( 0.25,  0.006, "left",  "bottom"),  # x=13, y=0.938
        "BRC":   ( 0.25, -0.012, "left",  "top"),     # x=13, y=0.920 (below PRC)
    }

    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.84, bottom=0.12)

    xs, ys = [], []
    for c in CANCERS:
        x   = int(supp.loc[c, "deploy_panel_size"])
        y   = float(supp.loc[c, "single_deploy_auc"])
        col = CANCER_COLOR[c]
        xs.append(x); ys.append(y)

        ax.scatter(x, y, s=180, color=col, zorder=4, ec="white", lw=1.0)

        dx, dy, ha, va = NUDGE[c]
        lbl = cancer_label(c)
        ax.text(x + dx, y + dy, lbl,
                ha=ha, va=va, fontsize=9.5, fontweight="bold", color=col)

    # Trend line
    xs_a, ys_a = np.array(xs), np.array(ys)
    r = np.corrcoef(xs_a, ys_a)[0, 1]
    m, b = np.polyfit(xs_a, ys_a, 1)
    xfit = np.linspace(1.0, 14.5, 200)
    ax.plot(xfit, m * xfit + b, color="#aaaaaa", lw=1.6, ls="--",
            zorder=1, label=f"Linear trend  r = {r:.2f}")

    ax.set_xlabel("Confirmation panel size  (number of proteins)", fontsize=13, labelpad=8)
    ax.set_ylabel("Single-panel internal test AUC", fontsize=13, labelpad=8)
    ax.set_xlim(0.5, 15.0)
    ax.set_ylim(0.80, 1.04)
    ax.set_xticks(range(1, 15))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.grid(axis="y", color="#eeeeee", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=10.5, framealpha=0.88)

    ax.set_title(
        "Cancer-specific panels: panel size vs. classification performance\n"
        "Focused single-cancer panels (2-13 proteins) achieve high AUC across all 12 cancers\n"
        "AML (3P, AUC 1.00)  *  CLL (2P, AUC 1.00)  *  ENDC hardest (10P, AUC 0.84)",
        fontsize=12, fontweight="bold", pad=10,
    )

    for ext in ("pdf", "png"):
        p = OUT / f"fig7a_panel_efficiency.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


# =============================================================================
# Panel 7b -- Part A confirmation confidence among true positive cases
# NOTE: an earlier version of this panel reused fig5b's Part B (multiclass
# screening) data, just retitled -- that was wrong for a slide about Part A
# single-class confirmation panels ("why does the x-axis say multiclass
# screening score if we're doing single-class performance now?"). Rebuilt
# from each cancer's OWN deploy panel: for the true-positive samples in that
# cancer's confirmation test set, what score does the confirmation panel
# itself assign them? This is the genuine single-class analog of fig5b/7b's
# original intent, not a relabeled Part B artifact.
# =============================================================================
def panel_ridgeline_s7():
    supp = pd.read_csv(SUPP).set_index("target_class")

    per_cancer = {}
    for c in CANCERS:
        n_prot = int(supp.loc[c, "deploy_panel_size"])
        panel_name = f"{c} Top {n_prot}"
        pred_path = PARTA_DIR / c.lower() / "tables" / f"{c.lower()}_test_predictions.csv"
        df = pd.read_csv(pred_path)
        sub = df[(df["panel"] == panel_name) & (df["partition"] == "test")]
        true_pos = sub[sub["y_true"] == 1]
        per_cancer[c] = dict(
            vals=true_pos["y_score"].values,
            correct=(true_pos["y_pred"] == 1).values,
        )

    medians = {c: float(np.median(per_cancer[c]["vals"])) for c in CANCERS}
    order = sorted(CANCERS, key=lambda c: medians[c])

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.subplots_adjust(left=0.24, right=0.96, top=0.85, bottom=0.09)

    slot_h = 1.0
    x_grid = np.linspace(0, 1, 300)

    for i, c in enumerate(order):
        vals   = per_cancer[c]["vals"]
        y_base = i * slot_h
        col    = CANCER_COLOR[c]

        if len(vals) >= 3 and np.std(vals) > 1e-9:
            try:
                kde       = gaussian_kde(vals, bw_method=0.28)
                dens      = kde(x_grid)
                dens_norm = dens / dens.max() * 0.85
                ax.fill_between(x_grid, y_base, y_base + dens_norm,
                                color=col, alpha=0.42, zorder=2)
                ax.plot(x_grid, y_base + dens_norm,
                        color=col, lw=1.6, zorder=3)
            except np.linalg.LinAlgError:
                pass

        ax.axhline(y_base, color="#cccccc", lw=0.6, zorder=1)

        med = medians[c]
        ax.plot([med, med], [y_base, y_base + 0.85],
                color="black", lw=1.6, zorder=4)

        rng = np.random.default_rng(42 + i)
        y_jit = rng.uniform(-0.10, 0.10, size=len(vals))
        correct_mask = per_cancer[c]["correct"]
        for v, yy, ok in zip(vals, y_jit, correct_mask):
            fc = col if ok else "white"
            ax.scatter(v, y_base + yy, s=20, fc=fc, ec=col, lw=0.9,
                       zorder=5, alpha=0.85)

        ax.text(1.015, y_base + 0.35, f"{med:.2f}  n={len(vals)}",
                va="center", ha="left", fontsize=9, fontweight="bold", color=col)

    ax.set_yticks([i * slot_h + 0.35 for i in range(len(order))])
    ax.set_yticklabels([cancer_label(c) for c in order],
                       fontsize=11, fontweight="bold")
    for tick, c in zip(ax.get_yticklabels(), order):
        tick.set_color(CANCER_COLOR[c])

    ax.set_xlim(-0.02, 1.16)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"],
                       fontsize=11, fontweight="bold")
    ax.set_xlabel("Confirmation panel score for the true positive cases",
                  fontsize=13, fontweight="bold", labelpad=8)

    ax.axvline(0.5, color="#888888", lw=1.4, ls="--", zorder=1)
    ax.set_ylim(-0.4, len(order) * slot_h + 0.4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.set_title(
        "Cancer-specific panels: score distribution among true positive cases\n"
        "Internal held-out test  *  deploy panel  *  sorted by median score\n"
        "solid dot = caught (score >= 0.50 confirmation threshold)  *  hollow dot = missed  *  vertical bar = median",
        fontsize=12.5, fontweight="bold", pad=10,
    )

    for ext in ("pdf", "png"):
        p = OUT / f"fig7b_ridgeline_confirmation.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


# =============================================================================
# Panel 7c -- Screening sensitivity (Part B) vs Confirmation specificity (Part A)
# Grouped horizontal bars, NOT a slope -- these are two different metrics for
# two different clinical jobs, so a connecting line would wrongly imply
# "performance dropped" when in fact Part B is judged on catching cases
# (recall) and Part A is judged on ruling out false alarms (specificity).
# =============================================================================
def panel_sens_spec_bars():
    auc_b = pd.read_csv(TABLES / "locked_per_class_recall_auc.csv").set_index("label")
    # seed-52-aligned Part A specificity at the fixed 0.80 confirmation
    # threshold -- NOT Supplementary_Table_S1's single_deploy_specificity,
    # which is the superseded seed-42/threshold-0.5 number
    parta_perf = pd.read_csv(PARTA_SEED52_PERF).set_index("target_class")

    records = []
    for c in CANCERS:
        records.append({
            "cancer": c,
            "sens":   float(auc_b.loc[c, "recall"]),
            "spec":   float(parta_perf.loc[c, "test_specificity"]),
        })
    df = pd.DataFrame(records).sort_values("sens", ascending=True).reset_index(drop=True)

    n = len(df)
    fig, ax = plt.subplots(figsize=(9.5, 9.0))
    fig.subplots_adjust(left=0.20, right=0.90, top=0.80, bottom=0.11)

    bar_h = 0.36
    for i, row in df.iterrows():
        c   = row["cancer"]
        col = CANCER_COLOR[c]
        light = tuple(np.array(mc.to_rgba(col))[:3] * 0.55 + 0.45)  # lighter shade

        ax.barh(i + bar_h/2 + 0.02, row["sens"], height=bar_h,
                color=light, ec=col, lw=1.1, zorder=3)
        ax.barh(i - bar_h/2 - 0.02, row["spec"], height=bar_h,
                color=col, ec=col, lw=1.1, zorder=3)

        ax.text(row["sens"] + 0.012, i + bar_h/2 + 0.02, f"{row['sens']:.2f}",
                va="center", ha="left", fontsize=9, fontweight="bold", color=col)
        ax.text(row["spec"] + 0.012, i - bar_h/2 - 0.02, f"{row['spec']:.2f}",
                va="center", ha="left", fontsize=9, fontweight="bold", color=col)

    ax.set_yticks(range(n))
    ax.set_yticklabels([cancer_label(c) for c in df["cancer"]],
                       fontsize=10.5, fontweight="bold")
    for tick, c in zip(ax.get_yticklabels(), df["cancer"]):
        tick.set_color(CANCER_COLOR[c])

    ax.set_xlim(0, 1.14)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Metric value", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylim(-0.7, n - 0.3)
    ax.axvline(1.0, color="#cccccc", lw=1.0, ls="--", zorder=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", lw=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#999999", ec="#555555",
                      label="Cancer-specific panel specificity"),
        plt.Rectangle((0, 0), 1, 1, fc=tuple(np.array(mc.to_rgba("#999999"))[:3]*0.55+0.45),
                      ec="#555555", label="Multiclass panel sensitivity (recall)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
              bbox_to_anchor=(0.56, 0.005), fontsize=9.5, framealpha=0.92)

    ax.set_title(
        "Different statistical objectives: multiclass recall and cancer-specific specificity\n"
        "Seed-52 patient-aligned internal split | both metrics vary by cancer\n"
        "see fig7d/7h/7i for the per-cancer confusion, calibration, and decision-curve detail",
        fontsize=11.5, fontweight="bold", pad=10,
    )

    for ext in ("pdf", "png"):
        p = OUT / f"fig7c_sens_spec_bars.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


# =============================================================================
# Panel 7d -- Part A internal confusion matrices (12 cancers, deploy panel)
# Each tile: the actual deployed panel "{CANCER} Top {deploy_panel_size}"
# (matches Supplementary Table S1 deploy_proteins), internal held-out test,
# fixed 0.5 threshold -- reproduces single_deploy_recall/specificity exactly.
# Note: best_panel in the supp table is the AUC-optimal search result, which
# is often LARGER than the deployed panel -- do not use it here.
# =============================================================================
def _blend(col, frac):
    rgb = np.array(mc.to_rgba(col))[:3]
    return tuple((rgb * frac + (1 - frac)).clip(0, 1))


def _cm_palette(col):
    return (
        _blend(col, 1.00),  # TP full color
        _blend(col, 0.62),  # TN medium shade
        _blend(col, 0.15),  # FN very light
        _blend(col, 0.09),  # FP lightest
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


def _bootstrap_auc_ci(y_true, y_score, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.sum() in (0, n):
            continue
        try:
            boots.append(roc_auc_score(yt, y_score[idx]))
        except Exception:
            pass
    arr = np.array(boots)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def panel_confusion_grid():
    """Mini radar (AUC/Spec/Sens) paired with the confusion matrix for each
    cancer -- deliberately NOT a plain CM grid like fig8c/f, and not bars
    either: the radar's triangle shape makes the sens/spec trade-off (e.g.
    ENDC's collapsed Sens spoke) visually obvious at a glance."""
    supp = pd.read_csv(SUPP).set_index("target_class")

    fig = plt.figure(figsize=(24, 6.7))
    gs = fig.add_gridspec(2, 12, width_ratios=[1, 1] * 6,
                          wspace=0.42, hspace=0.08,
                          left=0.035, right=0.995, top=0.96, bottom=0.06)

    metric_names  = ["AUC", "Sp", "Se"]
    angles        = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    for i, c in enumerate(CANCERS):
        row  = i // 6
        pair = i % 6
        ax_m = fig.add_subplot(gs[row, pair * 2])
        ax_r = fig.add_subplot(gs[row, pair * 2 + 1], projection="polar")
        # PolarAxes already forces its own box to be square (equal aspect built in);
        # the confusion-matrix axes needs it set explicitly so cells render as squares.
        ax_m.set_box_aspect(1)

        col = CANCER_COLOR[c]
        n_prot = int(supp.loc[c, "deploy_panel_size"])
        panel_name = f"{c} Top {n_prot}"

        pred_path = PARTA_DIR / c.lower() / "tables" / f"{c.lower()}_test_predictions.csv"
        df = pd.read_csv(pred_path)
        sub = df[(df["panel"] == panel_name) & (df["partition"] == "test")]

        y_true  = sub["y_true"].values
        y_score = sub["y_score"].values
        y_pred  = sub["y_pred"].values

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())

        sens, sens_lo, sens_hi = _wilson_ci(tp, tp + fn)
        spec, spec_lo, spec_hi = _wilson_ci(tn, tn + fp)
        auc_v = roc_auc_score(y_true, y_score)
        auc_lo, auc_hi = _bootstrap_auc_ci(y_true, y_score, seed=hash(c) % 10000)

        # ---- mini radar: AUC / Specificity / Sensitivity, with 95% CI band ----
        # Wilson CI for Sens/Spec (binomial), bootstrap CI for AUC -- shown as a
        # shaded band so a "Sens = 1.00" off a handful of positives (e.g. AML
        # n=14, CLL n=13) doesn't read as more certain than it is.
        vals        = [auc_v, spec, sens]
        vals_closed = vals + vals[:1]
        his         = [auc_hi, spec_hi, sens_hi]
        his_closed  = his + his[:1]
        los         = [auc_lo, spec_lo, sens_lo]
        los_closed  = los + los[:1]

        ax_r.set_theta_offset(np.pi / 2)
        ax_r.set_theta_direction(-1)
        ax_r.set_ylim(0, 1.20)
        ax_r.set_xticks(angles)
        ax_r.set_xticklabels([])
        ax_r.set_yticks([0.5, 1.0])
        ax_r.set_yticklabels([])
        ax_r.grid(color="#dddddd", lw=0.6)
        ax_r.spines["polar"].set_color("#cccccc")

        ax_r.fill(angles_closed, vals_closed, color=col, alpha=0.35, zorder=2)
        ax_r.plot(angles_closed, vals_closed, color=col, lw=1.8, zorder=3)

        # CI whisker per axis (radial error bar) -- the donut-fill CI band used
        # in fig5c's macro radar disappears here whenever the point estimate
        # sits at the CI upper bound (very common with small n, e.g. AML
        # Sens=1.00=Wilson upper bound), since it gets fully swallowed by the
        # point-estimate polygon's own fill. A thin same-hue tick wasn't
        # visible either (blended into the spoke it sits on) -- bold, fully
        # opaque black with perpendicular end-caps, drawn above everything
        # else, reads as a distinct range indicator regardless of what's
        # underneath.
        CAP = 0.05  # angular half-width of each end-cap, radians
        for ang, lo, hi in zip(angles, los, his):
            ax_r.plot([ang, ang], [lo, hi], color="black", lw=2.6, alpha=1.0,
                      zorder=6, solid_capstyle="butt")
            for r in (lo, hi):
                ax_r.plot([ang - CAP, ang + CAP], [r, r], color="black", lw=2.2,
                          alpha=1.0, zorder=6, solid_capstyle="butt")

        ax_r.scatter(angles, vals, s=26, fc=col, ec="white", lw=1.1, zorder=7)

        metric_offsets = {"AUC": (0, 8), "Sp": (12, -4), "Se": (-12, -4)}
        metric_align = {"AUC": ("center", "bottom"),
                        "Sp": ("left", "center"), "Se": ("right", "center")}
        # Match the prediction headers exactly so each mini-panel has one
        # consistent annotation scale.
        metric_fontsize = {"AUC": 10.5, "Sp": 10.5, "Se": 10.5}
        for metric, ang, v in zip(metric_names, angles, vals):
            ha, va = metric_align[metric]
            ax_r.annotate(f"{metric}\n{v:.2f}",
                          xy=(ang, min(v + 0.05, 1.08)),
                          xytext=metric_offsets[metric], textcoords="offset points",
                          ha=ha, va=va, fontsize=metric_fontsize[metric], fontweight="bold",
                          color="black", zorder=8, linespacing=0.98,
                          bbox=dict(facecolor="white", edgecolor="none",
                                    alpha=0.82, pad=0.45))

        # ---- confusion matrix (paired, same row) ----
        tp_c, tn_c, fn_c, fp_c = _cm_palette(col)
        fills = [[tp_c, fn_c], [fp_c, tn_c]]
        counts = [[tp, fn], [fp, tn]]
        txt_cols = [["white", "black"], ["black", "black"]]

        for r in range(2):
            for cc in range(2):
                ax_m.add_patch(plt.Rectangle(
                    (cc, 1 - r), 1, 1,
                    facecolor=fills[r][cc], edgecolor="white", linewidth=2.0, zorder=2))
                ax_m.text(cc + 0.5, 1.5 - r, str(counts[r][cc]),
                          ha="center", va="center", fontsize=17, fontweight="bold",
                          color=txt_cols[r][cc], zorder=3)

        ax_m.set_xlim(0, 2); ax_m.set_ylim(0, 2)
        ax_m.set_aspect("equal", adjustable="box")
        ax_m.set_xticks([0.5, 1.5]); ax_m.set_xticklabels(["+", "−"], fontsize=12, fontweight="bold")
        ax_m.xaxis.tick_top(); ax_m.xaxis.set_label_position("top")
        ax_m.set_xticklabels(["Pred +", "Pred -"], fontsize=10.5, fontweight="bold")
        ax_m.set_yticks([0.5, 1.5])
        ax_m.set_yticklabels(["Ctrl", "Case"] if pair == 0 else ["", ""],
                             fontsize=10.5, fontweight="bold")
        ax_m.tick_params(length=0, pad=2)
        for sp in ax_m.spines.values():
            sp.set_visible(False)

        ax_m.set_title(f"{display_label(c)} · {n_prot}P", color=col, fontsize=14.5,
                       fontweight="bold", pad=18, loc="left")


    for ext in ("pdf", "png"):
        p = OUT / f"fig7d_confusion_grid.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


# =============================================================================
# Panel 7e -- Overlaid precision-recall curves, all 12 Part A confirmation
# panels on one axis + macro-averaged AUPRC curve. Modernized version of the
# "One-vs-rest PR" panel from the old deck (Proteomics-canva9.pdf, slide 7):
# same concept, but CANCER_COLOR-consistent and using the locked deploy panel.
# PR (not ROC) matters here because several cancers have very few positives
# in this cohort (AML n=14, CLL n=13, MYEL n=9) where ROC can look
# over-optimistic; PR curves surface that scarcity directly via precision.
# =============================================================================
def panel_pr_overlay():
    supp = pd.read_csv(SUPP).set_index("target_class")

    fig, ax = plt.subplots(figsize=(11.5, 9.4))
    # Reserve a dedicated legend band below the axes.  In the three-column
    # composite an outside-right legend made the data area narrow and its
    # long labels became unreadably small.
    fig.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.42)

    recall_grid = np.linspace(0, 1, 200)
    interp_precisions = []
    auprc_vals = {}
    line_handles = {}

    for c in CANCERS:
        n_prot = int(supp.loc[c, "deploy_panel_size"])
        panel_name = f"{c} Top {n_prot}"
        pred_path = PARTA_DIR / c.lower() / "tables" / f"{c.lower()}_test_predictions.csv"
        df = pd.read_csv(pred_path)
        sub = df[(df["panel"] == panel_name) & (df["partition"] == "test")]

        y_true, y_score = sub["y_true"].values, sub["y_score"].values
        col = CANCER_COLOR[c]

        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        auprc_vals[c] = ap

        line, = ax.plot(rec, prec, color=col, lw=1.5, alpha=0.80, zorder=2)
        line_handles[c] = line

        # sklearn returns rec descending; sort ascending for interpolation
        order = np.argsort(rec)
        interp_precisions.append(np.interp(recall_grid, rec[order], prec[order]))

    macro_curve = np.mean(interp_precisions, axis=0)
    macro_auprc = float(np.mean(list(auprc_vals.values())))
    macro_line, = ax.plot(recall_grid, macro_curve, color="#A06CD5", lw=3.4, zorder=5,
                          solid_capstyle="round")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Recall (Sensitivity)", fontsize=17, fontweight="bold")
    ax.set_ylabel("Precision (PPV)", fontsize=17, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#eeeeee", lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Legend outside the axes, sorted by AUPRC descending -- avoids the
    # unreadable label pile-up that forms where all curves converge near
    # recall=1 (that's where the old in-plot end-of-curve labels collided).
    order_desc = sorted(CANCERS, key=lambda c: auprc_vals[c], reverse=True)
    handles = [macro_line] + [line_handles[c] for c in order_desc]
    labels  = [f"Macro AUPRC = {macro_auprc:.3f}"] + \
              [f"{display_label(c)}  AP={auprc_vals[c]:.2f}" for c in order_desc]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=3, fontsize=15.5, framealpha=0.96, handlelength=1.8,
              columnspacing=1.25, labelspacing=0.65, borderaxespad=0.0)


    for ext in ("pdf", "png"):
        p = OUT / f"fig7e_pr_overlay.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


# =============================================================================
# Shared helper: Part A deploy-panel confusion counts, all 12 cancers
# =============================================================================
def _load_parta_confusion():
    """TP/FN/TN/FP for each cancer's deployed panel (internal test, fixed 0.80 threshold)."""
    supp = pd.read_csv(SUPP).set_index("target_class")
    out = {}
    for c in CANCERS:
        n_prot = int(supp.loc[c, "deploy_panel_size"])
        panel_name = f"{c} Top {n_prot}"
        pred_path = PARTA_DIR / c.lower() / "tables" / f"{c.lower()}_test_predictions.csv"
        df = pd.read_csv(pred_path)
        sub = df[(df["panel"] == panel_name) & (df["partition"] == "test")]
        y_true, y_pred = sub["y_true"].values, sub["y_pred"].values
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        out[c] = dict(tp=tp, fn=fn, tn=tn, fp=fp, n_prot=n_prot)
    return out


# =============================================================================
# Panel 7f -- Part A sensitivity & specificity with 95% Wilson CI (forest plot)
# Directly answers the reviewer question fig7d's point estimates invite:
# AML/CLL show Sens = 1.00 off only 14/13 positives -- that needs an interval,
# not a bare "1.00". Wilson score interval computed from the exact TP/FN/TN/FP
# counts underlying fig7d (same deploy panel, same fixed 0.80 threshold).
# =============================================================================
def panel_sens_spec_ci():
    conf = _load_parta_confusion()

    records = []
    for c in CANCERS:
        d = conf[c]
        sens, sens_lo, sens_hi = _wilson_ci(d["tp"], d["tp"] + d["fn"])
        spec, spec_lo, spec_hi = _wilson_ci(d["tn"], d["tn"] + d["fp"])
        records.append(dict(cancer=c, sens=sens, sens_lo=sens_lo, sens_hi=sens_hi,
                            spec=spec, spec_lo=spec_lo, spec_hi=spec_hi,
                            n_pos=d["tp"] + d["fn"], n_neg=d["tn"] + d["fp"]))
    df = pd.DataFrame(records).sort_values("sens", ascending=True).reset_index(drop=True)
    n = len(df)

    fig, ax = plt.subplots(figsize=(10, 9.2))
    fig.subplots_adjust(left=0.24, right=0.90, top=0.82, bottom=0.09)

    for i, row in df.iterrows():
        c = row["cancer"]
        col = CANCER_COLOR[c]
        light = tuple(np.array(mc.to_rgba(col))[:3] * 0.55 + 0.45)
        y_sens, y_spec = i + 0.16, i - 0.16

        ax.plot([row["sens_lo"], row["sens_hi"]], [y_sens, y_sens],
                color=light, lw=2.6, zorder=2, solid_capstyle="round")
        ax.plot([row["spec_lo"], row["spec_hi"]], [y_spec, y_spec],
                color=col, lw=2.6, zorder=2, solid_capstyle="round")
        ax.scatter(row["sens"], y_sens, s=55, fc=light, ec=col, lw=1.3, zorder=4)
        ax.scatter(row["spec"], y_spec, s=55, fc=col, ec=col, lw=1.3, zorder=4)

        ax.text(1.015, y_sens, f"{row['sens']:.2f} [{row['sens_lo']:.2f}-{row['sens_hi']:.2f}]  n={int(row['n_pos'])}",
                va="center", ha="left", fontsize=8, color=col)
        ax.text(1.015, y_spec, f"{row['spec']:.2f} [{row['spec_lo']:.2f}-{row['spec_hi']:.2f}]  n={int(row['n_neg'])}",
                va="center", ha="left", fontsize=8, fontweight="bold", color=col)

    ax.set_yticks(range(n))
    ax.set_yticklabels([cancer_label(c) for c in df["cancer"]], fontsize=10.5, fontweight="bold")
    for tick, c in zip(ax.get_yticklabels(), df["cancer"]):
        tick.set_color(CANCER_COLOR[c])

    ax.set_xlim(0, 1.60)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Proportion", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylim(-0.7, n - 0.3)
    ax.axvline(1.0, color="#cccccc", lw=1.0, ls="--", zorder=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", lw=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)

    legend_handles = [
        plt.Line2D([0], [0], color="#555555", lw=2.6, label="Specificity (95% Wilson CI)"),
        plt.Line2D([0], [0], color="#aaaaaa", lw=2.6, label="Sensitivity (95% Wilson CI)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, 0.005), fontsize=10, framealpha=0.92)

    ax.set_title(
        "Cancer-specific panels | sensitivity and specificity with 95% CI\n"
        "Wilson score interval | fixed threshold 0.80 | small comparator groups produce wide intervals\n"
        "e.g. AML/CLL sensitivity = 1.00 is off only 14/13 positive samples",
        fontsize=11.5, fontweight="bold", pad=10,
    )

    for ext in ("pdf", "png"):
        p = OUT / f"fig7f_sens_spec_ci.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


# =============================================================================
# Panel 7g -- PPV / NPV vs. assumed prevalence (Bayes' theorem)
# Sensitivity/specificity are prevalence-independent, but PPV/NPV are not --
# this cohort's case:control ratio (e.g. AML 14/413 = 3.4%) is nowhere near
# real deployment prevalence. Rather than assert one "true" prevalence
# (unsourced), sweep across a plausible range so the reader can read off PPV
# at whatever prior they consider realistic for their use case.
# =============================================================================
def panel_ppv_npv_prevalence():
    """Single bold PPV-vs-prevalence line plot -- kept as a line chart (not
    another bar/forest panel like 7c/7f) for visual variety in the slide 7
    set, but stripped down to survive being shrunk into a mosaic cell:
    one axis (NPV dropped -- it was the less differentiated, less decisive
    half), thick lines, big fonts, one reference line, short 2-line legend."""
    conf = _load_parta_confusion()

    PREV_LO, PREV_HI = 0.001, 0.5
    prevalence = np.logspace(np.log10(PREV_LO), np.log10(PREV_HI), 300)

    fig, ax = plt.subplots(figsize=(9, 8.3))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.83, bottom=0.11)

    ppv_at_5pct = {}
    for c in CANCERS:
        d = conf[c]
        sens = d["tp"] / (d["tp"] + d["fn"])
        spec = d["tn"] / (d["tn"] + d["fp"])
        col = CANCER_COLOR[c]

        ppv = (sens * prevalence) / (sens * prevalence + (1 - spec) * (1 - prevalence))
        ppv_at_5pct[c] = sens * 0.05 / (sens * 0.05 + (1 - spec) * 0.95)

        ax.plot(prevalence, ppv, color=col, lw=3.0, alpha=0.88, zorder=2,
                solid_capstyle="round")

    ax.axvline(0.05, color="#999999", lw=1.3, ls="--", zorder=1)
    ax.text(0.05, 1.045, "5%", ha="center", va="bottom", fontsize=10,
            fontweight="bold", color="#999999")

    ax.set_xscale("log")
    ax.set_xlim(PREV_LO, PREV_HI)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Assumed prevalence (log scale)", fontsize=13, fontweight="bold")
    ax.set_ylabel("PPV  (P[cancer | test+])", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#eeeeee", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks([0.001, 0.01, 0.05, 0.10, 0.30, 0.5])
    ax.set_xticklabels(["0.1%", "1%", "5%", "10%", "30%", "50%"], fontsize=11)
    ax.tick_params(axis="y", labelsize=11)

    # Compact legend: short code + PPV@5% only, sorted descending, 2 columns
    order_desc = sorted(CANCERS, key=lambda c: ppv_at_5pct[c], reverse=True)
    handles = [plt.Line2D([0], [0], color=CANCER_COLOR[c], lw=3.0) for c in order_desc]
    labels  = [f"{display_label(c)}  {ppv_at_5pct[c]:.2f}" for c in order_desc]
    ax.legend(handles, labels, loc="lower right", ncol=2, fontsize=9.5,
              framealpha=0.92, handlelength=1.3, columnspacing=1.0,
              labelspacing=0.4, title="PPV @ 5% prevalence", title_fontsize=9.5)

    ax.set_title(
        "Cancer-specific panels | PPV versus assumed prevalence\n"
        "Bayes' theorem from each panel's own Sens/Spec  *  NOT this cohort's case:control ratio",
        fontsize=12.5, fontweight="bold", loc="left", pad=10,
    )

    for ext in ("pdf", "png"):
        p = OUT / f"fig7g_ppv_prevalence.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


# =============================================================================
# Panel 7h -- Calibration / reliability diagram (replaces 7b)
# Uses the FULL internal test set (cases AND controls), not just true
# positives -- a genuinely different diagnostic from anything else on the
# slide: is a raw score of 0.8 actually trustworthy as "80% likely cancer,"
# bin by bin? Quantile-binned per cancer so small-n cancers (MYEL n_pos=9,
# CLL n_pos=13) still get reasonably populated bins (n~413 total per cancer,
# since this is the full one-vs-rest test cohort, not just that cancer's cases).
# =============================================================================
def panel_calibration():
    """Layout v4: dropped the pooled "Overall" curve entirely -- this panel
    analyzes each cancer's confirmation confidence SEPARATELY (that's the
    point of Part A being single-class panels, not one pooled model), so an
    aggregate summary line added confusion rather than clarity. Back to 12
    per-cancer curves + the diagonal reference only, each with its own
    ECE/Brier in the legend. Line and marker opacity are now independent
    (line 80%, dot 100%) so the point estimates stay crisp even as the
    connecting lines blend together where curves overlap."""
    supp = pd.read_csv(SUPP).set_index("target_class")

    X_MAX = 0.70
    fig, ax = plt.subplots(figsize=(11.0, 7.2))
    fig.subplots_adjust(left=0.12, right=0.985, top=0.97, bottom=0.22)

    ax.plot([0, X_MAX], [0, X_MAX], color="#999999", lw=2.0, ls="--", zorder=1)

    N_BINS = 5
    ece_per_cancer = {}
    brier_per_cancer = {}
    for c in CANCERS:
        n_prot = int(supp.loc[c, "deploy_panel_size"])
        panel_name = f"{c} Top {n_prot}"
        pred_path = PARTA_DIR / c.lower() / "tables" / f"{c.lower()}_test_predictions.csv"
        df = pd.read_csv(pred_path)
        sub = df[(df["panel"] == panel_name) & (df["partition"] == "test")]
        y_true, y_score = sub["y_true"].values, sub["y_score"].values
        col = CANCER_COLOR[c]

        try:
            bins = pd.qcut(y_score, N_BINS, duplicates="drop")
        except ValueError:
            bins = pd.qcut(y_score, 3, duplicates="drop")
        bdf = pd.DataFrame({"bin": bins, "y_true": y_true, "y_score": y_score})
        grp = (bdf.groupby("bin", observed=True)
               .agg(mean_pred=("y_score", "mean"), obs_rate=("y_true", "mean"),
                    n=("y_true", "size"))
               .sort_values("mean_pred"))

        ece = float((grp["n"] / len(y_true) * (grp["mean_pred"] - grp["obs_rate"]).abs()).sum())
        ece_per_cancer[c] = ece
        brier_per_cancer[c] = float(np.mean((y_score - y_true) ** 2))

        # line and dot opacity split: line 80%, dot 100%
        ax.plot(grp["mean_pred"], grp["obs_rate"], color=col, lw=2.0, alpha=0.80, zorder=3)
        ax.scatter(grp["mean_pred"], grp["obs_rate"], color=col, s=42, alpha=1.0,
                  edgecolors="none", zorder=4)

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, X_MAX)
    ax.set_xlabel("Mean predicted score (bin)", fontsize=15.5, fontweight="bold")
    ax.set_ylabel("Observed positive rate (bin)", fontsize=15.5, fontweight="bold")
    ax.tick_params(labelsize=12.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#eeeeee", lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    order_asc = sorted(CANCERS, key=lambda c: ece_per_cancer[c])
    handles = [plt.Line2D([0], [0], color="#999999", lw=2.0, ls="--")] + \
              [plt.Line2D([0], [0], color=CANCER_COLOR[c], lw=2.0, marker="o", ms=6) for c in order_asc]
    labels = ["Ideal"] + [display_label(c) for c in order_asc]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=7, fontsize=11.5, frameon=False, handlelength=1.25,
              handletextpad=0.35, columnspacing=0.75, labelspacing=0.35,
              borderaxespad=0.0)
    # The adjacent DCA panel carries the shared cancer-color key plus its
    # treat-all/none keys; repeating the same 12-item legend here wastes the
    # calibration panel's limited reading area.
    if ax.legend_ is not None:
        ax.legend_.remove()


    for ext in ("pdf", "png"):
        p = OUT / f"fig7h_calibration.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


# =============================================================================
# Panel 7i -- Decision curve analysis (replaces 7g)
# Net benefit = TP/n - FP/n * pt/(1-pt), computed by thresholding each
# cancer's ACTUAL predicted scores at each candidate threshold pt -- unlike
# 7g's Bayes sweep, this uses this cohort's own OBSERVED prevalence and
# OBSERVED scores throughout, not an assumed/unsourced prevalence. Answers a
# concrete clinical question: does using this panel's score beat the naive
# treat-all/treat-none baselines, and over what threshold range?
# =============================================================================
def panel_decision_curve():
    supp = pd.read_csv(SUPP).set_index("target_class")
    pt_grid = np.linspace(0.01, 0.90, 200)

    fig, ax = plt.subplots(figsize=(11.0, 7.2))
    fig.subplots_adjust(left=0.12, right=0.985, top=0.97, bottom=0.22)

    treat_all_curves = []
    nb_at_ref = {}
    for c in CANCERS:
        n_prot = int(supp.loc[c, "deploy_panel_size"])
        panel_name = f"{c} Top {n_prot}"
        pred_path = PARTA_DIR / c.lower() / "tables" / f"{c.lower()}_test_predictions.csv"
        df = pd.read_csv(pred_path)
        sub = df[(df["panel"] == panel_name) & (df["partition"] == "test")]
        y_true, y_score = sub["y_true"].values, sub["y_score"].values
        n = len(y_true)
        prev = float(y_true.mean())
        col = CANCER_COLOR[c]

        nb = np.empty_like(pt_grid)
        for i, pt in enumerate(pt_grid):
            pred_pos = y_score >= pt
            tp = np.sum(pred_pos & (y_true == 1))
            fp = np.sum(pred_pos & (y_true == 0))
            nb[i] = tp / n - fp / n * (pt / (1 - pt))

        ax.plot(pt_grid, nb, color=col, lw=2.4, alpha=0.85, zorder=3)

        treat_all_curves.append(prev - (1 - prev) * (pt_grid / (1 - pt_grid)))
        idx_ref = int(np.argmin(np.abs(pt_grid - 0.50)))
        nb_at_ref[c] = float(nb[idx_ref])

    treat_all_arr = np.array(treat_all_curves)
    ax.fill_between(pt_grid, treat_all_arr.min(axis=0), treat_all_arr.max(axis=0),
                    color="#bbbbbb", alpha=0.30, zorder=1)
    ax.axhline(0, color="#555555", lw=2.0, ls="--", zorder=2)

    ax.set_xlim(0, 0.9)
    ax.set_ylim(-0.05, 0.2)
    ax.set_xlabel("Threshold probability (pt)", fontsize=15.5, fontweight="bold")
    ax.set_ylabel("Net benefit", fontsize=15.5, fontweight="bold")
    ax.tick_params(labelsize=12.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#eeeeee", lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    order_desc = sorted(CANCERS, key=lambda c: nb_at_ref[c], reverse=True)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#bbbbbb", alpha=0.5, ec="none"),
              plt.Line2D([0], [0], color="#555555", lw=2.0, ls="--")] + \
              [plt.Line2D([0], [0], color=CANCER_COLOR[c], lw=2.4) for c in order_desc]
    labels = ["Treat all", "Treat none"] + [display_label(c) for c in order_desc]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=7, fontsize=11.5, frameon=False, handlelength=1.25,
              handletextpad=0.35, columnspacing=0.75, labelspacing=0.35,
              borderaxespad=0.0)
    # The slide composite supplies one universal key below panels A-C.
    if ax.legend_ is not None:
        ax.legend_.remove()


    for ext in ("pdf", "png"):
        p = OUT / f"fig7i_decision_curve.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


# =============================================================================
if __name__ == "__main__":
    print("Building fig7 panels ...")
    print("  7c panel_sens_spec_bars ...")
    panel_sens_spec_bars()
    print("  7d panel_confusion_grid ...")
    panel_confusion_grid()
    print("  7e panel_pr_overlay ...")
    panel_pr_overlay()
    print("  7h panel_calibration ...")
    panel_calibration()
    print("  7i panel_decision_curve ...")
    panel_decision_curve()
    print("\nDone.")
