#!/usr/bin/env python3
"""
fig5_panels.py - Internal Part B multiclass results (novel visualizations)

Design goal: NON-repetitive vs slide 4 (heatmap), slide 6 (single-class CM/PR),
slide 8 (external box+ROC+CM). Each panel here shows something not shown elsewhere.

fig5a: Confusion Sankey - 413 samples flow from true class to predicted class
fig5b: True-class probability ridgeline - density per cancer, sorted by median
fig5c: Bootstrap forest - macro metrics + per-cancer OvR AUC with 95% CI
fig5d: Score matrix - mean model score_j among true-class-i samples (cross-talk)
fig5e: Decision curve analysis - net benefit vs threshold, one-vs-rest per
       cancer, from OBSERVED test data (same construction as fig7i for the
       Part A confirmation panels, applied here to the Part B multiclass
       screening scores)

Output: figurev5/output/fig5{a,b,c,d,e}_*.pdf/.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
import pandas as pd
from figure_labels import display_label
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score
from publication_style import enable_text_scaling

enable_text_scaling(factor=1.30, minimum=10.5, axis_label_scale=0.75)

ROOT   = Path(__file__).resolve().parents[2]
TABLES = ROOT / "revise_plan/part_b_multiclass/vRSX_v11_locked_reproducer/tables"
OUT    = ROOT / "figurev6/output"
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

def lighter(hexcol, blend=0.52):
    r, g, b = mc.to_rgb(hexcol)
    return (r+(1-r)*blend, g+(1-g)*blend, b+(1-b)*blend)

plt.rcParams.update({
    "font.family":"Arial","font.size":12,"font.weight":"bold",
    "axes.labelweight":"bold","axes.titleweight":"bold",
    "axes.linewidth":1.2,"xtick.major.width":1.2,"ytick.major.width":1.2,
})


# =============================================================================
# Panel 5a - 3-column alluvial: True class → Outcome → Predicted class
# Visually distinct from the 2-column cancer→protein-function Sankey in slide 7
# =============================================================================
def panel_sankey():
    prob = pd.read_csv(TABLES / "locked_test_predictions_with_probabilities.csv")
    cm = pd.crosstab(prob["true_class"], prob["predicted_class"])
    cm = cm.reindex(index=CANCERS, columns=CANCERS, fill_value=0).astype(int)

    total_n  = int(cm.values.sum())
    diag_n   = sum(int(cm.loc[c, c]) for c in CANCERS)   # correct
    misc_n   = total_n - diag_n                           # misclassified

    left_totals  = cm.sum(axis=1)
    right_totals = cm.sum(axis=0)

    GAP     = 4
    BIG_GAP = 22   # gap between CORRECT and MISC nodes in middle column
    N       = len(CANCERS)

    # ── Column positions (Y in raw count units, top-down) ──────────────────────
    def col_positions(totals_dict, keys):
        top_d, bot_d = {}, {}
        y = sum(totals_dict[k] for k in keys) + GAP * (len(keys) - 1)
        total = y
        for k in keys:
            top_d[k] = y
            y -= totals_dict[k]
            bot_d[k] = y
            y -= GAP
        return top_d, bot_d, total

    left_top,  left_bot,  lh = col_positions(left_totals.to_dict(),  CANCERS)
    right_top, right_bot, rh = col_positions(right_totals.to_dict(), CANCERS)

    # Middle: CORRECT on top, MISC on bottom (raw coords before centering)
    mid_h         = diag_n + BIG_GAP + misc_n
    correct_top_r = mid_h
    correct_bot_r = mid_h - diag_n
    misc_top_r    = misc_n
    misc_bot_r    = 0

    total_h = max(lh, mid_h, rh)

    # Center each column vertically
    l_off = (total_h - lh)  // 2
    m_off = (total_h - mid_h) // 2
    r_off = (total_h - rh)  // 2

    left_top  = {k: v + l_off for k, v in left_top.items()}
    left_bot  = {k: v + l_off for k, v in left_bot.items()}
    right_top = {k: v + r_off for k, v in right_top.items()}
    right_bot = {k: v + r_off for k, v in right_bot.items()}
    correct_top = correct_top_r + m_off
    correct_bot = correct_bot_r + m_off
    misc_top    = misc_top_r    + m_off
    misc_bot    = misc_bot_r    + m_off

    # ── Node X positions (normalized 0–1 axis) ─────────────────────────────────
    NW = 0.016
    X_L1, X_L2 = 0.135, 0.135 + NW
    X_M1, X_M2 = 0.475, 0.475 + NW
    X_R1, X_R2 = 0.815, 0.815 + NW

    fig, ax = plt.subplots(figsize=(17, 11))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03)

    # ── Bezier ribbon helper ───────────────────────────────────────────────────
    def ribbon(x1, x2, lt, lb, rt, rb, col, alpha, zo=2):
        dx  = x2 - x1
        cx1 = x1 + dx * 0.40
        cx2 = x1 + dx * 0.60
        verts = [
            (x1, lt), (cx1, lt), (cx2, rt), (x2, rt),
            (x2, rb), (cx2, rb), (cx1, lb), (x1, lb),
            (x1, lt),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]
        ax.add_patch(PathPatch(MplPath(verts, codes),
                               fc=col, alpha=alpha, ec="none", zorder=zo))

    # ── LEFT → MIDDLE ribbons ─────────────────────────────────────────────────
    left_used   = {c: 0 for c in CANCERS}
    corr_entry  = 0   # pointer into CORRECT node from top
    misc_entry  = 0   # pointer into MISC node from top

    # Pass 1: MISC ribbons (draw behind)
    for c in CANCERS:
        n_misc = int(left_totals[c]) - int(cm.loc[c, c])
        if n_misc == 0:
            continue
        # Reserve correct-size slot first so order within left node is: correct on top
        n_c = int(cm.loc[c, c])
        left_used[c] += n_c   # skip correct slot (drawn in pass 2)
        lt = left_top[c] - left_used[c]; lb = lt - n_misc
        left_used[c] += n_misc
        rt = misc_top - misc_entry; rb = rt - n_misc; misc_entry += n_misc
        ribbon(X_L2, X_M1, lt, lb, rt, rb,
               lighter(CANCER_COLOR[c], 0.38), 0.46, zo=2)
        left_used[c] = 0   # reset (will redo in pass 2 correctly)

    # Pass 2: CORRECT ribbons (draw on top)
    left_used   = {c: 0 for c in CANCERS}
    corr_entry  = 0
    for c in CANCERS:
        n_c = int(cm.loc[c, c])
        if n_c > 0:
            lt = left_top[c] - left_used[c]; lb = lt - n_c
            left_used[c] += n_c
            rt = correct_top - corr_entry; rb = rt - n_c; corr_entry += n_c
            ribbon(X_L2, X_M1, lt, lb, rt, rb, CANCER_COLOR[c], 0.75, zo=3)
        left_used[c] += int(left_totals[c]) - int(cm.loc[c, c])  # advance past misc slot

    # ── MIDDLE → RIGHT ribbons ────────────────────────────────────────────────
    # Correct flows go to TOP of each right node; misc flows fill the BOTTOM.
    right_corr_used = {c: 0 for c in CANCERS}
    right_misc_used = {c: 0 for c in CANCERS}
    # Misc starts just below the correct slot in each right node
    right_misc_start = {c: right_top[c] - int(cm.loc[c, c]) for c in CANCERS}
    corr_exit = 0
    misc_exit = 0

    # Pass 1: MISC → Right (draw behind, zo=2)
    for true_c in CANCERS:
        for pred_c in CANCERS:
            if true_c == pred_c:
                continue
            n = int(cm.loc[true_c, pred_c])
            if n == 0:
                continue
            lt = misc_top - misc_exit; lb = lt - n; misc_exit += n
            rt = right_misc_start[pred_c] - right_misc_used[pred_c]; rb = rt - n
            right_misc_used[pred_c] += n
            ribbon(X_M2, X_R1, lt, lb, rt, rb,
                   lighter(CANCER_COLOR[true_c], 0.38), 0.40, zo=2)

    # Pass 2: CORRECT → Right (draw on top, zo=3; same cancer on left & right)
    for c in CANCERS:
        n_c = int(cm.loc[c, c])
        if n_c == 0:
            continue
        lt = correct_top - corr_exit; lb = lt - n_c; corr_exit += n_c
        rt = right_top[c] - right_corr_used[c]; rb = rt - n_c
        right_corr_used[c] += n_c
        ribbon(X_M2, X_R1, lt, lb, rt, rb, CANCER_COLOR[c], 0.75, zo=3)

    # ── Draw nodes ─────────────────────────────────────────────────────────────
    # Left column
    for c in CANCERS:
        col = CANCER_COLOR[c]
        ax.add_patch(Rectangle((X_L1, left_bot[c]), NW,
                                left_top[c] - left_bot[c],
                                fc=col, ec="white", lw=0.5, zorder=5))
        mid = (left_top[c] + left_bot[c]) / 2
        n_t = int(left_totals[c])
        n_c = int(cm.loc[c, c])
        pct = 100 * n_c / n_t if n_t > 0 else 0
        ax.text(X_L1 - 0.010, mid,
                f"{cancer_label(c)}  n={n_t}  ({pct:.0f}%)",
                ha="right", va="center",
                fontsize=10, fontweight="bold", color=col, zorder=6)

    # Middle nodes
    corr_pct = 100 * diag_n / total_n
    misc_pct = 100 * misc_n  / total_n
    ax.add_patch(Rectangle((X_M1, correct_bot), NW,
                            correct_top - correct_bot,
                            fc="#2a9d2a", ec="white", lw=0.7, zorder=5))
    ax.text((X_M1 + X_M2) / 2, (correct_top + correct_bot) / 2,
            f"CORRECT\nn={diag_n}\n({corr_pct:.1f}%)",
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="white",
            rotation=90, zorder=6)

    ax.add_patch(Rectangle((X_M1, misc_bot), NW,
                            misc_top - misc_bot,
                            fc="#cc3333", ec="white", lw=0.7, zorder=5))
    ax.text((X_M1 + X_M2) / 2, (misc_top + misc_bot) / 2,
            f"MISCLAS.\nn={misc_n}\n({misc_pct:.1f}%)",
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="white",
            rotation=90, zorder=6)

    # Right column
    for c in CANCERS:
        col = CANCER_COLOR[c]
        ax.add_patch(Rectangle((X_R1, right_bot[c]), NW,
                                right_top[c] - right_bot[c],
                                fc=col, ec="white", lw=0.5, zorder=5))
        mid = (right_top[c] + right_bot[c]) / 2
        n_p = int(right_totals[c])
        ax.text(X_R2 + 0.010, mid, f"{display_label(c)}  n={n_p}",
                ha="left", va="center",
                fontsize=10, fontweight="bold", color=col, zorder=6)

    # Column headers
    for x, label in [((X_L1 + X_L2) / 2, "TRUE CLASS"),
                     ((X_M1 + X_M2) / 2, "OUTCOME"),
                     ((X_R1 + X_R2) / 2, "PREDICTED")]:
        ax.text(x, total_h + 24, label,
                ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#333333")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-18, total_h + 52)
    ax.axis("off")

    fig.suptitle(
        "Internal held-out test  *  413 samples  *  12 cancer types\n"
        "3-column alluvial: true label → outcome (correct / misclassified) → predicted label\n"
        "Ribbon width = sample count  *  color = true class  *  faded ribbons = misclassifications",
        fontsize=13.5, fontweight="bold", y=0.97,
    )

    for ext in ("pdf", "png"):
        p = OUT / f"fig5a_confusion_sankey.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


# =============================================================================
# Panel 5b - True-class probability ridgeline
# =============================================================================
def panel_ridgeline():
    prob = pd.read_csv(TABLES / "locked_test_predictions_with_probabilities.csv")
    prob["true_score"] = [row[f"score_{row['true_class']}"]
                          for _, row in prob.iterrows()]
    prob["correct"] = prob["true_class"] == prob["predicted_class"]

    medians = prob.groupby("true_class")["true_score"].median()
    order = medians.sort_values(ascending=True).index.tolist()

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.subplots_adjust(left=0.24, right=0.96, top=0.86, bottom=0.09)

    slot_h = 1.0
    x_grid = np.linspace(0, 1, 300)

    for i, c in enumerate(order):
        vals = prob.loc[prob["true_class"] == c, "true_score"].values
        y_base = i * slot_h
        col    = CANCER_COLOR[c]

        if len(vals) >= 3:
            try:
                kde  = gaussian_kde(vals, bw_method=0.28)
                dens = kde(x_grid)
                dens_norm = dens / dens.max() * 0.85
                ax.fill_between(x_grid, y_base, y_base + dens_norm,
                                color=col, alpha=0.42, zorder=2)
                ax.plot(x_grid, y_base + dens_norm,
                        color=col, lw=1.6, zorder=3)
            except np.linalg.LinAlgError:
                pass

        ax.axhline(y_base, color="#cccccc", lw=0.6, zorder=1)

        med = float(np.median(vals))
        ax.plot([med, med], [y_base, y_base + 0.85],
                color="black", lw=1.6, zorder=4)

        # Individual dots on baseline: correct = filled, wrong = hollow
        rng = np.random.default_rng(42 + i)
        y_jit = rng.uniform(-0.10, 0.10, size=len(vals))
        correct_mask = prob.loc[prob["true_class"] == c, "correct"].values
        for v, yy, ok in zip(vals, y_jit, correct_mask):
            fc = col if ok else "white"
            ax.scatter(v, y_base + yy, s=20, fc=fc, ec=col, lw=0.9,
                       zorder=5, alpha=0.85)

        # Median value annotation
        ax.text(1.015, y_base + 0.35, f"{med:.2f}",
                va="center", ha="left", fontsize=9, fontweight="bold", color=col)

    ax.set_yticks([i * slot_h + 0.35 for i in range(len(order))])
    ax.set_yticklabels([cancer_label(c) for c in order],
                       fontsize=11, fontweight="bold")
    for tick, c in zip(ax.get_yticklabels(), order):
        tick.set_color(CANCER_COLOR[c])

    ax.set_xlim(-0.02, 1.10)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"],
                       fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted probability assigned to true class",
                  fontsize=13, fontweight="bold", labelpad=8)

    ax.axvline(0.5, color="#888888", lw=1.4, ls="--", zorder=1)
    ax.set_ylim(-0.4, len(order) * slot_h + 0.4)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.set_title(
        "Per-cancer distribution of predicted probability for the true class\n"
        "Internal held-out test  *  413 samples  *  sorted by median probability\n"
        "solid dot = correct  *  hollow dot = misclassified  *  vertical black bar = median",
        fontsize=12.5, fontweight="bold", pad=10,
    )

    for ext in ("pdf", "png"):
        p = OUT / f"fig5b_prob_ridgeline.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


# =============================================================================
# Panel 5c - Bootstrap forest (macro + per-cancer OvR AUC)
# =============================================================================
def _forest_data():
    """Shared data prep for the radar and the per-cancer violin strip.

    Split out 2026-08-31 so the two can be emitted as SEPARATE panels. In the
    combined portrait figure the radar was allotted 0.270 of the figure
    height against the violin's 0.465 — the reverse of their importance —
    which is why the radar's metric text was unreadable at slide scale. The
    violin strip is also 12 categories side by side: it wants to be WIDE and
    short, not crammed into a narrow portrait column.
    """
    prob = pd.read_csv(TABLES / "locked_test_predictions_with_probabilities.csv")
    ci   = pd.read_csv(TABLES / "bootstrap_ci_summary.csv")

    y_true_arr = prob["true_class"].values
    scores_dict = {c: prob[f"score_{c}"].values for c in CANCERS}
    n = len(prob)

    # Compute per-cancer OvR AUC bootstrap on the fly
    rng = np.random.default_rng(52)
    n_boot = 1000
    per_cancer_boots = {c: [] for c in CANCERS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt  = y_true_arr[idx]
        for c in CANCERS:
            y_bin = (yt == c).astype(int)
            if y_bin.sum() in (0, n):
                continue
            try:
                per_cancer_boots[c].append(roc_auc_score(y_bin, scores_dict[c][idx]))
            except Exception:
                pass

    per_cancer_ci = {}
    for c in CANCERS:
        y_bin = (prob["true_class"] == c).astype(int).values
        pt    = roc_auc_score(y_bin, scores_dict[c])
        arr   = np.array(per_cancer_boots[c])
        per_cancer_ci[c] = (pt, np.percentile(arr, 2.5), np.percentile(arr, 97.5))

    macro_labels = [
        ("Accuracy",           "accuracy"),
        ("Balanced\naccuracy", "balanced_accuracy"),
        ("Macro F1",           "macro_f1"),
        ("Macro\nOvR AUC",     "macro_ovr_auc"),
        ("Min class\nrecall",  "min_class_recall"),
    ]
    macro_data = []
    for label, key in macro_labels:
        row = ci[ci["metric"] == key].iloc[0]
        macro_data.append((label,
                           float(row["point_estimate"]),
                           float(row["ci_lower_2_5"]),
                           float(row["ci_upper_97_5"])))

    # Keep the same canonical cancer order as the score matrix. This makes
    # the invisible columns of panels B and D correspond one-to-one.
    cancer_sorted = list(CANCERS)
    return macro_data, cancer_sorted, per_cancer_boots, per_cancer_ci


def _radar_ax(ax_r, macro_data, scale=1.0, show_title=True):
    """Radar of the 5 macro metrics. `scale` multiplies every font size —
    1.0 reproduces the old cramped combined figure, ~1.5 is the standalone
    panel where the CI text is actually readable at slide scale.

    show_title=False for the standalone panel: there the axes title and the
    figure suptitle occupy the same band and overprint each other, so the
    standalone folds both into one suptitle."""
    n_met = len(macro_data)
    angles = np.linspace(0, 2 * np.pi, n_met, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    pts = [m[1] for m in macro_data] + [macro_data[0][1]]
    los = [m[2] for m in macro_data] + [macro_data[0][2]]
    his = [m[3] for m in macro_data] + [macro_data[0][3]]

    # Keep the complete metric scale: min-class recall and its CI are defined
    # against a 0–1.0 performance range, so the radar center must remain 0.
    R_LO, R_HI = 0.00, 1.00
    ax_r.set_theta_offset(np.pi / 2)
    ax_r.set_theta_direction(-1)
    ax_r.set_ylim(R_LO, R_HI)
    ax_r.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], labels=["", "", "", "", ""])
    # Put all radial values on the Accuracy spoke (the top vertical line),
    # rather than on the right-hand spoke where they collided with the
    # Balanced accuracy annotation.
    for radius in (0.2, 0.4, 0.6, 0.8, 1.0):
        ax_r.text(angles[0], radius, f"{radius:.1f}",
                  fontsize=7 * scale, fontweight="bold", color="#555555",
                  ha="left", va="center", zorder=20,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=0.9))
    ax_r.set_xticks(angles)
    # Keep every reported number attached to its metric name. Floating
    # point-estimate/CI annotations around the polygon were ambiguous and
    # repeatedly collided on the lateral spokes.
    ax_r.set_xticklabels([])
    # Two-color stacked labels: metric name in black, numerical result in
    # purple. AnnotationBbox lets the two text styles remain one aligned
    # block, which ordinary polar tick labels cannot do.
    for ang, (label, pt, lo, hi) in zip(angles, macro_data):
        metric = TextArea(
            label,
            textprops=dict(color="#111111", fontsize=8.2 * scale,
                           fontweight="bold", ha="center",
                           multialignment="center"),
        )
        result = TextArea(
            f"{pt:.3f}",
            textprops=dict(color="#4a2d75", fontsize=9.2 * scale,
                           fontweight="bold", ha="center",
                           multialignment="center"),
        )
        ci = TextArea(
            f"[{lo:.2f}–{hi:.2f}]",
            textprops=dict(color="#9b8bae", fontsize=6.3 * scale,
                           fontweight="normal", ha="center",
                           multialignment="center"),
        )
        block = VPacker(children=[metric, result, ci], align="center", pad=0, sep=0.7)
        # Keep the metric/result/CI blocks close to the radar rim so the
        # standalone panel does not acquire a large blank outer margin.
        outward = (np.sin(ang) * 22 * scale, np.cos(ang) * 22 * scale)
        ax_r.add_artist(AnnotationBbox(
            block, (ang, R_HI), xybox=outward,
            xycoords="data", boxcoords="offset points",
            box_alignment=(0.5, 0.5), frameon=False, pad=0,
            annotation_clip=False,
        ))
    # pad was 20*scale, which at the standalone panel's scale pushed the spoke
    # labels ~31pt out and (with bbox_inches="tight") shrank the plotted circle
    # inside the saved image. 12 keeps the labels clear of the ring while
    # letting the circle itself occupy far more of the panel.
    ax_r.tick_params(axis="x", pad=0)
    ax_r.spines["polar"].set_visible(False)
    # Use a stronger grid for thumbnail readability, matching the explicit
    # uncertainty treatment used in Figure 7A.
    ax_r.grid(color="#b8b8b8", lw=1.25, alpha=0.95)

    # Filled point-estimate polygon: the color carries the overall profile,
    # while CI whiskers remain independently readable at every metric.
    ax_r.fill(angles_closed, pts, color="#9467bd", alpha=0.28, zorder=2)
    ax_r.plot(angles_closed, pts, color="#9467bd", lw=2.8 * scale, zorder=5,
              solid_capstyle="round")

    # Radial CI whisker with perpendicular end-caps for each metric.  Draw
    # these above the fill but below the estimate markers, as in Figure 7A.
    cap = 0.045
    for ang, lo, hi in zip(angles, los, his):
        lo_draw = max(R_LO, float(lo))
        hi_draw = min(R_HI, float(hi))
        ax_r.plot([ang, ang], [lo_draw, hi_draw], color="#222222",
                  lw=1.65 * scale, alpha=0.95, zorder=6,
                  solid_capstyle="butt")
        for radius in (lo_draw, hi_draw):
            ax_r.plot([ang - cap, ang + cap], [radius, radius],
                      color="#222222", lw=1.45 * scale, alpha=0.95,
                      zorder=6, solid_capstyle="butt")

    ax_r.scatter(angles, pts[:-1], s=60 * scale ** 2, fc="#9467bd", ec="white",
                 lw=1.5 * scale, zorder=7)

    if show_title:
        ax_r.set_title("Macro performance metrics  *  point estimate + bootstrap 95% CI",
                       fontsize=11.5 * scale, fontweight="bold", pad=14 * scale)


def _violin_ax(ax_v, cancer_sorted, per_cancer_boots, per_cancer_ci, scale=1.0,
               show_ylabel=True):
    """Per-cancer OvR AUC bootstrap KDE strip, 12 categories left to right."""
    HW   = 0.38   # violin half-width in cancer-index units
    KBND = 0.35
    DEGEN_THR = 0.003   # catches AML/CLL (std=0) and MYEL (std=0.00049)
    Y_LO, Y_HI = 0.60, 1.08   # reserve a label band above 1.0; violins remain unobscured

    for xi, c in enumerate(cancer_sorted):
        arr = np.array(per_cancer_boots[c])
        col = CANCER_COLOR[c]
        pt, ci_lo, ci_hi = per_cancer_ci[c]
        std = float(arr.std())

        if std < DEGEN_THR:
            bar_hw = max(std * 3, 0.002)
            ax_v.fill_between(
                [xi - HW * 0.55, xi + HW * 0.55],
                [pt - bar_hw, pt - bar_hw],
                [pt + bar_hw, pt + bar_hw],
                color=col, alpha=0.72, zorder=2,
            )
        else:
            try:
                kde  = gaussian_kde(arr, bw_method=KBND)
                bw   = kde.factor * std
                # Local y-range around this cancer's own data, NOT the shared
                # axis range -- prevents the KDE outline from trailing off as
                # a near-invisible-density line all the way to the axis edges
                y_local = np.linspace(max(Y_LO, arr.min() - 4 * bw),
                                      min(Y_HI, arr.max() + 4 * bw), 300)
                dens = kde(y_local)
                dens_norm = dens / dens.max() * HW
                ax_v.fill_betweenx(y_local, xi - dens_norm, xi + dens_norm,
                                    color=col, alpha=0.60, zorder=2)
                ax_v.plot(xi + dens_norm, y_local, color=col, lw=0.9, alpha=0.50, zorder=2)
                ax_v.plot(xi - dens_norm, y_local, color=col, lw=0.9, alpha=0.50, zorder=2)
            except np.linalg.LinAlgError:
                pass

            p5, p95 = np.percentile(arr, 5), np.percentile(arr, 95)
            outliers = arr[(arr < p5) | (arr > p95)]
            if len(outliers):
                rng_jit = np.random.default_rng(xi + 99)
                jit = rng_jit.uniform(-0.12, 0.12, size=len(outliers))
                ax_v.scatter(xi + jit, outliers,
                             s=9, fc=col, ec="none", alpha=0.38, zorder=3)

        # 95% CI whisker (vertical)
        ax_v.plot([xi, xi], [ci_lo, ci_hi],
                  color="black", lw=1.8, zorder=4, solid_capstyle="round")
        for ye in (ci_lo, ci_hi):
            ax_v.plot([xi - 0.12, xi + 0.12], [ye, ye],
                      color="black", lw=1.6, zorder=4)

        ax_v.scatter(xi, pt, s=60, fc="white", ec=col, lw=2.2, zorder=5)

        ax_v.text(xi, 1.045, f"{pt:.3f}",
                  ha="center", va="center",
                  fontsize=10.5 * scale, fontweight="bold", color=col, zorder=7,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.6))

    ax_v.set_xticks(range(len(cancer_sorted)))
    ax_v.set_xticklabels([display_label(c) for c in cancer_sorted],
                         fontsize=10.5 * scale, fontweight="bold",
                         rotation=45, ha="right", rotation_mode="anchor")
    for tick, c in zip(ax_v.get_xticklabels(), cancer_sorted):
        tick.set_color(CANCER_COLOR[c])
    ax_v.set_xlim(-0.6, len(cancer_sorted) - 0.4)
    ax_v.set_ylim(Y_LO, Y_HI)
    ax_v.grid(axis="y", color="#eeeeee", lw=0.8, zorder=0)
    ax_v.set_axisbelow(True)
    ax_v.spines[["top", "right"]].set_visible(False)
    ax_v.tick_params(axis="x", length=0)
    ax_v.tick_params(axis="y", labelsize=12 * scale)
    if show_ylabel:
        ax_v.set_ylabel("OvR AUC  *  bootstrap 95% CI",
                        fontsize=12 * scale, fontweight="bold", labelpad=8)


def panel_forest():
    """Combined portrait figure — SUPERSEDED for deck use 2026-08-31.

    Kept because it is referenced by older docs and by the archived 08-29 /
    08-30 packages. The live slide 5 now uses the two split panels below;
    see `panel_macro_radar` and `panel_percancer_violin`.
    """
    macro_data, cancer_sorted, per_cancer_boots, per_cancer_ci = _forest_data()

    fig = plt.figure(figsize=(7.5, 14.2))
    ax_r = fig.add_axes([0.13, 0.585, 0.74, 0.270], polar=True)
    ax_v = fig.add_axes([0.11, 0.062, 0.85, 0.465])
    _radar_ax(ax_r, macro_data)
    _violin_ax(ax_v, cancer_sorted, per_cancer_boots, per_cancer_ci)

    fig.suptitle(
        "25-protein multiclass panel | performance uncertainty\n"
        "1 000 bootstrap resamples  *  413 held-out test samples",
        fontsize=13.5, fontweight="bold", y=0.995,
    )
    for ext in ("pdf", "png"):
        p = OUT / f"fig5c_bootstrap_forest.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


def panel_macro_radar():
    """The radar ALONE, square and large — slide 5's headline panel.

    In the combined figure this got 0.270 of a 14.2in height (~3.8in) and
    its CI labels rendered at 7.2pt, which is why they could not be read at
    slide scale. Standalone it is ~8in square with every font 1.55x, so the
    five metric values are legible without zooming.
    """
    macro_data, *_ = _forest_data()
    # 2026-08-31 pass 2: scale 1.55 -> 2.05 and the canvas tightened from
    # 8.6x8.8. Font sizes are absolute points, so shrinking the canvas while
    # raising the multiplier makes the text bigger RELATIVE to the panel —
    # which is what governs legibility once the mosaic scales the image down
    # into its cell. The plot box also grew (0.78 -> 0.84 of the height).
    # A and B use identical 11.5 x 9.0 canvases. Their outer panel boxes now
    # align exactly in the 2 x 2 publication mosaic.
    fig = plt.figure(figsize=(11.5, 10.5))
    # Fill more of the standalone canvas; tight export below removes the
    # remaining outer white border before the mosaic is assembled.
    ax_r = fig.add_axes([0.10, 0.09, 0.80, 0.80], polar=True)
    _radar_ax(ax_r, macro_data, scale=1.85, show_title=False)
    # Kept SHORT and narrow on purpose: in the slide mosaic the panel letter is
    # drawn at this image's top-left corner, and a suptitle wide enough to reach
    # the left edge collides with it. The cohort details this used to repeat
    # (413 samples, 1 000 resamples) are already in the slide title.
    for ext in ("pdf", "png"):
        p = OUT / f"fig5f_macro_radar.{ext}"
        fig.savefig(p, dpi=DPI, facecolor="white", bbox_inches="tight",
                    pad_inches=0.04)
        print(f"  saved -> {p}")
    plt.close(fig)


def panel_percancer_violin():
    """The per-cancer strip ALONE, WIDE — 12 categories side by side.

    Portrait-cramped in the combined figure (12 violins across ~6.4in). At
    16in wide each cancer gets ~1.3in, so the KDE shapes and their CI
    whiskers separate cleanly.
    """
    _, cancer_sorted, per_cancer_boots, per_cancer_ci = _forest_data()
    fig = plt.figure(figsize=(11.5, 6.2))
    # B/D use the same fractional x bounds, so each cancer occupies the same
    # invisible column in the matrix above and violin strip below.
    ax_v = fig.add_axes([0.14, 0.22, 0.73, 0.72])
    _violin_ax(ax_v, cancer_sorted, per_cancer_boots, per_cancer_ci,
               scale=1.30, show_ylabel=False)
    for ext in ("pdf", "png"):
        p = OUT / f"fig5g_percancer_violin.{ext}"
        fig.savefig(p, dpi=DPI, facecolor="white")
        print(f"  saved -> {p}")
    plt.close(fig)


# =============================================================================
# Panel 5d - Score matrix (mean score_j among true-class-i samples)
# =============================================================================
def panel_score_matrix():
    prob = pd.read_csv(TABLES / "locked_test_predictions_with_probabilities.csv")
    score_cols = [f"score_{c}" for c in CANCERS]
    means = prob.groupby("true_class")[score_cols].mean().reindex(CANCERS)
    means.columns = CANCERS

    # 2026-08-31: canvas tightened 11x9.5 -> 9.8x9.0 and every font raised, so
    # this panel and the decision curve beside it end up at matching displayed
    # size in the slide-5 mosaic AND their text survives that downscale. The
    # cell values were 9-10pt on an 11in canvas; they are now 13pt on a 9.8in
    # one, which is roughly a 1.5x gain in apparent size on the slide.
    fig, ax = plt.subplots(figsize=(11.5, 10.5))
    fig.subplots_adjust(left=0.14, right=0.87, top=0.87, bottom=0.17)

    cmap = mc.LinearSegmentedColormap.from_list(
        "score_pur",
        ["#ffffff", "#e5d5f0", "#a888c2", "#5a2985", "#2a0d47"],
        N=256,
    )
    im = ax.imshow(means.values, cmap=cmap, vmin=0, vmax=1,
                   aspect="equal", interpolation="nearest")

    for i, true_c in enumerate(CANCERS):
        for j, pred_c in enumerate(CANCERS):
            val = float(means.iloc[i, j])
            tc = "white" if val > 0.35 else "#333333"
            fs = 14 if i == j else 12.5
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=fs, fontweight="bold", color=tc)

    # Diagonal outline (red) to visually highlight the "should be high" cells
    for i in range(len(CANCERS)):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1,
                                fc="none", ec="#e05555", lw=2.0, zorder=4))

    ax.set_xticks(range(len(CANCERS)))
    ax.set_xticklabels([display_label(c) for c in CANCERS], rotation=45, ha="right",
                       fontsize=14, fontweight="bold")
    for tick, c in zip(ax.get_xticklabels(), CANCERS):
        tick.set_color(CANCER_COLOR[c])
    ax.set_yticks(range(len(CANCERS)))
    ax.set_yticklabels([display_label(c) for c in CANCERS],
                       fontsize=14, fontweight="bold")
    for tick, c in zip(ax.get_yticklabels(), CANCERS):
        tick.set_color(CANCER_COLOR[c])

    # Implementation indices i/j add no information once cancer labels are
    # printed on both axes, so omit them from the publication panel.
    ax.set_xlabel("")
    ax.set_ylabel("")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.12)
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label("Mean score",
                 fontsize=12, fontweight="bold", labelpad=6)
    cb.ax.tick_params(labelsize=11)

    # Title shortened 2026-08-31. The old second line was wide enough that the
    # centred title reached the image's left edge, where the mosaic draws the
    # panel letter — the "b" printed on top of the word "Diagonal".

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    for ext in ("pdf", "png"):
        p = OUT / f"fig5d_score_matrix.{ext}"
        fig.savefig(p, dpi=DPI, facecolor="white")
        print(f"  saved -> {p}")
    plt.close(fig)


# =============================================================================
# Panel 5e - Decision curve analysis (multiclass one-vs-rest)
# Same construction as fig7i (Part A confirmation panels): net benefit =
# TP/n - FP/n * pt/(1-pt), computed by thresholding each cancer's ACTUAL
# one-vs-rest score_c at each candidate threshold pt, using this test
# cohort's own observed class prevalence -- not an assumed one. Answers the
# Part B screening equivalent of fig7i's question: does using this class's
# score beat the naive treat-all/treat-none baselines, and over what
# threshold range?
# =============================================================================
def panel_decision_curve():
    prob = pd.read_csv(TABLES / "locked_test_predictions_with_probabilities.csv")
    pt_grid = np.linspace(0.01, 0.90, 200)

    # 2026-08-31: 9.5x8.6 -> 8.7x9.0, chosen so this panel's SAVED aspect
    # ratio lands next to the score matrix's, which is what makes the two
    # render at matching size in the slide-5 mosaic (both cells are the same
    # width, so the taller-aspect image was previously the smaller one).
    # C and D use identical 11.5 x 6.2 canvases and identical plot bounds.
    # The curve legend is moved to the shared composite key, allowing the
    # DCA itself to occupy the full panel width.
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    fig.subplots_adjust(left=0.14, right=0.87, top=0.94, bottom=0.22)

    treat_all_curves = []
    nb_curves = []
    nb_at_ref = {}
    for c in CANCERS:
        y_true = (prob["true_class"] == c).astype(int).values
        y_score = prob[f"score_{c}"].values
        n = len(y_true)
        prev = float(y_true.mean())
        col = CANCER_COLOR[c]

        nb = np.empty_like(pt_grid)
        for i, pt in enumerate(pt_grid):
            pred_pos = y_score >= pt
            tp = np.sum(pred_pos & (y_true == 1))
            fp = np.sum(pred_pos & (y_true == 0))
            nb[i] = tp / n - fp / n * (pt / (1 - pt))

        # individual cancers -- bolder than the earlier "heavily faded
        # background" treatment (0.22->0.50, 1.3->1.8) per feedback
        ax.plot(pt_grid, nb, color=col, lw=2.8, alpha=0.75, zorder=3)
        nb_curves.append(nb)

        treat_all_curves.append(prev - (1 - prev) * (pt_grid / (1 - pt_grid)))
        idx_ref = int(np.argmin(np.abs(pt_grid - 0.5)))
        nb_at_ref[c] = float(nb[idx_ref])

    treat_all_arr = np.array(treat_all_curves)
    ax.fill_between(pt_grid, treat_all_arr.min(axis=0), treat_all_arr.max(axis=0),
                    color="#bbbbbb", alpha=0.30, zorder=1)
    ax.axhline(0, color="#555555", lw=2.7, ls="--", zorder=2)

    # Overall = macro-average net benefit across the 12 one-vs-rest curves --
    # purple, matching the established "macro/aggregate" color used
    # elsewhere in this deck (fig5c radar, fig7e macro AUPRC). No in-plot
    # arrow/label (removed per feedback -- legend is enough); line thinned
    # (4.4->3.0) now that the 12 components are bolder and no longer need
    # such a strong contrast to read correctly.
    MACRO_COLOR = "#A06CD5"
    nb_macro = np.mean(np.array(nb_curves), axis=0)
    ax.plot(pt_grid, nb_macro, color=MACRO_COLOR, lw=4.6, solid_capstyle="round", zorder=5)

    ax.set_xlim(0, 0.90)
    ax.set_ylim(-0.05, 0.20)
    ax.set_xlabel("Threshold probability (pt)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Net benefit", fontsize=18, fontweight="bold")
    ax.tick_params(labelsize=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#eeeeee", lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    order_desc = sorted(CANCERS, key=lambda c: nb_at_ref[c], reverse=True)
    handles = [plt.Line2D([0], [0], color=MACRO_COLOR, lw=3.0),
              plt.Rectangle((0, 0), 1, 1, fc="#bbbbbb", alpha=0.5, ec="none"),
              plt.Line2D([0], [0], color="#555555", lw=2.0, ls="--")] + \
              [plt.Line2D([0], [0], color=CANCER_COLOR[c], lw=3.0, alpha=0.72) for c in order_desc]
    labels = ["Macro", "Treat all", "Treat none"] + [display_label(c) for c in order_desc]
    # One legend is drawn below the complete Figure 5 mosaic.  A local key
    # would shrink panel C and repeat the same cancer-color mapping used by D.


    for ext in ("pdf", "png"):
        p = OUT / f"fig5e_decision_curve.{ext}"
        fig.savefig(p, dpi=DPI, facecolor="white")
        print(f"  saved -> {p}")
    plt.close(fig)


# =============================================================================
if __name__ == "__main__":
    print("=== fig5 novel panels ===")
    print("Panel 5a - Confusion Sankey ...")
    panel_sankey()
    print("Panel 5b - Probability ridgeline ...")
    panel_ridgeline()
    print("Panel 5c - Bootstrap forest (combined, superseded for deck) ...")
    panel_forest()
    print("Panel 5d - Score matrix ...")
    panel_score_matrix()
    print("Panel 5e - Decision curve analysis ...")
    panel_decision_curve()
    print("Panel 5f - Macro radar, standalone (LIVE slide 5) ...")
    panel_macro_radar()
    print("Panel 5g - Per-cancer violin, standalone wide (LIVE slide 5) ...")
    panel_percancer_violin()
    print("All done.")
