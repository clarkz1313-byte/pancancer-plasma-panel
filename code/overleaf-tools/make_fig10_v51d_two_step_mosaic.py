"""Manuscript v5.1 Figure 10, version d (versions a-c are kept unchanged).

Same column grammar as Figures 8 and 9 (one column per cancer, figurev6 order and
colours, row letters only, one legend), but no bar or line charts:
  A, E  Plasma / TCGA: one proportional square per cancer (mosaic tile). The square
        is all patients assigned to the cancer by step 1; its width is split into
        confirmed (left) and not confirmed (right) by step 2 (cutoff 0.50), and each
        part into correct (colour) and wrong (dark / grey). Above: % correct when
        confirmed and when not; below: confirmed / assigned.
  C, F  External / TCGA: ROC-space arrows. Each rule is a point at
        (1 - specificity, sensitivity): step 1 (open circle), step 2 alone (open
        square), two-step call (filled); the arrow runs from step 1 to the two-step
        call, i.e. the effect of adding step 2. The x axis is zoomed per row.
  B, D, G  Decision curves (standardised net benefit) with layered shading above the
        better of treat-all and treat-none: faded = benefit step 1 already gives;
        bold = extra benefit from adding step 2; grey hatch = benefit lost by adding
        step 2. Only the two-step curve is drawn as a line.
Inputs: saved outputs of scripts 62, 64 and 70.
Output: overleaf-source-git/figures_300/fig10_v51d_two_step_mosaic.png (new file).
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "revise_plan"
OUT = ROOT / "overleaf-source-git/figures_300/fig10_v51d_two_step_mosaic.png"
sys.path.insert(0, str(ROOT / "figurev6/scripts"))
import fig89_common as FC  # noqa: E402

ORDER = FC.CANCER_ORDER
NAME = {c: ("DLBCL" if c == "LYMPH" else c) for c in ORDER}
COL = FC.PLOT_COLOR
DARK, LIGHTGREY, GREY = "#2b2b2b", "#c9c9c9", "#6b6b6b"
COL_W, LEFT, RIGHT, TOP, BOTTOM = 2.40, 1.75, 0.18, 0.70, 1.45
ROWS = [("square", "plasma", "Plasma", 2.10), ("dca", "plasma", "Plasma", 1.75),
        ("roc", "external", "External", 2.00), ("dca", "external", "External", 1.75),
        ("square", "TCGA", "TCGA", 2.10), ("roc", "TCGA", "TCGA", 2.00), ("dca", "TCGA", "TCGA", 1.75)]
VAR = {"plasma": "step2>=0.50", "TCGA": "step2>=0.50", "external": "step2>=0.80"}
XMAX = {"external": 0.60, "TCGA": 0.05}
XTICKS = {"external": [0, 0.3, 0.6], "TCGA": [0, 0.02, 0.04]}
plt.rcParams.update({"font.family": "DejaVu Sans"})


def frame(ax):
    for sp in ax.spines.values():
        sp.set_color("#666666"); sp.set_linewidth(0.9)


def title(ax, c, show):
    if show:
        ax.set_title(NAME[c], color=COL[c], fontsize=19, fontweight="bold", pad=8)


def square(ax, c, r, show):
    col = COL[c]
    nc, nu = r.correct_confirmed + r.wrong_confirmed, r.correct_unconfirmed + r.wrong_unconfirmed
    n = nc + nu
    wc = nc / n if n else 0
    parts = [(0.0, wc, r.correct_confirmed, r.wrong_confirmed, FC._blend(col, 1.0), DARK),
             (wc, 1 - wc, r.correct_unconfirmed, r.wrong_unconfirmed, FC._blend(col, 0.35), LIGHTGREY)]
    for x0, w, ok, bad, c_ok, c_bad in parts:
        if w <= 0:
            continue
        h_ok = ok / (ok + bad)
        ax.add_patch(Rectangle((x0, 0), w, h_ok, facecolor=c_ok, edgecolor="none"))
        ax.add_patch(Rectangle((x0, h_ok), w, 1 - h_ok, facecolor=c_bad, edgecolor="none"))
    if 0 < wc < 1:
        ax.plot([wc, wc], [0, 1], color="white", lw=2.2)
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#666666", lw=1.0))
    acc_c = f"{100 * r.correct_confirmed / nc:.0f}%" if nc else "–"
    acc_u = f"{100 * r.correct_unconfirmed / nu:.0f}%" if nu else "–"
    ax.text(0.30, 1.05, acc_c, ha="center", va="bottom", fontsize=16, fontweight="bold", color="black")
    ax.text(0.80, 1.05, acc_u, ha="center", va="bottom", fontsize=13, fontweight="bold", color="#8a8a8a")
    ax.text(0.5, -0.06, f"{int(nc)}/{int(n)} confirmed", ha="center", va="top", fontsize=12.5,
            color="#444444", fontweight="bold")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.22, 1.22); ax.set_aspect("equal"); ax.axis("off")
    title(ax, c, show)


def roc(ax, c, g, lev, first, show):
    col = COL[c]
    g = g.set_index("rule")
    pts = {k: (1 - g.loc[k, "specificity"], g.loc[k, "sensitivity"]) for k in ("step1", "step2", "two_step")}
    ax.add_patch(FancyArrowPatch(pts["step1"], pts["two_step"], arrowstyle="-|>", mutation_scale=22,
                                 color=col, lw=2.6, shrinkA=7, shrinkB=9, zorder=3))
    ax.scatter(*pts["step2"], s=70, marker="s", facecolor="white", edgecolor=FC._blend(col, 0.55), lw=2, zorder=4)
    ax.scatter(*pts["step1"], s=90, marker="o", facecolor="white", edgecolor=GREY, lw=2.2, zorder=5)
    ax.scatter(*pts["two_step"], s=120, marker="o", facecolor=col, edgecolor="white", lw=1.2, zorder=6)
    xm = XMAX[lev]
    ax.set_xlim(-0.03 * xm, xm); ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(XTICKS[lev], ["0" if v == 0 else f"{v:g}" for v in XTICKS[lev]]); ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=12.5, length=3, pad=2)
    ax.grid(color="#eeeeee", lw=0.8)
    if first:
        ax.set_ylabel("Sensitivity", fontsize=12.5, fontweight="bold")
        ax.set_xlabel("1 − specificity", fontsize=12, fontweight="bold", labelpad=1)
    else:
        ax.set_yticklabels([])
    frame(ax)
    title(ax, c, show)


def dca(ax, c, g, prev, first):
    col = COL[c]
    get = lambda k: g[g.rule == k].sort_values("pt").net_benefit.to_numpy() / prev
    pt = g[g.rule == "two_step"].sort_values("pt").pt.to_numpy()
    s1, s2, ta = get("step1"), get("two_step"), get("treat_all")
    ref = np.maximum(ta, 0)
    low = np.minimum(s1, s2)
    ax.fill_between(pt, ref, low, where=low > ref, color=col, alpha=0.18, lw=0, interpolate=True)
    lo_gain = np.maximum(s1, ref)
    ax.fill_between(pt, lo_gain, s2, where=s2 > lo_gain, color=col, alpha=0.70, lw=0, interpolate=True)
    lo_loss = np.maximum(s2, ref)
    ax.fill_between(pt, lo_loss, s1, where=s1 > lo_loss, facecolor="none", edgecolor="#8a8a8a",
                    hatch="////", lw=0, interpolate=True)
    ax.plot(pt, ta, color="#999999", lw=1.2, ls=":")
    ax.axhline(0, color="#555555", lw=1.1, ls="--")
    ax.plot(pt, s2, color=col, lw=2.4)
    ax.set_xlim(0.05, 0.95); ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0.2, 0.5, 0.8]); ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=13, length=3, pad=2)
    if first:
        ax.set_ylabel("Standardised\nnet benefit", fontsize=12.5, fontweight="bold")
    else:
        ax.set_yticklabels([])
    frame(ax)


def empty(ax, c, show):
    title(ax, c, show)
    ax.text(0.5, 0.5, "not in TCGA", ha="center", va="center", fontsize=13, color="#9a9a9a",
            style="italic", transform=ax.transAxes)
    ax.axis("off")


def main():
    asg = {"plasma": pd.read_csv(R / "v5_seed52_common_split/internal/assign_confirm/assign_confirm_by_assigned_cancer.csv"),
           "TCGA": pd.read_csv(R / "tcga_two_step/results/tcga_assign_confirm_by_assigned_cancer.csv")}
    asg["TCGA"]["assigned"] = asg["TCGA"].assigned.replace({"DLBCL": "LYMPH"})
    met = pd.read_csv(R / "two_step_binary_rule/binary_rule_metrics.csv")
    cur = pd.read_csv(R / "two_step_binary_rule/binary_rule_decision_curves.csv")
    cmp_ = pd.read_csv(R / "two_step_binary_rule/binary_rule_paired_comparisons.csv")
    for d in (met, cur, cmp_):
        d["cancer"] = d.cancer.replace({"DLBCL": "LYMPH"})

    heights = [h for *_, h in ROWS]
    fig_w = LEFT + 12 * COL_W + RIGHT
    fig_h = TOP + sum(heights) * 1.30 + BOTTOM
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(len(ROWS), 12, left=LEFT / fig_w, right=1 - RIGHT / fig_w,
                          top=1 - TOP / fig_h, bottom=BOTTOM / fig_h,
                          wspace=0.12, hspace=0.45, height_ratios=heights)
    for i, (kind, lev, unit, _) in enumerate(ROWS):
        for j, c in enumerate(ORDER):
            ax = fig.add_subplot(gs[i, j])
            show = i == 0
            if kind == "square":
                d = asg[lev]; d = d[(d.threshold == 0.5) & (d.assigned == c)]
                empty(ax, c, show) if d.empty else square(ax, c, d.iloc[0], show)
            elif kind == "roc":
                g = met[(met.level == lev) & (met.variant == VAR[lev]) & (met.cancer == c)]
                empty(ax, c, show) if g.empty else roc(ax, c, g, lev, j == 0, show)
            else:
                g = cur[(cur.level == lev) & (cur.variant == VAR[lev]) & (cur.cancer == c)]
                if g.empty:
                    empty(ax, c, show); continue
                k = cmp_[(cmp_.level == lev) & (cmp_.variant == VAR[lev]) & (cmp_.cancer == c)].iloc[0]
                dca(ax, c, g, k.cases / k.n, j == 0)
        box = fig.axes[-1].get_position()
        fig.text(0.004, box.y1 + 0.006, "ABCDEFG"[i], fontsize=22, fontweight="bold", va="top")
        fig.text(0.017, (box.y1 + box.y0) / 2, unit, rotation=90, fontsize=15, fontweight="bold",
                 ha="center", va="center")
    ex = "#2e8b57"
    handles = [Patch(facecolor=ex, label="Confirmed: correct"), Patch(facecolor=DARK, label="Confirmed: wrong"),
               Patch(facecolor=FC._blend(ex, 0.35), label="Not confirmed: correct"),
               Patch(facecolor=LIGHTGREY, label="Not confirmed: wrong"),
               Line2D([], [], ls="", marker="o", mfc="white", mec=GREY, mew=2, ms=10, label="Step 1"),
               Line2D([], [], ls="", marker="s", mfc="white", mec=FC._blend(ex, 0.55), mew=2, ms=9, label="Step 2 alone"),
               Line2D([], [], ls="", marker="o", mfc=ex, mec="white", ms=12, label="Two-step call"),
               Line2D([], [], color=ex, lw=2.4, label="Two-step curve"),
               Line2D([], [], color="#999999", lw=1.2, ls=":", label="Treat all"),
               Line2D([], [], color="#555555", lw=1.1, ls="--", label="Treat none"),
               Patch(facecolor=ex, alpha=0.18, label="Benefit of step 1"),
               Patch(facecolor=ex, alpha=0.70, label="Gain from adding step 2"),
               Patch(facecolor="white", edgecolor="#8a8a8a", hatch="////", label="Loss from adding step 2")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=True, fontsize=13,
               bbox_to_anchor=(0.5, 0.003), handlelength=1.8, columnspacing=1.4)
    fig.savefig(OUT, dpi=300, facecolor="white")
    print(OUT, fig.get_size_inches())


if __name__ == "__main__":
    main()
