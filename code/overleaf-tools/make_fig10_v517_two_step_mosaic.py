"""Manuscript Figure 10, version v5.1.7 (fork of make_fig10_v51d_two_step_mosaic.py).

Twelve cancer columns, CLL and myeloma last (absent from TCGA). Four rows of
equal-size tiles, grouped by plot type so each legend sits beside its rows in the
empty TCGA slots:
  A External, B TCGA  ROC space: arrow from step 1 (open) to the two-step call (filled).
  C External, D TCGA  standardised net benefit of the two-step call; faded = benefit
                      of step 1, bold = gain from adding step 2, hatch = loss.
Inputs: saved outputs of scripts 64 and 70. Output: figures_300/fig10_v517_two_step_mosaic.png
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "revise_plan"
OUT = ROOT / "overleaf-source-git/figures_300/fig10_v517_two_step_mosaic.png"
sys.path.insert(0, str(ROOT / "figurev6/scripts"))
import fig89_common as FC  # noqa: E402

ORDER = [c for c in FC.CANCER_ORDER if c not in ("CLL", "MYEL")] + ["CLL", "MYEL"]
NAME = {c: ("DLBCL" if c == "LYMPH" else c) for c in ORDER}
COL = FC.PLOT_COLOR
GREY = "#6b6b6b"
ROWS = [("roc", "external", "External"), ("roc", "TCGA", "TCGA"),
        ("dca", "external", "External"), ("dca", "TCGA", "TCGA")]
VAR = {"TCGA": "step2>=0.50", "external": "step2>=0.80"}
XMAX = {"external": 0.60, "TCGA": 0.05}
XTICKS = {"external": [0, 0.2, 0.4], "TCGA": [0, 0.02, 0.04]}
TILE_W, TILE_H, LEFT, RIGHT, TOP, BOTTOM = 2.40, 2.05, 2.30, 0.20, 0.85, 0.75
WSP, HSP = 0.12, 0.42
FS = dict(title=26, tick=17, label=17, letter=30, cohort=21, legend=17)
plt.rcParams.update({"font.family": "DejaVu Sans"})


def frame(ax):
    for sp in ax.spines.values():
        sp.set_color("#666666"); sp.set_linewidth(0.9)
    ax.tick_params(labelsize=FS["tick"], length=4, pad=3)


def roc(ax, c, g, lev, first):
    col = COL[c]
    g = g.set_index("rule")
    p1, p2 = ((1 - g.loc[k, "specificity"], g.loc[k, "sensitivity"]) for k in ("step1", "two_step"))
    xm = XMAX[lev]
    if np.hypot((p1[0] - p2[0]) / xm, p1[1] - p2[1]) > 0.08:
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=24, color=col, lw=2.8,
                                     shrinkA=8, shrinkB=10, zorder=3))
    ax.scatter(*p1, s=120, marker="o", facecolor="white", edgecolor=GREY, lw=2.4, zorder=5, clip_on=False)
    ax.scatter(*p2, s=160, marker="o", facecolor=col, edgecolor="white", lw=1.3, zorder=6, clip_on=False)
    xm = XMAX[lev]
    ax.set_xlim(-0.07 * xm, xm); ax.set_ylim(-0.06, 1.09)
    ax.set_xticks(XTICKS[lev], ["0" if v == 0 else f"{v:g}" for v in XTICKS[lev]])
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(color="#eeeeee", lw=0.8)
    frame(ax)
    if first:
        ax.set_ylabel("Sensitivity", fontsize=FS["label"], fontweight="bold")
        ax.set_xlabel("1 − specificity", fontsize=FS["label"] - 1, fontweight="bold", labelpad=2)
    else:
        ax.set_yticklabels([])


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
    ax.plot(pt, ta, color="#999999", lw=1.3, ls=":")
    ax.axhline(0, color="#555555", lw=1.2, ls="--")
    ax.plot(pt, s2, color=col, lw=2.6)
    ax.set_xlim(0.05, 0.95); ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0.2, 0.5, 0.8]); ax.set_yticks([0, 0.5, 1.0])
    frame(ax)
    if first:
        ax.set_ylabel("Standardised\nnet benefit", fontsize=FS["label"], fontweight="bold")
        ax.set_xlabel("Decision threshold", fontsize=FS["label"] - 1, fontweight="bold", labelpad=2)
    else:
        ax.set_yticklabels([])


def legend(ax, handles):
    ax.axis("off")
    ax.legend(handles=handles, loc="center left", frameon=False, fontsize=FS["legend"],
              handlelength=2.0, labelspacing=0.55, borderaxespad=0.2)


def load():
    met = pd.read_csv(R / "two_step_binary_rule/binary_rule_metrics.csv")
    cur = pd.read_csv(R / "two_step_binary_rule/binary_rule_decision_curves.csv")
    cmp_ = pd.read_csv(R / "two_step_binary_rule/binary_rule_paired_comparisons.csv")
    tc = pd.read_csv(R / "tcga_two_step/results/tcga_assign_confirm_by_assigned_cancer.csv")
    tc = tc[tc.threshold == 0.5].copy()
    for d in (met, cur, cmp_):
        d["cancer"] = d.cancer.replace({"DLBCL": "LYMPH"})
    tc["assigned"] = tc.assigned.replace({"DLBCL": "LYMPH"})
    cmp_ = cmp_[cmp_.compared_with == "step1"]
    for lev in ("external", "TCGA"):
        for c in ORDER:
            m = met[(met.level == lev) & (met.variant == VAR[lev]) & (met.cancer == c)]
            if m.empty:
                assert lev == "TCGA" and c in ("CLL", "MYEL"), (lev, c)
                continue
            s1, two = (m[m.rule == k].iloc[0] for k in ("step1", "two_step"))
            k = cmp_[(cmp_.level == lev) & (cmp_.variant == VAR[lev]) & (cmp_.cancer == c)].iloc[0]
            assert (s1.tp - two.tp, s1.fp - two.fp) == (k.tp_lost, k.fp_removed), (lev, c)
            assert two.tp <= s1.tp and two.fp <= s1.fp, (lev, c)
            if lev == "TCGA":
                t = tc[tc.assigned == c].iloc[0]
                assert (t.correct_unconfirmed, t.wrong_unconfirmed) == (k.tp_lost, k.fp_removed), c
            assert np.isclose((two.tp + two.fn), (s1.tp + s1.fn)), (lev, c)
    return met, cur, cmp_


def main():
    met, cur, cmp_ = load()
    fig_w = LEFT + 12 * TILE_W + 11 * WSP * TILE_W + RIGHT
    fig_h = TOP + 4 * TILE_H + 3 * HSP * TILE_H + BOTTOM
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(4, 12, left=LEFT / fig_w, right=1 - RIGHT / fig_w, top=1 - TOP / fig_h,
                          bottom=BOTTOM / fig_h, wspace=WSP, hspace=HSP)
    first_ax = {}
    for i, (kind, lev, cohort) in enumerate(ROWS):
        for j, c in enumerate(ORDER):
            if lev == "TCGA" and c in ("CLL", "MYEL"):
                continue
            ax = fig.add_subplot(gs[i, j])
            first_ax.setdefault(i, ax)
            if i == 0:
                ax.set_title(NAME[c], color=COL[c], fontsize=FS["title"], fontweight="bold", pad=10)
            if kind == "roc":
                g = met[(met.level == lev) & (met.variant == VAR[lev]) & (met.cancer == c)]
                roc(ax, c, g, lev, j == 0)
            else:
                g = cur[(cur.level == lev) & (cur.variant == VAR[lev]) & (cur.cancer == c)]
                k = cmp_[(cmp_.level == lev) & (cmp_.variant == VAR[lev]) & (cmp_.cancer == c)].iloc[0]
                dca(ax, c, g, k.cases / k.n, j == 0)
        box = first_ax[i].get_position()
        fig.text(0.003, box.y1 + 0.006, "ABCD"[i], fontsize=FS["letter"], fontweight="bold", va="top")
        fig.text(0.013, (box.y0 + box.y1) / 2, cohort, rotation=90, fontsize=FS["cohort"],
                 fontweight="bold", ha="center", va="center")
    ink = "#3a3a3a"
    legend(fig.add_subplot(gs[1, 10:12]), [
        Line2D([], [], ls="", marker="o", mfc="white", mec=GREY, mew=2.4, ms=12, label="Step 1"),
        Line2D([], [], ls="", marker="o", mfc=ink, mec="white", ms=13, label="Two-step call"),
        Line2D([], [], color=ink, lw=2.8, marker=">", ms=11, label="Adding step 2")])
    legend(fig.add_subplot(gs[3, 10:12]), [
        Line2D([], [], color=ink, lw=2.6, label="Two-step call"),
        Line2D([], [], color="#999999", lw=1.4, ls=":", label="Treat all"),
        Line2D([], [], color="#555555", lw=1.2, ls="--", label="Treat none"),
        Patch(facecolor=ink, alpha=0.18, label="Benefit of step 1"),
        Patch(facecolor=ink, alpha=0.70, label="Gain from step 2"),
        Patch(facecolor="white", edgecolor="#8a8a8a", hatch="////", label="Loss from step 2")])
    fig.savefig(OUT, dpi=300, facecolor="white")
    print(OUT, fig.get_size_inches())


if __name__ == "__main__":
    main()
