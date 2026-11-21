"""Manuscript Figure 10, version v5.1.10 (fork of make_fig10_v519_two_step_mosaic.py).

Twelve cancer columns, CLL and myeloma last (absent from TCGA); equal-size tiles; two
cohort groups sharing one legend (in the empty TCGA slots).
  A External, C TCGA  ROC space, x = specificity: step 1 alone (open circle) to step 1 +
                      step 2 (filled), drawn as an L-shaped path: coloured horizontal bar =
                      specificity gained, hatched vertical bar = sensitivity lost.
  B External, D TCGA  change in net benefit from adding step 2 (two-step minus step 1),
                      net true positives per 100 patients, across decision thresholds;
                      colour = gain, hatch = loss; saved patient-bootstrap 95% intervals
                      at 0.50 and 0.80. y axis shared within each cohort.
Inputs: saved outputs of scripts 64 and 70. Output: figures_300/fig10_v5110_two_step_mosaic.png
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "revise_plan"
OUT = ROOT / "overleaf-source-git/figures_300/fig10_v5110_two_step_mosaic.png"
sys.path.insert(0, str(ROOT / "figurev6/scripts"))
import fig89_common as FC  # noqa: E402

ORDER = [c for c in FC.CANCER_ORDER if c not in ("CLL", "MYEL")] + ["CLL", "MYEL"]
NAME = {c: ("DLBCL" if c == "LYMPH" else c) for c in ORDER}
COL = FC.PLOT_COLOR
GREY, INK = "#6b6b6b", "#3a3a3a"
VAR = {"TCGA": "step2>=0.50", "external": "step2>=0.80"}
ROC_X = {"external": ((0.37, 1.025), [0.5, 0.75, 1.0], ["0.5", "0.75", "1"]),
         "TCGA": ((0.9655, 1.0015), [0.97, 0.985, 1.0], ["0.97", "0.985", "1"])}
ROC_Y = {"external": ((0.50, 1.025), [0.5, 0.75, 1.0], ["0.5", "0.75", "1"]),
         "TCGA": ((-0.04, 1.06), [0, 0.5, 1.0], ["0", "0.5", "1"])}
BAR_IN = 0.13
PT_RANGE = (0.05, 0.85)
YLIM = {"external": (-25, 36), "TCGA": (-6, 12)}
YTICKS = {"external": [-20, 0, 20], "TCGA": [-5, 0, 5, 10]}
ROWS = [("roc", "external"), ("dnb", "external"), (None, None), ("roc", "TCGA"), ("dnb", "TCGA")]
HR = [1, 1, 0.16, 1, 1]
TILE_W, TILE_H, LEFT, RIGHT, TOP, BOTTOM = 2.40, 2.05, 2.75, 0.20, 0.85, 0.75
WSP, HSP = 0.12, 0.40
FS = dict(title=26, tick=17, label=17, letter=30, cohort=23, legend=18)
plt.rcParams.update({"font.family": "DejaVu Sans"})


def frame(ax):
    for sp in ax.spines.values():
        sp.set_color("#666666"); sp.set_linewidth(0.9)
    ax.tick_params(labelsize=FS["tick"], length=4, pad=3)


def roc(ax, c, s1, two, lev, first):
    col = COL[c]
    (x0, x1), xt, xl = ROC_X[lev]
    (y0, y1), yt, yl = ROC_Y[lev]
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks(xt, xl); ax.set_yticks(yt, yl)
    ax.grid(color="#eeeeee", lw=0.8)
    hx = BAR_IN / TILE_W * (x1 - x0)
    hy = BAR_IN / TILE_H * (y1 - y0)
    sp1, se1, sp2, se2 = s1.specificity, s1.sensitivity, two.specificity, two.sensitivity
    if sp2 > sp1:
        ax.add_patch(Rectangle((sp1, se1 - hy / 2), sp2 - sp1, hy, facecolor=col, alpha=0.8,
                               edgecolor="none", zorder=3))
    if se2 < se1:
        ax.add_patch(Rectangle((sp2 - hx / 2, se2), hx, se1 - se2 + hy / 2, facecolor="white",
                               edgecolor="#8a8a8a", hatch="////", lw=0, zorder=3))
        ax.add_patch(Rectangle((sp2 - hx / 2, se2), hx, se1 - se2 + hy / 2, facecolor="none",
                               edgecolor=col, lw=1.6, zorder=4))
    ax.scatter(sp1, se1, s=260, marker="o", facecolor="white", edgecolor=GREY, lw=2.8, zorder=6,
               clip_on=False)
    ax.scatter(sp2, se2, s=330, marker="o", facecolor=col, edgecolor="white", lw=1.6, zorder=7,
               clip_on=False)
    frame(ax)
    if first:
        ax.set_ylabel("Sensitivity", fontsize=FS["label"], fontweight="bold")
        ax.set_xlabel("Specificity", fontsize=FS["label"] - 1, fontweight="bold", labelpad=2)
    else:
        ax.set_yticklabels([])

def dnb_curve(g):
    a = g[g.rule == "step1"].sort_values("pt"); b = g[g.rule == "two_step"].sort_values("pt")
    pt = a.pt.to_numpy()
    assert np.allclose(pt, b.pt.to_numpy())
    return pt, 100 * (b.net_benefit.to_numpy() - a.net_benefit.to_numpy())


def dnb(ax, c, g, k, lev, first):
    col = COL[c]
    pt, d = dnb_curve(g)
    m = (pt >= PT_RANGE[0] - 1e-9) & (pt <= PT_RANGE[1] + 1e-9)
    pt, d = pt[m], d[m]
    ax.fill_between(pt, 0, d, where=d > 0, color=col, alpha=0.75, lw=0, interpolate=True)
    ax.fill_between(pt, d, 0, where=d < 0, facecolor="none", edgecolor="#8a8a8a", hatch="////",
                    lw=0, interpolate=True)
    ax.axhline(0, color="#555555", lw=1.2, ls="--")
    ax.plot(pt, d, color=col, lw=2.6)
    for p0 in (0.50, 0.80):
        tag = f"{p0:.2f}"
        y, lo, hi = (100 * k[f"nb_diff_{tag}{s}"] for s in ("", "_lo", "_hi"))
        ax.errorbar(p0, y, yerr=[[y - lo], [hi - y]], fmt="o", color="black", ms=5.5, lw=1.7,
                    capsize=4, capthick=1.7, zorder=7)
    ax.set_xlim(PT_RANGE[0] - 0.02, PT_RANGE[1] + 0.03); ax.set_ylim(*YLIM[lev])
    ax.set_xticks([0.2, 0.5, 0.8]); ax.set_yticks(YTICKS[lev])
    frame(ax)
    if first:
        ax.set_ylabel("Change in net\nbenefit per 100", fontsize=FS["label"], fontweight="bold")
        ax.set_xlabel("Decision threshold", fontsize=FS["label"] - 1, fontweight="bold", labelpad=2)
    else:
        ax.set_yticklabels([])


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
            assert two.tp <= s1.tp and two.fp <= s1.fp and two.tp + two.fn == s1.tp + s1.fn, (lev, c)
            up, down = 100 * k.fp_removed / k.n, 100 * k.tp_lost / k.n
            for v, (lo_, hi_) in [(s1.specificity, ROC_X[lev][0]), (two.specificity, ROC_X[lev][0]),
                                  (s1.sensitivity, ROC_Y[lev][0]), (two.sensitivity, ROC_Y[lev][0])]:
                assert lo_ < v <= 1 + 1e-12 and v < hi_ + 1e-12, (lev, c, v)
            if lev == "TCGA":
                t = tc[tc.assigned == c].iloc[0]
                assert (t.correct_unconfirmed, t.wrong_unconfirmed) == (k.tp_lost, k.fp_removed), c
            pt, d = dnb_curve(cur[(cur.level == lev) & (cur.variant == VAR[lev]) & (cur.cancer == c)])
            for p0 in (0.50, 0.80):
                tag = f"{p0:.2f}"
                i = np.argmin(abs(pt - p0))
                assert np.isclose(d[i], 100 * k[f"nb_diff_{tag}"], atol=1e-6), (lev, c, p0)
                assert k[f"nb_diff_{tag}_lo"] - 1e-9 <= k[f"nb_diff_{tag}"] <= k[f"nb_diff_{tag}_hi"] + 1e-9
                # break-even rule: sign of the difference follows FP removed x odds - TP lost
                s = np.sign(round(k.fp_removed * p0 / (1 - p0) - k.tp_lost, 9))
                assert s == np.sign(round(d[i], 9)), (lev, c, p0)
                assert np.isclose(d[i], up * p0 / (1 - p0) - down, atol=1e-6), (lev, c, p0)
            mm = (pt >= PT_RANGE[0] - 1e-9) & (pt <= PT_RANGE[1] + 1e-9)
            lo, hi = YLIM[lev]
            ci = [100 * k[f"nb_diff_{t}{s}"] for t in ("0.50", "0.80") for s in ("_lo", "_hi")]
            assert lo < min(d[mm].min(), *ci) and max(d[mm].max(), *ci) < hi, (lev, c)
    return met, cur, cmp_


def main():
    met, cur, cmp_ = load()
    fig_w = LEFT + 12 * TILE_W + 11 * WSP * TILE_W + RIGHT
    fig_h = TOP + sum(HR) * TILE_H + (len(HR) - 1) * HSP * TILE_H + BOTTOM
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(len(ROWS), 12, left=LEFT / fig_w, right=1 - RIGHT / fig_w, top=1 - TOP / fig_h,
                          bottom=BOTTOM / fig_h, wspace=WSP, hspace=HSP, height_ratios=HR)
    letters = iter("ABCD")
    first_ax = {}
    for i, (kind, lev) in enumerate(ROWS):
        if kind is None:
            continue
        for j, c in enumerate(ORDER):
            if lev == "TCGA" and c in ("CLL", "MYEL"):
                continue
            ax = fig.add_subplot(gs[i, j])
            first_ax.setdefault(i, ax)
            if i == 0:
                ax.set_title(NAME[c], color=COL[c], fontsize=FS["title"], fontweight="bold", pad=10)
            g = cur if kind == "dnb" else met
            sel = (g.level == lev) & (g.variant == VAR[lev]) & (g.cancer == c)
            k = cmp_[(cmp_.level == lev) & (cmp_.variant == VAR[lev]) & (cmp_.cancer == c)].iloc[0]
            if kind == "roc":
                m = g[sel]
                roc(ax, c, m[m.rule == "step1"].iloc[0], m[m.rule == "two_step"].iloc[0], lev, j == 0)
            else:
                dnb(ax, c, g[sel], k, lev, j == 0)
        box = first_ax[i].get_position()
        fig.text(0.003, box.y1 + 0.006, next(letters), fontsize=FS["letter"], fontweight="bold", va="top")
    for rows, lab in [((0, 1), "External"), ((3, 4), "TCGA")]:
        y1, y0 = first_ax[rows[0]].get_position().y1, first_ax[rows[1]].get_position().y0
        fig.add_artist(Line2D([0.0285, 0.0285], [y0, y1], color="#444444", lw=1.6))
        fig.text(0.023, (y0 + y1) / 2, lab, rotation=90, fontsize=FS["cohort"], fontweight="bold",
                 ha="center", va="center")

    kw = dict(loc="center left", frameon=False, fontsize=FS["legend"], handlelength=2.4,
              labelspacing=0.7, borderaxespad=0.1)
    ax_l = fig.add_subplot(gs[3:5, 10:12]); ax_l.axis("off")
    ci = (Line2D([], [], ls="", marker="|", ms=18, mew=1.8, color="black"),
          Line2D([], [], ls="", marker="o", ms=6, color="black"))
    step1 = (Line2D([], [], ls="", marker="o", mfc="white", mec=GREY, mew=2.6, ms=15),
             Line2D([], [], color="#555555", lw=1.2, ls="--"))
    both = Line2D([], [], ls="", marker="o", mfc=INK, mec="white", ms=16)
    ax_l.legend([step1, both, Patch(facecolor=INK, alpha=0.8),
                 Patch(facecolor="white", edgecolor="#8a8a8a", hatch="////"), ci],
                ["Step 1 alone", "Step 1 + step 2", "Gain from adding step 2", "Loss from adding step 2",
                 "95% interval"],
                handler_map={step1: HandlerTuple(ndivide=None, pad=0.2), ci: HandlerTuple(ndivide=1, pad=0)}, loc="center left", frameon=False,
                fontsize=FS["legend"] + 1, handlelength=2.6, handleheight=1.4, labelspacing=1.2,
                borderaxespad=0.1)
    fig.savefig(OUT, dpi=300, facecolor="white")
    print(OUT, fig.get_size_inches())


if __name__ == "__main__":
    main()
