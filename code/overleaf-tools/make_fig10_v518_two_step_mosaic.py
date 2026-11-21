"""Manuscript Figure 10, version v5.1.8 (fork of make_fig10_v517_two_step_mosaic.py).

Twelve cancer columns, CLL and myeloma last (absent from TCGA); equal-size tiles.
External group (legend strip below it):
  A  ROC space, x = specificity: 25-protein model alone (open) -> both models positive (filled).
  B  standardised net benefit of 'both positive'; faded = benefit of the 25-protein model
     alone, bold = extra benefit from requiring both, hatch = benefit lost.
TCGA group (legends in the empty CLL / myeloma slots):
  C  assigned patients (% of those assigned to the cancer), confirmed vs not confirmed,
     stacked correct / wrong; % correct above each bar.
  D  ROC space as A: multiclass assignment alone -> assignment confirmed.
  E  net benefit as B.
Inputs: saved outputs of scripts 64 and 70. Output: figures_300/fig10_v518_two_step_mosaic.png
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "revise_plan"
OUT = ROOT / "overleaf-source-git/figures_300/fig10_v518_two_step_mosaic.png"
sys.path.insert(0, str(ROOT / "figurev6/scripts"))
import fig89_common as FC  # noqa: E402

ORDER = [c for c in FC.CANCER_ORDER if c not in ("CLL", "MYEL")] + ["CLL", "MYEL"]
NAME = {c: ("DLBCL" if c == "LYMPH" else c) for c in ORDER}
COL = FC.PLOT_COLOR
GREY, DARK, INK = "#6b6b6b", "#2b2b2b", "#3a3a3a"
VAR = {"TCGA": "step2>=0.50", "external": "step2>=0.80"}
XLIM = {"external": (0.37, 1.025), "TCGA": (0.9655, 1.0015)}
XTICKS = {"external": ([0.5, 0.75, 1.0], ["0.5", "0.75", "1"]),
          "TCGA": ([0.97, 0.985, 1.0], ["0.97", "0.985", "1"])}
# grid rows: A, B, external legend strip, C, D, E
ROWS = [("roc", "external"), ("dca", "external"), ("legend", None),
        ("bar", "TCGA"), ("roc", "TCGA"), ("dca", "TCGA")]
HR = [1, 1, 0.42, 1, 1, 1]
TILE_W, TILE_H, LEFT, RIGHT, TOP, BOTTOM = 2.40, 2.05, 2.75, 0.20, 0.85, 0.75
WSP, HSP = 0.12, 0.40
FS = dict(title=26, tick=17, label=17, letter=30, cohort=23, legend=17, slot=15.5, pct=15)
plt.rcParams.update({"font.family": "DejaVu Sans"})


def frame(ax):
    for sp in ax.spines.values():
        sp.set_color("#666666"); sp.set_linewidth(0.9)
    ax.tick_params(labelsize=FS["tick"], length=4, pad=3)


def roc(ax, c, g, lev, first):
    col = COL[c]
    g = g.set_index("rule")
    p1, p2 = ((g.loc[k, "specificity"], g.loc[k, "sensitivity"]) for k in ("step1", "two_step"))
    x0, x1 = XLIM[lev]
    if np.hypot((p1[0] - p2[0]) / (x1 - x0), p1[1] - p2[1]) > 0.08:
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=24, color=col, lw=2.8,
                                     shrinkA=8, shrinkB=10, zorder=3))
    ax.scatter(*p1, s=120, marker="o", facecolor="white", edgecolor=GREY, lw=2.4, zorder=5, clip_on=False)
    ax.scatter(*p2, s=160, marker="o", facecolor=col, edgecolor="white", lw=1.3, zorder=6, clip_on=False)
    ax.set_xlim(x0, x1); ax.set_ylim(-0.06, 1.09)
    ax.set_xticks(*XTICKS[lev]); ax.set_yticks([0, 0.5, 1.0])
    ax.grid(color="#eeeeee", lw=0.8)
    frame(ax)
    if first:
        ax.set_ylabel("Sensitivity", fontsize=FS["label"], fontweight="bold")
        ax.set_xlabel("Specificity", fontsize=FS["label"] - 1, fontweight="bold", labelpad=2)
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


def bars(ax, c, t, first):
    col = COL[c]
    n = t.n_assigned
    for x, ok, bad in [(0, t.correct_confirmed, t.wrong_confirmed),
                       (1, t.correct_unconfirmed, t.wrong_unconfirmed)]:
        h_ok, h_bad = 100 * ok / n, 100 * bad / n
        ax.bar(x, h_ok, width=0.62, color=col, edgecolor="none")
        ax.bar(x, h_bad, bottom=h_ok, width=0.62, color=DARK, edgecolor="none")
        tot = ok + bad
        lab = f"{100 * ok / tot:.0f}%" if tot else "–"
        ax.text(x, h_ok + h_bad + 3, lab, ha="center", va="bottom", fontsize=FS["pct"],
                fontweight="bold", color="black" if x == 0 else "#7a7a7a")
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 122)
    ax.set_xticks([0, 1], ["Yes", "No"]); ax.set_yticks([0, 50, 100])
    frame(ax)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if first:
        ax.set_ylabel("Assigned\npatients (%)", fontsize=FS["label"], fontweight="bold")
        ax.set_xlabel("Confirmed", fontsize=FS["label"] - 1, fontweight="bold", labelpad=2)
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
            x0, x1 = XLIM[lev]
            assert x0 < min(s1.specificity, two.specificity) and max(s1.specificity, two.specificity) <= 1, (lev, c)
            if lev == "TCGA":
                t = tc[tc.assigned == c].iloc[0]
                assert (t.correct_unconfirmed, t.wrong_unconfirmed) == (k.tp_lost, k.fp_removed), c
                assert t.correct_confirmed + t.wrong_confirmed == t.n_confirmed, c
                assert t.n_confirmed + t.correct_unconfirmed + t.wrong_unconfirmed == t.n_assigned, c
    return met, cur, cmp_, tc


def main():
    met, cur, cmp_, tc = load()
    fig_w = LEFT + 12 * TILE_W + 11 * WSP * TILE_W + RIGHT
    fig_h = TOP + sum(HR) * TILE_H + (len(HR) - 1) * HSP * TILE_H + BOTTOM
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(len(ROWS), 12, left=LEFT / fig_w, right=1 - RIGHT / fig_w, top=1 - TOP / fig_h,
                          bottom=BOTTOM / fig_h, wspace=WSP, hspace=HSP, height_ratios=HR)
    letters = iter("ABCDE")
    first_ax = {}
    for i, (kind, lev) in enumerate(ROWS):
        if kind == "legend":
            continue
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
            elif kind == "dca":
                g = cur[(cur.level == lev) & (cur.variant == VAR[lev]) & (cur.cancer == c)]
                k = cmp_[(cmp_.level == lev) & (cmp_.variant == VAR[lev]) & (cmp_.cancer == c)].iloc[0]
                dca(ax, c, g, k.cases / k.n, j == 0)
            else:
                bars(ax, c, tc[tc.assigned == c].iloc[0], j == 0)
        box = first_ax[i].get_position()
        fig.text(0.003, box.y1 + 0.006, next(letters), fontsize=FS["letter"], fontweight="bold", va="top")
    for rows, lab in [((0, 1), "External"), ((3, 5), "TCGA")]:
        y1, y0 = first_ax[rows[0]].get_position().y1, first_ax[rows[1]].get_position().y0
        fig.add_artist(Line2D([0.0285, 0.0285], [y0, y1], color="#444444", lw=1.6))
        fig.text(0.023, (y0 + y1) / 2, lab, rotation=90, fontsize=FS["cohort"], fontweight="bold",
                 ha="center", va="center")

    multi = tuple(Patch(facecolor=COL[c]) for c in ("AML", "CVX", "GLIOM", "LUNGC"))
    curve = Line2D([], [], color=INK, lw=2.6)
    call_all = Line2D([], [], color="#999999", lw=1.4, ls=":", label="Call everyone positive")
    call_none = Line2D([], [], color="#555555", lw=1.2, ls="--", label="Call no one positive")
    hatch = Patch(facecolor="white", edgecolor="#8a8a8a", hatch="////")
    open_c = Line2D([], [], ls="", marker="o", mfc="white", mec=GREY, mew=2.4, ms=12)
    fill_c = Line2D([], [], ls="", marker="o", mfc=INK, mec="white", ms=13)
    arrow = Line2D([], [], color=INK, lw=2.8, marker=">", ms=11)

    lax = fig.add_subplot(gs[2, :]); lax.axis("off")
    ext = [(open_c, "25-protein model alone"), ((fill_c, curve), "Both models positive"),
           (arrow, "Effect of requiring both"), (call_all, "Call everyone positive"),
           (call_none, "Call no one positive"),
           (Patch(facecolor=INK, alpha=0.18), "Benefit already given by 25-protein model"),
           (Patch(facecolor=INK, alpha=0.70), "Extra benefit from requiring both"),
           (hatch, "Benefit lost by requiring both")]
    lax.legend([h for h, _ in ext], [l for _, l in ext], loc="center", ncol=4, frameon=True,
               fontsize=FS["legend"], handlelength=2.4, columnspacing=2.0, edgecolor="#bbbbbb",
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})

    ax_c = fig.add_subplot(gs[3, 10:12]); ax_c.axis("off")
    ax_c.legend([multi, Patch(facecolor=DARK)], ["Correct assignment", "Wrong assignment"],
                loc="center left", frameon=False, fontsize=FS["slot"], handlelength=2.6,
                labelspacing=0.6, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
    ax_d = fig.add_subplot(gs[4, 10:12]); ax_d.axis("off")
    ax_d.legend([open_c, fill_c, arrow],
                ["Multiclass assignment alone", "Assignment confirmed by\ndisease-specific model",
                 "Effect of requiring\nconfirmation"],
                loc="center left", frameon=False, fontsize=FS["slot"], handlelength=2.2, labelspacing=0.6)
    ax_e = fig.add_subplot(gs[5, 10:12]); ax_e.axis("off")
    ax_e.legend([curve, call_all, call_none, Patch(facecolor=INK, alpha=0.18),
                 Patch(facecolor=INK, alpha=0.70), hatch],
                ["Confirmed assignment", "Call everyone positive", "Call no one positive",
                 "Benefit already given by\nmulticlass assignment", "Extra benefit from confirmation",
                 "Benefit lost by confirmation"],
                loc="center left", frameon=False, fontsize=FS["slot"], handlelength=2.2, labelspacing=0.45)
    fig.savefig(OUT, dpi=300, facecolor="white")
    print(OUT, fig.get_size_inches())


if __name__ == "__main__":
    main()
