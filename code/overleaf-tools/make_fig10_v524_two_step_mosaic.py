"""Manuscript Figure 10, version v5.2.4 (fork of make_fig10_v523_two_step_mosaic.py: all tick labels in a tile share one number of decimals); v5.2.3 was a fork of make_fig10_v522_two_step_mosaic.py ( tick labels spaced by width; circles closer than one symbol are drawn apart horizontally, true sensitivity kept); v5.2.2 was a fork of make_fig10_v521_two_step_mosaic.py: rows B and D baseline solid in cancer colour, step 1 + step 2 curve dashed; v5.2.1 was a fork of make_fig10_v520_two_step_mosaic.py: faint cancer-colour\nbackground in rows A and C; otherwise as v5.2.0, itself a fork of make_fig10_v5110_two_step_mosaic.py;
circles follow v5.1.7, numbered, joined by a dashed line with a mid-line arrow).

Twelve cancer columns, CLL and myeloma last (absent from TCGA); equal-size tiles; two
cohort groups sharing one legend (in the empty TCGA slots).
  A External, C TCGA  sensitivity against specificity: numbered circles for step 1 alone (1)
                      and step 1 + step 2 (2), joined by a dashed line with a mid-line arrow.
                      x axis zoomed per cancer; y axis shared within each cohort.
  B External, D TCGA  change in net benefit from adding step 2 (two-step minus step 1),
                      net true positives per 100 patients, across decision thresholds;
                      colour = gain, hatch = loss; saved patient-bootstrap 95% intervals
                      at 0.50 and 0.80. y axis shared within each cohort.
Inputs: saved outputs of scripts 64 and 70. Output: figures_300/fig10_v524_two_step_mosaic.png
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "revise_plan"
OUT = ROOT / "overleaf-source-git/figures_300/fig10_v524_two_step_mosaic.png"
sys.path.insert(0, str(ROOT / "figurev6/scripts"))
import fig89_common as FC  # noqa: E402

ORDER = [c for c in FC.CANCER_ORDER if c not in ("CLL", "MYEL")] + ["CLL", "MYEL"]
NAME = {c: ("DLBCL" if c == "LYMPH" else c) for c in ORDER}
COL = FC.PLOT_COLOR
GREY, INK = "#6b6b6b", "#3a3a3a"
VAR = {"TCGA": "step2>=0.50", "external": "step2>=0.80"}
MIN_XSPAN = {"external": 0.10, "TCGA": 0.008}
ROC_Y = {"external": ((0.50, 1.025), [0.5, 0.75, 1.0], ["0.5", "0.75", "1"]),
         "TCGA": ((-0.04, 1.06), [0, 0.5, 1.0], ["0", "0.5", "1"])}
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


def x_window(a, b, lev):
    lo, hi = min(a, b), max(a, b)
    span = max(hi - lo, MIN_XSPAN[lev])
    pad = 0.22 * span
    mid = (lo + hi) / 2
    x0, x1 = mid - span / 2 - pad, mid + span / 2 + pad
    if x1 > 1 + 0.9 * pad:
        x0, x1 = x0 - (x1 - 1 - 0.9 * pad), 1 + 0.9 * pad
    return x0, x1


def numbered(ax, x, y, label, face, edge, txt, fs=14):
    ax.scatter(x, y, s=430, marker="o", facecolor=face, edgecolor=edge, lw=2.4, zorder=6, clip_on=False)
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, fontweight="bold", color=txt, zorder=7,
            clip_on=False)


DODGED = []


def axis_labels(ax, first):
    frame(ax)
    if first:
        ax.set_ylabel("Sensitivity", fontsize=FS["label"], fontweight="bold")
        ax.set_xlabel("Specificity", fontsize=FS["label"] - 1, fontweight="bold", labelpad=2)
    else:
        ax.set_yticklabels([])


def spaced_ticks(ax, x0, x1):
    w = x1 - x0
    ticks = [v for v in MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(x0, x1)
             if x0 + 0.12 * w <= v <= min(1.0 + 1e-12, x1)]
    while True:
        step = ticks[1] - ticks[0] if len(ticks) > 1 else 0.01
        dec = max(1, int(-np.floor(np.log10(step) + 1e-9)))
        labels = [f"{v:.{dec}f}" for v in ticks]
        if len(ticks) < 2:
            break
        px_per_unit = ax.bbox.width / w
        char_px = 0.62 * FS["tick"] * ax.figure.dpi / 72
        gaps_ok = all((ticks[i + 1] - ticks[i]) * px_per_unit
                      >= (len(labels[i]) + len(labels[i + 1])) / 2 * char_px + 0.6 * char_px
                      for i in range(len(ticks) - 1))
        gap_px = WSP * ax.bbox.width
        half = [len(l) / 2 * char_px for l in labels]
        inside = [(v - x0) * px_per_unit - h >= -0.35 * gap_px and (x1 - v) * px_per_unit - h >= -0.35 * gap_px
                  for v, h in zip(ticks, half)]
        if not all(inside):
            ticks = [v for v, ok in zip(ticks, inside) if ok]
            continue
        if gaps_ok:
            break
        ticks = ticks[::-1][::2][::-1]
    ax.set_xticks(ticks, labels)


def roc(ax, c, s1, two, lev, first):
    col = COL[c]
    ax.set_facecolor(FC._blend(col, 0.09))
    p1 = (s1.specificity, s1.sensitivity)
    p2 = (two.specificity, two.sensitivity)
    x0, x1 = x_window(p1[0], p2[0], lev)
    (y0, y1), yt, yl = ROC_Y[lev]
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    spaced_ticks(ax, x0, x1)
    ax.set_yticks(yt, yl)
    ax.grid(color="white", lw=1.1)
    same = s1.tp == two.tp and s1.fp == two.fp
    q1, q2 = ax.transData.transform(p1), ax.transData.transform(p2)
    d = q2 - q1
    SEP = 72.0
    if np.hypot(*d) < SEP:
        # move apart horizontally only (true sensitivity kept); step 2 never lowers specificity
        dx = np.sqrt(max(SEP ** 2 - d[1] ** 2, 0.0))
        m = (q1[0] + q2[0]) / 2
        q1[0], q2[0] = m - dx / 2, m + dx / 2
        bb, r = ax.bbox, 17.0
        shift = (bb.x0 + r - q1[0]) if q1[0] < bb.x0 + r else ((bb.x1 - r - q2[0]) if q2[0] > bb.x1 - r else 0.0)
        q1[0] += shift; q2[0] += shift
        inv = ax.transData.inverted()
        o1, o2 = p1, p2
        p1, p2 = tuple(inv.transform(q1)), tuple(inv.transform(q2))
        assert np.isclose(p1[1], o1[1]) and np.isclose(p2[1], o2[1])
        DODGED.append((lev, NAME[c], "unchanged" if same else "moved",
                       round(max(abs(p1[0] - o1[0]), abs(p2[0] - o2[0])), 4)))
    if same:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, "=", ha="center", va="center", fontsize=20,
                fontweight="bold", color="#444444", zorder=7)
    else:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=col, lw=2.6, ls=(0, (3.0, 1.8)), zorder=4)
        a = (p1[0] + 0.40 * (p2[0] - p1[0]), p1[1] + 0.40 * (p2[1] - p1[1]))
        b = (p1[0] + 0.60 * (p2[0] - p1[0]), p1[1] + 0.60 * (p2[1] - p1[1]))
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=30, color=col, lw=0,
                                     shrinkA=0, shrinkB=0, zorder=5))
    numbered(ax, *p1, "1", "white", GREY, "#444444")
    numbered(ax, *p2, "2", col, "white", "white")
    axis_labels(ax, first)


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
    ax.axhline(0, color=col, lw=3.4, zorder=4)
    ax.plot(pt, d, color=col, lw=2.6, ls=(0, (3.0, 1.8)), zorder=5)
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
            xw = x_window(s1.specificity, two.specificity, lev)
            for v, (lo_, hi_) in [(s1.specificity, xw), (two.specificity, xw),
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
    step1 = ((Line2D([], [], ls="", marker="o", mfc="white", mec=GREY, mew=2.6, ms=22),
              Line2D([], [], ls="", marker="$1$", ms=11, color="#444444")),
             Line2D([], [], color=INK, lw=3.4))
    both = ((Line2D([], [], ls="", marker="o", mfc=INK, mec="white", ms=22),
             Line2D([], [], ls="", marker="$2$", ms=11, color="white")),
            Line2D([], [], color=INK, lw=2.6, ls=(0, (3.0, 1.8))))
    ax_l.legend([step1, both, Patch(facecolor=INK, alpha=0.8),
                 Patch(facecolor="white", edgecolor="#8a8a8a", hatch="////"), ci],
                ["Step 1 alone", "Step 1 + step 2", "Gain from adding step 2",
                 "Loss from adding step 2",
                 "95% interval"],
                handler_map={step1: HandlerTuple(ndivide=None, pad=0.35), both: HandlerTuple(ndivide=None, pad=0.35), ci: HandlerTuple(ndivide=1, pad=0)}, loc="center left", frameon=False,
                fontsize=FS["legend"] + 1, handlelength=2.6, handleheight=1.4, labelspacing=1.2,
                borderaxespad=0.1)
    fig.savefig(OUT, dpi=300, facecolor="white")
    print(OUT, fig.get_size_inches())
    print("drawn apart:", DODGED)


if __name__ == "__main__":
    main()
