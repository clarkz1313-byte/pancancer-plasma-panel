"""v7 Figure 9C (sample, standalone): two-step rule in the internal test and every mixed-cancer series, grouped by
independent family. Left: accuracy of accepted minus all predictions (points, 95% CI). Right: accuracy of step 2
minus the stricter multiclass cutoff at equal coverage; filled = global comparator, open = within-cancer comparator.
Primary series of each family solid and bold; checks lighter. Series below the 7-cancer rule marked with a dagger.
Inputs: results/B5_internal_two_step_summary.json, B8_mixed_cancer/B8_all_series_v7b.csv,
B8_mixed_cancer/B8_class_matched_comparator.csv. Output: figures/output/fig9c_v7_mixed_forest.png
"""
import ast
import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import v7_style as S
plt = S.plt

ser = pd.read_csv(S.RES / "B8_mixed_cancer/B8_all_series_v7b.csv").set_index("id")
cm = pd.read_csv(S.RES / "B8_mixed_cancer/B8_class_matched_comparator.csv").set_index("id")
b5 = json.load(open(S.RES / "B5_internal_two_step_summary.json"))["comparison_0.5"]

ROWS = [  # (id, label, family, primary)
    ("internal", "U-CAN plasma, internal test", "Internal", True),
    ("TCGA_xena", "Original processing", "TCGA", True),
    ("D2c", "Toil processing", "TCGA", False),
    ("D2b", "GDC processing", "TCGA", False),
    ("D2", "GDC + MMRF myeloma", "TCGA", False),
    ("D4", "PCAWG", "TCGA", False),
    ("D1", "GSE2109 microarray", "expO", True),
    ("D7", "11 cancers", "GDC non-TCGA", True),
    ("D8", "12 cancers (+ CLL-map)", "GDC non-TCGA", False),
    ("D6", "CPTAC RNA, 6 cancers †", "GDC non-TCGA", False),
    ("D3", "CPTAC proteins, 6 cancers †", "GDC non-TCGA", False),
    ("D5", "CCLE cell lines", "Cell lines", True),
]
FAMCOL = {"Internal": S.PURPLE_DARK, "TCGA": "#1f5f8b", "expO": "#8a5a00", "GDC non-TCGA": "#2e6b4f", "Cell lines": "#7a4f7a"}


def ci(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


rows = []
for sid, lab, fam, prim in ROWS:
    if sid == "internal":
        g, gc = b5["gain_points"], b5["gain_ci"]
        a, ac = b5["accuracy_diff_points"], b5.get("acc_ci", [np.nan, np.nan])
        n, k = 413, 12
    else:
        r = ser.loc[sid]
        g, gc, a, ac, n, k = r.gain_points, ci(r.gain_ci), r.accuracy_diff_points, ci(r.acc_ci), int(r.n), int(r.cancers)
    c = cm.loc[sid] if sid in cm.index else None
    rows.append(dict(sid=sid, lab=lab, fam=fam, prim=prim, n=n, k=k, g=g, gl=gc[0], gh=gc[1], a=a, al=ac[0], ah=ac[1],
                     ca=None if c is None else c.difference_points, cal=None if c is None else c.ci_lo, cah=None if c is None else c.ci_hi))
R = pd.DataFrame(rows)

# y positions with a gap between families
ys, y, prev = [], 0, None
for f in R.fam:
    if prev is not None and f != prev:
        y += 0.7
    ys.append(y); y += 1; prev = f
R["y"] = -np.array(ys)

fig = plt.figure(figsize=(24, 13))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.06, left=0.02, right=0.985, top=0.93, bottom=0.12)
axL, axG, axA = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
axL.axis("off")
for _, r in R.iterrows():
    col = FAMCOL[r.fam]
    w = "bold" if r.prim else "normal"
    axL.text(0.30, r.y, r.lab, ha="left", va="center", fontsize=S.FS["annot"] + 2, fontweight=w, color=col if r.prim else S.INK)
    axL.text(1.0, r.y, f"{r.n:,} / {r.k}", ha="right", va="center", fontsize=S.FS["annot"] + 1, color=S.INK, fontweight="normal")
for fam, grp in R.groupby("fam", sort=False):
    axL.text(0.0, grp.y.mean(), fam, ha="left", va="center", fontsize=S.FS["annot"] + 3, fontweight="bold", color=FAMCOL[fam])
    axL.plot([0.27, 0.27], [grp.y.min() - 0.3, grp.y.max() + 0.3], color=FAMCOL[fam], lw=3)
axL.text(1.0, 1.1, "Patients / cancers", ha="right", va="bottom", fontsize=S.FS["annot"] + 1, color=S.INK)
axL.set_xlim(0, 1.02); axL.set_ylim(R.y.min() - 0.8, 1.6)

for ax, (m, lo, hi), xl, lim in ((axG, ("g", "gl", "gh"), "Accepted minus all predictions\n(percentage points)", (-2, 50)),
                                  (axA, ("a", "al", "ah"), "Step 2 minus stricter cutoff\n(percentage points)", (-5, 12))):
    S.frame(ax)
    ax.axvline(0, color=S.GREY, lw=1.4)
    for _, r in R.iterrows():
        col = FAMCOL[r.fam]
        alpha = 1 if r.prim else 0.55
        if np.isfinite(r[lo]):
            ax.plot([r[lo], r[hi]], [r.y, r.y], color=col, lw=3 if r.prim else 2, alpha=alpha, solid_capstyle="round")
        ax.plot(r[m], r.y, "o" if m == "g" else "D", ms=13 if r.prim else 10, color=col, alpha=alpha, zorder=5)
        if m == "a" and r.ca is not None and np.isfinite(r.ca):
            if r.ca > lim[1]:   # off scale: arrow at the edge with the value
                ax.annotate("", xy=(lim[1] - 0.05, r.y - 0.28), xytext=(lim[1] - 1.3, r.y - 0.28),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
                ax.text(lim[1] - 1.5, r.y - 0.28, f"{r.ca:+.0f} ({r.cal:.0f} to {r.cah:.0f})", ha="right", va="center",
                        fontsize=S.FS["annot"] - 2, color=col, fontweight="normal")
            else:
                ax.plot([r.cal, r.cah], [r.y - 0.28, r.y - 0.28], color=col, lw=1.6, alpha=0.8)
                ax.plot(r.ca, r.y - 0.28, "D", ms=9, mfc="white", mec=col, mew=2, zorder=6)
    ax.set_ylim(R.y.min() - 0.8, 1.6); ax.set_yticks([]); ax.set_xlim(*lim)
    ax.set_xlabel(xl, fontsize=S.FS["label"])
    ax.grid(axis="x", color="#eeeeee"); ax.set_axisbelow(True)
    for _, grp in R.groupby("fam", sort=False):
        pass
S.letter(fig, axL, "C", dx=0.01, dy=-0.02)
leg = [Line2D([], [], ls="none", marker="o", ms=12, color=S.INK, label="Accepted vs all predictions"),
       Line2D([], [], ls="none", marker="D", ms=11, color=S.INK, label="vs stricter cutoff, all cancers"),
       Line2D([], [], ls="none", marker="D", ms=10, mfc="white", mec=S.INK, mew=2, label="vs stricter cutoff within cancer"),
       Line2D([], [], color=S.INK, lw=3, label="Primary series (bold)"), Line2D([], [], color=S.INK, lw=2, alpha=0.5, label="Check series")]
fig.legend(handles=leg, loc="lower center", ncol=5, frameon=False, fontsize=S.FS["legend"] - 1, bbox_to_anchor=(0.55, 0.0))
S.save(fig, "fig9c_v7_mixed_forest.png")
