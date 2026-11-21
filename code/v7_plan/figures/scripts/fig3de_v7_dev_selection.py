"""v7 Figure 3 D-E (sample): panel choice from the development set only (replaces the v6.x test-set size curves).
(D) The 3,000 multiclass candidates: development 4-fold CV objective by panel size (one point per candidate), the best
    candidate per size joined, and the chosen 26-protein candidate (highest objective) circled.
(E) Development CV metrics of the best candidate per size (macro AUC, balanced accuracy, macro F1, minimum class recall);
    the vertical line marks the chosen size. Lines and markers follow the v6.x Figure 4A style.
(F) Disease-specific size sweeps: development 5-fold CV AUC (solid) and balanced accuracy (dashed) for the top 1-18
    LASSO-ranked proteins; shading = within 0.02 of the best CV AUC; the dotted line and coloured number mark the size
    chosen by the rule (smallest size within 0.02 AUC and 0.05 balanced accuracy of the best).
No internal-test value is plotted. Inputs: stage1_cv_results.csv (v11 tournament), pilot_ds_dev_only_size/size_sweep_devonly.csv,
panels/panel_memberships_v7.csv. Output: figures/output/fig3de_v7_dev_selection.png
"""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import v7_style as S
plt = S.plt

cand = pd.read_csv(S.ROOT / "revise_plan/part_b_multiclass/v11_seed52_lr_l2_wide_compact_sensitivity/tables/stage1_cv_results.csv")
sw = pd.read_csv(S.ROOT / "revise_plan/pilot_ds_dev_only_size/size_sweep_devonly.csv")
pan = pd.read_csv(S.V7 / "panels/panel_memberships_v7.csv")
chosen = {c: int((pan.panel == f"single_{S.NAME[c]}").sum()) for c in S.ORDER}
win = cand.loc[cand.stage1_objective.idxmax()]
best = cand.loc[cand.groupby("feature_count").stage1_objective.idxmax()].sort_values("feature_count")

fig = plt.figure(figsize=(30, 17))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.38, left=0.06, right=0.985, top=0.95, bottom=0.08)
top = gs[0].subgridspec(1, 2, wspace=0.16)
axD, axE = fig.add_subplot(top[0]), fig.add_subplot(top[1])

S.frame(axD)
jit = np.random.default_rng(1).uniform(-0.25, 0.25, len(cand))
axD.scatter(cand.feature_count + jit, cand.stage1_objective, s=9, color=S.PURPLE_LIGHT, alpha=0.6, lw=0, zorder=2)
axD.plot(best.feature_count, best.stage1_objective, "-o", color=S.PURPLE_DARK, lw=2.6, ms=7, zorder=4)
axD.plot(win.feature_count, win.stage1_objective, "o", ms=24, mfc="none", mec=S.PURPLE_DARK, mew=3, zorder=5)
axD.annotate(f"chosen: {int(win.feature_count)} proteins", (win.feature_count, win.stage1_objective), xytext=(26, 0.664),
             ha="center", fontsize=S.FS["annot"] + 1, color=S.PURPLE_DARK, arrowprops=dict(arrowstyle="-", color=S.PURPLE_DARK, lw=1.2))
axD.set_xticks(range(12, 31, 2)); axD.set_xlim(11.3, 30.7)
axD.set_xlabel("Panel size (proteins)", fontsize=S.FS["label"]); axD.set_ylabel("Development CV objective", fontsize=S.FS["label"])
axD.grid(axis="y", color="#eeeeee"); axD.set_axisbelow(True)

S.frame(axE)
MET = [("cv_macro_ovr_auc_mean", "Macro OvR AUC", "-", "o"), ("cv_balanced_accuracy_mean", "Balanced accuracy", "--", "s"),
       ("cv_macro_f1_mean", "Macro F$_1$", "-.", "^"), ("cv_min_class_recall_mean", "Min class recall", ":", "D")]
shades = [S.PURPLE_DARK, S.PURPLE, "#8f63b8", "#b48fd6"]
for (col, lab, ls, mk), c in zip(MET, shades):
    sd = col.replace("_mean", "_std")
    axE.fill_between(best.feature_count, best[col] - best[sd], best[col] + best[sd], color=c, alpha=0.10, lw=0)
    axE.plot(best.feature_count, best[col], ls=ls, marker=mk, color=c, lw=2.4, ms=7, label=lab)
axE.axvline(win.feature_count, color=S.PURPLE, ls=":", lw=1.8)
axE.set_xticks(range(12, 31, 2)); axE.set_xlim(11.3, 30.7); axE.set_ylim(0.1, 1.02)
axE.set_xlabel("Panel size (proteins)", fontsize=S.FS["label"]); axE.set_ylabel("Development CV value", fontsize=S.FS["label"])
axE.grid(axis="y", color="#eeeeee"); axE.set_axisbelow(True)

low = gs[1].subgridspec(2, 6, hspace=0.55, wspace=0.18)
tiles = []
for j, c in enumerate(S.ORDER):
    ax = fig.add_subplot(low[j // 6, j % 6]); tiles.append(ax); S.frame(ax)
    s = sw[sw.cancer == c].sort_values("size")
    col = S.COL[c]
    ax.set_facecolor(S.tint(c, 0.07))
    b = s.cv_auc.max()
    ax.axhspan(b - 0.02, b, color=col, alpha=0.18, lw=0)
    ax.plot(s["size"], s.cv_auc, "-", color=col, lw=2.6)
    ax.plot(s["size"], s.cv_balanced_accuracy, "--", color=S.GREY, lw=2)
    k = chosen[c]
    ax.axvline(k, color=col, ls=":", lw=1.8)
    r = s[s["size"] == k].iloc[0]
    ax.plot(k, r.cv_auc, "o", ms=10, mfc="white", mec=col, mew=2.2, zorder=5)
    ax.plot(k, r.cv_balanced_accuracy, "o", ms=8, mfc="white", mec=S.GREY, mew=2, zorder=5)
    ax.set_xlim(0.5, 18.5); ax.set_ylim(0.4, 1.03); ax.set_xticks([1, 5, 10, 15, 18])
    lbls = ax.set_xticklabels(["1", "5", "10", "15", "18"])
    ax.text(k, -0.16, str(k), transform=ax.get_xaxis_transform(), ha="center", va="top", color=col, fontsize=S.FS["tick"] + 2, fontweight="bold")
    ax.set_title(S.NAME[c], color=col, fontsize=S.FS["title"] - 2, pad=8)
    if j % 6:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("Development CV", fontsize=S.FS["label"] - 1)
    if j == 6:
        ax.set_xlabel("Panel size (proteins)", fontsize=S.FS["label"] - 1, labelpad=22)

for ax, s_ in ((axD, "D"), (axE, "E"), (tiles[0], "F")):
    S.letter(fig, ax, s_, dx=-0.03, dy=0.015)
leg = [Line2D([], [], color=c, ls=ls, marker=mk, lw=2.4, ms=7, label=lab) for (col, lab, ls, mk), c in zip(MET, shades)]
leg += [Line2D([], [], ls="none", marker="o", ms=18, mfc="none", mec=S.PURPLE_DARK, mew=2.5, label="Chosen candidate"),
        Line2D([], [], color=S.INK, lw=2.6, label="CV AUC (disease-specific)"), Line2D([], [], color=S.GREY, lw=2, ls="--", label="CV balanced accuracy"),
        Patch(fc="#999999", alpha=0.35, label="Within 0.02 of the best CV AUC")]
fig.legend(handles=leg, loc="lower center", ncol=4, frameon=False, fontsize=S.FS["legend"] - 1, bbox_to_anchor=(0.52, -0.035))
S.save(fig, "fig3de_v7_dev_selection.png")
