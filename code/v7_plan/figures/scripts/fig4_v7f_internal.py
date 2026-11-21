"""v7 Figure 4 (revision f, 16 Oct: one purple shade for the 26-protein list; D-F differ by line style and marker;
CLL olive via v7_style; otherwise as revision e): internal test of the 413 patients.
Panels A-F are drawn in ONE canvas on a 2 x 3 grid of equal square plot areas (so tops, bottoms and columns align);
legends sit outside the plots, one strip per row.
A  multiclass macro metrics radar (accuracy, balanced accuracy, macro F1, macro one-vs-rest AUC, minimum class recall),
   95% bootstrap intervals (1,000 resamples, seed 52024) - redrawn after v6 fig5_panels.panel_macro_radar
B  mean predicted probability by true cancer (rows true, columns predicted), diagonal outlined
C  one-vs-rest decision curves of each cancer and their mean (macro), treat-all range shaded, treat-none dashed
D-F published multiclass models of the same cohort (Bueno Alvez et al. 2023, Supplementary Data 6; macro means over the
   12 cancers, minimum = lowest per-cancer recall) against proteins in the model, line with shaded area; ours at 26
   proteins with 95% interval and the difference from the nearest-sized published model (36 proteins).
G-H as revision d (fig4_v7d_internal.lower_gh): disease-specific confusion matrices over radars, and each panel against
   the published per-cancer size series, 12 shared cancer columns.
Output: figures/output/fig4_v7f_internal.png
"""
import importlib.util

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator
from PIL import Image

import v7_style as S
plt = S.plt
Image.MAX_IMAGE_PIXELS = None
B = S.V7 / "figures/build"
BUILD = B / "fig4e"
BUILD.mkdir(parents=True, exist_ok=True)
GREY_PUB = "#555555"
MAT = mcolors.LinearSegmentedColormap.from_list("mat", ["#ffffff", "#e7def2", "#b79ad8", "#6a3d9a"], N=256)
NEAREST = 36
spec = importlib.util.spec_from_file_location("f4d", S.V7 / "figures/scripts/fig4_v7d_internal.py")
f4d = importlib.util.module_from_spec(spec); spec.loader.exec_module(f4d)


def ci_table():
    ci = pd.read_csv(B / "fig4/bootstrap_ci_summary.csv").set_index("metric")
    return {k: ci.loc[k, ["point_estimate", "ci_lower_2_5", "ci_upper_97_5"]].astype(float).to_numpy()
            for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_auc", "min_class_recall")}


def panel_a(ax, ci):
    keys = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced\naccuracy"), ("macro_f1", "Macro F$_1$"),
            ("macro_ovr_auc", "Macro\nOvR AUC"), ("min_class_recall", "Min class\nrecall")]
    ang = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
    v = [ci[k][0] for k, _ in keys]
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1); ax.set_ylim(0, 1.0)
    ax.set_xticks(ang); ax.set_xticklabels([])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=12, color=S.GREY)
    ax.grid(color="#d9d9d9"); ax.spines["polar"].set_color("#cccccc")
    ax.fill(ang + ang[:1], v + v[:1], color=S.PURPLE_LIGHT, alpha=0.6, zorder=2)
    ax.plot(ang + ang[:1], v + v[:1], color=S.PURPLE, lw=3.4, zorder=3)
    ax.scatter(ang, v, s=110, color=S.PURPLE, ec="white", lw=1.5, zorder=5)
    for a, (k, lab) in zip(ang, keys):
        e, lo, hi = ci[k]
        ax.plot([a, a], [lo, hi], color="black", lw=2.4, zorder=6)
        for r in (lo, hi):
            ax.plot([a - 0.04, a + 0.04], [r, r], color="black", lw=2.2, zorder=6)
        ax.text(a, 1.27, lab, ha="center", va="bottom", fontsize=16, fontweight="bold", color=S.INK, linespacing=0.95)
        ax.text(a, 1.27, f"{e:.3f}\n[{lo:.2f}–{hi:.2f}]", ha="center", va="top", fontsize=14, color=S.PURPLE_DARK)


def panel_b(fig, ax):
    m = pd.read_csv(S.RES / "B1_multiclass/B1_mean_probability_by_true_cancer.csv", index_col=0)
    order = S.ORDER
    M = m.loc[order, order].to_numpy()
    ax.imshow(M, cmap=MAT, vmin=0, vmax=1, aspect="auto")
    for i in range(12):
        for j in range(12):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10.5,
                    color="white" if M[i, j] > 0.55 else S.INK, fontweight="bold" if i == j else "normal")
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec="#d1495b", lw=2.2))
    ax.set_xticks(range(12)); ax.set_xticklabels([S.NAME[c] for c in order], rotation=45, ha="right", fontsize=13)
    ax.set_yticks(range(12)); ax.set_yticklabels([S.NAME[c] for c in order], fontsize=13)
    for tl, c in list(zip(ax.get_xticklabels(), order)) + list(zip(ax.get_yticklabels(), order)):
        tl.set_color(S.COL[c])
    ax.set_xlabel("Predicted cancer", fontsize=S.FS["label"] - 1); ax.set_ylabel("True cancer", fontsize=S.FS["label"] - 1)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = ax.inset_axes([1.03, 0.0, 0.035, 1.0])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=MAT), cax=cax)
    cb.set_label("Mean predicted probability", fontsize=13); cb.ax.tick_params(labelsize=11); cb.outline.set_visible(False)


def panel_c(ax):
    dc = pd.read_csv(S.RES / "B1_multiclass/B1_decision_curves.csv")
    S.frame(ax); ax.set_facecolor(S.tint("#6a3d9a", 0.07))
    ta = dc.pivot(index="threshold", columns="cancer", values="nb_treat_all")
    ax.fill_between(ta.index, ta.min(1), ta.max(1), color="#bbbbbb", alpha=0.35, lw=0, zorder=1)
    ax.axhline(0, color="#555555", lw=2.4, ls="--", zorder=2)
    for c in S.ORDER:
        g = dc[dc.cancer == c].sort_values("threshold")
        ax.plot(g.threshold, g.nb_model, color=S.COL[c], lw=2.2, alpha=0.8, zorder=3)
    mac = dc.groupby("threshold").nb_model.mean()
    ax.plot(mac.index, mac.values, color=S.PURPLE, lw=4.4, zorder=5, solid_capstyle="round")
    ax.set_xlim(0, 0.9); ax.set_ylim(-0.05, 0.2)
    ax.set_xlabel("Threshold probability", fontsize=S.FS["label"] - 1); ax.set_ylabel("Net benefit", fontsize=S.FS["label"] - 1)
    ax.grid(color="#e6e0ef"); ax.set_axisbelow(True)


def panel_size(ax, macro, key, ours, title, marker, ylim, shade, ls):
    """Published models: grey line in this panel's line style, open markers of this panel's shape; ours: filled marker in
    this panel's purple shade with 95% interval. No fill under the curve (author request, 16 Oct)."""
    S.frame(ax)
    n = macro.index.to_numpy()
    ax.plot(n, macro[key], ls=ls, marker=marker, color=GREY_PUB, lw=3, ms=11, mfc="white", mew=2.2, zorder=3)
    e, lo, hi = ours
    ax.errorbar([26], [e], yerr=[[e - lo], [hi - e]], fmt=marker, color=shade, ms=18, mec="white", mew=1.4,
                elinewidth=3.6, capsize=7, capthick=2.6, zorder=5)
    gap = e - macro.loc[NEAREST, key]
    ax.annotate(f"{gap:+.3f} vs {NEAREST}-protein model", (26, hi), xytext=(-6, 10), textcoords="offset points",
                ha="left", va="bottom", fontsize=13, color=shade, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax.set_xscale("log"); ax.xaxis.set_major_locator(FixedLocator([12, 26, 36, 83, 1463])); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["12", "26", "36", "83", "1,463"]); ax.get_xticklabels()[1].set_color(shade)
    ax.set_xlim(9, 2400); ax.set_ylim(*ylim)
    ax.set_xlabel("Proteins in the multiclass model (log scale)", fontsize=S.FS["label"] - 2)
    ax.text(0.03, 0.96, title, transform=ax.transAxes, ha="left", va="top", fontsize=S.FS["label"], color=shade, fontweight="bold")
    ax.grid(color="#eeeeee"); ax.set_axisbelow(True)


def top_block():
    ci = ci_table()
    macro, _ = f4d.published()
    fig = plt.figure(figsize=(30, 22))
    gs = fig.add_gridspec(2, 3, left=0.05, right=0.955, top=0.95, bottom=0.08, wspace=0.42, hspace=0.50)
    axA = fig.add_subplot(gs[0, 0], projection="polar")
    axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[0, 2])
    axD, axE, axF = (fig.add_subplot(gs[1, i]) for i in range(3))
    for ax in (axB, axC, axD, axE, axF):
        ax.set_box_aspect(1)
    panel_a(axA, ci); panel_b(fig, axB); panel_c(axC)
    panel_size(axD, macro, "auc", ci["macro_ovr_auc"], "Macro one-versus-rest AUC", "D", (0.84, 1.005), S.PURPLE, "-")
    panel_size(axE, macro, "f1", ci["macro_f1"], "Macro F$_1$", "s", (0.5, 0.9), S.PURPLE, (0, (7, 3)))
    panel_size(axF, macro, "minrec", ci["min_class_recall"], "Minimum class recall", "^", (0.2, 0.80), S.PURPLE, (0, (9, 3, 2, 3)))
    fig.canvas.draw()
    for ax, s_ in ((axA, "A"), (axB, "B"), (axC, "C"), (axD, "D"), (axE, "E"), (axF, "F")):
        S.letter(fig, ax, s_, dx=-0.035 if ax is not axA else -0.01, dy=0.035 if ax is axA else 0.02)
    # row legends
    yC = axC.get_position().y0 - 0.075
    h1 = [Line2D([], [], color=S.PURPLE, lw=4.4, label="Macro (mean)"), Patch(fc="#bbbbbb", alpha=0.5, label="Treat all"),
          Line2D([], [], color="#555555", lw=2.4, ls="--", label="Treat none")] + \
         [Line2D([], [], color=S.COL[c], lw=3, label=S.NAME[c]) for c in S.ORDER]
    fig.legend(handles=h1, loc="upper center", ncol=8, frameon=False, fontsize=S.FS["legend"] - 3,
               bbox_to_anchor=(0.5, yC), columnspacing=1.4)
    yF = axF.get_position().y0 - 0.06
    h2 = [Line2D([], [], color=GREY_PUB, lw=3, marker="o", ms=10, mfc="white", mew=2.2, label="Published models of the same cohort (open markers)"),
          Line2D([], [], ls="none", marker="o", ms=13, color=S.PURPLE, label="This study, 26 proteins, 95% interval (filled marker)")]
    fig.legend(handles=h2, loc="upper center", ncol=2, frameon=False, fontsize=S.FS["legend"] - 2, bbox_to_anchor=(0.5, yF))
    p = BUILD / "top_AF.png"
    fig.savefig(p, dpi=250, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    top = Image.open(top_block()).convert("RGB")
    f4d.BUILD = BUILD
    bot = Image.open(f4d.lower_gh()).convert("RGB")
    w = max(top.width, bot.width)
    top = top.resize((w, round(top.height * w / top.width)), Image.LANCZOS)
    bot = bot.resize((w, round(bot.height * w / bot.width)), Image.LANCZOS)
    gap = int(0.01 * w)
    out = Image.new("RGB", (w, top.height + gap + bot.height), "white")
    out.paste(top, (0, 0)); out.paste(bot, (0, top.height + gap))
    p = S.OUT / "fig4_v7f_internal.png"
    out.save(p, optimize=True)
    print("wrote", p, out.size)


if __name__ == "__main__":
    main()
