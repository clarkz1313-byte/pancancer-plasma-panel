"""v7 Figure 4, revision l (18 Oct): rim Se labels of BRC-PRC radars moved inside towards 7 o'clock.
Revision k (17 Oct): white backgrounds, colour above the published curve only (D-F and I), gap
numbers in boxes of that colour (D-F) and bottom right in I. Revision j (17 Oct): D-F in the Fig 6 cohort colours with fill above the published curve, larger
"26" and gap numbers; G-H labels on AML only, larger cancer-coloured radar numbers; I own protein count on the x axis,
gap number next to each cancer. Revision i (17 Oct): D-F each in its own neutral colour family (background, published line, area
under it); row I bigger text, dashed gap line to the published model of the same size, * for same-sex scoring.
Based on revisions g and h (author review 17 Oct), built from fig4_v7f_internal.py / fig4_v7d_internal.py:
- bolder polar grids in A and in the disease-specific radars;
- the published-model gap is shown as the number only (no "vs 36-protein model");
- the disease-specific block is split: confusion matrices, radars and the published per-cancer comparison each get
  their own letter.
Version g (fig4_v7g_internal.png): D-F as three panels (macro AUC, macro F1, minimum class recall); G confusion
  matrices, H radars, I per-cancer comparison.
Version h (fig4_v7h_internal.png): D = the three metrics in one plot against proteins in the model (published models
  grey, one line style and marker per metric; this study filled purple at 26 proteins with 95% intervals);
  E confusion matrices, F radars, G per-cancer comparison.
Output: figures/output/fig4_v7g_internal.png, fig4_v7h_internal.png
"""
import importlib.util
import inspect
import sys

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator
from PIL import Image

import v7_style as S

plt = S.plt
Image.MAX_IMAGE_PIXELS = None
spec = importlib.util.spec_from_file_location("f4f", S.V7 / "figures/scripts/fig4_v7f_internal.py")
f4f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f4f)
f4d = f4f.f4d
BUILD = S.V7 / "figures/build/fig4l"
BUILD.mkdir(parents=True, exist_ok=True)
GRID = dict(color="#8f8f8f", lw=1.3)
METRICS = [("auc", "macro_ovr_auc", "Macro AUC", "D", "-"), ("f1", "macro_f1", "Macro F$_1$", "s", (0, (7, 3))),
           ("minrec", "min_class_recall", "Min class recall", "^", (0, (9, 3, 2, 3)))]


def patch(fn, reps, ns):
    src = inspect.getsource(fn)
    for o, n in reps.items():
        assert src.count(o) == 1, o
        src = src.replace(o, n)
    exec(compile(src, f"{fn.__name__} (v7g)", "exec"), ns)


# A: bolder grid
patch(f4f.panel_a, {'ax.grid(color="#d9d9d9"); ax.spines["polar"].set_color("#cccccc")':
                    'ax.grid(color="#8f8f8f", lw=1.3); ax.spines["polar"].set_color("#777777")'}, f4f.__dict__)
# D-F: the gap as a number only
patch(f4f.panel_size, {'ax.annotate(f"{gap:+.3f} vs {NEAREST}-protein model", (26, hi)':
                       'ax.annotate(f"{gap:+.3f}", (26, hi)'}, f4f.__dict__)
# lower block: bolder radar grid, three letters
patch(f4d.lower_gh, {'ar.grid(color="#dddddd", lw=0.6); ar.spines["polar"].set_color("#cccccc")':
                     'ar.grid(color="#8f8f8f", lw=1.2); ar.spines["polar"].set_color("#777777")',
                     'first = {"G": am, "H": af}': 'first = {"G": am, "R": ar, "H": af}',
                     '''    S.letter(fig, first["G"], "G", dx=-0.012, dy=0.045)
    S.letter(fig, first["H"], "H", dx=-0.012, dy=0.02)''':
                     '''    S.letter(fig, first["G"], LETTERS[0], dx=-0.012, dy=0.045)
    S.letter(fig, first["R"], LETTERS[1], dx=-0.012, dy=0.01)
    S.letter(fig, first["H"], LETTERS[2], dx=-0.012, dy=0.02)''',
                     'return save(fig, "lower_GH.png")': 'return save(fig, f"lower_{LETTERS}.png")',
                     '" · same sex" if name in SAME_SEX else ""), transform=af.transAxes,\n                ha="right", fontsize=11,':
                     '""), transform=af.transAxes,\n                ha="right", fontsize=16,',
                     'af.set_xticklabels(["3", "10", "50", "200", "1.5k"], fontsize=11)': '(af.set_xticks([n_prot, 50, 1463]) if j == 0 else af.set_xticks([n_prot])); af.set_xticklabels(([str(n_prot), "50", "1463"] if j == 0 else [str(n_prot)]), fontsize=15); af.get_xticklabels()[0].set_color(col); af.get_xticklabels()[0].set_fontsize(19)',
                     'am.set_xticks([0.5, 1.5]); am.set_xticklabels(["Pred +", "Pred −"], fontsize=12)':
                     'am.set_xticks([0.5, 1.5]); am.set_xticklabels(["Pred +", "Pred −"] if j == 0 else ["", ""], fontsize=12)',
                     'ar.annotate(f"{m}\\n{v:.2f}", xy=': 'ar.annotate((f"{m}\\n{v:.2f}" if j == 0 else f"{v:.2f}"), xy=',
                     'fontsize=11.5, linespacing=0.95,': 'fontsize=16, color=col, linespacing=0.95,',
                     'xy=(ang, min(v + 0.05, 1.08)), xytext=(dx, dy),':
                     'xy=((np.radians(210), 0.50) if (j > 0 and m == "Se" and v >= 0.6) else (ang, min(v + 0.05, 1.08))), xytext=((0, 0) if (j > 0 and m == "Se" and v >= 0.6) else (dx, dy)),',
                     'af.text(0.97, 0.06, f"{n_prot}P" + (': 'af.text(0.97, 0.06, "" + (',

                     'mec="white", mew=1.2, elinewidth=3, capsize=0, zorder=5)':
                     'mec="white", mew=1.2, elinewidth=3, capsize=0, zorder=5)\n'
                     '        pub_at = float(np.interp(np.log(n_prot), np.log(SIZES), pubv))\n'
                     '        af.plot([n_prot, n_prot], [pub_at, r.auc], color=col, lw=2.6, ls=(0, (2, 1.5)), zorder=4)\n'
                     '        af.scatter([n_prot], [pub_at], s=40, color=GREY_PUB, zorder=4)\n'
                     '        af.text(0.97, 0.07, (f"{r.auc - pub_at:+.2f}" if abs(r.auc - pub_at) >= 0.005 else "0.00") + ("*" if name in SAME_SEX else ""), transform=af.transAxes, ha="right", va="bottom", color=col, fontsize=18, fontweight="bold")',
                     'af.fill_between(SIZES, pubv, 0.55, color=S.tint(c, 0.22), zorder=1)': 'af.fill_between(SIZES, pubv, 1.03, color=S.tint(c, 0.22), zorder=1)',
                     'fmt="D", color=col, ms=11,': 'fmt="D", color=col, ms=15,',
                     'Line2D([], [], ls="none", marker="D", ms=12, color=S.INK, label="This study, with 95% interval")]':
                     'Line2D([], [], ls="none", marker="D", ms=12, color=S.INK, label="This study, with 95% interval"),\n'
                     '           Line2D([], [], color=S.INK, lw=2.4, ls=(0, (2, 1.5)), label="gap to the published model of the same size"),\n'
                     '           Line2D([], [], ls="none", label="* scored within one sex")]',
                     'loc="lower center", ncol=2, frameon=False, fontsize=S.FS["legend"], bbox_to_anchor=(0.5, 0.0))':
                     'loc="lower center", ncol=4, frameon=False, fontsize=S.FS["legend"], bbox_to_anchor=(0.5, 0.0))'}, f4d.__dict__)
f4d.BUILD = BUILD


# D-F: one neutral colour family each (no cancer colour, no purple) for background, published line and area under it
STY = [dict(bg=S.tint("#1f5f8b", 0.05), line="#1f5f8b", fill=S.tint("#1f5f8b", 0.22)),   # TCGA blue (Fig 6)
       dict(bg=S.tint("#b07a10", 0.05), line="#b07a10", fill=S.tint("#b07a10", 0.22)),   # expO amber
       dict(bg=S.tint("#2e6b4f", 0.05), line="#2e6b4f", fill=S.tint("#2e6b4f", 0.22))]   # GDC green


def styled(ax, macro, key, ours, title, marker, ylim, ls, st):
    f4f.panel_size(ax, macro, key, ours, title, marker, ylim, S.PURPLE, ls)
    ax.set_facecolor("white")
    ln = ax.lines[0]
    ln.set_color(st["line"]); ln.set_markeredgecolor(st["line"])
    x, y = ln.get_xdata(), ln.get_ydata()
    ax.fill_between(x, y, ylim[1], color=st["fill"], lw=0, zorder=1)    # above the published curve
    ax.texts[-1].set_color(S.INK)
    for t in ax.texts:
        if t.get_text()[:1] in "+-":
            t.set_fontsize(24); t.set_color(S.PURPLE)
            t.set_bbox(dict(facecolor=st["fill"], edgecolor="none", pad=3))
    lab = [t for t in ax.get_xticklabels() if t.get_text() == "26"]
    for t in lab:
        t.set_fontsize(24)
    ax.grid(color="white", lw=1.4)


def combined_d(ax, macro, ci):
    S.frame(ax)
    n = macro.index.to_numpy()
    for key, cikey, lab, mk, ls in METRICS:
        ax.plot(n, macro[key], ls=ls, marker=mk, color=f4f.GREY_PUB, lw=3, ms=12, mfc="white", mew=2.2, zorder=3)
        e, lo, hi = ci[cikey]
        ax.errorbar([26], [e], yerr=[[e - lo], [hi - e]], fmt=mk, color=S.PURPLE, ms=19, mec="white", mew=1.4,
                    elinewidth=3.6, capsize=7, capthick=2.6, zorder=5)
        ax.text(1463 * 1.12, macro[key].iloc[-1], lab, ha="left", va="center", fontsize=S.FS["label"], color=S.INK)
    ax.set_xscale("log"); ax.xaxis.set_major_locator(FixedLocator([12, 26, 36, 83, 1463])); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["12", "26", "36", "83", "1,463"]); ax.get_xticklabels()[1].set_color(S.PURPLE)
    ax.set_xlim(9, 7000); ax.set_ylim(0.2, 1.02)
    ax.set_xlabel("Proteins in the multiclass model (log scale)", fontsize=S.FS["label"] - 1)
    ax.set_ylabel("Internal test value", fontsize=S.FS["label"] - 1)
    ax.grid(color="#eeeeee"); ax.set_axisbelow(True)


def top_block(version):
    ci = f4f.ci_table()
    macro, _ = f4d.published()
    fig = plt.figure(figsize=(30, 22 if version == "g" else 19))
    gs = fig.add_gridspec(2, 3, left=0.05, right=0.955, top=0.95, bottom=0.08, wspace=0.42, hspace=0.50,
                          height_ratios=[1, 1] if version == "g" else [1, 0.62])
    axA = fig.add_subplot(gs[0, 0], projection="polar")
    axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[0, 2])
    for ax in (axB, axC):
        ax.set_box_aspect(1)
    f4f.panel_a(axA, ci); f4f.panel_b(fig, axB); f4f.panel_c(axC)
    if version == "g":
        axD, axE, axF = (fig.add_subplot(gs[1, i]) for i in range(3))
        for ax in (axD, axE, axF):
            ax.set_box_aspect(1)
        styled(axD, macro, "auc", ci["macro_ovr_auc"], "Macro one-versus-rest AUC", "D", (0.84, 1.005), "-", STY[0])
        styled(axE, macro, "f1", ci["macro_f1"], "Macro F$_1$", "s", (0.5, 0.9), (0, (7, 3)), STY[1])
        styled(axF, macro, "minrec", ci["min_class_recall"], "Minimum class recall", "^", (0.2, 0.80), (0, (9, 3, 2, 3)), STY[2])
        lower = [(axD, "D"), (axE, "E"), (axF, "F")]
    else:
        axD = fig.add_subplot(gs[1, 0:3])
        combined_d(axD, macro, ci)
        lower = [(axD, "D")]
    fig.canvas.draw()
    for ax, s_ in [(axA, "A"), (axB, "B"), (axC, "C")] + lower:
        S.letter(fig, ax, s_, dx=-0.035 if ax is not axA else -0.01, dy=0.035 if ax is axA else 0.02)
    yC = axC.get_position().y0 - 0.075
    h1 = [Line2D([], [], color=S.PURPLE, lw=4.4, label="Macro (mean)"), Patch(fc="#bbbbbb", alpha=0.5, label="Treat all"),
          Line2D([], [], color="#555555", lw=2.4, ls="--", label="Treat none")] + \
         [Line2D([], [], color=S.COL[c], lw=3, label=S.NAME[c]) for c in S.ORDER]
    fig.legend(handles=h1, loc="upper center", ncol=8, frameon=False, fontsize=S.FS["legend"] - 3,
               bbox_to_anchor=(0.5, yC), columnspacing=1.4)
    yF = axD.get_position().y0 - 0.06
    h2 = [Line2D([], [], color=f4f.GREY_PUB, lw=3, marker="o", ms=10, mfc="white", mew=2.2, label="Published models of the same cohort (open markers)"),
          Line2D([], [], ls="none", marker="o", ms=13, color=S.PURPLE, label="This study, 26 proteins, 95% interval (filled marker)")]
    fig.legend(handles=h2, loc="upper center", ncol=2, frameon=False, fontsize=S.FS["legend"] - 2, bbox_to_anchor=(0.5, yF))
    p = BUILD / f"top_{version}.png"
    fig.savefig(p, dpi=250, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def build(version):
    f4d.LETTERS = "GHI" if version == "g" else "EFG"
    top = Image.open(top_block(version)).convert("RGB")
    bot = Image.open(f4d.lower_gh()).convert("RGB")
    w = max(top.width, bot.width)
    top = top.resize((w, round(top.height * w / top.width)), Image.LANCZOS)
    bot = bot.resize((w, round(bot.height * w / bot.width)), Image.LANCZOS)
    gap = int(0.01 * w)
    out = Image.new("RGB", (w, top.height + gap + bot.height), "white")
    out.paste(top, (0, 0)); out.paste(bot, (0, top.height + gap))
    p = S.OUT / f"fig4_v7l_internal.png"
    out.save(p, optimize=True)
    print("wrote", p, out.size)


if __name__ == "__main__":
    for v in sys.argv[1:] or ["g"]:
        build(v)
