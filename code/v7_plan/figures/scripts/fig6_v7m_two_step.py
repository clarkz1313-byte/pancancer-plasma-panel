"""v7 Figure 6, revision m (18 Oct): "Incorrect" for wrong step-1 calls, blank line before counts, x ticks on
B's first row. Earlier: revision k (17 Oct): each flow one polygon (no stripes). Revision j: solid flow blocks. Revision i: left-to-right flows again (as revision g), flipped vertically so the wrong
calls and crossing ribbons are on top; labels name / count / % on three lines. Revision h (author review 17 Oct): flow diagrams vertical (step 1 bottom, step 2 top) with
name / count / percentage on three lines; curves on a tight y range from 10% kept, step 1 dashed and step 1 + 2
solid, bigger markers; tissue rows split into C (sensitivity/specificity) and D (net benefit); cohorts E-H.
Revision g: no section rules/titles; confirmed group (bracket, label, counts)
in the cohort colour and undecided group in grey; curves show correct among kept (higher = better). Revision f: segment labels as counts only, "Undecided" restored, correct
flows in cohort colours and wrong calls in plain greys, gains coloured and losses white (no hatching) in every
panel, bolder colours. Revision e description: the two-step rule in the three kinds of test, one canvas.
Internal test (U-CAN plasma):
  (A) flow of the 413 patients from step 1 (correct / wrong cancer named) to step 2 (confirmed / not confirmed at a
      step-2 probability of 0.50), every label as count (%), percentages within the group the label belongs to;
      next to it, wrong calls among the patients kept as more patients are kept, keeping patients in order of the
      step-1 probability (1, a stricter step-1 cutoff) or of the step-2 probability (2, step 1 + step 2); the gap is
      coloured where step 2 makes fewer errors and hatched where it makes more.
  (B) change in net benefit per cancer from adding step 2 (net true positives per 100 patients across decision
      thresholds; colour = gain, hatch = loss; 95% intervals at 0.50 and 0.80; B5).
Single-cancer external tests, 12 cancers (C): sensitivity against specificity moving from step 1 alone (1) to step 1 +
  step 2 (2), and the change in net benefit (v6.x Figure 10 grammar, B6b binary rule).
Mixed-cancer external cohorts (D TCGA, E expO, F other GDC cohorts, G cancer cell lines, CCLE): as A.
No subtitles, no operating-point markers; cohort sizes and processing are in the caption and Supplementary Table.
Inputs: as fig5_v7d_two_step.py. Output: figures/output/fig6_v7h_two_step.png
"""
import importlib.util

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import v7_style as S

plt = S.plt
spec = importlib.util.spec_from_file_location("f5c", S.V7 / "figures/scripts/fig5_v7c_two_step.py")
f5c = importlib.util.module_from_spec(spec); spec.loader.exec_module(f5c)
spec = importlib.util.spec_from_file_location("f9ab", S.V7 / "figures/scripts/fig9ab_v7_two_step_tissue.py")
f9ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(f9ab)

HUE = {"internal": "#6a3d9a", "tcga": "#1f5f8b", "expo": "#b07a10", "gdc": "#2e6b4f", "ccle": "#7a4f7a"}
NAMES = {"internal": "Internal test", "tcga": "TCGA", "expo": "expO", "gdc": "Other GDC cohorts",
         "ccle": "Cancer cell lines (CCLE)"}
PATH = {k: p for k, lab, p in f5c.SERIES}
WRONG_DARK, WRONG_LIGHT = "#555555", "#c8c8c8"   # wrong calls: grey only, no hatching (author, 17 Oct)
UND = "#7a7a7a"                                   # undecided group labels


def fade(c, f):
    r = np.array(mcolors.to_rgb(c))
    return tuple(r * f + (1 - f))


def ribbon(ax, x0, x1, a0, a1, b0, b1, **kw):
    """One closed polygon per flow (fill_between draws strips that show as stripes when rendered)."""
    from matplotlib.patches import Polygon
    xs = np.linspace(x0, x1, 120)
    s = (1 - np.cos(np.pi * (xs - x0) / (x1 - x0))) / 2
    lo, hi = a0 + (b0 - a0) * s, a1 + (b1 - a1) * s
    pts = np.vstack([np.column_stack([xs, lo]), np.column_stack([xs[::-1], hi[::-1]])])
    c = kw.pop("color", None) or kw.pop("facecolor", None)
    ax.add_patch(Polygon(pts, closed=True, facecolor=c, edgecolor='none', lw=0, alpha=kw.get("alpha", 1.0), zorder=2))


def spread(ys, gap):
    """Push label positions apart (keeping order) so no two are closer than gap."""
    order = np.argsort(ys)[::-1]
    out = np.array(ys, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[a] - out[b] < gap:
            out[b] = out[a] - gap
    return out


def pct(a, b):
    return f"{a:,} ({100 * a / b:.1f}%)" if b else f"{a:,}"


def alluvial(ax, key, t, p, second):
    """Correct calls in shades of the cohort colour, wrong calls in greys (no hatching); segment labels are counts,
    group labels count (% of patients)."""
    hue = HUE[key]
    ok, acc = t == p, second >= 0.5
    n = len(t)
    nc, nw = ok.sum(), (~ok).sum()
    ac, wc, aw, cw = (ok & acc).sum(), (ok & ~acc).sum(), (~ok & acc).sum(), (~ok & ~acc).sum()
    g = 0.035 * n
    top = n + 3 * g
    L_ok = (top - nc, top); L_bad = (L_ok[0] - g - nw, L_ok[0] - g)
    R_ac = (top - ac, top); R_aw = (R_ac[0] - aw, R_ac[0])
    R_cw = (R_aw[0] - 2 * g - cw, R_aw[0] - 2 * g); R_wc = (R_cw[0] - wc, R_cw[0])
    xl, xr, w = 0.0, 1.0, 0.07
    light = fade(hue, 0.45)
    for (x, (lo, hi), c) in ((xl - w, L_ok, hue), (xl - w, L_bad, WRONG_DARK), (xr, R_ac, hue), (xr, R_aw, WRONG_DARK),
                             (xr, R_cw, WRONG_LIGHT), (xr, R_wc, light)):
        ax.add_patch(plt.Rectangle((x, lo), w, hi - lo, fc=c, lw=0))
    ribbon(ax, xl, xr, L_ok[1] - ac, L_ok[1], R_ac[0], R_ac[1], color=hue, alpha=0.55)
    ribbon(ax, xl, xr, L_ok[0], L_ok[0] + wc, R_wc[0], R_wc[1], color=light, alpha=0.75)
    ribbon(ax, xl, xr, L_bad[1] - aw, L_bad[1], R_aw[0], R_aw[1], color=WRONG_DARK, alpha=0.55)
    ribbon(ax, xl, xr, L_bad[0], L_bad[0] + cw, R_cw[0], R_cw[1], color=WRONG_LIGHT, alpha=0.75)
    fs = S.FS["annot"] + 2
    ax.text(xl - w - 0.03, np.mean(L_ok), f"Correct\n\n{nc:,}\n{100 * nc / n:.1f}%", ha="right", va="center", fontsize=fs, color=hue, linespacing=1.1)
    ax.text(xl - w - 0.03, np.mean(L_bad), f"Incorrect\n\n{nw:,}\n{100 * nw / n:.1f}%", ha="right", va="center", fontsize=fs, color=WRONG_DARK, linespacing=1.1)
    segs = [(R_ac, hue, ac), (R_aw, hue, aw), (R_cw, UND, cw), (R_wc, UND, wc)]
    ys = spread([np.mean(s_[0]) for s_ in segs], 0.07 * n)
    for (rng, c, v), y in zip(segs, ys):
        ax.text(xr + w + 0.03, y, f"{v:,}", ha="left", va="center", fontsize=fs, color=c)
    for (lo, hi), lab, c in (((R_aw[0], R_ac[1]), f"Confirmed\n\n{ac + aw:,}\n{100 * (ac + aw) / n:.1f}%", hue),
                             ((R_wc[0], R_cw[1]), f"Undecided\n\n{wc + cw:,}\n{100 * (wc + cw) / n:.1f}%", UND)):
        xb = xr + w + 0.42
        ax.plot([xb, xb], [lo, hi], color=c, lw=2.2)
        ax.text(xb + 0.04, (lo + hi) / 2, lab, ha="left", va="center", fontsize=fs, color=c, linespacing=1.1)
    # flipped vertically (author, 17 Oct): wrong calls and the crossing ribbons on top, correct calls below
    ax.text(xl - w / 2, R_wc[0] - 0.6 * g, "Step 1", ha="center", va="bottom", fontsize=fs + 1, color=S.INK)
    ax.text(xr + w / 2, R_wc[0] - 0.6 * g, "Step 2", ha="center", va="bottom", fontsize=fs + 1, color=S.INK)
    ax.set_xlim(-0.70, 2.10); ax.set_ylim(top + g, R_wc[0] - 2.6 * g); ax.axis("off")


def numbered(ax, x, y, s, face, edge, txt):
    ax.scatter(x, y, s=900, color=face, edgecolor=edge, lw=2.6, zorder=8)
    ax.text(x, y, s, ha="center", va="center", fontsize=19, fontweight="bold", color=txt, zorder=9)


def curve(ax, key, t, p, conf, second):
    hue = HUE[key]
    ok = t == p
    n = len(t)
    S.frame(ax)
    k = np.arange(1, n + 1)
    cov = 100 * k / n
    r1 = 100 * np.cumsum(ok[np.argsort(-conf, kind="stable")]) / k
    r2 = 100 * np.cumsum(ok[np.argsort(-second, kind="stable")]) / k
    m = cov >= 10
    ax.fill_between(cov[m], r1[m], r2[m], where=r2[m] >= r1[m], color=hue, alpha=0.55, lw=0, interpolate=True)
    ax.plot(cov[m], r1[m], color=S.GREY, lw=3.2, ls=(0, (3.0, 1.8)))
    ax.plot(cov[m], r2[m], color=hue, lw=3.6)
    i = np.searchsorted(cov, 35)
    numbered(ax, cov[i], r1[i], "1", "white", S.GREY, "#444444")
    i = np.searchsorted(cov, 65)
    numbered(ax, cov[i], r2[i], "2", hue, "white", "white")
    lo, hi = min(r1[m].min(), r2[m].min()), max(r1[m].max(), r2[m].max())
    pad = 0.06 * (hi - lo)
    ax.set_xlim(9, 101); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Patients kept (%)", fontsize=S.FS["label"] - 1)
    ax.set_ylabel("Correct among kept (%)", fontsize=S.FS["label"] - 1)
    ax.grid(color="#ececec"); ax.set_axisbelow(True)


def pair(fig, cell, key, letter):
    sub = cell.subgridspec(1, 2, width_ratios=[1.45, 1], wspace=0.14)
    t, p, conf, second = f5c.load(PATH[key])
    a1, a2 = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    alluvial(a1, key, t, p, second)
    curve(a2, key, t, p, conf, second)
    box = a1.get_position()
    fig.text(box.x0 - 0.004, box.y1 + 0.004, letter, fontsize=S.FS["letter"], fontweight="bold", ha="left", va="bottom")
    if True:
        fig.text(box.x0 + 0.022, box.y1 + 0.006, NAMES[key], fontsize=S.FS["group"], color=HUE[key], fontweight="bold",
                 ha="left", va="bottom")


def internal_nb(fig, cell):
    d = pd.read_csv(S.RES / "B5_internal_per_patient.csv")
    nb = pd.read_csv(S.RES / "B5_net_benefit_per_cancer.csv")
    n = len(d)
    acc = d.second.to_numpy() >= 0.5
    cor = d.correct.to_numpy().astype(bool)
    pts = np.linspace(0.05, 0.85, 81)
    w = pts / (1 - pts)
    sub = cell.subgridspec(2, 6, wspace=0.12, hspace=0.42)
    first = None
    for j, c in enumerate(S.ORDER):
        ax = fig.add_subplot(sub[j // 6, j % 6]); S.frame(ax)
        nm = S.NAME[c]
        m = d.pred.to_numpy() == c
        tl, fr = (m & ~acc & cor).sum(), (m & ~acc & ~cor).sum()
        y = 100 * (fr * w - tl) / n
        col = S.COL[c]
        ax.fill_between(pts, 0, y, where=y > 0, color=col, alpha=1.0, lw=0, interpolate=True)
        ax.axhline(0, color=col, lw=3.0, ls=(0, (3.0, 1.8)), zorder=4)     # step 1 alone (dashed)
        ax.plot(pts, y, color=col, lw=3.2, zorder=5)                        # step 1 + step 2 (solid)
        rr = nb[nb.cancer == c].iloc[0]          # DLBCL stored as LYMPH
        for t_ in (0.5, 0.8):
            mid, lo, hi = rr[f"nb_diff_{t_}"], rr[f"nb_lo_{t_}"], rr[f"nb_hi_{t_}"]
            ax.errorbar(t_, mid, yerr=[[mid - lo], [hi - mid]], fmt="o", ms=9, color="black", capsize=5, lw=2.2, zorder=6)
        ax.set_xlim(0.03, 0.87); ax.set_ylim(-8, 22); ax.set_xticks([0.2, 0.5, 0.8]); ax.set_yticks([0, 10, 20])
        ax.set_title(nm, color=col, fontsize=S.FS["tick"] + 2, pad=4)
        ax.tick_params(labelsize=S.FS["tick"] - 3)
        if j % 6:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Change in net\nbenefit per 100", fontsize=S.FS["label"] - 3)
        if j == 6:
            ax.set_xlabel("Decision threshold", fontsize=S.FS["label"] - 3)
        first = first or ax
    box = first.get_position()
    fig.text(box.x0 - 0.035, box.y1 + 0.012, "B", fontsize=S.FS["letter"], fontweight="bold", ha="left", va="bottom")


def tissue(fig, cell):
    """Tile renderers of the v6.x Figure 10 builder, patched with the v7 inputs as in fig9ab_v7_two_step_tissue."""
    src = f9ab.SRC.read_text(encoding="utf-8")
    for o, n in f9ab.REPS.items():
        src = src.replace(o, n)
    for o, n in {'    ax.scatter(x, y, s=430, marker="o", facecolor=face, edgecolor=edge, lw=2.4, zorder=6, clip_on=False)':
                 '    ax.scatter(x, y, s=760, marker="o", facecolor=face, edgecolor=edge, lw=2.6, zorder=6, clip_on=False)',
                 "def numbered(ax, x, y, label, face, edge, txt, fs=14):": "def numbered(ax, x, y, label, face, edge, txt, fs=18):",
                 'fmt="o", color="black", ms=5.5, lw=1.7,': 'fmt="o", color="black", ms=9, lw=2.2,',
                 "    ax.axhline(0, color=col, lw=3.4, zorder=4)\n": "    ax.axhline(0, color=col, lw=3.0, ls=(0, (3.0, 1.8)), zorder=4)\n",
                 "    ax.plot(pt, d, color=col, lw=2.6, ls=(0, (3.0, 1.8)), zorder=5)\n": "    ax.plot(pt, d, color=col, lw=3.0, zorder=5)\n",
                 "    ax.fill_between(pt, 0, d, where=d > 0, color=col, alpha=0.75, lw=0, interpolate=True)\n":
                 "    ax.fill_between(pt, 0, d, where=d > 0, color=col, alpha=1.0, lw=0, interpolate=True)\n",
                 '''    ax.fill_between(pt, d, 0, where=d < 0, facecolor="none", edgecolor="#8a8a8a", hatch="////",
                    lw=0, interpolate=True)\n''': ""}.items():
        assert src.count(o) == 1, o
        src = src.replace(o, n)
    ns = {"__file__": str(f9ab.SRC), "__name__": "tissue_tiles"}
    exec(compile(src, "make_fig10_v53 (v7 tiles)", "exec"), ns)
    met, cur, cmp_ = ns["load"]()
    sub = cell.subgridspec(2, 12, wspace=0.12, hspace=0.50)
    first = second_ = None
    for j, c in enumerate(ns["ORDER"]):
        sel = lambda g: g[(g.level == "external") & (g.variant == ns["VAR"]["external"]) & (g.cancer == c)]
        k = sel(cmp_).iloc[0]
        m = sel(met)
        a = fig.add_subplot(sub[0, j])
        ns["roc"](a, c, m[m.rule == "step1"].iloc[0], m[m.rule == "two_step"].iloc[0], "external", j == 0)
        a.set_title(ns["NAME"][c], color=ns["COL"][c], fontsize=S.FS["title"] - 4, pad=8)
        b = fig.add_subplot(sub[1, j])
        ns["dnb"](b, c, sel(cur), k, "external", j == 0)
        first = first or a
        second_ = second_ or b
    for ax_, L in ((first, "C"), (second_, "D")):
        box = ax_.get_position()
        fig.text(box.x0 - 0.035, box.y1 + 0.012, L, fontsize=S.FS["letter"], fontweight="bold", ha="left", va="bottom")


def section(fig, y, title):
    fig.add_artist(Line2D([0.012, 0.988], [y, y], color="#9a9a9a", lw=1.4))
    fig.text(0.012, y + 0.003, title, fontsize=S.FS["group"] + 2, color=S.INK, fontweight="bold", ha="left", va="bottom")


def legend(fig):
    step1 = ((Line2D([], [], ls="", marker="o", mfc="white", mec=S.GREY, mew=2.4, ms=20),
              Line2D([], [], ls="", marker="$1$", ms=10, color="#444444")), Line2D([], [], color=S.GREY, lw=2.6, ls=(0, (3.0, 1.8))))
    both = ((Line2D([], [], ls="", marker="o", mfc=S.INK, mec="white", ms=20),
             Line2D([], [], ls="", marker="$2$", ms=10, color="white")),
            Line2D([], [], color=S.INK, lw=2.8))
    ci = (Line2D([], [], ls="", marker="|", ms=16, mew=1.8, color="black"), Line2D([], [], ls="", marker="o", ms=6, color="black"))
    fig.legend([step1, both, Patch(facecolor="#7a7a7a"), Patch(facecolor="white", edgecolor="#7a7a7a"), ci],
               ["Step 1 alone", "Step 1 + step 2", "Gain from adding step 2 (colour)", "Loss (white)", "95% interval"],
               handler_map={step1: HandlerTuple(ndivide=None, pad=0.3), both: HandlerTuple(ndivide=None, pad=0.3),
                            ci: HandlerTuple(ndivide=1, pad=0)},
               loc="lower center", ncol=5, frameon=False, fontsize=S.FS["legend"], bbox_to_anchor=(0.5, 0.002),
               handlelength=2.6, handleheight=1.3, columnspacing=2.2)


def main():
    fig = plt.figure(figsize=(32, 36))
    fig._figurev6_text_scaled = True
    H = [7.4, 1.0, 6.4, 1.7, 7.4, 0.9, 7.4]
    gs = fig.add_gridspec(len(H), 1, height_ratios=H, hspace=0.0, left=0.045, right=0.988, top=0.955, bottom=0.04)
    top_row = gs[0].subgridspec(1, 2, wspace=0.10)
    pair(fig, top_row[0], "internal", "A")
    internal_nb(fig, top_row[1])
    tissue(fig, gs[2])
    r1 = gs[4].subgridspec(1, 2, wspace=0.10)
    pair(fig, r1[0], "tcga", "E"); pair(fig, r1[1], "expo", "F")
    r2 = gs[6].subgridspec(1, 2, wspace=0.10)
    pair(fig, r2[0], "gdc", "G"); pair(fig, r2[1], "ccle", "H")
    legend(fig)
    S.save(fig, "fig6_v7m_two_step.png")


if __name__ == "__main__":
    main()
