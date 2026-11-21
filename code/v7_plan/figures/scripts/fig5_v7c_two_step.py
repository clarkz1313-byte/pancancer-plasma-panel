"""v7 Figure 5 (revision c, author review 16 Oct): the two-step rule in every data source, one grammar per row.

Rows with patients assigned among several cancers (internal plasma test and each mixed-cancer family) share three
panels, aligned in columns:
  (i)   alluvial flow of the patients: step 1 correct / wrong -> accepted (correct, wrong) or undecided (a wrong call
        held back, a correct call held back);
  (ii)  errors against coverage: share of wrong calls among the accepted patients when patients are accepted in order
        of the step-1 probability (solid = the stricter multiclass cutoff) or of the step-2 probability (dashed); the
        numbered circle is step 2 at 0.50 (and 0.80 internally), the open square the stricter cutoff accepting as many
        patients, the open triangle the same applied within each predicted cancer;
  (iii) per predicted cancer, the share of correct calls before (open circle) and after step 2 (filled circle).
Internal (A-C) is followed by the per-cancer net-benefit change (D, as v7 Fig 7C); tissue rows (E-F) are the
single-cancer binary tests (fig9ab_v7_two_step_tissue.py); mixed-cancer families are G TCGA (original processing),
H expO, I GDC non-TCGA (11 cancers), J CCLE cell lines (primary series of each family; checks in the supplement).
Output: figures/output/fig5_v7c_two_step.png
"""
import importlib.util

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

import v7_style as S
plt = S.plt
Image.MAX_IMAGE_PIXELS = None
BUILD = S.V7 / "figures/build/fig5c"
BUILD.mkdir(parents=True, exist_ok=True)
R = S.RES
C_OK, C_WIT, C_BAD, C_CAU = "#5b3a8c", "#cdbde3", "#b5453a", "#eab3a5"
C_L_OK, C_L_BAD = "#7b5aa8", "#b5453a"
W_RATIOS = [1.0, 1.0, 1.55]
DISP = lambda c: "DLBCL" if c == "LYMPH" else c
COLD = lambda c: S.COL.get(c, S.COL["LYMPH"] if c == "DLBCL" else "#777777")

SERIES = [("internal", "Internal test", R / "B5_internal_per_patient.csv"),
          ("tcga", "TCGA", R / "B7_tcga/per_patient_tcga_xena_dev_only.csv"),
          ("expo", "expO", R / "B8_mixed_cancer/pilot_series/per_patient_D1_dsdevonly.csv"),
          ("gdc", "GDC non-TCGA", R / "B8_mixed_cancer/per_patient_D7_gdc_nontcga.csv"),
          ("ccle", "Cell lines", R / "B8_mixed_cancer/pilot_series/per_patient_D5_dsdevonly.csv")]
SUB = {"internal": "U-CAN plasma, 413 patients, 12 cancers", "tcga": "original processing, 4,665 patients, 10 cancers",
       "expo": "GSE2109 microarray, 1,200 tumours, 7 cancers", "gdc": "11 cancers, 2,791 patients",
       "ccle": "CCLE, 491 cell lines, 8 cancers"}


def load(path):
    d = pd.read_csv(path)
    sfx = "_26" if "pred_26" in d else ""
    t = d.true.map(DISP).to_numpy(); p = d[f"pred{sfx}"].map(DISP).to_numpy()
    return t, p, d[f"conf{sfx}"].to_numpy(), d[f"second{sfx}"].to_numpy()


def ribbon(ax, x0, x1, a0, a1, b0, b1, color, alpha=0.55):
    xs = np.linspace(x0, x1, 80)
    s = (1 - np.cos(np.pi * (xs - x0) / (x1 - x0))) / 2
    ax.fill_between(xs, a0 + (b0 - a0) * s, a1 + (b1 - a1) * s, color=color, alpha=alpha, lw=0)


def alluvial(ax, t, p, second, cut=0.5):
    ok, acc = t == p, second >= cut
    n = len(t)
    nc, nw = ok.sum(), (~ok).sum()
    ac, wc = (ok & acc).sum(), (ok & ~acc).sum()
    aw, cw = (~ok & acc).sum(), (~ok & ~acc).sum()
    g = 0.035 * n
    # left stack (top = high y)
    top = n + 3 * g
    L_ok = (top - nc, top); L_bad = (L_ok[0] - g - nw, L_ok[0] - g)
    # right stack: accepted (correct, wrong), gap, undecided (wrong held back, correct held back)
    R_ac = (top - ac, top); R_aw = (R_ac[0] - aw, R_ac[0])
    R_cw = (R_aw[0] - 2 * g - cw, R_aw[0] - 2 * g); R_wc = (R_cw[0] - wc, R_cw[0])
    xl, xr, w = 0.0, 1.0, 0.06
    for (lo, hi), c in ((L_ok, C_L_OK), (L_bad, C_L_BAD)):
        ax.add_patch(plt.Rectangle((xl - w, lo), w, hi - lo, fc=c, ec="none"))
    for (lo, hi), c in ((R_ac, C_OK), (R_aw, C_BAD), (R_cw, C_CAU), (R_wc, C_WIT)):
        ax.add_patch(plt.Rectangle((xr, lo), w, hi - lo, fc=c, ec="none"))
    # flows: correct -> accepted correct (top part), correct -> held back (bottom part); wrong likewise
    ribbon(ax, xl, xr, L_ok[1] - ac, L_ok[1], R_ac[0], R_ac[1], C_OK)
    ribbon(ax, xl, xr, L_ok[0], L_ok[0] + wc, R_wc[0], R_wc[1], C_WIT, 0.7)
    ribbon(ax, xl, xr, L_bad[1] - aw, L_bad[1], R_aw[0], R_aw[1], C_BAD)
    ribbon(ax, xl, xr, L_bad[0], L_bad[0] + cw, R_cw[0], R_cw[1], C_CAU, 0.7)
    fs = S.FS["annot"]
    ax.text(xl - w - 0.03, np.mean(L_ok), f"{nc:,}\ncorrect", ha="right", va="center", fontsize=fs, color=C_L_OK)
    ax.text(xl - w - 0.03, np.mean(L_bad), f"{nw:,}\nwrong", ha="right", va="center", fontsize=fs, color=C_L_BAD)
    for (lo, hi), c, v in ((R_ac, C_OK, ac), (R_aw, C_BAD, aw), (R_cw, "#b0705f", cw), (R_wc, "#8a74ab", wc)):
        ax.text(xr + w + 0.03, (lo + hi) / 2, f"{v:,}", ha="left", va="center", fontsize=fs, color=c)
    # brackets for accepted / undecided
    for (lo, hi), lab in (((R_aw[0], R_ac[1]), f"accepted\n{100 * ac / max(ac + aw, 1):.1f}% correct"),
                          ((R_wc[0], R_cw[1]), "undecided")):
        xb = xr + w + 0.30
        ax.plot([xb, xb], [lo, hi], color=S.GREY, lw=1.6)
        ax.text(xb + 0.04, (lo + hi) / 2, lab, ha="left", va="center", fontsize=fs - 1, color=S.INK)
    ax.text(xl - w / 2, top + 0.6 * g, "Step 1", ha="center", va="bottom", fontsize=fs, color=S.INK)
    ax.text(xr + w / 2, top + 0.6 * g, "+ step 2", ha="center", va="bottom", fontsize=fs, color=S.INK)
    ax.set_xlim(-0.55, 1.75); ax.set_ylim(R_wc[0] - g, top + 2.2 * g); ax.axis("off")


def within_cancer_keep(pred, conf, gate):
    keep = np.zeros(len(pred), bool)
    for c in np.unique(pred):
        ii = np.flatnonzero(pred == c)
        keep[ii[np.argsort(-conf[ii], kind="stable")[:int(gate[ii].sum())]]] = True
    return keep


def risk_coverage(ax, t, p, conf, second, cuts=(0.5,)):
    ok = t == p
    n = len(t)
    S.frame(ax); ax.set_facecolor(S.tint("#6a3d9a", 0.06))
    k = np.arange(1, n + 1)
    for score, ls in ((conf, "-"), (second, "--")):
        o = np.argsort(-score, kind="stable")
        ax.plot(100 * k / n, 100 * np.cumsum(~ok[o]) / k, ls=ls, color=S.PURPLE_DARK, lw=2.6)
    ymax = 0
    for cut in cuts:
        a = second >= cut
        x, y = 100 * a.mean(), 100 * (~ok[a]).mean()
        o = np.argsort(-conf, kind="stable")[: a.sum()]
        ys = 100 * (~ok[o]).mean()
        keep = within_cancer_keep(p, conf, a)
        yw = 100 * (~ok[keep]).mean()
        ax.plot(x, ys, "s", ms=13, mfc="white", mec=S.PURPLE_DARK, mew=2.2, zorder=6)
        ax.plot(x, yw, "^", ms=13, mfc="white", mec=S.PURPLE_DARK, mew=2.2, zorder=6)
        ax.plot(x, y, "o", ms=24, mfc=S.PURPLE, mec="white", mew=1.4, zorder=7)
        ax.text(x, y, "2", ha="center", va="center", color="white", fontsize=13, fontweight="bold", zorder=8)
        if len(cuts) > 1:
            ax.annotate(f"{cut:.2f}", (x, y), xytext=(0, 16), textcoords="offset points", ha="center", fontsize=12, color=S.PURPLE_DARK)
        ymax = max(ymax, y, ys, yw)
    full = 100 * (~ok).mean()
    ax.set_xlim(0, 101); ax.set_ylim(0, max(full * 1.08, ymax * 1.6))
    ax.set_xlabel("Patients accepted (%)", fontsize=S.FS["label"] - 1)
    ax.set_ylabel("Wrong among accepted (%)", fontsize=S.FS["label"] - 1)
    ax.grid(color="#e6e0ef"); ax.set_axisbelow(True)


def per_cancer(ax, t, p, second, cut=0.5):
    ok, acc = t == p, second >= cut
    present = [S.NAME[c] for c in S.ORDER if S.NAME[c] in set(p)]
    S.frame(ax)
    lows = []
    for i, c in enumerate(present):
        m = p == c
        p1 = 100 * ok[m].mean()
        p2 = 100 * ok[m & acc].mean() if (m & acc).any() else np.nan
        col = COLD(c)
        ax.axvspan(i - 0.42, i + 0.42, color=S.tint(c if c != "DLBCL" else "LYMPH", 0.10), lw=0, zorder=0)
        if np.isfinite(p2):
            ax.annotate("", xy=(i, p2), xytext=(i, p1), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2, mutation_scale=16), zorder=3)
        ax.plot(i, p1, "o", ms=11, mfc="white", mec=col, mew=2.2, zorder=4)
        if np.isfinite(p2):
            ax.plot(i, p2, "o", ms=12, color=col, zorder=5)
        lows.append(p1)
    ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=45, ha="right", fontsize=S.FS["tick"] - 2)
    for tl, c in zip(ax.get_xticklabels(), present):
        tl.set_color(COLD(c))
    lo = max(0, np.floor(min(lows) / 10) * 10 - 5)
    ax.set_ylim(lo, 101); ax.set_xlim(-0.6, len(present) - 0.4)
    ax.set_ylabel("Correct among calls (%)", fontsize=S.FS["label"] - 1)
    ax.grid(axis="y", color="#eeeeee"); ax.set_axisbelow(True)


def nb_row(fig, spec):
    d = pd.read_csv(R / "B5_internal_per_patient.csv")
    nb = pd.read_csv(R / "B5_net_benefit_per_cancer.csv")
    n = len(d); acc = d.second.to_numpy() >= 0.5; cor = d.correct.to_numpy().astype(bool)
    pts = np.linspace(0.05, 0.85, 81); w = pts / (1 - pts)
    sub = spec.subgridspec(1, 12, wspace=0.12)
    first = None
    for j, c in enumerate(S.ORDER):
        ax = fig.add_subplot(sub[j]); S.frame(ax)
        m = d.pred.to_numpy() == c
        tl, fr = (m & ~acc & cor).sum(), (m & ~acc & ~cor).sum()
        y = 100 * (fr * w - tl) / n
        col = S.COL[c]
        ax.axhline(0, color=col, lw=2.6); ax.plot(pts, y, ls="--", color=col, lw=2.2)
        ax.fill_between(pts, 0, y, where=y > 0, color=col, alpha=0.75, lw=0)
        ax.fill_between(pts, 0, y, where=y < 0, facecolor="none", edgecolor="#777777", hatch="////", lw=0)
        r = nb[nb.cancer == c].iloc[0]
        for tt in (0.5, 0.8):
            mid, lo, hi = r[f"nb_diff_{tt}"], r[f"nb_lo_{tt}"], r[f"nb_hi_{tt}"]
            ax.errorbar(tt, mid, yerr=[[mid - lo], [hi - mid]], fmt="o", ms=5, color="black", capsize=4, lw=1.5, zorder=5)
        ax.set_xlim(0.03, 0.87); ax.set_ylim(-7, 21); ax.set_xticks([0.2, 0.5, 0.8])
        ax.set_title(S.NAME[c], color=col, fontsize=S.FS["title"], pad=10)
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Change in net\nbenefit per 100", fontsize=S.FS["label"])
            ax.set_xlabel("Decision threshold", fontsize=S.FS["label"] - 1)
            first = ax
    return first


def block_rows(names, with_nb, letters, stem):
    rows = len(names)
    hr = [1.0] * rows + ([0.95] if with_nb else [])
    fig = plt.figure(figsize=(32, 6.2 * rows + (5.6 if with_nb else 0) + 1.2))
    gs = fig.add_gridspec(len(hr), 1, height_ratios=hr, hspace=0.42, left=0.07, right=0.985, top=0.96, bottom=0.07)
    for r, key in enumerate(names):
        label, path = {k: (lab, pth) for k, lab, pth in SERIES}[key]
        t, p, conf, second = load(path)
        sub = gs[r].subgridspec(1, 3, width_ratios=W_RATIOS, wspace=0.22)
        a1, a2, a3 = (fig.add_subplot(sub[i]) for i in range(3))
        alluvial(a1, t, p, second)
        risk_coverage(a2, t, p, conf, second, cuts=(0.5, 0.8) if key == "internal" else (0.5,))
        per_cancer(a3, t, p, second)
        box = a1.get_position()
        fig.text(0.012, (box.y0 + box.y1) / 2, f"{label}", rotation=90, ha="center", va="center", fontsize=S.FS["group"], fontweight="bold")
        fig.text(a1.get_position().x0, box.y1 + 0.004, SUB[key], ha="left", va="bottom", fontsize=S.FS["annot"], color=S.GREY)
        if key == "internal":
            for ax, s_ in zip((a1, a2, a3), letters[:3]):
                S.letter(fig, ax, s_, dx=0.0, dy=0.02)
        else:
            S.letter(fig, a1, letters[r], dx=0.0, dy=0.02)
    if with_nb:
        first = nb_row(fig, gs[rows])
        S.letter(fig, first, letters[3], dx=-0.012, dy=0.035)
    p = BUILD / f"{stem}.png"
    fig.savefig(p, dpi=250, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def legend_strip():
    fig = plt.figure(figsize=(32, 1.6))
    h = [Patch(fc=C_L_OK, label="Step 1 correct"), Patch(fc=C_L_BAD, label="Step 1 wrong"),
         Patch(fc=C_OK, label="Accepted, correct"), Patch(fc=C_BAD, label="Accepted, wrong"),
         Patch(fc=C_CAU, label="Wrong call held back"), Patch(fc=C_WIT, label="Correct call held back"),
         Line2D([], [], color=S.PURPLE_DARK, lw=2.6, label="Accepted by step-1 probability"),
         Line2D([], [], color=S.PURPLE_DARK, lw=2.6, ls="--", label="Accepted by step-2 probability"),
         Line2D([], [], ls="none", marker="o", ms=16, color=S.PURPLE, label="Step 2 (0.50)"),
         Line2D([], [], ls="none", marker="s", ms=12, mfc="white", mec=S.PURPLE_DARK, mew=2, label="Stricter cutoff, same number"),
         Line2D([], [], ls="none", marker="^", ms=12, mfc="white", mec=S.PURPLE_DARK, mew=2, label="Stricter cutoff within cancer"),
         Line2D([], [], ls="none", marker="o", ms=11, mfc="white", mec=S.INK, mew=2, label="Correct among calls, step 1"),
         Line2D([], [], ls="none", marker="o", ms=11, color=S.INK, label="... among accepted calls")]
    fig.legend(handles=h, loc="center", ncol=7, frameon=False, fontsize=S.FS["legend"] - 2)
    p = BUILD / "legend.png"
    fig.savefig(p, dpi=250, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def tissue_block():
    spec = importlib.util.spec_from_file_location("f9ab", S.V7 / "figures/scripts/fig9ab_v7_two_step_tissue.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.BUILD = BUILD
    m.REPS['OUT = ROOT / "overleaf-source-git/figures_300/fig10_v53_two_step_mosaic.png"'] = f'OUT = Path(r"{BUILD / "tissue.png"}")'
    m.REPS['letters = iter("ABCD")'] = 'letters = iter("EFGH")'
    m.main()
    return BUILD / "tissue.png"


def stack(paths, out):
    ims = [Image.open(p).convert("RGB") for p in paths]
    w = max(i.width for i in ims)
    ims = [i.resize((w, round(i.height * w / i.width)), Image.LANCZOS) for i in ims]
    gap = int(0.012 * w)
    o = Image.new("RGB", (w, sum(i.height for i in ims) + gap * (len(ims) - 1)), "white")
    y = 0
    for i in ims:
        o.paste(i, (0, y)); y += i.height + gap
    o.save(out, optimize=True)
    return o.size


def main():
    b1 = block_rows(["internal"], True, "ABCD", "internal")
    b2 = tissue_block()
    b3 = block_rows(["tcga", "expo", "gdc", "ccle"], False, "GHIJ", "mixed")
    lg = legend_strip()
    p = S.OUT / "fig5_v7c_two_step.png"
    print("wrote", p, stack([b1, b2, b3, lg], p))


if __name__ == "__main__":
    main()
