"""v7 Figure 2, revision c (17 Oct): bigger cohort donut (outer thin development/test ring, inner thick cancer
ring started at 45 degrees), bigger per-cancer donuts, original RdBu_r heatmap scale, bolder UMAP points.
Revision b (author review 17 Oct). As fig2_v7_cohort.py, with:
(A) donuts instead of the bar chart: one large two-ring donut for the cohort (outer ring = share of each cancer, inner
    ring = development vs internal test, 1,375 in the centre) and one donut per cancer (development dark, test light,
    total in the centre, development · test counts below);
(B) a softer blue-white-red scale;
(C) the two strips in two light greys, the overlap as "Overlap 91%" in the curve area, each strip with the range of
    its 12 values, no all-sample density curve, no horizontal grid;
(D) unchanged.
Counts in A come from the seed-52 development/test split (asserted against the panel-A table of the earlier figure).
Output: figures/output/fig2_v7b_cohort.png
"""
import importlib.util

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import v7_style as S
import fig2_v7_cohort as F2

plt = S.plt
BUILD = S.V7 / "figures/build/fig2b"
SOFT = mcolors.LinearSegmentedColormap.from_list("soft_rdbu", ["#4f78a8", "#a9c1dc", "#f7f7f7", "#e8aaa3", "#c0605a"], N=256)


def counts():
    raw = pd.read_csv(S.ROOT / "data/processed/filtered_pancancer_data.csv", usecols=["Sample_ID", "Cancer"])
    tr, te = train_test_split(np.arange(len(raw)), test_size=0.30, random_state=52, stratify=raw.Cancer)
    dev = raw.Cancer.iloc[tr].value_counts(); test = raw.Cancer.iloc[te].value_counts()
    assert dev.sum() == 962 and test.sum() == 413
    return {c: (int(dev[c]), int(test[c])) for c in S.ORDER}


def light(c, f=0.42):
    r = np.array(mcolors.to_rgb(c))
    return tuple(r * f + (1 - f))


def panel_a(path):
    n = counts()
    tot = sum(a + b for a, b in n.values())
    fig = plt.figure(figsize=(24, 7.6))
    gs = fig.add_gridspec(2, 8, width_ratios=[2.5, 0.30, 1, 1, 1, 1, 1, 1], hspace=0.18, wspace=0.04,
                          left=0.01, right=0.99, top=0.93, bottom=0.04)
    ax = fig.add_subplot(gs[:, 0])
    sizes = [sum(n[c]) for c in S.ORDER]
    dv, ts = sum(a for a, _ in n.values()), sum(b for _, b in n.values())
    # outer thin ring: development vs internal test; inner thick ring: cancers, started at a different angle so the
    # two rings do not share a boundary at 12 o'clock
    ax.pie([dv, ts], colors=["#5a5a5a", "#c4c4c4"], radius=1.0, startangle=90, counterclock=False,
           wedgeprops=dict(width=0.13, edgecolor="white", linewidth=2))
    ax.pie(sizes, colors=[S.COL[c] for c in S.ORDER], radius=0.80, startangle=45, counterclock=False,
           wedgeprops=dict(width=0.32, edgecolor="white", linewidth=2))
    ax.text(0, 0.06, f"{tot:,}", ha="center", va="center", fontsize=36, fontweight="bold", color=S.INK)
    ax.text(0, -0.17, "patients", ha="center", va="center", fontsize=19, color=S.GREY)
    for val, start, col_ in ((dv, 0, "#4a4a4a"), (ts, dv, "#9a9a9a")):   # outside the outer ring, as on the small donuts
        th = np.radians(90 - 360 * (start + val / 2) / (dv + ts))
        ax.text(1.17 * np.cos(th), 1.17 * np.sin(th), f"{val}", ha="center", va="center", fontsize=24,
                fontweight="bold", color=col_)
    ax.set_xlim(-1.32, 1.32); ax.set_ylim(-1.32, 1.32)
    ax.set_aspect("equal")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="#5a5a5a"), plt.Rectangle((0, 0), 1, 1, fc="#c4c4c4")],
              labels=["Development", "Internal test"], loc="center left", bbox_to_anchor=(0.86, 0.10), ncol=1,
              frameon=False, fontsize=16, handlelength=1.2)
    for i, c in enumerate(S.ORDER):
        a = fig.add_subplot(gs[i // 6, 2 + i % 6])
        d, t = n[c]
        a.pie([d, t], colors=[S.COL[c], light(S.COL[c])], radius=1.0, startangle=90, counterclock=False,
              wedgeprops=dict(width=0.34, edgecolor="white", linewidth=2))
        a.text(0, 0, f"{d + t}", ha="center", va="center", fontsize=22, fontweight="bold", color=S.INK)
        a.set_title(S.NAME[c], color=S.COL[c], fontsize=21, fontweight="bold", pad=0)
        for val, start, col_ in ((d, 0, S.COL[c]), (t, d, light(S.COL[c], 0.75))):   # next to its own segment
            th = np.radians(90 - 360 * (start + val / 2) / (d + t))
            a.text(1.27 * np.cos(th), 1.27 * np.sin(th), f"{val}", ha="center", va="center", fontsize=18,
                   fontweight="bold", color=col_)
        a.set_xlim(-1.45, 1.45); a.set_ylim(-1.45, 1.30); a.set_aspect("equal")
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    src = F2.SRC.read_text(encoding="utf-8").replace('"c=CANCER_COLOR[c], s=26, alpha=0.55,"', '"c=CANCER_COLOR[c], s=34, alpha=0.70,"').replace('"c=CANCER_COLOR[c], s=26, alpha=0.85,"', '"c=CANCER_COLOR[c], s=36, alpha=0.88,"')
    reps = dict(F2.REPS)
    reps['BUILD = ROOT / "revise_plan/v5_seed52_common_split/figure_inputs/fig2_v51_build"'] = f'BUILD = Path(r"{BUILD}")'
    reps['LIVE = ROOT / "overleaf-source-git/figures_300/fig2_v51_cohort_mosaic.png"'] = f'LIVE = Path(r"{BUILD / "v51_copy.png"}")'
    for o, n_ in reps.items():
        assert src.count(o) == 1, o
        src = src.replace(o, n_)
    spec = importlib.util.spec_from_loader("fig2_v7b", loader=None)
    mod = importlib.util.module_from_spec(spec); mod.__file__ = str(F2.SRC)
    exec(compile(src, "make_fig2_v51_cohort (v7b)", "exec"), mod.__dict__)
    mod.main()                                             # all four panels as before (D is kept)
    out_line = 'OUT  = ROOT / "figurev6" / "output"'
    mod.run_patched("fig2b_npx_heatmap.py", {out_line: "OUT  = V5_BUILD",
                                             "fig, ax = plt.subplots(figsize=(14.0, 7.6))": "fig, ax = plt.subplots(figsize=(18.0, 5.6))",
                                             }, {})   # original RdBu_r scale (supervisor, 17 Oct)
    mod.run_patched("fig2b_pooled_distributions.py", {
        out_line: "OUT  = V5_BUILD",
        'ax.set_facecolor("#fafafa")': 'ax.set_facecolor("white")',
        "fig, ax = plt.subplots(figsize=(13, 6.2))": "fig, ax = plt.subplots(figsize=(18, 5.4))",
        'bbox_to_anchor=(0.5, -0.13),\n          ncol=8,': 'bbox_to_anchor=(0.5, -0.20),\n          ncol=15,',
        "ax.fill_between(centers, hist_pooled_all, color=\"black\", alpha=0.05, zorder=1)": "pass",
        "ax.plot(centers, hist_pooled_all, color=\"black\", lw=4.2, ls=\"-\",\n        alpha=0.95, zorder=5, label=\"All-sample density\")": "pass",
        'ax.axhspan(med_bot, med_top, color="#dce3ea", zorder=1, lw=0)': 'ax.axhspan(med_bot, med_top, color="#f1f1f1", zorder=1, lw=0)',
        'ax.axhspan(mean_bot, mean_top, color="#eae2da", zorder=1, lw=0)': 'ax.axhspan(mean_bot, mean_top, color="#e3e3e3", zorder=1, lw=0)',
        'ax.text(0.985, 0.96, f"Overlap: {mean_overlap:.0%}", transform=ax.transAxes,':
            'for _v, _y in ((median_per_cancer, (med_bot + med_top) / 2), (mean_per_cancer, (mean_bot + mean_top) / 2)):\n'
            '    ax.text(hi - 0.015 * (hi - lo), _y, f"range {min(_v.values()):.2f}–{max(_v.values()):.2f}", ha="right",\n'
            '            va="center", fontsize=12.5, fontweight="bold", color="#46515b", zorder=7)\n'
            'ax.text(0.985, 0.78, f"Overlap {mean_overlap:.0%}", transform=ax.transAxes,',
        'ax.grid(axis="y", color="#e2e2e2", lw=1.1, zorder=0)': "ax.grid(False)",
        'handles = [plt.Line2D([0], [0], color="black", lw=4.2, ls="-", alpha=0.95),\n          plt.Line2D':
            "handles = [plt.Line2D",
        'labels = ["All-sample density", "Pooled median", "Pooled mean"]': 'labels = ["Pooled median", "Pooled mean"]',
    })
    panel_a(BUILD / "fig2a_cohort.png")
    import build_slide_mosaics as bsm
    bsm.OUT = BUILD
    spec_ = dict(title=bsm.SLIDES["fig2_slide2_cohort_mosaic"]["title"], grid=(4, 1), label_band=104,
                 source_paths={k: BUILD / f"{k}.png" for k in
                               ("fig2a_cohort", "fig2b_npx_heatmap", "fig2b_pooled_distributions", "fig2f_umap")},
                 panels=[("fig2a_cohort", 0, 0, 1, 1), ("fig2b_npx_heatmap", 1, 0, 1, 1),
                         ("fig2b_pooled_distributions", 2, 0, 1, 1), ("fig2f_umap", 3, 0, 1, 1)])
    bsm.build("fig2_v7d_mosaic", spec_)
    im = Image.open(BUILD / "fig2_v7d_mosaic.png")
    p = S.OUT / "fig2_v7d_cohort.png"
    im.resize((3600, round(im.height * 3600 / im.width)), Image.LANCZOS).save(p, optimize=True, dpi=(300, 300))
    print("wrote", p)


if __name__ == "__main__":
    main()
