#!/usr/bin/env python3
"""
fig2b_pooled_distributions.py — full protein-value distributions per cancer
Every single NPX measurement (all 1,463 proteins x all samples) pooled per
cancer type, overlaid as density curves on one axis. Unlike fig2b_npx_violin
(which collapses each sample to ONE median number, letting outliers like
myeloma show through), this uses every data point with nothing summarized
away -- demonstrating that even at full resolution, the 12 cancer
distributions are nearly indistinguishable. This is the deliberate
"bulk data can't tell these apart" companion to the UMAP panel: it motivates
why the classification task needs a targeted ML approach rather than being
visible by eye in the raw data.
Output: figurev5/output/fig2b_pooled_distributions.pdf/.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

# NOTE: "protein_count" is a per-sample METADATA column (values 1206-1463),
# not an assayed protein. Excluding only Sample_ID/Cancer left it in the
# feature matrix, mixing values ~1500x the NPX scale (-9..+11) into the
# plotted data and into every derived statistic. The assay measured 1,463
# proteins, not 1,464.
_NON_FEATURE = ("Sample_ID", "Cancer", "protein_count")


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "filtered_pancancer_data.csv"
OUT  = ROOT / "figurev6" / "output"
OUT.mkdir(parents=True, exist_ok=True)

CANCERS = ["AML","BRC","CLL","CRC","CVX","ENDC","GLIOM","LUNGC","LYMPH","MYEL","OVC","PRC"]
CANCER_FULL = {
    "AML":"AML","BRC":"Breast Cancer","CLL":"CLL",
    "CRC":"Colorectal Cancer","CVX":"Cervical Cancer",
    "ENDC":"Endometrial Cancer","GLIOM":"Glioma",
    "LUNGC":"Lung Cancer","LYMPH":"DLBCL",
    "MYEL":"Myeloma","OVC":"Ovarian Cancer","PRC":"Prostate Cancer",
}
CANCER_COLOR = {
    "AML":"#b22222","BRC":"#c97b63","CLL":"#7a3e9d","CRC":"#d68600","CVX":"#c13d86",
    "ENDC":"#8e5d2c","GLIOM":"#1a8db8","LUNGC":"#2e8b57","LYMPH":"#3856a6",
    "MYEL":"#8c564b","OVC":"#d1495b","PRC":"#008b8b",
}

def cancer_label(code):
    full = CANCER_FULL[code]
    if code == "LYMPH":
        return "DLBCL"
    return full if full == code else f"{full} ({code})"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 15, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.linewidth": 1.8, "xtick.major.width": 1.8, "ytick.major.width": 1.8,
})

print("Loading data...")
df = pd.read_csv(DATA)
prot_cols = [c for c in df.columns if c not in _NON_FEATURE]

all_vals = df[prot_cols].values.astype(float).ravel()
all_vals = all_vals[~np.isnan(all_vals)]
# trim symmetric extreme tails (0.5/99.5 pct) for a readable shared x-range --
# applied identically to every cancer, so it doesn't distort relative shape
lo, hi = np.percentile(all_vals, [0.5, 99.5])
n_bins = 140
bins = np.linspace(lo, hi, n_bins + 1)
bin_w = bins[1] - bins[0]
centers = 0.5 * (bins[:-1] + bins[1:])

hist_pooled_all, _ = np.histogram(all_vals, bins=bins, density=True)
hist_pooled_all = gaussian_filter1d(hist_pooled_all, sigma=1.6)

hist_per_cancer = {}
n_per_cancer = {}
median_per_cancer = {}
mean_per_cancer = {}
for c in CANCERS:
    vals = df[df["Cancer"] == c][prot_cols].values.astype(float).ravel()
    vals = vals[~np.isnan(vals)]
    h, _ = np.histogram(vals, bins=bins, density=True)
    # light smoothing -- raw per-bin histogram noise (n bins x 12 cancers) was
    # making the curves look jagged/uneven, distracting from the actual
    # message (they overlap); a small Gaussian smooth on the density curve
    # cleans this up without changing the underlying shape or overlap number
    hist_per_cancer[c] = gaussian_filter1d(h, sigma=1.6)
    n_per_cancer[c] = len(vals)
    median_per_cancer[c] = float(np.median(vals))
    mean_per_cancer[c] = float(np.mean(vals))
    print(f"  {c}: {len(vals):,} pooled measurements")

# mean pairwise overlap coefficient across all 12 cancer distributions
# (NOT vs the pooled-all curve -- that would be circular, since every cancer
# contributes to it). Overlap = integral of min(density_i, density_j).
pairwise = []
for i in range(len(CANCERS)):
    for j in range(i + 1, len(CANCERS)):
        hi_, hj_ = hist_per_cancer[CANCERS[i]], hist_per_cancer[CANCERS[j]]
        pairwise.append(np.sum(np.minimum(hi_, hj_)) * bin_w)
mean_overlap = float(np.mean(pairwise))
print(f"\nMean pairwise distributional overlap (12 cancers, {len(pairwise)} pairs): {mean_overlap:.1%}")

fig, ax = plt.subplots(figsize=(13, 6.2))
fig.subplots_adjust(left=0.08, right=0.985, top=0.96, bottom=0.19)
ax.set_facecolor("#fafafa")

# Pool curve = black again (dotted style, not color, was the actual ask
# last round). Between median and mean, MEDIAN gets the one accent color:
# this distribution is heavily right-skewed by a long tail of high-abundance
# proteins, which drags the mean (1.94) away from where most values actually
# sit -- median (0.69) is the robust, representative "typical value," which
# is what the panel's argument actually depends on, so it's the one worth
# highlighting. Mean stays black/secondary.
MEDIAN_COLOR = "crimson"

ax.fill_between(centers, hist_pooled_all, color="black", alpha=0.05, zorder=1)

for c in CANCERS:
    ax.plot(centers, hist_per_cancer[c], color=CANCER_COLOR[c], lw=2.6,
            alpha=0.88, zorder=3)

# drawn AFTER (on top of) the colored cancer curves -- at zorder below them it
# was getting fully covered wherever the curves bunch up near the peak,
# exactly where a reader most needs to see it sitting in the middle of the pack.
ax.plot(centers, hist_pooled_all, color="black", lw=4.2, ls="-",
        alpha=0.95, zorder=5, label="All-sample density")

median_all = float(np.median(all_vals))
mean_all   = float(np.mean(all_vals))
ymax = float(max(hist_pooled_all.max(),
                 max(h.max() for h in hist_per_cancer.values())))

# median/mean already differentiated by color (crimson vs black) -- same
# dash style for both now; width cut 25% (2.2->1.65, 1.6->1.2)
ax.axvline(median_all, color=MEDIAN_COLOR, lw=1.65, ls="--", dashes=(6, 3), alpha=0.90, zorder=5)
ax.axvline(mean_all, color="black", lw=1.2, ls="--", dashes=(6, 3), alpha=0.65, zorder=5)

# Line-only descriptive addition, redesigned: the first version (ticks sitting
# right at y=0 on top of the curve fill) read as rendering noise/error bars,
# not deliberate marks. Now two dedicated shaded strips -- median on top,
# mean on the bottom -- each separated from the curve area by a visible gap
# and a baseline rule, so they unambiguously read as their own rows (mini rug
# plots), not an artifact of the density curves. Tick width bumped 30%
# (3.4 -> 4.4) per feedback. Each cancer's own pooled median/mean, in that
# cancer's own color, shows how tightly all 12 cluster on each statistic.
RUG_LW = 3.4 * 1.3
GAP    = ymax * 0.05
STRIP_H = ymax * 0.11

# top strip -- per-cancer median. Cool light blue-grey -- distinct from both
# the near-white curve-plot background (#fafafa) and the mean strip below,
# so neither strip reads as a continuation of the white plot area.
med_bot = ymax * 1.02 + GAP
med_top = med_bot + STRIP_H
ax.axhspan(med_bot, med_top, color="#dce3ea", zorder=1, lw=0)
ax.axhline(ymax * 1.02, color="#aaaaaa", lw=1.1, zorder=2)
for c in CANCERS:
    ax.plot([median_per_cancer[c]] * 2, [med_bot + STRIP_H * 0.18, med_top - STRIP_H * 0.18],
            color=CANCER_COLOR[c], lw=RUG_LW, alpha=0.95, zorder=6, solid_capstyle="butt")
ax.text(lo + 0.015 * (hi - lo), (med_bot + med_top) / 2, "Cancer medians",
        ha="left", va="center", fontsize=12.5, fontweight="bold",
        color="#46515b", zorder=7)

# bottom strip -- per-cancer mean. Warm light grey -- distinct from the
# cool blue-grey median strip above and from the white curve-plot background.
mean_top = -GAP * 0.4
mean_bot = mean_top - STRIP_H
ax.axhspan(mean_bot, mean_top, color="#eae2da", zorder=1, lw=0)
ax.axhline(0, color="#aaaaaa", lw=1.1, zorder=2)
for c in CANCERS:
    ax.plot([mean_per_cancer[c]] * 2, [mean_bot + STRIP_H * 0.18, mean_top - STRIP_H * 0.18],
            color=CANCER_COLOR[c], lw=RUG_LW, alpha=0.95, zorder=6, solid_capstyle="butt")
ax.text(lo + 0.015 * (hi - lo), (mean_bot + mean_top) / 2, "Cancer means",
        ha="left", va="center", fontsize=12.5, fontweight="bold",
        color="#5b5048", zorder=7)

ax.set_xlim(lo, hi)
ax.set_ylim(mean_bot - STRIP_H * 0.3, med_top + STRIP_H * 0.3)
ax.set_xlabel("NPX (log2)", fontsize=17, fontweight="bold", labelpad=8)
ax.set_ylabel("Density", fontsize=17, fontweight="bold", labelpad=8)
ax.text(0.985, 0.96, f"Overlap: {mean_overlap:.0%}", transform=ax.transAxes,
        ha="right", va="top", fontsize=17, fontweight="bold", color="#333333")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_linewidth(1.8)
# explicit density ticks only (0.0-0.6) -- the extended ylim for the two rug
# strips was pulling in an auto tick (0.7) that landed inside the median
# strip, which read as a stray density value rather than a strip label
ax.set_yticks(np.arange(0, 0.61, 0.1))
ax.grid(axis="y", color="#e2e2e2", lw=1.1, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="both", labelsize=15, width=1.8, length=6)

handles = [plt.Line2D([0], [0], color="black", lw=4.2, ls="-", alpha=0.95),
          plt.Line2D([0], [0], color=MEDIAN_COLOR, lw=1.65, ls="--", dashes=(6, 3), alpha=0.90),
          plt.Line2D([0], [0], color="black", lw=1.2, ls="--", dashes=(6, 3), alpha=0.65)] + \
          [plt.Line2D([0], [0], color=CANCER_COLOR[c], lw=3.0) for c in CANCERS]
labels = ["All-sample density", "Pooled median", "Pooled mean"] + ["DLBCL" if c == "LYMPH" else c for c in CANCERS]
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.13),
          ncol=8, fontsize=14, frameon=False, handlelength=1.5,
          handletextpad=0.35, labelspacing=0.35, columnspacing=0.8,
          borderaxespad=0.0)

for ext in ("pdf", "png"):
    p = OUT / f"fig2b_pooled_distributions.{ext}"
    fig.savefig(p, dpi=400, bbox_inches="tight")
    print(f"saved -> {p}")
plt.close(fig)
print("done.")
