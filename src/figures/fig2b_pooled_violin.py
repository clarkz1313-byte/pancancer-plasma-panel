#!/usr/bin/env python3
"""
fig2b_pooled_violin.py — full protein-value distributions per cancer, violin style
Same underlying calculation as fig2b_pooled_distributions.py (all 1,463 proteins
x all samples pooled per cancer, same histogram bins/smoothing, same mean
pairwise overlap statistic) -- just re-encoded as a violin (one per cancer,
x-axis = cancer, y-axis = NPX) instead of overlaid density lines, for a
side-by-side comparison against the per-sample-summary violin (fig2b_npx_violin).
Output: figurev5/output/fig2b_pooled_violin.pdf/.png
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


ROOT = Path("e:/Proteomics")
DATA = ROOT / "data" / "processed" / "filtered_pancancer_data.csv"
OUT  = ROOT / "figurev5" / "output"
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
    "font.family": "Arial", "font.size": 13, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.linewidth": 1.3, "xtick.major.width": 1.3, "ytick.major.width": 1.3,
})

print("Loading data...")
df = pd.read_csv(DATA)
prot_cols = [c for c in df.columns if c not in _NON_FEATURE]

all_vals = df[prot_cols].values.astype(float).ravel()
all_vals = all_vals[~np.isnan(all_vals)]
lo, hi = np.percentile(all_vals, [0.5, 99.5])
n_bins = 140
bins = np.linspace(lo, hi, n_bins + 1)
bin_w = bins[1] - bins[0]
centers = 0.5 * (bins[:-1] + bins[1:])

raw_per_cancer = {}
hist_per_cancer = {}
for c in CANCERS:
    vals = df[df["Cancer"] == c][prot_cols].values.astype(float).ravel()
    vals = vals[~np.isnan(vals)]
    raw_per_cancer[c] = vals
    h, _ = np.histogram(vals, bins=bins, density=True)
    hist_per_cancer[c] = gaussian_filter1d(h, sigma=1.6)
    print(f"  {c}: {len(vals):,} pooled measurements")

# identical calculation to fig2b_pooled_distributions.py
pairwise = []
for i in range(len(CANCERS)):
    for j in range(i + 1, len(CANCERS)):
        hi_, hj_ = hist_per_cancer[CANCERS[i]], hist_per_cancer[CANCERS[j]]
        pairwise.append(np.sum(np.minimum(hi_, hj_)) * bin_w)
mean_overlap = float(np.mean(pairwise))
print(f"\nMean pairwise distributional overlap: {mean_overlap:.1%}")

fig, ax = plt.subplots(figsize=(14, 7.5))
ax.set_facecolor("#fafafa")
xpos = np.arange(len(CANCERS))

width_max = 0.42
box_w = 0.10
for xi, c in enumerate(CANCERS):
    dens_norm = hist_per_cancer[c] / hist_per_cancer[c].max() * width_max
    col = CANCER_COLOR[c]

    ax.fill_betweenx(centers, xi - dens_norm, xi + dens_norm,
                     color=col, alpha=0.62, linewidth=0, zorder=2)
    ax.plot(xi - dens_norm, centers, color=col, lw=0.9, alpha=0.9, zorder=3)
    ax.plot(xi + dens_norm, centers, color=col, lw=0.9, alpha=0.9, zorder=3)

    # embedded box from the RAW pooled values (not the smoothed histogram)
    q1, med, q3 = np.percentile(raw_per_cancer[c], [25, 50, 75])
    iqr = q3 - q1
    w_lo = max(raw_per_cancer[c].min(), q1 - 1.5 * iqr)
    w_hi = min(raw_per_cancer[c].max(), q3 + 1.5 * iqr)
    ax.plot([xi, xi], [w_lo, q1], color="#555555", lw=1.3, zorder=4)
    ax.plot([xi, xi], [q3, w_hi], color="#555555", lw=1.3, zorder=4)
    ax.add_patch(plt.Rectangle((xi - box_w, q1), 2 * box_w, q3 - q1,
                               facecolor="#333333", alpha=0.85, zorder=4))
    ax.plot([xi - box_w, xi + box_w], [med, med], color="white", lw=2.3, zorder=5)

ax.axhline(np.median(all_vals), color="#999999", lw=1.2, ls="--", alpha=0.7, zorder=1)

ax.set_xticks(xpos)
ax.set_xticklabels([cancer_label(c) for c in CANCERS],
                   fontsize=10, fontweight="bold", rotation=32, ha="right")
for tick, c in zip(ax.get_xticklabels(), CANCERS):
    tick.set_color(CANCER_COLOR[c])

ax.set_xlim(-0.6, len(CANCERS) - 0.4)
ax.set_ylim(lo, hi)
ax.set_ylabel("NPX  (log2 arbitrary unit)", fontsize=13, fontweight="bold", labelpad=6)
ax.set_title(
    "Full protein-value distributions per cancer type (violin)\n"
    "All 1,463 proteins x all samples per cancer, pooled — not summarized to one number per sample\n"
    f"Mean pairwise distributional overlap across all 12 cancers = {mean_overlap:.0%}  (same calculation as the overlay version)",
    fontsize=13.5, fontweight="bold", pad=12,
)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e8e8e8", lw=0.9, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="y", labelsize=11)

fig.tight_layout()
for ext in ("pdf", "png"):
    p = OUT / f"fig2b_pooled_violin.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"saved -> {p}")
plt.close(fig)
print("done.")
