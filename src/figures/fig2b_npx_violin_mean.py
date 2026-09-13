#!/usr/bin/env python3
"""
fig2b_npx_violin_mean.py — Per-sample MEAN NPX distribution per cancer
Mean-variant of fig2b_npx_violin.py: for each sample, compute the MEAN NPX
across all 1,463 proteins -> one scalar (instead of the median). Same
violin + embedded box + jittered strip construction, for direct comparison
against the median version.
Output: figurev5/output/fig2b_npx_violin_mean.pdf/.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import pandas as pd
import numpy as np

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

def lighter(hexcol, blend=0.50):
    r, g, b = mc.to_rgb(hexcol)
    return (r + (1-r)*blend, g + (1-g)*blend, b + (1-b)*blend)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 13, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.linewidth": 1.3, "xtick.major.width": 1.3, "ytick.major.width": 1.3,
})

print("Loading data...")
df = pd.read_csv(DATA)
prot_cols = [c for c in df.columns if c not in _NON_FEATURE]

# Per-sample MEAN NPX across all 1,463 proteins
df["mean_npx"] = df[prot_cols].mean(axis=1)

data_per_cancer = []
for c in CANCERS:
    vals = df[df["Cancer"] == c]["mean_npx"].dropna().values
    data_per_cancer.append(vals)
    print(f"  {c}: n={len(vals)}  mean={np.mean(vals):.3f}  "
          f"IQR=[{np.percentile(vals,25):.3f},{np.percentile(vals,75):.3f}]")

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_facecolor("#fafafa")
xpos = np.arange(len(CANCERS))

# Violin
parts = ax.violinplot(data_per_cancer, positions=xpos,
                      widths=0.72, showmedians=False, showextrema=False)
for pc, c in zip(parts["bodies"], CANCERS):
    pc.set_facecolor(CANCER_COLOR[c])
    pc.set_alpha(0.65)
    pc.set_edgecolor("none")

# Embedded box (narrow, dark)
bp = ax.boxplot(data_per_cancer, positions=xpos, widths=0.16,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="white", linewidth=2.5),
                boxprops=dict(facecolor="#333333", alpha=0.85),
                whiskerprops=dict(color="#555555", linewidth=1.3),
                capprops=dict(color="#555555", linewidth=1.3))

# Jittered strip of individual sample dots
rng = np.random.default_rng(42)
for xi, (c, vals) in enumerate(zip(CANCERS, data_per_cancer)):
    jitter = rng.uniform(-0.22, 0.22, size=len(vals))
    ax.scatter(xi + jitter, vals,
               color=CANCER_COLOR[c], s=14, alpha=0.35,
               linewidths=0, zorder=1)

# X-axis cancer labels with abbreviation
ax.set_xticks(xpos)
ax.set_xticklabels(
    [cancer_label(c) for c in CANCERS],
    fontsize=10, fontweight="bold", rotation=32, ha="right",
)
for tick, c in zip(ax.get_xticklabels(), CANCERS):
    tick.set_color(CANCER_COLOR[c])

ax.set_ylabel("Per-sample mean NPX  (log2 arbitrary unit)",
              fontsize=13, fontweight="bold", labelpad=6)
ax.set_title(
    "Global plasma protein abundance per cancer type\n"
    "Mean NPX across 1,463 proteins per sample · 1,375 samples",
    fontsize=15, fontweight="bold", pad=10,
)
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=12)

ax.axhline(0, color="#999999", lw=1.0, ls="--", alpha=0.6, zorder=0)
ax.text(len(CANCERS) - 0.5, ax.get_ylim()[0] * 0.02, "NPX = 0  (log2 baseline)",
        fontsize=10, color="#999999", fontstyle="italic", ha="right")

ax.grid(axis="y", color="#e8e8e8", lw=0.9, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
for ext in ("pdf", "png"):
    p = OUT / f"fig2b_npx_violin_mean.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"saved -> {p}")
plt.close(fig)
print("done B-violin-mean.")
