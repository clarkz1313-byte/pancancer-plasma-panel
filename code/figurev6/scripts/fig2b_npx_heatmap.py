#!/usr/bin/env python3
"""
fig2b_npx_heatmap.py — NPX expression spectrum heatmap
Z-scored mean NPX per cancer × all 1,463 proteins.
Proteins sorted by argmax cancer (blocks only, no secondary Z sort) to avoid
the red→white gradient artifact seen when stacking by descending Z within block.
Cancer code labels annotated above each block on the x-axis.
Output: figurev5/output/fig2b_npx_heatmap.pdf/.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
from scipy.stats import zscore

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
    return "DLBCL" if code == "LYMPH" else code

plt.rcParams.update({
    "font.family": "Arial", "font.size": 13, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.linewidth": 1.3,
})

print("Loading data…")
df = pd.read_csv(DATA)
prot_cols = [c for c in df.columns if c not in _NON_FEATURE]
print(f"  {len(df)} samples × {len(prot_cols)} proteins")

print("Computing per-cancer mean NPX…")
mean_mat = df.groupby("Cancer")[prot_cols].mean().reindex(CANCERS)   # 12 × 1464

print("Z-scoring per protein across cancers…")
Z_vals = zscore(mean_mat.values, axis=0, nan_policy="omit")
Z = pd.DataFrame(Z_vals, index=CANCERS, columns=prot_cols).fillna(0)

print("Sorting proteins by argmax cancer (no secondary sort — avoids red gradient)…")
argmax_cancer = Z.values.argmax(axis=0)
# Sort ONLY by which cancer is highest — no descending-Z secondary sort.
# This prevents proteins from stacking brightest-first and creating a red→white gradient.
order = sorted(range(Z.shape[1]), key=lambda j: argmax_cancer[j])
Z_sorted    = Z.iloc[:, order].values        # 12 × 1464 numpy
argmax_ord  = argmax_cancer[order]           # for block boundaries

# ── block boundaries and centers ──────────────────────────────────────────────
block_edges  = []
block_starts = [0]
prev = argmax_ord[0]
for j in range(1, len(order)):
    if argmax_ord[j] != prev:
        block_edges.append(j - 0.5)
        block_starts.append(j)
        prev = argmax_ord[j]
block_starts.append(len(order))

# cancer order in which blocks appear (same as CANCERS since Z argmax follows row order)
block_cancer_idx = [argmax_ord[s] for s in block_starts[:-1]]
block_centers    = [(block_starts[i] + block_starts[i+1]) / 2
                    for i in range(len(block_starts) - 1)]

print("Plotting…")
fig, ax = plt.subplots(figsize=(14.0, 7.6))
im = ax.imshow(Z_sorted, aspect="auto", cmap="RdBu_r",
               vmin=-2.0, vmax=2.0, interpolation="nearest")

# Y-axis: cancer labels with abbreviation
ax.set_yticks(range(len(CANCERS)))
ax.set_yticklabels([cancer_label(c) for c in CANCERS], fontsize=15, fontweight="bold")
for tick, c in zip(ax.get_yticklabels(), CANCERS):
    tick.set_color(CANCER_COLOR[c])

# X-axis: suppress individual protein ticks
ax.set_xticks([])
ax.set_xlabel("1,463 proteins", fontsize=16, fontweight="bold", labelpad=12)

# Block boundary lines (white)
for xe in block_edges:
    ax.axvline(xe, color="white", lw=1.2, alpha=0.8)

# Cancer block labels omitted — viewers read cancer identity from the
# y-axis color bands and the corresponding high-NPX (red) block positions.

# Colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1.8%", pad=0.12)
cb  = fig.colorbar(im, cax=cax)
cb.set_label("Z-score\n(mean NPX)", fontsize=15, fontweight="bold", labelpad=4)
cb.ax.tick_params(labelsize=13)

ax.spines[["top","right","bottom","left"]].set_visible(False)

fig.tight_layout()
for ext in ("pdf", "png"):
    p = OUT / f"fig2b_npx_heatmap.{ext}"
    fig.savefig(p, dpi=400, bbox_inches="tight")
    print(f"saved -> {p}")
plt.close(fig)
print("done B-heatmap.")
