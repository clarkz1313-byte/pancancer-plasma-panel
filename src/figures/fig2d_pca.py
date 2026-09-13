#!/usr/bin/env python3
"""
fig2d_pca.py — PCA of all 1,375 plasma samples × 1,463 proteins
PC1 vs PC2 with 95% confidence ellipses per cancer type.
Individual sample dots kept at low alpha for density context.
Cancer code labels at centroids.
Output: figurev5/output/fig2d_pca.pdf/.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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


def confidence_ellipse(ax, x, y, color, n_std=2.0, alpha_fill=0.13, lw=2.2):
    """Draw a covariance-based confidence ellipse (n_std standard deviations)."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals  = np.maximum(vals[order], 0)
    vecs  = vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h  = 2 * n_std * np.sqrt(vals)
    ell   = Ellipse(xy=(np.mean(x), np.mean(y)),
                    width=w, height=h, angle=angle,
                    facecolor=color, alpha=alpha_fill,
                    edgecolor=color, linewidth=lw, zorder=3)
    ax.add_patch(ell)


plt.rcParams.update({
    "font.family": "Arial", "font.size": 13, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.linewidth": 1.3, "xtick.major.width": 1.3, "ytick.major.width": 1.3,
})

print("Loading data…")
df = pd.read_csv(DATA)
prot_cols = [c for c in df.columns if c not in _NON_FEATURE]
X = df[prot_cols].values
y = df["Cancer"].values

print("Imputing -> Scaling -> PCA...")
X_imp = SimpleImputer(strategy="mean").fit_transform(X)
X_sc  = StandardScaler().fit_transform(X_imp)
pca   = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_sc)
ev    = pca.explained_variance_ratio_ * 100
print(f"  PC1={ev[0]:.1f}%  PC2={ev[1]:.1f}%")

print("Plotting…")
fig, ax = plt.subplots(figsize=(9, 8))
ax.set_facecolor("#fafafa")

# Scatter: light dots for density context
for c in CANCERS:
    mask = y == c
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=CANCER_COLOR[c], s=18, alpha=0.25,
               linewidths=0, rasterized=True, zorder=2)

# 95% confidence ellipses (2 std dev ≈ 95% for 1D; ~87% for 2D bivariate normal)
for c in CANCERS:
    mask = y == c
    confidence_ellipse(ax, coords[mask, 0], coords[mask, 1],
                       color=CANCER_COLOR[c], n_std=2.0, alpha_fill=0.13, lw=2.2)

# Centroid labels
for c in CANCERS:
    mask = y == c
    cx = coords[mask, 0].mean()
    cy = coords[mask, 1].mean()
    ax.text(cx, cy, c,
            fontsize=10, fontweight="bold",
            ha="center", va="center", color="white", zorder=5,
            path_effects=[pe.withStroke(linewidth=3.2, foreground=CANCER_COLOR[c])])

ax.set_xlabel(f"PC1  ({ev[0]:.1f}% variance explained)", fontsize=14, fontweight="bold", labelpad=6)
ax.set_ylabel(f"PC2  ({ev[1]:.1f}% variance explained)", fontsize=14, fontweight="bold", labelpad=6)
ax.set_title(
    "PCA of plasma proteomics — 1,375 samples\n"
    "1,463 proteins · Z-scored · 95% confidence ellipses per cancer type",
    fontsize=15, fontweight="bold", pad=10,
)
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(labelsize=12)

legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=CANCER_COLOR[c],
           markersize=9, label=cancer_label(c), markeredgewidth=0)
    for c in CANCERS
]
ax.legend(handles=legend_handles, fontsize=10, framealpha=0.0,
          loc="upper right", ncol=2, handletextpad=0.4)

ax.grid(color="#e8e8e8", lw=0.7, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
for ext in ("pdf", "png"):
    p = OUT / f"fig2d_pca.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"saved -> {p}")
plt.close(fig)
print("done D.")
