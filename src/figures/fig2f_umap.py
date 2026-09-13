#!/usr/bin/env python3
"""
fig2f_umap.py — UMAP satellite figure
Left:  main UMAP of all 1,375 samples, colored by cancer type + 95% ellipses.
Right: 3x4 grid of 12 satellite panels — each highlights one cancer in color,
       all others shown as light grey background.
All panels share the same UMAP embedding coordinates.
Requires: umap-learn  (pip install umap-learn)
Output: figurev5/output/fig2f_umap.pdf/.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import umap

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

def main_label(code):
    return "DLBCL" if code == "LYMPH" else code

def confidence_ellipse(ax, x, y, color, n_std=2.0, alpha_fill=0.13, lw=2.0):
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
    "font.family": "Arial", "font.size": 12, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.linewidth": 1.2,
})

print("Loading data...")
df = pd.read_csv(DATA)
prot_cols = [c for c in df.columns if c not in _NON_FEATURE]
X = df[prot_cols].values
y = df["Cancer"].values
print(f"  {X.shape[0]} samples x {X.shape[1]} proteins")

print("Imputing -> Scaling...")
X_imp = SimpleImputer(strategy="mean").fit_transform(X)
X_sc  = StandardScaler().fit_transform(X_imp)

print("UMAP (n_neighbors=15, min_dist=0.15)...  [~60 s]")
reducer   = umap.UMAP(n_neighbors=15, min_dist=0.15,
                      n_components=2, random_state=42, verbose=False)
embedding = reducer.fit_transform(X_sc)
print("  done.")

# Fixed axis limits 0–16 on both axes to ensure all ellipses are fully visible.
# UMAP coordinates are arbitrary and can be negative. Normalize the display
# frame using the point cloud and ellipse extents so neither is clipped.
embedding = embedding - embedding.min(axis=0)
embedding *= 14.0 / embedding.max()

ellipse_bounds = []
for c in CANCERS:
    mask = y == c
    cov = np.cov(embedding[mask, 0], embedding[mask, 1])
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = np.maximum(vals[order], 0)
    vecs = vecs[:, order]
    angle = np.arctan2(vecs[1, 0], vecs[0, 0])
    semi_major, semi_minor = 2.0 * np.sqrt(vals)
    dx = np.sqrt((semi_major * np.cos(angle)) ** 2 +
                 (semi_minor * np.sin(angle)) ** 2)
    dy = np.sqrt((semi_major * np.sin(angle)) ** 2 +
                 (semi_minor * np.cos(angle)) ** 2)
    center = embedding[mask].mean(axis=0)
    ellipse_bounds.append((center[0] - dx, center[0] + dx,
                           center[1] - dy, center[1] + dy))

xmin = min(b[0] for b in ellipse_bounds)
xmax = max(b[1] for b in ellipse_bounds)
ymin = min(b[2] for b in ellipse_bounds)
ymax = max(b[3] for b in ellipse_bounds)
embedding[:, 0] = 0.5 + 14.0 * (embedding[:, 0] - xmin) / (xmax - xmin)
embedding[:, 1] = 0.5 + 14.0 * (embedding[:, 1] - ymin) / (ymax - ymin)
xlim = (0, 15)
ylim = (0, 15)

print("Plotting satellite figure...")
# Layout: main panel left (2.5 ratio), tiny spacer, 4 satellite columns (1.0 each)
# Satellite panels: bigger, minimal gap between boxes
fig = plt.figure(figsize=(21, 10))
gs  = GridSpec(3, 6, figure=fig,
               width_ratios=[2.5, 0.04, 1.0, 1.0, 1.0, 1.0])
fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.06,
                    hspace=0.08, wspace=0.11)

# ── Main panel (left, spanning all 3 rows) ───────────────────────────────────
ax_main = fig.add_subplot(gs[:, 0])
ax_main.set_facecolor("#fafafa")

for c in CANCERS:
    mask = y == c
    ax_main.scatter(embedding[mask, 0], embedding[mask, 1],
                    c=CANCER_COLOR[c], s=16, alpha=0.28,
                    linewidths=0, rasterized=True, zorder=2)

for c in CANCERS:
    mask = y == c
    confidence_ellipse(ax_main, embedding[mask, 0], embedding[mask, 1],
                       color=CANCER_COLOR[c], n_std=2.0, alpha_fill=0.13, lw=2.0)

for c in CANCERS:
    mask = y == c
    ax_main.text(embedding[mask, 0].mean(), embedding[mask, 1].mean(), main_label(c),
                 fontsize=9, fontweight="bold",
                 ha="center", va="center", color="white", zorder=5,
                 path_effects=[pe.withStroke(linewidth=3.0, foreground=CANCER_COLOR[c])])

ax_main.set_xlim(xlim)
ax_main.set_ylim(ylim)
ax_main.set_xlabel("UMAP-1", fontsize=13, fontweight="bold", labelpad=5)
ax_main.set_ylabel("UMAP-2", fontsize=13, fontweight="bold", labelpad=5)
ax_main.spines[["top","right"]].set_visible(False)
ax_main.tick_params(labelsize=10)
ax_main.grid(color="#e8e8e8", lw=0.7, zorder=0)
ax_main.set_axisbelow(True)

# ── 12 Satellite panels (3 rows × 4 cols, right side) ────────────────────────
for i, c in enumerate(CANCERS):
    row = i // 4
    col = (i % 4) + 2       # columns 2-5
    ax  = fig.add_subplot(gs[row, col])
    ax.set_facecolor("#f5f5f5")

    # All samples: light grey background
    ax.scatter(embedding[:, 0], embedding[:, 1],
               c="#c4c4c4", s=5, alpha=0.20, linewidths=0, rasterized=True, zorder=1)

    # Focal cancer: full color, larger
    mask = y == c
    ax.scatter(embedding[mask, 0], embedding[mask, 1],
               c=CANCER_COLOR[c], s=16, alpha=0.82,
               linewidths=0, rasterized=True, zorder=2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(main_label(c), fontsize=9.5, fontweight="bold",
                 color=CANCER_COLOR[c], pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(CANCER_COLOR[c])
        sp.set_linewidth(1.8)

for ext in ("pdf", "png"):
    p = OUT / f"fig2f_umap.{ext}"
    fig.savefig(p, dpi=400, bbox_inches="tight")
    print(f"saved -> {p}")
plt.close(fig)
print("done F-UMAP satellite.")
