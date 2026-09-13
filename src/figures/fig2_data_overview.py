#!/usr/bin/env python3
"""
fig2_data_overview.py
Data introduction / descriptive statistics mosaic  (Fig 2).

4-panel mosaic in figurev5 style (Arial bold, CANCER_COLOR, 300 dpi):

  A [top-left]     Internal cohort stacked horizontal bar
                   train+dev (solid) | held-out (lighter tint)
                   sorted by total n, annotated

  B [top-right]    Locked-25 expression heatmap
                   12 cancers x 25 proteins, Z-scored mean NPX
                   diverging RdBu_r, cancer labels colored

  C [bottom, wide] External validation dot matrix
                   cancer rows x modality columns (gene proxy | protein)
                   dot size proportional to sqrt(n_tumor), CANCER_COLOR

  D [bottom-right] Disease-specific confirmation panel sizes
                   horizontal lollipop (protein count per single panel)
                   CANCER_COLOR, sorted by panel size

Output: figurev5/output/fig2_data_overview.pdf/.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from figure_labels import display_label

ROOT = Path("e:/Proteomics")
PKG  = ROOT / "revise_plan" / "locked25_final_results_package"
V3   = ROOT / "figures_nature_v3" / "production" / "01_study_cohort" / "source_data"
OUT  = ROOT / "figurev5" / "output"
OUT.mkdir(parents=True, exist_ok=True)

# ── shared constants ──────────────────────────────────────────────────────────
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
LOCKED25 = ["FLT3","CNTN1","FCER2","PRDX6","LTA4H","XG","CCDC80","CXCL17","CXCL13",
            "SLAMF7","PAEP","PSPN","BMP4","WFDC2","TRAF2","KLK13","GLO1","GFAP",
            "CEACAM5","CGA","ADAMTS13","CRTAC1","TCL1A","ADAMTS15","NEFL"]

def lighter(hexcol, blend=0.55):
    """Blend hex color toward white by `blend` fraction."""
    r, g, b = mc.to_rgb(hexcol)
    return (r + (1-r)*blend, g + (1-g)*blend, b + (1-b)*blend)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "font.weight": "bold",
    "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
})
DPI = 300

def spine_style(ax, sides=("top","right")):
    for s in sides:
        ax.spines[s].set_visible(False)
    for s in ax.spines:
        if s not in sides:
            ax.spines[s].set_color("#888888")
            ax.spines[s].set_linewidth(0.7)

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

# A: internal cohort distribution
coh = pd.read_csv(V3 / "internal_pancancer_class_distribution.csv")
coh = coh.set_index("endpoint").reindex(CANCERS)
# Sort ascending by total n for the bar chart (smallest at top)
coh_sorted = coh.sort_values("internal_total_n", ascending=True)

# B: locked-25 NPX heatmap
expr = pd.read_csv(ROOT / "data" / "processed" / "filtered_pancancer_data.csv")
proteins = [c for c in expr.columns if c not in ["Sample_ID","Cancer"]]
avail_l25 = [p for p in LOCKED25 if p in expr.columns]   # should be 25/25
mean_npx = expr.groupby("Cancer")[avail_l25].mean().reindex(CANCERS)
# Z-score per protein across cancers
z_npx = (mean_npx - mean_npx.mean()) / mean_npx.std()

# C: external validation — use the full source file (14 datasets, includes BRC-PXD)
ext_path = ROOT / "figures_nature_v3" / "production" / "01_study_cohort" / \
           "source_data" / "external_locked_main_summary.csv"
ext = pd.read_csv(ext_path)
ext = ext[["endpoint","dataset","data_level","n_tumor","n_control"]].copy()
ext["modality"] = ext["data_level"].apply(
    lambda x: "Protein" if "protein" in str(x).lower() else "Gene proxy")

# D: single-panel sizes
s1_path = ROOT / "paper" / "supplementary_tables" / \
          "Supplementary_Table_S1_single_panel_protein_summary.csv"
s1 = pd.read_csv(s1_path)
s1["cancer"] = s1["target_class"].str.upper()
s1 = s1.set_index("cancer").reindex(CANCERS)
panel_sizes = s1["deploy_panel_size"].astype(float)

# ─────────────────────────────────────────────────────────────────────────────
# Figure layout
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(19, 13))
gs  = GridSpec(2, 3,
               width_ratios=[1.05, 1.55, 1.05],
               height_ratios=[1.15, 1.0],
               left=0.07, right=0.97, top=0.93, bottom=0.06,
               hspace=0.46, wspace=0.40)

ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1:3])
ax_C = fig.add_subplot(gs[1, 0:2])
ax_D = fig.add_subplot(gs[1, 2])

# ─────────────────────────────────────────────────────────────────────────────
# PANEL A — Internal cohort stacked bar
# ─────────────────────────────────────────────────────────────────────────────
cancers_A = list(coh_sorted.index)   # sorted smallest -> largest
y_pos     = np.arange(len(cancers_A))
bar_h     = 0.65

for i, c in enumerate(cancers_A):
    row     = coh_sorted.loc[c]
    n_train = int(row["internal_train_dev_n_estimate"])
    n_held  = int(row["heldout_test_n"])
    col     = CANCER_COLOR[c]
    col_lt  = lighter(col, blend=0.55)

    ax_A.barh(i, n_train, height=bar_h,
              color=col, alpha=0.88, zorder=3)
    ax_A.barh(i, n_held, left=n_train, height=bar_h,
              color=col_lt, edgecolor=col, linewidth=0.7, zorder=3)
    # total annotation
    n_tot = int(row["internal_total_n"])
    ax_A.text(n_tot + 4, i, f"n={n_tot}",
              va="center", ha="left", fontsize=7.5,
              fontweight="bold", color=col)

ax_A.set_yticks(y_pos)
ax_A.set_yticklabels(
    [display_label(CANCER_FULL[c]) for c in cancers_A],
    fontsize=8, fontweight="bold")
for tick, c in zip(ax_A.get_yticklabels(), cancers_A):
    tick.set_color(CANCER_COLOR[c])

ax_A.set_xlabel("Samples", fontsize=9, fontweight="bold")
ax_A.set_xlim(0, 320)
ax_A.set_ylim(-0.6, len(cancers_A) - 0.4)
ax_A.set_title("Internal cohort  (n=1,375)", fontweight="bold", pad=6)
spine_style(ax_A)
ax_A.tick_params(axis="x", labelsize=8, length=3)
ax_A.tick_params(axis="y", length=0, pad=3)
ax_A.xaxis.grid(True, color="#eeeeee", lw=0.6, zorder=0)
ax_A.set_axisbelow(True)

# legend for train/held-out
leg_patches = [
    mpatches.Patch(facecolor="#555555", alpha=0.88, label="Train / dev"),
    mpatches.Patch(facecolor="#aaaaaa", edgecolor="#555555",
                   linewidth=0.7, label="Held-out test (~30%)"),
]
ax_A.legend(handles=leg_patches, loc="lower right",
            fontsize=7, framealpha=0.9, edgecolor="#cccccc",
            handlelength=1.2, handletextpad=0.5)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL B — Locked-25 heatmap
# ─────────────────────────────────────────────────────────────────────────────
z_mat = z_npx[avail_l25].values   # shape (12, 25)

im = ax_B.imshow(z_mat, aspect="auto", cmap="RdBu_r",
                 vmin=-2.0, vmax=2.0, interpolation="nearest")

# Cancer labels (y-axis) colored
ax_B.set_yticks(np.arange(12))
ax_B.set_yticklabels(
    [display_label(CANCER_FULL[c]) for c in CANCERS],
    fontsize=8, fontweight="bold")
for tick, c in zip(ax_B.get_yticklabels(), CANCERS):
    tick.set_color(CANCER_COLOR[c])

# Protein labels (x-axis) diagonal
ax_B.set_xticks(np.arange(len(avail_l25)))
ax_B.set_xticklabels(avail_l25, rotation=40, ha="right",
                     fontsize=7.5, fontstyle="italic", fontweight="bold")

ax_B.set_title("Locked-25 panel: cancer-specific expression (Z-scored NPX)",
               fontweight="bold", pad=6)

# Colorbar
cbar = fig.colorbar(im, ax=ax_B, fraction=0.025, pad=0.02, aspect=20)
cbar.set_label("Z-score", fontsize=8, fontweight="bold")
cbar.ax.tick_params(labelsize=7)

# Cell border lines (thin grid between cells)
for x in np.arange(-0.5, len(avail_l25), 1):
    ax_B.axvline(x, color="white", lw=0.4, zorder=4)
for y in np.arange(-0.5, 12, 1):
    ax_B.axhline(y, color="white", lw=0.4, zorder=4)

ax_B.tick_params(axis="both", length=0, pad=4)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL C — External validation dot matrix
# ─────────────────────────────────────────────────────────────────────────────
MOD_X = {"Gene proxy": 0.5, "Protein": 1.5}
MOD_LABEL = {0.5: "Gene proxy\n(transcript)", 1.5: "Protein-level"}
S_SCALE = 12.0    # sqrt(n_tumor) * S_SCALE -> scatter size

cancers_C = list(reversed(CANCERS))   # AML at top to match typical display
y_map_C = {c: i for i, c in enumerate(cancers_C)}

for _, row in ext.iterrows():
    c   = row["endpoint"]
    mod = row["modality"]
    x   = MOD_X[mod]
    y   = y_map_C.get(c, None)
    if y is None:
        continue
    n_t = row["n_tumor"]
    s   = max(np.sqrt(n_t) * S_SCALE, 40)
    col = CANCER_COLOR[c]

    marker = "o" if mod == "Protein" else "s"
    ax_C.scatter(x, y, s=s, color=col, alpha=0.85,
                 marker=marker, edgecolors="white", linewidths=0.8, zorder=4)
    # Dataset label next to dot
    label_x = x + 0.12 if mod == "Gene proxy" else x - 0.12
    ha_val   = "left"   if mod == "Gene proxy" else "right"
    ax_C.text(label_x, y, row["dataset"],
              va="center", ha=ha_val, fontsize=6.5,
              fontweight="bold", color=col, clip_on=False)

# y-axis: cancer names
ax_C.set_yticks(np.arange(len(cancers_C)))
ax_C.set_yticklabels(
    [display_label(CANCER_FULL[c]) for c in cancers_C],
    fontsize=8, fontweight="bold")
for tick, c in zip(ax_C.get_yticklabels(), cancers_C):
    tick.set_color(CANCER_COLOR[c])

# x-axis: modality
ax_C.set_xticks([0.5, 1.5])
ax_C.set_xticklabels(["Gene proxy\n(transcript)", "Protein-level"],
                     fontsize=9, fontweight="bold")
ax_C.set_xlim(-0.0, 2.3)
ax_C.set_ylim(-0.8, len(cancers_C) - 0.2)
ax_C.set_title("External validation datasets  (14 cohorts · 12 cancers)",
               fontweight="bold", pad=6)
spine_style(ax_C)
ax_C.tick_params(axis="x", length=0, pad=8)
ax_C.tick_params(axis="y", length=0, pad=3)
ax_C.yaxis.grid(True, color="#eeeeee", lw=0.5, zorder=0)
ax_C.set_axisbelow(True)

# size legend
for n_ex, lab in [(50,"n=50"),(200,"n=200"),(500,"n=500")]:
    ax_C.scatter([], [], s=np.sqrt(n_ex)*S_SCALE,
                 color="#999999", alpha=0.85, edgecolors="white",
                 linewidths=0.8, label=lab)
leg_c = ax_C.legend(title="Tumor n", loc="lower right",
                    fontsize=7, title_fontsize=7.5,
                    framealpha=0.9, edgecolor="#cccccc")
leg_c.get_title().set_fontweight("bold")

# modality shape legend
ax_C.scatter([], [], marker="s", color="#777777", s=50, label="Gene proxy")
ax_C.scatter([], [], marker="o", color="#777777", s=50, label="Protein")
ax_C.legend(loc="upper right", fontsize=7.5,
            framealpha=0.9, edgecolor="#cccccc",
            title="Modality", title_fontsize=7.5).get_title().set_fontweight("bold")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL D — Disease-specific confirmation panel sizes
# ─────────────────────────────────────────────────────────────────────────────
# Sort by panel size ascending (smallest at top for horizontal lollipop)
valid = panel_sizes.dropna().sort_values(ascending=True)
cancers_D = list(valid.index)
sizes_D   = list(valid.values)
y_D = np.arange(len(cancers_D))

for i, (c, sz) in enumerate(zip(cancers_D, sizes_D)):
    col = CANCER_COLOR[c]
    ax_D.plot([0, sz], [i, i], color=col, lw=1.6, zorder=2, alpha=0.7)
    ax_D.scatter(sz, i, color=col, s=80, zorder=4, edgecolors="white", linewidths=0.8)
    ax_D.text(sz + 0.15, i, str(int(sz)),
              va="center", ha="left", fontsize=8,
              fontweight="bold", color=col)

ax_D.set_yticks(y_D)
ax_D.set_yticklabels(
    [display_label(CANCER_FULL[c]) for c in cancers_D],
    fontsize=8, fontweight="bold")
for tick, c in zip(ax_D.get_yticklabels(), cancers_D):
    tick.set_color(CANCER_COLOR[c])

ax_D.set_xlabel("Proteins in panel", fontsize=9, fontweight="bold")
ax_D.set_xlim(0, 16)
ax_D.set_ylim(-0.6, len(cancers_D) - 0.4)
ax_D.set_title("Disease-specific\nconfirmation panels", fontweight="bold", pad=6)
spine_style(ax_D)
ax_D.tick_params(axis="x", labelsize=8, length=3)
ax_D.tick_params(axis="y", length=0, pad=3)
ax_D.xaxis.grid(True, color="#eeeeee", lw=0.6, zorder=0)
ax_D.set_axisbelow(True)

# ─────────────────────────────────────────────────────────────────────────────
# Overall title
# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Study data overview  --  12 cancers, 1,375 samples, 1,464 proteins",
    fontsize=13, fontweight="bold", y=0.97,
)

# Panel labels
for ax, lbl in [(ax_A,"a"),(ax_B,"b"),(ax_C,"c"),(ax_D,"d")]:
    ax.text(-0.12, 1.04, lbl, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left", color="#222222")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
for ext_fmt in ("pdf", "png"):
    out_p = OUT / f"fig2_data_overview.{ext_fmt}"
    fig.savefig(out_p, dpi=DPI, bbox_inches="tight")
    print(f"saved -> {out_p}")
plt.close(fig)
print("done.")
