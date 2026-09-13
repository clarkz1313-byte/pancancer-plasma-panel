#!/usr/bin/env python3
"""
fig2c_dea_landscape.py — Differential protein landscape
Butterfly (diverging) horizontal bar chart:
  Right = upregulated proteins (cancer color, BH < 0.05)
  Left  = downregulated proteins (lighter tint, BH < 0.05)
Sorted by total significant count descending.
Output: figurev5/output/fig2c_dea_landscape.pdf/.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import Patch
import pandas as pd
import numpy as np

ROOT = Path("e:/Proteomics")
DEA  = (ROOT / "revise_plan" / "part_b_multiclass" /
        "vRSX_v11_locked_reproducer" / "tables" / "train_only_ovr_dea_all.csv")
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

def lighter(hexcol, blend=0.52):
    r, g, b = mc.to_rgb(hexcol)
    return (r + (1-r)*blend, g + (1-g)*blend, b + (1-b)*blend)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 13, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.linewidth": 1.3, "xtick.major.width": 1.3, "ytick.major.width": 1.3,
})

dea = pd.read_csv(DEA)
counts = {}
for c in CANCERS:
    sub = dea[(dea["target_class"] == c) & (dea["significant_bh"] == True)]
    up  = int((sub["fold_change"] > 0).sum())
    dn  = int((sub["fold_change"] <= 0).sum())
    counts[c] = {"up": up, "down": dn, "total": up + dn}

# Sort ascending so largest total is at top
order = sorted(CANCERS, key=lambda c: counts[c]["total"])

fig, ax = plt.subplots(figsize=(11, 7))

x_max = max(counts[c]["up"]   for c in CANCERS)
x_min = max(counts[c]["down"] for c in CANCERS)

for idx, c in enumerate(order):
    col   = CANCER_COLOR[c]
    up    = counts[c]["up"]
    dn    = counts[c]["down"]
    total = counts[c]["total"]

    ax.barh(idx,  up, color=col,          height=0.68, zorder=2)
    ax.barh(idx, -dn, color=lighter(col), height=0.68, zorder=2)

    # Per-side count labels (outside bars, compact)
    ax.text(up + 12,  idx, f"+{up}",
            va="center", ha="left",  fontsize=10, fontweight="bold", color=col)
    ax.text(-dn - 12, idx, f"−{dn}",
            va="center", ha="right", fontsize=10, fontweight="bold",
            color=lighter(col, blend=0.20))

# Y-axis labels with abbreviation
ax.set_yticks(range(len(order)))
ax.set_yticklabels([cancer_label(c) for c in order], fontsize=12, fontweight="bold")
for tick, c in zip(ax.get_yticklabels(), order):
    tick.set_color(CANCER_COLOR[c])

ax.axvline(0, color="#444444", lw=1.5, zorder=3)
ax.set_xlabel("Number of differentially expressed proteins  (BH-FDR < 0.05, OvR)",
              fontsize=13, fontweight="bold", labelpad=6)
ax.set_title(
    "Differential plasma protein landscape\n"
    "One-vs-rest DEA · 1,464 proteins · training set",
    fontsize=15, fontweight="bold", pad=10,
)

pad = 110
ax.set_xlim(-x_min - pad, x_max + pad)
ax.set_ylim(-0.55, len(order) - 0.45)   # explicit ylim — no room for floating text to clip into
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", length=0)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: str(abs(int(x)))))

# Legend at lower-left — the negative (downregulated) side has the most empty space
# at the bottom rows where MYEL/LYMPH/BRC/CLL have very short leftward bars.
ax.legend(handles=[
    Patch(facecolor="#555555", label="Upregulated vs rest"),
    Patch(facecolor=lighter("#555555"), label="Downregulated vs rest"),
], fontsize=11, framealpha=0.0, loc="lower left")

ax.grid(axis="x", color="#e8e8e8", lw=0.9, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
for ext in ("pdf", "png"):
    p = OUT / f"fig2c_dea_landscape.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"saved -> {p}")
plt.close(fig)
print("done C.")
