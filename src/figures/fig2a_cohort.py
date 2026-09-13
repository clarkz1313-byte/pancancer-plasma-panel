#!/usr/bin/env python3
"""
fig2a_cohort.py — Internal cohort sample distribution
Horizontal stacked bars: train (solid) | held-out (lighter tint)
Sorted by total n, largest at top.
Annotations: per-segment train / held-out counts + total.
Output: figurev5/output/fig2a_cohort.pdf/.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import Patch
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V3   = ROOT / "figures_nature_v3" / "production" / "01_study_cohort" / "source_data"
OUT  = ROOT / "figurev6" / "output"
OUT.mkdir(parents=True, exist_ok=True)

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

# Y-axis labels: "Full Name (CODE)" — skip redundant parenthetical for all-acronym names
def cancer_label(code):
    return "DLBCL" if code == "LYMPH" else code

def lighter(hexcol, blend=0.50):
    r, g, b = mc.to_rgb(hexcol)
    return (r + (1-r)*blend, g + (1-g)*blend, b + (1-b)*blend)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 13, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.linewidth": 1.3, "xtick.major.width": 1.3, "ytick.major.width": 1.3,
})

df = pd.read_csv(V3 / "internal_pancancer_class_distribution.csv")
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={"endpoint": "cancer"})
df["cancer"] = df["cancer"].str.upper()
df = df.sort_values("internal_total_n", ascending=True).reset_index(drop=True)
cancers_sorted = df["cancer"].tolist()

fig, ax = plt.subplots(figsize=(10, 7))

for idx, row in df.iterrows():
    c       = row["cancer"]
    col     = CANCER_COLOR.get(c, "#888888")
    traindev = int(round(row["internal_train_dev_n_estimate"]))
    held     = int(row["heldout_test_n"])
    total    = int(row["internal_total_n"])

    ax.barh(idx, traindev, color=col,          height=0.70, zorder=2)
    ax.barh(idx, held, left=traindev,           color=lighter(col), height=0.70, zorder=2)

    # Per-segment counts inside bars (only if wide enough)
    if traindev > 0:
        ax.text(traindev / 2, idx, str(traindev),
                va="center", ha="center", fontsize=13, fontweight="bold",
                color="white")
    if held > 0:
        ax.text(traindev + held / 2, idx, str(held),
                va="center", ha="center", fontsize=13, fontweight="bold",
                color=col)

    # Total label to the right (number only, no "n=")
    ax.text(total + 6, idx, str(total),
            va="center", fontsize=14, fontweight="bold", color=col)

ax.set_yticks(range(len(df)))
ax.set_yticklabels([cancer_label(c) for c in cancers_sorted],
                   fontsize=15, fontweight="bold")
for tick, c in zip(ax.get_yticklabels(), cancers_sorted):
    tick.set_color(CANCER_COLOR.get(c, "#333333"))

ax.set_xlabel("Number of samples", fontsize=16, fontweight="bold", labelpad=6)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, df["internal_total_n"].max() * 1.28)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", length=0)

ax.legend(handles=[
    Patch(facecolor="#555555",           label="Train"),
    Patch(facecolor=lighter("#555555"),  label="Held-out test"),
], fontsize=14, framealpha=0.0, loc="lower right")

ax.grid(axis="x", color="#e8e8e8", lw=0.9, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
for ext in ("pdf", "png"):
    p = OUT / f"fig2a_cohort.{ext}"
    fig.savefig(p, dpi=400, bbox_inches="tight")
    print(f"saved -> {p}")
plt.close(fig)
print("done A.")
