#!/usr/bin/env python3
"""
fig4c_membership.py  –  Protein membership dot matrix
Rows = 12 cancers  |  Cols = 25 locked Part-B proteins (in rank order)
Filled dot = protein appears in cancer's Part A single-panel deployment
Empty dot  = not in Part A deploy panel

Bridges the big Part-B heatmap (fig4a) with the 12 Part-A heatmaps (fig4b),
showing how many Part-B proteins were also selected for single-cancer binary panels.

Output: figurev5/output/fig4c_protein_membership.pdf/.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

ROOT   = Path("e:/Proteomics")
TABLES = ROOT / "revise_plan/part_b_multiclass/vRSX_v11_locked_reproducer/tables"
OUT    = ROOT / "figurev5/output"
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

plt.rcParams.update({
    "font.family": "Arial", "font.size": 11, "font.weight": "bold",
    "axes.titleweight": "bold",
})


def build_membership():
    # 25 locked proteins (rank order from locked_selected_features.csv)
    feats = pd.read_csv(TABLES / "locked_selected_features.csv")
    proteins = feats.sort_values("rank")["protein"].tolist()

    # Part-A single-panel deploy proteins (semicolon-separated in Supplementary Table S1)
    supp = pd.read_csv(
        ROOT / "paper/supplementary_tables/Supplementary_Table_S1_single_panel_protein_summary.csv"
    ).set_index("target_class")

    # Build binary membership matrix  [cancer × locked-protein]
    mat = pd.DataFrame(False, index=CANCERS, columns=proteins)
    for c in CANCERS:
        deploy_str = str(supp.loc[c, "deploy_proteins"])
        deploy_set = {p.strip() for p in deploy_str.split(";")}
        for p in proteins:
            if p in deploy_set:
                mat.loc[c, p] = True

    return mat, proteins


def panel_membership():
    mat, proteins = build_membership()
    n_cancers  = len(CANCERS)
    n_proteins = len(proteins)

    fig, ax = plt.subplots(figsize=(16, 5.5))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.82, bottom=0.26)

    DOT_FULL = 260
    DOT_EMPTY = 55

    for ci, c in enumerate(CANCERS):
        col = CANCER_COLOR[c]
        for pi, p in enumerate(proteins):
            member = bool(mat.loc[c, p])
            if member:
                ax.scatter(pi, ci, s=DOT_FULL, color=col,
                           zorder=3, ec="white", lw=0.6)
            else:
                ax.scatter(pi, ci, s=DOT_EMPTY, color="none",
                           zorder=2, ec=col, lw=0.8, alpha=0.35)

    # Y-axis: cancer names (bottom = AML, top = PRC)
    ax.set_yticks(range(n_cancers))
    ylabels = ["DLBCL" if c == "LYMPH" else
               (f"{CANCER_FULL[c]} ({c})" if CANCER_FULL[c] != c else c)
               for c in CANCERS]
    ax.set_yticklabels(ylabels, fontsize=10.5, fontweight="bold")
    for tick, c in zip(ax.get_yticklabels(), CANCERS):
        tick.set_color(CANCER_COLOR[c])

    # X-axis: protein names (rotated)
    ax.set_xticks(range(n_proteins))
    ax.set_xticklabels(
        [f"{i+1}. {p}" for i, p in enumerate(proteins)],
        rotation=55, ha="right", fontsize=9, fontweight="bold",
    )

    ax.set_xlim(-0.7, n_proteins - 0.3)
    ax.set_ylim(-0.7, n_cancers - 0.3)
    ax.grid(axis="x", color="#eeeeee", lw=0.7, zorder=0)
    ax.spines[["top","right","bottom"]].set_visible(False)
    ax.tick_params(axis="both", length=0)

    # Column fill-count annotation (how many cancers include each protein)
    col_sums = mat.astype(int).sum(axis=0)
    for pi, p in enumerate(proteins):
        n = col_sums[p]
        if n > 0:
            ax.text(pi, n_cancers - 0.1, str(n),
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#444444")

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor="#555555", label="In Part-A deploy panel"),
        mpatches.Patch(facecolor="none", edgecolor="#888888",
                       label="Not in Part-A deploy panel"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              framealpha=0.85, fontsize=10, handlelength=1.0)

    ax.set_title(
        "Part-B locked 25 proteins  ×  Part-A single-cancer binary panels\n"
        "Filled dot = protein selected for cancer's Part-A deploy panel  ·  "
        "Number above column = count of cancer panels that use this protein",
        fontsize=12, fontweight="bold", pad=10,
    )

    for ext in ("pdf", "png"):
        p = OUT / f"fig4c_protein_membership.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    print("=== fig4c protein membership dot matrix ===")
    panel_membership()
    print("Done.")
