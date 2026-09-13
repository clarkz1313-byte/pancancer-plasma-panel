#!/usr/bin/env python3
"""
Fig 6 panels — figurev5 (mosaic-ready)
Pathway enrichment for the locked 25-protein Part B panel
  Left:   GO MF/CC  — 6 FDR-significant terms   (purple scale)
  Middle: KEGG      — top 5 nominal terms        (orange scale)
  Right:  Reactome  — top 5 nominal terms        (blue-green scale)
DB colors consistent with fig6_chord_network.py and fig6_parta_opt3.
"""
from __future__ import annotations
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path("e:/Proteomics")
EV   = ROOT / "revise_plan" / "locked25_final_results_package" / "pathway_outputs"
OUT  = ROOT / "figurev5" / "output"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 12, "font.weight": "bold",
    "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.linewidth": 1.1,
})
DPI = 300
_SP = dict(color="#666666", lw=0.9)

# Consistent DB accent colors (matches chord network and opt3)
DB_COL = {"GO": "#7b3f9e", "KEGG": "#c97b00", "Reactome": "#1a8b72"}
DB_BG  = {"GO": "#fdf8ff", "KEGG": "#fffbf0", "Reactome": "#f0faf7"}

ENTREZ_SYM = {
    "2322": "FLT3",   "2208": "FCER2",  "284340": "CXCL17",
    "10563": "CXCL13","652":  "BMP4",   "5623":   "PSPN",
    "7186":  "TRAF2", "8115": "TCL1A",  "1272":   "CNTN1",
    "57823": "SLAMF7",
}


def save(fig, stem):
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


def parse_gr(s):
    a, b = str(s).split("/")
    return int(a) / int(b)


def wrap_label(s, width=30):
    return "\n".join(textwrap.wrap(str(s), width))


def bubble_panel(fig, ax, df, title, subtitle, cmap_name, significant, db="GO"):
    db_color = DB_COL.get(db, "#333333")

    df = df.copy().reset_index(drop=True)
    df["gr"]   = df["GeneRatio"].apply(parse_gr)
    df["logp"] = -np.log10(df["p.adjust"].clip(lower=1e-10))
    df = df.sort_values("gr", ascending=True).reset_index(drop=True)

    n    = len(df)
    ypos = np.arange(n)

    vmin = df["logp"].min() * 0.80
    vmax = df["logp"].max() * 1.10
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    bubble_s = df["Count"].values * 210

    sc = ax.scatter(df["gr"].values, ypos,
                    s=bubble_s, c=df["logp"].values,
                    cmap=cmap, norm=norm,
                    edgecolors="#444444", linewidths=1.0,
                    zorder=5, alpha=0.92)

    # Protein annotations right of each bubble
    x_max = df["gr"].max()
    for idx, row in df.iterrows():
        raw   = str(row.get("marker_symbols", row.get("geneID", ""))).strip()
        parts = [p.strip() for p in raw.replace("/", ",").split(",") if p.strip()]
        parts = [ENTREZ_SYM.get(p, p) for p in parts]
        syms  = ", ".join(p for p in parts if p and p != "nan" and not p.isdigit())
        if syms:
            ax.text(x_max * 1.10, idx, syms,
                    va="center", ha="left",
                    fontsize=10, color="#222222",
                    fontstyle="italic", clip_on=False)

    ax.set_yticks(ypos)
    ax.set_yticklabels([wrap_label(d) for d in df["Description"]],
                       fontsize=11, fontweight="bold")
    ax.set_xlabel("Gene Ratio", fontsize=13, fontweight="bold", labelpad=4)
    ax.set_xlim(0, x_max * 1.65)
    ax.set_ylim(-0.65, n - 0.35)
    ax.tick_params(axis="x", labelsize=11)

    # Compact colorbar — very short horizontal strip below axes
    cb = fig.colorbar(sc, ax=ax, location="bottom",
                      shrink=0.28, pad=0.08, aspect=14)
    cb.set_label("-log10(p.adj)", fontsize=11, fontweight="bold", labelpad=3)
    cb.ax.tick_params(labelsize=10)
    cb.ax.xaxis.set_label_position("bottom")

    ax.set_title(title, fontsize=16, fontweight="bold", pad=10, color=db_color)

    # Significance badge
    badge_col = db_color if significant else "#888888"
    badge_txt = "FDR < 0.05 (BH corrected)" if significant else "Nominal — no term FDR < 0.05"
    ax.text(0.98, 0.02, badge_txt, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, color=badge_col,
            fontstyle="italic", fontweight="normal")

    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=10, color="#555555",
            fontstyle="italic", fontweight="normal")

    # Gene count legend — below colorbar
    counts = sorted(df["Count"].unique())
    sz_map = {1: 7, 2: 10, 3: 13, 4: 16, 5: 18}
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor="#aaaaaa", markeredgecolor="#555555",
                      markersize=sz_map.get(c, 10), label=f"n = {c}")
               for c in counts]
    leg = ax.legend(handles=handles, title="Gene count",
                    fontsize=9.5, title_fontsize=9.5,
                    loc="upper left",
                    bbox_to_anchor=(0.01, -0.22),
                    bbox_transform=ax.transAxes,
                    frameon=True, framealpha=0.92,
                    edgecolor="#cccccc", ncol=len(counts))
    leg.get_title().set_fontweight("bold")

    ax.grid(axis="x", color="#eeeeee", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(_SP["color"])
        sp.set_linewidth(_SP["lw"])

    ax.set_facecolor(DB_BG.get(db, "white"))


# ─── Load & prepare data ─────────────────────────────────────────────────────
go_df = pd.read_csv(
    EV / "enrichment_visualizations/part_b_locked_25/go_significant_panel_terms.csv"
)

kegg_raw  = pd.read_csv(EV / "pathway_enrichment/part_b_locked_25/kegg_enrichment_results.csv")
KEGG_SKIP = {"hsa05169","hsa04913","hsa05320","hsa00620","hsa00480"}
kegg_df   = (kegg_raw[~kegg_raw["ID"].isin(KEGG_SKIP)]
             .nsmallest(5, "pvalue").reset_index(drop=True))

react_raw = pd.read_csv(EV / "pathway_enrichment/part_b_locked_25/reactome_enrichment_results.csv")
REACT_SKIP = ["nephric duct","Kidney development","B3GALTL",
              "Androgen biosynthesis","Thyroxine","Glycoprotein hormones",
              "Diseases associated with O-glycosylation","O-linked glycosylation"]
mask = react_raw["Description"].apply(
    lambda d: not any(kw.lower() in str(d).lower() for kw in REACT_SKIP))
react_df = (react_raw[mask & (react_raw["Count"] >= 2)]
            .nsmallest(5, "pvalue").reset_index(drop=True))

react_df["marker_symbols"] = react_df["geneID"].apply(
    lambda x: ", ".join(str(x).split("/")) if pd.notna(x) else ""
)

# ─── Build figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 9.0))
fig.subplots_adjust(wspace=0.55, left=0.04, right=0.97, top=0.87, bottom=0.18)

bubble_panel(fig, axes[0], go_df,
             "GO Molecular Function & Cell. Component",
             "6 significant terms · background = 1,463 proteins",
             "RdPu", significant=True, db="GO")

bubble_panel(fig, axes[1], kegg_df,
             "KEGG Pathways",
             "Top 5 by p-value · no term survives FDR correction",
             "YlOrBr", significant=False, db="KEGG")

bubble_panel(fig, axes[2], react_df,
             "Reactome Pathways",
             "Top 5 by p-value (Count ≥ 2) · no term survives FDR correction",
             "GnBu", significant=False, db="Reactome")

fig.suptitle(
    "Locked 25-protein panel — pathway enrichment  "
    "(ORA · background = 1,463 assayed proteins · bubble size = gene count)",
    fontsize=14, fontweight="bold", y=0.97)

save(fig, "fig6_pathway_enrichment")
print("Done.")
