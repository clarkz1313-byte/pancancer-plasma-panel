#!/usr/bin/env python3
"""
Fig 6 — Advanced pathway options (A / B / C) for PI selection
  A: Protein–term bipartite network (cnetplot style)
  B: Combined tri-database lollipop (GO + KEGG + Reactome in one axis)
  C: Protein × pathway membership heatmap (binary matrix)
All based on the locked 25-protein Part B multiclass panel.
"""
from __future__ import annotations
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

ROOT = Path("e:/Proteomics")
EV   = ROOT / "revise_plan" / "locked25_final_results_package" / "pathway_outputs"
OUT  = ROOT / "figurev5" / "output"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "font.weight": "bold",
    "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
})
DPI = 300
_SP = dict(color="#666666", lw=0.9)

# ─── Database palette ─────────────────────────────────────────────────────────
DB_COL  = {"GO": "#7b3f9e", "KEGG": "#c97b00", "Reactome": "#1a8b72"}
DB_LIGHT = {"GO": "#ede0f5", "KEGG": "#fff3cc", "Reactome": "#d6f5ed"}

# ─── Protein functional categories ───────────────────────────────────────────
PROT_CAT = {
    "FLT3": "Receptor/kinase",   "FCER2": "Immune receptor",
    "SLAMF7": "Immune receptor", "CXCL17": "Cytokine",
    "CXCL13": "Cytokine",        "BMP4": "Growth factor",
    "PSPN": "Growth factor",     "TRAF2": "Signaling",
    "TCL1A": "Signaling",        "CNTN1": "Neural adhesion",
    "GFAP": "Structural",        "ADAMTS13": "Metalloprotease",
    "ADAMTS15": "Metalloprotease","CCDC80": "ECM",
    "LTA4H": "Enzyme",           "NEFL": "Structural",
}
CAT_COL = {
    "Receptor/kinase":  "#b22222",
    "Immune receptor":  "#7a3e9d",
    "Cytokine":         "#c13d86",
    "Growth factor":    "#2e8b57",
    "Signaling":        "#d68600",
    "Neural adhesion":  "#3856a6",
    "Structural":       "#8c564b",
    "Metalloprotease":  "#d1495b",
    "ECM":              "#008b8b",
    "Enzyme":           "#c97b63",
}

# ENTREZ → gene symbol (KEGG uses ENTREZ IDs)
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


def _spine_style(ax):
    for sp in ax.spines.values():
        sp.set_color(_SP["color"])
        sp.set_linewidth(_SP["lw"])


# ─── Load data ────────────────────────────────────────────────────────────────
def _syms(s, db="GO"):
    """Parse protein symbols.  GO uses comma-sep marker_symbols; others use /."""
    raw = str(s)
    if db == "GO":
        return [x.strip() for x in raw.split(",") if x.strip()]
    elif db == "KEGG":
        return [ENTREZ_SYM.get(x.strip(), x.strip()) for x in raw.split("/")]
    else:
        return [x.strip() for x in raw.split("/")]


go_raw = pd.read_csv(
    EV / "enrichment_visualizations/part_b_locked_25/go_significant_panel_terms.csv"
)
kegg_raw = pd.read_csv(
    EV / "pathway_enrichment/part_b_locked_25/kegg_enrichment_results.csv"
)
react_raw = pd.read_csv(
    EV / "pathway_enrichment/part_b_locked_25/reactome_enrichment_results.csv"
)

KEGG_SKIP = {"hsa05169","hsa04913","hsa05320","hsa00620","hsa00480"}
REACT_SKIP = ["nephric duct","Kidney development","B3GALTL","Androgen",
              "Thyroxine","Glycoprotein hormones",
              "Diseases associated with O-glycosylation","O-linked glycosylation"]

kegg_df = (kegg_raw[~kegg_raw["ID"].isin(KEGG_SKIP)]
           .nsmallest(5, "pvalue").reset_index(drop=True))
mask = react_raw["Description"].apply(
    lambda d: not any(kw.lower() in str(d).lower() for kw in REACT_SKIP))
react_df = (react_raw[mask & (react_raw["Count"] >= 2)]
            .nsmallest(5, "pvalue").reset_index(drop=True))

# Unified term table — store both raw pval and BH-adjusted padj
rows = []
for _, r in go_raw.iterrows():
    rows.append({"db": "GO",       "term": r["Description"],
                 "pval": r["pvalue"], "padj": r["p.adjust"],
                 "count": r["Count"],
                 "proteins": _syms(r["marker_symbols"], "GO")})
for _, r in kegg_df.iterrows():
    rows.append({"db": "KEGG",     "term": r["Description"],
                 "pval": r["pvalue"], "padj": r["p.adjust"],
                 "count": r["Count"],
                 "proteins": _syms(r["geneID"], "KEGG")})
for _, r in react_df.iterrows():
    rows.append({"db": "Reactome", "term": r["Description"],
                 "pval": r["pvalue"], "padj": r["p.adjust"],
                 "count": r["Count"],
                 "proteins": _syms(r["geneID"], "Reactome")})

terms = pd.DataFrame(rows).reset_index(drop=True)

# Unique individual proteins across all terms
all_prots = sorted({p for ps in terms["proteins"] for p in ps})


# ══════════════════════════════════════════════════════════════════════════════
# OPTION A — Bipartite protein–term network
# ══════════════════════════════════════════════════════════════════════════════
def build_option_a():
    G = nx.Graph()
    # protein nodes
    for prot in all_prots:
        G.add_node(prot, kind="protein")
    # term nodes + edges
    for i, row in terms.iterrows():
        tnode = f"T{i}"
        G.add_node(tnode, kind="term", db=row["db"],
                   label=row["term"], count=row["count"],
                   logp=-np.log10(max(row["padj"], 1e-10)))
        for prot in row["proteins"]:
            if prot in all_prots:
                G.add_edge(prot, tnode, db=row["db"])

    # bipartite layout: proteins left, terms right
    prot_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "protein"]
    term_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "term"]
    pos = {}
    for i, n in enumerate(sorted(prot_nodes)):
        pos[n] = (0.0, i / max(len(prot_nodes) - 1, 1))
    for i, n in enumerate(term_nodes):
        pos[n] = (1.0, i / max(len(term_nodes) - 1, 1))

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-0.18, 1.42)
    ax.set_ylim(-0.06, 1.06)
    ax.axis("off")

    # Draw edges (color by database)
    for u, v, d in G.edges(data=True):
        col = DB_COL[d["db"]]
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=col, lw=1.1, alpha=0.45, zorder=1)

    # Draw protein nodes (circles, colored by functional category)
    for n in prot_nodes:
        x, y = pos[n]
        cat  = PROT_CAT.get(n, "Other")
        fc   = CAT_COL.get(cat, "#aaaaaa")
        circ = plt.Circle((x, y), 0.018,
                           facecolor=fc, edgecolor="white",
                           linewidth=1.2, zorder=4)
        ax.add_patch(circ)
        ax.text(x - 0.025, y, n, ha="right", va="center",
                fontsize=9, fontweight="bold", color="#111111")

    # Draw term nodes (square markers, colored by database)
    logp_vals = [G.nodes[n]["logp"] for n in term_nodes]
    logp_min, logp_max = min(logp_vals), max(logp_vals)
    for n in term_nodes:
        x, y   = pos[n]
        d_data = G.nodes[n]
        db     = d_data["db"]
        fc     = DB_COL[db]
        size   = 0.014 + 0.014 * (d_data["count"] - 1)
        rect   = plt.Rectangle((x - size, y - size * 0.75),
                                size * 2, size * 1.5,
                                facecolor=fc, edgecolor="white",
                                linewidth=1.2, alpha=0.85, zorder=4)
        ax.add_patch(rect)
        label  = "\n".join(textwrap.wrap(d_data["label"], 28))
        ax.text(x + 0.028, y, label,
                ha="left", va="center", fontsize=8,
                fontstyle="italic", color="#111111")

    # Protein category legend
    cat_seen = {PROT_CAT.get(n, "Other") for n in prot_nodes}
    leg_prot = [mpatches.Patch(facecolor=CAT_COL.get(c, "#aaaaaa"),
                               edgecolor="white", label=c)
                for c in sorted(cat_seen)]
    l1 = ax.legend(handles=leg_prot, title="Protein function",
                   fontsize=8, title_fontsize=8.5,
                   loc="upper left", framealpha=0.92,
                   edgecolor="#cccccc", ncol=1)
    l1.get_title().set_fontweight("bold")

    # Database legend for term boxes
    leg_db = [mpatches.Patch(facecolor=DB_COL[db], edgecolor="white", label=db)
              for db in ["GO", "KEGG", "Reactome"]]
    ax.legend(handles=leg_db, title="Database",
              fontsize=8, title_fontsize=8.5,
              loc="lower left", framealpha=0.92,
              edgecolor="#cccccc")
    ax.add_artist(l1)

    ax.set_title(
        "Option A — Protein–pathway bipartite network\n"
        "Locked 25-protein panel · circles = proteins · squares = enriched terms",
        fontsize=13, fontweight="bold", pad=10)

    save(fig, "fig6_optionA_network")


# ══════════════════════════════════════════════════════════════════════════════
# OPTION B — Combined tri-database lollipop
# ══════════════════════════════════════════════════════════════════════════════
def build_option_b():
    # Sort: Reactome top, KEGG middle, GO bottom (bottom = most prominent = FDR sig)
    # within each group sorted best→worst so best is at top of that group
    order = {"GO": 2, "KEGG": 1, "Reactome": 0}
    df = terms.copy()
    # x-axis = -log10(p.adjust) so FDR line is meaningful
    df["logpadj"] = -np.log10(df["padj"].clip(lower=1e-10))
    df["_ord"]    = df["db"].map(order)
    df = df.sort_values(["_ord", "padj"], ascending=[True, False]).reset_index(drop=True)

    n = len(df)
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(left=0.38, right=0.72, top=0.90, bottom=0.10)

    ypos = np.arange(n)
    fdr_line = -np.log10(0.05)   # = 1.30

    # Database background stripes + left labels
    db_bounds = {}
    for db in ["Reactome", "KEGG", "GO"]:
        idxs = df.index[df["db"] == db].tolist()
        if idxs:
            db_bounds[db] = (min(idxs) - 0.5, max(idxs) + 0.5)
    for db, (ylo, yhi) in db_bounds.items():
        ax.axhspan(ylo, yhi, color=DB_LIGHT[db], alpha=0.70, zorder=0)
        # Label flush-left outside the axes
        ax.annotate(db, xy=(0, (ylo + yhi) / 2),
                    xycoords=("axes fraction", "data"),
                    xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=10,
                    fontweight="bold", color=DB_COL[db])

    # Lollipop stems and heads
    for i, row in df.iterrows():
        col     = DB_COL[row["db"]]
        logpadj = row["logpadj"]
        is_sig  = logpadj >= fdr_line
        lw = 2.5 if is_sig else 1.8
        ax.plot([0, logpadj], [i, i], color=col, lw=lw, alpha=0.85, zorder=2)
        ax.scatter([logpadj], [i], s=row["count"] * 140,
                   color=col, edgecolors="white" if is_sig else col,
                   linewidths=1.5, zorder=3, alpha=0.92)
        # FDR-sig marker
        if is_sig:
            ax.text(logpadj + 0.04, i, "*",
                    va="center", ha="left", fontsize=13,
                    color="#cc0000", fontweight="bold")
        # Protein symbols to the right
        syms = ", ".join(row["proteins"])
        offset = 0.18 if is_sig else 0.06
        ax.text(logpadj + offset, i, syms,
                va="center", ha="left", fontsize=8,
                fontstyle="italic", color="#222222", clip_on=False)

    # FDR threshold line
    ax.axvline(fdr_line, color="#cc0000", lw=1.4, ls="--", zorder=1, alpha=0.8)
    ax.text(fdr_line + 0.03, n - 0.4, "FDR 0.05",
            color="#cc0000", fontsize=8, va="top", fontweight="bold")

    # Y-axis term labels
    ax.set_yticks(ypos)
    ax.set_yticklabels(["\n".join(textwrap.wrap(t, 34)) for t in df["term"]],
                       fontsize=8.5, fontweight="bold")
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("-log10(p.adj)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=10)

    # Size legend
    counts = sorted(df["count"].unique())
    sz_map = {1: 8, 2: 11, 3: 14, 4: 16}
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor="#aaaaaa", markeredgecolor="#555555",
                      markersize=sz_map.get(c, 12), label=f"n = {c}")
               for c in counts]
    ax.legend(handles=handles, title="Gene count",
              fontsize=8, title_fontsize=8.5, loc="lower right",
              frameon=True, framealpha=0.90,
              edgecolor="#cccccc").get_title().set_fontweight("bold")

    ax.grid(axis="x", color="#eeeeee", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    _spine_style(ax)

    ax.set_title(
        "Option B — Tri-database enrichment lollipop\n"
        "x-axis = -log10(p.adj BH) · * FDR < 0.05 · locked 25-protein panel",
        fontsize=13, fontweight="bold", pad=10)

    save(fig, "fig6_optionB_lollipop")


# ══════════════════════════════════════════════════════════════════════════════
# OPTION C — Protein × pathway membership heatmap
# ══════════════════════════════════════════════════════════════════════════════
def build_option_c():
    # Build binary membership matrix
    mat = pd.DataFrame(0, index=all_prots,
                       columns=[f"{r['db']}::{r['term']}" for _, r in terms.iterrows()])
    for _, row in terms.iterrows():
        col = f"{row['db']}::{row['term']}"
        for prot in row["proteins"]:
            if prot in mat.index:
                mat.loc[prot, col] = 1

    # Drop proteins with zero membership
    mat = mat.loc[mat.sum(axis=1) > 0]

    # Cluster rows and columns
    def _order(m, axis=0):
        if m.shape[axis] <= 2:
            return list(range(m.shape[axis]))
        d = pdist(m if axis == 0 else m.T, metric="jaccard")
        d = np.nan_to_num(d, nan=0)
        return leaves_list(linkage(d, method="average"))

    row_ord = _order(mat.values, axis=0)
    col_ord = _order(mat.values, axis=1)
    mat = mat.iloc[row_ord, col_ord]

    cols_db = [c.split("::")[0] for c in mat.columns]
    cols_label = ["\n".join(textwrap.wrap(c.split("::", 1)[1], 22))
                  for c in mat.columns]

    fig_w = max(14, len(mat.columns) * 1.1)
    fig_h = max(6, len(mat.index) * 0.65 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.78, bottom=0.38)

    nr, nc = mat.shape
    # Draw cells
    for ci, col in enumerate(mat.columns):
        db = cols_db[ci]
        fc_on  = DB_COL[db]
        fc_off = DB_LIGHT[db]
        for ri in range(nr):
            val = mat.iloc[ri, ci]
            ax.add_patch(plt.Rectangle(
                (ci, ri), 1, 1,
                facecolor=fc_on if val else fc_off,
                edgecolor="white", linewidth=1.2, zorder=2))

    # Column labels (term names, rotated)
    ax.set_xticks([i + 0.5 for i in range(nc)])
    ax.set_xticklabels(cols_label, rotation=55, ha="right",
                       rotation_mode="anchor", fontsize=7.5, fontweight="bold")

    # Row labels (protein names)
    ax.set_yticks([i + 0.5 for i in range(nr)])
    ax.set_yticklabels(mat.index, fontsize=9.5, fontweight="bold")

    ax.set_xlim(0, nc); ax.set_ylim(0, nr)
    ax.tick_params(length=0)

    # Database color header strip above columns
    for ci, db in enumerate(cols_db):
        ax.add_patch(plt.Rectangle(
            (ci, nr), 1, 0.45,
            facecolor=DB_COL[db], edgecolor="white",
            linewidth=1.2, zorder=3, clip_on=False))

    # Database legend
    leg = [mpatches.Patch(facecolor=DB_COL[db], label=db)
           for db in ["GO", "KEGG", "Reactome"]]
    ax.legend(handles=leg, title="Database", fontsize=8.5,
              title_fontsize=8.5, loc="upper left",
              bbox_to_anchor=(0, 1.14), framealpha=0.92,
              edgecolor="#cccccc", ncol=3).get_title().set_fontweight("bold")

    # Protein category color strip (left side)
    for ri, prot in enumerate(mat.index):
        cat = PROT_CAT.get(prot, "Other")
        fc  = CAT_COL.get(cat, "#aaaaaa")
        ax.add_patch(plt.Rectangle(
            (-0.5, ri), 0.38, 1,
            facecolor=fc, edgecolor="white",
            linewidth=1.0, zorder=3, clip_on=False))

    _spine_style(ax)
    ax.set_title(
        "Option C — Protein × pathway membership heatmap\n"
        "Filled = protein present in term · column color = database · "
        "left strip = protein function",
        fontsize=12, fontweight="bold", pad=40)

    # Category legend
    cats_seen = {PROT_CAT.get(p, "Other") for p in mat.index}
    leg2 = [mpatches.Patch(facecolor=CAT_COL.get(c, "#aaaaaa"), label=c)
            for c in sorted(cats_seen)]
    ax.legend(handles=leg2, title="Protein function",
              fontsize=7.5, title_fontsize=8,
              loc="upper right", bbox_to_anchor=(1.0, 1.14),
              framealpha=0.92, edgecolor="#cccccc",
              ncol=2).get_title().set_fontweight("bold")

    save(fig, "fig6_optionC_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building advanced pathway options ...")
    print("\n--- Option A: bipartite network ---")
    build_option_a()
    print("\n--- Option B: tri-database lollipop ---")
    build_option_b()
    print("\n--- Option C: protein-pathway heatmap ---")
    build_option_c()
    print("\nDone. Three PDFs + PNGs in figurev5/output/")
