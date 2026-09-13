#!/usr/bin/env python3
"""
fig6_ora_assay_background_2026-09-05.py — replaces fig6_slide6_pathway_mosaic.

WHY THE OLD FIGURE IS GONE
--------------------------
`fig6_slide6_pathway_mosaic` drew six FDR-significant GO terms whose
BgRatio denominators were 18,737 and 19,960 — the whole annotated genome,
not the 1,463-protein assay the Methods claimed. Its KEGG and Reactome
sources used 9,380 and 11,146. Against a genome background, a panel drawn
from a plasma secretome assay returns secretome terms whether or not panel
selection had anything to do with them.

WHAT THIS FIGURE SHOWS INSTEAD
------------------------------
The corrected over-representation run, against the measured assay universe.
It returns no term at BH q < 0.05 anywhere, so the figure's job is to show
that honestly and to show WHY the old result looked different — which is a
methodological point worth a panel, not an embarrassment to hide.

  a  every panel x database test, smallest nominal p against the BH line
  b  annotation coverage — how much of each panel and of the assay universe
     each database actually annotates, which bounds the achievable power
  c  descriptive term map for the 25-protein panel: which proteins land in
     which processes, with no enrichment claim attached
  d  what changed: background size, and the ontology scope that was NOT
     re-run (GO MF and CC), so the six withdrawn terms are recorded as
     withdrawn rather than refuted

Sources:
  revision_package_2026-09-04/05_data_audit/ora_assay_background_2026-09-05/
Output: figurev5/output/fig6_ora_assay_background.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

ROOT = Path("e:/Proteomics")
ORA = ROOT / "revision_package_2026-09-04" / "05_data_audit" / "ora_assay_background_2026-09-05"
OUT = ROOT / "figurev5" / "output"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 300

C_INK, C_MUTE = "#222222", "#6b7680"
DB_COL = {"GO_BP_2023": "#5b7fa6", "KEGG_2021_Human": "#c47a2c", "Reactome_2022": "#4c7a5c"}
DB_LAB = {"GO_BP_2023": "GO Biological Process", "KEGG_2021_Human": "KEGG", "Reactome_2022": "Reactome"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 11, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})

PANEL_LAB = {"multiclass_25": "25-protein"}


def nice(p):
    return PANEL_LAB.get(p, p.replace("single_", ""))


def panel_a(ax, s):
    """Smallest nominal p per test, against the BH-significance boundary."""
    order = sorted(s.panel.unique(), key=lambda x: (x != "multiclass_25", x))
    x = np.arange(len(order))
    for db in ["GO_BP_2023", "KEGG_2021_Human", "Reactome_2022"]:
        d = s[s.database == db].set_index("panel").reindex(order)
        ax.scatter(x, -np.log10(d.minimum_p.values), s=64, color=DB_COL[db],
                   edgecolor="white", linewidth=0.8, zorder=4, label=DB_LAB[db])
    ax.axhline(-np.log10(0.05), color="#33414d", lw=1.0, ls="--", zorder=2)
    ax.text(len(order) - 0.4, -np.log10(0.05) + 0.06, "nominal $p$ = 0.05",
            fontsize=8, fontweight="bold", color="#33414d", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([nice(p) for p in order], rotation=45, ha="right", fontsize=8.4)
    ax.set_ylabel("smallest $-\\log_{10}\\ p$ in test", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.2, loc="upper right", handletextpad=0.3)
    ax.set_title("a   No test reached significance after correction "
                 "(all 42 BH $q$ > 0.05)", loc="left", fontsize=11.5,
                 fontweight="bold", pad=6)


def panel_b(ax, m):
    """How much of each panel, and of the assay, each database annotates."""
    g = m.groupby("database").agg(
        ann_uni=("annotated_universe_size", "first"),
        uni=("assay_universe_size", "first")).reset_index()
    dbs = ["GO_BP_2023", "KEGG_2021_Human", "Reactome_2022"]
    g = g.set_index("database").reindex(dbs).reset_index()
    y = np.arange(len(dbs))
    frac = g.ann_uni / g.uni
    ax.barh(y, frac, color=[DB_COL[d] for d in dbs], edgecolor="white", height=0.55)
    for i, (f, a, u) in enumerate(zip(frac, g.ann_uni, g.uni)):
        ax.text(f + 0.015, i, f"{int(a):,} / {int(u):,}", va="center",
                fontsize=8.6, fontweight="bold", color=C_INK)
    ax.set_yticks(y)
    ax.set_yticklabels([DB_LAB[d] for d in dbs], fontsize=9)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("fraction of the 1,463-protein assay that is annotated", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    ax.set_title("b   Annotation coverage bounds the achievable power",
                 loc="left", fontsize=11.5, fontweight="bold", pad=6)


def panel_c(ax, t):
    """Descriptive: which 25-panel proteins land in which processes."""
    d = t[(t.panel == "multiclass_25")].copy()
    d = d.sort_values(["overlap_count", "p_value"], ascending=[False, True]).head(10)
    y = np.arange(len(d))[::-1]
    ax.barh(y, d.overlap_count.values, color=[DB_COL[x] for x in d.database],
            edgecolor="white", height=0.6)
    for yi, (n, genes) in zip(y, zip(d.overlap_count, d.overlap_genes)):
        ax.text(n + 0.08, yi, str(genes).replace(";", ", "), va="center",
                fontsize=7.4, fontweight="bold", color=C_MUTE)
    labs = [str(x)[:52] + ("..." if len(str(x)) > 52 else "") for x in d.term]
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=7.8)
    ax.set_xlabel("panel proteins annotated to the term", fontsize=9)
    ax.set_xlim(0, max(d.overlap_count) + 2.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("c   Where the 25 proteins sit, descriptively "
                 "(no term is enriched)", loc="left", fontsize=11.5,
                 fontweight="bold", pad=6)


def panel_d(ax):
    """What actually changed between the withdrawn run and this one."""
    ax.axis("off")
    ax.set_title("d   Why the earlier result differed", loc="left",
                 fontsize=11.5, fontweight="bold", pad=6)
    rows = [
        ("", "withdrawn run", "corrected run"),
        ("background", "whole annotated genome\n(18,737 / 19,960 genes)",
         "measured assay\n(1,463 proteins; 1,263 annotated)"),
        ("GO scope", "BP + MF + CC", "BP only"),
        ("terms at BH $q$ < 0.05", "6 (5 MF, 1 CC)", "0 of 42 tests"),
    ]
    xs = [0.02, 0.30, 0.63]
    y0, dy = 0.86, 0.225
    for r, row in enumerate(rows):
        y = y0 - r * dy
        for c, cell in enumerate(row):
            bold = (r == 0) or (c == 0)
            ax.text(xs[c], y, cell, transform=ax.transAxes, ha="left", va="top",
                    fontsize=8.8 if r else 9.4, fontweight="bold",
                    color=C_INK if bold else "#3a3a3a", linespacing=1.35)
        if r == 0:
            ax.plot([0.02, 0.98], [y - 0.055] * 2, transform=ax.transAxes,
                    color="#d5dbe0", lw=1.2)
    ax.text(0.02, y0 - len(rows) * dy - 0.02,
            "The six previously reported terms were Molecular Function and Cellular\n"
            "Component. Those ontologies were not re-run, so those terms are\n"
            "withdrawn rather than refuted.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            fontweight="bold", color="#9aa4ad", linespacing=1.5)


def main():
    s = pd.read_csv(ORA / "ora_summary.csv")
    m = pd.read_csv(ORA / "ora_mapping_and_test_counts.csv")
    t = pd.read_csv(ORA / "ora_terms_with_panel_overlap.csv")

    fig = plt.figure(figsize=(17.5, 10.4))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0],
                           hspace=0.52, wspace=0.30,
                           left=0.055, right=0.98, top=0.87, bottom=0.075)
    panel_a(fig.add_subplot(gs[0, :]), s)
    panel_b(fig.add_subplot(gs[1, 0]), m)
    panel_c(fig.add_subplot(gs[1, 1]), t)

    fig.suptitle("Over-representation against the measured assay: no panel-level "
                 "enrichment survives correction",
                 fontsize=16.5, fontweight="bold", x=0.055, ha="left", y=0.965)
    fig.text(0.055, 0.925,
             "42 panel-by-database tests against the 1,463-protein assay universe. "
             "Smallest adjusted value anywhere was $q$ = 0.059.",
             fontsize=9.8, fontweight="bold", color=C_MUTE, ha="left", va="top")

    for ext in ("png", "pdf"):
        p = OUT / f"fig6_ora_assay_background.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)

    # Panel d is rendered as a standalone note figure so the main figure stays
    # data-only; it is a methods explanation, not a result.
    f2 = plt.figure(figsize=(8.6, 3.6))
    panel_d(f2.add_subplot(111))
    for ext in ("png", "pdf"):
        p = OUT / f"fig6_supp_background_change.{ext}"
        f2.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(f2)


if __name__ == "__main__":
    print("Building fig6 (corrected assay-background ORA) ...")
    main()
    print("Done.")
