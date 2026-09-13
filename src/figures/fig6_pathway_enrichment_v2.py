#!/usr/bin/env python3
"""
fig6_pathway_enrichment_v2.py -- panel a of slide 6, rebuilt 2026-09-05.

REPLACES the withdrawn `fig6_panels.py` bubble row.

WHY THE OLD PANEL WENT
----------------------
The old bubbles were drawn from `go_significant_panel_terms.csv`, whose six
"FDR-significant" GO terms carried BgRatio denominators of 18,737 and 19,960 --
the whole annotated genome. Its own header nevertheless read "against a
1,463-protein assayed background", which was not what had been computed. The
KEGG and Reactome sources behind the same row used 9,380 and 11,146.

WHAT THIS PANEL DOES INSTEAD
----------------------------
Same visual language -- gene ratio on x, bubble area = overlap count, colour =
-log10(p), member proteins named beside each bubble -- but computed against the
measured assay universe, and with the honest outcome stated on the panel: no
term survives BH correction anywhere.

The second row is the diagnostic that explains the change. Holding the gene-set
library, the ontology scope and the test code FIXED, and varying only the
background, 9 GO Biological Process terms and 3 Reactome terms lose
significance. The background is the whole story; the other three differences
between the runs are not.

Sources:
  05_data_audit/ora_assay_background_2026-09-05/ora_terms_with_panel_overlap.csv
  05_data_audit/ora_assay_background_2026-09-05/ora_background_isolation_2026-09-05.csv
Output: figurev5/output/fig6_pathway_enrichment.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ORA = ROOT / "revision_package_2026-09-04" / "05_data_audit" / "ora_assay_background_2026-09-05"
OUT = ROOT / "figurev6" / "output"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 400

DB = [
    ("GO_BP_2023",      "GO Biological Process", "#7b3fa0", "#f4eefa", "RdPu"),
    ("KEGG_2021_Human", "KEGG Pathways",         "#c4761c", "#fdf6e7", "YlOrBr"),
    ("Reactome_2022",   "Reactome Pathways",     "#2e7d5b", "#eaf5f0", "GnBu"),
]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 11, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.linewidth": 1.3, "xtick.major.width": 1.3, "ytick.major.width": 1.3,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
})


def wrap(s, w=34):
    words, lines, cur = str(s).split(), [], ""
    for x in words:
        if len(cur) + len(x) + 1 <= w:
            cur = (cur + " " + x).strip()
        else:
            lines.append(cur)
            cur = x
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def clean_term(t):
    t = str(t)
    t = t.split(" R-HSA-")[0]
    t = t.split(" (GO:")[0]
    substitutions = {
        "Lymphoid Progenitor Cell Differentiation": "Lymphoid progenitor\ndifferentiation",
        "Intermediate Filament Organization": "Intermediate filament\norganization",
        "Regulation Of Cellular Component Biogenesis": "Cellular component\nbiogenesis",
        "Regulation Of Programmed Cell Death": "Programmed cell death",
        "Amyotrophic lateral sclerosis": "ALS",
        "Epstein-Barr virus infection": "EBV infection",
        "Acute myeloid leukemia": "AML",
        "Arachidonic acid metabolism": "Arachidonic acid\nmetabolism",
        "Signaling By NOTCH2": "NOTCH2 signaling",
        "Defective B3GALTL Causes PpS": "B3GALTL / PpS",
        "O-glycosylation Of TSR Domain-Containing Proteins": "TSR O-glycosylation",
        "Diseases Associated With O-glycosylation Of Proteins": "O-glycosylation\ndisorders",
    }
    return substitutions.get(t, t)


def bubble_panel(fig, ax, d, label, colour, bg, cmap, n_panel):
    ax.set_facecolor(bg)
    if d.empty:
        ax.text(.5, .5, "no term with overlap", ha="center", va="center",
                transform=ax.transAxes, color=colour)
        return
    d = d.sort_values("p_value").head(4).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(d))
    ratio = d.overlap_count / n_panel
    size = 60 + (d.overlap_count / max(d.overlap_count.max(), 1)) * 340
    col = -np.log10(d.p_value)

    sc = ax.scatter(ratio, y, s=size, c=col, cmap=cmap, edgecolor="#33414d",
                    linewidth=1.0, zorder=5,
                    vmin=col.min() * 0.92, vmax=col.max() * 1.03)
    ax.set_yticks(y)
    # Keep labels readable but reserve a real gutter between neighbouring
    # ontology panels; long y labels previously ran into the next plot.
    ax.set_yticklabels([wrap(clean_term(t), 19) for t in d.term], fontsize=14.5)
    ax.tick_params(axis="y", pad=5)
    ax.set_xlabel("Gene ratio", fontsize=16)
    ax.tick_params(axis="x", labelsize=12.5)
    ax.set_xlim(0, max(ratio.max() * 1.35, 0.22))
    ax.set_ylim(-0.75, len(d) - 0.25)
    ax.grid(axis="x", color="white", lw=1.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#b9c2ca")
    # A short source tag is sufficient here; verbose panel titles belong in
    # the manuscript caption, not in a dense multipanel graphic.
    short_label = {"GO Biological Process": "GO",
                   "KEGG Pathways": "KEGG",
                   "Reactome Pathways": "Reactome"}[label]
    ax.text(0.02, 1.015, short_label, transform=ax.transAxes,
            fontsize=16, fontweight="bold", color=colour, va="bottom")
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.075,
                      pad=0.20, aspect=26)
    cb.set_label("$-\\log_{10}\\ P$", fontsize=18, fontweight="bold")
    cb.ax.tick_params(labelsize=11)


def isolation_panel(ax, iso):
    """Same library, same test, two backgrounds."""
    order = ["GO_BP_2023", "Reactome_2022"]
    lab = {"GO_BP_2023": "GO Biological Process", "Reactome_2022": "Reactome"}
    C_FULL, C_ASSAY = "#c0392b", "#2e7d5b"

    yticks, ylabels = [], []
    row = 0
    for db in order:
        for bgname, colr in (("full_annotation", C_FULL), ("measured_assay", C_ASSAY)):
            d = iso[(iso.database == db) & (iso.background == bgname)]
            if d.empty:
                continue
            q = d.bh_q.values
            n_sig = int((q < 0.05).sum())
            uni = int(d.universe_size.iloc[0])
            jitter = (np.random.RandomState(0).rand(len(q)) - 0.5) * 0.26
            ax.scatter(-np.log10(np.clip(q, 1e-6, None)), row + jitter,
                       s=95, color=colr, alpha=0.80, edgecolor="none", zorder=3)
            ax.scatter([-np.log10(max(q.min(), 1e-6))], [row], s=240, marker="D",
                       color=colr, edgecolor="white", linewidth=1.3, zorder=6)
            ax.text(-np.log10(max(q.min(), 1e-6)) + 0.10, row,
                    f"qmin {q.min():.3f}  |  n={n_sig}",
                    va="center", fontsize=18, fontweight="bold", color=colr)
            yticks.append(row)
            ylabels.append(f"{'GO' if db == 'GO_BP_2023' else 'Reactome'}  |  "
                           f"{'Full' if bgname=='full_annotation' else 'Assay'}")
            row += 1
        row += 0.55

    ax.axvline(-np.log10(0.05), color="#33414d", lw=1.2, ls="--", zorder=2)
    ax.annotate("q = 0.05", xy=(-np.log10(0.05), 1),
                xycoords=("data", "axes fraction"), xytext=(7, -6),
                textcoords="offset points", fontsize=15, fontweight="bold",
                color="#33414d", va="top",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=18, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("$-\\log_{10}$ BH $q$", fontsize=16)
    ax.tick_params(axis="x", labelsize=12.5)
    ax.set_xlim(-0.05, None)
    ax.grid(axis="x", color="#f0f0f0", lw=1.1, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        Line2D([0], [0], marker="D", ls="", ms=9, mfc=C_FULL, mec="white",
               label="Full annotation"),
        Line2D([0], [0], marker="D", ls="", ms=9, mfc=C_ASSAY, mec="white",
               label="Measured assay")],
        loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
        frameon=False, fontsize=17)


def main():
    t = pd.read_csv(ORA / "ora_terms_with_panel_overlap.csv")
    t25 = t[t.panel == "multiclass_25"]
    iso_path = ORA / "ora_background_isolation_2026-09-05.csv"

    fig = plt.figure(figsize=(17.6, 9.4))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.0, 0.66],
                           hspace=0.50, wspace=0.68,
                           left=0.070, right=0.985, top=0.94, bottom=0.125)

    for i, (db, label, colour, bg, cmap) in enumerate(DB):
        bubble_panel(fig, fig.add_subplot(gs[0, i]),
                     t25[t25.database == db], label, colour, bg, cmap, 25)

    if iso_path.exists():
        isolation_panel(fig.add_subplot(gs[1, :]), pd.read_csv(iso_path))

    for ext in ("png", "pdf"):
        p = OUT / f"fig6_pathway_enrichment.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


if __name__ == "__main__":
    print("Building fig6 panel a (corrected assay-background ORA) ...")
    main()
    print("Done.")
