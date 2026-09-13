#!/usr/bin/env python3
"""
Fig 12 — SLIDE 12: Hallmark GSEA on the full proteome.

Rebuilt 2026-08-13 in figurev5 style (Arial bold, CANCER_COLOR, save()
PDF+PNG) to replace the June 2026 `revise_plan/gsea_analysis/figures/`
"Prototype C" figure, which predated the figurev5 design system and stacked
five competing encodings on one axes (NES fill + M# boxes + S# boxes + gold
strip + green strip + a six-line legend).

What this analysis is and is NOT
--------------------------------
GSEA was run on the FULL 1,463-protein list per cancer (one-vs-rest ranked
lists), NOT on the deployed panels, and it was NOT used to select any panel.
It answers one preliminary question: does the proteome carry recognisable
cancer biology, or is it a random protein list? So this slide is a sanity /
context layer, not a headline performance claim. FDR < 0.25 is a permissive
threshold and is stated on every panel rather than buried in a subtitle.

  12a  NES heatmap, significant cells only, with ONE panel-overlap marker
       (replaces the old four-way M#/S#/gold/green annotation)
  12b  significant Hallmark count per cancer
  12c  Hallmark breadth — how many cancers each term is significant in

Source: revise_plan/gsea_analysis/figures/
        gsea_hallmarks_all_panels_overlay_source_data_fdr25.csv
Output: figurev5/output/fig12{a,b,c}_*.pdf/.png
Run:    python fig12_panels.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from figure_labels import display_label

ROOT = Path("e:/Proteomics")
SRC = ROOT / "revise_plan/gsea_analysis/figures/gsea_hallmarks_all_panels_overlay_source_data_fdr25.csv"
OUT = ROOT / "figurev5/output"
OUT.mkdir(parents=True, exist_ok=True)

CANCERS = ["AML", "BRC", "CLL", "CRC", "CVX", "ENDC", "GLIOM", "LUNGC",
           "LYMPH", "MYEL", "OVC", "PRC"]
CANCER_COLOR = {
    "AML": "#b22222", "BRC": "#c97b63", "CLL": "#7a3e9d", "CRC": "#d68600",
    "CVX": "#c13d86", "ENDC": "#8e5d2c", "GLIOM": "#1a8db8", "LUNGC": "#2e8b57",
    "LYMPH": "#3856a6", "MYEL": "#8c564b", "OVC": "#d1495b", "PRC": "#008b8b",
}
# panel-overlap marker colours -- ONE encoding, three states, matching the
# MACRO/panel conventions used elsewhere in figurev5
MULTI_COL = "#d8a200"    # locked 25-protein multiclass panel
SINGLE_COL = "#2a9d5c"   # cancer-specific single panel
BOTH_COL = "#7a3e9d"     # both panel types hit the same cell

FDR = 0.25
DPI = 300

plt.rcParams.update({
    "font.family": "Arial", "font.size": 12, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
})


def save(fig, stem):
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


def load():
    d = pd.read_csv(SRC)
    d["sig"] = d["significant_under_threshold"].astype(bool)
    return d


# ── 12a: NES heatmap ─────────────────────────────────────────────────────────
def panel_nes_heatmap(d):
    sig = d[d["sig"]]
    # keep only Hallmarks significant somewhere; order by breadth then |NES|
    breadth = sig["Term"].value_counts()
    terms = list(breadth.index)
    strength = sig.groupby("Term")["NES"].apply(lambda s: s.abs().mean())
    terms.sort(key=lambda t: (-breadth[t], -strength.get(t, 0)))
    # cancers ordered by how much pathway signal they carry
    csig = sig["cancer_type"].value_counts()
    cancers = sorted(CANCERS, key=lambda c: -csig.get(c, 0))

    M = np.full((len(terms), len(cancers)), np.nan)
    cat = {}
    for _, r in sig.iterrows():
        if r["Term"] in terms and r["cancer_type"] in cancers:
            i, j = terms.index(r["Term"]), cancers.index(r["cancer_type"])
            M[i, j] = r["NES"]
            cat[(i, j)] = r["support_category"]

    vmax = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(11.5, 12.0))
    fig.subplots_adjust(left=0.30, right=0.90, top=0.90, bottom=0.11)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "nes", ["#1a4b8c", "#7fb3d9", "#f4f4f4", "#e8927c", "#a01e28"])
    ax.set_facecolor("white")
    im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    # faint grid so empty (non-significant) cells still read as tested cells
    for i in range(len(terms) + 1):
        ax.axhline(i - 0.5, color="#e6e6e6", lw=0.7, zorder=1)
    for j in range(len(cancers) + 1):
        ax.axvline(j - 0.5, color="#e6e6e6", lw=0.7, zorder=1)

    # ONE panel-overlap marker per cell (was: 2 count boxes + 2 side strips)
    for (i, j), c in cat.items():
        if c == "significant_no_panel":
            continue
        col = {"multi_only": MULTI_COL, "single_only": SINGLE_COL,
               "both": BOTH_COL}.get(c)
        if col:
            ax.scatter(j, i, s=95, marker="o", facecolor=col,
                       edgecolor="white", linewidth=1.6, zorder=4)

    ax.set_xticks(range(len(cancers)))
    ax.set_xticklabels([display_label(c) for c in cancers],
                       fontsize=12, fontweight="bold",
                       rotation=40, ha="right", rotation_mode="anchor")
    for t, c in zip(ax.get_xticklabels(), cancers):
        t.set_color(CANCER_COLOR[c])
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms, fontsize=10.5, fontweight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("Normalized Enrichment Score (NES)", fontsize=12,
                 fontweight="bold")
    cb.outline.set_linewidth(1.0)

    handles = [
        plt.Line2D([0], [0], marker="o", ls="", ms=10, mfc=MULTI_COL,
                   mec="white", mew=1.6, label="locked 25-protein panel protein in leading edge"),
        plt.Line2D([0], [0], marker="o", ls="", ms=10, mfc=SINGLE_COL,
                   mec="white", mew=1.6, label="single-panel protein in leading edge"),
        plt.Line2D([0], [0], marker="o", ls="", ms=10, mfc=BOTH_COL,
                   mec="white", mew=1.6, label="both panel types"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10.5,
               bbox_to_anchor=(0.58, -0.005), framealpha=0.94)

    ax.set_title(
        "Hallmark GSEA on the full 1,463-protein proteome\n"
        f"One-vs-rest ranked list per cancer  *  significant cells only, FDR < {FDR}\n"
        "Red = enriched toward that cancer  *  blue = depleted  *  blank = not significant",
        fontsize=13, fontweight="bold", pad=14)

    save(fig, "fig12a_gsea_nes_heatmap")


# ── 12b: significant Hallmark count per cancer ───────────────────────────────
def panel_counts_per_cancer(d):
    sig = d[d["sig"]]
    counts = {c: int((sig["cancer_type"] == c).sum()) for c in CANCERS}
    order = sorted(CANCERS, key=lambda c: -counts[c])
    vals = [counts[c] for c in order]

    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.82, bottom=0.12)

    bars = ax.bar(range(len(order)), vals,
                  color=[CANCER_COLOR[c] for c in order],
                  edgecolor="white", linewidth=1.6, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.35, str(v),
                ha="center", va="bottom", fontsize=13, fontweight="bold",
                color="#222222")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([display_label(c) for c in order],
                       fontsize=12, fontweight="bold",
                       rotation=40, ha="right", rotation_mode="anchor")
    for t, c in zip(ax.get_xticklabels(), order):
        t.set_color(CANCER_COLOR[c])
    ax.set_ylabel("Significant Hallmark pathways", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    n_zero = sum(1 for v in vals if v == 0)
    zero_note = (f"  *  {n_zero} cancer with no significant Hallmark"
                 if n_zero == 1 else
                 f"  *  {n_zero} cancers with none" if n_zero else "")
    ax.set_title(
        "How much Hallmark signal each cancer carries\n"
        f"Count of Hallmark pathways significant at FDR < {FDR}{zero_note}",
        fontsize=13, fontweight="bold", pad=12)

    save(fig, "fig12b_gsea_counts_per_cancer")


# ── 12c: Hallmark breadth ────────────────────────────────────────────────────
def panel_term_breadth(d):
    sig = d[d["sig"]]
    breadth = sig["Term"].value_counts()
    breadth = breadth[breadth >= 2].sort_values()   # shared terms only
    # colour each bar by a stack of the contributing cancers' colours would be
    # over-busy; use a neutral fill and let 12a carry the per-cancer detail
    fig, ax = plt.subplots(figsize=(9.6, 7.6))
    fig.subplots_adjust(left=0.42, right=0.96, top=0.85, bottom=0.10)

    ypos = range(len(breadth))
    ax.barh(list(ypos), breadth.values, color="#5a7fa6",
            edgecolor="white", linewidth=1.4, zorder=3)
    for y, v in zip(ypos, breadth.values):
        ax.text(v + 0.08, y, str(int(v)), va="center", ha="left",
                fontsize=12, fontweight="bold", color="#222222")

    ax.set_yticks(list(ypos))
    ax.set_yticklabels(breadth.index, fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of cancers where significant", fontsize=13,
                  fontweight="bold")
    ax.set_xlim(0, breadth.values.max() * 1.14)
    ax.set_xticks(range(0, int(breadth.values.max()) + 1))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    ax.set_title(
        "Shared vs cancer-specific Hallmark biology\n"
        f"Hallmarks significant (FDR < {FDR}) in 2 or more cancers\n"
        "Terms significant in only one cancer are omitted here — see 12a",
        fontsize=13, fontweight="bold", pad=12)

    save(fig, "fig12c_gsea_term_breadth")


if __name__ == "__main__":
    print("Building fig12 panels (slide 12 — Hallmark GSEA) ...")
    d = load()
    n_sig = int(d["sig"].sum())
    print(f"  {len(d)} cancer x Hallmark cells, {n_sig} significant at FDR<{FDR}")
    panel_nes_heatmap(d)
    panel_counts_per_cancer(d)
    panel_term_breadth(d)
    print("\nDone.")
