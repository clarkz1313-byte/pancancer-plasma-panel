#!/usr/bin/env python3
"""
Fig 6 (Part A pathway family) — single-panel pathway enrichment · 12-cancer mosaic

Each subplot: top GO BP terms for that cancer's DEA-significant proteins
              (background = that cancer's selected feature pool, ~100-300 proteins)
Single-panel proteins highlighted in red; others in grey.
Layout: 4 rows × 3 cols.

STATUS: superseded alternative, kept for provenance. The PI selected
`fig6_parta_opt3_legacy_dea` (panel-direct ORA) for the deck instead; this
mosaic is not currently placed on any slide.

RENAMED 2026-08-13 from `fig9_single_panel_pathway.py` (stem
`fig9_single_panel_pathway_mosaic`). The old name predated the slide-8/9
split, where `fig9_*` became the external-validation multi-panel slide —
keeping it would have made "fig9" mean two unrelated things.
"""
from __future__ import annotations
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path("e:/Proteomics")
LEG  = ROOT / "revise_plan" / "legacy_pathway_enrichment" / "result12classes" / "tables" / "enrichment"
PKG  = ROOT / "revise_plan" / "locked25_final_results_package"
OUT  = ROOT / "figurev5" / "output"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "font.weight": "bold",
    "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.9,
})
DPI = 300
_SP = dict(color="#666666", lw=0.8)

CANCER_ORDER = ["AML","BRC","CLL","CRC","CVX","ENDC","GLIOM","LUNGC","LYMPH","MYEL","OVC","PRC"]
CANCER_COLOR = {
    "AML":"#b22222","BRC":"#c97b63","CLL":"#7a3e9d",
    "CRC":"#d68600","CVX":"#c13d86","ENDC":"#8e5d2c",
    "GLIOM":"#1a8db8","LUNGC":"#2e8b57","LYMPH":"#3856a6",
    "MYEL":"#8c564b","OVC":"#d1495b","PRC":"#008b8b",
}
CANCER_FULL = {
    "AML":"AML","BRC":"Breast Cancer","CLL":"CLL",
    "CRC":"Colorectal Cancer","CVX":"Cervical Cancer",
    "ENDC":"Endometrial Cancer","GLIOM":"Glioma",
    "LUNGC":"Lung Cancer","LYMPH":"DLBCL",
    "MYEL":"Myeloma","OVC":"Ovarian Cancer","PRC":"Prostate Cancer",
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


def load_single_panel_proteins(cancer: str) -> list[str]:
    """Return the deployed single-panel protein list for one cancer."""
    si = list(PKG.rglob("single_internal.csv"))
    for f in si:
        df = pd.read_csv(f)
        row = df[df["target_class"].str.upper() == cancer.upper()]
        if not row.empty:
            raw = str(row.iloc[0]["deploy_proteins"])
            return [p.strip() for p in raw.split(";") if p.strip()]
    return []


def load_go_bp(cancer: str, n_terms: int = 5) -> pd.DataFrame:
    """Top n GO BP terms (by p_adjusted_bh) for a cancer's DEA pool."""
    f = LEG / f"{cancer.lower()}_go_bp_ora.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    # Keep terms where p_adjusted_bh < 0.05, or fall back to top n by p_value
    sig = df[df["p_adjusted_bh"] < 0.05]
    if len(sig) >= n_terms:
        return sig.nsmallest(n_terms, "p_value").reset_index(drop=True)
    # fallback: nominal top n
    return df.nsmallest(n_terms, "p_value").reset_index(drop=True)


def panel(ax, cancer: str, single_prots: list[str], go_df: pd.DataFrame,
          first_col: bool, first_row: bool):
    col = CANCER_COLOR[cancer]
    ax.set_facecolor("#fafafa")

    if go_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#888888")
        ax.set_title(CANCER_FULL[cancer], color=col,
                     fontsize=11, fontweight="bold", pad=4)
        _spine_style(ax)
        return

    n = len(go_df)
    ypos = np.arange(n)
    go_df = go_df.copy().reset_index(drop=True)
    go_df["logp"]  = -np.log10(go_df["p_value"].clip(lower=1e-10))
    go_df["logfdr"] = -np.log10(go_df["p_adjusted_bh"].clip(lower=1e-10))
    # Sort ascending so best term is at top (reversed y)
    go_df = go_df.sort_values("p_value", ascending=False).reset_index(drop=True)

    fdr_cut = -np.log10(0.05)

    for i, row in go_df.iterrows():
        logp    = row["logp"]
        is_sig  = row["logfdr"] >= fdr_cut
        bar_col = col if is_sig else "#cccccc"

        ax.barh(i, logp, color=bar_col, alpha=0.85,
                height=0.62, edgecolor="white", linewidth=0.8, zorder=2)

        # Highlight single-panel proteins in the overlapping genes
        overlapping = str(row.get("overlapping_genes", "")).split(";")
        panel_hit = [p.strip() for p in overlapping if p.strip() in single_prots]
        other_hit = [p.strip() for p in overlapping
                     if p.strip() and p.strip() not in single_prots]

        # Annotate proteins to the right of bar
        if panel_hit:
            prot_str = ", ".join(panel_hit)
            ax.text(logp + 0.08, i, prot_str,
                    va="center", ha="left", fontsize=6.5,
                    fontweight="bold", color="#cc0000", clip_on=False)
        if other_hit and not panel_hit:
            # Only show if no panel hits (avoid clutter)
            ax.text(logp + 0.08, i, other_hit[0],
                    va="center", ha="left", fontsize=6,
                    color="#888888", clip_on=False, fontstyle="italic")

    # FDR line
    ax.axvline(fdr_cut, color="#cc0000", lw=1.0, ls="--", alpha=0.7, zorder=1)

    # Y-axis: wrapped term labels
    ax.set_yticks(ypos)
    ax.set_yticklabels(
        ["\n".join(textwrap.wrap(
            str(t).replace("(GO:", "\n(GO:") if len(str(t)) > 30 else str(t), 28))
         for t in go_df["term_name"]],
        fontsize=6.5, fontweight="bold")
    ax.set_ylim(-0.6, n - 0.4)

    xmax = go_df["logp"].max() * 1.3
    ax.set_xlim(0, max(xmax, fdr_cut * 1.5))
    ax.tick_params(axis="x", labelsize=7, length=2, pad=1)
    ax.tick_params(axis="y", length=0)

    if first_col:
        pass  # y-labels already shown
    ax.set_xlabel("-log10(p)", fontsize=7, fontweight="bold", labelpad=2)

    ax.set_title(CANCER_FULL[cancer], color=col,
                 fontsize=11, fontweight="bold", pad=4)

    # Single-panel proteins badge (top-left of each subplot)
    sp_str = " · ".join(single_prots[:4])
    if len(single_prots) > 4:
        sp_str += f" +{len(single_prots)-4}"
    ax.text(0.01, 0.99, sp_str, transform=ax.transAxes,
            ha="left", va="top", fontsize=5.5,
            color="#cc0000", fontweight="bold", fontstyle="italic")

    ax.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    _spine_style(ax)


# ─── Build 4×3 mosaic ────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 3, figsize=(21, 22))
fig.subplots_adjust(wspace=0.52, hspace=0.55,
                    left=0.06, right=0.96, top=0.94, bottom=0.04)

for idx, cancer in enumerate(CANCER_ORDER):
    row, col = divmod(idx, 3)
    ax = axes[row][col]

    single_prots = load_single_panel_proteins(cancer)
    go_df = load_go_bp(cancer, n_terms=5)
    panel(ax, cancer, single_prots, go_df,
          first_col=(col == 0), first_row=(row == 0))

fig.suptitle(
    "Single-panel pathway context — GO Biological Process (top 5 per cancer)\n"
    "Background = cancer-specific DEA feature pool · "
    "red labels = deployed single-panel proteins · dashed = FDR 0.05",
    fontsize=11, fontweight="bold", y=0.97)

# Global legend
leg_handles = [
    mpatches.Patch(facecolor="#b22222", label="FDR < 0.05 (colored bar)"),
    mpatches.Patch(facecolor="#cccccc", label="Nominal only"),
    mpatches.Patch(facecolor="white", edgecolor="#cc0000",
                   label="Red label = in single panel"),
]
fig.legend(handles=leg_handles, fontsize=9, framealpha=0.92,
           loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.00))

save(fig, "fig6_parta_pathway_mosaic")
print("Done.")
