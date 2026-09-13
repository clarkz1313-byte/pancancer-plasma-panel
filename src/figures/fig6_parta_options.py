#!/usr/bin/env python3
"""
Fig 6 — Part A single-panel pathway options (for PI selection)
  parta_opt1: Protein functional annotation heatmap (12 cancers × 10 function categories)
  parta_opt2: Per-cancer feature importance lollipop (12-panel mosaic, dots = proteins, color = function)
  parta_opt3: Legacy DEA pathway mosaic with -log10(p) gradient + single-panel highlight (improved fig9)
All renamed to fig6_* to belong to slide 6 family.
"""
from __future__ import annotations
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path("e:/Proteomics")
PKG  = ROOT / "revise_plan" / "locked25_final_results_package"
LEG  = ROOT / "revise_plan" / "legacy_pathway_enrichment" / "result12classes" / "tables" / "enrichment"
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

# Curated protein → primary function category
FUNC = {
    # Immune/hematopoietic
    "FLT3":"Immune/Hematop.","CD244":"Immune/Hematop.","FCER2":"Immune/Hematop.",
    "TCL1A":"Immune/Hematop.","SLAMF7":"Immune/Hematop.","FCRLB":"Immune/Hematop.",
    "MZB1":"Immune/Hematop.","TNFRSF13C":"Immune/Hematop.","TNFRSF10A":"Immune/Hematop.",
    "C1QA":"Immune/Hematop.","PDCD1":"Immune/Hematop.","CORO1A":"Immune/Hematop.",
    "PGLYRP1":"Immune/Hematop.",
    # Cytokine/Chemokine
    "CXCL8":"Cytokine","CXCL13":"Cytokine","CXCL17":"Cytokine",
    # Growth factor / receptor
    "BMP4":"Growth factor","BMP6":"Growth factor","ARTN":"Growth factor",
    "NTF3":"Growth factor","IGFBP1":"Growth factor","AREG":"Growth factor",
    "ENPP2":"Growth factor","LEFTY2":"Growth factor","PROK1":"Growth factor",
    "NTRK3":"Growth factor","ADGRG1":"Growth factor","PSPN":"Growth factor",
    # ECM / adhesion
    "SDC4":"ECM/Adhesion","SDC1":"ECM/Adhesion","SPOCK1":"ECM/Adhesion",
    "LGALS4":"ECM/Adhesion","CDH15":"ECM/Adhesion","CNTN5":"ECM/Adhesion",
    "CNTN3":"ECM/Adhesion","TSPAN1":"ECM/Adhesion","BCAN":"ECM/Adhesion",
    "FAP":"ECM/Adhesion","CRTAC1":"ECM/Adhesion","CDHR2":"ECM/Adhesion","S100A4":"ECM/Adhesion",
    # Enzyme / protease
    "LTA4H":"Enzyme/Protease","ADAMTS13":"Enzyme/Protease","KLK14":"Enzyme/Protease",
    "KLK13":"Enzyme/Protease","THOP1":"Enzyme/Protease","HAGH":"Enzyme/Protease",
    "PRDX6":"Enzyme/Protease","PRDX5":"Enzyme/Protease","GLO1":"Enzyme/Protease",
    "CA3":"Enzyme/Protease","PSMD9":"Enzyme/Protease","ALPP":"Enzyme/Protease",
    "QPCT":"Enzyme/Protease","AFP":"Enzyme/Protease",
    # Cancer / secreted marker
    "WFDC2":"Cancer marker","PAEP":"Cancer marker","CGA":"Cancer marker",
    "MSMB":"Cancer marker","TACSTD2":"Cancer marker","SEZ6L2":"Cancer marker",
    "SPINK4":"Cancer marker","CEACAM5":"Cancer marker",
    # Signaling
    "PSIP1":"Signaling","ILKAP":"Signaling","NFATC1":"Signaling",
    "TRAF2":"Signaling","BID":"Signaling",
    # Neural / structural
    "GFAP":"Neural/Struct.","NEFL":"Neural/Struct.",
    # Metabolic / endocrine
    "GCG":"Metabolic/Endocr.","LEP":"Metabolic/Endocr.","PLIN1":"Metabolic/Endocr.",
    "CHGB":"Metabolic/Endocr.","VAT1":"Metabolic/Endocr.","TFRC":"Metabolic/Endocr.",
    "XG":"Metabolic/Endocr.","FKBP1B":"Metabolic/Endocr.","CRNN":"Metabolic/Endocr.",
    "F3":"Metabolic/Endocr.",
}
FUNC_COL = {
    "Immune/Hematop.":  "#7a3e9d",
    "Cytokine":         "#c13d86",
    "Growth factor":    "#2e8b57",
    "ECM/Adhesion":     "#008b8b",
    "Enzyme/Protease":  "#d1495b",
    "Cancer marker":    "#b22222",
    "Signaling":        "#d68600",
    "Neural/Struct.":   "#8c564b",
    "Metabolic/Endocr.":"#3856a6",
    "Other":            "#aaaaaa",
}
FUNC_ORDER = list(FUNC_COL.keys())


def save(fig, stem):
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


def _spine(ax):
    for sp in ax.spines.values():
        sp.set_color(_SP["color"]); sp.set_linewidth(_SP["lw"])


def get_deploy_proteins(si_df, cancer):
    row = si_df[si_df["target_class"].str.upper() == cancer.upper()]
    if row.empty:
        return []
    return [p.strip() for p in str(row.iloc[0]["deploy_proteins"]).split(";") if p.strip()]


def get_importance(pkgs, cancer, proteins):
    files = list(pkgs.rglob(f"{cancer.lower()}_top_feature_ranking.csv"))
    if not files:
        return {p: 0 for p in proteins}
    df = pd.read_csv(files[0])
    mp = dict(zip(df["protein"], df["importance"]))
    return {p: mp.get(p, 0) for p in proteins}


# ─── Load single-panel data ──────────────────────────────────────────────────
si_files = list(PKG.rglob("single_internal.csv"))
si_df    = pd.read_csv(si_files[0]) if si_files else pd.DataFrame()

deploy = {c: get_deploy_proteins(si_df, c) for c in CANCER_ORDER}


# ══════════════════════════════════════════════════════════════════════════════
# OPTION 1 — Functional annotation heatmap (12 cancers × function categories)
# ══════════════════════════════════════════════════════════════════════════════
def build_opt1():
    # Build binary matrix: cancer × function category
    mat = pd.DataFrame(0, index=CANCER_ORDER, columns=FUNC_ORDER)
    count_mat = pd.DataFrame(0, index=CANCER_ORDER, columns=FUNC_ORDER)

    for cancer in CANCER_ORDER:
        for prot in deploy[cancer]:
            cat = FUNC.get(prot, "Other")
            if cat in mat.columns:
                mat.loc[cancer, cat] = 1
                count_mat.loc[cancer, cat] += 1

    # Remove empty columns
    mat = mat.loc[:, mat.sum() > 0]
    count_mat = count_mat[mat.columns]

    nc = len(mat.columns)
    nr = len(mat.index)

    fig, ax = plt.subplots(figsize=(nc * 1.4, nr * 0.9 + 2))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.84, bottom=0.28)

    for ci, cat in enumerate(mat.columns):
        fc_on  = FUNC_COL.get(cat, "#aaaaaa")
        fc_off = "#f4f4f4"
        for ri, cancer in enumerate(mat.index):
            val   = mat.loc[cancer, cat]
            cnt   = count_mat.loc[cancer, cat]
            color = fc_on if val else fc_off
            ax.add_patch(plt.Rectangle(
                (ci, ri), 1, 1,
                facecolor=color, edgecolor="white",
                linewidth=1.5, zorder=2))
            if cnt > 0:
                ax.text(ci + 0.5, ri + 0.5, str(cnt),
                        ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if val else "#888888", zorder=3)

    # Cancer labels (y-axis)
    ax.set_yticks([i + 0.5 for i in range(nr)])
    ax.set_yticklabels(
        [f"{CANCER_FULL[c]}" for c in CANCER_ORDER],
        fontsize=9, fontweight="bold")
    # Color cancer labels
    for label, cancer in zip(ax.get_yticklabels(), CANCER_ORDER):
        label.set_color(CANCER_COLOR[cancer])

    # Category labels (x-axis, rotated)
    ax.set_xticks([i + 0.5 for i in range(nc)])
    ax.set_xticklabels(list(mat.columns), rotation=40, ha="right",
                       rotation_mode="anchor", fontsize=8.5, fontweight="bold")

    ax.set_xlim(0, nc); ax.set_ylim(0, nr)
    ax.tick_params(length=0, pad=4)

    # Category color header strip
    for ci, cat in enumerate(mat.columns):
        ax.add_patch(plt.Rectangle(
            (ci, nr), 1, 0.4,
            facecolor=FUNC_COL.get(cat, "#aaa"),
            edgecolor="white", linewidth=1.2,
            zorder=3, clip_on=False))

    _spine(ax)
    ax.set_title(
        "Option 1 — Single-panel functional annotation heatmap\n"
        "Rows = 12 cancers · Columns = protein function categories · "
        "Number = count of single-panel proteins with that function",
        fontsize=10, fontweight="bold", pad=24)

    save(fig, "fig6_parta_opt1_annotation_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# OPTION 2 — Per-cancer feature importance lollipop (12-panel mosaic)
# ══════════════════════════════════════════════════════════════════════════════
def build_opt2():
    fig, axes = plt.subplots(4, 3, figsize=(21, 22))
    fig.subplots_adjust(wspace=0.52, hspace=0.60,
                        left=0.06, right=0.97, top=0.94, bottom=0.04)

    for idx, cancer in enumerate(CANCER_ORDER):
        row_i, col_i = divmod(idx, 3)
        ax  = axes[row_i][col_i]
        col = CANCER_COLOR[cancer]

        prots   = deploy[cancer]
        imp_map = get_importance(PKG, cancer, prots)
        # Sort by importance descending → bottom to top in barh
        order   = sorted(prots, key=lambda p: imp_map.get(p, 0))
        imps    = [imp_map.get(p, 0) for p in order]
        cats    = [FUNC.get(p, "Other") for p in order]
        colors  = [FUNC_COL.get(c, "#aaaaaa") for c in cats]

        ypos = np.arange(len(order))

        # Horizontal lollipop
        for yi, (prot, imp, c) in enumerate(zip(order, imps, colors)):
            ax.plot([0, imp], [yi, yi], color=c, lw=2.0, alpha=0.8, zorder=2)
            ax.scatter([imp], [yi], s=70, color=c,
                       edgecolors="white", linewidths=1.0, zorder=3)

        ax.set_yticks(ypos)
        ax.set_yticklabels(order, fontsize=8, fontweight="bold")
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel("Feature importance", fontsize=7.5, fontweight="bold", labelpad=2)
        ax.tick_params(axis="x", labelsize=7, length=2, pad=1)
        ax.tick_params(axis="y", length=0)

        ax.set_title(CANCER_FULL[cancer], color=col,
                     fontsize=11, fontweight="bold", pad=4)

        # AUC badge
        auc = si_df[si_df["target_class"].str.upper() == cancer]["best_panel_test_auc"]
        if not auc.empty:
            ax.text(0.98, 0.02, f"AUC {auc.values[0]:.2f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, fontweight="bold", color=col)

        ax.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        _spine(ax)

    # Global function legend
    leg_handles = [mpatches.Patch(facecolor=FUNC_COL[c], label=c)
                   for c in FUNC_ORDER if FUNC_COL.get(c)]
    fig.legend(handles=leg_handles, title="Protein function",
               fontsize=8, title_fontsize=8.5, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, -0.01),
               framealpha=0.92, edgecolor="#cccccc").get_title().set_fontweight("bold")

    fig.suptitle(
        "Option 2 — Single-panel protein importance by cancer\n"
        "Dots = deployed single-panel proteins · color = function category · "
        "x = feature importance score",
        fontsize=11, fontweight="bold", y=0.97)

    save(fig, "fig6_parta_opt2_importance_lollipop")


# ══════════════════════════════════════════════════════════════════════════════
# OPTION 3 — Panel-direct ORA: Fisher's exact test on single-panel proteins
# Background = cancer-specific DEA pool (not all 1,463 proteins).
# Selects best of GO BP / Reactome per cancer by (n_FDR, best_p) score.
# Consistent DB colors: GO=#7b3f9e  KEGG=#c97b00  Reactome=#1a8b72
# ══════════════════════════════════════════════════════════════════════════════
from scipy.stats import fisher_exact as _fisher_exact

DB_COL_MAP = {"GO BP": "#7b3f9e", "Reactome": "#1a8b72", "None": "#888888"}


def _bh_correct(pvals):
    """Manual Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    ps = np.array(pvals, dtype=float)[order]
    bh = np.minimum.accumulate((ps * n / np.arange(1, n + 1))[::-1])[::-1]
    bh = np.minimum(bh, 1.0)
    out = np.empty(n)
    out[order] = bh
    return out


def load_panel_enrichment(cancer: str, panel_prots: list, n_terms: int = 5):
    """
    ORA directly on single-panel proteins (3-13 per cancer).
    Background = cancer DEA pool (selected_gene_count in the CSV).
    For each GO/Reactome term that contains ≥1 panel protein, compute
    Fisher's exact p-value (one-sided: over-representation).
    Returns (result_df, db_name).
    """
    panel_set = set(panel_prots)
    K = len(panel_prots)
    if K == 0:
        return pd.DataFrame(), "None"

    go_f = LEG / f"{cancer.lower()}_go_bp_ora.csv"
    re_f = LEG / f"{cancer.lower()}_reactome_ora.csv"

    rows = []
    for f, db in [(go_f, "GO BP"), (re_f, "Reactome")]:
        if not f.exists():
            continue
        raw = pd.read_csv(f)
        N = int(raw["selected_gene_count"].iloc[0])   # DEA pool size

        for _, r in raw.iterrows():
            T = int(r["overlap_count"])               # DEA proteins in term
            gene_set = {g.strip() for g in str(r["overlapping_genes"]).split(";") if g.strip()}
            hits = [p for p in panel_prots if p in gene_set]
            a = len(hits)
            if a == 0:
                continue

            b = K - a                   # panel proteins NOT in term
            c = T - a                   # DEA-in-term, not panel
            d = max(N - T - b, 0)       # neither (guard rounding)
            _, pval = _fisher_exact([[a, b], [c, d]], alternative="greater")

            rows.append({
                "db":         db,
                "term_name":  r["term_name"],
                "p_value":    pval,
                "panel_hits": hits,
                "a": a, "K": K, "T": T, "N": N,
            })

    if not rows:
        return pd.DataFrame(), "None"

    res = pd.DataFrame(rows)
    res["p_adjusted_bh"] = _bh_correct(res["p_value"].values)

    # Pick best DB: most FDR terms first, then best nominal p as tiebreaker
    def _score(sub):
        if sub.empty:
            return (-1, 0.0)
        return (int((sub["p_adjusted_bh"] < 0.05).sum()),
                float(-np.log10(sub["p_value"].min() + 1e-15)))

    go_s = _score(res[res["db"] == "GO BP"])
    re_s = _score(res[res["db"] == "Reactome"])

    if not any(res["db"] == "GO BP"):
        best_db = "Reactome"
    elif not any(res["db"] == "Reactome"):
        best_db = "GO BP"
    else:
        best_db = "GO BP" if go_s >= re_s else "Reactome"

    final = (res[res["db"] == best_db]
             .nsmallest(n_terms, "p_value")
             .reset_index(drop=True))
    return final, best_db


def build_opt3():
    fig, axes = plt.subplots(4, 3, figsize=(22, 24))
    fig.subplots_adjust(wspace=0.48, hspace=0.58,
                        left=0.07, right=0.97, top=0.94, bottom=0.05)

    fdr_cut = -np.log10(0.05)

    for idx, cancer in enumerate(CANCER_ORDER):
        row_i, col_i = divmod(idx, 3)
        ax     = axes[row_i][col_i]
        col    = CANCER_COLOR[cancer]
        single = deploy[cancer]

        df, db_name = load_panel_enrichment(cancer, single, n_terms=5)
        db_text_col = DB_COL_MAP.get(db_name, "#888888")

        # ── Empty panel: no pathway hits for these proteins ───────────────────
        if df.empty:
            ax.set_facecolor("#f8f8f8")
            ax.text(0.5, 0.58, "No pathway annotations\nfound for panel proteins",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#888888", fontsize=10, fontstyle="italic")
            if single:
                ax.text(0.5, 0.35,
                        "\n".join(textwrap.wrap(", ".join(single), 28)),
                        ha="center", va="center", transform=ax.transAxes,
                        color=col, fontsize=9, fontweight="bold")
            ax.set_title(CANCER_FULL[cancer], color=col,
                         fontsize=14, fontweight="bold")
            _spine(ax)
            continue

        df = df.copy()
        df["logp"]   = -np.log10(df["p_value"].clip(lower=1e-10))
        df["logfdr"] = -np.log10(df["p_adjusted_bh"].clip(lower=1e-10))
        df = df.sort_values("logp", ascending=True).reset_index(drop=True)

        lp_min, lp_max = df["logp"].min(), df["logp"].max()
        lp_range = max(lp_max - lp_min, 1e-6)

        n_fdr_shown = 0
        for i, row in df.iterrows():
            logp   = row["logp"]
            is_nom = logp >= fdr_cut          # p < 0.05 nominal
            is_fdr = row["logfdr"] >= fdr_cut # BH-corrected p < 0.05

            t         = (logp - lp_min) / lp_range
            alpha_val = 0.28 + 0.62 * t
            edge_col  = col     if is_fdr else ("#888888" if is_nom else "#d8d8d8")
            edge_lw   = 2.2     if is_fdr else (1.0 if is_nom else 0.4)

            ax.barh(i, logp, color=col, alpha=alpha_val,
                    height=0.65, edgecolor=edge_col, linewidth=edge_lw, zorder=2)

            if is_fdr:
                n_fdr_shown += 1

            hits = row["panel_hits"]
            K    = row["K"]
            if hits:
                label = f"{', '.join(hits)}  ({len(hits)}/{K})"
                ax.text(logp + 0.06, i, label,
                        va="center", ha="left", fontsize=7.5,
                        fontweight="bold", color="#cc0000", clip_on=False)

        ax.axvline(fdr_cut, color="#cc0000", lw=1.1, ls="--", alpha=0.65, zorder=1)

        ax.set_yticks(np.arange(len(df)))
        ax.set_yticklabels(
            ["\n".join(textwrap.wrap(
                t.split("(GO:")[0].strip() if "(GO:" in t else
                (t.split(" R-HSA-")[0].strip() if " R-HSA-" in t else t[:40]), 28))
             for t in df["term_name"]],
            fontsize=8, fontweight="bold")
        ax.set_ylim(-0.6, len(df) - 0.4)
        xmax = max(df["logp"].max(), fdr_cut * 1.1)
        ax.set_xlim(0, xmax * 1.60)
        ax.set_xlabel("-log10(p · panel ORA)", fontsize=9, fontweight="bold", labelpad=2)
        ax.tick_params(axis="x", labelsize=8.5, length=2, pad=1)
        ax.tick_params(axis="y", length=0)

        # Significance badge (bottom-right)
        if n_fdr_shown:
            sig_txt = f"{n_fdr_shown} FDR<0.05"
            sig_col = col
        elif any(df["logp"] >= fdr_cut):
            sig_txt = "Nominal p<0.05"
            sig_col = "#666666"
        else:
            sig_txt = "All p>0.05"
            sig_col = "#aaaaaa"
        ax.text(0.98, 0.02, sig_txt, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5,
                color=sig_col, fontweight="bold", fontstyle="italic")

        # Panel proteins badge (top-left)
        sp_str = " · ".join(single[:4])
        if len(single) > 4:
            sp_str += f" +{len(single)-4}"
        ax.text(0.01, 0.99, sp_str, transform=ax.transAxes,
                ha="left", va="top", fontsize=7,
                color="#cc0000", fontweight="bold", fontstyle="italic")

        ax.set_title(CANCER_FULL[cancer], color=col,
                     fontsize=14, fontweight="bold", pad=10)
        ax.text(0.5, 1.01, f"[{db_name}]", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9,
                color=db_text_col, fontstyle="italic", fontweight="bold")

        ax.set_facecolor("#fafafa")
        ax.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        _spine(ax)

    leg_handles = [
        mpatches.Patch(facecolor="#7a3e9d", alpha=0.90, edgecolor="#7a3e9d",
                       linewidth=2.2, label="FDR < 0.05 · BH-corrected (bold edge)"),
        mpatches.Patch(facecolor="#7a3e9d", alpha=0.55, edgecolor="#888888",
                       linewidth=1.0, label="Nominal p < 0.05 (medium edge)"),
        mpatches.Patch(facecolor="#7a3e9d", alpha=0.30, edgecolor="#d8d8d8",
                       linewidth=0.4, label="p > 0.05 (light, gradient by p-value)"),
        mpatches.Patch(facecolor="white", edgecolor="#cc0000",
                       label="Red label = panel protein(s) driving the hit  (n/K)"),
    ]
    fig.legend(handles=leg_handles, fontsize=10, framealpha=0.92,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.00))

    fig.suptitle(
        "Single-panel pathway enrichment — ORA directly on deployed proteins (3–13 per cancer)\n"
        "Background = cancer DEA pool · Fisher's exact test · BH FDR correction · "
        "bars = terms containing ≥1 panel protein · red = panel protein(s) · (n/K) = fraction of panel",
        fontsize=11, fontweight="bold", y=0.97)

    save(fig, "fig6_parta_opt3_legacy_dea")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building Part A single-panel options ...")
    print("\n--- Option 1: annotation heatmap ---")
    build_opt1()
    print("\n--- Option 2: importance lollipop ---")
    build_opt2()
    print("\n--- Option 3: legacy DEA gradient ---")
    build_opt3()
    print("\nDone.")
