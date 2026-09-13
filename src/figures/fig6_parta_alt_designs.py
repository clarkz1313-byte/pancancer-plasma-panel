#!/usr/bin/env python3
"""
Fig 6 Part A — four alternative designs (no ORA statistics needed)
  D  — Protein-first lollipop: one dot per panel protein, best annotation label
  A  — 12-cancer × function-category dot matrix
  Ch — Chord / bipartite arc: cancer ↔ function-category arcs (matplotlib)
  Sk — Sankey: cancer → function-category → protein (plotly HTML)

Run:  python3 figurev5/scripts/fig6_parta_alt_designs.py
"""
from __future__ import annotations
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact as _fisher_exact

ROOT = Path("e:/Proteomics")
PKG  = ROOT / "revise_plan" / "locked25_final_results_package"
LEG  = ROOT / "revise_plan" / "legacy_pathway_enrichment" / "result12classes" / "tables" / "enrichment"
OUT  = ROOT / "figurev5" / "output"
OUT.mkdir(parents=True, exist_ok=True)

# ── shared constants ──────────────────────────────────────────────────────────
CANCERS = ["AML","BRC","CLL","CRC","CVX","ENDC","GLIOM","LUNGC","LYMPH","MYEL","OVC","PRC"]
CANCER_FULL = {
    "AML":"AML","BRC":"Breast","CLL":"CLL",
    "CRC":"Colorectal","CVX":"Cervical",
    "ENDC":"Endometrial","GLIOM":"Glioma",
    "LUNGC":"Lung","LYMPH":"DLBCL",
    "MYEL":"Myeloma","OVC":"Ovarian","PRC":"Prostate",
}
CANCER_COLOR = {
    "AML":"#b22222","BRC":"#c97b63","CLL":"#7a3e9d",
    "CRC":"#d68600","CVX":"#c13d86","ENDC":"#8e5d2c",
    "GLIOM":"#1a8db8","LUNGC":"#2e8b57","LYMPH":"#3856a6",
    "MYEL":"#8c564b","OVC":"#d1495b","PRC":"#008b8b",
}

FUNC = {
    "FLT3":"Immune/Hematop.","CD244":"Immune/Hematop.","FCER2":"Immune/Hematop.",
    "TCL1A":"Immune/Hematop.","SLAMF7":"Immune/Hematop.","FCRLB":"Immune/Hematop.",
    "MZB1":"Immune/Hematop.","TNFRSF13C":"Immune/Hematop.","TNFRSF10A":"Immune/Hematop.",
    "C1QA":"Immune/Hematop.","PDCD1":"Immune/Hematop.","CORO1A":"Immune/Hematop.",
    "PGLYRP1":"Immune/Hematop.",
    "CXCL8":"Cytokine","CXCL13":"Cytokine","CXCL17":"Cytokine",
    "BMP4":"Growth factor","BMP6":"Growth factor","ARTN":"Growth factor",
    "NTF3":"Growth factor","IGFBP1":"Growth factor","AREG":"Growth factor",
    "ENPP2":"Growth factor","LEFTY2":"Growth factor","PROK1":"Growth factor",
    "NTRK3":"Growth factor","ADGRG1":"Growth factor","PSPN":"Growth factor",
    "SDC4":"ECM/Adhesion","SDC1":"ECM/Adhesion","SPOCK1":"ECM/Adhesion",
    "LGALS4":"ECM/Adhesion","CDH15":"ECM/Adhesion","CNTN5":"ECM/Adhesion",
    "CNTN3":"ECM/Adhesion","TSPAN1":"ECM/Adhesion","BCAN":"ECM/Adhesion",
    "FAP":"ECM/Adhesion","CRTAC1":"ECM/Adhesion","CDHR2":"ECM/Adhesion","S100A4":"ECM/Adhesion",
    "LTA4H":"Enzyme/Protease","ADAMTS13":"Enzyme/Protease","KLK14":"Enzyme/Protease",
    "KLK13":"Enzyme/Protease","THOP1":"Enzyme/Protease","HAGH":"Enzyme/Protease",
    "PRDX6":"Enzyme/Protease","PRDX5":"Enzyme/Protease","GLO1":"Enzyme/Protease",
    "CA3":"Enzyme/Protease","PSMD9":"Enzyme/Protease","ALPP":"Enzyme/Protease",
    "QPCT":"Enzyme/Protease","AFP":"Enzyme/Protease",
    "WFDC2":"Cancer marker","PAEP":"Cancer marker","CGA":"Cancer marker",
    "MSMB":"Cancer marker","TACSTD2":"Cancer marker","SEZ6L2":"Cancer marker",
    "SPINK4":"Cancer marker","CEACAM5":"Cancer marker",
    "PSIP1":"Signaling","ILKAP":"Signaling","NFATC1":"Signaling",
    "TRAF2":"Signaling","BID":"Signaling","TFRC":"Signaling",
    "GFAP":"Neural/Struct.","NEFL":"Neural/Struct.",
    "GCG":"Metabolic/Endocr.","LEP":"Metabolic/Endocr.","PLIN1":"Metabolic/Endocr.",
    "CHGB":"Metabolic/Endocr.","VAT1":"Metabolic/Endocr.",
    "XG":"Metabolic/Endocr.","FKBP1B":"Metabolic/Endocr.","CRNN":"Metabolic/Endocr.",
    "F3":"Metabolic/Endocr.",
}
FUNC_ORDER = [
    "Immune/Hematop.","Cytokine","Growth factor","ECM/Adhesion",
    "Enzyme/Protease","Cancer marker","Signaling","Metabolic/Endocr.","Neural/Struct.",
]
FUNC_COL = {
    "Immune/Hematop.":  "#7a3e9d",
    "Cytokine":         "#c13d86",
    "Growth factor":    "#2e8b57",
    "ECM/Adhesion":     "#008b8b",
    "Enzyme/Protease":  "#d1495b",
    "Cancer marker":    "#b22222",
    "Signaling":        "#d68600",
    "Metabolic/Endocr.":"#3856a6",
    "Neural/Struct.":   "#8c564b",
    "Other":            "#aaaaaa",
}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "font.weight": "bold",
    "axes.titlesize": 13, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": 0.9,
})
DPI = 300

# ── load deploy lists + AUC ───────────────────────────────────────────────────
si_files = list(PKG.rglob("single_internal.csv"))
si_df    = pd.read_csv(si_files[0]) if si_files else pd.DataFrame()

def get_deploy(cancer):
    row = si_df[si_df["target_class"].str.upper() == cancer.upper()]
    return [p.strip() for p in str(row.iloc[0]["deploy_proteins"]).split(";") if p.strip()] if not row.empty else []

def get_auc(cancer):
    row = si_df[si_df["target_class"].str.upper() == cancer.upper()]
    return float(row.iloc[0]["best_panel_test_auc"]) if not row.empty else None

deploy = {c: get_deploy(c) for c in CANCERS}
aucs   = {c: get_auc(c)    for c in CANCERS}


# ── BH correction + Fisher ORA (for best-term annotation in Option D) ─────────
def _bh(pvals):
    n = len(pvals)
    if n == 0: return np.array([])
    order = np.argsort(pvals)
    ps = np.array(pvals, dtype=float)[order]
    bk = np.minimum.accumulate((ps * n / np.arange(1, n + 1))[::-1])[::-1]
    bk = np.minimum(bk, 1.0)
    out = np.empty(n); out[order] = bk
    return out

def best_term_per_protein(cancer, panel_prots):
    """For each panel protein, find its best (lowest p) pathway term."""
    K = len(panel_prots)
    if K == 0: return {}
    rows = []
    for f, db in [(LEG / f"{cancer.lower()}_go_bp_ora.csv", "GO BP"),
                  (LEG / f"{cancer.lower()}_reactome_ora.csv", "Reactome")]:
        if not f.exists(): continue
        raw = pd.read_csv(f)
        N = int(raw["selected_gene_count"].iloc[0])
        for _, r in raw.iterrows():
            T = int(r["overlap_count"])
            gene_set = {g.strip() for g in str(r["overlapping_genes"]).split(";") if g.strip()}
            hits = [p for p in panel_prots if p in gene_set]
            if not hits: continue
            a = len(hits); b = K-a; c = T-a; d = max(N-T-b, 0)
            _, pval = _fisher_exact([[a,b],[c,d]], alternative="greater")
            for h in hits:
                rows.append({"protein": h, "term": r["term_name"], "p": pval, "db": db})
    if not rows: return {}
    res = pd.DataFrame(rows)
    best = {}
    for prot, grp in res.groupby("protein"):
        best_row = grp.nsmallest(1, "p").iloc[0]
        raw_term = str(best_row["term"])
        # strip GO/Reactome IDs from term name
        if "(GO:" in raw_term:
            clean = raw_term.split("(GO:")[0].strip()
        elif " R-HSA-" in raw_term:
            clean = raw_term.split(" R-HSA-")[0].strip()
        else:
            clean = raw_term[:45]
        best[prot] = {"term": clean, "p": best_row["p"], "db": best_row["db"]}
    return best


def _spine(ax, color="#888888", lw=0.8):
    for sp in ax.spines.values():
        sp.set_color(color); sp.set_linewidth(lw)


def save(fig, stem):
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# OPTION D — Protein-first lollipop
# One dot per panel protein, colored by function, best-term annotation on right
# Grouped by cancer, with AUC badge; avoids identical-bar artifact entirely
# ══════════════════════════════════════════════════════════════════════════════
def build_optD():
    print("Building Option D -- protein-first lollipop ...")

    # Pre-compute best terms (can be slow — cache in dict)
    best_terms = {}
    for c in CANCERS:
        best_terms[c] = best_term_per_protein(c, deploy[c])

    fig, axes = plt.subplots(4, 3, figsize=(23, 26))
    fig.subplots_adjust(wspace=0.52, hspace=0.60,
                        left=0.06, right=0.97, top=0.93, bottom=0.05)

    for idx, cancer in enumerate(CANCERS):
        row_i, col_i = divmod(idx, 3)
        ax  = axes[row_i][col_i]
        col = CANCER_COLOR[cancer]
        prots = deploy[cancer]
        bt    = best_terms[cancer]

        # Sort: proteins with a term hit first (by p), then unannotated
        def sort_key(p):
            if p in bt:
                return (0, bt[p]["p"])
            return (1, 1.0)
        order = sorted(prots, key=sort_key)

        fdr_cut = -np.log10(0.05)
        ypos = np.arange(len(order))

        x_vals  = []
        colors  = []
        markers = []

        for p in order:
            fc = FUNC_COL.get(FUNC.get(p, "Other"), "#aaaaaa")
            if p in bt:
                lp = -np.log10(bt[p]["p"] + 1e-15)
            else:
                lp = 0.0
            x_vals.append(lp)
            colors.append(fc)
            markers.append("o")

        x_arr = np.array(x_vals)
        x_max = max(x_arr.max() if len(x_arr) else 1.0, fdr_cut * 1.2)

        for yi, (prot, xv, fc) in enumerate(zip(order, x_arr, colors)):
            # Stem
            ax.plot([0, xv], [yi, yi], color=fc, lw=2.0, alpha=0.75, zorder=2)
            # Dot — filled circle
            edge = "#cc0000" if (xv >= fdr_cut) else "white"
            ax.scatter([xv], [yi], s=90, color=fc,
                       edgecolors=edge, linewidths=1.5, zorder=3)
            # Best-term annotation
            if prot in bt:
                term_str = textwrap.shorten(bt[prot]["term"], width=36, placeholder="…")
                sig_col  = "#cc0000" if xv >= fdr_cut else "#555555"
                ax.text(x_max * 1.04, yi, term_str,
                        va="center", ha="left", fontsize=6.5,
                        color=sig_col, fontstyle="italic", clip_on=False)
            else:
                ax.text(x_max * 1.04, yi, "no annotation",
                        va="center", ha="left", fontsize=6,
                        color="#bbbbbb", fontstyle="italic", clip_on=False)

        # FDR line
        ax.axvline(fdr_cut, color="#cc0000", lw=1.0, ls="--", alpha=0.55, zorder=1)

        ax.set_yticks(ypos)
        ax.set_yticklabels(order, fontsize=8.5, fontweight="bold")
        ax.set_ylim(-0.65, len(order) - 0.35)
        ax.set_xlim(0, x_max * 1.08)
        ax.set_xlabel("-log10(best-term p, panel ORA)", fontsize=8, fontweight="bold", labelpad=2)
        ax.tick_params(axis="x", labelsize=8, length=2, pad=1)
        ax.tick_params(axis="y", length=0)

        ax.set_title(CANCER_FULL[cancer], color=col, fontsize=14, fontweight="bold", pad=8)

        # AUC badge
        auc = aucs.get(cancer)
        if auc:
            ax.text(0.99, 0.01, f"AUC {auc:.3f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, fontweight="bold", color=col)

        ax.set_facecolor("#fafafa")
        ax.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        _spine(ax)

    # Global function-category legend
    leg_handles = [mpatches.Patch(facecolor=FUNC_COL[c], label=c)
                   for c in FUNC_ORDER if c in FUNC_COL]
    fig.legend(handles=leg_handles, title="Protein function", fontsize=9,
               title_fontsize=9.5, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, 0.00), framealpha=0.93,
               edgecolor="#cccccc").get_title().set_fontweight("bold")

    fig.suptitle(
        "Single-panel proteins -- best pathway annotation per protein  "
        "(panel-direct ORA, Fisher's exact, x = -log10 p of best term)\n"
        "Dot color = protein function · red outline = p < 0.05 · "
        "italic label = best-matching pathway term",
        fontsize=11, fontweight="bold", y=0.97)

    save(fig, "fig6_parta_optD_protein_lollipop")
    print("  Option D done.")


# ══════════════════════════════════════════════════════════════════════════════
# OPTION A — 12-cancer × function-category dot matrix
# Rows = cancers, cols = function categories
# Dot size ∝ protein count; color = cancer; annotation = protein names
# ══════════════════════════════════════════════════════════════════════════════
def build_optA():
    print("Building Option A -- dot matrix ...")

    # Build count matrix
    cats_present = [c for c in FUNC_ORDER if any(
        FUNC.get(p, "Other") == c for ps in deploy.values() for p in ps)]

    # matrix[cancer][cat] = list of proteins
    mat = {c: {cat: [] for cat in cats_present} for c in CANCERS}
    for c in CANCERS:
        for p in deploy[c]:
            cat = FUNC.get(p, "Other")
            if cat in mat[c]:
                mat[c][cat].append(p)

    nc = len(cats_present)
    nr = len(CANCERS)

    fig, ax = plt.subplots(figsize=(nc * 1.55 + 1.0, nr * 1.05 + 2.2))
    fig.subplots_adjust(left=0.18, right=0.98, top=0.87, bottom=0.26)

    # Max count for dot scaling
    all_counts = [len(mat[c][cat]) for c in CANCERS for cat in cats_present]
    max_count  = max(all_counts) if all_counts else 1
    S_MAX = 700   # max dot area in pt²

    for ci, cat in enumerate(cats_present):
        fc_col = FUNC_COL.get(cat, "#aaa")
        for ri, cancer in enumerate(CANCERS):
            prots = mat[cancer][cat]
            n = len(prots)
            # Background cell
            bg = "#fafafa" if n == 0 else "#f0f4ff"
            ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                         facecolor=bg, edgecolor="#e8e8e8",
                         linewidth=0.8, zorder=1))
            if n == 0:
                continue
            # Dot sized by count
            s = S_MAX * (n / max_count)
            ax.scatter([ci], [ri], s=s,
                       color=CANCER_COLOR[cancer],
                       edgecolors=fc_col, linewidths=1.8,
                       alpha=0.88, zorder=3)
            # Count label inside dot
            ax.text(ci, ri, str(n), ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white", zorder=4)
            # Protein names below dot (small, italic)
            names = "\n".join(textwrap.wrap(", ".join(prots), 14))
            ax.text(ci, ri - 0.38, names,
                    ha="center", va="top", fontsize=5.8,
                    color="#444444", fontstyle="italic",
                    zorder=4, clip_on=True)

    # Category header strip above grid
    for ci, cat in enumerate(cats_present):
        fc_col = FUNC_COL.get(cat, "#aaa")
        ax.add_patch(plt.Rectangle((ci - 0.5, nr - 0.5), 1, 0.55,
                     facecolor=fc_col, edgecolor="white",
                     linewidth=1.2, zorder=5, clip_on=False))

    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(-0.5, nr - 0.5)
    ax.set_xticks(range(nc))
    ax.set_xticklabels(cats_present, rotation=38, ha="right",
                       rotation_mode="anchor", fontsize=9, fontweight="bold")
    ax.set_yticks(range(nr))
    ax.set_yticklabels([CANCER_FULL[c] for c in CANCERS],
                       fontsize=10, fontweight="bold")
    for label, cancer in zip(ax.get_yticklabels(), CANCERS):
        label.set_color(CANCER_COLOR[cancer])
    ax.tick_params(length=0, pad=5)
    ax.invert_yaxis()

    # Dot-size legend
    for n_ex in [1, 2, 4]:
        s_ex = S_MAX * (n_ex / max_count)
        ax.scatter([], [], s=s_ex, color="#888888",
                   edgecolors="#555555", linewidths=1.2,
                   label=f"n = {n_ex}")
    ax.legend(title="Protein count", fontsize=9, title_fontsize=9,
              loc="lower right", bbox_to_anchor=(1.0, -0.30),
              bbox_transform=ax.transAxes,
              framealpha=0.93, edgecolor="#cccccc",
              ncol=3).get_title().set_fontweight("bold")

    _spine(ax)
    ax.set_title(
        "Option A — Single-panel proteins by function category\n"
        "Rows = 12 cancers · Columns = protein function · "
        "Dot color = cancer · size = protein count · white number = count",
        fontsize=10.5, fontweight="bold", pad=28)

    save(fig, "fig6_parta_optA_dot_matrix")
    print("  Option A done.")


# ══════════════════════════════════════════════════════════════════════════════
# OPTION Ch — Bipartite arc / chord-style (matplotlib, no external lib)
# Left arc: cancers  |  Right arc: function categories
# Bezier ribbons connect cancer → its function categories
# Ribbon width proportional to protein count; color = cancer
# ══════════════════════════════════════════════════════════════════════════════
def build_optCh():
    print("Building Option Ch -- chord/arc diagram ...")

    # Build edge table: cancer × category → count
    edges = []
    for c in CANCERS:
        from collections import Counter
        cat_cnt = Counter(FUNC.get(p, "Other") for p in deploy[c])
        for cat, n in cat_cnt.items():
            if cat in FUNC_COL:
                edges.append({"cancer": c, "cat": cat, "n": n})
    edf = pd.DataFrame(edges)

    cats_present = [c for c in FUNC_ORDER if c in edf["cat"].values]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.5, max(len(CANCERS), len(cats_present)) + 0.5)

    n_c = len(CANCERS)
    n_f = len(cats_present)

    # Vertical positions
    c_y = {c: (n_c - 1 - i) * (max(n_c, n_f) - 1) / (n_c - 1)
           for i, c in enumerate(CANCERS)}
    f_y = {f: (n_f - 1 - i) * (max(n_c, n_f) - 1) / (n_f - 1)
           for i, f in enumerate(cats_present)}

    x_left, x_right = 0.18, 1.82
    x_ctrl = 1.0   # control point x (midline)

    # Draw Bezier ribbons
    max_n = edf["n"].max()
    for _, row in edf.iterrows():
        c, cat, n = row["cancer"], row["cat"], row["n"]
        y0 = c_y[c]
        y1 = f_y[cat]
        lw   = 0.8 + 5.5 * (n / max_n)
        alpha = 0.30 + 0.40 * (n / max_n)
        col   = CANCER_COLOR[c]
        verts = [(x_left,  y0),
                 (x_ctrl,  y0),
                 (x_ctrl,  y1),
                 (x_right, y1)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
        patch = mpatches.PathPatch(MPath(verts, codes),
                                   facecolor="none", edgecolor=col,
                                   lw=lw, alpha=alpha, zorder=2)
        ax.add_patch(patch)

    # Left nodes (cancers)
    for c in CANCERS:
        y = c_y[c]
        col = CANCER_COLOR[c]
        ax.plot(x_left, y, "o", ms=14, color=col,
                mec="white", mew=1.5, zorder=4)
        k = len(deploy[c])
        auc = aucs.get(c)
        label = f"{CANCER_FULL[c]}  (K={k}, AUC={auc:.2f})" if auc else f"{CANCER_FULL[c]}  (K={k})"
        ax.text(x_left - 0.06, y, label,
                ha="right", va="center", fontsize=10,
                fontweight="bold", color=col)

    # Right nodes (function categories)
    for cat in cats_present:
        y = f_y[cat]
        fc = FUNC_COL.get(cat, "#aaa")
        ax.plot(x_right, y, "s", ms=13, color=fc,
                mec="white", mew=1.5, zorder=4)
        # protein count across all cancers for this category
        total = edf[edf["cat"] == cat]["n"].sum()
        ax.text(x_right + 0.06, y, f"{cat}  (n={total})",
                ha="left", va="center", fontsize=10,
                fontweight="bold", color=fc)

    # Section headers
    ax.text(x_left,  max(c_y.values()) + 0.65, "CANCERS",
            ha="center", va="bottom", fontsize=14,
            fontweight="bold", color="#333333")
    ax.text(x_right, max(f_y.values()) + 0.65, "PROTEIN FUNCTION",
            ha="center", va="bottom", fontsize=14,
            fontweight="bold", color="#333333")

    # Ribbon-width legend
    for n_ex, lbl in [(1, "1 protein"), (3, "3 proteins")]:
        lw_ex = 0.8 + 5.5 * (n_ex / max_n)
        ax.plot([], [], color="#888888", lw=lw_ex, label=lbl)
    leg = ax.legend(title="Ribbon width", fontsize=9.5, title_fontsize=10,
                    loc="lower center", bbox_to_anchor=(0.5, -0.03),
                    ncol=2, framealpha=0.93, edgecolor="#cccccc")
    leg.get_title().set_fontweight("bold")

    ax.set_title(
        "Single-panel proteins — cancer to function-category connections\n"
        "Ribbon width proportional to protein count · color = cancer · square = function category",
        fontsize=12, fontweight="bold", pad=12)

    save(fig, "fig6_parta_optCh_chord_arc")
    print("  Option Ch done.")


# ══════════════════════════════════════════════════════════════════════════════
# OPTION Sk — Sankey: cancer → function-category → protein  (plotly HTML)
# Three layers: Cancer | Function category | Protein
# Saved as interactive HTML (open in browser) + static PNG via kaleido
# ══════════════════════════════════════════════════════════════════════════════
def build_optSk():
    print("Building Option Sk -- Sankey diagram ...")
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly not available -- skipping Sankey")
        return

    # Nodes: cancers first, then categories, then proteins
    all_prots_ordered = []
    seen = set()
    for c in CANCERS:
        for p in deploy[c]:
            if p not in seen:
                all_prots_ordered.append(p)
                seen.add(p)

    cats_present = [c for c in FUNC_ORDER
                    if any(FUNC.get(p, "Other") == c
                           for ps in deploy.values() for p in ps)]

    nodes   = CANCERS + cats_present + all_prots_ordered
    node_idx = {n: i for i, n in enumerate(nodes)}

    node_colors = []
    for n in nodes:
        if n in CANCERS:
            node_colors.append(CANCER_COLOR[n])
        elif n in FUNC_COL:
            # rgba with alpha
            hx = FUNC_COL[n].lstrip("#")
            r,g,b = int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)
            node_colors.append(f"rgba({r},{g},{b},0.85)")
        else:
            node_colors.append("rgba(180,180,180,0.7)")

    sources, targets, values, link_colors = [], [], [], []

    for c in CANCERS:
        from collections import Counter
        cat_cnt = Counter(FUNC.get(p, "Other") for p in deploy[c])
        for cat, n in cat_cnt.items():
            if cat not in cats_present: continue
            sources.append(node_idx[c])
            targets.append(node_idx[cat])
            values.append(n)
            hx = CANCER_COLOR[c].lstrip("#")
            r,g,b = int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)
            link_colors.append(f"rgba({r},{g},{b},0.35)")

    for c in CANCERS:
        for p in deploy[c]:
            cat = FUNC.get(p, "Other")
            if cat not in cats_present: continue
            sources.append(node_idx[cat])
            targets.append(node_idx[p])
            values.append(1)
            hx = FUNC_COL.get(cat, "#aaaaaa").lstrip("#")
            r,g,b = int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)
            link_colors.append(f"rgba({r},{g},{b},0.25)")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18, thickness=18,
            line=dict(color="white", width=0.8),
            label=nodes,
            color=node_colors,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate="%{source.label} → %{target.label}: %{value}<extra></extra>",
        ),
    ))
    fig.update_layout(
        title_text=(
            "Single-panel proteins: Cancer → Function category → Protein  "
            "(Sankey · flow width = protein count)"
        ),
        title_font_size=15,
        font_family="Arial",
        font_size=12,
        width=1400, height=900,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
    )

    html_path = OUT / "fig6_parta_optSk_sankey.html"
    fig.write_html(str(html_path))
    print(f"  saved -> {html_path}")

    # Try static PNG via kaleido
    try:
        png_path = OUT / "fig6_parta_optSk_sankey.png"
        fig.write_image(str(png_path), scale=2)
        print(f"  saved -> {png_path}")
        pdf_path = OUT / "fig6_parta_optSk_sankey.pdf"
        fig.write_image(str(pdf_path))
        print(f"  saved -> {pdf_path}")
    except Exception as e:
        print(f"  kaleido static export failed ({e}) -- HTML only")

    print("  Option Sk done.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building four alternative designs for fig6 Part A ...\n")
    build_optD()
    print()
    build_optA()
    print()
    build_optCh()
    print()
    build_optSk()
    print("\nAll done.")
