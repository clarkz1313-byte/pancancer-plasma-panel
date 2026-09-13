#!/usr/bin/env python3
"""
Fig 4 panels — figurev5 v10
  4a: Part B grouped heatmap
      - colorbar in dedicated thin row ABOVE the bar plot, beneath suptitle
      - 4-row GridSpec: [colorbar | bar | stripe | heatmap+dendrogram]
      - no annotation text
  4b: Part A single 1×12 figure
      - global Z-score scale vmin=-4, vmax=+4, NO row normalization
      - large vmax prevents solid stripes (Z≈3 peaks at ~87% color, not 100%)
      - per-cancer diverging colormap: gray(depleted) → white(avg) → 75%-sat cancer color(elevated)
      - 1 row × 12 columns layout
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from figure_labels import display_label
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
try:
    from adjustText import adjust_text
except ImportError:
    # Optional label-repulsion dependency.  Keep the figure reproducible in
    # the repository's plotting environment; labels retain their deterministic
    # initial offsets when adjustText is unavailable.
    def adjust_text(texts, *args, **kwargs):
        return texts
from publication_style import enable_text_scaling

enable_text_scaling(factor=1.40, minimum=9.5)

ROOT   = Path(__file__).resolve().parents[2]
REVISE = ROOT / "revise_plan"
OUT    = ROOT / "figurev6" / "output"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_DATA     = ROOT / "data/processed/filtered_pancancer_data.csv"
PART_B_RANKER = REVISE / "part_b_multiclass/vRSX_v11_locked_reproducer/tables/train_only_ranker_table.csv"
PART_B_DEA    = REVISE / "part_b_multiclass/vRSX_v11_locked_reproducer/tables/train_only_ovr_dea_all.csv"
PART_A_SINGLE = REVISE / "part_a_single"
SUPP_S1       = ROOT / "supplementary_tables/Supplementary_Table_S1_single_panel_protein_summary.csv"

CANCER_ORDER = ["AML","BRC","CLL","CRC","CVX","ENDC","GLIOM","LUNGC","LYMPH","MYEL","OVC","PRC"]
CANCER_COLOR = {
    "AML":"#b22222","BRC":"#c97b63","CLL":"#7a3e9d",
    "CRC":"#d68600","CVX":"#c13d86","ENDC":"#8e5d2c",
    "GLIOM":"#7ec8e3","LUNGC":"#2e8b57","LYMPH":"#3856a6",
    "MYEL":"#8c564b","OVC":"#d1495b","PRC":"#008b8b",
}
PLOT_COLOR = {**CANCER_COLOR, "GLIOM": "#1a8db8"}

LOCKED_25 = [
    "FLT3","CNTN1","FCER2","PRDX6","LTA4H","XG","CCDC80","CXCL17","CXCL13","SLAMF7",
    "PAEP","PSPN","BMP4","WFDC2","TRAF2","KLK13","GLO1","GFAP","CEACAM5","CGA",
    "ADAMTS13","CRTAC1","TCL1A","ADAMTS15","NEFL",
]

# 5-stop ramp (not a flat 3-point linear gray-white-purple): 74% of all Z-score
# cells fall within |Z|<1.0, which a plain 3-stop linspace compresses into the
# near-white 40% dead zone around the midpoint (invisible at mosaic scale).
# Extra stops at +-0.28/+-0.72 (of vmax) force visible color well before the
# extremes, so the bulk of ordinary cells are still legible, not just outliers.
PARTB_CMAP = LinearSegmentedColormap.from_list(
    "gray_white_purple",
    [(0.00, "#4d4d4d"), (0.28, "#b0b0b0"), (0.50, "#efeef2"),
     (0.72, "#c3a3dc"), (1.00, "#6a3d9a")],
    N=256,
)

def _soft(col, frac=0.75):
    rgba = np.array(mcolors.to_rgba(col))
    return tuple((rgba[:3] * frac + (1 - frac)).clip(0, 1))

def _cancer_cmap(col):
    """Same 5-stop steep-ramp design as PARTB_CMAP: dark gray (depleted, shared
    across all cancers so "depleted" always reads the same) -> off-white (avg)
    -> cancer-tinted (elevated). Saturation raised vs. the old 75%-blend so each
    panel's hue still reads as that cancer's identity even at small size."""
    mid_end = _soft(col, 0.50)
    hi_end  = _soft(col, 0.92)
    return LinearSegmentedColormap.from_list(
        f"gray_white_{col}",
        [(0.00, "#4d4d4d"), (0.28, "#b0b0b0"), (0.50, "#efeef2"),
         (0.72, mid_end), (1.00, hi_end)],
        N=256,
    )


def _cell_gridlines(ax, n_rows, n_cols):
    """Subtle semi-transparent hairlines at every cell boundary so cells stay
    visually distinct even when the fill color is near-white -- a fixed-hue
    grid line (pure white or pure gray) disappears against similarly-shaded
    fills, but a low-alpha black line stays faintly visible against any color."""
    for x in range(n_cols + 1):
        ax.axvline(x - 0.5, color="black", lw=0.4, alpha=0.10, zorder=3)
    for y in range(n_rows + 1):
        ax.axhline(y - 0.5, color="black", lw=0.4, alpha=0.10, zorder=3)

plt.rcParams.update({
    "font.family":"Arial","font.size":9,"font.weight":"bold",
    "axes.titlesize":10,"axes.labelsize":8,
    "xtick.labelsize":7,"ytick.labelsize":8,
    "axes.linewidth":0.7,"legend.fontsize":7,"legend.frameon":False,
})
DPI = 400


def save(fig, stem):
    for ext in ("pdf","png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


def _load_deploy(s1, cancer):
    row = s1[s1["target_class"] == cancer].iloc[0]
    pts = [p.strip() for p in str(row["deploy_proteins"]).split(";") if p.strip()]
    return pts[:int(row["deploy_panel_size"])]


def _row_dendrogram(ax, link, NC):
    dend  = dendrogram(link, no_plot=True)
    max_d = max(max(d) for d in dend["dcoord"])
    for icoord, dcoord in zip(dend["icoord"], dend["dcoord"]):
        ys = [(v / 10.0 - 0.5) for v in icoord]
        xs = [(-v / max_d)     for v in dcoord]
        ax.plot(xs, ys, color="#222222", lw=1.3, solid_capstyle="butt")
    for k in range(NC):
        ax.plot([0.0, 0.025], [k, k], color="#222222", lw=1.1)
    ax.set_ylim(NC - 0.5, -0.5)
    ax.set_xlim(-1.02, 0.06)
    ax.axis("off")
    return dend["leaves"]


# =============================================================================
# PANEL 4a — Part B grouped heatmap
# =============================================================================
def panel_a():
    print("Building panel 4a ...")

    dea = pd.read_csv(PART_B_DEA)
    pivot_fc = (dea[dea["protein"].isin(LOCKED_25)]
                .pivot(index="protein", columns="target_class", values="fold_change"))
    prot_to_cancer = pivot_fc.idxmax(axis=1).to_dict()

    rk  = pd.read_csv(PART_B_RANKER)
    lr  = rk[rk["ranker"]=="LOGISTIC_COEF"].set_index("protein")["importance"]

    prot_ord  = []
    grp_spans = {}
    for cancer in CANCER_ORDER:
        grp = sorted([p for p in LOCKED_25 if prot_to_cancer.get(p) == cancer],
                     key=lambda p: -float(lr.get(p, 0)))
        if grp:
            x0 = len(prot_ord)
            prot_ord.extend(grp)
            grp_spans[cancer] = (x0, len(prot_ord) - 1)

    NP = len(prot_ord)
    NC = len(CANCER_ORDER)
    imp_ord = np.array([float(lr.get(p, 0)) for p in prot_ord])

    raw      = pd.read_csv(MAIN_DATA, usecols=["Cancer"] + LOCKED_25)
    mean_npx = raw.groupby("Cancer")[LOCKED_25].mean().loc[CANCER_ORDER]
    z        = (mean_npx - mean_npx.mean()) / mean_npx.std().replace(0, 1)
    z_col    = z[prot_ord]

    cl_link = linkage(pdist(z_col.values, "euclidean"), "average")

    fig = plt.figure(figsize=(10.5, 6.8))

    # 4-row GridSpec: colorbar row | bar | stripe | heatmap+dendrogram
    # Row 0 (thin) holds the colorbar just below the suptitle and above the bar chart
    gs = mgridspec.GridSpec(4, 2, figure=fig,
                            height_ratios=[0.12, 0.18, 0.04, 1.0],
                            width_ratios=[0.11, 1.0],
                            hspace=0.0, wspace=0.01)
    ax_hidden0 = fig.add_subplot(gs[0, 0]); ax_hidden0.set_axis_off()
    ax_cbrow   = fig.add_subplot(gs[0, 1])   # colorbar lives here
    ax_hidden1 = fig.add_subplot(gs[1, 0]); ax_hidden1.set_axis_off()
    ax_bar     = fig.add_subplot(gs[1, 1])
    ax_stripe  = fig.add_subplot(gs[2, 1])
    ax_dend    = fig.add_subplot(gs[3, 0])
    ax_hm      = fig.add_subplot(gs[3, 1])

    leaves     = _row_dendrogram(ax_dend, cl_link, NC)
    cancer_ord = [CANCER_ORDER[i] for i in leaves]

    z_plot = z_col.loc[cancer_ord].values
    vmax   = 2.0   # was 2.5 -- 74% of cells have |Z|<1.0; tighter scale + 5-stop
                   # cmap keeps ordinary cells legible instead of just outliers
    im = ax_hm.imshow(z_plot, aspect="auto", cmap=PARTB_CMAP,
                      vmin=-vmax, vmax=vmax, interpolation="nearest")
    _cell_gridlines(ax_hm, NC, NP)

    ax_hm.set_yticks(range(NC))
    ax_hm.set_yticklabels([display_label(c) for c in cancer_ord],
                          fontsize=8.5, fontweight="bold")
    ax_hm.yaxis.tick_right()
    for tk, c in zip(ax_hm.get_yticklabels(), cancer_ord):
        tk.set_color(PLOT_COLOR[c])

    ax_hm.set_xticks(range(NP))
    ax_hm.set_xticklabels(prot_ord, rotation=55, ha="right",
                           fontsize=10, fontweight="bold")
    ax_hm.xaxis.tick_bottom()
    ax_hm.tick_params(length=0)
    for sp in ax_hm.spines.values():
        sp.set_visible(False)

    for cancer, (x0, _) in grp_spans.items():
        if x0 > 0:
            ax_hm.axvline(x0 - 0.5, color="white", lw=1.8, zorder=5)

    # Colorbar in dedicated row (gs[0,1]) — compact, centered, above the bar chart
    ax_cbrow.set_axis_off()
    cax = ax_cbrow.inset_axes([0.20, 0.06, 0.60, 0.78])
    cb  = fig.colorbar(im, cax=cax, orientation="horizontal")
    # Put the numeric anchors inside the spectrum itself.  Tick labels below
    # this very shallow inset were clipped when the panel was reduced in the
    # mosaic, leaving a coloured bar with no readable meaning.
    cb.set_ticks([])
    cb.ax.text(0.025, 0.50, "−2", transform=cb.ax.transAxes,
               ha="left", va="center", fontsize=10.5, fontweight="bold",
               color="white")
    cb.ax.text(0.50, 0.50, "0", transform=cb.ax.transAxes,
               ha="center", va="center", fontsize=10.5, fontweight="bold",
               color="#222222")
    cb.ax.text(0.975, 0.50, "+2", transform=cb.ax.transAxes,
               ha="right", va="center", fontsize=10.5, fontweight="bold",
               color="white")
    ax_cbrow.text(0.50, 0.98, "Mean NPX z-score", transform=ax_cbrow.transAxes,
                  ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                  color="#222222")

    # Cancer color stripe
    for xi, protein in enumerate(prot_ord):
        pri = prot_to_cancer[protein]
        ax_stripe.add_patch(Rectangle((xi - 0.5, 0), 1, 1,
                                       facecolor=PLOT_COLOR[pri], linewidth=0))
    for cancer, (x0, x1) in grp_spans.items():
        cx        = (x0 + x1) / 2.0
        is_single = (x0 == x1)
        ax_stripe.text(cx, 0.5, display_label(cancer),
                       ha="center", va="center",
                       fontsize=5.5, fontweight="bold", color="white",
                       rotation=90 if is_single else 0)
    ax_stripe.set_xlim(-0.5, NP - 0.5)
    ax_stripe.set_ylim(0, 1)
    ax_stripe.axis("off")

    # Importance bars
    for xi, protein in enumerate(prot_ord):
        pri = prot_to_cancer[protein]
        ax_bar.bar(xi, imp_ord[xi], color=PLOT_COLOR[pri], width=0.72, alpha=0.90)
    for cancer, (x0, _) in grp_spans.items():
        if x0 > 0:
            ax_bar.axvline(x0 - 0.5, color="#cccccc", lw=0.8)
    ax_bar.set_xlim(-0.5, NP - 0.5)
    ax_bar.set_ylim(0, imp_ord.max() * 1.18)
    ax_bar.set_xticks([])
    ax_bar.set_ylabel("|LR-L2\ncoef.|", fontsize=6.5, fontweight="bold", labelpad=2)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["bottom"].set_visible(False)
    ax_bar.tick_params(axis="y", labelsize=6)

    save(fig, "fig4a_partb_expression_heatmap")


# =============================================================================
# PANEL 4b — Part A: 2×6 figure, global Z-score scale
# =============================================================================
def panel_b():
    print("Building panel 4b (2×6, global Z-score vmax=2.0) ...")

    s1 = pd.read_csv(SUPP_S1)

    cancer_data = {}
    all_uniq: list[str] = []
    seen: set[str] = set()

    for cancer in CANCER_ORDER:
        deployed = _load_deploy(s1, cancer)
        slug     = cancer.lower()
        rk_path  = PART_A_SINGLE / slug / "tables" / f"{slug}_top_feature_ranking.csv"
        rk       = pd.read_csv(rk_path).set_index("protein")
        imps     = {p: float(rk.loc[p, "importance"]) if p in rk.index else 0.0
                    for p in deployed}
        dep_s    = sorted(deployed, key=lambda p: -imps[p])
        cancer_data[cancer] = {"proteins": dep_s}
        for p in dep_s:
            if p not in seen:
                all_uniq.append(p)
                seen.add(p)

    raw      = pd.read_csv(MAIN_DATA, usecols=["Cancer"] + all_uniq)
    mean_npx = raw.groupby("Cancer")[all_uniq].mean().loc[CANCER_ORDER]
    z = (mean_npx - mean_npx.mean()) / mean_npx.std().replace(0, 1)

    NROW, NCOL = 2, 6
    VMAX = 2.0   # was 4.0 -- same 74%-within-|Z|<1.0 distribution as 4a; matched
                 # to 4a's scale so both panels share one consistent color language
    _SP4 = dict(color="#666666", lw=0.9)   # tile spine style matching fig8

    fig, axes = plt.subplots(NROW, NCOL, figsize=(24.0, 13.4),
                             sharey=False, sharex=False, squeeze=False)
    axes = axes.ravel()
    fig.subplots_adjust(hspace=0.70, wspace=0.34,
                        left=0.035, right=0.99, top=0.94, bottom=0.14)

    for idx, cancer in enumerate(CANCER_ORDER):
        ax    = axes[idx]
        prots = cancer_data[cancer]["proteins"]
        n     = len(prots)
        col   = PLOT_COLOR[cancer]
        cmap  = _cancer_cmap(col)

        z_mat = z.loc[CANCER_ORDER, prots].T.values   # (n_prots, 12)

        im = ax.imshow(z_mat, aspect="auto", cmap=cmap,
                       vmin=-VMAX, vmax=VMAX, interpolation="nearest")
        _cell_gridlines(ax, n, 12)

        # Protein y-labels — bigger and readable
        ax.set_yticks(range(n))
        ax.set_yticklabels(prots, fontsize=8, fontweight="bold")

        # Cancer x-ticks — all colored, target bolded and larger
        ax.set_xticks(range(12))
        ax.set_xticklabels([display_label(c) for c in CANCER_ORDER],
                           rotation=62, ha="right", fontsize=12)
        target_xi = CANCER_ORDER.index(cancer)
        for i, (tk, c) in enumerate(zip(ax.get_xticklabels(), CANCER_ORDER)):
            tk.set_color(PLOT_COLOR[c])
            if i == target_xi:
                tk.set_fontweight("bold")
                tk.set_fontsize(14)

        # Target column highlight rectangle
        ax.add_patch(Rectangle((target_xi - 0.5, -0.5), 1, n,
                                fill=False, edgecolor=col, lw=2.5, zorder=5))

        # Mini colorbar above panel (left 52% width)
        cax = ax.inset_axes([0.0, 1.03, 0.52, 0.10])
        cb  = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks([-2.0, 0.0, 2.0])
        cb.ax.set_xticklabels(["−2", "0", "+2"], fontsize=10.5,
                              fontweight="bold")
        cb.ax.tick_params(labelsize=10.5, length=2.5, pad=2)
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.xaxis.set_label_position("top")

        # Large bold colored title right of colorbar
        ax.text(0.55, 1.065, f"{display_label(cancer)}  n={n}",
                color=col, fontweight="bold", fontsize=14,
                transform=ax.transAxes, ha="left", va="center")

        # Mosaic-style tile border — all 4 spines visible
        ax.tick_params(length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(_SP4["color"])
            sp.set_linewidth(_SP4["lw"])

    save(fig, "fig4b_parta_expression_heatmap")


# =============================================================================
# PANEL 4d — Global (Part B) vs local (Part A) importance concordance
# Fills slide 4's whitespace with a genuine analytical question the other
# panels don't address: are proteins that matter for distinguishing ALL 12
# cancers (Part B) also the proteins that matter for confirming ONE specific
# cancer (Part A)? One point per (protein, cancer) pair where a locked-25
# protein appears in that cancer's own deploy panel -- proteins reused across
# multiple cancers appear as multiple points at the same x, different y/color.
# =============================================================================
def panel_d():
    print("Building panel 4d (importance concordance) ...")

    s1   = pd.read_csv(SUPP_S1)
    rk_b = pd.read_csv(PART_B_RANKER)
    lr   = rk_b[rk_b["ranker"] == "LOGISTIC_COEF"].set_index("protein")["importance"]

    records = []
    for cancer in CANCER_ORDER:
        deployed = _load_deploy(s1, cancer)
        rk_path  = PART_A_SINGLE / cancer.lower() / "tables" / f"{cancer.lower()}_top_feature_ranking.csv"
        rk_a     = pd.read_csv(rk_path).set_index("protein")
        for p in deployed:
            if p in LOCKED_25 and p in lr.index and p in rk_a.index:
                records.append(dict(protein=p, cancer=cancer,
                                    b_imp=float(lr.loc[p]),
                                    a_imp=float(rk_a.loc[p, "importance"])))
    df = pd.DataFrame(records)

    rho, pval = spearmanr(df["b_imp"], df["a_imp"])

    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    fig.subplots_adjust(left=0.12, right=0.985, top=0.97, bottom=0.20)

    x_norm = (df["b_imp"] - df["b_imp"].min()) / max(df["b_imp"].max() - df["b_imp"].min(), 1e-12)
    y_norm = (df["a_imp"] - df["a_imp"].min()) / max(df["a_imp"].max() - df["a_imp"].min(), 1e-12)
    label_idx = set(
        df.assign(_priority=np.maximum(x_norm, y_norm) + 0.15 * (x_norm + y_norm))
        .sort_values("_priority", ascending=False)
        .drop_duplicates("protein")
        .head(8)
        .index
    )
    texts = []
    label_offsets = {
        "TRAF2": (12, 18),
        "KLK13": (12, -10),
        "ADAMTS13": (10, 13),
        "CRTAC1": (10, -11),
    }
    for idx, row in df.iterrows():
        col = PLOT_COLOR[row["cancer"]]
        ax.scatter(row["b_imp"], row["a_imp"], s=180, color=col,
                   ec="white", lw=1.4, zorder=3, alpha=0.94)
        if idx not in label_idx:
            continue
        # Offset labels from their markers so the dot cannot hide the first
        # character (for example the T in TCL1A).
        dx, dy = label_offsets.get(row["protein"], (7, 5))
        texts.append(ax.annotate(
            row["protein"], (row["b_imp"], row["a_imp"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=8.8, fontweight="bold", color=col, zorder=4,
            ha="left", va="center",
        ))

    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6, alpha=0.7),
                expand=(2.0, 2.4), force_text=(1.3, 1.6), force_points=(1.0, 1.3))

    # Legend: one swatch per cancer that actually appears
    used_cancers = [c for c in CANCER_ORDER if c in df["cancer"].values]
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=PLOT_COLOR[c],
                          markersize=8, label=display_label(c))
              for c in used_cancers]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              fontsize=9.5, frameon=False, ncol=6, handletextpad=0.3,
              columnspacing=0.75, borderaxespad=0.0)

    ax.set_xlabel("Multiclass importance  |β|",
                 fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Cancer-specific panel importance",
                 fontsize=12, fontweight="bold", labelpad=8)
    ax.grid(color="#eeeeee", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    ax.text(0.03, 0.97, f"Spearman ρ = {rho:.2f}",
           transform=ax.transAxes, ha="left", va="top",
           fontsize=11.5, fontweight="bold", color="#333333",
           bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                     edgecolor="#cccccc", linewidth=1.0))

    save(fig, "fig4d_importance_concordance")


# =============================================================================
if __name__ == "__main__":
    panel_a()
    panel_b()
    panel_d()
    print("\nAll fig4 panels written.")
