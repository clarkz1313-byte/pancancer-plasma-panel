#!/usr/bin/env python3
"""
Fig 3 panels for FASEB submission - figurev5 v6
  volcano : 12 rows x 1 col  (saved to figurev5/output/)
  b       : Bonferroni DEA  -- narrow horizontal bars (3.2" wide)
  c       : Part A LASSO    -- 12 rows x 1 col  (3.2" wide)
  d       : Part B LR-L2    -- portrait, importance on X / rank on Y

Run from repo root:
    python figurev5/scripts/fig3_panels.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_labels import display_label
from publication_style import enable_text_scaling

enable_text_scaling(factor=1.50, minimum=10.0)

# -- paths --------------------------------------------------------------------
ROOT   = Path(__file__).resolve().parents[2]
REVISE = ROOT / "revise_plan"
OUT    = ROOT / "figurev6" / "output"
OUT.mkdir(parents=True, exist_ok=True)
V4OUT  = ROOT / "figurev4"

PART_B_DEA    = REVISE / "part_b_multiclass/vRSX_v11_locked_reproducer/tables/train_only_ovr_dea_all.csv"
PART_B_RANKER = REVISE / "part_b_multiclass/vRSX_v11_locked_reproducer/tables/train_only_ranker_table.csv"
PART_A_SINGLE = REVISE / "part_a_single"
SUPP_S1       = ROOT / "supplementary_tables/Supplementary_Table_S1_single_panel_protein_summary.csv"
VOLCANO_SRC   = ROOT / "figurev4" / "source_data" / "bonferroni_volcano_source.csv"

# -- constants ----------------------------------------------------------------
CANCER_ORDER = ["AML", "BRC", "CLL", "CRC", "CVX", "ENDC",
                "GLIOM", "LUNGC", "LYMPH", "MYEL", "OVC", "PRC"]

CANCER_COLOR = {
    "AML":   "#b22222", "BRC":  "#c97b63", "CLL":  "#7a3e9d",
    "CRC":   "#d68600", "CVX":  "#c13d86", "ENDC": "#8e5d2c",
    "GLIOM": "#7ec8e3", "LUNGC":"#2e8b57", "LYMPH":"#3856a6",
    "MYEL":  "#8c564b", "OVC":  "#d1495b", "PRC":  "#008b8b",
}
# Darker GLIOM for white-background plots only; global CANCER_COLOR unchanged
PLOT_COLOR = {**CANCER_COLOR, "GLIOM": "#1a8db8"}

LOCKED_25 = [
    "FLT3",    "CNTN1",   "FCER2",   "PRDX6",   "LTA4H",
    "XG",      "CCDC80",  "CXCL17",  "CXCL13",  "SLAMF7",
    "PAEP",    "PSPN",    "BMP4",    "WFDC2",   "TRAF2",
    "KLK13",   "GLO1",    "GFAP",    "CEACAM5", "CGA",
    "ADAMTS13","CRTAC1",  "TCL1A",   "ADAMTS15","NEFL",
]
PARTB_ACCENT = "#9467bd"   # purple — distinct from all 12 cancer colors

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "font.weight": "bold",
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 8,
    "legend.frameon": False,
})
DPI = 400

COL_W = 3.2   # shared figure width for b, c, and volcano


def save(fig: plt.Figure, stem: str, out: Path = OUT) -> None:
    for ext in ("pdf", "png"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


def save_v4(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png", "svg"):
        p = V4OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)


def _spread_labels(y_positions: np.ndarray, min_gap: float,
                   y_min: float, y_max: float,
                   iterations: int = 500) -> np.ndarray:
    ys = np.array(y_positions, dtype=float).copy()
    n  = len(ys)
    if (n - 1) * min_gap > (y_max - y_min):
        return np.linspace(y_min, y_max, n)
    for _ in range(iterations):
        moved = False
        for i in range(1, n):
            if ys[i] - ys[i - 1] < min_gap:
                mid       = (ys[i] + ys[i - 1]) / 2.0
                ys[i - 1] = max(y_min, mid - min_gap / 2.0)
                ys[i]     = ys[i - 1] + min_gap
                moved     = True
        if ys[-1] > y_max:
            ys[-1] = y_max
            for i in range(n - 2, -1, -1):
                ys[i] = min(ys[i], ys[i + 1] - min_gap)
        if not moved:
            break
    return ys


# =============================================================================
# VOLCANO  --  12 rows x 1 col
# =============================================================================
# ─── reusable per-cancer tile renderers ──────────────────────────────────────
# Extracted from the panel_*() loop bodies so the stacked slide-3 mosaic
# (fig3_slide3_mosaic.py) can draw the SAME tiles into a shared GridSpec
# instead of re-implementing them. panel_volcano/panel_b/panel_c below now
# call these, so a style fix lands on both the standalone panels and the
# mosaic -- same rule fig89_common.py follows for slides 8/9.
def _volcano_ax(ax, src, cancer, is_bottom, is_top, show_label=True,
                show_protein_labels=True):
    sub   = src[src["target_class"] == cancer].copy()
    color = PLOT_COLOR[cancer]
    sig   = sub[sub["significant_bonf"] == True]
    ns    = sub[sub["significant_bonf"] == False]
    # Kept for the compact badge cleanup block shared with the DEA tile.
    fc_up = sig[sig["fold_change"] > 0]["fold_change"].values
    fc_dn = sig[sig["fold_change"] < 0]["fold_change"].values

    ax.scatter(ns["fold_change"],  ns["minus_log10_raw_p"],
               color="#cccccc", s=6, linewidths=0, zorder=1)
    ax.scatter(sig["fold_change"], sig["minus_log10_raw_p"],
               color=color, s=14, linewidths=0, zorder=2)

    thresh_y = float(-np.log10(sub["bonferroni_raw_p_threshold"].iloc[0]))
    ax.axhline(thresh_y, color="#888888", linewidth=0.7, linestyle="--", zorder=0)

    if show_protein_labels:
        labeled = sub[sub["label_in_figure"] == True]
        for _, row in labeled.iterrows():
            x_off = 3 if row["fold_change"] >= 0 else -3
            ax.annotate(row["protein"],
                        xy=(row["fold_change"], row["minus_log10_raw_p"]),
                        xytext=(x_off, 1), textcoords="offset points",
                        fontsize=7, fontweight="bold", color=color,
                        va="bottom", ha="left" if x_off > 0 else "right")

    # Replace the tiny legacy count labels with high-contrast badges.
    for _txt in ax.texts[-2:]:
        _txt.set_visible(False)
    ax.text(0.99, 0.77, f"↑{len(fc_up)}", transform=ax.transAxes,
            va="center", ha="right", fontsize=11, fontweight="bold", color="#e05c5c",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.3))
    ax.text(0.99, 0.25, f"↓{len(fc_dn)}", transform=ax.transAxes,
            va="center", ha="right", fontsize=11, fontweight="bold", color="#4878cf",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.3))

    if show_label:
        ax.set_title(display_label(cancer), color=color, fontweight="bold",
                     fontsize=11, pad=2, loc="left")
    for _txt in ax.texts[-2:]:
        _txt.set_visible(False)
    ax.set_xlim(-6, 6)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Mean abundance difference" if is_bottom else "",
                  fontsize=8, fontweight="bold", labelpad=1)
    ax.set_ylabel("$-\log_{10}$(P)" if is_top else "",
                  fontsize=8, fontweight="bold", labelpad=1)
    if not is_bottom:
        ax.tick_params(axis="x", labelbottom=False)


_BXP = dict(
    vert=False, patch_artist=True, widths=[0.80],
    whiskerprops=dict(linewidth=0.7, color="#888888"),
    capprops=dict(linewidth=0.7, color="#888888"),
    flierprops=dict(marker=".", markersize=1.2, alpha=0.40, color="#aaaaaa"),
    medianprops=dict(color="white", linewidth=1.6),
)


def _bonf_ax(ax, dea, cancer, is_bottom, show_label=True):
    sub   = dea[dea["target_class"] == cancer]
    sig   = sub[sub["significant_bonf"] == True]
    fc_up = sig[sig["fold_change"] > 0]["fold_change"].values
    fc_dn = sig[sig["fold_change"] < 0]["fold_change"].values
    color = PLOT_COLOR[cancer]

    if len(fc_dn) >= 2:
        bp = ax.boxplot(fc_dn, positions=[0], **_BXP)
        bp["boxes"][0].set_facecolor("#4878cf")
        bp["boxes"][0].set_alpha(0.82)
        bp["boxes"][0].set_linewidth(0.5)
    if len(fc_up) >= 2:
        bp = ax.boxplot(fc_up, positions=[1], **_BXP)
        bp["boxes"][0].set_facecolor("#e05c5c")
        bp["boxes"][0].set_alpha(0.82)
        bp["boxes"][0].set_linewidth(0.5)

    ax.axvline(0, color="#cccccc", linewidth=0.6, zorder=0)
    ax.text(0.99, 0.77, f"↑{len(fc_up)}", transform=ax.transAxes,
            va="center", ha="right", fontsize=8, fontweight="bold", color="#e05c5c")
    ax.text(0.99, 0.25, f"↓{len(fc_dn)}", transform=ax.transAxes,
            va="center", ha="right", fontsize=8, fontweight="bold", color="#4878cf")

    if show_label:
        ax.set_title(display_label(cancer), color=color, fontweight="bold",
                     fontsize=11, pad=2, loc="left")
    # Replace the thin legacy counts with high-contrast badges.
    for _txt in ax.texts[-2:]:
        _txt.set_visible(False)
    ax.text(0.99, 0.77, f"↑{len(fc_up)}", transform=ax.transAxes,
            va="center", ha="right", fontsize=11, fontweight="bold", color="#e05c5c",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.3))
    ax.text(0.99, 0.25, f"↓{len(fc_dn)}", transform=ax.transAxes,
            va="center", ha="right", fontsize=11, fontweight="bold", color="#4878cf",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.3))
    ax.set_xlim(-4.5, 6.0)
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xlabel("Mean abundance difference" if is_bottom else "",
                  fontsize=8, fontweight="bold", labelpad=1)
    if not is_bottom:
        ax.tick_params(axis="x", labelbottom=False)


def _lasso_ax(ax, s1, cancer, is_bottom, is_top, show_label=True):
    slug  = cancer.lower()
    rpath = PART_A_SINGLE / slug / "tables" / f"{slug}_top_feature_ranking.csv"
    all_df = pd.read_csv(rpath).sort_values("importance", ascending=False).reset_index(drop=True)

    nz_df = all_df[all_df["importance"] > 0].copy().reset_index(drop=True)
    nz_df["nz_rank"] = range(1, len(nz_df) + 1)
    n_nz   = len(nz_df)
    y_ceil = max(n_nz + 2, 6)

    deployed   = _load_deploy_proteins(s1, cancer)
    color      = PLOT_COLOR[cancer]
    dep_df     = nz_df[nz_df["protein"].isin(deployed)].copy()
    non_dep_df = nz_df[~nz_df["protein"].isin(deployed)].copy()

    ax.plot(nz_df["importance"], nz_df["nz_rank"],
            color="#dedede", linewidth=0.9, zorder=1)
    ax.scatter(non_dep_df["importance"], non_dep_df["nz_rank"],
               color="#c0c3c8", s=9, zorder=2, alpha=0.75, linewidths=0)
    for _, row in dep_df.iterrows():
        ax.hlines(y=row["nz_rank"], xmin=0, xmax=row["importance"],
                  colors=color, linewidths=1.4, alpha=0.85, zorder=3)
    ax.scatter(dep_df["importance"], dep_df["nz_rank"],
               color=color, s=30, zorder=4, linewidths=0)

    ax.invert_yaxis()
    ax.set_ylim(y_ceil, 0)
    ax.set_xlim(left=0)
    if show_label:
        ax.set_title(display_label(cancer), color=color, fontweight="bold",
                     fontsize=11, pad=2, loc="left")
    ax.tick_params(axis="both", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("|LASSO coef.|" if is_bottom else "", fontsize=8,
                  fontweight="bold", labelpad=1)
    ax.set_ylabel("LASSO rank" if is_top else "", fontsize=8,
                  fontweight="bold", labelpad=1)
    # Match the v5 strip: report the number of non-zero ranked proteins in
    # each cancer tile without adding another legend or title.
    ax.text(0.99, 0.13, str(n_nz), transform=ax.transAxes,
            ha="right", va="center", fontsize=8, fontweight="bold",
            color="#b7b7b7")
    if not is_bottom:
        ax.tick_params(axis="x", labelbottom=False)


def panel_volcano() -> None:
    print("Building volcano (12 x 1) ...")
    src = pd.read_csv(VOLCANO_SRC)

    fig, axes = plt.subplots(12, 1, figsize=(COL_W, 12.0))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.99, bottom=0.04, hspace=0.40)

    for ax, cancer in zip(axes, CANCER_ORDER):
        _volcano_ax(ax, src, cancer,
                    is_bottom=(cancer == CANCER_ORDER[-1]),
                    is_top=(cancer == CANCER_ORDER[0]))

    save(fig, "fig3a_volcano")


# =============================================================================
# PANEL B  --  Bonferroni DEA counts  (narrow horizontal bars, COL_W wide)
# =============================================================================
def panel_b() -> None:
    print("Building panel b ...")
    dea = pd.read_csv(PART_B_DEA,
                      usecols=["target_class", "significant_bonf", "fold_change"])

    fig, axes = plt.subplots(12, 1, figsize=(COL_W, 12.0))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.99, bottom=0.04, hspace=0.40)

    for ax, cancer in zip(axes, CANCER_ORDER):
        _bonf_ax(ax, dea, cancer, is_bottom=(cancer == CANCER_ORDER[-1]))

    handles = [
        mpatches.Patch(color="#e05c5c", alpha=0.82, label="sig. up"),
        mpatches.Patch(color="#4878cf", alpha=0.82, label="sig. down"),
    ]
    axes[0].legend(handles=handles, loc="upper right", ncol=1,
                   fontsize=8, frameon=False)

    save(fig, "fig3b_bonferroni_counts")


# =============================================================================
# PANEL C  --  Part A LASSO profiles  (12 rows x 1 col, COL_W wide)
# =============================================================================
def _load_deploy_proteins(s1: pd.DataFrame, cancer: str) -> list[str]:
    row = s1[s1["target_class"] == cancer].iloc[0]
    proteins = [p.strip() for p in str(row["deploy_proteins"]).split(";") if p.strip()]
    return proteins[:int(row["deploy_panel_size"])]


def panel_c() -> None:
    print("Building panel c ...")
    s1 = pd.read_csv(SUPP_S1)

    fig, axes = plt.subplots(12, 1, figsize=(COL_W, 12.0))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.99, bottom=0.04, hspace=0.40)

    for ax, cancer in zip(axes, CANCER_ORDER):
        _lasso_ax(ax, s1, cancer,
                  is_bottom=(cancer == CANCER_ORDER[-1]),
                  is_top=(cancer == CANCER_ORDER[0]))

    save(fig, "fig3c_parta_lasso_rank_profiles")


# =============================================================================
# PANEL D  --  Part B LR-L2 importance curve  (portrait: importance X, rank Y)
# =============================================================================
def draw_partb_rank_profile(ax) -> None:
    """Part B LR-L2 selection curve into a caller-supplied axes, so both the
    standalone panel d and the slide-3 mosaic render the identical figure."""
    from matplotlib.lines import Line2D
    ranker = pd.read_csv(PART_B_RANKER)
    lr_df  = (
        ranker[ranker["ranker"] == "LOGISTIC_COEF"]
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    lr_df["rank"]        = np.arange(1, len(lr_df) + 1)
    lr_df["is_selected"] = lr_df["protein"].isin(LOCKED_25)

    sel     = lr_df[lr_df["is_selected"]].copy()
    non_sel = lr_df[~lr_df["is_selected"]].copy()
    x_max   = float(lr_df["importance"].max())
    n_all   = len(lr_df)

    ax.plot(lr_df["importance"], lr_df["rank"],
            color="#dedede", linewidth=0.8, zorder=1)
    ax.scatter(non_sel["importance"], non_sel["rank"],
               color="#b8bbc0", s=9, zorder=2, alpha=0.75, linewidths=0)
    ax.scatter(sel["importance"], sel["rank"],
               color=PARTB_ACCENT, s=60, zorder=4,
               linewidths=0.8, edgecolors="white")

    sel_sorted = sel.sort_values("rank").reset_index(drop=True)
    raw_ys     = sel_sorted["rank"].values.astype(float)
    label_ys   = _spread_labels(raw_ys, min_gap=7.0, y_min=2.0, y_max=n_all - 2.0)
    x_label    = x_max * 1.08

    for i, (_, row) in enumerate(sel_sorted.iterrows()):
        ax.annotate(
            row["protein"],
            xy=(row["importance"], row["rank"]),
            xytext=(x_label, label_ys[i]),
            textcoords="data",
            fontsize=9.2, fontweight="bold", color="#222", va="center", ha="left",
            arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.5),
        )

    ax.invert_yaxis()
    ax.set_xlim(left=0, right=max(0.33, x_label * 1.52))
    ax.set_ylim(n_all + 3, 0)
    ax.set_xlabel("Mean |LR-L2 coefficient|",
                  fontsize=9, fontweight="bold")
    ax.set_ylabel("Importance rank", fontsize=6.5,
                  fontweight="bold", labelpad=0)
    # A footer states the set size explicitly.  The former lower-left legend
    # sat beside the last-ranked point and made “25” look like that point's
    # rank rather than the total number selected.
    ax.text(0.50, 0.018, f"●  {len(sel)} selected / {n_all} total",
            transform=ax.transAxes, ha="center", va="bottom", clip_on=False,
            fontsize=10.5, fontweight="bold", color=PARTB_ACCENT,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=1.2))

    # Compact set-size key lives outside the plot; the legacy in-axes footer
    # is hidden to prevent it being read as a point annotation.
    if False:
        ax.legend(
        handles=[Line2D([0], [0], marker="o", linestyle="None", markersize=7,
                        markerfacecolor=PARTB_ACCENT, markeredgecolor="white",
                        label=f"{len(sel)}/{n_all} selected")],
        loc="upper right", bbox_to_anchor=(1.0, -0.018), frameon=True,
        facecolor="white", edgecolor="none", framealpha=0.86,
        fontsize=10, handletextpad=0.35, borderaxespad=0.0,
    )
    if ax.texts:
        ax.texts[-1].set_visible(False)
    # Keep the set-size key INSIDE the plotting area, immediately below the
    # lowest labelled candidate (CXCL13).  Data coordinates make its meaning
    # unambiguous and prevent collision with the x-axis title.
    ax.text(ax.get_xlim()[1] * 0.98, n_all - 0.5,
            f"{len(sel)}/{n_all} selected",
            transform=ax.transData, ha="right", va="center", clip_on=True,
            fontsize=10.5, fontweight="bold", color="#222222",
            bbox=dict(facecolor="white", edgecolor=PARTB_ACCENT,
                      linewidth=0.9, alpha=0.96, pad=2.0))


def panel_d() -> None:
    print("Building panel d ...")
    fig, ax = plt.subplots(figsize=(3.2, 7.2))
    draw_partb_rank_profile(ax)
    fig.tight_layout()
    save(fig, "fig3d_partb_lr_l2_rank_profile")


# =============================================================================
# PANEL E  --  25-protein Bonferroni significance membership  (dot matrix)
# =============================================================================
def draw_locked25_membership(ax) -> None:
    """Locked-25 x cancer Bonferroni-significance dot matrix into a
    caller-supplied axes (shared by standalone panel e and the mosaic)."""
    dea = pd.read_csv(PART_B_DEA,
                      usecols=["target_class", "protein", "significant_bonf"])

    sig_sets = {}
    for cancer in CANCER_ORDER:
        sub = dea[dea["target_class"] == cancer]
        sig_sets[cancer] = set(sub[sub["significant_bonf"] == True]["protein"])

    breadth  = {p: sum(1 for c in CANCER_ORDER if p in sig_sets[c]) for p in LOCKED_25}
    proteins = sorted(LOCKED_25, key=lambda p: (-breadth[p], p))

    n_prot = len(proteins)
    n_canc = len(CANCER_ORDER)
    YSPACE = 2.5
    SIG_S, NS_S = 180, 22

    for yi_idx, cancer in enumerate(CANCER_ORDER):
        y = yi_idx * YSPACE
        for xi, protein in enumerate(proteins):
            if protein in sig_sets[cancer]:
                ax.scatter(xi, y, s=SIG_S, color=PLOT_COLOR[cancer],
                           zorder=3, linewidths=0)
            else:
                ax.scatter(xi, y, s=NS_S, color="#e0e0e0",
                           zorder=1, linewidths=0)

    y_positions = [i * YSPACE for i in range(n_canc)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels([display_label(c) for c in CANCER_ORDER],
                       fontsize=11, fontweight="bold")
    for tick, cancer in zip(ax.yaxis.get_majorticklabels(), CANCER_ORDER):
        tick.set_color(PLOT_COLOR[cancer])

    for yi_idx, cancer in enumerate(CANCER_ORDER):
        y = yi_idx * YSPACE
        n_sig = sum(1 for p in proteins if p in sig_sets[cancer])
        ax.text(n_prot + 0.4, y, str(n_sig), va="center", ha="left",
                fontsize=9, fontweight="bold", color=PLOT_COLOR[cancer])

    ax.set_xticks(range(n_prot))
    ax.set_xticklabels(proteins, rotation=75, ha="right",
                       fontsize=7.2, fontweight="bold", color="#222222")

    y_max = (n_canc - 1) * YSPACE
    ax.set_ylim(y_max + 1.1, -1.1)
    ax.set_xlim(-0.7, n_prot + 1.2)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0, pad=4)


def panel_e() -> None:
    print("Building panel e (significance membership matrix) ...")
    fig, ax = plt.subplots(figsize=(6.0, 14.0))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.99, bottom=0.14)
    draw_locked25_membership(ax)
    save(fig, "fig3e_locked25_significance_membership")


# =============================================================================
if __name__ == "__main__":
    panel_volcano()
    panel_b()
    panel_c()
    panel_d()
    panel_e()
    print("\nAll panels written.")
