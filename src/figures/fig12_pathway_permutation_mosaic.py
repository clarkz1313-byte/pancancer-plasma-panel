#!/usr/bin/env python3
"""Reaudited Hallmark leading-edge overlap figure.

The primary single-panel null preserves cancer, signed rank position, and
panel size. The locked-25 comparison is shown only as candidate-bank
sensitivity context because the full locked-panel selection pipeline was not
repeated inside each draw.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from publication_style import enable_text_scaling

enable_text_scaling(factor=1.0, minimum=8.0)


ROOT = Path(__file__).resolve().parents[2]
GSEA_FIG = ROOT / "revise_plan" / "gsea_analysis" / "figures"
OUT = ROOT / "figurev6" / "output"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 400

INK = "#111820"
MUTED = "#46535f"
GRID = "#cbd5dc"
SINGLE = "#6a1b9a"
LOCKED = "#007c91"
ACCENT = "#e07a00"
PANEL_BG = "#fbfcfd"
CANCER_COLOR = {
    "AML": "#b22222", "BRC": "#c97b63", "CLL": "#7a3e9d",
    "CRC": "#d68600", "CVX": "#c13d86", "ENDC": "#8e5d2c",
    "GLIOM": "#1a8db8", "LUNGC": "#2e8b57", "DLBCL": "#3856a6",
    "MYEL": "#8c564b", "OVC": "#d1495b", "PRC": "#008b8b",
}

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 1.0,
    }
)


def load_data():
    summary = pd.read_csv(GSEA_FIG / "direction_matched_permutation_summary_v2.csv").iloc[0]
    null_single = pd.read_csv(GSEA_FIG / "direction_matched_null_single_v2.csv")["overlap_count"].to_numpy()
    per_cancer = pd.read_csv(GSEA_FIG / "direction_matched_permutation_per_cancer_v2.csv")
    tolerance = pd.read_csv(GSEA_FIG / "direction_matched_tolerance_sensitivity_v2.csv")
    locked = pd.read_csv(GSEA_FIG / "locked25_pathway_overlap_descriptive_v2.csv").iloc[0]
    null_locked = pd.read_csv(GSEA_FIG / "locked25_candidate_bank_null_v2.csv")["overlap_count"].to_numpy()
    return summary, null_single, per_cancer, tolerance, locked, null_locked


def clean_axis(ax):
    ax.set_facecolor(PANEL_BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.9, zorder=0)


def null_panel(ax, values, observed, color, letter, name, p_text):
    parts = ax.violinplot([values], positions=[0], widths=0.78, showextrema=False)
    body = parts["bodies"][0]
    body.set_facecolor(color)
    body.set_edgecolor(color)
    body.set_alpha(0.38)
    rng = np.random.default_rng(52)
    sample = rng.choice(values, size=min(700, len(values)), replace=False)
    ax.scatter(rng.uniform(-0.20, 0.20, len(sample)), sample, s=20,
               color=color, alpha=0.28, linewidth=0)
    # Keep the observed value and its star as two distinct visual marks.  The
    # previous arrangement overprinted the number on the star in dense null
    # distributions, making both difficult to read at manuscript scale.
    ax.scatter([0], [observed], marker="*", s=790, color=ACCENT,
               edgecolor="#7a3c00", linewidth=2.1, zorder=8)
    observed_text = ax.text(0, observed + 1.35, f"{int(observed)}",
                            ha="center", color=ACCENT, fontsize=18,
                            fontweight="bold", zorder=9)
    observed_text.set_path_effects([
        pe.withStroke(linewidth=3.0, foreground="white"), pe.Normal()
    ])
    ax.text(0.03, 0.04, p_text, transform=ax.transAxes, color=INK,
            fontsize=11, fontweight="bold", va="bottom")
    ax.set_xlim(-0.62, 0.62)
    ax.set_xticks([])
    ax.set_ylabel("Leading-edge overlap", fontsize=10)
    ax.set_title(letter, loc="left", fontsize=14)
    clean_axis(ax)


def per_cancer_panel(ax, data):
    order = ["AML", "BRC", "CLL", "CRC", "CVX", "ENDC", "GLIOM", "LUNGC", "DLBCL", "MYEL", "OVC", "PRC"]
    data = data.set_index("display_cancer").reindex(order).reset_index()
    y = np.arange(len(data))
    for index, row in data.iterrows():
        if pd.isna(row["null_mean"]):
            ax.text(0.08, index, "none at FDR < .25", va="center",
                    color=MUTED, fontsize=9.5, fontweight="bold")
            continue
        ax.plot([row["null_mean"], row["observed_overlap"]], [index, index],
                color=GRID, linewidth=4, zorder=1)
        ax.scatter(row["null_mean"], index, s=105, facecolor="white",
                   edgecolor=LOCKED, linewidth=2.3, zorder=3)
        ax.scatter(row["observed_overlap"], index, s=165,
                   color=CANCER_COLOR[row["display_cancer"]],
                   edgecolor="white", linewidth=1.0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Overlap count", fontsize=10)
    ax.set_title("B", loc="left", fontsize=14)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", ls="", markersize=9.5,
                       mfc=INK, mec="white", mew=0.9, label="Observed"),
            plt.Line2D([0], [0], marker="o", ls="", markersize=7.5,
                       mfc="white", mec=LOCKED, mew=1.4, label="Null"),
        ],
        frameon=False,
        loc="lower right", fontsize=10,
    )
    clean_axis(ax)


def tolerance_panel(ax, data):
    ax.plot(data["tolerance_pct_points"], data["empirical_p"], color=SINGLE,
            marker="o", markersize=11, linewidth=4.0)
    for _, row in data.iterrows():
        ax.text(row["tolerance_pct_points"], row["empirical_p"] + 0.035,
                f"{row['empirical_p']:.3f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.axhline(0.05, color=ACCENT, linestyle=":", linewidth=2.0)
    ax.text(1.6, 0.065, "0.05", color=ACCENT, fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_xticks(data["tolerance_pct_points"])
    ax.set_xlabel("Rank tolerance (percentage points)", fontsize=10)
    ax.set_ylabel("Empirical p", fontsize=10)
    ax.set_title("C", loc="left", fontsize=14)
    clean_axis(ax)


def main() -> None:
    summary, null_single, per_cancer, tolerance, locked, null_locked = load_data()
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5),
                             gridspec_kw={"hspace": 0.34, "wspace": 0.27})

    null_panel(
        axes[0, 0],
        null_single,
        summary["observed"],
        SINGLE,
        "A",
        "",
        f"p={summary['empirical_p']:.3f}  [{summary['mc_ci_lower']:.3f}–{summary['mc_ci_upper']:.3f}]",
    )
    per_cancer_panel(axes[0, 1], per_cancer)
    tolerance_panel(axes[1, 0], tolerance)
    null_panel(
        axes[1, 1],
        null_locked,
        locked["observed"],
        LOCKED,
        "D",
        "",
        f"descriptive: {locked['sensitivity_empirical_p']:.3f}",
    )

    fig.subplots_adjust(top=0.97, bottom=0.10, left=0.09, right=0.985)
    for extension in ("pdf", "png"):
        path = OUT / f"fig12_pathway_permutation_mosaic.{extension}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
