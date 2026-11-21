#!/usr/bin/env python3
"""
fig3_5_and_fig4_5_panel_size_selection.py

Candidate figures answering "why this panel size?", built standalone so that
neither Figure 3 nor Figure 4 has to be re-laid out:

  fig3_5   multiclass panel size selection (sizes 12-30, locked at 25 markers)
  fig4_5   disease-specific panel size selection (sizes 1-18, 12 cancers)
  fig3_6b  the multiclass sweep alone, for the Figure 3-6 mosaic cell

Design decisions these plots share, so that 3-5, 3-6 and 4-6 read as one family
-----------------------------------------------------------------------------
Colour. fig3_5 describes the MULTICLASS panel, which the deck renders in purple
(PARTB_CMAP in fig3_panels.py and fig4_panels.py). Its four metric series run
down that colormap's own stops, so none of the twelve cancer hues appears and a
metric line can never be misread as a cancer. fig4_5 is per-cancer, so each
tile keeps its own cancer colour.

Separating four flat curves. Shade alone does not do it once balanced accuracy
and macro F1 start overlapping, so every series also carries its own dash
pattern and its own marker. Four redundant channels means the reader never has
to resolve two of them at once.

The decision point. The selected size is not announced in a text box; it is
drawn. A hairline drops from the top of the axes, every series carries an open
ring where it crosses that size, and the size itself replaces a tick on the
x axis in the panel's own colour. The eye lands on the crossing, which is the
quantity the figure exists to justify.

Legends sit BELOW the x axis, never inside the data area, so the plotting
rectangle keeps its full height and can be matched against a heatmap of the
same cell width in a mosaic.

Sources, all seed 52:
  fig3_5  revise_plan/part_b_multiclass/v11_seed52_lr_l2_wide_compact_sensitivity/
          tables/best_lr_l2_by_feature_count.csv
  fig4_5  revise_plan/sensitivity_seed52_exact/part_a_single/<slug>/tables/
          <slug>_panel_performance.csv

The rule drawn in fig4_5 is the one implemented in revise_plan/scripts/
common.py (build_panel_review_tables): keep every panel whose held-out AUC is
within 0.02 of that cancer's best AND whose balanced accuracy is within 0.05
of its best, then take the smallest survivor.

Output: figurev6/output/ and overleaf-source-git/figures/ at 600 dpi.
Run:    python fig3_5_and_fig4_5_panel_size_selection.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fig_candidate_common import save_measured
from figure_labels import display_label, panel_letter
from publication_style import enable_text_scaling

enable_text_scaling(factor=1.40, minimum=10.0)

ROOT = Path(__file__).resolve().parents[2]
REVISE = ROOT / "revise_plan"
OUT = ROOT / "figurev6" / "output"
OVERLEAF = ROOT / "overleaf-source-git" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MULTI_SWEEP = (REVISE / "part_b_multiclass"
               / "v11_seed52_lr_l2_wide_compact_sensitivity"
               / "tables" / "best_lr_l2_by_feature_count.csv")
SINGLE_ROOT = REVISE / "sensitivity_seed52_exact" / "part_a_single"

LOCKED_SIZE = 25
AUC_TOL = 0.02        # common.py: Test_AUC >= max - 0.02
BAL_TOL = 0.05        # common.py: balanced_accuracy >= max - 0.05

CANCER_ORDER = ["AML", "BRC", "CLL", "CRC", "CVX", "ENDC",
                "GLIOM", "LUNGC", "LYMPH", "MYEL", "OVC", "PRC"]
PLOT_COLOR = {
    "AML": "#b22222", "BRC": "#c97b63", "CLL": "#7a3e9d",
    "CRC": "#d68600", "CVX": "#c13d86", "ENDC": "#8e5d2c",
    "GLIOM": "#1a8db8", "LUNGC": "#2e8b57", "LYMPH": "#3856a6",
    "MYEL": "#8c564b", "OVC": "#d1495b", "PRC": "#008b8b",
}
DEPLOYED = {"AML": 3, "BRC": 13, "CLL": 2, "CRC": 10, "CVX": 11, "ENDC": 10,
            "GLIOM": 3, "LUNGC": 6, "LYMPH": 5, "MYEL": 8, "OVC": 9, "PRC": 13}

# Part B purple ramp, taken from the PARTB_CMAP stops used by figures 3 and 4.
PARTB_DEEP = "#3f1d5e"
PARTB_MID = "#6a3d9a"
PARTB_ACCENT = "#9467bd"
PARTB_LIGHT = "#b98fd6"

# (value column, bootstrap-CI stem, label, colour, marker, dash)
METRICS = [
    ("test_macro_ovr_auc", "macro_ovr_auc", "Macro OvR AUC",
     PARTB_DEEP, "o", "-"),
    ("test_balanced_accuracy", "balanced_accuracy", "Balanced accuracy",
     PARTB_MID, "s", (0, (5, 1.4))),
    ("test_macro_f1", "macro_f1", "Macro $F_1$",
     PARTB_ACCENT, "^", (0, (3, 1.2, 1, 1.2))),
    ("test_min_class_recall", "min_class_recall", "Min class recall",
     PARTB_LIGHT, "D", (0, (1.4, 1.4))),
]

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "font.weight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 8,
    "legend.frameon": False,
})
DPI = 600


# ── shared drawing idioms ────────────────────────────────────────────────────
def mark_choice(ax, x, colour, ymax=1.0):
    """Hairline dropping from the top of the axes to the chosen size."""
    ax.axvline(x, color=colour, linestyle=(0, (2.5, 1.6)), linewidth=1.1,
               alpha=0.85, ymax=ymax, zorder=2)


def ring(ax, x, y, colour, size=10.5):
    """Open ring where a series crosses the chosen size."""
    ax.plot([x], [y], marker="o", markersize=size, markerfacecolor="white",
            markeredgecolor=colour, markeredgewidth=1.7, linestyle="none",
            zorder=8, clip_on=False)


def colour_choice_tick(ax, base_ticks, chosen, colour, keep_gap=1,
                       rotation=0):
    """Label the x axis, with the chosen size picked out in colour.

    keep_gap=0 keeps every tick and only recolours the chosen one, which is
    what the multiclass sweep wants: a reader checking "why 25 and not 24"
    needs 24 and 26 on the axis, so the labels are set diagonally rather than
    thinned.
    """
    ticks = [t for t in base_ticks if abs(t - chosen) > keep_gap]
    ticks = sorted(set(ticks) | {chosen})
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], rotation=rotation,
                       ha="right" if rotation else "center",
                       rotation_mode="anchor" if rotation else None)
    for tick, value in zip(ax.get_xticklabels(), ticks):
        if value == chosen:
            tick.set_color(colour)
            tick.set_fontweight("bold")
            tick.set_fontsize(tick.get_fontsize() * 1.3)


def _load_multi() -> pd.DataFrame:
    return (pd.read_csv(MULTI_SWEEP)
            .sort_values("feature_count")
            .reset_index(drop=True))


def multi_legend_handles():
    return [plt.Line2D([], [], color=c, linewidth=1.8, linestyle=st, marker=m,
                       markersize=7, label=lab)
            for _, _, lab, c, m, st in METRICS] + [
        plt.Line2D([], [], color=PARTB_DEEP, linewidth=1.1,
                   linestyle=(0, (2.5, 1.6)), marker="o", markersize=7,
                   markerfacecolor="white", markeredgecolor=PARTB_DEEP,
                   label="Selected size (25 markers)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=PARTB_ACCENT, alpha=0.20,
                      label="95% bootstrap interval")]


def draw_multi_sweep(ax, df, shortfall=False):
    """The multiclass sweep: four held-out metrics against panel size.

    On the value panel each series carries its 95% bootstrap interval as a
    band. That is not decoration: the minimum-class-recall band runs 0.44 to
    0.65 at the selected size, so a reader can see at once that the peak the
    figure is built around is the widest-intervalled of the four.
    """
    x = df["feature_count"]
    for col, ci, label, colour, marker, style in METRICS:
        y = (df[col].max() - df[col]) if shortfall else df[col]
        if not shortfall:
            ax.fill_between(x, df[f"{ci}_ci_lower"], df[f"{ci}_ci_upper"],
                            color=colour, alpha=0.16, linewidth=0, zorder=1)
        # Large markers, because four near-parallel curves in one hue family
        # are told apart by shape long before they are told apart by shade.
        ax.plot(x, y, marker=marker, markersize=6.2, linewidth=1.7,
                linestyle=style, color=colour, label=label,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        ring(ax, LOCKED_SIZE, float(y[x == LOCKED_SIZE].iloc[0]), colour)

    mark_choice(ax, LOCKED_SIZE, PARTB_DEEP)
    ax.set_xlabel("Panel size (markers)")
    ax.set_xlim(11.3, 30.7)
    colour_choice_tick(ax, list(range(12, 31)), LOCKED_SIZE, PARTB_DEEP,
                       keep_gap=0, rotation=55)
    ax.grid(axis="y", linewidth=0.4, alpha=0.32, zorder=0)
    ax.set_axisbelow(True)
    if shortfall:
        ax.axhline(0.0, color="#555555", linewidth=0.7, zorder=1)
        ax.set_ylabel("Shortfall from that metric's best")
    else:
        ax.set_ylabel("Held-out metric value")
        ax.set_ylim(0.05, 1.03)


# ── fig 3-5 ──────────────────────────────────────────────────────────────────
def build_fig3_5() -> pd.DataFrame:
    df = _load_multi()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.6, 4.5))
    draw_multi_sweep(ax_a, df)
    panel_letter(ax_a, "A", dx=-0.10, dy=1.01, size=20)
    draw_multi_sweep(ax_b, df, shortfall=True)
    panel_letter(ax_b, "B", dx=-0.10, dy=1.01, size=20)

    row25 = df.loc[df["feature_count"] == LOCKED_SIZE].iloc[0]
    hits = sum((df[c].max() - row25[c]) < 1e-9 for c, *_ in METRICS)
    ax_b.text(0.97, 0.93, f"{hits} of 4 metrics best here",
              transform=ax_b.transAxes, color=PARTB_DEEP, fontsize=8.5,
              ha="right", va="top")

    fig.legend(handles=multi_legend_handles(), loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.015), handlelength=2.2,
               columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.075, 1, 1), w_pad=3.0)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_5_multiclass_panel_size_selection.{ext}",
                    dpi=DPI, facecolor="white", bbox_inches="tight")
    fig.savefig(OVERLEAF / "fig3_5_multiclass_panel_size_selection.png",
                dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # Standalone sweep for the Figure 3-6 mosaic cell. The legend sits below
    # the axis so the plotting rectangle keeps its full height; the mosaic
    # builder then pads this image to fig4a's aspect so the two cells render
    # at exactly the same height.
    # Taller than wide-ish: after the mosaic pads for baseline alignment the
    # axes should still fill its cell rather than float above white space.
    fig2, ax = plt.subplots(figsize=(9.6, 7.8))
    draw_multi_sweep(ax, df)
    fig2.legend(handles=multi_legend_handles(), loc="lower center", ncol=3,
                frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=2.2,
                columnspacing=1.4, fontsize=10)
    fig2.tight_layout(rect=(0, 0.135, 1, 1))
    save_measured(fig2, OUT / "fig3_6a_multiclass_size_sweep.png", ax, dpi=400)
    plt.close(fig2)
    return df


# ── fig 4-5 ──────────────────────────────────────────────────────────────────
def load_single(slug: str) -> pd.DataFrame:
    path = SINGLE_ROOT / slug / "tables" / f"{slug}_panel_performance.csv"
    d = pd.read_csv(path)
    d = d[d["Model"].str.contains("Top", na=False)].copy()
    d["Features"] = d["Features"].astype(int)
    return d.sort_values("Features").reset_index(drop=True)


# AUC and F1 both take the cancer's colour and are separated by dash pattern;
# balanced accuracy stays grey. Two coloured curves against one neutral reads
# faster than three curves that differ only in dash, which is what the author
# could not tell apart in the previous draft.
SINGLE_SERIES = [
    ("Test_AUC", "Held-out AUC", 1.9, "-", True),
    ("balanced_accuracy", "Held-out balanced accuracy", 1.2, (0, (5, 1.4)), False),
    # Long dash, long gap: at Figure 4-6's tile size a tight dash-dot in the
    # cancer's own colour was not separable from the solid AUC curve.
    ("f1", "Held-out $F_1$", 1.5, (0, (7, 3, 1.6, 3)), True),
]


def draw_single_sweep(ax, cancer, d, show_ylabel=True, show_xlabel=True,
                      show_title=True, tick_size=None, axis_label_size=None):
    """One cancer's size sweep: AUC, balanced accuracy and F1 against size.

    tick_size / axis_label_size exist for Figure 4-7, where the tile beside
    each sweep sets its own label size explicitly; without them the sweep's
    ticks come out visibly smaller than the protein names next to them once
    the whole strip is downscaled into a mosaic cell.
    """
    colour = PLOT_COLOR[cancer]
    if tick_size is not None:
        ax.tick_params(labelsize=tick_size)
    auc_best = d["Test_AUC"].max()
    chosen = DEPLOYED[cancer]
    at_chosen = d[d["Features"] == chosen]

    ax.axhspan(auc_best - AUC_TOL, auc_best, color=colour, alpha=0.22,
               linewidth=0, zorder=0)
    ax.axhline(auc_best - AUC_TOL, color=colour, linewidth=0.6, alpha=0.55,
               zorder=1)
    for col, _label, lw, style, tinted in SINGLE_SERIES:
        shade = colour if tinted else "#5a5a5a"
        ax.plot(d["Features"], d[col], color=shade, linewidth=lw,
                linestyle=style, marker="o" if col == "Test_AUC" else None,
                markersize=2.6, zorder=4 if col == "Test_AUC" else 3)
        if len(at_chosen):
            ring(ax, chosen, float(at_chosen[col].iloc[0]), shade, size=6.4)

    mark_choice(ax, chosen, colour)
    if show_title:
        ax.set_title(display_label(cancer), color=colour, pad=3)
    ax.set_xlim(0.3, 18.7)
    ax.set_ylim(-0.03, 1.06)
    colour_choice_tick(ax, [1, 5, 10, 15, 18], chosen, colour, keep_gap=2)
    ax.grid(axis="y", linewidth=0.35, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    if show_xlabel:
        ax.set_xlabel("Panel size (markers)", fontsize=axis_label_size)
    if show_ylabel:
        ax.set_ylabel("Held-out value", fontsize=axis_label_size)
    if axis_label_size is not None and show_title:
        ax.title.set_fontsize(axis_label_size * 1.25)

    keep = d[(d["Test_AUC"] >= auc_best - AUC_TOL)
             & (d["balanced_accuracy"] >= d["balanced_accuracy"].max() - BAL_TOL)]
    return {"cancer": display_label(cancer), "deployed_size": chosen,
            "rule_size_seed52": int(keep["Features"].min()) if len(keep) else np.nan,
            "auc_best": round(float(auc_best), 4),
            "f1_at_deployed": round(float(at_chosen["f1"].iloc[0])
                                    if len(at_chosen) else np.nan, 4)}


def sweep_legend_handles():
    """Series key only. The 0.02 tolerance band is described in the caption:
    as a legend swatch it was an undifferentiated grey rectangle that carried
    no information about where the band actually sits in a tile."""
    handles = [
        plt.Line2D([], [], color="#333333", linewidth=lw + 0.4, linestyle=style,
                   marker="o" if col == "Test_AUC" else None, markersize=3,
                   label=label)
        for col, label, lw, style, _t in SINGLE_SERIES]
    handles.append(
        plt.Line2D([], [], color="#444444", linewidth=1.1,
                   linestyle=(0, (2.5, 1.6)), marker="o", markersize=6,
                   markerfacecolor="white", markeredgecolor="#444444",
                   label="Selected panel size"))
    return handles


def build_fig4_5() -> pd.DataFrame:
    # No sharex: every tile carries its own selected size as a coloured tick,
    # and a shared x axis would let the last tile's ticks overwrite the rest.
    fig, axes = plt.subplots(3, 4, figsize=(11.6, 7.8))
    rows = []
    for idx, (ax, cancer) in enumerate(zip(axes.ravel(), CANCER_ORDER)):
        rows.append(draw_single_sweep(
            ax, cancer, load_single(cancer.lower()),
            show_ylabel=(idx % 4 == 0), show_xlabel=(idx >= 8)))

    fig.legend(handles=sweep_legend_handles(), loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=2.2,
               fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.6, w_pad=1.4)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig4_5_disease_specific_panel_size_selection.{ext}",
                    dpi=DPI, facecolor="white", bbox_inches="tight")
    fig.savefig(OVERLEAF / "fig4_5_disease_specific_panel_size_selection.png",
                dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    multi = build_fig3_5()
    r25 = multi.loc[multi["feature_count"] == LOCKED_SIZE].iloc[0]
    print("fig3-5 + fig3-6b written. At 25 markers:")
    for col, _ci, label, *_ in METRICS:
        print(f"  {label:<20} {r25[col]:.4f}   best over 12-30 "
              f"{multi[col].max():.4f} at size "
              f"{int(multi.loc[multi[col].idxmax(), 'feature_count'])}")

    single = build_fig4_5()
    print("\nfig4-5 written.")
    print(single.to_string(index=False))
    single.to_csv(OUT / "fig4_5_source_panel_size_selection.csv", index=False)
