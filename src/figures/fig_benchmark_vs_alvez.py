#!/usr/bin/env python3
"""
fig_benchmark_vs_alvez.py — the locked 25-protein panel against the source
study's own panel-size sweep.

WHY THIS EXISTS
---------------
Added 2026-08-31, from a manuscript review gap. The draft never compares its
performance to ANY published multi-cancer proteomic signature — and a Q1
reviewer will ask in the first round. The obvious comparator was already in
the repository: Álvez et al., *Next generation pan-cancer blood proteome
profiling using proximity extension assay*, Nat Commun 14:4308 (2023),
`references/alves_source_study/alvez_multiclasss_perf.csv`.

That comparator is unusually clean, for one reason and with one caveat.

  THE REASON: it is the SAME COHORT. Our discovery data IS the Álvez
  dataset — same 1,375 plasma samples, same 1,463 proteins, same 12 cancer
  classes. So the comparison is not the usual apples-to-oranges cross-study
  guess; the underlying measurements are identical.

  THE CAVEAT: it is NOT a controlled head-to-head. Álvez used their own
  train/test splits and their own modelling; we use a single seed-52
  70/30 split and class-BALANCED L2 logistic regression. Any difference
  therefore confounds *panel selection* with *model choice* — in particular
  our class weighting is expected to help exactly the metrics (macro F1,
  minimum class recall) where the gap is largest. This figure is evidence
  that a compact panel is not obviously worse; it is NOT evidence that our
  selection procedure beats theirs.

WHAT THE COMPARISON SHOWS
-------------------------
Álvez report multiclass performance at four panel sizes (12, 36, 83, and
all 1,463). Plotting our single 25-protein point against that curve is the
most direct answer available to "what does a compact panel cost?".

Metrics are macro-averaged over the 12 classes in both cases. Note that
Álvez's per-class "Accuracy" column is one-vs-rest accuracy, not multiclass
accuracy, so it is deliberately NOT compared here — only AUC, F1 and recall,
which are computed the same way on both sides.

Source: references/alves_source_study/alvez_multiclasss_perf.csv
        revise_plan/.../tables/bootstrap_ci_summary.csv (ours, for the CI)
        revise_plan/.../tables/locked_per_class_recall_auc.csv
Output: figurev5/output/fig_benchmark_vs_alvez.{png,pdf}
        figurev5/output/fig_benchmark_vs_alvez_table.csv
Run:    python fig_benchmark_vs_alvez.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("e:/Proteomics")
REF = ROOT / "references/alves_source_study/alvez_multiclasss_perf.csv"
TABLES = ROOT / "revise_plan/part_b_multiclass/vRSX_v11_locked_reproducer/tables"
OUT = ROOT / "figurev5/output"
OUT.mkdir(parents=True, exist_ok=True)

REF_COL = "#5d6b78"
OURS_COL = "#7c3fa0"
FG, MUTED = "#1a1a1a", "#555555"

plt.rcParams.update({"font.family": "Arial", "font.size": 11,
                     "font.weight": "bold", "axes.labelweight": "bold",
                     "axes.titleweight": "bold"})

METRICS = [
    ("AUC",    "Macro one-vs-rest AUC",   "macro_auc"),
    ("F1",     "Macro F1",                "macro_f1"),
    ("Recall", "Macro recall\n(= balanced accuracy)", "macro_recall"),
]


def main() -> None:
    ref = pd.read_csv(REF)
    ref = ref[ref["Type"] == "Multiclass"]
    g = ref.groupby("Num_proteins").agg(
        macro_auc=("AUC", "mean"),
        macro_f1=("F1", "mean"),
        macro_recall=("Recall", "mean"),
        min_recall=("Recall", "min"),
    ).reset_index().sort_values("Num_proteins")

    # ours — read from the same tables every other Part B number comes from
    per = pd.read_csv(TABLES / "locked_per_class_recall_auc.csv")
    ours = {
        "n": 25,
        "macro_auc": float(per["ovr_auc"].mean()),
        "macro_f1": 0.7840,          # bootstrap_ci_summary.csv, macro F1
        "macro_recall": float(per["recall"].mean()),
        "min_recall": float(per["recall"].min()),
    }

    tab = g.copy()
    tab.insert(0, "study", "Alvez et al. 2023")
    ours_row = pd.DataFrame([{
        "study": "this study (locked panel)", "Num_proteins": 25,
        "macro_auc": ours["macro_auc"], "macro_f1": ours["macro_f1"],
        "macro_recall": ours["macro_recall"], "min_recall": ours["min_recall"],
    }])
    tab = pd.concat([tab, ours_row], ignore_index=True)
    tab.to_csv(OUT / "fig_benchmark_vs_alvez_table.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(19.0, 5.3))
    fig.subplots_adjust(left=0.045, right=0.99, top=0.775, bottom=0.155,
                        wspace=0.26)

    panels = METRICS + [("min", "Minimum class recall\n(the weakest cancer)",
                         "min_recall")]

    for ax, (_, title, key) in zip(axes, panels):
        ax.plot(g["Num_proteins"], g[key], "-o", color=REF_COL, lw=2.6, ms=10,
                mec="white", mew=1.6, zorder=3, label="Álvez et al. 2023")
        ax.set_xscale("log")

        ax.axhline(ours[key], color=OURS_COL, ls=":", lw=1.6, zorder=2)
        ax.plot([25], [ours[key]], marker="*", ms=27, color=OURS_COL,
                mec="white", mew=1.6, zorder=6,
                label="this study — 25 proteins")
        # placed LEFT of the star: at x=25 the next reference point (36) is
        # close enough on a log axis that a right-hand label overprints its
        # value annotation
        ax.annotate(f"{ours[key]:.3f}", (25, ours[key]),
                    textcoords="offset points", xytext=(-13, -24),
                    ha="right", fontsize=12, fontweight="bold", color=OURS_COL)

        for x, y in zip(g["Num_proteins"], g[key]):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=9.5,
                        color=MUTED)

        ax.set_xticks([12, 25, 36, 83, 1463])
        ax.set_xticklabels(["12", "25", "36", "83", "1,463"], fontsize=10.5)
        ax.minorticks_off()
        ax.set_xlabel("proteins in panel (log scale)", fontsize=11.5)
        ax.set_title(title, fontsize=12.5, loc="left", pad=9)
        ax.grid(color="#eeeeee", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        lo = min(g[key].min(), ours[key])
        hi = max(g[key].max(), ours[key])
        pad = (hi - lo) * 0.22 + 0.01
        ax.set_ylim(lo - pad, hi + pad)

    axes[0].legend(fontsize=10.5, frameon=False, loc="lower right")

    fig.suptitle(
        "Compact panel versus the source study's own panel-size sweep  ·  "
        "same cohort, same 12 cancers, same assay  ·  "
        "different splits and different models — not a controlled benchmark",
        fontsize=14.5, fontweight="bold", x=0.045, ha="left", y=0.955)

    for ext in ("png", "pdf"):
        p = OUT / f"fig_benchmark_vs_alvez.{ext}"
        fig.savefig(p, dpi=300, facecolor="white", bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)

    print("\n" + tab.to_string(index=False))
    print("\nReading:")
    for key, label in [("macro_auc", "macro AUC"), ("macro_f1", "macro F1"),
                       ("macro_recall", "macro recall"),
                       ("min_recall", "min class recall")]:
        beaten = g[g[key] < ours[key]]["Num_proteins"].tolist()
        print(f"  {label:18s} ours {ours[key]:.3f}  "
              f"exceeds Alvez at sizes {beaten}")


if __name__ == "__main__":
    print("Building benchmark vs Alvez et al. 2023 ...")
    main()
    print("Done.")
