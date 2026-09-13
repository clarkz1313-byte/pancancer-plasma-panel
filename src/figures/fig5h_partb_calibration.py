#!/usr/bin/env python3
"""
fig5h_partb_calibration.py — internal CALIBRATION for the locked 25-protein
Part B multiclass panel.

WHY THIS EXISTS
---------------
Added 2026-08-31, from a manuscript review gap. The draft reports expected
calibration error, Brier score and reliability curves for **Part A** (the 12
cancer-specific panels, `fig7_panels.panel_calibration`) but reports **none**
of it for Part B — the locked panel, which is the paper's headline model.
For a prediction-model paper that asymmetry is conspicuous: a reviewer sees
calibration demanded of the supporting analysis and not of the main one.

Nothing new is fitted here. This reads the same locked test predictions that
every other Part B number comes from and reports how well their probabilities
behave.

WHAT IS COMPUTED
----------------
Multiclass calibration has no single definition, so three complementary
views are reported rather than one number:

  a  TOP-LABEL reliability. For each sample take the winning class and its
     probability, bin by that probability, and plot bin accuracy against bin
     mean confidence. This answers "when the model says 70%, is it right 70%
     of the time?" — the question that matters if a predicted cancer type is
     ever acted on. Summarised by top-label ECE and by the mean
     confidence-minus-accuracy gap, which carries a SIGN: positive is
     overconfident, negative is underconfident.

  b  CLASSWISE reliability, one-vs-rest, per cancer. A model can be well
     calibrated on its winning label while being badly calibrated for an
     individual cancer, which is exactly the failure mode that matters for
     the smallest classes. Summarised by classwise ECE per cancer.

  c  Per-class Brier score, decomposed into RELIABILITY and RESOLUTION
     (Murphy decomposition). Brier alone conflates "probabilities are
     miscalibrated" with "the task is hard"; the decomposition separates
     them, which is the honest way to report a 12-class problem where class
     sizes run from 9 to 77 in the test set.

BINNING. Equal-count (quantile) bins, not equal-width. With 413 samples an
equal-width top-label binning leaves several bins with 2-3 samples, whose
accuracy is 0 or 1 by construction and which then dominate a plotted curve.
Bin count is capped so no bin holds fewer than MIN_BIN_N samples.

Source: revise_plan/part_b_multiclass/vRSX_v11_locked_reproducer/tables/
        locked_test_predictions_with_probabilities.csv   (413 x 12 scores)
Output: figurev5/output/fig5h_partb_calibration.{png,pdf}
        figurev5/output/fig5h_partb_calibration_summary.csv
Run:    python fig5h_partb_calibration.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_labels import display_label

ROOT = Path("e:/Proteomics")
TABLES = ROOT / "revise_plan/part_b_multiclass/vRSX_v11_locked_reproducer/tables"
OUT = ROOT / "figurev5/output"
OUT.mkdir(parents=True, exist_ok=True)

CANCERS = ["AML", "BRC", "CLL", "CRC", "CVX", "ENDC",
           "GLIOM", "LUNGC", "LYMPH", "MYEL", "OVC", "PRC"]
CANCER_COLOR = {
    "AML": "#b22222", "BRC": "#c97b63", "CLL": "#7a3e9d", "CRC": "#d68600",
    "CVX": "#c13d86", "ENDC": "#8e5d2c", "GLIOM": "#1a8db8", "LUNGC": "#2e8b57",
    "LYMPH": "#3856a6", "MYEL": "#8c564b", "OVC": "#d1495b", "PRC": "#008b8b",
}
ACCENT = "#7c3fa0"
FG, MUTED = "#1a1a1a", "#555555"
MIN_BIN_N = 25

plt.rcParams.update({"font.family": "Arial", "font.size": 11,
                     "font.weight": "bold", "axes.labelweight": "bold",
                     "axes.titleweight": "bold"})


def quantile_bins(p: np.ndarray, n_bins: int):
    """Equal-COUNT bin edges. Returns edges with duplicates removed, so a
    heavily tied score vector silently yields fewer bins rather than empty
    ones."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(p, qs))
    edges[0], edges[-1] = edges[0] - 1e-9, edges[-1] + 1e-9
    return edges


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int):
    """Bin (confidence, hit) pairs. Returns per-bin mean confidence, observed
    frequency, count — plus ECE and the SIGNED mean gap."""
    edges = quantile_bins(p, n_bins)
    idx = np.digitize(p, edges) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    conf, freq, cnt = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        conf.append(p[m].mean())
        freq.append(y[m].mean())
        cnt.append(int(m.sum()))
    conf, freq, cnt = np.array(conf), np.array(freq), np.array(cnt)
    w = cnt / cnt.sum()
    ece = float(np.sum(w * np.abs(conf - freq)))
    gap = float(np.sum(w * (conf - freq)))   # signed: + = overconfident
    return conf, freq, cnt, ece, gap


def murphy(p: np.ndarray, y: np.ndarray, n_bins: int):
    """Brier = reliability - resolution + uncertainty."""
    edges = quantile_bins(p, n_bins)
    idx = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    base = y.mean()
    brier = float(np.mean((p - y) ** 2))
    rel = res = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        w = m.sum() / len(y)
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - base) ** 2
    return brier, float(rel), float(res), float(base * (1 - base))


def main() -> None:
    d = pd.read_csv(TABLES / "locked_test_predictions_with_probabilities.csv")
    score_cols = [f"score_{c}" for c in CANCERS]
    P = d[score_cols].to_numpy(dtype=float)
    truth = d["true_class"].to_numpy()

    # Rows should already be normalised; renormalise defensively so the
    # top-label confidence is a genuine probability rather than a raw score.
    P = P / P.sum(axis=1, keepdims=True)

    top_idx = P.argmax(axis=1)
    top_conf = P[np.arange(len(P)), top_idx]
    top_pred = np.array(CANCERS)[top_idx]
    top_hit = (top_pred == truth).astype(float)

    n_bins = max(3, min(10, len(d) // MIN_BIN_N))
    conf, freq, cnt, top_ece, top_gap = reliability(top_conf, top_hit, n_bins)

    # multiclass Brier (sum of squared error over the full probability vector)
    Y = np.zeros_like(P)
    for i, t in enumerate(truth):
        Y[i, CANCERS.index(t)] = 1.0
    mc_brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))

    rows = []
    for j, c in enumerate(CANCERS):
        y = Y[:, j]
        p = P[:, j]
        nb = max(3, min(8, int(y.sum()) // 3 or 3))
        _, _, _, ece_c, gap_c = reliability(p, y, nb)
        br, rel, res, unc = murphy(p, y, nb)
        rows.append(dict(cancer=display_label(c), n_true=int(y.sum()),
                         classwise_ece=ece_c, signed_gap=gap_c,
                         brier=br, reliability=rel, resolution=res,
                         uncertainty=unc))
    summ = pd.DataFrame(rows)

    # ── figure ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17.0, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.18, 1.18],
                          left=0.045, right=0.988, top=0.80, bottom=0.145,
                          wspace=0.28)

    # a — top-label reliability
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 1], [0, 1], ls="--", lw=1.8, color="#999999", zorder=1)
    ax.plot(conf, freq, "-o", color=ACCENT, lw=2.8, ms=9, mec="white",
            mew=1.6, zorder=4)
    for x, y_, n in zip(conf, freq, cnt):
        ax.annotate(f"n={n}", (x, y_), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=9, color=MUTED)
    lo = min(conf.min(), freq.min()) - 0.08
    ax.set_xlim(max(0, lo), 1.0)
    ax.set_ylim(max(0, lo), 1.0)
    ax.set_xlabel("mean predicted probability of the winning class", fontsize=11.5)
    ax.set_ylabel("observed proportion correct", fontsize=11.5)
    ax.set_title(f"Top-label reliability\nECE = {top_ece:.3f}  ·  "
                 f"mean gap = {top_gap:+.3f} "
                 f"({'over' if top_gap > 0 else 'under'}confident)",
                 fontsize=12.5, loc="left", pad=9)
    ax.grid(color="#eeeeee", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # b — classwise ECE, sorted
    ax = fig.add_subplot(gs[0, 1])
    s = summ.sort_values("classwise_ece")
    cols = [CANCER_COLOR[c] for c in
            [k for lbl in s["cancer"] for k in CANCERS
             if display_label(k) == lbl]]
    ax.barh(range(len(s)), s["classwise_ece"], color=cols, alpha=0.85,
            edgecolor=FG, lw=0.7)
    for i, (v, n) in enumerate(zip(s["classwise_ece"], s["n_true"])):
        ax.text(v + 0.0012, i, f"{v:.3f}  (n={n})", va="center",
                fontsize=9.5, color=FG)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s["cancer"], fontsize=10.5)
    for tick, lbl in zip(ax.get_yticklabels(), s["cancer"]):
        for k in CANCERS:
            if display_label(k) == lbl:
                tick.set_color(CANCER_COLOR[k])
    ax.set_xlim(0, float(s["classwise_ece"].max()) * 1.42)
    ax.set_xlabel("one-vs-rest expected calibration error", fontsize=11.5)
    ax.set_title("Classwise calibration\nsmaller is better  ·  n = test-set "
                 "cases of that cancer", fontsize=12.5, loc="left", pad=9)
    ax.grid(axis="x", color="#eeeeee", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # c — Brier decomposition
    ax = fig.add_subplot(gs[0, 2])
    s2 = summ.sort_values("brier")
    x = np.arange(len(s2))
    ax.bar(x, s2["reliability"], color="#c1666b", edgecolor=FG, lw=0.7,
           label="reliability (miscalibration — lower better)")
    ax.bar(x, s2["resolution"], bottom=s2["reliability"], color="#4a9fd4",
           edgecolor=FG, lw=0.7,
           label="resolution (discrimination — higher better)")
    ax.plot(x, s2["brier"], "o-", color=FG, lw=1.8, ms=7, label="Brier score")
    ax.set_xticks(x)
    ax.set_xticklabels(s2["cancer"], rotation=45, ha="right", fontsize=10)
    for tick, lbl in zip(ax.get_xticklabels(), s2["cancer"]):
        for k in CANCERS:
            if display_label(k) == lbl:
                tick.set_color(CANCER_COLOR[k])
    ax.set_ylabel("score component", fontsize=11.5)
    ax.set_title("Brier decomposition (Murphy)\nreliability is the "
                 "miscalibration part; resolution is the signal",
                 fontsize=12.5, loc="left", pad=9)
    ax.legend(fontsize=9.5, frameon=False, loc="upper left")
    ax.grid(axis="y", color="#eeeeee", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Internal calibration of the 25-protein multiclass panel | "
        f"{len(d)} held-out samples | multiclass Brier = {mc_brier:.3f} | "
        f"top-label ECE = {top_ece:.3f}",
        fontsize=15, fontweight="bold", x=0.045, ha="left", y=0.965)

    for ext in ("png", "pdf"):
        p = OUT / f"fig5h_partb_calibration.{ext}"
        fig.savefig(p, dpi=300, facecolor="white", bbox_inches="tight")
        print(f"  saved -> {p}")
    plt.close(fig)

    summ_out = OUT / "fig5h_partb_calibration_summary.csv"
    summ.to_csv(summ_out, index=False)
    print(f"  saved -> {summ_out}")

    print(f"\n  multiclass Brier      {mc_brier:.4f}")
    print(f"  top-label ECE         {top_ece:.4f}")
    print(f"  top-label signed gap  {top_gap:+.4f}"
          f"  ({'overconfident' if top_gap > 0 else 'underconfident'})")
    print(f"  top-label accuracy    {top_hit.mean():.4f}")
    print(f"  mean confidence       {top_conf.mean():.4f}")
    print(f"  classwise ECE  mean {summ['classwise_ece'].mean():.4f}  "
          f"range {summ['classwise_ece'].min():.4f}-{summ['classwise_ece'].max():.4f}")
    print("\n" + summ.to_string(index=False))


if __name__ == "__main__":
    print("Building fig5h (Part B internal calibration) ...")
    main()
    print("Done.")
