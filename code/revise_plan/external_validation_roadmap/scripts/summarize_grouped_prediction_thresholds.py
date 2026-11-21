#!/usr/bin/env python3
"""Summarize repeated grouped predictions at fixed thresholds.

Repeated out-of-fold probabilities are averaged within participant before
computing AUC, confusion counts, and Wilson confidence intervals. This avoids
treating repeated predictions from the same participant as independent data.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)) / denominator
    return estimate, max(0.0, center - half_width), min(1.0, center + half_width)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--thresholds", default="0.50,0.80")
    args = parser.parse_args()

    raw = pd.read_csv(args.predictions)
    required = {"sample_id", "true_label", "predicted_probability_tumor"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    per_sample = (
        raw.groupby("sample_id", as_index=False)
        .agg(
            true_label=("true_label", "first"),
            predicted_probability_tumor=("predicted_probability_tumor", "mean"),
            n_oof_predictions=("predicted_probability_tumor", "size"),
        )
    )
    y = per_sample["true_label"].to_numpy(dtype=int)
    score = per_sample["predicted_probability_tumor"].to_numpy(dtype=float)
    auc = auc_score(y, score)

    rows = []
    for threshold in [float(value.strip()) for value in args.thresholds.split(",") if value.strip()]:
        predicted = (score >= threshold).astype(int)
        tp = int(((y == 1) & (predicted == 1)).sum())
        fn = int(((y == 1) & (predicted == 0)).sum())
        tn = int(((y == 0) & (predicted == 0)).sum())
        fp = int(((y == 0) & (predicted == 1)).sum())
        sensitivity, sensitivity_low, sensitivity_high = wilson_interval(tp, tp + fn)
        specificity, specificity_low, specificity_high = wilson_interval(tn, tn + fp)
        rows.append(
            {
                "cohort": args.cohort,
                "threshold": threshold,
                "auc_per_sample_mean_probability": auc,
                "n_samples": len(per_sample),
                "n_positive": int((y == 1).sum()),
                "n_negative": int((y == 0).sum()),
                "tp": tp,
                "fn": fn,
                "tn": tn,
                "fp": fp,
                "sensitivity": sensitivity,
                "sensitivity_ci95_low": sensitivity_low,
                "sensitivity_ci95_high": sensitivity_high,
                "specificity": specificity,
                "specificity_ci95_low": specificity_low,
                "specificity_ci95_high": specificity_high,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    per_sample.to_csv(args.output.with_name(f"{args.output.stem}_per_sample.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
