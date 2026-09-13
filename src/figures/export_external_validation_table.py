#!/usr/bin/env python3
"""Export the canonical external-validation results table.

WHY THIS EXISTS
---------------
Before 2026-08-17 the manuscript tables (`paper/tables/*.csv`,
`paper/supplementary_tables/S4/S5`) were generated separately from the
figures, and drifted:

  * LYMPH still carried the invalid pooled 114-case GSE32018 cohort that
    was retired in the painful-trio work (correct cohort: 22 DLBCL cases).
  * CVX still carried GSE63514 on the Part B side after GSE9750 was
    promoted.
  * LUNGC's main table row used GSE19804 while the figures use CPTAC.
  * Table values were mean-of-per-split metrics, while the figures
    threshold *participant-averaged* probabilities -- so the two disagreed
    by ~0.005-0.06 even where the cohort matched.

This script removes the drift by deriving the table from **exactly the same
objects the figures read** (`fig89_common.PARTA_PRED` / `PARTB_PRED`) using
**exactly the same aggregation** (`fig89_common.load_preds`, which averages
repeated out-of-fold probabilities per participant before thresholding).

If a cohort map changes in `fig89_common.py`, rerun this script; the tables
and figures then cannot disagree.

Thresholds are the live figure constants, not re-derived here:
  Part A (confirmation) = 0.80   Part B (screening) = 0.50

Usage:  python export_external_validation_table.py
Writes: paper/tables/external_validation_canonical_<date>.csv
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from pathlib import PurePath

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig89_common as F  # noqa: E402

PART_A_THR = 0.80   # must match fig8_panels.CONFIRM_THR
PART_B_THR = 0.50   # must match fig9_panels.SCREEN_THR

OUT = F.ROOT / "paper" / "tables" / f"external_validation_canonical_{date.today()}.csv"


def study_of(path: Path) -> str:
    parts = PurePath(path).parts
    return parts[parts.index("external_validation_roadmap") + 1]


def auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def coverage(pred_path: Path) -> tuple:
    try:
        c = pd.read_csv(pred_path.parent / "cohort_summary.csv")
        return int(c["mapped_panel_features"][0]), int(c["total_panel_features"][0])
    except Exception:
        return (np.nan, np.nan)


def perm_p(pred_path: Path):
    f = pred_path.parent / "locked25_leakage_diagnostic_summary.csv"
    if not f.exists():
        return np.nan
    return float(pd.read_csv(f).iloc[0]["empirical_p_value_ge_observed"])


def build() -> pd.DataFrame:
    rows = []
    for part, pred_map, thr in (("A", F.PARTA_PRED, PART_A_THR),
                                ("B", F.PARTB_PRED, PART_B_THR)):
        for cancer in F.CANCER_ORDER:
            path = pred_map[cancer]
            d = F.load_preds(path)
            y = d["true_label"].to_numpy()
            s = d["prob"].to_numpy()
            pred = (s >= thr).astype(int)

            tp = int(((y == 1) & (pred == 1)).sum())
            fn = int(((y == 1) & (pred == 0)).sum())
            fp = int(((y == 0) & (pred == 1)).sum())
            tn = int(((y == 0) & (pred == 0)).sum())

            sens, sens_lo, sens_hi = F._wilson_ci(tp, tp + fn)
            spec, spec_lo, spec_hi = F._wilson_ci(tn, tn + fp)
            used, total = coverage(path)

            rows.append({
                "part": part,
                "stage": "confirmation" if part == "A" else "screening",
                "panel": "single-class" if part == "A" else "locked-25 multiclass",
                "endpoint": F.TILE_LABEL.get(cancer, cancer),
                "internal_code": cancer,
                "study": study_of(path),
                "n_total": len(y), "n_case": int((y == 1).sum()), "n_control": int((y == 0).sum()),
                "panel_features_used": used, "panel_features_total": total,
                "threshold": thr,
                "auc": round(auc(y, s), 4),
                "sensitivity": round(sens, 4),
                "sensitivity_wilson_lo": round(sens_lo, 4),
                "sensitivity_wilson_hi": round(sens_hi, 4),
                "specificity": round(spec, 4),
                "specificity_wilson_lo": round(spec_lo, 4),
                "specificity_wilson_hi": round(spec_hi, 4),
                "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                "permutation_p": perm_p(path),
                "train_samples_per_feature": (round((len(y) * 0.75) / used, 2)
                                              if used and not np.isnan(used) else np.nan),
                "prediction_source": str(path.relative_to(F.ROOT)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    pd.set_option("display.width", 250)
    show = ["part", "endpoint", "study", "n_case", "n_control",
            "panel_features_used", "panel_features_total",
            "threshold", "auc", "sensitivity", "specificity"]
    print(df[show].to_string(index=False))
    print()
    for p in ("A", "B"):
        s = df[df.part == p]
        print(f"Part {p} macro: AUC={s.auc.mean():.3f} "
              f"sens={s.sensitivity.mean():.3f} spec={s.specificity.mean():.3f}")
    print(f"\nwrote -> {OUT}")


if __name__ == "__main__":
    main()
