#!/usr/bin/env python3
"""Generic locked-panel validation for external sample-level matrices —
scikit-learn variant.

This is a MINIMAL-DIFF fork of `generic_locked_panel_external_validation.py`
(the production script every canonical external row was generated with).
Every line of preprocessing, splitting, thresholding, and output-file schema
is unchanged; the only edits are:

  1. `fit_logistic()` (hand-written L2 gradient descent) is replaced by
     `sklearn.linear_model.LogisticRegression(C=1.0, solver="lbfgs",
     max_iter=5000, penalty="l2")` — the same estimator family and
     regularisation strength the INTERNAL pipeline uses (`LR-L2 C=1.0
     lbfgs`), so the internal and external arms now share one estimator.
  2. `auc_score()` (hand-rolled Mann-Whitney statistic) is replaced by
     `sklearn.metrics.roc_auc_score`.

Nothing else changed: same CLI, same grouped-split protocol (50 repeats,
75/25, participant-grouped), same median-impute + z-standardise fitted on
the train fold only, same 0.50/0.80 thresholding, same output file names
and columns. Output is written to a SEPARATE `--out-dir` so it never
overwrites the production run's canonical files.

Why this script exists
-----------------------
Not because the hand-written estimator was found to be wrong — it wasn't.
`estimator_sensitivity_sklearn_vs_handrolled.py` and
`estimator_sensitivity_component_metrics.py` ran both estimators on
IDENTICAL splits for all 24 canonical rows and found:

    macro AUC        Part A -0.0000, Part B -0.0010
    sensitivity       mean |delta| 0.003, max 0.024 (A/AML)
    specificity        mean |delta| 0.005, max 0.044 (B/GLIOM)
    cohort rankings   weakest/strongest unchanged in both parts
    coefficient cosine similarity  0.9969

Every specificity swing traces to exactly one flipped participant at the
fixed threshold (verified against each row's n_control) -- discreteness
noise, not a systematic shift. See
`estimator_sensitivity_component_metrics.csv` for the full per-row audit.

This script is kept as a validated drop-in alternative: if a reviewer
specifically wants the external refit to use a standard library estimator,
this can regenerate any cohort's row with one command and the same CLI as
production, and the result is known in advance to agree with the reported
numbers to within noise.

Input requirements (unchanged from production):
- matrix CSV/TSV with one row per sample.
- metadata CSV/TSV with sample_id and binary label columns.
- feature columns named by gene/protein symbols.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PANEL = ROOT / "revise_plan/external_validation_roadmap/tables/pathway_input_part_b_25_panel.csv"


def read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=sep)


def load_panel(path: Path) -> list[str]:
    panel = pd.read_csv(path)
    col = "gene_symbol" if "gene_symbol" in panel.columns else "protein"
    return [str(x).strip().upper() for x in panel[col].dropna() if str(x).strip()]


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    """scikit-learn AUC. Replaces the hand-rolled Mann-Whitney statistic.
    (Verified elsewhere to agree with it to machine precision on real
    scores; this is a swap of implementation, not of definition.)"""
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def impute_standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # unchanged from production -- isolates the comparison to the estimator only
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x = x.copy()
    missing = np.where(~np.isfinite(x))
    x[missing] = np.take(med, missing[1])
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return (x - mean) / std, med, mean, std


def fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """scikit-learn LogisticRegression, matching the internal pipeline's
    configuration (LR-L2 C=1.0 lbfgs). Replaces the hand-written gradient
    descent. Returns (coef, intercept) in the same shape the caller expects,
    so downstream scoring code is untouched."""
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, penalty="l2")
    clf.fit(x, y)
    return clf.coef_.ravel(), float(clf.intercept_[0])


def grouped_splits(y: np.ndarray, groups: np.ndarray, repeats: int, test_fraction: float, seed: int):
    # unchanged from production
    rng = np.random.default_rng(seed)
    unique_groups = np.array(list(dict.fromkeys(groups)))
    indices = np.arange(len(y))
    for _ in range(repeats):
        for _attempt in range(500):
            shuffled = unique_groups.copy()
            rng.shuffle(shuffled)
            n_test = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * test_fraction))))
            test_groups = set(shuffled[:n_test])
            test = indices[np.array([group in test_groups for group in groups])]
            train = indices[np.array([group not in test_groups for group in groups])]
            if len(np.unique(y[test])) == 2 and len(np.unique(y[train])) == 2:
                yield train, test
                break
        else:
            raise ValueError("Could not create a valid grouped split")


def run_model(
    matrix: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    seed: int,
    repeats: int,
    prefix: str | None = "locked25_grouped",
) -> pd.DataFrame:
    # unchanged from production except coef,intercept now come from sklearn
    y = matrix["label"].to_numpy(dtype=int)
    groups = matrix["group"].astype(str).to_numpy()
    x_raw = matrix[feature_cols].to_numpy(dtype=float)
    rows = []
    preds = []
    for split_id, (train_idx, test_idx) in enumerate(grouped_splits(y, groups, repeats, 0.25, seed), start=1):
        x_train, train_med, mean, std = impute_standardize(x_raw[train_idx])
        x_test = x_raw[test_idx].copy()
        missing = np.where(~np.isfinite(x_test))
        x_test[missing] = np.take(train_med, missing[1])
        x_test = (x_test - mean) / std
        coef, intercept = fit_logistic(x_train, y[train_idx])
        score = 1.0 / (1.0 + np.exp(-np.clip(x_test @ coef + intercept, -35, 35)))
        pred = (score >= 0.5).astype(int)
        yt = y[test_idx]
        sens = float(((pred == 1) & (yt == 1)).sum() / max(1, (yt == 1).sum()))
        spec = float(((pred == 0) & (yt == 0)).sum() / max(1, (yt == 0).sum()))
        rows.append(
            {
                "split_id": split_id,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "auc": auc_score(yt, score),
                "accuracy": float((pred == yt).mean()),
                "sensitivity": sens,
                "specificity": spec,
                "balanced_accuracy": (sens + spec) / 2,
            }
        )
        for sample, truth, s, p in zip(matrix.iloc[test_idx]["sample_id"], yt, score, pred):
            preds.append({"split_id": split_id, "sample_id": sample, "true_label": truth, "predicted_probability_tumor": s, "predicted_label": p})
    metrics = pd.DataFrame(rows)
    if prefix:
        metrics.to_csv(out_dir / f"{prefix}_metrics.csv", index=False)
        metrics.select_dtypes(include=[np.number]).agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"}).to_csv(
            out_dir / f"{prefix}_metric_summary.csv", index=False
        )
        pd.DataFrame(preds).to_csv(out_dir / f"{prefix}_predictions.csv", index=False)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--sample-column", default="sample_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--positive-values", default="tumor,cancer,case,1")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--min-observed-fraction", type=float, default=0.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel(args.panel)
    matrix = read_table(args.matrix)
    metadata = read_table(args.metadata)

    positives = {x.strip().lower() for x in args.positive_values.split(",")}
    metadata = metadata.copy()
    metadata["sample_id"] = metadata[args.sample_column].astype(str)
    metadata["label"] = metadata[args.label_column].astype(str).str.lower().isin(positives).astype(int)
    metadata["group"] = metadata[args.group_column].astype(str) if args.group_column else metadata["sample_id"]

    matrix = matrix.copy()
    matrix["sample_id"] = matrix[args.sample_column].astype(str) if args.sample_column in matrix.columns else matrix.iloc[:, 0].astype(str)
    matrix = matrix.drop(columns=[col for col in ("label", "group", args.label_column, args.group_column) if col and col in matrix.columns and col != "sample_id"])
    merged = metadata[["sample_id", "label", "group"]].merge(matrix, on="sample_id", how="inner")

    upper_to_col = {str(col).strip().upper(): col for col in merged.columns}
    present = [symbol for symbol in panel if symbol in upper_to_col]
    audit_rows = []
    for symbol in panel:
        source_column = upper_to_col.get(symbol, "")
        observed_fraction = float(pd.to_numeric(merged[source_column], errors="coerce").notna().mean()) if source_column else 0.0
        audit_rows.append(
            {
                "panel_symbol": symbol,
                "mapping_status": "present" if symbol in present else "missing",
                "source_column": source_column,
                "observed_fraction": observed_fraction,
                "model_included": bool(source_column and observed_fraction >= args.min_observed_fraction),
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.out_dir / "locked25_marker_coverage_audit.csv", index=False)
    included_symbols = audit.loc[audit["model_included"], "panel_symbol"].tolist()
    feature_cols = [upper_to_col[symbol] for symbol in included_symbols]
    if len(feature_cols) < 2:
        raise ValueError(f"Only {len(feature_cols)} locked-panel features are present; need at least two")

    kept = merged[["sample_id", "label", "group"] + feature_cols].copy()
    kept.to_csv(args.out_dir / "locked25_feature_matrix.csv", index=False)
    pd.DataFrame(
        [
            {
                "n_samples": len(kept),
                "n_tumor": int((kept["label"] == 1).sum()),
                "n_control": int((kept["label"] == 0).sum()),
                "mapped_panel_features": len(feature_cols),
                "mapped_any_observed_panel_features": len(present),
                "total_panel_features": len(panel),
                "min_observed_fraction": args.min_observed_fraction,
            }
        ]
    ).to_csv(args.out_dir / "cohort_summary.csv", index=False)

    y = kept["label"].to_numpy(dtype=int)
    univariate = []
    for symbol, col in zip(included_symbols, feature_cols):
        values = pd.to_numeric(kept[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        raw_auc = auc_score(y[finite], values[finite])
        univariate.append({"panel_symbol": symbol, "n_unique_values": int(pd.Series(values[finite]).nunique()), "raw_auc_tumor_high": raw_auc, "best_direction_auc": max(raw_auc, 1 - raw_auc)})
    pd.DataFrame(univariate).to_csv(args.out_dir / "locked25_univariate_marker_auc.csv", index=False)

    observed = run_model(kept, feature_cols, args.out_dir, args.seed, args.repeats, prefix="locked25_grouped")["auc"].mean()
    rng = np.random.default_rng(args.seed)
    permutation_rows = []
    for i in range(args.permutations):
        shuffled = kept.copy()
        shuffled["label"] = rng.permutation(y)
        permutation_rows.append({"permutation_id": i + 1, "auc_mean": run_model(shuffled, feature_cols, args.out_dir, args.seed + i + 1, 5, prefix=None)["auc"].mean()})
    permutations = pd.DataFrame(permutation_rows)
    permutations.to_csv(args.out_dir / "locked25_permutation_auc.csv", index=False)
    pd.DataFrame(
        [
            {
                "observed_auc_mean": observed,
                "permutation_auc_mean": float(permutations["auc_mean"].mean()),
                "permutation_auc_p95": float(permutations["auc_mean"].quantile(0.95)),
                "empirical_p_value_ge_observed": float(((permutations["auc_mean"] >= observed).sum() + 1) / (len(permutations) + 1)),
            }
        ]
    ).to_csv(args.out_dir / "locked25_leakage_diagnostic_summary.csv", index=False)
    print(pd.read_csv(args.out_dir / "cohort_summary.csv").to_string(index=False))
    print(pd.read_csv(args.out_dir / "locked25_grouped_metric_summary.csv").to_string(index=False))
    print(pd.read_csv(args.out_dir / "locked25_leakage_diagnostic_summary.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
