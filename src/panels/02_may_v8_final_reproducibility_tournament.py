#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
REVISE_ROOT = ROOT / "revise_plan"
PART_B_ROOT = REVISE_ROOT / "part_b_multiclass"

_V2_PATH = SCRIPT_DIR / "27_apr_multiclass_compact_tournament_v2.py"
_V2_SPEC = importlib.util.spec_from_file_location("compact_tournament_v2", _V2_PATH)
v2 = importlib.util.module_from_spec(_V2_SPEC)
assert _V2_SPEC.loader is not None
_V2_SPEC.loader.exec_module(v2)

v1 = v2.v1
base = v2.base

DEFAULT_OUTPUT = PART_B_ROOT / "final_v8_reproducibility_tournament"
DEFAULT_SEARCH_BUDGETS = (24, 25, 26, 27, 28)
DEFAULT_FOCUS_SIZES = (25, 26)
DEFAULT_SEEDS = (42, 52, 62, 72, 82)
RESCUE_CLASSES = ("BRC", "CVX", "ENDC")
PAIRWISE_PAIRS = (
    frozenset(("BRC", "CVX")),
    frozenset(("BRC", "ENDC")),
    frozenset(("CVX", "ENDC")),
    frozenset(("CRC", "LUNGC")),
)
SEARCH_RANKERS = (
    "CONSENSUS_SOFT_VOTE",
    "PAIRWISE_STABILITY",
    "PAIRWISE_MINMAX",
    "OVR_STABILITY",
    "OVR_MINMAX",
    "COVERAGE_IMPORTANCE",
    "OVR_BONF_QUOTA",
    "LOGISTIC_COEF",
    "UNIVAR_OVR_AUC",
    "RF_IMPORTANCE",
)
FINAL_COACHES = ("SOFT_VOTE", "LR_L2", "LR_EN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V8 reproducibility tournament for compact 12-class cancer panels. "
            "Every seed starts from raw processed data, train-only imputation, train-only OVR DEA, "
            "train-only panel selection, then one held-out test audit."
        )
    )
    parser.add_argument("--input", type=Path, default=v1.SOURCE_INPUT)
    parser.add_argument("--class-column", default="Cancer")
    parser.add_argument("--sample-id-column", default="Sample_ID")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--imputer", choices=sorted(base.IMPUTER_METHODS), default="knn")
    parser.add_argument("--knn-neighbors", type=int, default=base.KNN_NEIGHBORS)
    parser.add_argument("--per-cancer-top-m", type=int, default=14)
    parser.add_argument("--min-cancer-coverage", type=int, default=2)
    parser.add_argument("--search-budgets", nargs="+", type=int, default=list(DEFAULT_SEARCH_BUDGETS))
    parser.add_argument("--focus-sizes", nargs="+", type=int, default=list(DEFAULT_FOCUS_SIZES))
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--candidate-bank-size", type=int, default=160)
    parser.add_argument("--random-panels-per-size", type=int, default=260)
    parser.add_argument("--max-panels-per-seed", type=int, default=850)
    parser.add_argument("--final-coaches", nargs="+", default=list(FINAL_COACHES), choices=["SOFT_VOTE", "LR_L2", "LR_EN", "RF"])
    parser.add_argument("--primary-coach", default="SOFT_VOTE", choices=["SOFT_VOTE", "LR_L2", "LR_EN", "RF"])
    parser.add_argument("--target-features", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def output_paths(output_root: Path) -> dict[str, Path]:
    return {
        "root": output_root,
        "tables": output_root / "tables",
        "figures": output_root / "figures",
        "logs": output_root / "logs",
        "manifests": output_root / "manifests",
    }


def ensure_dirs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def log_message(paths: dict[str, Path], message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with (paths["logs"] / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def append_table(rows: list[dict[str, object]], path: Path) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def split_features(value: Any) -> list[str]:
    if isinstance(value, str):
        return [feature for feature in value.split(";") if feature]
    return []


def panel_signature(features: list[str]) -> str:
    return ";".join(features)


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def normalize(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if math.isclose(min_value, max_value):
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return (numeric - min_value) / (max_value - min_value)


def collect_train_only_de_sources(
    train_df: pd.DataFrame,
    protein_columns: list[str],
    class_column: str,
    classes: list[str],
    per_cancer_top_m: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bonf_rows: list[dict[str, object]] = []
    quota_rows: list[dict[str, object]] = []
    all_de_rows: list[pd.DataFrame] = []

    for target_class in classes:
        binary_train = train_df.copy()
        binary_train["Label"] = (binary_train[class_column] == target_class).astype(int)
        de_df = base.differential_expression(
            train_df=binary_train,
            protein_columns=protein_columns,
            case_mean_column=f"{target_class.lower()}_mean",
            case_n_column=f"{target_class.lower()}_n",
            control_n_column="rest_n",
        )
        de_df.insert(0, "target_class", target_class)
        all_de_rows.append(de_df)

        bonf_df = de_df.loc[
            de_df["significant_bonf"],
            ["target_class", "protein", "p_value", "p_adjusted_bonf", "fold_change"],
        ].copy()
        bonf_df = bonf_df.sort_values(["p_adjusted_bonf", "p_value", "protein"]).reset_index(drop=True)
        for rank, row in enumerate(bonf_df.itertuples(index=False), start=1):
            bonf_rows.append(
                {
                    "target_class": row.target_class,
                    "protein": row.protein,
                    "source_rank": rank,
                    "p_value": row.p_value,
                    "p_adjusted_bonf": row.p_adjusted_bonf,
                    "fold_change": row.fold_change,
                    "selection_rule": "bonf_significant",
                }
            )

        quota_df = bonf_df.head(min(per_cancer_top_m, len(bonf_df))).copy()
        selection_rule = "quota_bonf_top_m"
        if quota_df.empty:
            quota_df = de_df[["target_class", "protein", "p_value", "p_adjusted_bonf", "fold_change"]].copy()
            quota_df = quota_df.sort_values(["p_value", "protein"]).head(per_cancer_top_m).reset_index(drop=True)
            selection_rule = "quota_pvalue_fallback"
        for rank, row in enumerate(quota_df.itertuples(index=False), start=1):
            quota_rows.append(
                {
                    "target_class": row.target_class,
                    "protein": row.protein,
                    "source_rank": rank,
                    "p_value": row.p_value,
                    "p_adjusted_bonf": row.p_adjusted_bonf,
                    "fold_change": row.fold_change,
                    "selection_rule": selection_rule,
                }
            )

    return pd.DataFrame(bonf_rows), pd.DataFrame(quota_rows), pd.concat(all_de_rows, ignore_index=True)


def build_candidate_feature_table(
    bonf_df: pd.DataFrame,
    quota_df: pd.DataFrame,
    protein_columns: list[str],
    min_cancer_coverage: int,
    candidate_bank_size: int,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    available = set(protein_columns)
    coverage_df = (
        bonf_df.groupby("protein", as_index=False)
        .agg(
            cancer_count=("target_class", "nunique"),
            cancers=("target_class", lambda values: ";".join(sorted(set(values)))),
            best_bonf_rank=("source_rank", "min"),
            best_p_adjusted_bonf=("p_adjusted_bonf", "min"),
        )
        .sort_values(["cancer_count", "best_bonf_rank", "protein"], ascending=[False, True, True])
        .reset_index(drop=True)
        if not bonf_df.empty
        else pd.DataFrame(columns=["protein", "cancer_count", "cancers", "best_bonf_rank", "best_p_adjusted_bonf"])
    )
    quota_proteins = {protein for protein in quota_df["protein"].tolist() if protein in available}
    coverage_proteins = {
        protein
        for protein in coverage_df.loc[coverage_df["cancer_count"] >= min_cancer_coverage, "protein"].tolist()
        if protein in available
    }
    selected = sorted(quota_proteins | coverage_proteins)
    if len(selected) < 40 and not bonf_df.empty:
        extra = [protein for protein in bonf_df.sort_values(["p_adjusted_bonf", "p_value"])["protein"].tolist() if protein in available]
        selected = ordered_unique([*selected, *extra])[:max(40, len(selected))]

    candidate_df = coverage_df.loc[coverage_df["protein"].isin(selected)].copy()
    missing = sorted(set(selected) - set(candidate_df["protein"].tolist()))
    if missing:
        candidate_df = pd.concat(
            [
                candidate_df,
                pd.DataFrame(
                    {
                        "protein": missing,
                        "cancer_count": 0,
                        "cancers": "",
                        "best_bonf_rank": np.nan,
                        "best_p_adjusted_bonf": np.nan,
                    }
                ),
            ],
            ignore_index=True,
        )
    candidate_df["quota_count"] = candidate_df["protein"].map(quota_df.groupby("protein")["target_class"].nunique()).fillna(0).astype(int)
    candidate_df["candidate_score"] = (
        0.45 * normalize(candidate_df["cancer_count"])
        + 0.35 * normalize(candidate_df["quota_count"])
        + 0.20 * (1.0 / (1.0 + pd.to_numeric(candidate_df["best_bonf_rank"], errors="coerce").fillna(999.0)))
    )
    candidate_df = candidate_df.sort_values(
        ["candidate_score", "cancer_count", "quota_count", "protein"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    candidate_df["candidate_rank"] = np.arange(1, len(candidate_df) + 1)
    candidate_df = candidate_df.head(candidate_bank_size).copy()
    return candidate_df["protein"].tolist(), candidate_df, coverage_df


def ranking_priority(rankings: dict[str, list[str]], rank_table: pd.DataFrame, candidate_df: pd.DataFrame) -> dict[str, float]:
    rows: list[dict[str, object]] = []
    for ranker, proteins in rankings.items():
        weight = {
            "CONSENSUS_SOFT_VOTE": 1.25,
            "PAIRWISE_STABILITY": 1.15,
            "PAIRWISE_MINMAX": 1.10,
            "OVR_STABILITY": 1.05,
            "COVERAGE_IMPORTANCE": 1.00,
            "OVR_BONF_QUOTA": 0.95,
            "LOGISTIC_COEF": 0.95,
            "UNIVAR_OVR_AUC": 0.90,
            "RF_IMPORTANCE": 0.75,
        }.get(ranker, 0.8)
        for rank, protein in enumerate(proteins, start=1):
            rows.append({"protein": protein, "score": weight / (rank + 2.0)})
    score_df = pd.DataFrame(rows).groupby("protein", as_index=False)["score"].sum()
    merged = candidate_df[["protein", "candidate_score"]].merge(score_df, on="protein", how="outer").fillna(0.0)
    merged["priority_score"] = merged["score"] + 0.35 * merged["candidate_score"]
    merged = merged.sort_values(["priority_score", "protein"], ascending=[False, True]).reset_index(drop=True)
    return {row.protein: float(row.priority_score) for row in merged.itertuples(index=False)}


def quota_panel_order(quota_df: pd.DataFrame, classes: list[str], min_per_class: int) -> list[str]:
    if quota_df.empty or min_per_class <= 0:
        return []
    sorted_quota = quota_df.sort_values(["target_class", "source_rank", "p_adjusted_bonf", "p_value", "protein"])
    per_class = {
        class_name: sorted_quota.loc[sorted_quota["target_class"] == class_name, "protein"].tolist()
        for class_name in classes
    }
    output: list[str] = []
    for quota_index in range(min_per_class):
        for class_name in classes:
            values = per_class.get(class_name, [])
            if quota_index < len(values):
                output.append(values[quota_index])
    return ordered_unique(output)


def resize_panel(features: list[str], size: int, filler: list[str]) -> list[str]:
    selected = ordered_unique(features)
    if len(selected) > size:
        return selected[:size]
    for protein in filler:
        if protein not in selected:
            selected.append(protein)
        if len(selected) == size:
            break
    return selected


def panel_static_score(
    features: list[str],
    priority: dict[str, float],
    quota_df: pd.DataFrame,
    size: int,
) -> float:
    if not features:
        return -1.0
    feature_score = float(np.mean([priority.get(protein, 0.0) for protein in features]))
    cancers = set(quota_df.loc[quota_df["protein"].isin(features), "target_class"].tolist()) if not quota_df.empty else set()
    rescue_hits = len(cancers & set(RESCUE_CLASSES)) / len(RESCUE_CLASSES)
    coverage = len(cancers) / 12.0
    return feature_score + 0.20 * coverage + 0.20 * rescue_hits - 0.01 * abs(size - 25.5)


def generate_panel_candidates(
    rankings: dict[str, list[str]],
    quota_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    candidate_features: list[str],
    priority: dict[str, float],
    classes: list[str],
    budgets: list[int],
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    filler = sorted(candidate_features, key=lambda protein: (-priority.get(protein, 0.0), protein))
    shared = (
        coverage_df.loc[coverage_df["cancer_count"] >= args.min_cancer_coverage]
        .sort_values(["cancer_count", "best_bonf_rank", "protein"], ascending=[False, True, True])["protein"]
        .tolist()
        if not coverage_df.empty and "best_bonf_rank" in coverage_df.columns
        else []
    )

    def add_candidate(features: list[str], size: int, origin: str, ranker: str, min_per_class: int, n_shared: int) -> None:
        features = resize_panel(features, size, filler)
        if len(features) != size:
            return
        rows.append(
            {
                "origin": origin,
                "ranker": ranker,
                "feature_count": size,
                "min_per_class": min_per_class,
                "n_shared": n_shared,
                "features": panel_signature(features),
                "static_score": panel_static_score(features, priority, quota_df, size),
            }
        )

    for size in budgets:
        for ranker in SEARCH_RANKERS:
            ranker_features = rankings.get(ranker, filler)
            for min_per_class in (0, 1, 2):
                for n_shared in sorted({0, 4, 8, 12, min(size, 16), size}):
                    seed_features = [
                        *quota_panel_order(quota_df, classes, min_per_class),
                        *shared[:n_shared],
                        *ranker_features,
                    ]
                    add_candidate(seed_features, size, "structural", ranker, min_per_class, n_shared)

        top_seed = resize_panel(rankings.get("CONSENSUS_SOFT_VOTE", filler), size, filler)
        add_candidate(top_seed, size, "consensus_top", "CONSENSUS_SOFT_VOTE", 0, 0)
        add_candidate([*quota_panel_order(quota_df, classes, 1), *top_seed], size, "quota_consensus", "CONSENSUS_SOFT_VOTE", 1, 0)

        weights = np.asarray([max(priority.get(protein, 0.0), 0.001) for protein in filler], dtype=float)
        weights = weights / weights.sum()
        for _ in range(args.random_panels_per_size):
            quota_min = int(rng.choice([0, 1, 1, 2]))
            locked = quota_panel_order(quota_df, classes, quota_min)
            remaining = [protein for protein in filler if protein not in locked]
            remaining_weights = np.asarray([max(priority.get(protein, 0.0), 0.001) for protein in remaining], dtype=float)
            remaining_weights = remaining_weights / remaining_weights.sum()
            take_n = max(0, size - len(ordered_unique(locked)))
            if take_n > len(remaining):
                continue
            sampled = rng.choice(remaining, size=take_n, replace=False, p=remaining_weights).tolist()
            add_candidate([*locked, *sampled], size, "weighted_random", "CONSENSUS_SOFT_VOTE", quota_min, 0)

    candidate_df = pd.DataFrame(rows)
    if candidate_df.empty:
        return candidate_df
    candidate_df = candidate_df.drop_duplicates("features", keep="first")
    candidate_df = candidate_df.sort_values(["static_score", "feature_count"], ascending=[False, True]).reset_index(drop=True)
    candidate_df["candidate_id"] = np.arange(1, len(candidate_df) + 1)
    return candidate_df.head(args.max_panels_per_seed).copy()


def class_recalls(y_true: pd.Series, y_pred: np.ndarray, classes: list[str]) -> dict[str, float]:
    y_values = y_true.to_numpy()
    output = {}
    for class_name in classes:
        mask = y_values == class_name
        output[class_name] = float(np.mean(y_pred[mask] == class_name)) if np.any(mask) else math.nan
    return output


def pairwise_confusion_penalty(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_values = y_true.to_numpy()
    if len(y_values) == 0:
        return 0.0
    count = 0
    for true_label, pred_label in zip(y_values, y_pred):
        if true_label != pred_label and frozenset((str(true_label), str(pred_label))) in PAIRWISE_PAIRS:
            count += 1
    return float(count / len(y_values))


def stage2_objective(row: pd.Series, args: argparse.Namespace) -> float:
    macro_auc = 0.0 if math.isnan(row["cv_macro_ovr_auc_mean"]) else float(row["cv_macro_ovr_auc_mean"])
    min_recall = 0.0 if math.isnan(row["cv_min_class_recall_mean"]) else float(row["cv_min_class_recall_mean"])
    rescue_recall = 0.0 if math.isnan(row["cv_rescue_recall_mean"]) else float(row["cv_rescue_recall_mean"])
    pair_penalty = 0.0 if math.isnan(row["cv_pairwise_confusion_mean"]) else float(row["cv_pairwise_confusion_mean"])
    feature_penalty = max(0.0, (int(row["feature_count"]) - args.target_features) / max(1, max(args.search_budgets) - args.target_features))
    return float(
        0.31 * row["cv_balanced_accuracy_mean"]
        + 0.20 * row["cv_macro_f1_mean"]
        + 0.12 * macro_auc
        + 0.17 * min_recall
        + 0.16 * rescue_recall
        - 0.10 * pair_penalty
        - 0.10 * row["cv_balanced_accuracy_std"]
        - 0.035 * feature_penalty
    )


def cv_evaluate_panel(
    train_df: pd.DataFrame,
    features: list[str],
    classes: list[str],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, float]:
    x_train = train_df[features]
    y_train = train_df[args.class_column]
    splitter = v1.StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=seed)
    metric_rows: list[dict[str, float]] = []
    for fold_index, (fit_idx, val_idx) in enumerate(splitter.split(x_train, y_train), start=1):
        estimator = v2.make_estimator_v2("LR_L2", seed + fold_index, {"C": 0.1})
        x_fit = x_train.iloc[fit_idx]
        y_fit = y_train.iloc[fit_idx]
        x_val = x_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]
        estimator.fit(x_fit, y_fit)
        y_pred = estimator.predict(x_val)
        y_score = v1.aligned_probabilities(estimator, x_val, classes)
        metrics = v1.metric_bundle(y_val, y_pred, y_score, classes)
        recalls = class_recalls(y_val, np.asarray(y_pred), classes)
        metrics["rescue_recall"] = float(np.nanmean([recalls.get(label, math.nan) for label in RESCUE_CLASSES]))
        metrics["pairwise_confusion"] = pairwise_confusion_penalty(y_val, np.asarray(y_pred))
        metric_rows.append(metrics)
    metric_df = pd.DataFrame(metric_rows)
    summary = {f"cv_{column}_mean": float(metric_df[column].mean()) for column in metric_df.columns}
    summary.update({f"cv_{column}_std": float(metric_df[column].std(ddof=0)) for column in metric_df.columns})
    return summary


def run_stage2_for_seed(
    seed: int,
    candidate_df: pd.DataFrame,
    train_df: pd.DataFrame,
    classes: list[str],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> pd.DataFrame:
    checkpoint = paths["tables"] / f"stage2_dev_scores_seed_{seed}.csv"
    if checkpoint.exists() and args.resume:
        return pd.read_csv(checkpoint)
    if checkpoint.exists():
        checkpoint.unlink()

    rows: list[dict[str, object]] = []
    total = len(candidate_df)
    for index, candidate in enumerate(candidate_df.itertuples(index=False), start=1):
        features = split_features(candidate.features)
        metrics = cv_evaluate_panel(train_df, features, classes, args, seed + index)
        row = {
            "seed": seed,
            "candidate_id": int(candidate.candidate_id),
            "origin": candidate.origin,
            "ranker": candidate.ranker,
            "feature_count": int(candidate.feature_count),
            "min_per_class": int(candidate.min_per_class),
            "n_shared": int(candidate.n_shared),
            "features": candidate.features,
            "static_score": float(candidate.static_score),
            **metrics,
        }
        row["stage2_objective"] = stage2_objective(pd.Series(row), args)
        rows.append(row)
        if index % 25 == 0 or index == 1 or index == total:
            append_table(rows, checkpoint)
            rows = []
            log_message(paths, f"[seed {seed}] stage2 CV {index}/{total}")

    if rows:
        append_table(rows, checkpoint)
    df = pd.read_csv(checkpoint).drop_duplicates(["seed", "features"], keep="last")
    df = df.sort_values(["stage2_objective", "cv_balanced_accuracy_mean", "feature_count"], ascending=[False, False, True])
    write_table(df, checkpoint)
    return df


def evaluate_test_panel(
    seed: int,
    panel_row: pd.Series,
    coach: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    classes: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = split_features(panel_row["features"])
    estimator = v2.make_estimator_v2(coach, seed, {})
    estimator.fit(train_df[features], train_df[args.class_column])
    y_test = test_df[args.class_column]
    y_pred = estimator.predict(test_df[features])
    y_score = v1.aligned_probabilities(estimator, test_df[features], classes)
    metrics = v1.metric_bundle(y_test, y_pred, y_score, classes)
    recalls = class_recalls(y_test, np.asarray(y_pred), classes)
    row = {
        "seed": seed,
        "coach": coach,
        "primary_coach": coach == args.primary_coach,
        "feature_count": len(features),
        "features": panel_signature(features),
        "source_origin": panel_row["origin"],
        "source_ranker": panel_row["ranker"],
        "stage2_objective": panel_row["stage2_objective"],
        "cv_balanced_accuracy_mean": panel_row["cv_balanced_accuracy_mean"],
        "cv_macro_f1_mean": panel_row["cv_macro_f1_mean"],
        "cv_macro_ovr_auc_mean": panel_row["cv_macro_ovr_auc_mean"],
        "cv_min_class_recall_mean": panel_row["cv_min_class_recall_mean"],
        "cv_rescue_recall_mean": panel_row["cv_rescue_recall_mean"],
        "test_accuracy": metrics["accuracy"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_macro_ovr_auc": metrics["macro_ovr_auc"],
        "test_min_class_recall": metrics["min_class_recall"],
        "test_rescue_recall": float(np.nanmean([recalls.get(label, math.nan) for label in RESCUE_CLASSES])),
        "test_pairwise_confusion": pairwise_confusion_penalty(y_test, np.asarray(y_pred)),
    }

    report_df = pd.DataFrame(
        classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)
    ).transpose().reset_index().rename(columns={"index": "label"})
    report_df.insert(0, "seed", seed)
    report_df.insert(1, "coach", coach)
    report_df.insert(2, "feature_count", len(features))

    y_values = y_test.to_numpy()
    auc_rows = []
    for class_index, class_name in enumerate(classes):
        binary = (y_values == class_name).astype(int)
        auc = float(roc_auc_score(binary, y_score[:, class_index])) if len(np.unique(binary)) >= 2 else math.nan
        auc_rows.append(
            {
                "seed": seed,
                "coach": coach,
                "feature_count": len(features),
                "label": class_name,
                "recall": recalls.get(class_name, math.nan),
                "ovr_auc": auc,
            }
        )
    per_class_df = pd.DataFrame(auc_rows)

    confusion_df = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=classes), index=classes, columns=classes)
    confusion_long = confusion_df.reset_index().melt(id_vars="index", var_name="predicted_class", value_name="n")
    confusion_long = confusion_long.rename(columns={"index": "true_class"})
    confusion_long.insert(0, "seed", seed)
    confusion_long.insert(1, "coach", coach)
    confusion_long.insert(2, "feature_count", len(features))
    return row, report_df, per_class_df, confusion_long


def run_seed(seed: int, full_df: pd.DataFrame, protein_columns: list[str], classes: list[str], paths: dict[str, Path], args: argparse.Namespace) -> None:
    seed_dir = paths["tables"] / "seed_details"
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_message(paths, f"[seed {seed}] split + train-only KNN imputation")
    train_df, test_df = v1.split_and_impute(
        full_df=full_df,
        protein_columns=protein_columns,
        class_column=args.class_column,
        sample_id_column=args.sample_id_column,
        seed=seed,
        test_size=args.test_size,
        imputer_method=args.imputer,
        knn_neighbors=args.knn_neighbors,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    log_message(paths, f"[seed {seed}] train-only OVR DEA Bonferroni feature sources")
    bonf_df, quota_df, de_df = collect_train_only_de_sources(
        train_df=train_df,
        protein_columns=protein_columns,
        class_column=args.class_column,
        classes=classes,
        per_cancer_top_m=args.per_cancer_top_m,
    )
    bonf_df.to_csv(seed_dir / f"seed_{seed}_bonf_sources.csv", index=False)
    quota_df.to_csv(seed_dir / f"seed_{seed}_quota_sources.csv", index=False)
    de_df.to_csv(seed_dir / f"seed_{seed}_ovr_dea_all.csv", index=False)

    candidate_features, candidate_feature_df, coverage_df = build_candidate_feature_table(
        bonf_df=bonf_df,
        quota_df=quota_df,
        protein_columns=protein_columns,
        min_cancer_coverage=args.min_cancer_coverage,
        candidate_bank_size=args.candidate_bank_size,
    )
    candidate_feature_df.to_csv(seed_dir / f"seed_{seed}_candidate_feature_bank.csv", index=False)

    log_message(paths, f"[seed {seed}] rich train-only rankers on {len(candidate_features)} candidate proteins")
    rankings, ranking_table = v2.build_rankings_v2(
        x_train=train_df[candidate_features],
        y_train=train_df[args.class_column],
        classes=classes,
        quota_sources_df=quota_df,
        coverage_df=coverage_df,
        seed=seed,
    )
    ranking_table.to_csv(seed_dir / f"seed_{seed}_ranker_tables.csv", index=False)
    priority = ranking_priority(rankings, ranking_table, candidate_feature_df)

    rng = np.random.default_rng(seed)
    panel_candidates = generate_panel_candidates(
        rankings=rankings,
        quota_df=quota_df,
        coverage_df=coverage_df,
        candidate_features=candidate_features,
        priority=priority,
        classes=classes,
        budgets=args.search_budgets,
        rng=rng,
        args=args,
    )
    panel_candidates.to_csv(seed_dir / f"seed_{seed}_panel_candidates_prescreen.csv", index=False)
    log_message(paths, f"[seed {seed}] panel candidates after static prescreen: {len(panel_candidates)}")

    stage2_df = run_stage2_for_seed(seed, panel_candidates, train_df, classes, paths, args)
    chosen = (
        stage2_df.loc[stage2_df["feature_count"].isin(args.focus_sizes)]
        .sort_values(["feature_count", "stage2_objective", "cv_balanced_accuracy_mean"], ascending=[True, False, False])
        .drop_duplicates("feature_count", keep="first")
        .reset_index(drop=True)
    )
    chosen.to_csv(seed_dir / f"seed_{seed}_selected_panels_by_dev.csv", index=False)

    test_rows: list[dict[str, object]] = []
    report_rows: list[pd.DataFrame] = []
    per_class_rows: list[pd.DataFrame] = []
    confusion_rows: list[pd.DataFrame] = []
    feature_rows: list[dict[str, object]] = []
    for _, panel in chosen.iterrows():
        features = split_features(panel["features"])
        for rank, protein in enumerate(features, start=1):
            feature_rows.append(
                {
                    "seed": seed,
                    "feature_count": int(panel["feature_count"]),
                    "panel_rank": rank,
                    "protein": protein,
                    "features": panel["features"],
                }
            )
        for coach in args.final_coaches:
            row, report_df, per_class_df, confusion_df = evaluate_test_panel(
                seed=seed,
                panel_row=panel,
                coach=coach,
                train_df=train_df,
                test_df=test_df,
                classes=classes,
                args=args,
            )
            test_rows.append(row)
            report_rows.append(report_df)
            per_class_rows.append(per_class_df)
            confusion_rows.append(confusion_df)
            log_message(
                paths,
                (
                    f"[seed {seed}] test size={row['feature_count']} coach={coach} "
                    f"BA={row['test_balanced_accuracy']:.4f} F1={row['test_macro_f1']:.4f} "
                    f"minRecall={row['test_min_class_recall']:.4f}"
                ),
            )

    append_table(test_rows, paths["tables"] / "all_test_results.csv")
    append_table(feature_rows, paths["tables"] / "selected_features_by_seed.csv")
    if report_rows:
        pd.concat(report_rows, ignore_index=True).to_csv(
            paths["tables"] / "classification_reports_by_seed.csv",
            mode="a",
            header=not (paths["tables"] / "classification_reports_by_seed.csv").exists(),
            index=False,
        )
    if per_class_rows:
        pd.concat(per_class_rows, ignore_index=True).to_csv(
            paths["tables"] / "per_class_recall_auc_by_seed.csv",
            mode="a",
            header=not (paths["tables"] / "per_class_recall_auc_by_seed.csv").exists(),
            index=False,
        )
    if confusion_rows:
        pd.concat(confusion_rows, ignore_index=True).to_csv(
            paths["tables"] / "confusion_matrices_long_by_seed.csv",
            mode="a",
            header=not (paths["tables"] / "confusion_matrices_long_by_seed.csv").exists(),
            index=False,
        )


def build_summary_tables(paths: dict[str, Path], args: argparse.Namespace) -> None:
    test_path = paths["tables"] / "all_test_results.csv"
    if not test_path.exists():
        return
    test_df = pd.read_csv(test_path).drop_duplicates(["seed", "coach", "feature_count", "features"], keep="last")
    test_df.to_csv(test_path, index=False)
    primary_df = test_df.loc[test_df["coach"] == args.primary_coach].copy()

    summary = (
        primary_df.groupby("feature_count", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            test_balanced_accuracy_mean=("test_balanced_accuracy", "mean"),
            test_balanced_accuracy_std=("test_balanced_accuracy", "std"),
            test_macro_f1_mean=("test_macro_f1", "mean"),
            test_macro_f1_std=("test_macro_f1", "std"),
            test_macro_ovr_auc_mean=("test_macro_ovr_auc", "mean"),
            test_macro_ovr_auc_std=("test_macro_ovr_auc", "std"),
            test_min_class_recall_mean=("test_min_class_recall", "mean"),
            test_min_class_recall_std=("test_min_class_recall", "std"),
            test_rescue_recall_mean=("test_rescue_recall", "mean"),
            test_rescue_recall_std=("test_rescue_recall", "std"),
        )
        .sort_values("feature_count")
    )
    write_table(summary, paths["tables"] / "final_decision_summary_primary_coach.csv")

    all_coach_summary = (
        test_df.groupby(["feature_count", "coach"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            test_balanced_accuracy_mean=("test_balanced_accuracy", "mean"),
            test_macro_f1_mean=("test_macro_f1", "mean"),
            test_macro_ovr_auc_mean=("test_macro_ovr_auc", "mean"),
            test_min_class_recall_mean=("test_min_class_recall", "mean"),
            test_rescue_recall_mean=("test_rescue_recall", "mean"),
        )
        .sort_values(["feature_count", "test_balanced_accuracy_mean"], ascending=[True, False])
    )
    write_table(all_coach_summary, paths["tables"] / "coach_comparison_summary.csv")

    feature_path = paths["tables"] / "selected_features_by_seed.csv"
    if feature_path.exists():
        features_df = pd.read_csv(feature_path).drop_duplicates(["seed", "feature_count", "protein"])
        recurrence = (
            features_df.groupby(["feature_count", "protein"], as_index=False)
            .agg(seed_count=("seed", "nunique"), median_panel_rank=("panel_rank", "median"))
            .sort_values(["feature_count", "seed_count", "median_panel_rank", "protein"], ascending=[True, False, True, True])
        )
        recurrence["seed_frequency"] = recurrence["seed_count"] / max(1, features_df["seed"].nunique())
        write_table(recurrence, paths["tables"] / "feature_recurrence_by_size.csv")

    per_class_path = paths["tables"] / "per_class_recall_auc_by_seed.csv"
    if per_class_path.exists():
        per_class_df = pd.read_csv(per_class_path)
        primary_per_class = per_class_df.loc[per_class_df["coach"] == args.primary_coach]
        per_class_summary = (
            primary_per_class.groupby(["feature_count", "label"], as_index=False)
            .agg(
                recall_mean=("recall", "mean"),
                recall_std=("recall", "std"),
                ovr_auc_mean=("ovr_auc", "mean"),
                ovr_auc_std=("ovr_auc", "std"),
            )
            .sort_values(["feature_count", "label"])
        )
        write_table(per_class_summary, paths["tables"] / "per_class_summary_primary_coach.csv")


def save_figures(paths: dict[str, Path], args: argparse.Namespace) -> None:
    test_path = paths["tables"] / "all_test_results.csv"
    if not test_path.exists():
        return
    test_df = pd.read_csv(test_path)
    primary_df = test_df.loc[test_df["coach"] == args.primary_coach].copy()
    if not primary_df.empty:
        plot_df = primary_df.melt(
            id_vars=["seed", "feature_count"],
            value_vars=["test_balanced_accuracy", "test_macro_f1", "test_macro_ovr_auc", "test_min_class_recall", "test_rescue_recall"],
            var_name="metric",
            value_name="value",
        )
        figure, ax = base.plt.subplots(figsize=(9.5, 5.4))
        sns.lineplot(data=plot_df, x="feature_count", y="value", hue="metric", marker="o", errorbar="sd", ax=ax)
        ax.axhline(0.75, color="#9c2f2f", linestyle="--", linewidth=1)
        ax.set_title(f"V8 reproducibility tournament held-out test ({args.primary_coach})")
        ax.set_xlabel("Feature count")
        ax.set_ylabel("Metric")
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        figure.tight_layout()
        figure.savefig(paths["figures"] / "primary_test_metrics_25_vs_26.png", dpi=300, bbox_inches="tight")
        base.plt.close(figure)

    per_class_path = paths["tables"] / "per_class_summary_primary_coach.csv"
    if per_class_path.exists():
        per_class_df = pd.read_csv(per_class_path)
        for metric, filename in [("recall_mean", "per_class_recall_heatmap.png"), ("ovr_auc_mean", "per_class_ovr_auc_heatmap.png")]:
            pivot = per_class_df.pivot(index="label", columns="feature_count", values=metric)
            figure, ax = base.plt.subplots(figsize=(5.8, 7.2))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
            ax.set_title(f"V8 primary {metric.replace('_', ' ')}")
            ax.set_xlabel("Feature count")
            ax.set_ylabel("Cancer class")
            figure.tight_layout()
            figure.savefig(paths["figures"] / filename, dpi=300, bbox_inches="tight")
            base.plt.close(figure)

    recurrence_path = paths["tables"] / "feature_recurrence_by_size.csv"
    if recurrence_path.exists():
        recurrence_df = pd.read_csv(recurrence_path)
        for size in sorted(recurrence_df["feature_count"].unique()):
            sub = recurrence_df.loc[recurrence_df["feature_count"] == size].head(35)
            figure, ax = base.plt.subplots(figsize=(8, max(5, 0.26 * len(sub))))
            sns.barplot(data=sub, y="protein", x="seed_frequency", color="#4c78a8", ax=ax)
            ax.set_title(f"V8 feature recurrence, {size}-protein panel")
            ax.set_xlabel("Fraction of seeds selected")
            ax.set_ylabel("Protein")
            ax.set_xlim(0, 1)
            figure.tight_layout()
            figure.savefig(paths["figures"] / f"feature_recurrence_{size}.png", dpi=300, bbox_inches="tight")
            base.plt.close(figure)


def clean_previous_outputs(paths: dict[str, Path], args: argparse.Namespace) -> None:
    if args.resume:
        return
    for filename in [
        "all_test_results.csv",
        "selected_features_by_seed.csv",
        "classification_reports_by_seed.csv",
        "per_class_recall_auc_by_seed.csv",
        "confusion_matrices_long_by_seed.csv",
    ]:
        path = paths["tables"] / filename
        if path.exists():
            path.unlink()
    for path in paths["tables"].glob("stage2_dev_scores_seed_*.csv"):
        path.unlink()


def main() -> None:
    args = parse_args()
    args.search_budgets = sorted({int(value) for value in args.search_budgets if int(value) > 0})
    args.focus_sizes = sorted({int(value) for value in args.focus_sizes if int(value) in set(args.search_budgets)})
    if not args.focus_sizes:
        raise ValueError("No valid focus sizes. They must be included in --search-budgets.")
    if args.primary_coach not in args.final_coaches:
        args.final_coaches = [args.primary_coach, *args.final_coaches]

    paths = output_paths(args.output_dir)
    ensure_dirs(paths)
    clean_previous_outputs(paths, args)
    full_df = pd.read_csv(args.input)
    protein_columns = base.protein_columns_from_df(full_df, args.class_column, args.sample_id_column)
    classes = sorted(full_df[args.class_column].dropna().unique().tolist())
    if len(classes) != 12:
        log_message(paths, f"WARNING: discovered {len(classes)} classes, not 12: {classes}")

    log_message(paths, f"V8 starting. seeds={args.seeds}, search_budgets={args.search_budgets}, focus_sizes={args.focus_sizes}")
    log_message(paths, "Leakage control: imputation, DEA, ranking, and panel selection are repeated inside train only for each seed.")
    for seed in args.seeds:
        run_seed(seed, full_df, protein_columns, classes, paths, args)

    build_summary_tables(paths, args)
    save_figures(paths, args)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "input": str(args.input.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "seeds": args.seeds,
        "test_size": args.test_size,
        "imputer": args.imputer,
        "knn_neighbors": args.knn_neighbors,
        "per_cancer_top_m": args.per_cancer_top_m,
        "min_cancer_coverage": args.min_cancer_coverage,
        "search_budgets": args.search_budgets,
        "focus_sizes": args.focus_sizes,
        "cv_folds": args.cv_folds,
        "candidate_bank_size": args.candidate_bank_size,
        "random_panels_per_size": args.random_panels_per_size,
        "max_panels_per_seed": args.max_panels_per_seed,
        "primary_coach": args.primary_coach,
        "final_coaches": args.final_coaches,
        "notes": [
            "No V5/V6/V7 held-out test output is used as feature input or selection input.",
            "Held-out test is evaluated only after each seed selects one panel per focus size by train-only CV objective.",
            "Primary comparison should use primary_coach rows; other coaches are secondary sensitivity checks.",
        ],
    }
    with (paths["manifests"] / "v8_run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    log_message(paths, f"V8 complete. Main summary: {paths['tables'] / 'final_decision_summary_primary_coach.csv'}")


if __name__ == "__main__":
    main()
