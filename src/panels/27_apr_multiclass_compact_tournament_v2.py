#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
REVISE_ROOT = ROOT / "revise_plan"
PART_B_ROOT = REVISE_ROOT / "part_b_multiclass"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
for pipeline_dir in (ROOT / "scripts", ROOT / "obsolete_scripts"):
    if pipeline_dir.exists() and str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))

_V1_PATH = SCRIPT_DIR / "27_apr_multiclass_compact_tournament.py"
_V1_SPEC = importlib.util.spec_from_file_location("compact_tournament_v1", _V1_PATH)
v1 = importlib.util.module_from_spec(_V1_SPEC)
assert _V1_SPEC.loader is not None
_V1_SPEC.loader.exec_module(v1)

base = v1.base
optuna = v1.optuna

RANKER_NAMES = (
    "LOGISTIC_COEF",
    "RF_IMPORTANCE",
    "UNIVAR_OVR_AUC",
    "OVR_STABILITY",
    "OVR_MINMAX",
    "PAIRWISE_STABILITY",
    "PAIRWISE_MINMAX",
    "COVERAGE_IMPORTANCE",
    "CONSENSUS_SOFT_VOTE",
    "OVR_BONF_QUOTA",
)
MAIN_GM_NAMES = ("LR_L2", "LR_EN", "RF")
HGB_GM_NAMES = ("HGB",)
GM_NAMES = MAIN_GM_NAMES + HGB_GM_NAMES
COACH_NAMES = ("LR_L2", "LR_EN", "RF", "SOFT_VOTE")
FINAL_STAGE_COACH_NAMES = ("HGB",)
DEFAULT_SIZE_BUCKETS = (12, 18, 24, 30, 36, 39)
RESULT_COLUMNS = [
    "trial",
    "source_trial",
    "block_trial",
    "objective_score",
    "ranker",
    "gm",
    "coach",
    "n_total",
    "min_per_class",
    "n_shared",
    "feature_count",
    "features",
    "model_params",
    "cv_objective_score",
    "cv_accuracy_mean",
    "cv_balanced_accuracy_mean",
    "cv_macro_f1_mean",
    "cv_macro_ovr_auc_mean",
    "cv_min_class_recall_mean",
    "cv_accuracy_std",
    "cv_balanced_accuracy_std",
    "cv_macro_f1_std",
    "cv_macro_ovr_auc_std",
    "cv_min_class_recall_std",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_f1",
    "test_macro_ovr_auc",
    "test_min_class_recall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compact 12-class multiclass v2 tournament with LOCO-like cancer-consensus rankers "
            "and size-bucket finalist selection."
        )
    )
    parser.add_argument("--input", type=Path, default=v1.SOURCE_INPUT)
    parser.add_argument("--class-column", default="Cancer")
    parser.add_argument("--sample-id-column", default="Sample_ID")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--imputer", choices=sorted(base.IMPUTER_METHODS), default="knn")
    parser.add_argument("--knn-neighbors", type=int, default=base.KNN_NEIGHBORS)
    parser.add_argument("--per-cancer-top-m", type=int, default=10)
    parser.add_argument("--min-cancer-coverage", type=int, default=2)
    parser.add_argument("--min-features", type=int, default=6)
    parser.add_argument("--max-features", type=int, default=39)
    parser.add_argument("--trials-per-block", type=int, default=1000)
    parser.add_argument(
        "--hgb-trials-per-block",
        type=int,
        default=100,
        help="Late-stage lightweight HGB GM trials per ranker. Main non-HGB GMs still use --trials-per-block.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--size-buckets", nargs="+", type=int, default=list(DEFAULT_SIZE_BUCKETS))
    parser.add_argument("--top-per-bucket", type=int, default=3)
    parser.add_argument("--top-global", type=int, default=10)
    parser.add_argument(
        "--evaluate-all-trials",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, evaluate every CV trial with every coach, giving rankers x GMs x trials x coaches rows.",
    )
    parser.add_argument(
        "--hgb-gm-stage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run lightweight HGB as a late GM stage after all non-HGB GM blocks finish.",
    )
    parser.add_argument(
        "--hgb-coach-stage",
        "--hgb-finalist-coach",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run lightweight HGB as a final coach on finalists and HGB-GM candidates only.",
    )
    parser.add_argument(
        "--hgb-coach-max-candidates",
        type=int,
        default=2000,
        help="Maximum candidates for final HGB coach stage after adding finalists and HGB-GM candidates.",
    )
    parser.add_argument("--output-dir", type=Path, default=PART_B_ROOT / "compact_tournament_27apr_v2")
    return parser.parse_args()


def output_paths(output_root: Path) -> dict[str, Path]:
    return {
        "root": output_root,
        "tables": output_root / "tables",
        "figures": output_root / "figures",
        "manifests": output_root / "manifests",
        "logs": output_root / "logs",
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


def normalize_results_df(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in RESULT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = np.nan
    return normalized[RESULT_COLUMNS]


def append_checkpoint(rows: pd.DataFrame | list[dict[str, object]], path: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df = normalize_results_df(df)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def cv_block_plan(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    blocks: list[tuple[str, str, int]] = []
    for ranker in RANKER_NAMES:
        for gm in MAIN_GM_NAMES:
            blocks.append((ranker, gm, args.trials_per_block))
    if args.hgb_gm_stage and args.hgb_trials_per_block > 0:
        for ranker in RANKER_NAMES:
            blocks.append((ranker, "HGB", args.hgb_trials_per_block))
    return blocks


def expected_cv_trials(args: argparse.Namespace) -> int:
    return int(sum(block_trials for _, _, block_trials in cv_block_plan(args)))


def expected_main_test_rows(args: argparse.Namespace) -> int:
    return expected_cv_trials(args) * len(COACH_NAMES)


def directional_auc(y_binary: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_binary)) < 2:
        return 0.5
    try:
        auc = float(roc_auc_score(y_binary, scores))
    except ValueError:
        return 0.5
    return max(auc, 1.0 - auc)


def rank_ovr_stability(x_train: pd.DataFrame, y_train: pd.Series, classes: list[str]) -> pd.DataFrame:
    y_values = y_train.to_numpy()
    rows: list[dict[str, object]] = []
    for protein in x_train.columns:
        scores = x_train[protein].to_numpy()
        aucs = np.asarray([directional_auc((y_values == label).astype(int), scores) for label in classes], dtype=float)
        rows.append(
            {
                "protein": protein,
                "importance": float(0.60 * aucs.mean() + 0.25 * aucs.min() - 0.15 * aucs.std()),
                "mean_auc": float(aucs.mean()),
                "min_auc": float(aucs.min()),
                "std_auc": float(aucs.std()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["importance", "mean_auc", "min_auc", "protein"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def rank_ovr_minmax(x_train: pd.DataFrame, y_train: pd.Series, classes: list[str]) -> pd.DataFrame:
    y_values = y_train.to_numpy()
    rows: list[dict[str, object]] = []
    for protein in x_train.columns:
        scores = x_train[protein].to_numpy()
        aucs = np.asarray([directional_auc((y_values == label).astype(int), scores) for label in classes], dtype=float)
        rows.append(
            {
                "protein": protein,
                "importance": float(0.45 * aucs.mean() + 0.45 * aucs.min() + 0.10 * np.quantile(aucs, 0.25)),
                "mean_auc": float(aucs.mean()),
                "min_auc": float(aucs.min()),
                "q25_auc": float(np.quantile(aucs, 0.25)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["importance", "min_auc", "mean_auc", "protein"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def rank_pairwise_stability(x_train: pd.DataFrame, y_train: pd.Series, classes: list[str]) -> pd.DataFrame:
    y_values = y_train.to_numpy()
    pairs = list(combinations(classes, 2))
    rows: list[dict[str, object]] = []
    for protein in x_train.columns:
        protein_values = x_train[protein].to_numpy()
        aucs: list[float] = []
        for class_a, class_b in pairs:
            mask = (y_values == class_a) | (y_values == class_b)
            pair_y = (y_values[mask] == class_a).astype(int)
            pair_scores = protein_values[mask]
            aucs.append(directional_auc(pair_y, pair_scores))
        auc_array = np.asarray(aucs, dtype=float)
        rows.append(
            {
                "protein": protein,
                "importance": float(0.65 * auc_array.mean() + 0.20 * auc_array.min() - 0.15 * auc_array.std()),
                "mean_pairwise_auc": float(auc_array.mean()),
                "min_pairwise_auc": float(auc_array.min()),
                "std_pairwise_auc": float(auc_array.std()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["importance", "mean_pairwise_auc", "min_pairwise_auc", "protein"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def rank_pairwise_minmax(x_train: pd.DataFrame, y_train: pd.Series, classes: list[str]) -> pd.DataFrame:
    y_values = y_train.to_numpy()
    pairs = list(combinations(classes, 2))
    rows: list[dict[str, object]] = []
    for protein in x_train.columns:
        protein_values = x_train[protein].to_numpy()
        aucs: list[float] = []
        for class_a, class_b in pairs:
            mask = (y_values == class_a) | (y_values == class_b)
            pair_y = (y_values[mask] == class_a).astype(int)
            aucs.append(directional_auc(pair_y, protein_values[mask]))
        auc_array = np.asarray(aucs, dtype=float)
        rows.append(
            {
                "protein": protein,
                "importance": float(
                    0.45 * auc_array.mean() + 0.35 * np.quantile(auc_array, 0.10) + 0.20 * auc_array.min()
                ),
                "mean_pairwise_auc": float(auc_array.mean()),
                "q10_pairwise_auc": float(np.quantile(auc_array, 0.10)),
                "min_pairwise_auc": float(auc_array.min()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["importance", "q10_pairwise_auc", "mean_pairwise_auc", "protein"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def reciprocal_rank_vote(ranking_tables: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    score_map: dict[str, float] = {}
    detail_map: dict[str, list[str]] = {}
    for ranker, table in ranking_tables.items():
        weight = weights.get(ranker, 1.0)
        proteins = table.drop_duplicates("protein", keep="first")["protein"].tolist()
        for rank, protein in enumerate(proteins, start=1):
            score_map[protein] = score_map.get(protein, 0.0) + weight / math.log2(rank + 1)
            detail_map.setdefault(protein, []).append(ranker)
    rows = [
        {
            "protein": protein,
            "importance": score,
            "ranker_votes": ";".join(sorted(detail_map.get(protein, []))),
            "n_ranker_votes": len(set(detail_map.get(protein, []))),
        }
        for protein, score in score_map.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["importance", "n_ranker_votes", "protein"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_rankings_v2(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    classes: list[str],
    quota_sources_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    logistic_df = v1.rank_logistic_coef(x_train, y_train, seed)
    rf_df = v1.rank_rf_importance(x_train, y_train, seed)
    univar_df = v1.rank_univar_ovr_auc(x_train, y_train, classes)
    ovr_stability_df = rank_ovr_stability(x_train, y_train, classes)
    ovr_minmax_df = rank_ovr_minmax(x_train, y_train, classes)
    pairwise_stability_df = rank_pairwise_stability(x_train, y_train, classes)
    pairwise_minmax_df = rank_pairwise_minmax(x_train, y_train, classes)
    quota_df = v1.rank_ovr_bonf_quota(quota_sources_df, classes, logistic_df["protein"].tolist())

    coverage_importance_df = logistic_df.merge(
        coverage_df[["protein", "cancer_count"]] if not coverage_df.empty else pd.DataFrame(columns=["protein", "cancer_count"]),
        on="protein",
        how="left",
    )
    coverage_importance_df["cancer_count"] = coverage_importance_df["cancer_count"].fillna(0)
    coverage_importance_df["coverage_score"] = v1.normalize_series(coverage_importance_df["cancer_count"])
    coverage_importance_df["importance_score"] = v1.normalize_series(coverage_importance_df["importance"])
    coverage_importance_df["importance"] = (
        0.50 * coverage_importance_df["importance_score"]
        + 0.35 * coverage_importance_df["coverage_score"]
        + 0.15 * coverage_importance_df["importance_score"] * coverage_importance_df["coverage_score"]
    )
    coverage_importance_df = coverage_importance_df.sort_values(
        ["importance", "cancer_count", "protein"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    base_tables = {
        "LOGISTIC_COEF": logistic_df,
        "RF_IMPORTANCE": rf_df,
        "UNIVAR_OVR_AUC": univar_df,
        "OVR_STABILITY": ovr_stability_df,
        "OVR_MINMAX": ovr_minmax_df,
        "PAIRWISE_STABILITY": pairwise_stability_df,
        "PAIRWISE_MINMAX": pairwise_minmax_df,
        "COVERAGE_IMPORTANCE": coverage_importance_df,
        "OVR_BONF_QUOTA": quota_df,
    }
    consensus_df = reciprocal_rank_vote(
        base_tables,
        weights={
            "LOGISTIC_COEF": 1.15,
            "RF_IMPORTANCE": 0.75,
            "UNIVAR_OVR_AUC": 0.90,
            "OVR_STABILITY": 1.25,
            "OVR_MINMAX": 1.10,
            "PAIRWISE_STABILITY": 1.30,
            "PAIRWISE_MINMAX": 1.20,
            "COVERAGE_IMPORTANCE": 1.00,
            "OVR_BONF_QUOTA": 0.85,
        },
    )
    ranking_tables = {**base_tables, "CONSENSUS_SOFT_VOTE": consensus_df}

    ranking_rows: list[pd.DataFrame] = []
    rankings: dict[str, list[str]] = {}
    for ranker, table in ranking_tables.items():
        ranked = table.drop_duplicates("protein", keep="first").copy().reset_index(drop=True)
        ranked["ranker"] = ranker
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranking_rows.append(ranked)
        rankings[ranker] = ranked["protein"].tolist()
    return rankings, pd.concat(ranking_rows, ignore_index=True)


def compact_objective(metrics: dict[str, float], feature_count: int, max_features: int) -> float:
    macro_auc = 0.0 if math.isnan(metrics["cv_macro_ovr_auc_mean"]) else metrics["cv_macro_ovr_auc_mean"]
    min_recall = 0.0 if math.isnan(metrics["cv_min_class_recall_mean"]) else metrics["cv_min_class_recall_mean"]
    return float(
        0.25 * metrics["cv_balanced_accuracy_mean"]
        + 0.25 * metrics["cv_macro_f1_mean"]
        + 0.20 * macro_auc
        + 0.20 * min_recall
        - 0.20 * (feature_count / max_features)
        - 0.10 * metrics["cv_macro_f1_std"]
    )


def make_estimator_v2(model_name: str, seed: int, params: dict[str, object] | None = None):
    params = params or {}
    if model_name == "LR_L2":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(params.get("C", 0.1)),
                        class_weight="balanced",
                        max_iter=4000,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    if model_name == "LR_EN":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(params.get("C", 0.1)),
                        class_weight="balanced",
                        l1_ratio=float(params.get("l1_ratio", 0.5)),
                        max_iter=4000,
                        n_jobs=1,
                        penalty="elasticnet",
                        random_state=seed,
                        solver="saga",
                    ),
                ),
            ]
        )
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    if model_name == "HGB":
        return HistGradientBoostingClassifier(
            max_iter=int(params.get("max_iter", 75)),
            learning_rate=float(params.get("learning_rate", 0.08)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
            l2_regularization=float(params.get("l2_regularization", 0.1)),
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=seed,
        )
    if model_name == "SOFT_VOTE":
        return VotingClassifier(
            estimators=[
                ("lr_l2", make_estimator_v2("LR_L2", seed, {"C": params.get("lr_l2_C", 0.1)})),
                (
                    "lr_en",
                    make_estimator_v2(
                        "LR_EN",
                        seed,
                        {"C": params.get("lr_en_C", 0.1), "l1_ratio": params.get("lr_en_l1_ratio", 0.5)},
                    ),
                ),
                (
                    "rf",
                    make_estimator_v2(
                        "RF",
                        seed,
                        {
                            "n_estimators": params.get("rf_n_estimators", 300),
                            "max_depth": params.get("rf_max_depth", None),
                            "min_samples_leaf": params.get("rf_min_samples_leaf", 1),
                        },
                    ),
                ),
            ],
            voting="soft",
            n_jobs=1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def suggest_model_params_v2(trial: optuna.Trial, model_name: str) -> dict[str, object]:
    if model_name in {"LR_L2", "LR_EN", "RF"}:
        return v1.suggest_model_params(trial, model_name)
    if model_name == "HGB":
        return {
            "max_iter": trial.suggest_categorical("hgb_max_iter", [50, 75, 100]),
            "learning_rate": trial.suggest_categorical("hgb_learning_rate", [0.05, 0.08, 0.1]),
            "max_leaf_nodes": trial.suggest_categorical("hgb_max_leaf_nodes", [7, 15, 31]),
            "l2_regularization": trial.suggest_categorical("hgb_l2_regularization", [0.01, 0.1, 1.0]),
        }
    return {}


def random_model_params(rng: np.random.Generator, model_name: str) -> dict[str, object]:
    if model_name == "HGB":
        return {
            "max_iter": int(rng.choice([50, 75, 100])),
            "learning_rate": float(rng.choice([0.05, 0.08, 0.1])),
            "max_leaf_nodes": int(rng.choice([7, 15, 31])),
            "l2_regularization": float(rng.choice([0.01, 0.1, 1.0])),
        }
    return v1.random_model_params(rng, model_name)


def cv_evaluate_v2(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    features: list[str],
    labels: list[str],
    model_name: str,
    model_params: dict[str, object],
    cv_folds: int,
    seed: int,
) -> dict[str, float]:
    splitter = v1.StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    rows: list[dict[str, float]] = []
    for fold_index, (fit_idx, val_idx) in enumerate(splitter.split(x_train[features], y_train), start=1):
        estimator = make_estimator_v2(model_name, seed + fold_index, model_params)
        x_fit = x_train.iloc[fit_idx][features]
        x_val = x_train.iloc[val_idx][features]
        y_fit = y_train.iloc[fit_idx]
        y_val = y_train.iloc[val_idx]
        estimator.fit(x_fit, y_fit)
        y_pred = estimator.predict(x_val)
        y_score = v1.aligned_probabilities(estimator, x_val, labels)
        rows.append(v1.metric_bundle(y_val, y_pred, y_score, labels))
    cv_df = pd.DataFrame(rows)
    summary = {f"cv_{column}_mean": float(cv_df[column].mean()) for column in cv_df.columns}
    summary.update({f"cv_{column}_std": float(cv_df[column].std(ddof=0)) for column in cv_df.columns})
    return summary


def evaluate_on_test_v2(
    candidate_row: pd.Series,
    coach_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    labels: list[str],
    seed: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    features = str(candidate_row["features"]).split(";")
    estimator = make_estimator_v2(coach_name, seed, {})
    estimator.fit(x_train[features], y_train)
    y_pred = estimator.predict(x_test[features])
    y_score = v1.aligned_probabilities(estimator, x_test[features], labels)
    metrics = v1.metric_bundle(y_test, y_pred, y_score, labels)
    row = {
        "source_trial": int(candidate_row["trial"]),
        "coach": coach_name,
        "ranker": candidate_row["ranker"],
        "gm": candidate_row["gm"],
        "feature_count": len(features),
        "features": ";".join(features),
        "cv_objective_score": candidate_row["objective_score"],
        "cv_balanced_accuracy_mean": candidate_row.get("cv_balanced_accuracy_mean", math.nan),
        "cv_macro_f1_mean": candidate_row.get("cv_macro_f1_mean", math.nan),
        "cv_macro_ovr_auc_mean": candidate_row.get("cv_macro_ovr_auc_mean", math.nan),
        "cv_min_class_recall_mean": candidate_row.get("cv_min_class_recall_mean", math.nan),
        "test_accuracy": metrics["accuracy"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_macro_ovr_auc": metrics["macro_ovr_auc"],
        "test_min_class_recall": metrics["min_class_recall"],
    }
    return row, np.asarray(y_pred), y_score


def structural_configs(size_buckets: list[int], max_features: int) -> list[dict[str, Any]]:
    sizes = sorted({size for size in size_buckets if 1 <= size <= max_features} | {max_features})
    configs: list[dict[str, Any]] = []
    for ranker in RANKER_NAMES:
        for gm in GM_NAMES:
            for size in sizes:
                for min_per_class in (1, 2, min(4, max(1, size // 12))):
                    if gm == "LR_L2":
                        model_params = {"C": 0.1}
                    elif gm == "LR_EN":
                        model_params = {"C": 0.1, "l1_ratio": 0.5}
                    elif gm == "RF":
                        model_params = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1}
                    else:
                        model_params = {
                            "max_iter": 75,
                            "learning_rate": 0.08,
                            "max_leaf_nodes": 15,
                            "l2_regularization": 0.1,
                        }
                    configs.append(
                        {
                            "ranker": ranker,
                            "gm": gm,
                            "n_total": size,
                            "min_per_class": int(min_per_class),
                            "n_shared": min(12, size),
                            "model_params": model_params,
                        }
                    )
    return configs


def random_configs(
    n_configs: int,
    min_features: int,
    max_features: int,
    n_classes: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    configs: list[dict[str, Any]] = []
    for _ in range(n_configs):
        n_total = int(rng.integers(min_features, max_features + 1))
        max_quota = max(0, min(6, n_total // max(n_classes, 1)))
        gm = str(rng.choice(GM_NAMES))
        configs.append(
            {
                "ranker": str(rng.choice(RANKER_NAMES)),
                "gm": gm,
                "n_total": n_total,
                "min_per_class": int(rng.integers(0, max_quota + 1)),
                "n_shared": int(rng.integers(0, min(40, n_total) + 1)),
                "model_params": random_model_params(rng, gm),
            }
        )
    return configs


def evaluate_config(
    config: dict[str, Any],
    trial_number: int,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    classes: list[str],
    rankings: dict[str, list[str]],
    quota_sources_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    cv_folds: int,
    seed: int,
    min_features: int,
    max_features: int,
) -> dict[str, object] | None:
    features = v1.build_panel(
        ranker_name=str(config["ranker"]),
        n_total=int(config["n_total"]),
        min_per_class=int(config["min_per_class"]),
        n_shared=int(config["n_shared"]),
        rankings=rankings,
        quota_sources_df=quota_sources_df,
        coverage_df=coverage_df,
        classes=classes,
    )
    if len(features) < min_features:
        return None
    metrics = cv_evaluate_v2(
        x_train=x_train,
        y_train=y_train,
        features=features,
        labels=classes,
        model_name=str(config["gm"]),
        model_params=dict(config.get("model_params", {})),
        cv_folds=cv_folds,
        seed=seed,
    )
    return {
        "trial": trial_number,
        "objective_score": compact_objective(metrics, len(features), max_features),
        "ranker": str(config["ranker"]),
        "gm": str(config["gm"]),
        "n_total": int(config["n_total"]),
        "min_per_class": int(config["min_per_class"]),
        "n_shared": int(config["n_shared"]),
        "feature_count": len(features),
        "features": ";".join(features),
        "model_params": json.dumps(dict(config.get("model_params", {})), sort_keys=True),
        **metrics,
    }


def optuna_search(
    args: argparse.Namespace,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    classes: list[str],
    rankings: dict[str, list[str]],
    quota_sources_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    cv_folds: int,
    max_search_features: int,
    paths: dict[str, Path],
) -> pd.DataFrame:
    all_rows: list[pd.DataFrame] = []
    global_trial = 0
    seed_configs = structural_configs(args.size_buckets, max_search_features)
    checkpoint_path = paths["tables"] / "cv_results_checkpoint.csv"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    blocks = cv_block_plan(args)
    total_blocks = len(blocks)
    for block_index, (ranker, gm, block_trials) in enumerate(blocks, start=1):
        log_message(paths, f"[CV block {block_index}/{total_blocks}] ranker={ranker} gm={gm} trials={block_trials}")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed + block_index),
        )
        for config in seed_configs:
            if config["ranker"] == ranker and config["gm"] == gm:
                payload = {key: value for key, value in config.items() if key not in {"model_params", "ranker", "gm"}}
                study.enqueue_trial(payload)

        def objective(trial: optuna.Trial, ranker_name: str = ranker, gm_name: str = gm) -> float:
            n_total = trial.suggest_int("n_total", args.min_features, max_search_features)
            max_quota = max(0, min(6, n_total // max(len(classes), 1)))
            config = {
                "ranker": ranker_name,
                "gm": gm_name,
                "n_total": n_total,
                "min_per_class": trial.suggest_int("min_per_class", 0, max_quota),
                "n_shared": trial.suggest_int("n_shared", 0, min(40, n_total)),
                "model_params": suggest_model_params_v2(trial, gm_name),
            }
            row = evaluate_config(
                config=config,
                trial_number=trial.number,
                x_train=x_train,
                y_train=y_train,
                classes=classes,
                rankings=rankings,
                quota_sources_df=quota_sources_df,
                coverage_df=coverage_df,
                cv_folds=cv_folds,
                seed=args.seed,
                min_features=args.min_features,
                max_features=args.max_features,
            )
            if row is None:
                raise optuna.TrialPruned()
            for key, value in row.items():
                if key not in {"trial", "objective_score", "ranker", "gm", "n_total", "min_per_class", "n_shared"}:
                    trial.set_user_attr(key, value)
            return float(row["objective_score"])

        study.optimize(objective, n_trials=block_trials, show_progress_bar=False)
        block_df = v1.trial_records(study)
        if block_df.empty:
            log_message(paths, f"[CV block {block_index}/{total_blocks}] no completed trials")
            continue
        block_df["ranker"] = ranker
        block_df["gm"] = gm
        block_df["block_trial"] = block_df["trial"]
        block_df["trial"] = np.arange(global_trial, global_trial + len(block_df))
        global_trial += len(block_df)
        all_rows.append(block_df)
        append_checkpoint(block_df, checkpoint_path)
        best = block_df.sort_values("objective_score", ascending=False).iloc[0]
        log_message(
            paths,
            f"[CV block {block_index}/{total_blocks}] completed={len(block_df)} "
            f"best_n={int(best['feature_count'])} best_obj={best['objective_score']:.3f} "
            f"best_BA={best['cv_balanced_accuracy_mean']:.3f} best_F1={best['cv_macro_f1_mean']:.3f}",
        )
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def fallback_search(
    args: argparse.Namespace,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    classes: list[str],
    rankings: dict[str, list[str]],
    quota_sources_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    cv_folds: int,
    max_search_features: int,
    paths: dict[str, Path],
) -> pd.DataFrame:
    seed_configs = structural_configs(args.size_buckets, max_search_features)
    rows: list[dict[str, object]] = []
    global_trial = 0
    checkpoint_path = paths["tables"] / "cv_results_checkpoint.csv"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    blocks = cv_block_plan(args)
    total_blocks = len(blocks)
    for block_index, (ranker, gm, block_trials) in enumerate(blocks, start=1):
        log_message(paths, f"[CV block {block_index}/{total_blocks}] ranker={ranker} gm={gm} trials={block_trials}")
        block_configs = [
            config for config in seed_configs if config["ranker"] == ranker and config["gm"] == gm
        ]
        remaining = max(0, block_trials - len(block_configs))
        random_block = random_configs(
            remaining,
            args.min_features,
            max_search_features,
            len(classes),
            args.seed + 1000 * block_index,
        )
        for config in random_block:
            config["ranker"] = ranker
            config["gm"] = gm
            config["model_params"] = random_model_params(np.random.default_rng(args.seed + global_trial), gm)
        block_configs.extend(random_block)
        block_rows: list[dict[str, object]] = []
        for block_trial, config in enumerate(block_configs[:block_trials]):
            row = evaluate_config(
                config=config,
                trial_number=global_trial,
                x_train=x_train,
                y_train=y_train,
                classes=classes,
                rankings=rankings,
                quota_sources_df=quota_sources_df,
                coverage_df=coverage_df,
                cv_folds=cv_folds,
                seed=args.seed,
                min_features=args.min_features,
                max_features=args.max_features,
            )
            global_trial += 1
            if row is not None:
                row["block_trial"] = block_trial
                rows.append(row)
                block_rows.append(row)
        block_df = pd.DataFrame(block_rows)
        if not block_df.empty:
            append_checkpoint(block_df, checkpoint_path)
            best = block_df.sort_values("objective_score", ascending=False).iloc[0]
            log_message(
                paths,
                f"[CV block {block_index}/{total_blocks}] completed={len(block_df)} "
                f"best_n={int(best['feature_count'])} best_obj={best['objective_score']:.3f} "
                f"best_BA={best['cv_balanced_accuracy_mean']:.3f} best_F1={best['cv_macro_f1_mean']:.3f}",
            )
    return pd.DataFrame(rows)


def select_finalists(cv_results_df: pd.DataFrame, size_buckets: list[int], top_per_bucket: int, top_global: int) -> pd.DataFrame:
    if cv_results_df.empty:
        return cv_results_df
    sorted_df = cv_results_df.sort_values(
        ["objective_score", "cv_balanced_accuracy_mean", "cv_macro_f1_mean"],
        ascending=[False, False, False],
    )
    finalists = [sorted_df.head(top_global)]
    for bucket in sorted(set(size_buckets)):
        bucket_df = sorted_df.loc[sorted_df["feature_count"] <= bucket].head(top_per_bucket)
        if not bucket_df.empty:
            bucket_df = bucket_df.copy()
            bucket_df["size_bucket"] = bucket
            finalists.append(bucket_df)
    result = pd.concat(finalists, ignore_index=True)
    result["feature_signature"] = result["features"]
    result = result.drop_duplicates("feature_signature", keep="first")
    return result.sort_values(["feature_count", "objective_score"], ascending=[True, False]).reset_index(drop=True)


def select_hgb_coach_candidates(
    cv_results_df: pd.DataFrame,
    finalists_df: pd.DataFrame,
    max_candidates: int,
) -> pd.DataFrame:
    pieces = []
    if not finalists_df.empty:
        pieces.append(finalists_df)
    if not cv_results_df.empty and "gm" in cv_results_df.columns:
        hgb_gm_df = cv_results_df.loc[cv_results_df["gm"] == "HGB"].copy()
        if not hgb_gm_df.empty:
            pieces.append(hgb_gm_df)
    if not pieces:
        return pd.DataFrame()
    candidates = pd.concat(pieces, ignore_index=True)
    candidates["feature_signature"] = candidates["features"].astype(str)
    candidates = candidates.sort_values(
        ["objective_score", "cv_balanced_accuracy_mean", "cv_macro_f1_mean", "feature_count"],
        ascending=[False, False, False, True],
    )
    candidates = candidates.drop_duplicates("feature_signature", keep="first").reset_index(drop=True)
    if max_candidates > 0:
        candidates = candidates.head(max_candidates).copy()
    return candidates


def test_finalists(
    finalists_df: pd.DataFrame,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    classes: list[str],
    seed: int,
    paths: dict[str, Path],
    coaches: tuple[str, ...] = COACH_NAMES,
    checkpoint_name: str = "test_results_checkpoint.csv",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    checkpoint_path = paths["tables"] / checkpoint_name
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    total_candidates = len(finalists_df)
    total_rows = total_candidates * len(coaches)
    log_message(paths, f"[TEST] evaluating {total_candidates} candidates x {len(coaches)} coaches = {total_rows} rows")
    for candidate_index, (_, candidate) in enumerate(finalists_df.iterrows(), start=1):
        chunk_rows: list[dict[str, object]] = []
        for coach in coaches:
            row, y_pred, y_score = evaluate_on_test_v2(candidate, coach, x_train, y_train, x_test, y_test, classes, seed)
            rows.append(row)
            chunk_rows.append(row)
        append_checkpoint(chunk_rows, checkpoint_path)
        if candidate_index == 1 or candidate_index % 100 == 0 or candidate_index == total_candidates:
            best_so_far = pd.DataFrame(rows).sort_values(
                ["test_balanced_accuracy", "test_macro_f1", "test_macro_ovr_auc", "feature_count"],
                ascending=[False, False, False, True],
            ).iloc[0]
            log_message(
                paths,
                f"[TEST] candidate {candidate_index}/{total_candidates}; "
                f"best_n={int(best_so_far['feature_count'])} best_BA={best_so_far['test_balanced_accuracy']:.3f} "
                f"best_F1={best_so_far['test_macro_f1']:.3f}",
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["test_balanced_accuracy", "test_macro_f1", "test_macro_ovr_auc", "feature_count"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    return result


def build_size_tradeoff(test_results_df: pd.DataFrame, size_buckets: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bucket in sorted(set(size_buckets)):
        sub = test_results_df.loc[test_results_df["feature_count"] <= bucket].copy()
        if sub.empty:
            continue
        best = sub.sort_values(
            ["test_balanced_accuracy", "test_macro_f1", "test_macro_ovr_auc", "feature_count"],
            ascending=[False, False, False, True],
        ).iloc[0]
        rows.append({"size_bucket_max": bucket, **best.to_dict()})
    return pd.DataFrame(rows)


def save_best_outputs(
    best_row: pd.Series,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    test_imputed: pd.DataFrame,
    sample_id_column: str,
    classes: list[str],
    paths: dict[str, Path],
    seed: int,
) -> None:
    features = str(best_row["features"]).split(";")
    estimator = make_estimator_v2(str(best_row["coach"]), seed, {})
    estimator.fit(x_train[features], y_train)
    y_pred = estimator.predict(x_test[features])
    y_score = v1.aligned_probabilities(estimator, x_test[features], classes)

    best_row.to_frame().T.to_csv(paths["tables"] / "best_panel_summary.csv", index=False)
    pd.DataFrame({"rank": np.arange(1, len(features) + 1), "protein": features}).to_csv(
        paths["tables"] / "best_panel_features.csv",
        index=False,
    )

    report_df = pd.DataFrame(classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)).transpose()
    report_df.reset_index().rename(columns={"index": "label"}).to_csv(paths["tables"] / "best_panel_per_class_metrics.csv", index=False)

    auc_rows = []
    y_values = y_test.to_numpy()
    for index, label in enumerate(classes):
        binary = (y_values == label).astype(int)
        auc_value = float(roc_auc_score(binary, y_score[:, index])) if len(np.unique(binary)) >= 2 else math.nan
        auc_rows.append({"label": label, "ovr_auc": auc_value})
    pd.DataFrame(auc_rows).to_csv(paths["tables"] / "best_panel_ovr_auc.csv", index=False)

    predictions_df = pd.DataFrame(
        {
            sample_id_column: test_imputed[sample_id_column],
            "true_class": y_test,
            "predicted_class": y_pred,
        }
    )
    for index, label in enumerate(classes):
        predictions_df[f"score_{label}"] = y_score[:, index]
    predictions_df.to_csv(paths["tables"] / "best_panel_test_predictions.csv", index=False)

    confusion = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=classes), index=classes, columns=classes)
    confusion.to_csv(paths["tables"] / "best_panel_confusion_matrix.csv")
    figure, ax = base.plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.3, linecolor="white", ax=ax)
    ax.set_title("Best compact v2 multiclass panel confusion matrix")
    ax.set_xlabel("Predicted cancer")
    ax.set_ylabel("True cancer")
    figure.tight_layout()
    figure.savefig(paths["figures"] / "best_panel_confusion_matrix.png", dpi=300, bbox_inches="tight")
    base.plt.close(figure)

    per_class_df = report_df.reset_index().rename(columns={"index": "label"})
    per_class_df = per_class_df.loc[per_class_df["label"].isin(classes)].copy()
    figure, ax = base.plt.subplots(figsize=(11, 5))
    per_class_long = per_class_df.melt(
        id_vars="label",
        value_vars=["precision", "recall", "f1-score"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=per_class_long, x="label", y="value", hue="metric", ax=ax)
    ax.set_title("Best compact v2 panel per-class metrics")
    ax.set_xlabel("Cancer class")
    ax.set_ylabel("Metric")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(paths["figures"] / "best_panel_per_class_metrics.png", dpi=300, bbox_inches="tight")
    base.plt.close(figure)

    auc_df = pd.DataFrame(auc_rows)
    figure, ax = base.plt.subplots(figsize=(10, 4.8))
    sns.barplot(data=auc_df, x="label", y="ovr_auc", ax=ax, color="#4c78a8")
    ax.set_title("Best compact v2 panel one-vs-rest AUC by class")
    ax.set_xlabel("Cancer class")
    ax.set_ylabel("OVR AUC")
    ax.set_ylim(0.5, 1.02)
    ax.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(paths["figures"] / "best_panel_ovr_auc.png", dpi=300, bbox_inches="tight")
    base.plt.close(figure)


def save_overview_figures(
    cv_results_df: pd.DataFrame,
    test_results_df: pd.DataFrame,
    size_tradeoff_df: pd.DataFrame,
    ranker_table: pd.DataFrame,
    paths: dict[str, Path],
    seed: int,
) -> None:
    if not size_tradeoff_df.empty:
        figure, ax = base.plt.subplots(figsize=(9, 5))
        plot_df = size_tradeoff_df.melt(
            id_vars="size_bucket_max",
            value_vars=["test_balanced_accuracy", "test_macro_f1", "test_macro_ovr_auc", "test_min_class_recall"],
            var_name="metric",
            value_name="value",
        )
        sns.lineplot(data=plot_df, x="size_bucket_max", y="value", hue="metric", marker="o", linewidth=2, ax=ax)
        ax.set_title("Compact v2 performance by feature budget")
        ax.set_xlabel("Maximum feature count")
        ax.set_ylabel("Held-out test metric")
        ax.set_ylim(0.0, 1.02)
        ax.legend(frameon=False, loc="best")
        figure.tight_layout()
        figure.savefig(paths["figures"] / "size_bucket_tradeoff.png", dpi=300, bbox_inches="tight")
        base.plt.close(figure)

    if not cv_results_df.empty:
        figure, ax = base.plt.subplots(figsize=(9, 5))
        sample_df = cv_results_df.sample(min(len(cv_results_df), 30000), random_state=seed)
        sns.scatterplot(
            data=sample_df,
            x="feature_count",
            y="cv_balanced_accuracy_mean",
            hue="ranker",
            alpha=0.45,
            s=22,
            linewidth=0,
            ax=ax,
        )
        ax.set_title("CV balanced accuracy vs compact panel size")
        ax.set_xlabel("Feature count")
        ax.set_ylabel("CV balanced accuracy")
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        figure.tight_layout()
        figure.savefig(paths["figures"] / "cv_feature_count_scatter.png", dpi=300, bbox_inches="tight")
        base.plt.close(figure)

        heatmap_df = cv_results_df.pivot_table(
            index="ranker",
            columns="gm",
            values="objective_score",
            aggfunc="max",
        )
        figure, ax = base.plt.subplots(figsize=(8, 6))
        sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
        ax.set_title("Best CV objective by ranker and GM")
        ax.set_xlabel("GM")
        ax.set_ylabel("Ranker")
        figure.tight_layout()
        figure.savefig(paths["figures"] / "cv_ranker_gm_heatmap.png", dpi=300, bbox_inches="tight")
        base.plt.close(figure)

    if not test_results_df.empty:
        figure, ax = base.plt.subplots(figsize=(9, 5))
        sample_df = test_results_df.sample(min(len(test_results_df), 30000), random_state=seed)
        sns.scatterplot(
            data=sample_df,
            x="feature_count",
            y="test_balanced_accuracy",
            hue="coach",
            style="gm",
            alpha=0.45,
            s=24,
            linewidth=0,
            ax=ax,
        )
        ax.set_title("Held-out test balanced accuracy vs compact panel size")
        ax.set_xlabel("Feature count")
        ax.set_ylabel("Test balanced accuracy")
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        figure.tight_layout()
        figure.savefig(paths["figures"] / "test_feature_count_scatter.png", dpi=300, bbox_inches="tight")
        base.plt.close(figure)

        heatmap_df = test_results_df.pivot_table(
            index="ranker",
            columns="coach",
            values="test_balanced_accuracy",
            aggfunc="max",
        )
        figure, ax = base.plt.subplots(figsize=(8, 6))
        sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlOrBr", ax=ax)
        ax.set_title("Best test balanced accuracy by ranker and coach")
        ax.set_xlabel("Coach")
        ax.set_ylabel("Ranker")
        figure.tight_layout()
        figure.savefig(paths["figures"] / "test_ranker_coach_heatmap.png", dpi=300, bbox_inches="tight")
        base.plt.close(figure)

    consensus_df = ranker_table.loc[ranker_table["ranker"] == "CONSENSUS_SOFT_VOTE"].head(25).copy()
    if not consensus_df.empty and "importance" in consensus_df.columns:
        figure, ax = base.plt.subplots(figsize=(8, 7))
        sns.barplot(data=consensus_df, y="protein", x="importance", ax=ax, color="#2e8b57")
        ax.set_title("Top 25 consensus soft-vote features")
        ax.set_xlabel("Consensus score")
        ax.set_ylabel("Protein")
        figure.tight_layout()
        figure.savefig(paths["figures"] / "consensus_top25_features.png", dpi=300, bbox_inches="tight")
        base.plt.close(figure)


def main() -> None:
    args = parse_args()
    paths = output_paths(args.output_dir)
    ensure_dirs(paths)
    log_file = paths["logs"] / "run_log.txt"
    if log_file.exists():
        log_file.unlink()
    log_message(paths, "Starting compact v2 multiclass tournament")
    log_message(
        paths,
        f"Configuration: max_features={args.max_features}, size_buckets={args.size_buckets}, "
        f"trials_per_block={args.trials_per_block}, hgb_trials_per_block={args.hgb_trials_per_block}, "
        f"hgb_gm_stage={args.hgb_gm_stage}, hgb_coach_stage={args.hgb_coach_stage}, "
        f"evaluate_all_trials={args.evaluate_all_trials}",
    )

    log_message(paths, "[1/8] Loading data")
    full_df = pd.read_csv(args.input)
    protein_columns = base.protein_columns_from_df(full_df, args.class_column, args.sample_id_column)
    classes = sorted(full_df[args.class_column].dropna().unique().tolist())
    class_counts = full_df[args.class_column].value_counts()
    cv_folds = min(args.cv_folds, int(class_counts.min()))
    if cv_folds < 2:
        raise ValueError("At least two samples per class are required for multiclass CV.")

    log_message(paths, "[2/8] Splitting and train-only KNN imputation")
    train_imputed, test_imputed = v1.split_and_impute(
        full_df=full_df,
        protein_columns=protein_columns,
        class_column=args.class_column,
        sample_id_column=args.sample_id_column,
        seed=args.seed,
        test_size=args.test_size,
        imputer_method=args.imputer,
        knn_neighbors=args.knn_neighbors,
    )

    split_df = (
        train_imputed[args.class_column]
        .value_counts()
        .rename_axis("class")
        .reset_index(name="train_n")
        .merge(
            test_imputed[args.class_column].value_counts().rename_axis("class").reset_index(name="test_n"),
            on="class",
            how="outer",
        )
    )
    split_df.to_csv(paths["tables"] / "split_distribution.csv", index=False)

    log_message(paths, "[3/8] Building train-only DEA Bonferroni candidate pool")
    bonf_sources_df, quota_sources_df = v1.collect_train_only_feature_sources(
        train_imputed=train_imputed,
        protein_columns=protein_columns,
        class_column=args.class_column,
        classes=classes,
        per_cancer_top_m=args.per_cancer_top_m,
    )
    candidate_features, candidate_df = v1.build_candidate_features(
        bonf_sources_df=bonf_sources_df,
        quota_sources_df=quota_sources_df,
        min_cancer_coverage=args.min_cancer_coverage,
    )
    coverage_df = v1.build_coverage_table(bonf_sources_df)
    candidate_df.to_csv(paths["tables"] / "candidate_features.csv", index=False)
    log_message(paths, f"Candidate features: {len(candidate_features)}")

    max_search_features = min(args.max_features, len(candidate_features))
    if max_search_features < args.min_features:
        raise ValueError(f"Only {len(candidate_features)} candidate features are available.")

    x_train = train_imputed[candidate_features].copy()
    y_train = train_imputed[args.class_column].copy()
    x_test = test_imputed[candidate_features].copy()
    y_test = test_imputed[args.class_column].copy()

    log_message(paths, "[4/8] Precomputing 10 rankers")
    rankings, ranker_table = build_rankings_v2(
        x_train=x_train,
        y_train=y_train,
        classes=classes,
        quota_sources_df=quota_sources_df,
        coverage_df=coverage_df,
        seed=args.seed,
    )
    ranker_table.loc[ranker_table["rank"] <= 100].to_csv(paths["tables"] / "ranker_top100_features.csv", index=False)

    log_message(paths, "[5/8] Running CV block tournament")
    if optuna is not None:
        cv_results_df = optuna_search(
            args=args,
            x_train=x_train,
            y_train=y_train,
            classes=classes,
            rankings=rankings,
            quota_sources_df=quota_sources_df,
            coverage_df=coverage_df,
            cv_folds=cv_folds,
            max_search_features=max_search_features,
            paths=paths,
        )
    else:
        log_message(paths, "Optuna is not installed in proteomics env; using deterministic fallback search.")
        cv_results_df = fallback_search(
            args=args,
            x_train=x_train,
            y_train=y_train,
            classes=classes,
            rankings=rankings,
            quota_sources_df=quota_sources_df,
            coverage_df=coverage_df,
            cv_folds=cv_folds,
            max_search_features=max_search_features,
            paths=paths,
        )

    log_message(paths, "[6/8] Writing CV results and selecting size-bucket finalists")
    normalize_results_df(cv_results_df).to_csv(paths["tables"] / "cv_results.csv", index=False)
    finalists_df = select_finalists(cv_results_df, args.size_buckets, args.top_per_bucket, args.top_global)
    finalists_df.to_csv(paths["tables"] / "finalists.csv", index=False)

    log_message(paths, "[7/8] Evaluating held-out test coaches")
    evaluation_candidates_df = cv_results_df if args.evaluate_all_trials else finalists_df
    test_results_df = test_finalists(
        evaluation_candidates_df,
        x_train,
        y_train,
        x_test,
        y_test,
        classes,
        args.seed,
        paths,
        coaches=COACH_NAMES,
        checkpoint_name="test_results_checkpoint.csv",
    )
    if args.hgb_coach_stage:
        hgb_candidates_df = select_hgb_coach_candidates(
            cv_results_df=cv_results_df,
            finalists_df=finalists_df,
            max_candidates=args.hgb_coach_max_candidates,
        )
    else:
        hgb_candidates_df = pd.DataFrame()
    if args.hgb_coach_stage and not hgb_candidates_df.empty:
        log_message(
            paths,
            f"[FINAL HGB] Evaluating lightweight HGB coach on {len(hgb_candidates_df)} finalists/HGB-GM candidates",
        )
        hgb_results_df = test_finalists(
            hgb_candidates_df,
            x_train,
            y_train,
            x_test,
            y_test,
            classes,
            args.seed,
            paths,
            coaches=FINAL_STAGE_COACH_NAMES,
            checkpoint_name="hgb_coach_results_checkpoint.csv",
        )
        normalize_results_df(hgb_results_df).to_csv(paths["tables"] / "hgb_coach_results.csv", index=False)
        test_results_df = pd.concat([test_results_df, hgb_results_df], ignore_index=True)
        test_results_df = test_results_df.sort_values(
            ["test_balanced_accuracy", "test_macro_f1", "test_macro_ovr_auc", "feature_count"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    normalize_results_df(test_results_df).to_csv(paths["tables"] / "test_results.csv", index=False)
    size_tradeoff_df = build_size_tradeoff(test_results_df, args.size_buckets)
    size_tradeoff_df.to_csv(paths["tables"] / "size_tradeoff_summary.csv", index=False)

    if not test_results_df.empty:
        log_message(paths, "[8/8] Saving best panel tables and figures")
        save_best_outputs(
            best_row=test_results_df.iloc[0],
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            test_imputed=test_imputed,
            sample_id_column=args.sample_id_column,
            classes=classes,
            paths=paths,
            seed=args.seed,
        )
        save_overview_figures(
            cv_results_df=cv_results_df,
            test_results_df=test_results_df,
            size_tradeoff_df=size_tradeoff_df,
            ranker_table=ranker_table,
            paths=paths,
            seed=args.seed,
        )

    manifest = {
        "input": str(args.input.resolve()),
        "output_root": str(paths["root"].resolve()),
        "seed": args.seed,
        "test_size": args.test_size,
        "imputer": args.imputer,
        "knn_neighbors": args.knn_neighbors,
        "per_cancer_top_m": args.per_cancer_top_m,
        "min_cancer_coverage": args.min_cancer_coverage,
        "min_features": args.min_features,
        "max_features": args.max_features,
        "trials_per_block": args.trials_per_block,
        "hgb_trials_per_block": args.hgb_trials_per_block,
        "expected_cv_trials": expected_cv_trials(args),
        "expected_main_test_rows_if_all_trials": expected_main_test_rows(args),
        "evaluate_all_trials": args.evaluate_all_trials,
        "hgb_gm_stage": args.hgb_gm_stage,
        "hgb_coach_stage": args.hgb_coach_stage,
        "hgb_coach_max_candidates": args.hgb_coach_max_candidates,
        "hgb_coach_candidate_count": int(len(hgb_candidates_df)),
        "cv_folds_used": cv_folds,
        "size_buckets": args.size_buckets,
        "rankers": list(RANKER_NAMES),
        "gms": list(GM_NAMES),
        "main_gms": list(MAIN_GM_NAMES),
        "late_hgb_gms": list(HGB_GM_NAMES),
        "coaches": list(COACH_NAMES),
        "final_stage_coaches": list(FINAL_STAGE_COACH_NAMES),
        "candidate_feature_count": len(candidate_features),
        "completed_cv_trials": int(len(cv_results_df)),
        "finalist_count": int(len(finalists_df)),
    }
    (paths["manifests"] / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log_message(paths, "Run completed")
    print(f"Output directory: {paths['root']}")
    print(f"Candidate features: {len(candidate_features)}")
    print(f"Completed CV trials: {len(cv_results_df)}")
    print(f"Finalists selected: {len(finalists_df)}")
    print(f"Test rows written: {len(test_results_df)}")
    if not test_results_df.empty:
        best = test_results_df.iloc[0]
        print(
            "Best v2 panel: "
            f"{best['feature_count']} features, ranker={best['ranker']}, coach={best['coach']}, "
            f"BA={best['test_balanced_accuracy']:.3f}, "
            f"macroF1={best['test_macro_f1']:.3f}, "
            f"macroOVR_AUC={best['test_macro_ovr_auc']:.3f}"
        )


if __name__ == "__main__":
    main()
