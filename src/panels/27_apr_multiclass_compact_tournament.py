#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency guard
    optuna = None

from common import SOURCE_INPUT
from run_pan_feature_multiclass import collect_train_only_feature_sources


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[2]
REVISE_ROOT = ROOT / "revise_plan"
PART_B_ROOT = REVISE_ROOT / "part_b_multiclass"
PIPELINE_SCRIPTS = ROOT / "scripts"
if str(PIPELINE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PIPELINE_SCRIPTS))

import run_12class_ovr_pipeline as base  # noqa: E402


RANKER_NAMES = (
    "LOGISTIC_COEF",
    "RF_IMPORTANCE",
    "UNIVAR_OVR_AUC",
    "OVR_BONF_QUOTA",
    "COVERAGE_IMPORTANCE",
)
GM_NAMES = ("LR_L2", "LR_EN", "RF")
COACH_NAMES = ("LR_L2", "LR_EN", "RF", "SOFT_VOTE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search compact 12-class multiclass protein panels with train-only DEA sources, "
            "multiple rankers, CV objective, and held-out test reporting."
        )
    )
    parser.add_argument("--input", type=Path, default=SOURCE_INPUT)
    parser.add_argument("--class-column", default="Cancer")
    parser.add_argument("--sample-id-column", default="Sample_ID")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--imputer", choices=sorted(base.IMPUTER_METHODS), default="knn")
    parser.add_argument("--knn-neighbors", type=int, default=base.KNN_NEIGHBORS)
    parser.add_argument("--per-cancer-top-m", type=int, default=10)
    parser.add_argument("--min-cancer-coverage", type=int, default=2)
    parser.add_argument("--min-features", type=int, default=24)
    parser.add_argument("--max-features", type=int, default=83)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--top-candidates", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=PART_B_ROOT / "compact_tournament_27apr")
    return parser.parse_args()


def output_paths(output_root: Path) -> dict[str, Path]:
    return {
        "root": output_root,
        "tables": output_root / "tables",
        "figures": output_root / "figures",
        "manifests": output_root / "manifests",
    }


def ensure_dirs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def split_and_impute(
    full_df: pd.DataFrame,
    protein_columns: list[str],
    class_column: str,
    sample_id_column: str,
    seed: int,
    test_size: float,
    imputer_method: str,
    knn_neighbors: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_idx, test_idx = train_test_split(
        full_df.index.to_numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=full_df[class_column],
    )
    train_raw = full_df.loc[train_idx].reset_index(drop=True)
    test_raw = full_df.loc[test_idx].reset_index(drop=True)

    imputer = base.build_imputer(imputer_method, knn_neighbors, seed)
    imputer.fit(train_raw[protein_columns])

    train_imputed = pd.concat(
        [
            train_raw[[sample_id_column, class_column]].reset_index(drop=True),
            pd.DataFrame(imputer.transform(train_raw[protein_columns]), columns=protein_columns),
        ],
        axis=1,
    )
    test_imputed = pd.concat(
        [
            test_raw[[sample_id_column, class_column]].reset_index(drop=True),
            pd.DataFrame(imputer.transform(test_raw[protein_columns]), columns=protein_columns),
        ],
        axis=1,
    )
    return train_imputed, test_imputed


def build_coverage_table(bonf_sources_df: pd.DataFrame) -> pd.DataFrame:
    if bonf_sources_df.empty:
        return pd.DataFrame(columns=["protein", "cancer_count", "cancers"])
    return (
        bonf_sources_df.groupby("protein", as_index=False)
        .agg(
            cancer_count=("target_class", "nunique"),
            cancers=("target_class", lambda values: ";".join(sorted(set(values)))),
        )
        .sort_values(["cancer_count", "protein"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_candidate_features(
    bonf_sources_df: pd.DataFrame,
    quota_sources_df: pd.DataFrame,
    min_cancer_coverage: int,
) -> tuple[list[str], pd.DataFrame]:
    coverage_df = build_coverage_table(bonf_sources_df)
    quota_proteins = set(quota_sources_df["protein"].tolist()) if not quota_sources_df.empty else set()
    coverage_proteins = set(
        coverage_df.loc[coverage_df["cancer_count"] >= min_cancer_coverage, "protein"].tolist()
    )
    candidate_features = sorted(quota_proteins | coverage_proteins)
    candidate_df = coverage_df.loc[coverage_df["protein"].isin(candidate_features)].copy()
    missing_from_coverage = sorted(set(candidate_features) - set(candidate_df["protein"].tolist()))
    if missing_from_coverage:
        candidate_df = pd.concat(
            [
                candidate_df,
                pd.DataFrame(
                    {
                        "protein": missing_from_coverage,
                        "cancer_count": 0,
                        "cancers": "",
                    }
                ),
            ],
            ignore_index=True,
        )
    candidate_df = candidate_df.sort_values(["cancer_count", "protein"], ascending=[False, True]).reset_index(drop=True)
    candidate_df["candidate_rank_by_coverage"] = np.arange(1, len(candidate_df) + 1)
    return candidate_features, candidate_df


def normalize_series(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    min_value = float(values.min())
    max_value = float(values.max())
    if math.isclose(max_value, min_value):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - min_value) / (max_value - min_value)


def rank_logistic_coef(x_train: pd.DataFrame, y_train: pd.Series, seed: int) -> pd.DataFrame:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    coef = model.named_steps["model"].coef_
    importance = np.mean(np.abs(coef), axis=0)
    ranking_df = pd.DataFrame({"protein": x_train.columns, "importance": importance})
    return ranking_df.sort_values(["importance", "protein"], ascending=[False, True]).reset_index(drop=True)


def rank_rf_importance(x_train: pd.DataFrame, y_train: pd.Series, seed: int) -> pd.DataFrame:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    ranking_df = pd.DataFrame({"protein": x_train.columns, "importance": model.feature_importances_})
    return ranking_df.sort_values(["importance", "protein"], ascending=[False, True]).reset_index(drop=True)


def safe_binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        return 0.5
    return max(auc, 1.0 - auc)


def rank_univar_ovr_auc(x_train: pd.DataFrame, y_train: pd.Series, classes: list[str]) -> pd.DataFrame:
    y_values = y_train.to_numpy()
    rows: list[dict[str, object]] = []
    for protein in x_train.columns:
        scores = x_train[protein].to_numpy()
        aucs = [safe_binary_auc((y_values == label).astype(int), scores) for label in classes]
        rows.append(
            {
                "protein": protein,
                "importance": float(np.mean(aucs)),
                "max_ovr_auc": float(np.max(aucs)),
                "min_ovr_auc": float(np.min(aucs)),
            }
        )
    return pd.DataFrame(rows).sort_values(["importance", "max_ovr_auc", "protein"], ascending=[False, False, True]).reset_index(drop=True)


def rank_ovr_bonf_quota(quota_sources_df: pd.DataFrame, classes: list[str], fallback: Iterable[str]) -> pd.DataFrame:
    selected: list[str] = []
    seen: set[str] = set()
    if not quota_sources_df.empty:
        sorted_quota = quota_sources_df.sort_values(["target_class", "p_adjusted_bonf", "p_value", "protein"])
        per_class = {
            label: sorted_quota.loc[sorted_quota["target_class"] == label, "protein"].tolist()
            for label in classes
        }
        max_len = max((len(values) for values in per_class.values()), default=0)
        for index in range(max_len):
            for label in classes:
                values = per_class.get(label, [])
                if index >= len(values):
                    continue
                protein = values[index]
                if protein not in seen:
                    selected.append(protein)
                    seen.add(protein)
    for protein in fallback:
        if protein not in seen:
            selected.append(protein)
            seen.add(protein)
    ranking_df = pd.DataFrame({"protein": selected, "importance": np.arange(len(selected), 0, -1)})
    return ranking_df


def build_rankings(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    classes: list[str],
    quota_sources_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    seed: int,
) -> dict[str, list[str]]:
    logistic_df = rank_logistic_coef(x_train, y_train, seed)
    rf_df = rank_rf_importance(x_train, y_train, seed)
    univar_df = rank_univar_ovr_auc(x_train, y_train, classes)
    quota_df = rank_ovr_bonf_quota(quota_sources_df, classes, logistic_df["protein"].tolist())

    coverage_importance_df = logistic_df.merge(
        coverage_df[["protein", "cancer_count"]] if not coverage_df.empty else pd.DataFrame(columns=["protein", "cancer_count"]),
        on="protein",
        how="left",
    )
    coverage_importance_df["cancer_count"] = coverage_importance_df["cancer_count"].fillna(0)
    coverage_importance_df["coverage_score"] = normalize_series(coverage_importance_df["cancer_count"])
    coverage_importance_df["importance_score"] = normalize_series(coverage_importance_df["importance"])
    coverage_importance_df["hybrid_score"] = (
        0.55 * coverage_importance_df["importance_score"] + 0.45 * coverage_importance_df["coverage_score"]
    )
    coverage_importance_df = coverage_importance_df.sort_values(
        ["hybrid_score", "cancer_count", "importance", "protein"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    ranking_tables = {
        "LOGISTIC_COEF": logistic_df,
        "RF_IMPORTANCE": rf_df,
        "UNIVAR_OVR_AUC": univar_df,
        "OVR_BONF_QUOTA": quota_df,
        "COVERAGE_IMPORTANCE": coverage_importance_df.rename(columns={"hybrid_score": "importance"}),
    }
    rankings: dict[str, list[str]] = {}
    for name, table in ranking_tables.items():
        deduped = table.drop_duplicates("protein", keep="first").reset_index(drop=True)
        deduped["ranker"] = name
        deduped["rank"] = np.arange(1, len(deduped) + 1)
        rankings[name] = deduped["protein"].tolist()
    return rankings


def save_rankings(
    rankings: dict[str, list[str]],
    output_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for ranker, proteins in rankings.items():
        for rank, protein in enumerate(proteins, start=1):
            rows.append({"ranker": ranker, "rank": rank, "protein": protein})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def add_unique(target: list[str], seen: set[str], proteins: Iterable[str], limit: int) -> None:
    for protein in proteins:
        if len(target) >= limit:
            return
        if protein in seen:
            continue
        target.append(protein)
        seen.add(protein)


def quota_panel_order(quota_sources_df: pd.DataFrame, classes: list[str], min_per_class: int) -> list[str]:
    if min_per_class <= 0 or quota_sources_df.empty:
        return []
    sorted_quota = quota_sources_df.sort_values(["target_class", "p_adjusted_bonf", "p_value", "protein"])
    per_class = {
        label: sorted_quota.loc[sorted_quota["target_class"] == label, "protein"].tolist()
        for label in classes
    }
    selected: list[str] = []
    for quota_index in range(min_per_class):
        for label in classes:
            proteins = per_class.get(label, [])
            if quota_index < len(proteins):
                selected.append(proteins[quota_index])
    return selected


def shared_feature_order(coverage_df: pd.DataFrame, min_coverage: int = 2) -> list[str]:
    if coverage_df.empty:
        return []
    shared_df = coverage_df.loc[coverage_df["cancer_count"] >= min_coverage].copy()
    shared_df = shared_df.sort_values(["cancer_count", "protein"], ascending=[False, True])
    return shared_df["protein"].tolist()


def build_panel(
    ranker_name: str,
    n_total: int,
    min_per_class: int,
    n_shared: int,
    rankings: dict[str, list[str]],
    quota_sources_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    classes: list[str],
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    add_unique(selected, seen, quota_panel_order(quota_sources_df, classes, min_per_class), n_total)
    add_unique(selected, seen, shared_feature_order(coverage_df), min(n_total, len(selected) + n_shared))
    add_unique(selected, seen, rankings[ranker_name], n_total)
    return selected


def make_estimator(model_name: str, seed: int, params: dict[str, object] | None = None):
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
                        max_iter=5000,
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
                        max_iter=6000,
                        n_jobs=-1,
                        penalty="elasticnet",
                        random_state=seed,
                        solver="saga",
                    ),
                ),
            ]
        )
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "SOFT_VOTE":
        return VotingClassifier(
            estimators=[
                ("lr_l2", make_estimator("LR_L2", seed, {"C": params.get("lr_l2_C", 0.1)})),
                (
                    "lr_en",
                    make_estimator(
                        "LR_EN",
                        seed,
                        {"C": params.get("lr_en_C", 0.1), "l1_ratio": params.get("lr_en_l1_ratio", 0.5)},
                    ),
                ),
                (
                    "rf",
                    make_estimator(
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
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def suggest_model_params(trial: optuna.Trial, model_name: str) -> dict[str, object]:
    if model_name == "LR_L2":
        return {"C": trial.suggest_categorical("lr_l2_C", [0.01, 0.03, 0.1, 0.3, 1.0, 3.0])}
    if model_name == "LR_EN":
        return {
            "C": trial.suggest_categorical("lr_en_C", [0.01, 0.03, 0.1, 0.3, 1.0]),
            "l1_ratio": trial.suggest_categorical("lr_en_l1_ratio", [0.2, 0.5, 0.8]),
        }
    if model_name == "RF":
        return {
            "n_estimators": trial.suggest_categorical("rf_n_estimators", [200, 300]),
            "max_depth": trial.suggest_categorical("rf_max_depth", [6, 10, None]),
            "min_samples_leaf": trial.suggest_categorical("rf_min_samples_leaf", [1, 2, 4]),
        }
    return {}


def random_model_params(rng: np.random.Generator, model_name: str) -> dict[str, object]:
    if model_name == "LR_L2":
        return {"C": float(rng.choice([0.01, 0.03, 0.1, 0.3, 1.0, 3.0]))}
    if model_name == "LR_EN":
        return {
            "C": float(rng.choice([0.01, 0.03, 0.1, 0.3, 1.0])),
            "l1_ratio": float(rng.choice([0.2, 0.5, 0.8])),
        }
    if model_name == "RF":
        max_depth_choice = rng.choice([6, 10, None])
        return {
            "n_estimators": int(rng.choice([200, 300])),
            "max_depth": int(max_depth_choice) if max_depth_choice is not None else None,
            "min_samples_leaf": int(rng.choice([1, 2, 4])),
        }
    return {}


def structural_seed_configs(min_features: int, max_features: int) -> list[dict[str, Any]]:
    seed_sizes = sorted({min_features, min(max_features, 36), min(max_features, 50), min(max_features, 60), max_features})
    configs: list[dict[str, Any]] = []
    for ranker in RANKER_NAMES:
        for gm in ("LR_L2", "LR_EN"):
            for size in seed_sizes:
                configs.append(
                    {
                        "ranker": ranker,
                        "gm": gm,
                        "n_total": size,
                        "min_per_class": min(2, max(0, size // 12)),
                        "n_shared": min(12, size),
                        "model_params": {"C": 0.1} if gm == "LR_L2" else {"C": 0.1, "l1_ratio": 0.5},
                    }
                )
    return configs


def random_trial_configs(
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
        max_quota = max(0, min(5, n_total // max(n_classes, 1)))
        gm_name = str(rng.choice(GM_NAMES))
        configs.append(
            {
                "ranker": str(rng.choice(RANKER_NAMES)),
                "gm": gm_name,
                "n_total": n_total,
                "min_per_class": int(rng.integers(0, max_quota + 1)),
                "n_shared": int(rng.integers(0, min(40, n_total) + 1)),
                "model_params": random_model_params(rng, gm_name),
            }
        )
    return configs


def aligned_probabilities(estimator, x: pd.DataFrame, labels: list[str]) -> np.ndarray:
    probabilities = estimator.predict_proba(x)
    estimator_classes = list(estimator.classes_)
    aligned = np.zeros((len(x), len(labels)), dtype=float)
    for index, label in enumerate(labels):
        if label in estimator_classes:
            aligned[:, index] = probabilities[:, estimator_classes.index(label)]
    return aligned


def metric_bundle(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_score: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    recalls = recall_score(y_true_array, y_pred_array, labels=labels, average=None, zero_division=0)
    try:
        macro_ovr_auc = float(
            roc_auc_score(y_true_array, y_score, labels=labels, multi_class="ovr", average="macro")
        )
    except ValueError:
        macro_ovr_auc = math.nan
    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred_array)),
        "macro_f1": float(f1_score(y_true_array, y_pred_array, labels=labels, average="macro", zero_division=0)),
        "macro_ovr_auc": macro_ovr_auc,
        "min_class_recall": float(np.min(recalls)) if len(recalls) else math.nan,
    }


def cv_evaluate(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    features: list[str],
    labels: list[str],
    model_name: str,
    model_params: dict[str, object],
    cv_folds: int,
    seed: int,
) -> dict[str, float]:
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    rows: list[dict[str, float]] = []
    for fold_index, (fit_idx, val_idx) in enumerate(splitter.split(x_train[features], y_train), start=1):
        estimator = make_estimator(model_name, seed + fold_index, model_params)
        x_fit = x_train.iloc[fit_idx][features]
        x_val = x_train.iloc[val_idx][features]
        y_fit = y_train.iloc[fit_idx]
        y_val = y_train.iloc[val_idx]
        estimator.fit(x_fit, y_fit)
        y_pred = estimator.predict(x_val)
        y_score = aligned_probabilities(estimator, x_val, labels)
        rows.append(metric_bundle(y_val, y_pred, y_score, labels))
    cv_df = pd.DataFrame(rows)
    summary = {f"cv_{column}_mean": float(cv_df[column].mean()) for column in cv_df.columns}
    summary.update({f"cv_{column}_std": float(cv_df[column].std(ddof=0)) for column in cv_df.columns})
    return summary


def objective_score(metrics: dict[str, float], feature_count: int, max_features: int) -> float:
    macro_ovr_auc = 0.0 if math.isnan(metrics["cv_macro_ovr_auc_mean"]) else metrics["cv_macro_ovr_auc_mean"]
    min_class_recall = 0.0 if math.isnan(metrics["cv_min_class_recall_mean"]) else metrics["cv_min_class_recall_mean"]
    return float(
        0.30 * metrics["cv_balanced_accuracy_mean"]
        + 0.30 * metrics["cv_macro_f1_mean"]
        + 0.25 * macro_ovr_auc
        + 0.15 * min_class_recall
        - 0.05 * (feature_count / max_features)
        - 0.10 * metrics["cv_macro_f1_std"]
    )


def enqueue_seed_trials(study: optuna.Study, min_features: int, max_features: int) -> None:
    seed_sizes = sorted({min_features, min(max_features, 36), min(max_features, 50), min(max_features, 60), max_features})
    for ranker in RANKER_NAMES:
        for gm in ("LR_L2", "LR_EN"):
            for size in seed_sizes:
                study.enqueue_trial(
                    {
                        "ranker": ranker,
                        "gm": gm,
                        "n_total": size,
                        "min_per_class": min(2, max(0, size // 12)),
                        "n_shared": min(12, size),
                    }
                )


def trial_records(study: optuna.Study) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row: dict[str, object] = {
            "trial": trial.number,
            "objective_score": trial.value,
            **trial.params,
            **trial.user_attrs,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def select_top_candidates(cv_results_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if cv_results_df.empty:
        return cv_results_df
    candidate_df = cv_results_df.copy()
    candidate_df["feature_signature"] = candidate_df["features"]
    candidate_df = candidate_df.sort_values("objective_score", ascending=False)
    candidate_df = candidate_df.drop_duplicates("feature_signature", keep="first")
    return candidate_df.head(top_n).reset_index(drop=True)


def evaluate_on_test(
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
    estimator = make_estimator(coach_name, seed, {})
    estimator.fit(x_train[features], y_train)
    y_pred = estimator.predict(x_test[features])
    y_score = aligned_probabilities(estimator, x_test[features], labels)
    metrics = metric_bundle(y_test, y_pred, y_score, labels)
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


def classification_report_df(y_true: pd.Series, y_pred: np.ndarray, labels: list[str]) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})


def ovr_auc_df(y_true: pd.Series, y_score: np.ndarray, labels: list[str]) -> pd.DataFrame:
    y_values = y_true.to_numpy()
    rows: list[dict[str, object]] = []
    for index, label in enumerate(labels):
        binary = (y_values == label).astype(int)
        auc = float(roc_auc_score(binary, y_score[:, index])) if len(np.unique(binary)) >= 2 else math.nan
        rows.append({"label": label, "ovr_auc": auc})
    return pd.DataFrame(rows)


def save_tradeoff_figure(results_df: pd.DataFrame, output_path: Path, prefix: str) -> None:
    if results_df.empty:
        return
    figure, ax = base.plt.subplots(figsize=(10, 5.5))
    if prefix == "cv":
        y_columns = ["cv_balanced_accuracy_mean", "cv_macro_f1_mean", "cv_macro_ovr_auc_mean"]
        y_labels = ["CV balanced accuracy", "CV macro F1", "CV macro OVR AUC"]
    else:
        y_columns = ["test_balanced_accuracy", "test_macro_f1", "test_macro_ovr_auc"]
        y_labels = ["Test balanced accuracy", "Test macro F1", "Test macro OVR AUC"]
    for column, label in zip(y_columns, y_labels):
        if column not in results_df.columns:
            continue
        sns.scatterplot(data=results_df, x="feature_count", y=column, ax=ax, label=label, s=55)
    ax.axvline(83, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title(f"Compact tournament {prefix.upper()} performance trade-off")
    ax.set_xlabel("Feature count")
    ax.set_ylabel("Metric")
    ax.legend(loc="best", frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    base.plt.close(figure)


def save_confusion_figure(matrix: pd.DataFrame, output_path: Path) -> None:
    figure, ax = base.plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.3, linecolor="white", ax=ax)
    ax.set_title("Best compact multiclass panel confusion matrix")
    ax.set_xlabel("Predicted cancer")
    ax.set_ylabel("True cancer")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    base.plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.min_features < 1:
        raise ValueError("--min-features must be >= 1")
    if args.max_features < args.min_features:
        raise ValueError("--max-features must be >= --min-features")

    paths = output_paths(args.output_dir)
    ensure_dirs(paths)

    full_df = pd.read_csv(args.input)
    protein_columns = base.protein_columns_from_df(full_df, args.class_column, args.sample_id_column)
    classes = sorted(full_df[args.class_column].dropna().unique().tolist())
    class_counts = full_df[args.class_column].value_counts()
    cv_folds = min(args.cv_folds, int(class_counts.min()))
    if cv_folds < 2:
        raise ValueError("At least two samples per class are required for multiclass CV.")

    train_imputed, test_imputed = split_and_impute(
        full_df=full_df,
        protein_columns=protein_columns,
        class_column=args.class_column,
        sample_id_column=args.sample_id_column,
        seed=args.seed,
        test_size=args.test_size,
        imputer_method=args.imputer,
        knn_neighbors=args.knn_neighbors,
    )

    train_distribution = train_imputed[args.class_column].value_counts().rename_axis("class").reset_index(name="train_n")
    test_distribution = test_imputed[args.class_column].value_counts().rename_axis("class").reset_index(name="test_n")
    train_distribution.merge(test_distribution, on="class", how="outer").to_csv(
        paths["tables"] / "compact_tournament_split_distribution.csv",
        index=False,
    )

    bonf_sources_df, quota_sources_df = collect_train_only_feature_sources(
        train_imputed=train_imputed,
        protein_columns=protein_columns,
        class_column=args.class_column,
        classes=classes,
        per_cancer_top_m=args.per_cancer_top_m,
    )
    bonf_sources_df.to_csv(paths["tables"] / "compact_tournament_bonf_sources.csv", index=False)
    quota_sources_df.to_csv(paths["tables"] / "compact_tournament_quota_sources.csv", index=False)

    candidate_features, candidate_df = build_candidate_features(
        bonf_sources_df=bonf_sources_df,
        quota_sources_df=quota_sources_df,
        min_cancer_coverage=args.min_cancer_coverage,
    )
    if not candidate_features:
        raise ValueError("No candidate features were found from Bonferroni/quota sources.")
    coverage_df = build_coverage_table(bonf_sources_df)
    candidate_df.to_csv(paths["tables"] / "compact_tournament_candidate_features.csv", index=False)
    coverage_df.to_csv(paths["tables"] / "compact_tournament_feature_coverage.csv", index=False)

    x_train = train_imputed[candidate_features].copy()
    y_train = train_imputed[args.class_column].copy()
    x_test = test_imputed[candidate_features].copy()
    y_test = test_imputed[args.class_column].copy()

    rankings = build_rankings(
        x_train=x_train,
        y_train=y_train,
        classes=classes,
        quota_sources_df=quota_sources_df,
        coverage_df=coverage_df,
        seed=args.seed,
    )
    save_rankings(rankings, paths["tables"] / "compact_tournament_ranker_outputs.csv")

    max_search_features = min(args.max_features, len(candidate_features))
    if max_search_features < args.min_features:
        raise ValueError(
            f"Only {len(candidate_features)} candidate features are available, "
            f"which is less than --min-features={args.min_features}."
        )

    def evaluate_config(config: dict[str, Any], trial_number: int) -> dict[str, object] | None:
        ranker_name = str(config["ranker"])
        gm_name = str(config["gm"])
        n_total = int(config["n_total"])
        min_per_class = int(config["min_per_class"])
        n_shared = int(config["n_shared"])
        model_params = dict(config.get("model_params", {}))
        features = build_panel(
            ranker_name=ranker_name,
            n_total=n_total,
            min_per_class=min_per_class,
            n_shared=n_shared,
            rankings=rankings,
            quota_sources_df=quota_sources_df,
            coverage_df=coverage_df,
            classes=classes,
        )
        if len(features) < args.min_features:
            return None
        metrics = cv_evaluate(
            x_train=x_train,
            y_train=y_train,
            features=features,
            labels=classes,
            model_name=gm_name,
            model_params=model_params,
            cv_folds=cv_folds,
            seed=args.seed,
        )
        score = objective_score(metrics, len(features), args.max_features)
        return {
            "trial": trial_number,
            "objective_score": score,
            "ranker": ranker_name,
            "gm": gm_name,
            "n_total": n_total,
            "min_per_class": min_per_class,
            "n_shared": n_shared,
            "feature_count": len(features),
            "features": ";".join(features),
            "model_params": json.dumps(model_params, sort_keys=True),
            **metrics,
        }

    if optuna is not None:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
        enqueue_seed_trials(study, args.min_features, max_search_features)

        def objective(trial: optuna.Trial) -> float:
            ranker_name = trial.suggest_categorical("ranker", list(RANKER_NAMES))
            gm_name = trial.suggest_categorical("gm", list(GM_NAMES))
            n_total = trial.suggest_int("n_total", args.min_features, max_search_features)
            max_quota = max(0, min(5, n_total // max(len(classes), 1)))
            min_per_class = trial.suggest_int("min_per_class", 0, max_quota)
            n_shared = trial.suggest_int("n_shared", 0, min(40, n_total))
            model_params = suggest_model_params(trial, gm_name)
            row = evaluate_config(
                {
                    "ranker": ranker_name,
                    "gm": gm_name,
                    "n_total": n_total,
                    "min_per_class": min_per_class,
                    "n_shared": n_shared,
                    "model_params": model_params,
                },
                trial.number,
            )
            if row is None:
                raise optuna.TrialPruned()
            for key, value in row.items():
                if key not in {"trial", "objective_score", "ranker", "gm", "n_total", "min_per_class", "n_shared"}:
                    trial.set_user_attr(key, value)
            return float(row["objective_score"])

        study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
        cv_results_df = trial_records(study)
    else:
        print("Optuna is not installed; using deterministic random/seeded tournament fallback.")
        configs = structural_seed_configs(args.min_features, max_search_features)
        remaining = max(0, args.trials - len(configs))
        configs.extend(random_trial_configs(remaining, args.min_features, max_search_features, len(classes), args.seed))
        rows: list[dict[str, object]] = []
        for trial_number, config in enumerate(configs[: args.trials]):
            row = evaluate_config(config, trial_number)
            if row is not None:
                rows.append(row)
        cv_results_df = pd.DataFrame(rows)
    cv_results_df.to_csv(paths["tables"] / "compact_tournament_cv_results.csv", index=False)
    save_tradeoff_figure(cv_results_df, paths["figures"] / "compact_tournament_cv_tradeoff.png", "cv")

    top_candidates_df = select_top_candidates(cv_results_df, args.top_candidates)
    top_candidates_df.to_csv(paths["tables"] / "compact_tournament_top_candidates.csv", index=False)

    test_rows: list[dict[str, object]] = []
    prediction_payloads: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for _, candidate_row in top_candidates_df.iterrows():
        for coach_name in COACH_NAMES:
            row, y_pred, y_score = evaluate_on_test(
                candidate_row=candidate_row,
                coach_name=coach_name,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                labels=classes,
                seed=args.seed,
            )
            test_rows.append(row)
            prediction_payloads[(row["source_trial"], row["coach"])] = (y_pred, y_score)

    test_results_df = pd.DataFrame(test_rows)
    if not test_results_df.empty:
        test_results_df = test_results_df.sort_values(
            ["test_balanced_accuracy", "test_macro_f1", "test_macro_ovr_auc", "feature_count"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    test_results_df.to_csv(paths["tables"] / "compact_tournament_test_results.csv", index=False)
    save_tradeoff_figure(test_results_df, paths["figures"] / "compact_tournament_test_tradeoff.png", "test")

    if not test_results_df.empty:
        best_row = test_results_df.iloc[0]
        best_features = str(best_row["features"]).split(";")
        best_key = (int(best_row["source_trial"]), str(best_row["coach"]))
        best_pred, best_score = prediction_payloads[best_key]

        pd.DataFrame(
            {
                "rank": np.arange(1, len(best_features) + 1),
                "protein": best_features,
            }
        ).to_csv(paths["tables"] / "compact_tournament_best_panel_features.csv", index=False)

        predictions_df = pd.DataFrame(
            {
                args.sample_id_column: test_imputed[args.sample_id_column],
                "true_class": y_test,
                "predicted_class": best_pred,
            }
        )
        for index, label in enumerate(classes):
            predictions_df[f"score_{label}"] = best_score[:, index]
        predictions_df.to_csv(paths["tables"] / "compact_tournament_best_panel_test_predictions.csv", index=False)

        classification_report_df(y_test, best_pred, classes).to_csv(
            paths["tables"] / "compact_tournament_best_panel_per_class_metrics.csv",
            index=False,
        )
        ovr_auc_df(y_test, best_score, classes).to_csv(
            paths["tables"] / "compact_tournament_best_panel_ovr_auc.csv",
            index=False,
        )
        confusion = pd.DataFrame(confusion_matrix(y_test, best_pred, labels=classes), index=classes, columns=classes)
        confusion.to_csv(paths["tables"] / "compact_tournament_best_panel_confusion_matrix.csv")
        save_confusion_figure(confusion, paths["figures"] / "compact_tournament_best_confusion_matrix.png")

        best_row.to_frame().T.to_csv(paths["tables"] / "compact_tournament_best_panel_summary.csv", index=False)

    manifest = {
        "input": str(args.input.resolve()),
        "output_root": str(paths["root"].resolve()),
        "class_column": args.class_column,
        "sample_id_column": args.sample_id_column,
        "seed": args.seed,
        "test_size": args.test_size,
        "imputer": args.imputer,
        "knn_neighbors": args.knn_neighbors,
        "per_cancer_top_m": args.per_cancer_top_m,
        "min_cancer_coverage": args.min_cancer_coverage,
        "min_features": args.min_features,
        "max_features": args.max_features,
        "trials": args.trials,
        "cv_folds_requested": args.cv_folds,
        "cv_folds_used": cv_folds,
        "classes": classes,
        "candidate_feature_count": len(candidate_features),
    }
    (paths["manifests"] / "compact_tournament_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Output directory: {paths['root']}")
    print(f"Candidate features: {len(candidate_features)}")
    print(f"Completed CV trials: {len(cv_results_df)}")
    if not test_results_df.empty:
        best = test_results_df.iloc[0]
        print(
            "Best compact panel: "
            f"{best['feature_count']} features, coach={best['coach']}, "
            f"BA={best['test_balanced_accuracy']:.3f}, "
            f"macroF1={best['test_macro_f1']:.3f}, "
            f"macroOVR_AUC={best['test_macro_ovr_auc']:.3f}"
        )


if __name__ == "__main__":
    main()
