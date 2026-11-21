from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from run_dea import IMPUTER_METHODS, KNN_NEIGHBORS, RANDOM_SEED, TEST_SIZE, build_imputer, differential_expression
from run_enrichment import (
    DEFAULT_DATABASES,
    collapse_to_genes,
    load_or_build_mapping,
    load_or_fetch_library,
    ora,
    preranked_gsea,
    save_dotplot,
    save_preranked_barplot,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "filtered_pancancer_data.csv"
DEFAULT_OUTPUT = ROOT / "result12classes"
DEFAULT_PANEL_SIZES = (1, 3, 10, 18)
TOP_FEATURE_CAP = 18
FALLBACK_FEATURE_COUNT = 50

MODEL_NAMES = ("Random Forest", "XGBoost", "Lasso Logistic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 12-class one-vs-rest proteomics pipeline from a pre-filtered pancancer dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--class-column", default="Cancer")
    parser.add_argument("--sample-id-column", default="Sample_ID")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--imputer", choices=sorted(IMPUTER_METHODS), default="knn")
    parser.add_argument("--knn-neighbors", type=int, default=KNN_NEIGHBORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-sizes", nargs="+", type=int, default=list(DEFAULT_PANEL_SIZES))
    parser.add_argument("--run-enrichment", action="store_true")
    parser.add_argument("--enrichment-databases", nargs="+", default=DEFAULT_DATABASES, choices=["go_bp", "reactome", "kegg"])
    parser.add_argument("--top-terms", type=int, default=15)
    parser.add_argument("--gsea-permutations", type=int, default=250)
    parser.add_argument("--run-robustness", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--robustness-repeats", type=int, default=0)
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def protein_columns_from_df(df: pd.DataFrame, class_column: str, sample_id_column: str) -> list[str]:
    metadata = {sample_id_column, class_column, "Label", "protein_count"}
    return [column for column in df.columns if column not in metadata]


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    paths = {
        "root": output_root,
        "data": output_root / "data",
        "dea": output_root / "tables" / "dea",
        "model": output_root / "tables" / "model_summary",
        "robustness": output_root / "tables" / "robustness",
        "enrichment": output_root / "tables" / "enrichment",
        "figures": output_root / "figures",
        "robustness_figures": output_root / "figures" / "robustness",
        "enrichment_figures": output_root / "figures" / "enrichment",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def resolve_split_sizes(n_samples: int, test_size: float) -> tuple[int, int]:
    if not 0 < test_size < 1:
        raise ValueError(f"`test_size` must be between 0 and 1. Received: {test_size}")
    n_test = math.ceil(test_size * n_samples)
    n_train = n_samples - n_test
    return n_train, n_test


def validate_stratification(df: pd.DataFrame, class_column: str, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    counts = df[class_column].value_counts()
    n_samples = len(df)
    n_classes = int(counts.shape[0])
    n_train, n_test = resolve_split_sizes(n_samples, test_size)

    too_rare = counts[counts < 2]
    if not too_rare.empty:
        details = "; ".join(
            f"{index}: total_samples={value} -> cannot place at least one sample in both train and test"
            for index, value in too_rare.items()
        )
        raise ValueError(
            "Invalid class-balanced stratified train/test split across 12 classes. "
            f"With test_size={test_size}, the following classes are too rare to appear in both partitions: {details}. "
            "Each class needs at least 2 samples overall. Increase the number of samples for those classes or revise the class set."
        )

    partition_errors: list[str] = []
    if n_test < n_classes:
        partition_errors.append(
            f"test partition would contain {n_test} samples for {n_classes} classes, so at least {n_classes - n_test} classes would be absent from test"
        )
    if n_train < n_classes:
        partition_errors.append(
            f"train partition would contain {n_train} samples for {n_classes} classes, so at least {n_classes - n_train} classes would be absent from train"
        )
    if partition_errors:
        raise ValueError(
            "Invalid class-balanced stratified train/test split across 12 classes. "
            f"Dataset size={n_samples}, test_size={test_size}, resolved_train_size={n_train}, resolved_test_size={n_test}. "
            + " ; ".join(partition_errors)
            + ". Reduce `test_size`, increase the dataset size, or merge/expand rare classes."
        )

    try:
        train_idx, test_idx = train_test_split(
            df.index.to_numpy(),
            test_size=test_size,
            random_state=seed,
            stratify=df[class_column],
        )
    except ValueError as exc:
        class_details = "; ".join(f"{index}: total_samples={value}" for index, value in counts.sort_index().items())
        raise ValueError(
            "Unable to execute class-balanced stratified train/test split across 12 classes. "
            f"Dataset size={n_samples}, test_size={test_size}, resolved_train_size={n_train}, resolved_test_size={n_test}. "
            f"Per-class counts: {class_details}. "
            "Adjust `test_size` or the class composition so every class can be represented in both partitions."
        ) from exc

    train_counts = df.loc[train_idx, class_column].value_counts()
    test_counts = df.loc[test_idx, class_column].value_counts()
    missing_classes: list[str] = []
    for class_name, total_count in counts.sort_index().items():
        train_count = int(train_counts.get(class_name, 0))
        test_count = int(test_counts.get(class_name, 0))
        if train_count == 0 or test_count == 0:
            missing_classes.append(
                f"{class_name}: total_samples={total_count}, train_samples={train_count}, test_samples={test_count}"
            )

    if missing_classes:
        raise ValueError(
            "The current class-balanced stratified train/test split would leave some classes absent from one partition. "
            f"test_size={test_size}. Violating classes: {'; '.join(missing_classes)}. "
            "Increase the number of samples for those classes or adjust `test_size`."
        )

    return train_idx, test_idx


def split_and_impute_multiclass(
    df: pd.DataFrame,
    protein_columns: list[str],
    class_column: str,
    sample_id_column: str,
    seed: int,
    test_size: float,
    imputer_method: str,
    knn_neighbors: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_columns = [column for column in [sample_id_column, class_column] if column in df.columns]
    metadata = df[metadata_columns].copy()
    feature_matrix = df[protein_columns].copy()
    classes = df[class_column].copy()

    train_idx, test_idx = validate_stratification(df, class_column, test_size, seed)
    x_train = feature_matrix.loc[train_idx]
    x_test = feature_matrix.loc[test_idx]
    y_train = classes.loc[train_idx]
    y_test = classes.loc[test_idx]
    meta_train = metadata.loc[train_idx]
    meta_test = metadata.loc[test_idx]

    imputer = build_imputer(imputer_method, knn_neighbors, seed)
    imputer.fit(x_train)

    x_train_imputed = pd.DataFrame(imputer.transform(x_train), columns=protein_columns, index=x_train.index)
    x_test_imputed = pd.DataFrame(imputer.transform(x_test), columns=protein_columns, index=x_test.index)

    train_df = pd.concat(
        [
            meta_train.reset_index(drop=True),
            x_train_imputed.reset_index(drop=True),
        ],
        axis=1,
    )
    test_df = pd.concat(
        [
            meta_test.reset_index(drop=True),
            x_test_imputed.reset_index(drop=True),
        ],
        axis=1,
    )
    train_df[class_column] = y_train.reset_index(drop=True)
    test_df[class_column] = y_test.reset_index(drop=True)
    return train_df, test_df


def class_distribution_table(df: pd.DataFrame, class_column: str, partition: str) -> pd.DataFrame:
    summary = df[class_column].value_counts(dropna=False).rename_axis(class_column).reset_index(name="n_samples")
    summary["partition"] = partition
    summary["proportion"] = summary["n_samples"] / summary["n_samples"].sum()
    return summary[[class_column, "partition", "n_samples", "proportion"]]


def prepare_binary_partition(
    base_train_df: pd.DataFrame,
    base_test_df: pd.DataFrame,
    class_column: str,
    target_class: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = base_train_df.copy()
    test_df = base_test_df.copy()
    train_df["Label"] = (train_df[class_column] == target_class).astype(int)
    test_df["Label"] = (test_df[class_column] == target_class).astype(int)
    return train_df, test_df


def correction_columns(correction: str) -> tuple[str, str]:
    if correction == "bh":
        return "significant_bh", "p_adjusted_bh"
    return "significant_bonf", "p_adjusted_bonf"


def dynamic_cv_folds(y: pd.Series, max_folds: int = 5) -> int | None:
    class_counts = y.value_counts()
    if class_counts.empty:
        return None
    min_class = int(class_counts.min())
    if min_class < 2:
        return None
    return max(2, min(max_folds, min_class))


def compute_binary_metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    npv = tn / (tn + fn) if (tn + fn) else math.nan
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "npv": npv,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "average_precision": average_precision_score(y_true, y_score),
        "brier_score": brier_score_loss(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score, bin_edges[1:-1], right=True)
    ece = 0.0
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        bin_confidence = float(np.mean(y_score[mask]))
        bin_accuracy = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / len(y_true)) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def train_youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_index = int(np.argmax(tpr - fpr))
    return float(thresholds[best_index])


def bootstrap_panel_intervals(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    metrics = {
        key: []
        for key in [
            "auc",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "npv",
            "mcc",
            "average_precision",
            "brier_score",
        ]
    }
    n_samples = len(y_true)
    completed = 0
    attempts = 0

    while completed < n_bootstrap and attempts < n_bootstrap * 5:
        attempts += 1
        indices = rng.integers(0, n_samples, size=n_samples)
        y_boot = y_true[indices]
        if len(np.unique(y_boot)) < 2:
            continue
        score_boot = y_score[indices]
        boot_metrics = compute_binary_metrics(pd.Series(y_boot), score_boot, threshold=threshold)
        metrics["auc"].append(float(roc_auc_score(y_boot, score_boot)))
        metrics["accuracy"].append(float(boot_metrics["accuracy"]))
        metrics["balanced_accuracy"].append(float(boot_metrics["balanced_accuracy"]))
        metrics["precision"].append(float(boot_metrics["precision"]))
        metrics["recall"].append(float(boot_metrics["recall"]))
        metrics["f1"].append(float(boot_metrics["f1"]))
        metrics["specificity"].append(float(boot_metrics["specificity"]))
        metrics["npv"].append(float(boot_metrics["npv"]))
        metrics["mcc"].append(float(boot_metrics["mcc"]))
        metrics["average_precision"].append(float(boot_metrics["average_precision"]))
        metrics["brier_score"].append(float(boot_metrics["brier_score"]))
        completed += 1

    summary: dict[str, float] = {"bootstrap_completed": float(completed)}
    for metric_name, values in metrics.items():
        if values:
            summary[f"{metric_name}_ci_lower"] = float(np.quantile(values, 0.025))
            summary[f"{metric_name}_ci_upper"] = float(np.quantile(values, 0.975))
        else:
            summary[f"{metric_name}_ci_lower"] = math.nan
            summary[f"{metric_name}_ci_upper"] = math.nan
    return summary


def top_n_features(proteins: list[str], scores: np.ndarray, top_n: int = TOP_FEATURE_CAP) -> tuple[list[str], pd.DataFrame]:
    importance_df = pd.DataFrame({"protein": proteins, "importance": scores})
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)
    return importance_df.head(min(top_n, len(importance_df)))["protein"].tolist(), importance_df


def model_configs(seed: int) -> dict[str, dict[str, object]]:
    return {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=seed, n_jobs=-1),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [10, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
            "scale": False,
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(
                random_state=seed,
                eval_metric="logloss",
                verbosity=0,
                n_estimators=100,
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
            },
            "scale": False,
        },
        "Lasso Logistic": {
            "model": LogisticRegression(random_state=seed, max_iter=1000),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l1"],
                "solver": ["liblinear"],
            },
            "scale": True,
        },
    }


def fit_grid_or_default(
    estimator: object,
    params: dict[str, list[object]],
    x_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series,
    cv_folds: int | None,
) -> tuple[object, dict[str, object], float]:
    if cv_folds is None:
        estimator.fit(x_train, y_train)
        return estimator, {}, math.nan

    grid = GridSearchCV(
        estimator,
        params,
        cv=cv_folds,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(x_train, y_train)
    return grid.best_estimator_, grid.best_params_, float(grid.best_score_)


def select_feature_pool(
    de_df: pd.DataFrame,
    train_df: pd.DataFrame,
    preferred_order: tuple[str, ...] = ("bonf", "bh"),
) -> tuple[list[str], str]:
    for correction in preferred_order:
        sig_col, _ = correction_columns(correction)
        pool = [protein for protein in de_df.loc[de_df[sig_col], "protein"].tolist() if protein in train_df.columns]
        if pool:
            return pool, correction

    fallback = [protein for protein in de_df["protein"].tolist() if protein in train_df.columns][:FALLBACK_FEATURE_COUNT]
    return fallback, "p_value_fallback"


def rank_features_with_lasso(
    train_df: pd.DataFrame,
    feature_pool: list[str],
    seed: int,
    source_label: str,
) -> tuple[pd.DataFrame, list[str]]:
    if not feature_pool:
        return pd.DataFrame(columns=["protein", "importance", "rank", "in_top18", "feature_source"]), []

    x_train = train_df[feature_pool].copy()
    y_train = train_df["Label"].copy()
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    model = LogisticRegression(
        C=0.1,
        penalty="l1",
        solver="liblinear",
        random_state=seed,
        max_iter=1000,
    )
    model.fit(x_train_scaled, y_train)
    importance_scores = np.abs(model.coef_[0])
    top_features, ranking_df = top_n_features(feature_pool, importance_scores)
    ranking_df["in_top18"] = ranking_df["protein"].isin(top_features)
    ranking_df["feature_source"] = source_label
    return ranking_df, top_features


def evaluate_model_pool(
    model_name: str,
    config: dict[str, object],
    feature_columns: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> dict[str, object]:
    x_train = train_df[feature_columns].copy()
    y_train = train_df["Label"].copy()
    x_test = test_df[feature_columns].copy()
    y_test = test_df["Label"].copy()

    if config["scale"]:
        scaler = StandardScaler()
        x_train_processed = scaler.fit_transform(x_train)
        x_test_processed = scaler.transform(x_test)
    else:
        x_train_processed = x_train
        x_test_processed = x_test

    cv_folds = dynamic_cv_folds(y_train)
    fitted_model, best_params, cv_auc = fit_grid_or_default(
        estimator=config["model"],
        params=config["params"],
        x_train=x_train_processed,
        y_train=y_train,
        cv_folds=cv_folds,
    )
    y_score = fitted_model.predict_proba(x_test_processed)[:, 1]

    return {
        "method": model_name,
        "cv_auc": cv_auc,
        "test_auc": float(roc_auc_score(y_test, y_score)),
        "input_features": len(feature_columns),
        "selected_features": len(feature_columns),
        "best_params": json.dumps(best_params, sort_keys=True),
        "feature_set": ";".join(feature_columns),
        "cv_folds": cv_folds if cv_folds is not None else 0,
    }


def build_model_comparison(
    shared_features: list[str],
    feature_source: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    configs = model_configs(seed)
    rows: list[dict[str, object]] = []

    if not shared_features:
        rows.append(
            {
                "feature_source": feature_source,
                "panel_label": "No shared features",
                "method": "No significant features",
                "cv_auc": math.nan,
                "test_auc": math.nan,
                "input_features": 0,
                "selected_features": 0,
                "best_params": "{}",
                "feature_set": "",
                "cv_folds": 0,
            }
        )
        return pd.DataFrame(rows)

    panel_label = f"Shared Top {len(shared_features)}"
    for model_name in MODEL_NAMES:
        result = evaluate_model_pool(model_name, configs[model_name], shared_features, train_df, test_df, seed)
        result["feature_source"] = feature_source
        result["panel_label"] = panel_label
        rows.append(result)

    return pd.DataFrame(rows)


def build_panel_definitions(
    target_class: str,
    top_features: list[str],
    panel_sizes: list[int],
) -> dict[str, list[str]]:
    panels: dict[str, list[str]] = {}
    for size in sorted({size for size in panel_sizes if size > 0}):
        available = top_features[: min(size, len(top_features))]
        if available:
            panels[f"{target_class} Top {len(available)}"] = available
    return panels


def evaluate_panel(
    name: str,
    proteins: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> dict[str, object] | None:
    available = [protein for protein in proteins if protein in train_df.columns and protein in test_df.columns]
    if not available:
        return None

    x_train = train_df[available]
    y_train = train_df["Label"]
    x_test = test_df[available]
    y_test = test_df["Label"]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    cv_folds = dynamic_cv_folds(y_train)
    estimator = LogisticRegression(random_state=seed, max_iter=1000)
    params = {"C": [0.01, 0.1, 1, 10], "penalty": ["l1"], "solver": ["liblinear"]}
    fitted_model, best_params, cv_auc = fit_grid_or_default(
        estimator=estimator,
        params=params,
        x_train=x_train_scaled,
        y_train=y_train,
        cv_folds=cv_folds,
    )

    train_score = fitted_model.predict_proba(x_train_scaled)[:, 1]
    y_score = fitted_model.predict_proba(x_test_scaled)[:, 1]
    metrics = compute_binary_metrics(y_test, y_score)
    youden_threshold = train_youden_threshold(y_train.to_numpy(), train_score)
    youden_metrics = compute_binary_metrics(y_test, y_score, threshold=youden_threshold)
    importances = np.abs(fitted_model.coef_[0]) if hasattr(fitted_model, "coef_") else np.zeros(len(available))

    return {
        "Model": name,
        "Features": len(available),
        "CV_AUC": cv_auc,
        "Test_AUC": float(roc_auc_score(y_test, y_score)),
        "Best_Params": json.dumps(best_params, sort_keys=True),
        "cv_folds": cv_folds if cv_folds is not None else 0,
        "threshold": 0.5,
        "proteins": available,
        "importances": importances,
        "y_test": y_test.to_numpy(),
        "y_train": y_train.to_numpy(),
        "train_score": train_score,
        "y_score": y_score,
        "train_youden_threshold": youden_threshold,
        "youden_accuracy": youden_metrics["accuracy"],
        "youden_precision": youden_metrics["precision"],
        "youden_recall": youden_metrics["recall"],
        "youden_f1": youden_metrics["f1"],
        "youden_tn": youden_metrics["tn"],
        "youden_fp": youden_metrics["fp"],
        "youden_fn": youden_metrics["fn"],
        "youden_tp": youden_metrics["tp"],
        "ece_10bin": expected_calibration_error(y_test.to_numpy(), y_score, n_bins=10),
        **metrics,
    }


def build_panel_outputs(
    target_class: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    top_features: list[str],
    panel_sizes: list[int],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    panels = build_panel_definitions(target_class, top_features, panel_sizes)
    panel_results: list[dict[str, object]] = []
    plot_payloads: list[dict[str, object]] = []

    for name, proteins in panels.items():
        result = evaluate_panel(name, proteins, train_df, test_df, seed)
        if result is None:
            continue
        plot_payloads.append(result)
        panel_results.append(
            {
                "Model": result["Model"],
                "Features": result["Features"],
                "CV_AUC": result["CV_AUC"],
                "Test_AUC": result["Test_AUC"],
                "Best_Params": result["Best_Params"],
                "cv_folds": result["cv_folds"],
                "threshold": result["threshold"],
                "accuracy": result["accuracy"],
                "balanced_accuracy": result["balanced_accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "specificity": result["specificity"],
                "npv": result["npv"],
                "mcc": result["mcc"],
                "average_precision": result["average_precision"],
                "brier_score": result["brier_score"],
                "tn": result["tn"],
                "fp": result["fp"],
                "fn": result["fn"],
                "tp": result["tp"],
                "proteins": ";".join(result["proteins"]),
            }
        )

    if panel_results:
        panel_df = pd.DataFrame(panel_results).sort_values(["Features", "Model"]).reset_index(drop=True)
    else:
        panel_df = pd.DataFrame(
            columns=[
                "Model",
                "Features",
                "CV_AUC",
                "Test_AUC",
                "Best_Params",
                "cv_folds",
                "threshold",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "specificity",
                "npv",
                "mcc",
                "average_precision",
                "brier_score",
                "tn",
                "fp",
                "fn",
                "tp",
                "proteins",
            ]
        )
    ranking_df = pd.DataFrame({"protein": top_features[:TOP_FEATURE_CAP], "rank": range(1, len(top_features[:TOP_FEATURE_CAP]) + 1)})
    if not ranking_df.empty:
        ranking_df["target_class"] = target_class

    selected_dataset_columns = ["Label"] + [column for column in ["Sample_ID", "Cancer"] if column in train_df.columns]
    selected_dataset_columns += top_features[:TOP_FEATURE_CAP]
    panel_dataset = pd.concat(
        [
            train_df[selected_dataset_columns],
            test_df[selected_dataset_columns],
        ],
        ignore_index=True,
    )
    return panel_df, ranking_df, panel_dataset, plot_payloads


def build_panel_confidence_intervals(
    target_class: str,
    plot_payloads: list[dict[str, object]],
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for payload in plot_payloads:
        intervals = bootstrap_panel_intervals(
            y_true=np.asarray(payload["y_test"]),
            y_score=np.asarray(payload["y_score"]),
            threshold=0.5,
            n_bootstrap=bootstrap_samples,
            seed=seed + int(payload["Features"]),
        )
        rows.append(
            {
                "target_class": target_class,
                "Model": payload["Model"],
                "Features": payload["Features"],
                "test_auc": payload["Test_AUC"],
                "accuracy": payload["accuracy"],
                "balanced_accuracy": payload["balanced_accuracy"],
                "precision": payload["precision"],
                "recall": payload["recall"],
                "f1": payload["f1"],
                "specificity": payload["specificity"],
                "npv": payload["npv"],
                "mcc": payload["mcc"],
                "average_precision": payload["average_precision"],
                "brier_score": payload["brier_score"],
                **intervals,
            }
        )
    return pd.DataFrame(rows).sort_values("Features").reset_index(drop=True)


def build_threshold_justification_table(target_class: str, plot_payloads: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for payload in plot_payloads:
        rows.append(
            {
                "target_class": target_class,
                "Model": payload["Model"],
                "Features": payload["Features"],
                "default_threshold": 0.5,
                "train_youden_threshold": payload["train_youden_threshold"],
                "test_auc": payload["Test_AUC"],
                "test_accuracy_at_0_5": payload["accuracy"],
                "test_balanced_accuracy_at_0_5": payload["balanced_accuracy"],
                "test_precision_at_0_5": payload["precision"],
                "test_recall_at_0_5": payload["recall"],
                "test_f1_at_0_5": payload["f1"],
                "test_specificity_at_0_5": payload["specificity"],
                "test_npv_at_0_5": payload["npv"],
                "test_mcc_at_0_5": payload["mcc"],
                "test_accuracy_at_youden": payload["youden_accuracy"],
                "test_precision_at_youden": payload["youden_precision"],
                "test_recall_at_youden": payload["youden_recall"],
                "test_f1_at_youden": payload["youden_f1"],
            }
        )
    return pd.DataFrame(rows).sort_values("Features").reset_index(drop=True)


def build_calibration_summary(target_class: str, plot_payloads: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for payload in plot_payloads:
        rows.append(
            {
                "target_class": target_class,
                "Model": payload["Model"],
                "Features": payload["Features"],
                "brier_score": payload["brier_score"],
                "average_precision": payload["average_precision"],
                "ece_10bin": payload["ece_10bin"],
                "mean_predicted_probability": float(np.mean(payload["y_score"])),
                "observed_event_rate": float(np.mean(payload["y_test"])),
            }
        )
    return pd.DataFrame(rows).sort_values("Features").reset_index(drop=True)


def select_reference_panel_payload(plot_payloads: list[dict[str, object]]) -> dict[str, object] | None:
    if not plot_payloads:
        return None
    return max(plot_payloads, key=lambda payload: (int(payload["Features"]), float(payload["Test_AUC"])))


def save_panel_auc_figure(target_class: str, panel_df: pd.DataFrame, output_path: Path) -> None:
    if panel_df.empty:
        return
    figure, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=panel_df, x="Model", y="Test_AUC", ax=ax, color="#4c78a8")
    ax.set_title(f"{target_class} one-vs-rest panel performance")
    ax.set_xlabel("Panel")
    ax.set_ylabel("Test AUC")
    ax.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_feature_importance_heatmap(importance_df: pd.DataFrame, output_path: Path) -> None:
    if importance_df.empty:
        return

    heatmap_df = importance_df.pivot(index="target_class", columns="protein", values="importance").fillna(0.0)
    column_order = (
        importance_df.groupby("protein")
        .agg(class_count=("target_class", "nunique"), mean_importance=("importance", "mean"))
        .sort_values(["class_count", "mean_importance", "protein"], ascending=[False, False, True])
        .index.tolist()
    )
    heatmap_df = heatmap_df.loc[sorted(heatmap_df.index), column_order]

    figure_width = max(12, min(40, 0.42 * heatmap_df.shape[1] + 4))
    figure_height = max(6, 0.6 * heatmap_df.shape[0] + 2)
    figure, ax = plt.subplots(figsize=(figure_width, figure_height))
    sns.heatmap(heatmap_df, cmap="viridis", linewidths=0.2, linecolor="white", ax=ax)
    ax.set_title("Feature importance heatmap across all classes")
    ax.set_xlabel("Protein")
    ax.set_ylabel("Target class")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_all_class_roc_figure(reference_payloads: list[dict[str, object]], output_path: Path) -> None:
    if not reference_payloads:
        return

    figure, ax = plt.subplots(figsize=(10.5, 8))
    palette = sns.color_palette("tab20", n_colors=max(len(reference_payloads), 3))
    for color, payload in zip(palette, sorted(reference_payloads, key=lambda item: str(item["target_class"]))):
        y_true = np.asarray(payload["y_test"])
        y_score = np.asarray(payload["y_score"])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            color=color,
            label=f"{payload['target_class']} (AUC={float(payload['Test_AUC']):.3f}, n={int(payload['Features'])})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
    ax.set_title("ROC curves across all classes")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_performance_vs_panel_size_figure(panel_df: pd.DataFrame, output_path: Path) -> None:
    if panel_df.empty:
        return

    performance_df = panel_df.sort_values(["target_class", "Features"]).copy()
    mean_df = (
        performance_df.groupby("Features", as_index=False)
        .agg(Test_AUC=("Test_AUC", "mean"))
        .sort_values("Features")
        .reset_index(drop=True)
    )

    figure, ax = plt.subplots(figsize=(11, 6.5))
    sns.lineplot(
        data=performance_df,
        x="Features",
        y="Test_AUC",
        hue="target_class",
        marker="o",
        linewidth=1.8,
        ax=ax,
    )
    ax.plot(
        mean_df["Features"],
        mean_df["Test_AUC"],
        color="black",
        linestyle="--",
        linewidth=2.5,
        marker="o",
        label="Mean across classes",
    )
    ax.set_title("Performance vs panel size across all classes")
    ax.set_xlabel("Panel size")
    ax.set_ylabel("Test AUC")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_calibration_figure(target_class: str, plot_payloads: list[dict[str, object]], output_path: Path) -> None:
    if not plot_payloads:
        return
    selected_payloads = sorted(plot_payloads, key=lambda payload: payload["Features"])[:4]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5, label="Perfect calibration")

    calibration_rows: list[dict[str, object]] = []
    for payload in selected_payloads:
        prob_true, prob_pred = calibration_curve(payload["y_test"], payload["y_score"], n_bins=10, strategy="quantile")
        axes[0].plot(prob_pred, prob_true, marker="o", linewidth=2, label=payload["Model"])
        calibration_rows.append({"Model": payload["Model"], "ece_10bin": payload["ece_10bin"]})

    axes[0].set_title(f"{target_class} calibration curves")
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed event rate")
    axes[0].legend(loc="best", fontsize=8)

    calibration_df = pd.DataFrame(calibration_rows)
    sns.barplot(data=calibration_df, x="Model", y="ece_10bin", ax=axes[1], color="#9ecae1")
    axes[1].set_title(f"{target_class} expected calibration error")
    axes[1].set_xlabel("Panel")
    axes[1].set_ylabel("ECE (10-bin)")
    axes[1].tick_params(axis="x", rotation=25)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_enrichment_for_class(
    target_class: str,
    target_slug: str,
    de_df: pd.DataFrame,
    enrichment_dir: Path,
    enrichment_figure_dir: Path,
    databases: list[str],
    top_terms: int,
    gsea_permutations: int,
) -> list[dict[str, object]]:
    proteins = de_df["protein"].tolist()
    mapping_df = load_or_build_mapping(proteins)
    collapsed_df = collapse_to_genes(de_df, mapping_df)
    selected_df = collapsed_df[collapsed_df["significant_bonf"]].copy()
    selected_genes = set(selected_df["mapped_gene"])
    background_genes = set(collapsed_df["mapped_gene"])

    ranked_df = collapsed_df[["mapped_gene", "fold_change", "p_adjusted_bonf", "p_adjusted_bh"]].copy()
    ranked_df = ranked_df.rename(columns={"mapped_gene": "gene_symbol"})
    ranked_df.to_csv(enrichment_dir / f"{target_slug}_ranked_gene_list.csv", index=False)
    selected_df[["protein", "mapped_gene", "mapping_status", "mapping_source", "p_adjusted_bonf", "fold_change"]].to_csv(
        enrichment_dir / f"{target_slug}_selected_gene_mapping.csv",
        index=False,
    )

    summary_rows: list[dict[str, object]] = []
    for database in databases:
        try:
            gene_sets = load_or_fetch_library(database)
            ora_results = ora(selected_genes, background_genes, gene_sets, database)
            gsea_results = preranked_gsea(collapsed_df, gene_sets, database, gsea_permutations)
        except Exception as exc:  # noqa: BLE001
            summary_rows.append(
                {
                    "target_class": target_class,
                    "database": database,
                    "selected_genes": len(selected_genes),
                    "background_genes": len(background_genes),
                    "ora_terms": 0,
                    "preranked_terms": 0,
                    "status": f"failed: {exc}",
                }
            )
            continue

        ora_output = enrichment_dir / f"{target_slug}_{database}_ora.csv"
        top_output = enrichment_dir / f"{target_slug}_{database}_top_terms.csv"
        plot_output = enrichment_figure_dir / f"{target_slug}_{database}_dotplot.png"
        gsea_output = enrichment_dir / f"{target_slug}_{database}_preranked_gsea.csv"
        gsea_top_output = enrichment_dir / f"{target_slug}_{database}_preranked_top_terms.csv"
        gsea_plot_output = enrichment_figure_dir / f"{target_slug}_{database}_preranked_barplot.png"

        ora_results.to_csv(ora_output, index=False)
        ora_results.head(top_terms).to_csv(top_output, index=False)
        gsea_results.to_csv(gsea_output, index=False)
        gsea_results.head(top_terms).to_csv(gsea_top_output, index=False)
        save_dotplot(ora_results, f"{target_class} {database.upper()} enrichment", plot_output, top_terms)
        save_preranked_barplot(
            gsea_results,
            f"{target_class} {database.upper()} preranked enrichment",
            gsea_plot_output,
            top_terms,
        )
        summary_rows.append(
            {
                "target_class": target_class,
                "database": database,
                "selected_genes": len(selected_genes),
                "background_genes": len(background_genes),
                "ora_terms": len(ora_results),
                "preranked_terms": len(gsea_results),
                "status": "ok",
            }
        )
    return summary_rows


def evaluate_top_panel_for_repeat(
    target_class: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    protein_columns: list[str],
    seed: int,
    panel_sizes: list[int],
) -> tuple[dict[str, object] | None, int]:
    de_df = differential_expression(
        train_df=train_df,
        protein_columns=protein_columns,
        case_mean_column="case_mean",
        case_n_column="case_n",
        control_n_column="rest_n",
    )
    feature_pool, feature_source = select_feature_pool(de_df, train_df)
    _, top_features = rank_features_with_lasso(train_df, feature_pool, seed, feature_source)
    panels = build_panel_definitions(target_class, top_features, panel_sizes)
    target_panel_name = next(
        (name for name in panels if name.endswith(f"Top {min(TOP_FEATURE_CAP, len(top_features))}")),
        None,
    )
    if target_panel_name is None and panels:
        target_panel_name = next(iter(panels))
    if target_panel_name is None:
        return None, int(de_df["significant_bonf"].sum())
    result = evaluate_panel(target_panel_name, panels[target_panel_name], train_df, test_df, seed)
    return result, int(de_df["significant_bonf"].sum())


def run_repeated_split_robustness(
    full_df: pd.DataFrame,
    class_column: str,
    sample_id_column: str,
    protein_columns: list[str],
    classes: list[str],
    imputer_method: str,
    knn_neighbors: int,
    repeats: int,
    seed: int,
    test_size: float,
    panel_sizes: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset in range(repeats):
        current_seed = seed + offset
        train_base, test_base = split_and_impute_multiclass(
            df=full_df,
            protein_columns=protein_columns,
            class_column=class_column,
            sample_id_column=sample_id_column,
            seed=current_seed,
            test_size=test_size,
            imputer_method=imputer_method,
            knn_neighbors=knn_neighbors,
        )
        for target_class in classes:
            train_df, test_df = prepare_binary_partition(train_base, test_base, class_column, target_class)
            result, bonf_count = evaluate_top_panel_for_repeat(
                target_class=target_class,
                train_df=train_df,
                test_df=test_df,
                protein_columns=protein_columns,
                seed=current_seed,
                panel_sizes=panel_sizes,
            )
            if result is None:
                continue
            rows.append(
                {
                    "target_class": target_class,
                    "repeat": offset + 1,
                    "seed": current_seed,
                    "Model": result["Model"],
                    "Features": result["Features"],
                    "Test_AUC": result["Test_AUC"],
                    "average_precision": result["average_precision"],
                    "balanced_accuracy": result["balanced_accuracy"],
                    "mcc": result["mcc"],
                    "bonferroni_feature_count": bonf_count,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dirs = ensure_output_dirs(args.output_dir)

    full_df = pd.read_csv(args.input)
    if args.class_column not in full_df.columns:
        raise ValueError(f"Column '{args.class_column}' was not found in {args.input}.")
    if args.sample_id_column not in full_df.columns:
        raise ValueError(f"Column '{args.sample_id_column}' was not found in {args.input}.")

    protein_columns = protein_columns_from_df(full_df, args.class_column, args.sample_id_column)
    classes = sorted(full_df[args.class_column].dropna().unique().tolist())

    train_base, test_base = split_and_impute_multiclass(
        df=full_df,
        protein_columns=protein_columns,
        class_column=args.class_column,
        sample_id_column=args.sample_id_column,
        seed=args.seed,
        test_size=args.test_size,
        imputer_method=args.imputer,
        knn_neighbors=args.knn_neighbors,
    )

    train_base.to_csv(output_dirs["data"] / "multiclass_train_imputed.csv", index=False)
    test_base.to_csv(output_dirs["data"] / "multiclass_test_imputed.csv", index=False)
    distribution_df = pd.concat(
        [
            class_distribution_table(train_base, args.class_column, "train"),
            class_distribution_table(test_base, args.class_column, "test"),
        ],
        ignore_index=True,
    )
    distribution_df.to_csv(output_dirs["data"] / "class_distribution_summary.csv", index=False)

    aggregate_rows: list[dict[str, object]] = []
    enrichment_rows: list[dict[str, object]] = []
    all_panel_frames: list[pd.DataFrame] = []
    all_reference_rows: list[dict[str, object]] = []
    all_reference_payloads: list[dict[str, object]] = []
    all_importance_rows: list[dict[str, object]] = []

    for target_class in classes:
        target_slug = slugify(target_class)
        train_df, test_df = prepare_binary_partition(train_base, test_base, args.class_column, target_class)
        train_df.to_csv(output_dirs["data"] / f"{target_slug}_train_ovr.csv", index=False)
        test_df.to_csv(output_dirs["data"] / f"{target_slug}_test_ovr.csv", index=False)

        de_df = differential_expression(
            train_df=train_df,
            protein_columns=protein_columns,
            case_mean_column="case_mean",
            case_n_column="case_n",
            control_n_column="rest_n",
        )
        de_df.insert(0, "target_class", target_class)
        de_df.to_csv(output_dirs["dea"] / f"{target_slug}_differential_expression_results.csv", index=False)

        feature_pool, feature_source = select_feature_pool(de_df, train_df)
        ranking_df, top_features = rank_features_with_lasso(train_df, feature_pool, args.seed, feature_source)
        model_comparison_df = build_model_comparison(top_features, feature_source, train_df, test_df, args.seed)
        model_comparison_df.insert(0, "target_class", target_class)
        model_comparison_df.to_csv(output_dirs["model"] / f"{target_slug}_model_comparison.csv", index=False)
        ranking_df.insert(0, "target_class", target_class)
        panel_df, _, panel_dataset_df, plot_payloads = build_panel_outputs(
            target_class=target_class,
            train_df=train_df,
            test_df=test_df,
            top_features=top_features,
            panel_sizes=args.panel_sizes,
            seed=args.seed,
        )
        panel_df.insert(0, "target_class", target_class)
        panel_dataset_df.insert(0, "target_class", target_class)
        if not panel_df.empty:
            all_panel_frames.append(panel_df.copy())

        ranking_df.to_csv(output_dirs["model"] / f"{target_slug}_top_feature_ranking.csv", index=False)
        panel_df.to_csv(output_dirs["model"] / f"{target_slug}_panel_performance.csv", index=False)
        panel_dataset_df.to_csv(output_dirs["model"] / f"{target_slug}_selected_panel_dataset.csv", index=False)

        save_panel_auc_figure(target_class, panel_df, output_dirs["figures"] / f"{target_slug}_panel_auc.png")

        reference_payload = select_reference_panel_payload(plot_payloads)
        if reference_payload is not None:
            all_reference_rows.append(
                {
                    "target_class": target_class,
                    "Model": reference_payload["Model"],
                    "Features": reference_payload["Features"],
                    "Test_AUC": reference_payload["Test_AUC"],
                    "accuracy": reference_payload["accuracy"],
                    "balanced_accuracy": reference_payload["balanced_accuracy"],
                    "precision": reference_payload["precision"],
                    "recall": reference_payload["recall"],
                    "f1": reference_payload["f1"],
                    "mcc": reference_payload["mcc"],
                    "average_precision": reference_payload["average_precision"],
                }
            )
            reference_payload_with_class = {**reference_payload, "target_class": target_class}
            all_reference_payloads.append(reference_payload_with_class)
            all_importance_rows.extend(
                {
                    "target_class": target_class,
                    "protein": protein,
                    "importance": float(importance),
                }
                for protein, importance in zip(reference_payload["proteins"], reference_payload["importances"])
            )

        if args.run_robustness:
            panel_ci_df = build_panel_confidence_intervals(target_class, plot_payloads, args.bootstrap_samples, args.seed)
            threshold_df = build_threshold_justification_table(target_class, plot_payloads)
            calibration_df = build_calibration_summary(target_class, plot_payloads)
            panel_ci_df.to_csv(output_dirs["robustness"] / f"{target_slug}_panel_confidence_intervals.csv", index=False)
            threshold_df.to_csv(output_dirs["robustness"] / f"{target_slug}_threshold_justification.csv", index=False)
            calibration_df.to_csv(output_dirs["robustness"] / f"{target_slug}_calibration_summary.csv", index=False)
            save_calibration_figure(
                target_class,
                plot_payloads,
                output_dirs["robustness_figures"] / f"{target_slug}_calibration.png",
            )

        if args.run_enrichment:
            enrichment_rows.extend(
                run_enrichment_for_class(
                    target_class=target_class,
                    target_slug=target_slug,
                    de_df=de_df,
                    enrichment_dir=output_dirs["enrichment"],
                    enrichment_figure_dir=output_dirs["enrichment_figures"],
                    databases=args.enrichment_databases,
                    top_terms=args.top_terms,
                    gsea_permutations=args.gsea_permutations,
                )
            )

        positive_train = int(train_df["Label"].sum())
        positive_test = int(test_df["Label"].sum())
        best_panel_auc = float(panel_df["Test_AUC"].max()) if not panel_df.empty else math.nan
        best_panel_name = panel_df.sort_values("Test_AUC", ascending=False).iloc[0]["Model"] if not panel_df.empty else ""
        aggregate_rows.append(
            {
                "target_class": target_class,
                "train_case_n": positive_train,
                "train_rest_n": int(len(train_df) - positive_train),
                "test_case_n": positive_test,
                "test_rest_n": int(len(test_df) - positive_test),
                "bh_significant_features": int(de_df["significant_bh"].sum()),
                "bonferroni_significant_features": int(de_df["significant_bonf"].sum()),
                "feature_source": feature_source,
                "selected_feature_pool_size": len(feature_pool),
                "top_feature_count": len(top_features),
                "best_panel": best_panel_name,
                "best_panel_test_auc": best_panel_auc,
            }
        )

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values("target_class").reset_index(drop=True)
    aggregate_df.to_csv(output_dirs["root"] / "class_iteration_summary.csv", index=False)

    if all_panel_frames:
        combined_panel_df = pd.concat(all_panel_frames, ignore_index=True)
        combined_panel_df.to_csv(output_dirs["model"] / "all_classes_panel_performance.csv", index=False)
        save_performance_vs_panel_size_figure(
            combined_panel_df,
            output_dirs["figures"] / "all_classes_performance_vs_panel_size.png",
        )

    if all_reference_rows:
        reference_df = pd.DataFrame(all_reference_rows).sort_values("target_class").reset_index(drop=True)
        reference_df.to_csv(output_dirs["model"] / "all_classes_reference_panel_summary.csv", index=False)

    if all_importance_rows:
        importance_df = pd.DataFrame(all_importance_rows).sort_values(["target_class", "importance"], ascending=[True, False]).reset_index(drop=True)
        importance_df.to_csv(output_dirs["model"] / "all_classes_reference_feature_importance.csv", index=False)
        save_feature_importance_heatmap(
            importance_df,
            output_dirs["figures"] / "all_classes_feature_importance_heatmap.png",
        )

    if all_reference_payloads:
        save_all_class_roc_figure(
            all_reference_payloads,
            output_dirs["figures"] / "all_classes_roc_curves.png",
        )

    if enrichment_rows:
        pd.DataFrame(enrichment_rows).to_csv(output_dirs["enrichment"] / "enrichment_run_summary.csv", index=False)

    if args.run_robustness and args.robustness_repeats > 0:
        repeat_df = run_repeated_split_robustness(
            full_df=full_df,
            class_column=args.class_column,
            sample_id_column=args.sample_id_column,
            protein_columns=protein_columns,
            classes=classes,
            imputer_method=args.imputer,
            knn_neighbors=args.knn_neighbors,
            repeats=args.robustness_repeats,
            seed=args.seed,
            test_size=args.test_size,
            panel_sizes=args.panel_sizes,
        )
        repeat_df.to_csv(output_dirs["robustness"] / "repeated_split_summary.csv", index=False)
        if not repeat_df.empty:
            aggregate_repeat_df = (
                repeat_df.groupby(["target_class", "Model", "Features"], as_index=False)
                .agg(
                    repeats=("repeat", "count"),
                    test_auc_mean=("Test_AUC", "mean"),
                    test_auc_std=("Test_AUC", "std"),
                    average_precision_mean=("average_precision", "mean"),
                    average_precision_std=("average_precision", "std"),
                    balanced_accuracy_mean=("balanced_accuracy", "mean"),
                    balanced_accuracy_std=("balanced_accuracy", "std"),
                    mcc_mean=("mcc", "mean"),
                    mcc_std=("mcc", "std"),
                    bonferroni_feature_count_mean=("bonferroni_feature_count", "mean"),
                )
                .sort_values(["target_class", "test_auc_mean"], ascending=[True, False])
                .reset_index(drop=True)
            )
            aggregate_repeat_df.to_csv(output_dirs["robustness"] / "repeated_split_aggregate.csv", index=False)

    manifest_rows = []
    for section, path in output_dirs.items():
        manifest_rows.append({"section": section, "path": str(path.resolve())})
    pd.DataFrame(manifest_rows).to_csv(output_dirs["root"] / "output_manifest.csv", index=False)

    print(f"Input dataset: {args.input}")
    print(f"Class column: {args.class_column}")
    print(f"Classes discovered: {len(classes)}")
    print(f"Output directory: {output_dirs['root']}")
    print(f"Saved multiclass train split: {output_dirs['data'] / 'multiclass_train_imputed.csv'}")
    print(f"Saved multiclass test split: {output_dirs['data'] / 'multiclass_test_imputed.csv'}")
    print(f"Saved class summary: {output_dirs['root'] / 'class_iteration_summary.csv'}")
    print(f"Saved output manifest: {output_dirs['root'] / 'output_manifest.csv'}")


if __name__ == "__main__":
    main()
