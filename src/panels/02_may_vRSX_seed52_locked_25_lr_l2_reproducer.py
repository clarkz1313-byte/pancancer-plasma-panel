#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
OUTPUT_ROOT = ROOT / "revise_plan" / "part_b_multiclass" / "vRSX_v11_locked_reproducer"
V8_PATH = SCRIPT_DIR / "02_may_v8_final_reproducibility_tournament.py"

spec = importlib.util.spec_from_file_location("v8", V8_PATH)
v8 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v8)

SEED = 52
BOOTSTRAP_SEED = 52024
BOOTSTRAP_SAMPLES = 1000
TEST_SIZE = 0.3
INPUT = ROOT / "data" / "processed" / "filtered_pancancer_data.csv"
CLASS_COLUMN = "Cancer"
SAMPLE_ID_COLUMN = "Sample_ID"
IMPUTER = "knn"
KNN_NEIGHBORS = 5
PER_CANCER_TOP_M = 14
MIN_CANCER_COVERAGE = 2
CANDIDATE_BANK_SIZE = 180
FEATURE_COUNT = 25
MIN_PER_CLASS = 1
N_SHARED = 0
RANKER = "LOGISTIC_COEF"
COACH = "LR_L2"

EXPECTED_FEATURES = [
    "FLT3",
    "CNTN1",
    "FCER2",
    "PRDX6",
    "LTA4H",
    "XG",
    "CCDC80",
    "CXCL17",
    "CXCL13",
    "SLAMF7",
    "PAEP",
    "PSPN",
    "BMP4",
    "WFDC2",
    "TRAF2",
    "KLK13",
    "GLO1",
    "GFAP",
    "CEACAM5",
    "CGA",
    "ADAMTS13",
    "CRTAC1",
    "TCL1A",
    "ADAMTS15",
    "NEFL",
]

EXPECTED = {
    "accuracy": 0.7530266343825666,
    "balanced_accuracy": 0.7986397136842692,
    "macro_f1": 0.7839556965531164,
    "macro_ovr_auc": 0.9557774180952709,
    "min_class_recall": 0.6190476190476191,
}


def ensure_dirs() -> dict[str, Path]:
    output = {
        "root": OUTPUT_ROOT,
        "tables": OUTPUT_ROOT / "tables",
        "figures": OUTPUT_ROOT / "figures",
        "logs": OUTPUT_ROOT / "logs",
        "manifests": OUTPUT_ROOT / "manifests",
    }
    for path in output.values():
        path.mkdir(parents=True, exist_ok=True)
    return output


def log(paths: dict[str, Path], message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (paths["logs"] / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def ordered_panel(
    rankings: dict[str, list[str]],
    quota_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    candidate_features: list[str],
    classes: list[str],
) -> list[str]:
    shared = (
        coverage_df.loc[coverage_df["cancer_count"] >= MIN_CANCER_COVERAGE]
        .sort_values(["cancer_count", "best_bonf_rank", "protein"], ascending=[False, True, True])["protein"]
        .tolist()
    )
    seed_features = [
        *v8.quota_panel_order(quota_df, classes, MIN_PER_CLASS),
        *shared[:N_SHARED],
        *rankings[RANKER],
    ]
    return v8.resize_panel(seed_features, FEATURE_COUNT, candidate_features)


def per_class_auc(y_true: pd.Series, y_score: np.ndarray, classes: list[str]) -> pd.DataFrame:
    y_values = y_true.to_numpy()
    rows: list[dict[str, object]] = []
    for class_index, class_name in enumerate(classes):
        binary = (y_values == class_name).astype(int)
        auc = float(roc_auc_score(binary, y_score[:, class_index])) if len(np.unique(binary)) >= 2 else math.nan
        rows.append({"label": class_name, "ovr_auc": auc})
    return pd.DataFrame(rows)


def bootstrap_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray, classes: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    y_values = y_true.to_numpy()
    rows: list[dict[str, object]] = []
    for bootstrap_index in range(1, BOOTSTRAP_SAMPLES + 1):
        indices = rng.integers(0, len(y_values), size=len(y_values))
        y_boot = y_values[indices]
        pred_boot = y_pred[indices]
        score_boot = y_score[indices, :]
        recalls = recall_score(y_boot, pred_boot, labels=classes, average=None, zero_division=0)
        try:
            macro_auc = float(roc_auc_score(y_boot, score_boot, labels=classes, multi_class="ovr", average="macro"))
        except ValueError:
            macro_auc = math.nan
        rows.append(
            {
                "bootstrap_index": bootstrap_index,
                "accuracy": float(accuracy_score(y_boot, pred_boot)),
                "balanced_accuracy": float(balanced_accuracy_score(y_boot, pred_boot)),
                "macro_f1": float(f1_score(y_boot, pred_boot, labels=classes, average="macro", zero_division=0)),
                "macro_ovr_auc": macro_auc,
                "min_class_recall": float(np.min(recalls)),
                "brc_recall": float(recalls[classes.index("BRC")]),
                "cvx_recall": float(recalls[classes.index("CVX")]),
                "endc_recall": float(recalls[classes.index("ENDC")]),
            }
        )
    return pd.DataFrame(rows)


def ci_summary(bootstrap_df: pd.DataFrame, point: dict[str, float]) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_ovr_auc",
        "min_class_recall",
        "brc_recall",
        "cvx_recall",
        "endc_recall",
    ]
    rows: list[dict[str, object]] = []
    for metric in metrics:
        values = pd.to_numeric(bootstrap_df[metric], errors="coerce").dropna()
        rows.append(
            {
                "metric": metric,
                "point_estimate": point.get(metric, math.nan),
                "bootstrap_mean": float(values.mean()),
                "ci_lower_2_5": float(values.quantile(0.025)),
                "ci_upper_97_5": float(values.quantile(0.975)),
                "bootstrap_samples": int(values.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def save_figures(paths: dict[str, Path], confusion: pd.DataFrame, per_class_df: pd.DataFrame, bootstrap_df: pd.DataFrame) -> None:
    figure, ax = v8.base.plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.3, linecolor="white", ax=ax)
    ax.set_title("vRSX locked 25-protein LR_L2 confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    figure.tight_layout()
    figure.savefig(paths["figures"] / "locked_confusion_matrix.png", dpi=300, bbox_inches="tight")
    v8.base.plt.close(figure)

    plot_df = per_class_df.melt(id_vars="label", value_vars=["recall", "ovr_auc"], var_name="metric", value_name="value")
    figure, ax = v8.base.plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=plot_df, x="label", y="value", hue="metric", ax=ax)
    ax.set_title("vRSX per-class recall and one-vs-rest AUC")
    ax.set_xlabel("Cancer class")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1.03)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(paths["figures"] / "per_class_recall_auc.png", dpi=300, bbox_inches="tight")
    v8.base.plt.close(figure)

    boot_plot = bootstrap_df.melt(
        id_vars="bootstrap_index",
        value_vars=["balanced_accuracy", "macro_f1", "macro_ovr_auc", "min_class_recall"],
        var_name="metric",
        value_name="value",
    )
    figure, ax = v8.base.plt.subplots(figsize=(9, 5.2))
    sns.boxplot(data=boot_plot, x="metric", y="value", color="#9ecae1", ax=ax)
    sns.stripplot(data=boot_plot, x="metric", y="value", color="#1f4e79", alpha=0.22, size=1.8, ax=ax)
    ax.set_title(f"vRSX bootstrap metric distributions (n={BOOTSTRAP_SAMPLES})")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Bootstrap value")
    ax.set_ylim(0, 1.03)
    ax.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(paths["figures"] / "bootstrap_metric_distributions.png", dpi=300, bbox_inches="tight")
    v8.base.plt.close(figure)


def main() -> None:
    paths = ensure_dirs()
    log(paths, "vRSX locked 25-protein LR_L2 reproducer starts from processed NPX data.")
    full_df = pd.read_csv(INPUT)
    protein_columns = v8.base.protein_columns_from_df(full_df, CLASS_COLUMN, SAMPLE_ID_COLUMN)
    classes = sorted(full_df[CLASS_COLUMN].dropna().unique().tolist())

    train_df, test_df = v8.v1.split_and_impute(
        full_df=full_df,
        protein_columns=protein_columns,
        class_column=CLASS_COLUMN,
        sample_id_column=SAMPLE_ID_COLUMN,
        seed=SEED,
        test_size=TEST_SIZE,
        imputer_method=IMPUTER,
        knn_neighbors=KNN_NEIGHBORS,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    bonf_df, quota_df, de_df = v8.collect_train_only_de_sources(
        train_df=train_df,
        protein_columns=protein_columns,
        class_column=CLASS_COLUMN,
        classes=classes,
        per_cancer_top_m=PER_CANCER_TOP_M,
    )
    candidate_features, candidate_feature_df, coverage_df = v8.build_candidate_feature_table(
        bonf_df=bonf_df,
        quota_df=quota_df,
        protein_columns=protein_columns,
        min_cancer_coverage=MIN_CANCER_COVERAGE,
        candidate_bank_size=CANDIDATE_BANK_SIZE,
    )
    rankings, ranking_table = v8.v2.build_rankings_v2(
        x_train=train_df[candidate_features],
        y_train=train_df[CLASS_COLUMN],
        classes=classes,
        quota_sources_df=quota_df,
        coverage_df=coverage_df,
        seed=SEED,
    )
    selected_features = ordered_panel(rankings, quota_df, coverage_df, candidate_features, classes)
    if selected_features != EXPECTED_FEATURES:
        raise RuntimeError(
            "Selected features drifted from the locked vRSX panel. "
            f"Observed={';'.join(selected_features)} Expected={';'.join(EXPECTED_FEATURES)}"
        )

    estimator = v8.v2.make_estimator_v2(COACH, SEED, {})
    estimator.fit(train_df[selected_features], train_df[CLASS_COLUMN])
    y_true = test_df[CLASS_COLUMN]
    y_pred = np.asarray(estimator.predict(test_df[selected_features]))
    y_score = v8.v1.aligned_probabilities(estimator, test_df[selected_features], classes)
    metrics = v8.v1.metric_bundle(y_true, y_pred, y_score, classes)
    for key, expected_value in EXPECTED.items():
        if not math.isclose(metrics[key], expected_value, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(f"Metric {key} drifted: observed={metrics[key]} expected={expected_value}")

    recalls = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    per_class_df = pd.DataFrame({"label": classes, "recall": recalls}).merge(per_class_auc(y_true, y_score, classes), on="label", how="left")
    point = {
        **metrics,
        "brc_recall": float(per_class_df.loc[per_class_df["label"] == "BRC", "recall"].iloc[0]),
        "cvx_recall": float(per_class_df.loc[per_class_df["label"] == "CVX", "recall"].iloc[0]),
        "endc_recall": float(per_class_df.loc[per_class_df["label"] == "ENDC", "recall"].iloc[0]),
    }
    bootstrap_df = bootstrap_metrics(y_true, y_pred, y_score, classes)
    ci_df = ci_summary(bootstrap_df, point)
    confusion = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=classes), index=classes, columns=classes)

    summary = {
        "seed": SEED,
        "feature_count": FEATURE_COUNT,
        "coach": COACH,
        "ranker": RANKER,
        "min_per_class": MIN_PER_CLASS,
        "n_shared": N_SHARED,
        "features": ";".join(selected_features),
        "test_accuracy": metrics["accuracy"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_macro_ovr_auc": metrics["macro_ovr_auc"],
        "test_min_class_recall": metrics["min_class_recall"],
        "test_brc_recall": point["brc_recall"],
        "test_cvx_recall": point["cvx_recall"],
        "test_endc_recall": point["endc_recall"],
    }
    pd.DataFrame([summary]).to_csv(paths["tables"] / "locked_reproduction_summary.csv", index=False)
    pd.DataFrame({"rank": range(1, len(selected_features) + 1), "protein": selected_features}).to_csv(
        paths["tables"] / "locked_selected_features.csv",
        index=False,
    )
    per_class_df.to_csv(paths["tables"] / "locked_per_class_recall_auc.csv", index=False)
    pd.DataFrame(classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)).transpose().to_csv(
        paths["tables"] / "locked_classification_report.csv"
    )
    confusion.to_csv(paths["tables"] / "locked_confusion_matrix.csv")
    bootstrap_df.to_csv(paths["tables"] / "bootstrap_metrics_1000.csv", index=False)
    ci_df.to_csv(paths["tables"] / "bootstrap_ci_summary.csv", index=False)
    candidate_feature_df.to_csv(paths["tables"] / "train_only_candidate_feature_bank.csv", index=False)
    ranking_table.to_csv(paths["tables"] / "train_only_ranker_table.csv", index=False)
    quota_df.to_csv(paths["tables"] / "train_only_quota_sources.csv", index=False)
    bonf_df.to_csv(paths["tables"] / "train_only_bonf_sources.csv", index=False)
    de_df.to_csv(paths["tables"] / "train_only_ovr_dea_all.csv", index=False)
    save_figures(paths, confusion, per_class_df, bootstrap_df)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "purpose": "Deterministic locked reproducer for the V11 trial_no=1 25-protein LR_L2 compact multiclass candidate with 1000 held-out bootstrap resamples.",
        "input": str(INPUT.resolve()),
        "output_dir": str(OUTPUT_ROOT.resolve()),
        "selection_rule": {
            "seed": SEED,
            "test_size": TEST_SIZE,
            "imputer": IMPUTER,
            "knn_neighbors": KNN_NEIGHBORS,
            "per_cancer_top_m": PER_CANCER_TOP_M,
            "min_cancer_coverage": MIN_CANCER_COVERAGE,
            "candidate_bank_size": CANDIDATE_BANK_SIZE,
            "ranker": RANKER,
            "feature_count": FEATURE_COUNT,
            "min_per_class": MIN_PER_CLASS,
            "n_shared": N_SHARED,
            "coach": COACH,
        },
        "expected_metrics": EXPECTED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "notes": [
            "Panel is regenerated from train-only DEA/rankers and asserted against the locked V11 trial_no=1 panel.",
            "Bootstrap CI resamples held-out test predictions only; it does not retrain or reselect features.",
            "This is an internal locked reproducer, not external clinical validation.",
        ],
    }
    with (paths["manifests"] / "vRSX_locked_reproducer_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    log(
        paths,
        (
            f"[seed {SEED}] test size={FEATURE_COUNT} coach={COACH} "
            f"BA={metrics['balanced_accuracy']:.4f} F1={metrics['macro_f1']:.4f} "
            f"minRecall={metrics['min_class_recall']:.4f}"
        ),
    )
    log(paths, f"vRSX complete: {paths['tables'] / 'locked_reproduction_summary.csv'}")


if __name__ == "__main__":
    main()
