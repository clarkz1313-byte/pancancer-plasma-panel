#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import SOURCE_INPUT

ROOT = Path(__file__).resolve().parents[2]
REVISE_ROOT = ROOT / "revise_plan"
PART_B_ROOT = REVISE_ROOT / "part_b_multiclass"
PART_A_PAN_ROOT = REVISE_ROOT / "part_a_pan_cancer"
PIPELINE_SCRIPTS = ROOT / "scripts"
if str(PIPELINE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PIPELINE_SCRIPTS))

import run_12class_ovr_pipeline as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the main 12-class multiclass model using the balanced hybrid train-only pan-feature assembly."
    )
    parser.add_argument("--input", type=Path, default=SOURCE_INPUT)
    parser.add_argument("--class-column", default="Cancer")
    parser.add_argument("--sample-id-column", default="Sample_ID")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--imputer", default="knn")
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--per-cancer-top-m", type=int, default=10)
    parser.add_argument("--min-cancer-coverage", type=int, default=2)
    parser.add_argument("--panel-max", type=int, default=18)
    return parser.parse_args()


def multiclass_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = PART_B_ROOT / f"hybrid_main_m{args.per_cancer_top_m}_k{args.min_cancer_coverage}"
    return {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures",
        "manifests": root / "manifests",
    }


def ensure_dirs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix_figure(matrix: pd.DataFrame, output_path: Path) -> None:
    figure, ax = base.plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.3, linecolor="white", ax=ax)
    ax.set_title("Pan-feature multiclass confusion matrix")
    ax.set_xlabel("Predicted cancer")
    ax.set_ylabel("True cancer")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    base.plt.close(figure)


def save_panel_size_performance_figure(panel_df: pd.DataFrame, output_path: Path) -> None:
    if panel_df.empty:
        return
    figure, ax = base.plt.subplots(figsize=(9.5, 5.5))
    sns.lineplot(data=panel_df, x="feature_count", y="test_accuracy", marker="o", linewidth=2, ax=ax, color="#2e8b57", label="Accuracy")
    sns.lineplot(data=panel_df, x="feature_count", y="test_macro_f1", marker="o", linewidth=2, ax=ax, color="#7ec8e3", label="Macro F1")
    sns.lineplot(data=panel_df, x="feature_count", y="test_macro_ovr_auc", marker="o", linewidth=2, ax=ax, color="#c97b63", label="Macro OVR AUC")
    ax.set_title("Multiclass performance vs panel size")
    ax.set_xlabel("Panel size")
    ax.set_ylabel("Metric value")
    ax.legend(loc="best", frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    base.plt.close(figure)


def save_single_vs_multiclass_auc_figure(comparison_df: pd.DataFrame, output_path: Path) -> None:
    if comparison_df.empty:
        return
    plot_df = comparison_df.melt(
        id_vars="target_class",
        value_vars=["single_best_auc", "multiclass_ovr_auc"],
        var_name="source",
        value_name="auc",
    )
    figure, ax = base.plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_df, x="target_class", y="auc", hue="source", ax=ax)
    ax.set_title("Part A single-model AUC vs Part B multiclass one-vs-rest AUC")
    ax.set_xlabel("Cancer")
    ax.set_ylabel("AUC")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="best", frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    base.plt.close(figure)


def build_classification_report_df(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})
    per_class_df = report_df.loc[report_df["label"].isin(labels)].reset_index(drop=True)
    summary_df = report_df.loc[~report_df["label"].isin(labels)].reset_index(drop=True)
    return per_class_df, summary_df


def build_ovr_auc_tables(
    y_true: pd.Series,
    y_score: np.ndarray,
    labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    auc_rows: list[dict[str, object]] = []
    y_true_values = y_true.to_numpy()
    for index, label in enumerate(labels):
        binary_true = (y_true_values == label).astype(int)
        auc_value = float(roc_auc_score(binary_true, y_score[:, index])) if len(np.unique(binary_true)) >= 2 else float("nan")
        auc_rows.append({"label": label, "ovr_auc": auc_value})
    per_class_df = pd.DataFrame(auc_rows)
    weights = np.asarray([(y_true_values == label).sum() for label in labels], dtype=float)
    summary_df = pd.DataFrame(
        [
            {
                "macro_ovr_auc": float(np.nanmean(per_class_df["ovr_auc"])),
                "weighted_ovr_auc": float(np.average(per_class_df["ovr_auc"].fillna(0.0), weights=weights)),
            }
        ]
    )
    return per_class_df, summary_df


def rank_pan_features_multiclass(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> pd.DataFrame:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    model = LogisticRegression(max_iter=3000, solver="lbfgs", C=0.1, random_state=seed)
    model.fit(x_scaled, y_train)
    coef = model.coef_
    importance = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    ranking_df = pd.DataFrame({"protein": x_train.columns.tolist(), "importance": importance})
    ranking_df = ranking_df.sort_values(["importance", "protein"], ascending=[False, True]).reset_index(drop=True)
    ranking_df["rank"] = np.arange(1, len(ranking_df) + 1)
    return ranking_df


def evaluate_multiclass_panel(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: list[str],
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray, list[str]]:
    estimator = GridSearchCV(
        Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, solver="lbfgs")),
            ]
        ),
        param_grid={"model__C": [0.01, 0.1, 1, 10]},
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    estimator.fit(x_train[feature_columns], y_train)
    y_pred = estimator.predict(x_test[feature_columns])
    y_score = estimator.predict_proba(x_test[feature_columns])
    class_order = estimator.best_estimator_.classes_.tolist()
    _, auc_summary_df = build_ovr_auc_tables(y_test, y_score, class_order)
    result = {
        "model": "Multinomial Logistic",
        "feature_count": len(feature_columns),
        "train_cv_score": float(estimator.best_score_),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "test_macro_ovr_auc": float(auc_summary_df.iloc[0]["macro_ovr_auc"]),
        "best_params": json.dumps(estimator.best_params_, sort_keys=True),
    }
    predictions = pd.DataFrame({"true_class": y_test.to_numpy(), "predicted_class": y_pred})
    return result, predictions, y_score, class_order


def collect_train_only_feature_sources(
    train_imputed: pd.DataFrame,
    protein_columns: list[str],
    class_column: str,
    classes: list[str],
    per_cancer_top_m: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bonf_rows: list[dict[str, object]] = []
    quota_rows: list[dict[str, object]] = []
    for target_class in classes:
        binary_train = train_imputed.copy()
        binary_train["Label"] = (binary_train[class_column] == target_class).astype(int)
        de_df = base.differential_expression(
            train_df=binary_train,
            protein_columns=protein_columns,
            case_mean_column=f"{target_class.lower()}_mean",
            case_n_column=f"{target_class.lower()}_n",
            control_n_column="rest_n",
        )
        bonf_df = de_df.loc[de_df["significant_bonf"], ["protein", "p_value", "p_adjusted_bonf", "fold_change"]].copy()
        bonf_df = bonf_df.sort_values(["p_adjusted_bonf", "p_value", "protein"]).reset_index(drop=True)
        for _, row in bonf_df.iterrows():
            bonf_rows.append(
                {
                    "target_class": target_class,
                    "protein": row["protein"],
                    "p_value": row["p_value"],
                    "p_adjusted_bonf": row["p_adjusted_bonf"],
                    "fold_change": row["fold_change"],
                    "selection_rule": "bonf_significant",
                }
            )
        quota_df = bonf_df.head(min(per_cancer_top_m, len(bonf_df))).copy()
        if quota_df.empty:
            fallback_df = de_df.loc[:, ["protein", "p_value", "p_adjusted_bonf", "fold_change"]].copy()
            quota_df = fallback_df.sort_values(["p_value", "protein"]).head(per_cancer_top_m).reset_index(drop=True)
            selection_rule = "quota_pvalue_fallback"
        else:
            selection_rule = "quota_bonf_top_m"
        for _, row in quota_df.iterrows():
            quota_rows.append(
                {
                    "target_class": target_class,
                    "protein": row["protein"],
                    "p_value": row["p_value"],
                    "p_adjusted_bonf": row["p_adjusted_bonf"],
                    "fold_change": row["fold_change"],
                    "selection_rule": selection_rule,
                }
            )
    return pd.DataFrame(bonf_rows), pd.DataFrame(quota_rows)


def build_candidate_feature_set(
    bonf_sources_df: pd.DataFrame,
    quota_sources_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage_df = (
        bonf_sources_df.groupby("protein", as_index=False)
        .agg(
            cancer_count=("target_class", "nunique"),
            cancers=("target_class", lambda series: ";".join(sorted(series.unique()))),
        )
        .sort_values(["cancer_count", "protein"], ascending=[False, True])
        .reset_index(drop=True)
    )
    quota_proteins = set(quota_sources_df["protein"].tolist())
    coverage_proteins = set(
        coverage_df.loc[coverage_df["cancer_count"] >= args.min_cancer_coverage, "protein"].tolist()
    )
    selected_proteins = sorted(quota_proteins | coverage_proteins)
    candidate_df = coverage_df.loc[coverage_df["protein"].isin(selected_proteins)].copy()
    candidate_df["candidate_rule"] = (
        f"balanced_hybrid_top_{args.per_cancer_top_m}_plus_min_coverage_{args.min_cancer_coverage}"
    )
    candidate_df = candidate_df.sort_values(["cancer_count", "protein"], ascending=[False, True]).reset_index(drop=True)
    candidate_df["candidate_rank_by_coverage"] = np.arange(1, len(candidate_df) + 1)
    return candidate_df, coverage_df


def save_model_outputs(
    output_prefix: str,
    paths: dict[str, Path],
    classes: list[str],
    sample_id_column: str,
    sample_ids: pd.Series,
    y_test: pd.Series,
    best_model,
    best_model_score: np.ndarray | None,
    best_predictions_df: pd.DataFrame,
) -> None:
    if best_model is None:
        return
    best_pred = best_model.predict(best_model.feature_names_in_ if False else None)


def main() -> None:
    args = parse_args()
    paths = multiclass_paths(args)
    ensure_dirs(paths)

    full_df = pd.read_csv(args.input)
    protein_columns = base.protein_columns_from_df(full_df, args.class_column, args.sample_id_column)
    classes = sorted(full_df[args.class_column].dropna().unique().tolist())

    train_idx, test_idx = train_test_split(
        full_df.index.to_numpy(),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=full_df[args.class_column],
    )
    train_df_raw = full_df.loc[train_idx].reset_index(drop=True)
    test_df_raw = full_df.loc[test_idx].reset_index(drop=True)

    imputer = base.build_imputer(args.imputer, args.knn_neighbors, args.seed)
    imputer.fit(train_df_raw[protein_columns])
    train_imputed = pd.concat(
        [
            train_df_raw[[args.sample_id_column, args.class_column]].reset_index(drop=True),
            pd.DataFrame(imputer.transform(train_df_raw[protein_columns]), columns=protein_columns),
        ],
        axis=1,
    )
    test_imputed = pd.concat(
        [
            test_df_raw[[args.sample_id_column, args.class_column]].reset_index(drop=True),
            pd.DataFrame(imputer.transform(test_df_raw[protein_columns]), columns=protein_columns),
        ],
        axis=1,
    )

    bonf_sources_df, quota_sources_df = collect_train_only_feature_sources(
        train_imputed=train_imputed,
        protein_columns=protein_columns,
        class_column=args.class_column,
        classes=classes,
        per_cancer_top_m=args.per_cancer_top_m,
    )
    bonf_sources_df.to_csv(paths["tables"] / "pan_feature_bonf_sources.csv", index=False)
    quota_sources_df.to_csv(paths["tables"] / "pan_feature_quota_sources.csv", index=False)

    candidate_df, coverage_df = build_candidate_feature_set(
        bonf_sources_df=bonf_sources_df,
        quota_sources_df=quota_sources_df,
        args=args,
    )
    candidate_df.to_csv(paths["tables"] / "pan_feature_candidates.csv", index=False)
    coverage_df.to_csv(paths["tables"] / "pan_feature_coverage_summary.csv", index=False)

    selected_features = candidate_df["protein"].tolist()
    x_train = train_imputed[selected_features]
    y_train = train_imputed[args.class_column]
    x_test = test_imputed[selected_features]
    y_test = test_imputed[args.class_column]

    models = {
        "Multinomial Logistic": GridSearchCV(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(max_iter=3000, solver="lbfgs")),
                ]
            ),
            param_grid={"model__C": [0.01, 0.1, 1, 10]},
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        ),
        "Random Forest": GridSearchCV(
            RandomForestClassifier(random_state=args.seed, n_jobs=-1),
            param_grid={"n_estimators": [200], "max_depth": [10, None], "min_samples_leaf": [1, 2]},
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        ),
    }

    multiclass_rows: list[dict[str, object]] = []
    best_model_name = ""
    best_model = None
    best_accuracy = -1.0
    best_model_score: np.ndarray | None = None
    best_model_class_order: list[str] = classes
    best_model_predictions_df = pd.DataFrame()

    for model_name, estimator in models.items():
        estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_test)
        y_score = estimator.predict_proba(x_test)
        class_order = estimator.best_estimator_.classes_.tolist()
        _, auc_summary_df = build_ovr_auc_tables(y_test, y_score, class_order)
        row = {
            "model": model_name,
            "train_cv_score": float(estimator.best_score_),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "test_macro_ovr_auc": float(auc_summary_df.iloc[0]["macro_ovr_auc"]),
            "feature_count": len(selected_features),
            "best_params": json.dumps(estimator.best_params_, sort_keys=True),
        }
        multiclass_rows.append(row)
        if row["test_accuracy"] > best_accuracy:
            best_accuracy = row["test_accuracy"]
            best_model_name = model_name
            best_model = estimator
            best_model_score = y_score
            best_model_class_order = class_order
            best_model_predictions_df = pd.DataFrame(
                {
                    args.sample_id_column: test_imputed[args.sample_id_column],
                    "true_class": y_test,
                    "predicted_class": y_pred,
                }
            )

    pd.DataFrame(multiclass_rows).to_csv(paths["tables"] / "multiclass_model_summary.csv", index=False)
    best_model_predictions_df.to_csv(paths["tables"] / "multiclass_test_predictions.csv", index=False)

    if best_model is not None:
        best_model_pred = best_model.predict(x_test)
        confusion = pd.DataFrame(
            confusion_matrix(y_test, best_model_pred, labels=classes),
            index=classes,
            columns=classes,
        )
        confusion.to_csv(paths["tables"] / "multiclass_confusion_matrix.csv")
        save_confusion_matrix_figure(confusion, paths["figures"] / "multiclass_confusion_matrix.png")
        per_class_df, summary_df = build_classification_report_df(y_test, best_model_pred, classes)
        per_class_df.to_csv(paths["tables"] / "multiclass_per_class_metrics.csv", index=False)
        summary_df.to_csv(paths["tables"] / "multiclass_summary_metrics.csv", index=False)
        if best_model_score is not None:
            auc_per_class_df, auc_summary_df = build_ovr_auc_tables(y_test, best_model_score, best_model_class_order)
            auc_per_class_df.to_csv(paths["tables"] / "multiclass_per_class_ovr_auc.csv", index=False)
            auc_summary_df.to_csv(paths["tables"] / "multiclass_summary_ovr_auc.csv", index=False)

    ranking_df = rank_pan_features_multiclass(x_train, y_train, args.seed)
    ranking_df.to_csv(paths["tables"] / "multiclass_feature_ranking.csv", index=False)

    panel_rows: list[dict[str, object]] = []
    panel_membership_rows: list[dict[str, object]] = []
    best_panel_accuracy = -1.0
    best_panel_name = ""
    best_panel_predictions_df = pd.DataFrame()
    best_panel_scores: np.ndarray | None = None
    best_panel_class_order: list[str] = classes

    for panel_size in range(1, min(args.panel_max, len(ranking_df)) + 1):
        features = ranking_df.head(panel_size)["protein"].tolist()
        result, predictions, y_score, class_order = evaluate_multiclass_panel(x_train, y_train, x_test, y_test, features)
        result["panel"] = f"Top {panel_size}"
        result["features"] = ";".join(features)
        panel_rows.append(result)
        for rank, protein in enumerate(features, start=1):
            panel_membership_rows.append(
                {
                    "panel": f"Top {panel_size}",
                    "feature_count": panel_size,
                    "panel_rank": rank,
                    "protein": protein,
                }
            )
        if result["test_accuracy"] > best_panel_accuracy:
            best_panel_accuracy = result["test_accuracy"]
            best_panel_name = f"Top {panel_size}"
            best_panel_predictions_df = predictions.copy()
            best_panel_scores = y_score
            best_panel_class_order = class_order

    all_features = ranking_df["protein"].tolist()
    all_result, all_predictions, all_scores, all_class_order = evaluate_multiclass_panel(x_train, y_train, x_test, y_test, all_features)
    all_result["panel"] = "All Selected"
    all_result["features"] = ";".join(all_features)
    panel_rows.append(all_result)
    for rank, protein in enumerate(all_features, start=1):
        panel_membership_rows.append(
            {
                "panel": "All Selected",
                "feature_count": len(all_features),
                "panel_rank": rank,
                "protein": protein,
            }
        )
    if all_result["test_accuracy"] > best_panel_accuracy:
        best_panel_accuracy = all_result["test_accuracy"]
        best_panel_name = "All Selected"
        best_panel_predictions_df = all_predictions.copy()
        best_panel_scores = all_scores
        best_panel_class_order = all_class_order

    panel_df = pd.DataFrame(panel_rows).sort_values(["feature_count", "panel"]).reset_index(drop=True)
    panel_df.to_csv(paths["tables"] / "multiclass_panel_performance.csv", index=False)
    pd.DataFrame(panel_membership_rows).to_csv(paths["tables"] / "multiclass_panel_membership.csv", index=False)
    save_panel_size_performance_figure(panel_df.loc[panel_df["panel"].str.startswith("Top ")], paths["figures"] / "multiclass_panel_performance.png")

    if not best_panel_predictions_df.empty:
        best_panel_predictions_df.insert(0, args.sample_id_column, test_imputed[args.sample_id_column].to_numpy())
        best_panel_predictions_df.to_csv(paths["tables"] / "multiclass_best_panel_test_predictions.csv", index=False)
        best_confusion = pd.DataFrame(
            confusion_matrix(best_panel_predictions_df["true_class"], best_panel_predictions_df["predicted_class"], labels=classes),
            index=classes,
            columns=classes,
        )
        best_confusion.to_csv(paths["tables"] / "multiclass_best_panel_confusion_matrix.csv")
        save_confusion_matrix_figure(best_confusion, paths["figures"] / "multiclass_best_panel_confusion_matrix.png")
        panel_per_class_df, panel_summary_df = build_classification_report_df(
            best_panel_predictions_df["true_class"],
            best_panel_predictions_df["predicted_class"],
            classes,
        )
        panel_per_class_df.to_csv(paths["tables"] / "multiclass_best_panel_per_class_metrics.csv", index=False)
        panel_summary_df.to_csv(paths["tables"] / "multiclass_best_panel_summary_metrics.csv", index=False)
        if best_panel_scores is not None:
            panel_auc_per_class_df, panel_auc_summary_df = build_ovr_auc_tables(
                best_panel_predictions_df["true_class"],
                best_panel_scores,
                best_panel_class_order,
            )
            panel_auc_per_class_df.to_csv(paths["tables"] / "multiclass_best_panel_per_class_ovr_auc.csv", index=False)
            panel_auc_summary_df.to_csv(paths["tables"] / "multiclass_best_panel_summary_ovr_auc.csv", index=False)

    single_summary_path = PART_A_PAN_ROOT / "tables" / "all_classes_reference_panel_summary.csv"
    multiclass_auc_path = paths["tables"] / "multiclass_per_class_ovr_auc.csv"
    if single_summary_path.exists() and multiclass_auc_path.exists():
        single_df = pd.read_csv(single_summary_path)[["target_class", "Test_AUC"]].rename(columns={"Test_AUC": "single_best_auc"})
        multiclass_auc_df = pd.read_csv(multiclass_auc_path).rename(columns={"label": "target_class", "ovr_auc": "multiclass_ovr_auc"})
        comparison_df = single_df.merge(multiclass_auc_df, on="target_class", how="inner").sort_values("target_class").reset_index(drop=True)
        comparison_df.to_csv(paths["tables"] / "single_vs_multiclass_auc_comparison.csv", index=False)
        save_single_vs_multiclass_auc_figure(comparison_df, paths["figures"] / "single_vs_multiclass_auc_comparison.png")

    manifest = {
        "source_input": str(args.input.resolve()),
        "class_column": args.class_column,
        "sample_id_column": args.sample_id_column,
        "seed": args.seed,
        "test_size": args.test_size,
        "imputer": args.imputer,
        "knn_neighbors": args.knn_neighbors,
        "feature_mode": "balanced_hybrid_main",
        "per_cancer_top_m": args.per_cancer_top_m,
        "min_cancer_coverage": args.min_cancer_coverage,
        "panel_max": args.panel_max,
        "dea_feature_filter": "bonf_only_train_features",
        "candidate_feature_count": len(selected_features),
        "best_model": best_model_name,
        "best_panel": best_panel_name,
        "output_root": str(paths["root"].resolve()),
    }
    (paths["manifests"] / "multiclass_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print("Completed multiclass run in mode: balanced_hybrid_main")
    print(f"Candidate features: {len(selected_features)}")
    print(f"Best model: {best_model_name}")
    print(f"Output folder: {paths['root']}")


if __name__ == "__main__":
    main()
