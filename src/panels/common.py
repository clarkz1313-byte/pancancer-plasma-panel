#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
REVISE_ROOT = ROOT / "revise_plan"
PART_A_SINGLE_ROOT = REVISE_ROOT / "part_a_single"
PART_A_PAN_ROOT = REVISE_ROOT / "part_a_pan_cancer"
SOURCE_INPUT = ROOT / "data" / "processed" / "filtered_pancancer_data.csv"
PIPELINE_SCRIPTS = ROOT / "scripts"
if str(PIPELINE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PIPELINE_SCRIPTS))

import run_12class_ovr_pipeline as base  # noqa: E402


CANCER_SLUGS = (
    "aml",
    "brc",
    "cll",
    "crc",
    "cvx",
    "endc",
    "gliom",
    "lungc",
    "lymph",
    "myel",
    "ovc",
    "prc",
)
SLUG_TO_LABEL = {
    "aml": "AML",
    "brc": "BRC",
    "cll": "CLL",
    "crc": "CRC",
    "cvx": "CVX",
    "endc": "ENDC",
    "gliom": "GLIOM",
    "lungc": "LUNGC",
    "lymph": "LYMPH",
    "myel": "MYEL",
    "ovc": "OVC",
    "prc": "PRC",
}
SLUG_TO_COLOR = {
    "aml": "#b22222",
    "brc": "#c97b63",
    "cll": "#7a3e9d",
    "crc": "#d68600",
    "cvx": "#c13d86",
    "endc": "#8e5d2c",
    "gliom": "#7ec8e3",
    "lungc": "#2e8b57",
    "lymph": "#3856a6",
    "myel": "#8c564b",
    "ovc": "#d1495b",
    "prc": "#008b8b",
}
LABEL_TO_SLUG = {label: slug for slug, label in SLUG_TO_LABEL.items()}
COMBINED_VOLCANO_LABEL_COUNT = 10


@dataclass(frozen=True)
class CancerPaths:
    slug: str
    root: Path
    data: Path
    tables: Path
    figures: Path
    logs: Path
    manifests: Path


def cancer_paths(slug: str) -> CancerPaths:
    root = PART_A_SINGLE_ROOT / slug
    return CancerPaths(
        slug=slug,
        root=root,
        data=root / "data",
        tables=root / "tables",
        figures=root / "figures",
        logs=root / "logs",
        manifests=root / "manifests",
    )


def pan_cancer_paths() -> dict[str, Path]:
    root = PART_A_PAN_ROOT
    return {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures",
        "manifests": root / "manifests",
    }


def ensure_cancer_dirs(paths: CancerPaths) -> None:
    for path in (paths.root, paths.data, paths.tables, paths.figures, paths.logs, paths.manifests):
        path.mkdir(parents=True, exist_ok=True)


def ensure_pan_dirs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def cancer_menu_text() -> str:
    lines = ["Cancer selection menu:"]
    for index, slug in enumerate(CANCER_SLUGS, start=1):
        lines.append(f"{index:2d}. {slug} ({SLUG_TO_LABEL[slug]})")
    return "\n".join(lines)


def slug_from_index(index: int) -> str:
    if not 1 <= index <= len(CANCER_SLUGS):
        raise ValueError(f"Invalid cancer index {index}. Valid range is 1-{len(CANCER_SLUGS)}.")
    return CANCER_SLUGS[index - 1]


def slugs_from_indexes(indexes: list[int]) -> list[str]:
    return [slug_from_index(index) for index in indexes]


def build_target_vs_rest_dataset(
    full_df: pd.DataFrame,
    class_column: str,
    target_label: str,
) -> pd.DataFrame:
    dataset = full_df.copy()
    dataset["Label"] = (dataset[class_column] == target_label).astype(int)
    return dataset


def split_and_impute_binary(
    df: pd.DataFrame,
    protein_columns: list[str],
    class_column: str,
    sample_id_column: str,
    seed: int,
    test_size: float,
    imputer_method: str,
    knn_neighbors: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df[protein_columns].copy()
    y = df["Label"].copy()
    metadata_columns = [column for column in [sample_id_column, class_column] if column in df.columns]
    metadata = df[metadata_columns].copy()

    x_train, x_test, y_train, y_test, meta_train, meta_test = base.train_test_split(
        x,
        y,
        metadata,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    imputer = base.build_imputer(imputer_method, knn_neighbors, seed)
    imputer.fit(x_train)

    x_train_imputed = pd.DataFrame(
        imputer.transform(x_train),
        columns=protein_columns,
        index=x_train.index,
    )
    x_test_imputed = pd.DataFrame(
        imputer.transform(x_test),
        columns=protein_columns,
        index=x_test.index,
    )

    train_df = pd.concat(
        [
            meta_train.reset_index(drop=True),
            y_train.reset_index(drop=True).rename("Label"),
            x_train_imputed.reset_index(drop=True),
        ],
        axis=1,
    )
    test_df = pd.concat(
        [
            meta_test.reset_index(drop=True),
            y_test.reset_index(drop=True).rename("Label"),
            x_test_imputed.reset_index(drop=True),
        ],
        axis=1,
    )
    return train_df, test_df


def summarize_missingness(df: pd.DataFrame, protein_columns: list[str], stage: str) -> pd.DataFrame:
    feature_df = df[protein_columns]
    sample_missing = feature_df.isna().sum(axis=1)
    protein_missing = feature_df.isna().sum(axis=0)
    rows = [
        {"stage": stage, "scope": "dataset", "metric": "samples", "value": int(len(df))},
        {"stage": stage, "scope": "dataset", "metric": "proteins", "value": int(len(protein_columns))},
        {"stage": stage, "scope": "dataset", "metric": "total_missing_values", "value": int(feature_df.isna().sum().sum())},
        {"stage": stage, "scope": "dataset", "metric": "mean_missing_per_sample", "value": float(sample_missing.mean())},
        {"stage": stage, "scope": "dataset", "metric": "median_missing_per_sample", "value": float(sample_missing.median())},
        {"stage": stage, "scope": "dataset", "metric": "max_missing_per_sample", "value": float(sample_missing.max())},
        {"stage": stage, "scope": "dataset", "metric": "mean_missing_per_protein", "value": float(protein_missing.mean())},
        {"stage": stage, "scope": "dataset", "metric": "median_missing_per_protein", "value": float(protein_missing.median())},
        {"stage": stage, "scope": "dataset", "metric": "max_missing_per_protein", "value": float(protein_missing.max())},
    ]
    return pd.DataFrame(rows)


def build_cohort_description(
    df: pd.DataFrame,
    class_column: str,
    target_label: str,
) -> pd.DataFrame:
    composition = (
        df[class_column]
        .value_counts(dropna=False)
        .rename_axis("source_cancer")
        .reset_index(name="n_samples")
        .sort_values(["n_samples", "source_cancer"], ascending=[False, True])
        .reset_index(drop=True)
    )
    composition["proportion"] = composition["n_samples"] / composition["n_samples"].sum()
    composition["is_target"] = composition["source_cancer"] == target_label
    composition["Label"] = composition["is_target"].astype(int)
    return composition


def select_feature_pool_bonf_only(
    de_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> tuple[list[str], str]:
    pool = [protein for protein in de_df.loc[de_df["significant_bonf"], "protein"].tolist() if protein in train_df.columns]
    if pool:
        return pool, "bonf"
    fallback = [protein for protein in de_df["protein"].tolist() if protein in train_df.columns][: base.FALLBACK_FEATURE_COUNT]
    return fallback, "p_value_fallback"


def build_panel_membership_table(
    target_class: str,
    panel_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in panel_df.iterrows():
        proteins = [protein for protein in str(row["proteins"]).split(";") if protein]
        for rank, protein in enumerate(proteins, start=1):
            rows.append(
                {
                    "target_class": target_class,
                    "panel": row["Model"],
                    "panel_size": int(row["Features"]),
                    "panel_rank": rank,
                    "protein": protein,
                }
            )
    return pd.DataFrame(rows)


def target_color(target_class: str) -> str:
    slug = LABEL_TO_SLUG.get(target_class, target_class.lower())
    return SLUG_TO_COLOR.get(slug, "#4c78a8")


def save_volcano_figure(target_class: str, de_df: pd.DataFrame, output_path: Path) -> None:
    if de_df.empty:
        return
    accent = target_color(target_class)
    plot_df = de_df.copy()
    plot_df["minus_log10_p"] = plot_df["p_value"].clip(lower=1e-300).map(lambda value: -math.log10(value))
    plot_df["significance"] = plot_df["significant_bonf"].map({True: "Bonferroni significant", False: "Not significant"})
    figure, ax = base.plt.subplots(figsize=(8.5, 6))
    sns.scatterplot(
        data=plot_df,
        x="fold_change",
        y="minus_log10_p",
        hue="significance",
        palette={"Bonferroni significant": accent, "Not significant": "#c5c8cc"},
        s=24,
        linewidth=0,
        ax=ax,
    )
    ax.set_title(f"{target_class} DEA volcano plot")
    ax.set_xlabel("Fold change")
    ax.set_ylabel("-log10(p-value)")
    ax.legend(loc="best", frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    base.plt.close(figure)


def save_pvalue_histogram(target_class: str, de_df: pd.DataFrame, output_path: Path) -> None:
    if de_df.empty:
        return
    accent = target_color(target_class)
    figure, ax = base.plt.subplots(figsize=(8, 5))
    ax.hist(de_df["p_value"].dropna(), bins=40, color=accent, edgecolor="white")
    ax.set_title(f"{target_class} DEA p-value histogram")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Protein count")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    base.plt.close(figure)


def save_panel_auc_figure_themed(target_class: str, panel_df: pd.DataFrame, output_path: Path) -> None:
    if panel_df.empty:
        return
    accent = target_color(target_class)
    figure, ax = base.plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=panel_df, x="Model", y="Test_AUC", ax=ax, color=accent)
    ax.set_title(f"{target_class} one-vs-rest panel performance")
    ax.set_xlabel("Panel")
    ax.set_ylabel("Test AUC")
    ax.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    base.plt.close(figure)


def build_revised_panel_definitions(
    target_class: str,
    top_features: list[str],
    all_selected_features: list[str],
    panel_sizes: list[int],
    include_all_panel: bool,
) -> list[tuple[str, list[str]]]:
    panels: list[tuple[str, list[str]]] = []
    seen_names: set[str] = set()

    for size in panel_sizes if panel_sizes else []:
        features = top_features[: min(size, len(top_features))]
        if not features:
            continue
        name = f"{target_class} Top {len(features)}"
        if name in seen_names:
            continue
        panels.append((name, features))
        seen_names.add(name)

    if include_all_panel and all_selected_features:
        name = f"{target_class} All Selected"
        if name not in seen_names:
            panels.append((name, all_selected_features))
            seen_names.add(name)

    return panels


def build_panel_outputs_revised(
    target_class: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    top_features: list[str],
    all_selected_features: list[str],
    panel_sizes: list[int],
    seed: int,
    sample_id_column: str,
    class_column: str,
    include_all_panel: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    panel_definitions = build_revised_panel_definitions(
        target_class=target_class,
        top_features=top_features,
        all_selected_features=all_selected_features,
        panel_sizes=panel_sizes,
        include_all_panel=include_all_panel,
    )
    panel_results: list[dict[str, object]] = []
    plot_payloads: list[dict[str, object]] = []

    for name, proteins in panel_definitions:
        result = base.evaluate_panel(name, proteins, train_df, test_df, seed)
        if result is None:
            continue
        result["sample_ids"] = test_df[sample_id_column].tolist() if sample_id_column in test_df.columns else list(test_df.index)
        result["source_cancers"] = test_df[class_column].tolist() if class_column in test_df.columns else []
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

    panel_df = pd.DataFrame(panel_results)
    if not panel_df.empty:
        panel_df = panel_df.sort_values(["Features", "Model"]).reset_index(drop=True)

    ranking_df = pd.DataFrame({"protein": top_features[: base.TOP_FEATURE_CAP], "rank": range(1, len(top_features[: base.TOP_FEATURE_CAP]) + 1)})
    if not ranking_df.empty:
        ranking_df["target_class"] = target_class

    selected_columns: list[str] = ["Label"]
    for column in [sample_id_column, class_column]:
        if column in train_df.columns and column not in selected_columns:
            selected_columns.append(column)
    for feature in top_features[: base.TOP_FEATURE_CAP]:
        if feature not in selected_columns:
            selected_columns.append(feature)
    panel_dataset = pd.concat(
        [
            train_df[selected_columns].assign(partition="dev"),
            test_df[selected_columns].assign(partition="test"),
        ],
        ignore_index=True,
    )
    return panel_df, ranking_df, panel_dataset, plot_payloads


def save_panel_roc_figure(target_class: str, plot_payloads: list[dict[str, object]], output_path: Path) -> None:
    if not plot_payloads:
        return
    accent = target_color(target_class)
    figure, ax = base.plt.subplots(figsize=(9, 6))
    palette = sns.light_palette(accent, n_colors=max(len(plot_payloads) + 2, 4), reverse=True)
    for color, payload in zip(palette, plot_payloads):
        fpr, tpr, _ = base.roc_curve(payload["y_test"], payload["y_score"])
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            color=color,
            label=f"{payload['Model']} (AUC={float(payload['Test_AUC']):.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
    ax.set_title(f"{target_class} panel ROC curves")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    base.plt.close(figure)


def save_test_predictions(
    target_class: str,
    plot_payloads: list[dict[str, object]],
    output_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for payload in plot_payloads:
        sample_ids = payload.get("sample_ids", [])
        source_cancers = payload.get("source_cancers", [])
        y_true = payload["y_test"]
        y_score = payload["y_score"]
        youden_threshold = payload["train_youden_threshold"]
        for index, (sample_id, source_cancer, y_value, score_value) in enumerate(
            zip(sample_ids, source_cancers, y_true, y_score)
        ):
            rows.append(
                {
                    "target_class": target_class,
                    "panel": payload["Model"],
                    "sample_rank": index + 1,
                    "Sample_ID": sample_id,
                    "source_cancer": source_cancer,
                    "partition": "test",
                    "y_true": int(y_value),
                    "y_score": float(score_value),
                    "y_pred_0_5": int(score_value >= 0.5),
                    "y_pred_youden": int(score_value >= youden_threshold),
                    "train_youden_threshold": float(youden_threshold),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def combined_volcano_marker_labels(result: dict[str, Any], plot_df: pd.DataFrame) -> pd.DataFrame:
    ranking_df = result.get("ranking_df")
    if ranking_df is None or ranking_df.empty:
        reference_payload = result.get("reference_payload") or {}
        proteins = list(reference_payload.get("proteins", []))[:COMBINED_VOLCANO_LABEL_COUNT]
        ranking_df = pd.DataFrame({"protein": proteins, "rank": range(1, len(proteins) + 1)})
    required_columns = {"protein", "rank"}
    if ranking_df.empty or not required_columns.issubset(ranking_df.columns):
        return pd.DataFrame()

    label_rank_df = (
        ranking_df.loc[:, ["protein", "rank"]]
        .dropna(subset=["protein", "rank"])
        .sort_values("rank")
        .head(COMBINED_VOLCANO_LABEL_COUNT)
    )
    label_df = plot_df.merge(label_rank_df, on="protein", how="inner")
    return label_df.sort_values("rank").reset_index(drop=True)


def save_combined_volcano_figure(results: list[dict[str, Any]], output_path: Path) -> None:
    if not results:
        return
    figure, axes = base.plt.subplots(3, 4, figsize=(18, 13))
    axes_flat = axes.flatten()
    for ax, result in zip(axes_flat, sorted(results, key=lambda item: item["label"])):
        de_df = result.get("de_df")
        if de_df is None or de_df.empty:
            ax.axis("off")
            continue
        target_class = result["label"]
        accent = target_color(target_class)
        plot_df = de_df.copy()
        plot_df["minus_log10_p"] = plot_df["p_value"].clip(lower=1e-300).map(lambda value: -math.log10(value))
        plot_df["significant"] = plot_df["significant_bonf"]
        ax.scatter(plot_df.loc[~plot_df["significant"], "fold_change"], plot_df.loc[~plot_df["significant"], "minus_log10_p"], s=8, color="#d0d3d8", alpha=0.8)
        ax.scatter(plot_df.loc[plot_df["significant"], "fold_change"], plot_df.loc[plot_df["significant"], "minus_log10_p"], s=10, color=accent, alpha=0.9)
        label_df = combined_volcano_marker_labels(result, plot_df)
        if not label_df.empty:
            ax.scatter(
                label_df["fold_change"],
                label_df["minus_log10_p"],
                s=24,
                color=accent,
                edgecolors="#202124",
                linewidths=0.4,
                zorder=4,
            )
            ax.margins(x=0.08, y=0.12)
            x_min, x_max = ax.get_xlim()
            x_mid = (x_min + x_max) / 2
            y_offsets = (5, 10, -8, -14, 15)
            for index, marker in label_df.iterrows():
                x_value = float(marker["fold_change"])
                y_value = float(marker["minus_log10_p"])
                ha = "right" if x_value > x_mid else "left"
                x_offset = -4 if ha == "right" else 4
                ax.annotate(
                    str(marker["protein"]),
                    xy=(x_value, y_value),
                    xytext=(x_offset, y_offsets[index % len(y_offsets)]),
                    textcoords="offset points",
                    fontsize=5,
                    color="#202124",
                    ha=ha,
                    va="center",
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75},
                    zorder=5,
                )
        ax.set_title(target_class, color=accent, fontsize=11, fontweight="bold")
        ax.set_xlabel("Fold change", fontsize=8)
        ax.set_ylabel("-log10(p)", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes_flat[len(results):]:
        ax.axis("off")
    figure.suptitle("Pan-cancer Bonferroni volcano overview", fontsize=16, y=0.995)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    base.plt.close(figure)


def save_feature_overlap_heatmap(overlap_df: pd.DataFrame, output_path: Path) -> None:
    if overlap_df.empty:
        return
    jaccard = pd.DataFrame(index=overlap_df.index, columns=overlap_df.index, dtype=float)
    for left in overlap_df.index:
        left_set = set(overlap_df.columns[overlap_df.loc[left] > 0])
        for right in overlap_df.index:
            right_set = set(overlap_df.columns[overlap_df.loc[right] > 0])
            union = left_set | right_set
            jaccard.loc[left, right] = len(left_set & right_set) / len(union) if union else 0.0
    figure, ax = base.plt.subplots(figsize=(10, 8))
    sns.heatmap(jaccard.astype(float), cmap="YlGnBu", linewidths=0.3, linecolor="white", ax=ax)
    ax.set_title("Pan-cancer feature overlap (Jaccard)")
    ax.set_xlabel("Target class")
    ax.set_ylabel("Target class")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    base.plt.close(figure)


def save_all_class_roc_figure_themed(reference_payloads: list[dict[str, object]], output_path: Path) -> None:
    if not reference_payloads:
        return
    figure, ax = base.plt.subplots(figsize=(10.5, 8))
    for payload in sorted(reference_payloads, key=lambda item: str(item["target_class"])):
        y_true = base.np.asarray(payload["y_test"])
        y_score = base.np.asarray(payload["y_score"])
        fpr, tpr, _ = base.roc_curve(y_true, y_score)
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            color=target_color(str(payload["target_class"])),
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
    base.plt.close(figure)


def save_performance_vs_panel_size_figure_themed(panel_df: pd.DataFrame, output_path: Path) -> None:
    if panel_df.empty:
        return
    performance_df = panel_df.sort_values(["target_class", "Features"]).copy()
    performance_df = performance_df.loc[performance_df["Model"].str.contains("Top ")]
    mean_df = (
        performance_df.groupby("Features", as_index=False)
        .agg(Test_AUC=("Test_AUC", "mean"))
        .sort_values("Features")
        .reset_index(drop=True)
    )
    figure, ax = base.plt.subplots(figsize=(11, 6.5))
    for target_class in sorted(performance_df["target_class"].unique()):
        subset = performance_df.loc[performance_df["target_class"] == target_class]
        ax.plot(
            subset["Features"],
            subset["Test_AUC"],
            marker="o",
            linewidth=1.8,
            color=target_color(target_class),
            label=target_class,
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
    base.plt.close(figure)


def panel_assay_feasibility(panel_size: int) -> str:
    if panel_size <= 3:
        return "high"
    if panel_size <= 6:
        return "medium"
    return "low"


def prototype_priority(assay_feasibility: str, single_auc: float, multi_auc: float) -> str:
    if assay_feasibility == "high" and single_auc >= 0.95 and multi_auc >= 0.9:
        return "high"
    if assay_feasibility in {"high", "medium"} and single_auc >= 0.9 and multi_auc >= 0.85:
        return "medium"
    return "low"


def drop_severity(delta_auc: float, delta_recall: float) -> str:
    if delta_auc >= -0.02 and delta_recall >= -0.05:
        return "mild"
    if delta_auc >= -0.05 and delta_recall >= -0.15:
        return "moderate"
    return "high"


def drop_interpretation(severity: str) -> str:
    if severity == "mild":
        return "Single-class signal remains largely usable in multiclass."
    if severity == "moderate":
        return "Marker set weakens in multiclass and needs panel support."
    return "Strong single-class signal does not transfer cleanly to multiclass."


def recommend_strategy(delta_auc: float, delta_recall: float, deploy_size: int) -> str:
    if delta_auc >= -0.02 and delta_recall >= -0.05:
        return "shared_multiclass_panel_candidate"
    if deploy_size <= 3:
        return "small_dedicated_panel"
    return "disease_specific_minimal_panel"


def preferred_use_case(strategy: str) -> str:
    if strategy == "shared_multiclass_panel_candidate":
        return "shared_multiclass_triage"
    if strategy == "small_dedicated_panel":
        return "disease_specific_triage"
    return "disease_specific_confirmatory_panel"


def clinical_rationale_stub(is_class_specific: bool, top1_cancer_count: float | int) -> str:
    if is_class_specific:
        return "needs_review: class-specific candidate with higher face validity"
    if top1_cancer_count <= 2:
        return "needs_review: limited sharing across classes, plausible supportive marker"
    return "needs_review: shared marker, verify disease specificity before kit inclusion"


def build_panel_review_tables(
    pan_paths: dict[str, Path],
    aggregate_df: pd.DataFrame,
    combined_panel_df: pd.DataFrame,
    top1_summary_df: pd.DataFrame,
) -> None:
    panel_dir = pan_paths["tables"] / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)

    multiclass_root = REVISE_ROOT / "part_b_multiclass" / "hybrid_main_m10_k2" / "tables"
    required_multiclass_files = [
        multiclass_root / "multiclass_best_panel_per_class_metrics.csv",
        multiclass_root / "multiclass_best_panel_per_class_ovr_auc.csv",
        multiclass_root / "multiclass_feature_ranking.csv",
    ]
    if not all(path.exists() for path in required_multiclass_files):
        return

    multiclass_metrics_df = pd.read_csv(multiclass_root / "multiclass_best_panel_per_class_metrics.csv").rename(
        columns={
            "label": "target_class",
            "precision": "multi_precision",
            "recall": "multi_recall",
            "f1-score": "multi_f1",
            "support": "multi_support",
        }
    )
    multiclass_auc_df = pd.read_csv(multiclass_root / "multiclass_best_panel_per_class_ovr_auc.csv").rename(
        columns={"label": "target_class", "ovr_auc": "multi_ovr_auc"}
    )
    multiclass_ranking_df = pd.read_csv(multiclass_root / "multiclass_feature_ranking.csv").rename(
        columns={"importance": "multiclass_importance", "rank": "multiclass_rank"}
    )
    top1_review_path = pan_paths["tables"] / "top1_marker_review_shortlist.csv"
    if top1_review_path.exists():
        top1_review_df = pd.read_csv(top1_review_path)
    else:
        top1_review_df = top1_summary_df.copy()
        top1_review_df["is_class_specific"] = False
        top1_review_df["shared_with_classes"] = ""

    max_balanced_by_class = (
        combined_panel_df.groupby("target_class", as_index=False)
        .agg(max_balanced_accuracy=("balanced_accuracy", "max"))
    )
    max_auc_by_class = (
        combined_panel_df.groupby("target_class", as_index=False)
        .agg(max_test_auc=("Test_AUC", "max"))
    )
    stat_candidate_df = (
        combined_panel_df.merge(max_balanced_by_class, on="target_class", how="left")
        .merge(max_auc_by_class, on="target_class", how="left")
    )
    stat_candidate_df = stat_candidate_df.loc[
        (stat_candidate_df["Test_AUC"] >= stat_candidate_df["max_test_auc"] - 0.02)
        & (stat_candidate_df["balanced_accuracy"] >= stat_candidate_df["max_balanced_accuracy"] - 0.05)
    ].copy()
    stat_min_df = (
        stat_candidate_df.sort_values(["target_class", "Features"])
        .groupby("target_class", as_index=False)
        .first()[["target_class", "Features"]]
        .rename(columns={"Features": "n_stat_min"})
    )

    minimal_panel_df = (
        aggregate_df[["target_class", "best_panel", "best_panel_test_auc"]]
        .merge(
            top1_review_df[
                [
                    "target_class",
                    "top1_protein",
                    "top1_importance",
                    "top1_test_auc",
                    "top1_cancer_count",
                    "is_class_specific",
                    "shared_with_classes",
                ]
            ],
            on="target_class",
            how="left",
        )
        .merge(stat_min_df, on="target_class", how="left")
        .merge(multiclass_metrics_df, on="target_class", how="left")
        .merge(multiclass_auc_df, on="target_class", how="left")
    )
    minimal_panel_df["n_stat_min"] = minimal_panel_df["n_stat_min"].fillna(1).astype(int)
    minimal_panel_df["n_clinical_min"] = minimal_panel_df.apply(
        lambda row: max(
            int(row["n_stat_min"]),
            2 if (not bool(row["is_class_specific"])) and float(row["top1_cancer_count"]) > 1 else 1,
        ),
        axis=1,
    )
    minimal_panel_df["n_deploy"] = minimal_panel_df["n_clinical_min"]

    deploy_panel_metrics = combined_panel_df[
        [
            "target_class",
            "Features",
            "Test_AUC",
            "recall",
            "specificity",
            "balanced_accuracy",
            "precision",
            "f1",
            "proteins",
        ]
    ].rename(
        columns={
            "Features": "n_deploy",
            "Test_AUC": "single_deploy_auc",
            "recall": "single_deploy_recall",
            "specificity": "single_deploy_specificity",
            "balanced_accuracy": "single_deploy_balanced_accuracy",
            "precision": "single_deploy_precision",
            "f1": "single_deploy_f1",
            "proteins": "deploy_proteins",
        }
    )
    minimal_panel_df = minimal_panel_df.merge(
        deploy_panel_metrics,
        on=["target_class", "n_deploy"],
        how="left",
    )
    minimal_panel_df["single_to_multi_drop_auc"] = minimal_panel_df["multi_ovr_auc"] - minimal_panel_df["single_deploy_auc"]
    minimal_panel_df["single_to_multi_drop_recall"] = minimal_panel_df["multi_recall"] - minimal_panel_df["single_deploy_recall"]
    minimal_panel_df["recommended_strategy"] = minimal_panel_df.apply(
        lambda row: recommend_strategy(float(row["single_to_multi_drop_auc"]), float(row["single_to_multi_drop_recall"]), int(row["n_deploy"])),
        axis=1,
    )
    minimal_panel_df["assay_feasibility"] = minimal_panel_df["n_deploy"].map(panel_assay_feasibility)
    minimal_panel_df["clinical_rationale"] = minimal_panel_df.apply(
        lambda row: clinical_rationale_stub(bool(row["is_class_specific"]), float(row["top1_cancer_count"])),
        axis=1,
    )
    minimal_panel_df = minimal_panel_df[
        [
            "target_class",
            "top1_protein",
            "top1_test_auc",
            "best_panel",
            "best_panel_test_auc",
            "n_stat_min",
            "n_clinical_min",
            "n_deploy",
            "single_deploy_auc",
            "single_deploy_recall",
            "single_deploy_specificity",
            "multi_ovr_auc",
            "multi_recall",
            "multi_precision",
            "multi_f1",
            "single_to_multi_drop_auc",
            "single_to_multi_drop_recall",
            "recommended_strategy",
            "assay_feasibility",
            "clinical_rationale",
            "deploy_proteins",
        ]
    ].sort_values("target_class").reset_index(drop=True)
    minimal_panel_df.to_csv(panel_dir / "minimal_panel_decision_per_cancer.csv", index=False)

    marker_rows: list[dict[str, object]] = []
    for row in minimal_panel_df.to_dict(orient="records"):
        target_class = str(row["target_class"])
        deploy_proteins = [protein for protein in str(row["deploy_proteins"]).split(";") if protein]
        class_ranking_df = combined_panel_df.loc[
            (combined_panel_df["target_class"] == target_class)
            & (combined_panel_df["Features"] == int(row["n_deploy"]))
        ]
        deploy_panel_label = class_ranking_df.iloc[0]["Model"] if not class_ranking_df.empty else f"{target_class} Top {int(row['n_deploy'])}"
        for rank, protein in enumerate(deploy_proteins, start=1):
            multi_hit = multiclass_ranking_df.loc[multiclass_ranking_df["protein"] == protein]
            multiclass_rank = int(multi_hit.iloc[0]["multiclass_rank"]) if not multi_hit.empty else math.nan
            multiclass_importance = float(multi_hit.iloc[0]["multiclass_importance"]) if not multi_hit.empty else math.nan
            shared_info = top1_review_df.loc[
                (top1_review_df["target_class"] == target_class) & (top1_review_df["top1_protein"] == protein),
                ["top1_cancer_count", "shared_with_classes", "is_class_specific"],
            ]
            if not shared_info.empty:
                is_class_specific = bool(shared_info.iloc[0]["is_class_specific"])
                shared_with_classes = str(shared_info.iloc[0]["shared_with_classes"])
            else:
                class_specific_path = panel_dir.parent / "per_class_specific_selected_features.csv"
                shared_path = panel_dir.parent / "per_class_shared_selected_features.csv"
                is_class_specific = False
                shared_with_classes = ""
                if class_specific_path.exists():
                    class_specific_df = pd.read_csv(class_specific_path)
                    is_class_specific = not class_specific_df.loc[
                        (class_specific_df["target_class"] == target_class) & (class_specific_df["protein"] == protein)
                    ].empty
                if not is_class_specific and shared_path.exists():
                    shared_df = pd.read_csv(shared_path)
                    shared_match = shared_df.loc[
                        (shared_df["target_class"] == target_class) & (shared_df["protein"] == protein)
                    ]
                    if not shared_match.empty:
                        shared_with_classes = ";".join(
                            class_name
                            for class_name in str(shared_match.iloc[0]["cancers"]).split(";")
                            if class_name and class_name != target_class
                        )
            marker_rows.append(
                {
                    "target_class": target_class,
                    "protein": protein,
                    "rank_within_deploy_panel": rank,
                    "deploy_panel": deploy_panel_label,
                    "panel_membership": f"Top {int(row['n_deploy'])}",
                    "is_class_specific": is_class_specific,
                    "shared_with_classes": shared_with_classes,
                    "single_class_value": float(row["single_deploy_auc"]),
                    "multi_class_value": float(row["multi_ovr_auc"]),
                    "multiclass_rank": multiclass_rank,
                    "multiclass_importance": multiclass_importance,
                    "biological_plausibility": "needs_review",
                    "clinical_plausibility": "needs_review",
                    "marker_role": "lead_marker" if rank == 1 else "supportive_marker",
                    "assay_feasibility": panel_assay_feasibility(rank),
                    "keep_decision": "keep_candidate",
                }
            )
    marker_audit_df = pd.DataFrame(marker_rows).sort_values(["target_class", "rank_within_deploy_panel"]).reset_index(drop=True)
    marker_audit_df.to_csv(panel_dir / "marker_audit_table.csv", index=False)

    single_vs_multi_df = minimal_panel_df[
        [
            "target_class",
            "n_deploy",
            "single_deploy_auc",
            "single_deploy_recall",
            "single_deploy_specificity",
            "multi_ovr_auc",
            "multi_recall",
            "multi_precision",
            "multi_f1",
            "single_to_multi_drop_auc",
            "single_to_multi_drop_recall",
        ]
    ].copy()
    single_vs_multi_df["drop_severity"] = single_vs_multi_df.apply(
        lambda row: drop_severity(float(row["single_to_multi_drop_auc"]), float(row["single_to_multi_drop_recall"])),
        axis=1,
    )
    single_vs_multi_df["interpretation"] = single_vs_multi_df["drop_severity"].map(drop_interpretation)
    single_vs_multi_df.to_csv(panel_dir / "single_vs_multi_drop_table.csv", index=False)

    final_kit_df = minimal_panel_df[
        [
            "target_class",
            "n_deploy",
            "deploy_proteins",
            "recommended_strategy",
            "assay_feasibility",
            "clinical_rationale",
            "multi_ovr_auc",
            "multi_recall",
            "single_deploy_auc",
        ]
    ].copy()
    final_kit_df["final_panel_type"] = final_kit_df["recommended_strategy"]
    final_kit_df["final_panel_size"] = final_kit_df["n_deploy"]
    final_kit_df["final_proteins"] = final_kit_df["deploy_proteins"]
    final_kit_df["single_class_ready"] = final_kit_df["single_deploy_auc"] >= 0.9
    final_kit_df["multi_class_ready"] = (final_kit_df["multi_ovr_auc"] >= 0.9) & (final_kit_df["multi_recall"] >= 0.7)
    final_kit_df["preferred_use_case"] = final_kit_df["recommended_strategy"].map(preferred_use_case)
    final_kit_df["prototype_priority"] = final_kit_df.apply(
        lambda row: prototype_priority(str(row["assay_feasibility"]), float(row["single_deploy_auc"]), float(row["multi_ovr_auc"])),
        axis=1,
    )
    final_kit_df = final_kit_df[
        [
            "target_class",
            "final_panel_type",
            "final_panel_size",
            "final_proteins",
            "single_class_ready",
            "multi_class_ready",
            "preferred_use_case",
            "clinical_rationale",
            "assay_feasibility",
            "prototype_priority",
        ]
    ].sort_values("target_class").reset_index(drop=True)
    final_kit_df.to_csv(panel_dir / "final_kit_strategy_table.csv", index=False)


def run_single_cancer_pipeline(
    cancer_slug: str,
    input_path: Path,
    class_column: str,
    sample_id_column: str,
    seed: int,
    test_size: float,
    imputer: str,
    knn_neighbors: int,
    panel_sizes: list[int],
    include_all_panel: bool,
    run_enrichment: bool,
    run_robustness: bool,
    bootstrap_samples: int,
) -> dict[str, Any]:
    target_label = SLUG_TO_LABEL[cancer_slug]
    paths = cancer_paths(cancer_slug)
    ensure_cancer_dirs(paths)

    full_df = pd.read_csv(input_path)
    if class_column not in full_df.columns:
        raise ValueError(f"Column '{class_column}' was not found in {input_path}.")
    if sample_id_column not in full_df.columns:
        raise ValueError(f"Column '{sample_id_column}' was not found in {input_path}.")

    target_df = build_target_vs_rest_dataset(full_df, class_column, target_label)
    positive_count = int(target_df["Label"].sum())
    negative_count = int(len(target_df) - positive_count)
    if positive_count < 2 or negative_count < 2:
        raise ValueError(
            f"Target cancer {target_label} does not have enough samples for a stratified split. "
            f"positives={positive_count}, negatives={negative_count}."
        )

    protein_columns = base.protein_columns_from_df(target_df, class_column, sample_id_column)
    cohort_description_df = build_cohort_description(target_df, class_column, target_label)
    cohort_description_df.to_csv(paths.tables / f"{cancer_slug}_cohort_description.csv", index=False)

    pre_imputation_missing_df = summarize_missingness(target_df, protein_columns, "pre_imputation_full")
    pre_imputation_missing_df.to_csv(paths.tables / f"{cancer_slug}_missingness_pre_imputation.csv", index=False)

    train_df, test_df = split_and_impute_binary(
        df=target_df,
        protein_columns=protein_columns,
        class_column=class_column,
        sample_id_column=sample_id_column,
        seed=seed,
        test_size=test_size,
        imputer_method=imputer,
        knn_neighbors=knn_neighbors,
    )

    target_df.to_csv(paths.data / f"{cancer_slug}_full_binary_dataset.csv", index=False)
    train_df.to_csv(paths.data / f"{cancer_slug}_dev_imputed.csv", index=False)
    test_df.to_csv(paths.data / f"{cancer_slug}_test_imputed.csv", index=False)

    post_imputation_missing_df = pd.concat(
        [
            summarize_missingness(train_df, protein_columns, "post_imputation_dev"),
            summarize_missingness(test_df, protein_columns, "post_imputation_test"),
        ],
        ignore_index=True,
    )
    post_imputation_missing_df.to_csv(paths.tables / f"{cancer_slug}_missingness_post_imputation.csv", index=False)

    split_manifest = pd.concat(
        [
            train_df[[sample_id_column, class_column, "Label"]].assign(partition="dev"),
            test_df[[sample_id_column, class_column, "Label"]].assign(partition="test"),
        ],
        ignore_index=True,
    )
    split_manifest.to_csv(paths.manifests / f"{cancer_slug}_split_manifest.csv", index=False)
    rest_composition_df = (
        split_manifest.loc[split_manifest["Label"] == 0, [class_column, "partition"]]
        .value_counts()
        .rename("n_samples")
        .reset_index()
        .rename(columns={class_column: "source_cancer"})
        .sort_values(["partition", "n_samples", "source_cancer"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    rest_composition_df.to_csv(paths.tables / f"{cancer_slug}_rest_pool_composition.csv", index=False)

    de_df = base.differential_expression(
        train_df=train_df,
        protein_columns=protein_columns,
        case_mean_column=f"{cancer_slug}_mean",
        case_n_column=f"{cancer_slug}_n",
        control_n_column="rest_n",
    )
    de_df.insert(0, "target_class", target_label)
    de_df.to_csv(paths.tables / f"{cancer_slug}_dea.csv", index=False)
    de_df.loc[de_df["significant_bonf"]].to_csv(paths.tables / f"{cancer_slug}_dea_bonf_significant.csv", index=False)
    de_df.sort_values("fold_change", ascending=False).head(25).to_csv(
        paths.tables / f"{cancer_slug}_dea_top_upregulated.csv",
        index=False,
    )
    de_df.sort_values("fold_change", ascending=True).head(25).to_csv(
        paths.tables / f"{cancer_slug}_dea_top_downregulated.csv",
        index=False,
    )
    save_volcano_figure(target_label, de_df, paths.figures / f"{cancer_slug}_dea_volcano.png")
    save_pvalue_histogram(target_label, de_df, paths.figures / f"{cancer_slug}_dea_pvalue_histogram.png")

    feature_pool, feature_source = select_feature_pool_bonf_only(de_df, train_df)
    ranking_df, top_features = base.rank_features_with_lasso(train_df, feature_pool, seed, feature_source)
    ranking_df.insert(0, "target_class", target_label)
    ranking_df.to_csv(paths.tables / f"{cancer_slug}_top_feature_ranking.csv", index=False)
    selected_feature_pool_df = pd.DataFrame(
        {
            "target_class": [target_label] * len(feature_pool),
            "feature_source": [feature_source] * len(feature_pool),
            "protein": feature_pool,
        }
    )
    selected_feature_pool_df.to_csv(paths.tables / f"{cancer_slug}_selected_feature_pool.csv", index=False)

    model_comparison_df = base.build_model_comparison(top_features, feature_source, train_df, test_df, seed)
    model_comparison_df.insert(0, "target_class", target_label)
    model_comparison_df.to_csv(paths.tables / f"{cancer_slug}_model_comparison.csv", index=False)

    panel_df, panel_rank_df, panel_dataset_df, plot_payloads = build_panel_outputs_revised(
        target_class=target_label,
        train_df=train_df,
        test_df=test_df,
        top_features=top_features,
        all_selected_features=feature_pool,
        panel_sizes=panel_sizes,
        seed=seed,
        sample_id_column=sample_id_column,
        class_column=class_column,
        include_all_panel=include_all_panel,
    )

    if not panel_df.empty:
        panel_df.insert(0, "target_class", target_label)
    if not panel_rank_df.empty and "target_class" not in panel_rank_df.columns:
        panel_rank_df.insert(0, "target_class", target_label)
    panel_dataset_df.insert(0, "target_class", target_label)

    panel_df.to_csv(paths.tables / f"{cancer_slug}_panel_performance.csv", index=False)
    panel_rank_df.to_csv(paths.tables / f"{cancer_slug}_panel_top18_ranking.csv", index=False)
    panel_dataset_df.to_csv(paths.tables / f"{cancer_slug}_selected_panel_dataset.csv", index=False)
    build_panel_membership_table(target_label, panel_df).to_csv(
        paths.tables / f"{cancer_slug}_panel_membership.csv",
        index=False,
    )
    save_test_predictions(target_label, plot_payloads, paths.tables / f"{cancer_slug}_test_predictions.csv")

    save_panel_auc_figure_themed(target_label, panel_df, paths.figures / f"{cancer_slug}_panel_auc.png")
    save_panel_roc_figure(target_label, plot_payloads, paths.figures / f"{cancer_slug}_panel_roc.png")

    if run_robustness:
        panel_ci_df = base.build_panel_confidence_intervals(target_label, plot_payloads, bootstrap_samples, seed)
        threshold_df = base.build_threshold_justification_table(target_label, plot_payloads)
        calibration_df = base.build_calibration_summary(target_label, plot_payloads)
        panel_ci_df.to_csv(paths.tables / f"{cancer_slug}_panel_confidence_intervals.csv", index=False)
        threshold_df.to_csv(paths.tables / f"{cancer_slug}_threshold_justification.csv", index=False)
        calibration_df.to_csv(paths.tables / f"{cancer_slug}_calibration_summary.csv", index=False)
        base.save_calibration_figure(target_label, plot_payloads, paths.figures / f"{cancer_slug}_calibration.png")

    enrichment_rows: list[dict[str, object]] = []
    if run_enrichment:
        enrichment_rows = base.run_enrichment_for_class(
            target_class=target_label,
            target_slug=cancer_slug,
            de_df=de_df,
            enrichment_dir=paths.tables,
            enrichment_figure_dir=paths.figures,
            databases=base.DEFAULT_DATABASES,
            top_terms=15,
            gsea_permutations=250,
        )
        if enrichment_rows:
            pd.DataFrame(enrichment_rows).to_csv(paths.tables / f"{cancer_slug}_enrichment_summary.csv", index=False)

    cohort_manifest = {
        "target_slug": cancer_slug,
        "target_label": target_label,
        "input": str(input_path.resolve()),
        "class_column": class_column,
        "sample_id_column": sample_id_column,
        "seed": seed,
        "test_size": test_size,
        "imputer": imputer,
        "knn_neighbors": knn_neighbors,
        "panel_sizes": panel_sizes,
        "include_all_panel": include_all_panel,
        "run_enrichment": run_enrichment,
        "run_robustness": run_robustness,
        "bootstrap_samples": bootstrap_samples,
        "n_samples": int(len(target_df)),
        "positive_n": positive_count,
        "negative_n": negative_count,
        "feature_pool_size": len(feature_pool),
        "top_feature_count": len(top_features),
        "dea_feature_filter": "bonf_only",
    }
    write_json(paths.manifests / f"{cancer_slug}_cohort_manifest.json", cohort_manifest)
    write_json(
        paths.manifests / f"{cancer_slug}_imputation_manifest.json",
        {
            "target_slug": cancer_slug,
            "imputer": imputer,
            "knn_neighbors": knn_neighbors,
            "protein_feature_count": len(protein_columns),
            "pre_imputation_missing_values": int(target_df[protein_columns].isna().sum().sum()),
            "post_dev_missing_values": int(train_df[protein_columns].isna().sum().sum()),
            "post_test_missing_values": int(test_df[protein_columns].isna().sum().sum()),
        },
    )

    reference_payload = base.select_reference_panel_payload(plot_payloads)
    reference_summary: dict[str, object] | None = None
    importance_rows: list[dict[str, object]] = []
    if reference_payload is not None:
        reference_summary = {
            "target_class": target_label,
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
        importance_rows = [
            {
                "target_class": target_label,
                "protein": protein,
                "importance": float(importance),
            }
            for protein, importance in zip(reference_payload["proteins"], reference_payload["importances"])
        ]

    aggregate_row = {
        "target_class": target_label,
        "dev_case_n": int(train_df["Label"].sum()),
        "dev_rest_n": int(len(train_df) - int(train_df["Label"].sum())),
        "test_case_n": int(test_df["Label"].sum()),
        "test_rest_n": int(len(test_df) - int(test_df["Label"].sum())),
        "bh_significant_features": int(de_df["significant_bh"].sum()),
        "bonferroni_significant_features": int(de_df["significant_bonf"].sum()),
        "feature_source": feature_source,
        "selected_feature_pool_size": len(feature_pool),
        "top_feature_count": len(top_features),
        "best_panel": panel_df.sort_values("Test_AUC", ascending=False).iloc[0]["Model"] if not panel_df.empty else "",
        "best_panel_test_auc": float(panel_df["Test_AUC"].max()) if not panel_df.empty else base.math.nan,
    }

    return {
        "slug": cancer_slug,
        "label": target_label,
        "paths": paths,
        "aggregate_row": aggregate_row,
        "panel_df": panel_df,
        "de_df": de_df,
        "ranking_df": ranking_df,
        "selected_feature_pool_df": selected_feature_pool_df,
        "reference_summary": reference_summary,
        "importance_rows": importance_rows,
        "reference_payload": {**reference_payload, "target_class": target_label} if reference_payload is not None else None,
        "enrichment_rows": enrichment_rows,
    }


def aggregate_pan_cancer_outputs(results: list[dict[str, Any]]) -> None:
    pan_paths = pan_cancer_paths()
    ensure_pan_dirs(pan_paths)
    class_feature_list_dir = pan_paths["tables"] / "class_feature_lists"
    class_feature_list_dir.mkdir(parents=True, exist_ok=True)

    aggregate_df = pd.DataFrame([result["aggregate_row"] for result in results]).sort_values("target_class").reset_index(drop=True)
    aggregate_df.to_csv(pan_paths["tables"] / "class_iteration_summary.csv", index=False)

    pool_annotation_df = pd.DataFrame()
    selected_feature_pool_frames = [
        result["selected_feature_pool_df"] for result in results if not result["selected_feature_pool_df"].empty
    ]
    if selected_feature_pool_frames:
        combined_selected_df = pd.concat(selected_feature_pool_frames, ignore_index=True)
        combined_selected_df.to_csv(pan_paths["tables"] / "all_classes_selected_feature_pool.csv", index=False)
        pool_summary_df = (
            combined_selected_df.groupby("protein", as_index=False)
            .agg(
                cancer_count=("target_class", "nunique"),
                cancers=("target_class", lambda series: ";".join(sorted(series.unique()))),
            )
            .sort_values(["cancer_count", "protein"], ascending=[False, True])
            .reset_index(drop=True)
        )
        pool_summary_df.to_csv(pan_paths["tables"] / "all_classes_selected_feature_pool_summary.csv", index=False)
        pool_summary_df.loc[pool_summary_df["cancer_count"] == 1].to_csv(
            pan_paths["tables"] / "class_specific_selected_features.csv",
            index=False,
        )
        pool_summary_df.loc[pool_summary_df["cancer_count"] > 1].to_csv(
            pan_paths["tables"] / "shared_selected_features.csv",
            index=False,
        )
        pool_annotation_df = combined_selected_df.merge(pool_summary_df, on="protein", how="left")
        per_class_specific_df = pool_annotation_df.loc[pool_annotation_df["cancer_count"] == 1].copy()
        per_class_specific_df = per_class_specific_df.sort_values(["target_class", "protein"]).reset_index(drop=True)
        per_class_specific_df.to_csv(
            pan_paths["tables"] / "per_class_specific_selected_features.csv",
            index=False,
        )
        for target_class, class_df in per_class_specific_df.groupby("target_class"):
            class_slug = base.slugify(str(target_class))
            class_df.to_csv(
                class_feature_list_dir / f"{class_slug}_class_specific_selected_features.csv",
                index=False,
            )
        per_class_shared_df = pool_annotation_df.loc[pool_annotation_df["cancer_count"] > 1].copy()
        per_class_shared_df = per_class_shared_df.sort_values(["target_class", "cancer_count", "protein"], ascending=[True, False, True]).reset_index(drop=True)
        per_class_shared_df.to_csv(
            pan_paths["tables"] / "per_class_shared_selected_features.csv",
            index=False,
        )
        for target_class, class_df in per_class_shared_df.groupby("target_class"):
            class_slug = base.slugify(str(target_class))
            class_df.to_csv(
                class_feature_list_dir / f"{class_slug}_shared_selected_features.csv",
                index=False,
            )

    ranking_frames = [result["ranking_df"] for result in results if not result["ranking_df"].empty]
    top1_summary_df = pd.DataFrame()
    if ranking_frames:
        combined_ranking_df = pd.concat(ranking_frames, ignore_index=True)
        combined_ranking_df.to_csv(pan_paths["tables"] / "all_classes_top_feature_ranking.csv", index=False)
        top1_rows: list[dict[str, object]] = []
        for result in results:
            ranking_df = result["ranking_df"]
            panel_df = result["panel_df"]
            if ranking_df.empty:
                continue
            top1_protein = str(ranking_df.iloc[0]["protein"])
            top1_importance = float(ranking_df.iloc[0]["importance"])
            top1_panel_df = panel_df.loc[panel_df["Features"] == 1]
            top1_auc = float(top1_panel_df.iloc[0]["Test_AUC"]) if not top1_panel_df.empty else base.math.nan
            top1_rows.append(
                {
                    "target_class": result["label"],
                    "top1_protein": top1_protein,
                    "top1_importance": top1_importance,
                    "top1_test_auc": top1_auc,
                    "best_panel": result["aggregate_row"]["best_panel"],
                    "best_panel_test_auc": result["aggregate_row"]["best_panel_test_auc"],
                }
            )
        top1_summary_df = pd.DataFrame(top1_rows).sort_values("target_class").reset_index(drop=True)
        top1_summary_df.to_csv(pan_paths["tables"] / "top1_marker_summary.csv", index=False)
        if not pool_annotation_df.empty:
            top1_review_df = top1_summary_df.merge(
                pool_annotation_df[["target_class", "protein", "feature_source", "cancer_count", "cancers"]].drop_duplicates(),
                left_on=["target_class", "top1_protein"],
                right_on=["target_class", "protein"],
                how="left",
            ).drop(columns=["protein"])
            top1_review_df["is_class_specific"] = top1_review_df["cancer_count"].fillna(0).eq(1)
            top1_review_df["shared_with_classes"] = top1_review_df.apply(
                lambda row: ";".join(
                    [
                        class_name
                        for class_name in str(row["cancers"]).split(";")
                        if class_name and class_name != str(row["target_class"])
                    ]
                ),
                axis=1,
            )
            top1_review_df = top1_review_df[
                [
                    "target_class",
                    "top1_protein",
                    "feature_source",
                    "top1_importance",
                    "top1_test_auc",
                    "best_panel",
                    "best_panel_test_auc",
                    "cancer_count",
                    "cancers",
                    "is_class_specific",
                    "shared_with_classes",
                ]
            ].rename(
                columns={
                    "feature_source": "top1_feature_source",
                    "cancer_count": "top1_cancer_count",
                    "cancers": "top1_present_in_classes",
                }
            )
            top1_review_df.to_csv(
                pan_paths["tables"] / "top1_marker_review_shortlist.csv",
                index=False,
            )

    panel_frames = [result["panel_df"] for result in results if not result["panel_df"].empty]
    combined_panel_df = pd.DataFrame()
    if panel_frames:
        combined_panel_df = pd.concat(panel_frames, ignore_index=True)
        combined_panel_df.to_csv(pan_paths["tables"] / "all_classes_panel_performance.csv", index=False)
        save_performance_vs_panel_size_figure_themed(
            combined_panel_df,
            pan_paths["figures"] / "all_classes_performance_vs_panel_size.png",
        )
    if not combined_panel_df.empty and not top1_summary_df.empty:
        build_panel_review_tables(
            pan_paths=pan_paths,
            aggregate_df=aggregate_df,
            combined_panel_df=combined_panel_df,
            top1_summary_df=top1_summary_df,
        )

    reference_rows = [result["reference_summary"] for result in results if result["reference_summary"] is not None]
    if reference_rows:
        reference_df = pd.DataFrame(reference_rows).sort_values("target_class").reset_index(drop=True)
        reference_df.to_csv(pan_paths["tables"] / "all_classes_reference_panel_summary.csv", index=False)

    importance_rows = [row for result in results for row in result["importance_rows"]]
    if importance_rows:
        importance_df = pd.DataFrame(importance_rows).sort_values(["target_class", "importance"], ascending=[True, False]).reset_index(drop=True)
        importance_df.to_csv(pan_paths["tables"] / "all_classes_reference_feature_importance.csv", index=False)
        base.save_feature_importance_heatmap(
            importance_df,
            pan_paths["figures"] / "all_classes_feature_importance_heatmap.png",
        )
        feature_frequency_df = (
            importance_df.groupby("protein", as_index=False)
            .agg(
                cancer_count=("target_class", "nunique"),
                mean_importance=("importance", "mean"),
                max_importance=("importance", "max"),
            )
            .sort_values(["cancer_count", "mean_importance", "protein"], ascending=[False, False, True])
            .reset_index(drop=True)
        )
        feature_frequency_df.to_csv(pan_paths["tables"] / "pan_feature_frequency.csv", index=False)
        overlap_matrix = pd.crosstab(importance_df["target_class"], importance_df["protein"]).astype(int)
        overlap_matrix.to_csv(pan_paths["tables"] / "pan_feature_overlap_matrix.csv")
        save_feature_overlap_heatmap(overlap_matrix, pan_paths["figures"] / "pan_feature_overlap_heatmap.png")
        shared_feature_df = importance_df.groupby("protein", as_index=False).agg(
            cancer_count=("target_class", "nunique"),
            cancers=("target_class", lambda series: ";".join(sorted(series.unique()))),
        )
        shared_feature_df = shared_feature_df.loc[shared_feature_df["cancer_count"] > 1].sort_values(
            ["cancer_count", "protein"], ascending=[False, True]
        )
        shared_feature_df.to_csv(pan_paths["tables"] / "shared_pan_features.csv", index=False)

    reference_payloads = [result["reference_payload"] for result in results if result["reference_payload"] is not None]
    if reference_payloads:
        save_all_class_roc_figure_themed(
            reference_payloads,
            pan_paths["figures"] / "all_classes_roc_curves.png",
        )
    save_combined_volcano_figure(results, pan_paths["figures"] / "all_classes_combined_volcano.png")

    output_manifest_rows = []
    for result in results:
        for category_name in ["data", "tables", "figures", "logs", "manifests"]:
            category_path = getattr(result["paths"], category_name)
            for file_path in sorted(category_path.glob("*")):
                if file_path.is_file():
                    output_manifest_rows.append(
                        {
                            "target_class": result["label"],
                            "category": category_name,
                            "path": str(file_path.resolve()),
                        }
                    )
    for category_name, category_path in pan_paths.items():
        if category_name == "root":
            continue
        for file_path in sorted(category_path.glob("*")):
            if file_path.is_file():
                output_manifest_rows.append(
                    {
                        "target_class": "PAN_CANCER",
                        "category": category_name,
                        "path": str(file_path.resolve()),
                    }
                )
    pd.DataFrame(output_manifest_rows).to_csv(pan_paths["manifests"] / "output_manifest.csv", index=False)

    write_json(
        pan_paths["manifests"] / "run_manifest.json",
        {
            "source_input": str(SOURCE_INPUT.resolve()),
            "cancers_processed": [result["slug"] for result in results],
            "pan_cancer_root": str(pan_paths["root"].resolve()),
        },
    )
