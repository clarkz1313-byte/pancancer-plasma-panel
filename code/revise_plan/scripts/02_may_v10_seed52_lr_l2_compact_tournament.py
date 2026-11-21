#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROTEOMICS_PYTHON = Path("/home/hien/occho/envs/proteomics/bin/python")
if Path(sys.executable).resolve() != PROTEOMICS_PYTHON.resolve() and PROTEOMICS_PYTHON.exists():
    os.execv(str(PROTEOMICS_PYTHON), [str(PROTEOMICS_PYTHON), *sys.argv])

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
PART_B_ROOT = ROOT / "revise_plan" / "part_b_multiclass"
V9_PATH = SCRIPT_DIR / "02_may_v9_seed52_compact_tournament.py"

spec = importlib.util.spec_from_file_location("v9", V9_PATH)
v9 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v9)

v8 = v9.v8

DEFAULT_OUTPUT = PART_B_ROOT / "v10_seed52_lr_l2_compact_tournament"
DEFAULT_BUDGETS = (24, 25, 26, 27, 28)
DEFAULT_RANKERS = (
    "LOGISTIC_COEF",
    "CONSENSUS_SOFT_VOTE",
    "OVR_BONF_QUOTA",
    "COVERAGE_IMPORTANCE",
    "PAIRWISE_STABILITY",
    "PAIRWISE_MINMAX",
    "OVR_STABILITY",
    "OVR_MINMAX",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V10 LR_L2-only seed-52 compact tournament. This is the cleaner final tournament script: "
            "from processed NPX data, train-only feature discovery/CV, and LR_L2-only held-out test audit."
        )
    )
    parser.add_argument("--input", type=Path, default=v8.v1.SOURCE_INPUT)
    parser.add_argument("--class-column", default="Cancer")
    parser.add_argument("--sample-id-column", default="Sample_ID")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--imputer", choices=sorted(v8.base.IMPUTER_METHODS), default="knn")
    parser.add_argument("--knn-neighbors", type=int, default=v8.base.KNN_NEIGHBORS)
    parser.add_argument("--per-cancer-top-m", type=int, default=14)
    parser.add_argument("--min-cancer-coverage", type=int, default=2)
    parser.add_argument("--candidate-bank-size", type=int, default=160)
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--rankers", nargs="+", default=list(DEFAULT_RANKERS), choices=list(v8.SEARCH_RANKERS))
    parser.add_argument("--random-panels-per-size", type=int, default=360)
    parser.add_argument("--max-panels", type=int, default=1200)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--test-top-per-size", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=52024)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def log(output: dict[str, Path], message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (output["logs"] / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def clean_outputs(output: dict[str, Path], resume: bool) -> None:
    if resume:
        return
    for csv_path in output["tables"].glob("*.csv"):
        csv_path.unlink()


def split_features(value: object) -> list[str]:
    return [feature for feature in str(value).split(";") if feature]


def bootstrap_metric_summary(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    classes: list[str],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    y_values = y_true.to_numpy()
    metric_values = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "macro_ovr_auc": [],
        "min_class_recall": [],
        "rescue_recall": [],
    }
    rescue_indices = [classes.index(label) for label in v8.RESCUE_CLASSES if label in classes]
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_values), size=len(y_values))
        y_boot = y_values[indices]
        pred_boot = y_pred[indices]
        score_boot = y_score[indices, :]
        recalls = recall_score(y_boot, pred_boot, labels=classes, average=None, zero_division=0)
        try:
            macro_auc = float(roc_auc_score(y_boot, score_boot, labels=classes, multi_class="ovr", average="macro"))
        except ValueError:
            macro_auc = np.nan
        metric_values["accuracy"].append(float(accuracy_score(y_boot, pred_boot)))
        metric_values["balanced_accuracy"].append(float(balanced_accuracy_score(y_boot, pred_boot)))
        metric_values["macro_f1"].append(float(f1_score(y_boot, pred_boot, labels=classes, average="macro", zero_division=0)))
        metric_values["macro_ovr_auc"].append(macro_auc)
        metric_values["min_class_recall"].append(float(np.min(recalls)))
        metric_values["rescue_recall"].append(float(np.nanmean([recalls[index] for index in rescue_indices])))

    output: dict[str, float] = {}
    for metric, values in metric_values.items():
        clean = pd.Series(values, dtype="float64").dropna()
        output[f"{metric}_boot_mean"] = float(clean.mean()) if not clean.empty else np.nan
        output[f"{metric}_ci_lower"] = float(clean.quantile(0.025)) if not clean.empty else np.nan
        output[f"{metric}_ci_upper"] = float(clean.quantile(0.975)) if not clean.empty else np.nan
    output["bootstrap_samples"] = int(n_bootstrap)
    return output


def add_bootstrap_ci_columns(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    classes: list[str],
    output: dict[str, Path],
    args: argparse.Namespace,
) -> pd.DataFrame:
    ci_prefixes = (
        "accuracy_",
        "balanced_accuracy_",
        "macro_f1_",
        "macro_ovr_auc_",
        "min_class_recall_",
        "rescue_recall_",
    )
    drop_cols = [
        column
        for column in test_df.columns
        if column == "bootstrap_samples" or any(column.startswith(prefix) and ("_ci_" in column or column.endswith("_boot_mean")) for prefix in ci_prefixes)
    ]
    if drop_cols:
        test_df = test_df.drop(columns=drop_cols)

    rows: list[dict[str, float]] = []
    y_train = train_df[args.class_column]
    y_true = heldout_df[args.class_column]
    for row_index, row in enumerate(test_df.itertuples(index=False), start=1):
        features = split_features(getattr(row, "features"))
        estimator = v8.v2.make_estimator_v2("LR_L2", args.seed, {})
        estimator.fit(train_df[features], y_train)
        y_pred = np.asarray(estimator.predict(heldout_df[features]))
        y_score = v8.v1.aligned_probabilities(estimator, heldout_df[features], classes)
        rng = np.random.default_rng(args.bootstrap_seed + row_index)
        rows.append(bootstrap_metric_summary(y_true, y_pred, y_score, classes, rng, args.bootstrap_samples))
        if row_index == 1 or row_index % 25 == 0 or row_index == len(test_df):
            log(output, f"[BOOTSTRAP] row {row_index}/{len(test_df)}")
    ci_df = pd.DataFrame(rows)
    ci_long_rows = []
    for row_index, ci_row in enumerate(ci_df.itertuples(index=False), start=1):
        trial_no = int(test_df.iloc[row_index - 1]["trial_mo"]) if "trial_mo" in test_df.columns else row_index
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_auc", "min_class_recall", "rescue_recall"]:
            ci_long_rows.append(
                {
                    "trial_no": trial_no,
                    "metric": metric,
                    "bootstrap_mean": getattr(ci_row, f"{metric}_boot_mean"),
                    "ci_lower_2_5": getattr(ci_row, f"{metric}_ci_lower"),
                    "ci_upper_97_5": getattr(ci_row, f"{metric}_ci_upper"),
                    "bootstrap_samples": getattr(ci_row, "bootstrap_samples"),
                }
            )
    pd.DataFrame(ci_long_rows).to_csv(output["tables"] / "stage2_bootstrap_ci_long.csv", index=False)
    return pd.concat([test_df.reset_index(drop=True), ci_df.reset_index(drop=True)], axis=1)


def save_lr_l2_specific_summaries(output: dict[str, Path]) -> None:
    for misleading in [
        output["tables"] / "best_softvote_by_feature_count.csv",
        output["tables"] / "high_performance_candidates.csv",
        output["figures"] / "softvote_test_ba_by_feature_count.png",
    ]:
        if misleading.exists():
            misleading.unlink()

    all_path = output["tables"] / "stage2_test_audit_all_rows.csv"
    if not all_path.exists():
        return
    df = pd.read_csv(all_path)
    if "trial_mo" in df.columns:
        df = df.rename(columns={"trial_mo": "trial_no"})
    if "trial_no" not in df.columns:
        df.insert(0, "trial_no", range(1, len(df) + 1))
    else:
        df["trial_no"] = range(1, len(df) + 1)
        df = df[["trial_no", *[column for column in df.columns if column != "trial_no"]]]
    df.to_csv(all_path, index=False)
    lr_path = output["tables"] / "stage2_test_audit_lr_l2_all_rows.csv"
    df.to_csv(lr_path, index=False)

    best = df.iloc[0].copy()
    pd.DataFrame([best]).to_csv(output["tables"] / "best_lr_l2_candidate_summary.csv", index=False)
    compact = df.loc[(df["feature_count"] <= 26) & (df["test_balanced_accuracy"] >= 0.75)].copy()
    compact.to_csv(output["tables"] / "high_performance_lr_l2_candidates.csv", index=False)

    figure, ax = v8.base.plt.subplots(figsize=(8.5, 5.2))
    sns.scatterplot(
        data=df,
        x="feature_count",
        y="test_balanced_accuracy",
        hue="source_ranker",
        style="source_origin",
        s=58,
        ax=ax,
    )
    ax.axhline(0.75, color="#9c2f2f", linestyle="--", linewidth=1)
    ax.set_title("V10 LR_L2-only compact tournament")
    ax.set_xlabel("Feature count")
    ax.set_ylabel("Held-out test balanced accuracy")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    figure.tight_layout()
    figure.savefig(output["figures"] / "lr_l2_test_ba_by_feature_count.png", dpi=300, bbox_inches="tight")
    v8.base.plt.close(figure)

    figure, ax = v8.base.plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="cv_balanced_accuracy_mean", y="test_balanced_accuracy", hue="feature_count", palette="viridis", s=55, ax=ax)
    ax.axhline(0.75, color="#9c2f2f", linestyle="--", linewidth=1)
    ax.set_title("V10 LR_L2 CV-vs-test audit")
    ax.set_xlabel("Train-CV balanced accuracy")
    ax.set_ylabel("Held-out test balanced accuracy")
    ax.legend(frameon=False, title="Features")
    figure.tight_layout()
    figure.savefig(output["figures"] / "lr_l2_cv_vs_test_balanced_accuracy.png", dpi=300, bbox_inches="tight")
    v8.base.plt.close(figure)


def main() -> None:
    args = parse_args()
    args.budgets = sorted({int(value) for value in args.budgets if int(value) > 0})
    args.final_coaches = ["LR_L2"]
    output = v9.paths(args.output_dir)
    clean_outputs(output, args.resume)

    log(output, f"V10 LR_L2-only starts. seed={args.seed}, budgets={args.budgets}, max_panels={args.max_panels}")
    log(output, "Only LR_L2 is audited on held-out test. No final-coach switching is possible inside V10.")

    full_df = pd.read_csv(args.input)
    protein_columns = v8.base.protein_columns_from_df(full_df, args.class_column, args.sample_id_column)
    classes = sorted(full_df[args.class_column].dropna().unique().tolist())
    train_df, heldout_df = v8.v1.split_and_impute(
        full_df=full_df,
        protein_columns=protein_columns,
        class_column=args.class_column,
        sample_id_column=args.sample_id_column,
        seed=args.seed,
        test_size=args.test_size,
        imputer_method=args.imputer,
        knn_neighbors=args.knn_neighbors,
    )
    train_df = train_df.reset_index(drop=True)
    heldout_df = heldout_df.reset_index(drop=True)

    bonf_df, quota_df, de_df = v8.collect_train_only_de_sources(
        train_df=train_df,
        protein_columns=protein_columns,
        class_column=args.class_column,
        classes=classes,
        per_cancer_top_m=args.per_cancer_top_m,
    )
    candidate_features, candidate_feature_df, coverage_df = v8.build_candidate_feature_table(
        bonf_df=bonf_df,
        quota_df=quota_df,
        protein_columns=protein_columns,
        min_cancer_coverage=args.min_cancer_coverage,
        candidate_bank_size=args.candidate_bank_size,
    )
    rankings, ranking_table = v8.v2.build_rankings_v2(
        x_train=train_df[candidate_features],
        y_train=train_df[args.class_column],
        classes=classes,
        quota_sources_df=quota_df,
        coverage_df=coverage_df,
        seed=args.seed,
    )
    priority = v8.ranking_priority(rankings, ranking_table, candidate_feature_df)
    candidates = v9.make_candidates(rankings, quota_df, coverage_df, candidate_features, priority, classes, args)
    log(output, f"Candidate panels after static prescreen: {len(candidates)}")

    bonf_df.to_csv(output["tables"] / "train_only_bonf_sources.csv", index=False)
    quota_df.to_csv(output["tables"] / "train_only_quota_sources.csv", index=False)
    de_df.to_csv(output["tables"] / "train_only_ovr_dea_all.csv", index=False)
    candidate_feature_df.to_csv(output["tables"] / "train_only_candidate_feature_bank.csv", index=False)
    ranking_table.to_csv(output["tables"] / "train_only_ranker_table.csv", index=False)
    candidates.to_csv(output["tables"] / "stage0_panel_candidates_prescreen.csv", index=False)

    cv_df = v9.cv_score_candidates(candidates, train_df, classes, output, args)
    audit_candidates = v9.select_for_test(cv_df, args)
    test_df = v9.test_audit(audit_candidates, train_df, heldout_df, classes, output, args)
    test_df = add_bootstrap_ci_columns(test_df, train_df, heldout_df, classes, output, args)
    test_df.to_csv(output["tables"] / "stage2_test_audit_all_rows.csv", index=False)
    v9.save_summaries(cv_df, audit_candidates, test_df, train_df, heldout_df, classes, output)
    save_lr_l2_specific_summaries(output)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "input": str(args.input.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "seed": args.seed,
        "test_size": args.test_size,
        "budgets": args.budgets,
        "rankers": args.rankers,
        "final_coach": "LR_L2",
        "test_top_per_size": args.test_top_per_size,
        "bootstrap_samples_per_row": args.bootstrap_samples,
        "notes": [
            "V10 is LR_L2-only to avoid final-coach switching after test audit.",
            "Feature discovery and stage-1 selection are train-only; test rows are internal audit rows.",
            "This is not external clinical validation.",
        ],
    }
    with (output["manifests"] / "v10_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    best_path = output["tables"] / "best_lr_l2_candidate_summary.csv"
    if best_path.exists():
        best = pd.read_csv(best_path).iloc[0]
        log(
            output,
            (
                f"[seed {int(best['seed'])}] test size={int(best['feature_count'])} coach=LR_L2 "
                f"BA={best['test_balanced_accuracy']:.4f} F1={best['test_macro_f1']:.4f} "
                f"minRecall={best['test_min_class_recall']:.4f}"
            ),
        )
    log(output, "V10 complete.")


if __name__ == "__main__":
    main()
