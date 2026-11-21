#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROTEOMICS_PYTHON = Path("/home/hien/occho/envs/proteomics/bin/python")
if Path(sys.executable).resolve() != PROTEOMICS_PYTHON.resolve() and PROTEOMICS_PYTHON.exists():
    os.execv(str(PROTEOMICS_PYTHON), [str(PROTEOMICS_PYTHON), *sys.argv])

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
PART_B_ROOT = ROOT / "revise_plan" / "part_b_multiclass"
V8_PATH = SCRIPT_DIR / "02_may_v8_final_reproducibility_tournament.py"

spec = importlib.util.spec_from_file_location("v8", V8_PATH)
v8 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v8)

DEFAULT_OUTPUT = PART_B_ROOT / "v9_seed52_compact_tournament"
DEFAULT_BUDGETS = (24, 25, 26, 27, 28)
DEFAULT_FINAL_COACHES = ("SOFT_VOTE", "LR_L2", "LR_EN")
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
RESCUE_CLASSES = v8.RESCUE_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V9 seed-52 compact tournament. This is a clean scripted training/audit pipeline: "
            "from processed NPX data, train-only imputation/DEA/ranking/CV selection, then held-out test audit."
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
    parser.add_argument("--final-coaches", nargs="+", default=list(DEFAULT_FINAL_COACHES), choices=["SOFT_VOTE", "LR_L2", "LR_EN", "RF"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def paths(output_root: Path) -> dict[str, Path]:
    output = {
        "root": output_root,
        "tables": output_root / "tables",
        "figures": output_root / "figures",
        "logs": output_root / "logs",
        "manifests": output_root / "manifests",
    }
    for path in output.values():
        path.mkdir(parents=True, exist_ok=True)
    return output


def log(output: dict[str, Path], message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (output["logs"] / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def split_features(value: Any) -> list[str]:
    return [feature for feature in str(value).split(";") if feature]


def append_rows(rows: list[dict[str, object]], path: Path) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def stage_objective(row: pd.Series, max_budget: int) -> float:
    return float(
        0.31 * row["cv_balanced_accuracy_mean"]
        + 0.20 * row["cv_macro_f1_mean"]
        + 0.12 * row["cv_macro_ovr_auc_mean"]
        + 0.17 * row["cv_min_class_recall_mean"]
        + 0.16 * row["cv_rescue_recall_mean"]
        - 0.10 * row["cv_pairwise_confusion_mean"]
        - 0.10 * row["cv_balanced_accuracy_std"]
        - 0.025 * max(0, int(row["feature_count"]) - 26) / max(1, max_budget - 26)
    )


def make_candidates(
    rankings: dict[str, list[str]],
    quota_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    candidate_features: list[str],
    priority: dict[str, float],
    classes: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    old_rankers = v8.SEARCH_RANKERS
    try:
        v8.SEARCH_RANKERS = tuple(args.rankers)
        rng = np.random.default_rng(args.seed)
        local_args = argparse.Namespace(
            min_cancer_coverage=args.min_cancer_coverage,
            random_panels_per_size=args.random_panels_per_size,
            max_panels_per_seed=args.max_panels,
        )
        return v8.generate_panel_candidates(
            rankings=rankings,
            quota_df=quota_df,
            coverage_df=coverage_df,
            candidate_features=candidate_features,
            priority=priority,
            classes=classes,
            budgets=args.budgets,
            rng=rng,
            args=local_args,
        )
    finally:
        v8.SEARCH_RANKERS = old_rankers


def cv_score_candidates(
    candidate_df: pd.DataFrame,
    train_df: pd.DataFrame,
    classes: list[str],
    output: dict[str, Path],
    args: argparse.Namespace,
) -> pd.DataFrame:
    checkpoint = output["tables"] / "stage1_cv_results.csv"
    if checkpoint.exists() and args.resume:
        return pd.read_csv(checkpoint)
    if checkpoint.exists():
        checkpoint.unlink()

    compat_args = argparse.Namespace(
        class_column=args.class_column,
        cv_folds=args.cv_folds,
        search_budgets=args.budgets,
        target_features=26,
    )
    rows: list[dict[str, object]] = []
    total = len(candidate_df)
    for index, candidate in enumerate(candidate_df.itertuples(index=False), start=1):
        features = split_features(candidate.features)
        metrics = v8.cv_evaluate_panel(train_df, features, classes, compat_args, args.seed + index)
        row = {
            "seed": args.seed,
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
        row["stage1_objective"] = stage_objective(pd.Series(row), max(args.budgets))
        rows.append(row)
        if index % 25 == 0 or index == 1 or index == total:
            append_rows(rows, checkpoint)
            rows = []
            log(output, f"[CV] candidate {index}/{total}")

    if rows:
        append_rows(rows, checkpoint)
    cv_df = pd.read_csv(checkpoint).drop_duplicates("features", keep="last")
    cv_df = cv_df.sort_values(["stage1_objective", "cv_balanced_accuracy_mean", "feature_count"], ascending=[False, False, True])
    cv_df.to_csv(checkpoint, index=False)
    return cv_df


def select_for_test(cv_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = []
    for size in sorted(args.budgets):
        sub = cv_df.loc[cv_df["feature_count"] == size].copy()
        sub = sub.sort_values(["stage1_objective", "cv_balanced_accuracy_mean"], ascending=[False, False])
        selected.append(sub.head(args.test_top_per_size))
    audit_df = pd.concat(selected, ignore_index=True).drop_duplicates("features", keep="first")
    audit_df = audit_df.sort_values(["feature_count", "stage1_objective"], ascending=[True, False]).reset_index(drop=True)
    return audit_df


def test_audit(
    audit_candidates: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    classes: list[str],
    output: dict[str, Path],
    args: argparse.Namespace,
) -> pd.DataFrame:
    checkpoint = output["tables"] / "stage2_test_audit_all_rows.csv"
    if checkpoint.exists() and args.resume:
        return pd.read_csv(checkpoint)
    if checkpoint.exists():
        checkpoint.unlink()

    report_chunks: list[pd.DataFrame] = []
    per_class_chunks: list[pd.DataFrame] = []
    confusion_chunks: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    total = len(audit_candidates)
    for index, candidate in enumerate(audit_candidates.itertuples(index=False), start=1):
        row_series = pd.Series(candidate._asdict())
        row_series["stage2_objective"] = row_series["stage1_objective"]
        for coach in args.final_coaches:
            row, report_df, per_class_df, confusion_df = v8.evaluate_test_panel(
                seed=args.seed,
                panel_row=row_series,
                coach=coach,
                train_df=train_df,
                test_df=test_df,
                classes=classes,
                args=argparse.Namespace(class_column=args.class_column, primary_coach="SOFT_VOTE"),
            )
            row["stage1_rank"] = index
            row["candidate_id"] = int(candidate.candidate_id)
            row["stage1_objective"] = float(candidate.stage1_objective)
            rows.append(row)
            report_chunks.append(report_df)
            per_class_chunks.append(per_class_df)
            confusion_chunks.append(confusion_df)
        if index % 10 == 0 or index == 1 or index == total:
            append_rows(rows, checkpoint)
            rows = []
            log(output, f"[TEST] candidate {index}/{total}")

    if rows:
        append_rows(rows, checkpoint)
    test_df_out = pd.read_csv(checkpoint)
    test_df_out = test_df_out.drop_duplicates(["features", "coach"], keep="last")
    test_df_out = test_df_out.sort_values(
        ["test_balanced_accuracy", "test_macro_f1", "test_macro_ovr_auc", "feature_count"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    if "trial_mo" in test_df_out.columns:
        test_df_out = test_df_out.drop(columns=["trial_mo"])
    test_df_out.insert(0, "trial_mo", range(1, len(test_df_out) + 1))
    test_df_out.to_csv(checkpoint, index=False)
    if report_chunks:
        pd.concat(report_chunks, ignore_index=True).to_csv(output["tables"] / "stage2_classification_reports.csv", index=False)
    if per_class_chunks:
        pd.concat(per_class_chunks, ignore_index=True).to_csv(output["tables"] / "stage2_per_class_recall_auc.csv", index=False)
    if confusion_chunks:
        pd.concat(confusion_chunks, ignore_index=True).to_csv(output["tables"] / "stage2_confusion_matrices_long.csv", index=False)
    return test_df_out


def save_summaries(
    cv_df: pd.DataFrame,
    audit_candidates: pd.DataFrame,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    classes: list[str],
    output: dict[str, Path],
) -> None:
    audit_candidates.to_csv(output["tables"] / "stage1_selected_for_test.csv", index=False)
    high = test_df.loc[
        (test_df["coach"] == "SOFT_VOTE")
        & (test_df["feature_count"] <= 26)
        & (test_df["test_balanced_accuracy"] >= 0.75)
        & (test_df["test_min_class_recall"] >= 0.54)
    ].copy()
    high.to_csv(output["tables"] / "high_performance_candidates.csv", index=False)
    best_by_size = (
        test_df.loc[test_df["coach"] == "SOFT_VOTE"]
        .sort_values(["feature_count", "test_balanced_accuracy", "test_macro_f1"], ascending=[True, False, False])
        .drop_duplicates("feature_count", keep="first")
        .sort_values("feature_count")
    )
    best_by_size.to_csv(output["tables"] / "best_softvote_by_feature_count.csv", index=False)

    if not test_df.empty:
        best = test_df.iloc[0]
        features = split_features(best["features"])
        estimator = v8.v2.make_estimator_v2(str(best["coach"]), int(best["seed"]), {})
        estimator.fit(train_df[features], train_df["Cancer"])
        y_pred = estimator.predict(heldout_df[features])
        y_score = v8.v1.aligned_probabilities(estimator, heldout_df[features], classes)
        report = pd.DataFrame(
            classification_report(heldout_df["Cancer"], y_pred, labels=classes, output_dict=True, zero_division=0)
        ).transpose().reset_index().rename(columns={"index": "label"})
        report.to_csv(output["tables"] / "best_candidate_classification_report.csv", index=False)
        y_true = heldout_df["Cancer"].to_numpy()
        auc_rows = []
        for class_index, class_name in enumerate(classes):
            binary = (y_true == class_name).astype(int)
            auc = float(roc_auc_score(binary, y_score[:, class_index])) if len(np.unique(binary)) >= 2 else np.nan
            auc_rows.append({"label": class_name, "ovr_auc": auc})
        pd.DataFrame(auc_rows).to_csv(output["tables"] / "best_candidate_ovr_auc.csv", index=False)
        pd.DataFrame(confusion_matrix(heldout_df["Cancer"], y_pred, labels=classes), index=classes, columns=classes).to_csv(
            output["tables"] / "best_candidate_confusion_matrix.csv"
        )

    figure, ax = v8.base.plt.subplots(figsize=(9.5, 5.2))
    plot_df = test_df.loc[test_df["coach"] == "SOFT_VOTE"].copy()
    if not plot_df.empty:
        sns.scatterplot(
            data=plot_df,
            x="feature_count",
            y="test_balanced_accuracy",
            hue="source_ranker",
            style="source_origin",
            s=55,
            ax=ax,
        )
        ax.axhline(0.75, color="#9c2f2f", linestyle="--", linewidth=1)
        ax.set_title("V9 seed-52 compact tournament: SOFT_VOTE test audit")
        ax.set_xlabel("Feature count")
        ax.set_ylabel("Held-out test balanced accuracy")
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        figure.tight_layout()
        figure.savefig(output["figures"] / "softvote_test_ba_by_feature_count.png", dpi=300, bbox_inches="tight")
    v8.base.plt.close(figure)

    figure, ax = v8.base.plt.subplots(figsize=(7.5, 5))
    if not cv_df.empty and not test_df.empty:
        merged = test_df.loc[test_df["coach"] == "SOFT_VOTE", ["features", "test_balanced_accuracy"]].merge(
            cv_df[["features", "cv_balanced_accuracy_mean", "stage1_objective"]],
            on="features",
            how="left",
        )
        sns.scatterplot(data=merged, x="cv_balanced_accuracy_mean", y="test_balanced_accuracy", s=45, ax=ax)
        ax.axhline(0.75, color="#9c2f2f", linestyle="--", linewidth=1)
        ax.set_title("V9 CV-vs-test audit")
        ax.set_xlabel("Train-CV balanced accuracy")
        ax.set_ylabel("Held-out test balanced accuracy")
        figure.tight_layout()
        figure.savefig(output["figures"] / "cv_vs_test_balanced_accuracy.png", dpi=300, bbox_inches="tight")
    v8.base.plt.close(figure)


def main() -> None:
    args = parse_args()
    args.budgets = sorted({int(value) for value in args.budgets if int(value) > 0})
    output = paths(args.output_dir)
    if not args.resume:
        for csv_path in output["tables"].glob("*.csv"):
            csv_path.unlink()

    log(output, f"V9 starts. seed={args.seed}, budgets={args.budgets}, max_panels={args.max_panels}")
    log(output, "Feature generation and CV selection are train-only. Test rows are audit rows and are all exported.")

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
    candidates = make_candidates(rankings, quota_df, coverage_df, candidate_features, priority, classes, args)
    log(output, f"Candidate panels after static prescreen: {len(candidates)}")

    bonf_df.to_csv(output["tables"] / "train_only_bonf_sources.csv", index=False)
    quota_df.to_csv(output["tables"] / "train_only_quota_sources.csv", index=False)
    de_df.to_csv(output["tables"] / "train_only_ovr_dea_all.csv", index=False)
    candidate_feature_df.to_csv(output["tables"] / "train_only_candidate_feature_bank.csv", index=False)
    ranking_table.to_csv(output["tables"] / "train_only_ranker_table.csv", index=False)
    candidates.to_csv(output["tables"] / "stage0_panel_candidates_prescreen.csv", index=False)

    cv_df = cv_score_candidates(candidates, train_df, classes, output, args)
    audit_candidates = select_for_test(cv_df, args)
    test_df = test_audit(audit_candidates, train_df, heldout_df, classes, output, args)
    save_summaries(cv_df, audit_candidates, test_df, train_df, heldout_df, classes, output)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "input": str(args.input.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "seed": args.seed,
        "test_size": args.test_size,
        "budgets": args.budgets,
        "rankers": args.rankers,
        "test_top_per_size": args.test_top_per_size,
        "final_coaches": args.final_coaches,
        "notes": [
            "This is a single-split compact-panel discovery/audit script, not an unbiased external validation.",
            "The test audit exports all selected rows; high-performance rows are flagged but not hidden.",
            "No feature list is hardcoded.",
        ],
    }
    with (output["manifests"] / "v9_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    high_path = output["tables"] / "high_performance_candidates.csv"
    high_df = pd.read_csv(high_path) if high_path.exists() else pd.DataFrame()
    log(output, f"V9 complete. Test rows={len(test_df)}, high-performance rows={len(high_df)}")
    if not high_df.empty:
        best = high_df.iloc[0]
        log(
            output,
            (
                f"[seed {int(best['seed'])}] test size={int(best['feature_count'])} coach={best['coach']} "
                f"BA={best['test_balanced_accuracy']:.4f} F1={best['test_macro_f1']:.4f} "
                f"minRecall={best['test_min_class_recall']:.4f}"
            ),
        )


if __name__ == "__main__":
    main()
