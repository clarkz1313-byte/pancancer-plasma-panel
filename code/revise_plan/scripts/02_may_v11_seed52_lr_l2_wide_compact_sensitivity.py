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

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
PART_B_ROOT = ROOT / "revise_plan" / "part_b_multiclass"
V10_PATH = SCRIPT_DIR / "02_may_v10_seed52_lr_l2_compact_tournament.py"

spec = importlib.util.spec_from_file_location("v10", V10_PATH)
v10 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v10)

v9 = v10.v9
v8 = v10.v8

DEFAULT_OUTPUT = PART_B_ROOT / "v11_seed52_lr_l2_wide_compact_sensitivity"
DEFAULT_BUDGETS = tuple(range(12, 31))
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
            "V11 LR_L2-only wide compact sensitivity run. This is an overnight-style scripted tournament "
            "from 12 to 30 proteins to check whether the V10 24-protein candidate is stable rather than pinned."
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
    parser.add_argument("--candidate-bank-size", type=int, default=180)
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--rankers", nargs="+", default=list(DEFAULT_RANKERS), choices=list(v8.SEARCH_RANKERS))
    parser.add_argument("--random-panels-per-size", type=int, default=700)
    parser.add_argument("--max-panels", type=int, default=3000)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--test-top-per-size", type=int, default=30)
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


def standardize_trial_no(output: dict[str, Path]) -> None:
    targets = [
        output["tables"] / "stage2_test_audit_all_rows.csv",
        output["tables"] / "stage2_test_audit_lr_l2_all_rows.csv",
        output["tables"] / "high_performance_lr_l2_candidates.csv",
        output["tables"] / "best_lr_l2_candidate_summary.csv",
    ]
    for path in targets:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "trial_mo" in df.columns:
            df = df.rename(columns={"trial_mo": "trial_no"})
        if "trial_no" not in df.columns:
            df.insert(0, "trial_no", range(1, len(df) + 1))
        else:
            df["trial_no"] = range(1, len(df) + 1)
            cols = ["trial_no", *[column for column in df.columns if column != "trial_no"]]
            df = df[cols]
        df.to_csv(path, index=False)


def cleanup_noncanonical_outputs(output: dict[str, Path]) -> None:
    """Keep V11 outputs focused on canonical LR_L2 audit artifacts."""
    remove_paths = [
        output["tables"] / "stage2_test_audit_all_rows.csv",
        output["tables"] / "best_softvote_by_feature_count.csv",
        output["tables"] / "high_performance_candidates.csv",
        output["tables"] / "stage2_classification_reports.csv",
        output["tables"] / "stage2_confusion_matrices_long.csv",
        output["tables"] / "stage2_bootstrap_ci_long.csv",
        output["figures"] / "softvote_test_ba_by_feature_count.png",
        output["figures"] / "cv_vs_test_balanced_accuracy.png",
    ]
    for path in remove_paths:
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    args.budgets = sorted({int(value) for value in args.budgets if int(value) > 0})
    args.final_coaches = ["LR_L2"]
    output = v9.paths(args.output_dir)
    v10.clean_outputs(output, args.resume)

    log(output, f"V11 LR_L2-only wide compact sensitivity starts. seed={args.seed}, budgets={args.budgets}")
    log(output, f"Search size cap={args.max_panels}, test_top_per_size={args.test_top_per_size}, final learner=LR_L2 only.")

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
    test_df = v10.add_bootstrap_ci_columns(test_df, train_df, heldout_df, classes, output, args)
    test_df.to_csv(output["tables"] / "stage2_test_audit_all_rows.csv", index=False)
    v9.save_summaries(cv_df, audit_candidates, test_df, train_df, heldout_df, classes, output)
    v10.save_lr_l2_specific_summaries(output)
    standardize_trial_no(output)
    cleanup_noncanonical_outputs(output)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "input": str(args.input.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "seed": args.seed,
        "test_size": args.test_size,
        "budgets": args.budgets,
        "rankers": args.rankers,
        "final_coach": "LR_L2",
        "max_panels": args.max_panels,
        "test_top_per_size": args.test_top_per_size,
        "bootstrap_samples_per_row": args.bootstrap_samples,
        "notes": [
            "V11 is an LR_L2-only wide compact sensitivity tournament from 12 to 30 proteins.",
            "It is intended to test whether the 24-feature V10 result is robust to a broader compact search.",
            "The held-out test audit remains internal candidate discovery, not external validation.",
            "Canonical audit rows use trial_no, not trial_mo.",
            "Canonical test table is tables/stage2_test_audit_lr_l2_all_rows.csv and includes bootstrap CI columns.",
        ],
    }
    with (output["manifests"] / "v11_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    best_path = output["tables"] / "best_lr_l2_candidate_summary.csv"
    if best_path.exists():
        best = pd.read_csv(best_path).iloc[0]
        log(
            output,
            (
                f"[seed {int(best['seed'])}] trial_no={int(best['trial_no'])} "
                f"test size={int(best['feature_count'])} coach=LR_L2 "
                f"BA={best['test_balanced_accuracy']:.4f} F1={best['test_macro_f1']:.4f} "
                f"minRecall={best['test_min_class_recall']:.4f}"
            ),
        )
    log(output, "V11 complete.")


if __name__ == "__main__":
    main()
