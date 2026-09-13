#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    CANCER_SLUGS,
    SOURCE_INPUT,
    cancer_menu_text,
    run_single_cancer_pipeline,
    slug_from_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the revised independent target-vs-rest workflow for one cancer."
    )
    parser.add_argument("--cancer", choices=CANCER_SLUGS, default=None)
    parser.add_argument("--index", type=int, default=None, help="Cancer index from the numbered menu.")
    parser.add_argument("--list-cancers", action="store_true")
    parser.add_argument("--input", type=Path, default=SOURCE_INPUT)
    parser.add_argument("--class-column", default="Cancer")
    parser.add_argument("--sample-id-column", default="Sample_ID")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--imputer", default="knn")
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--panel-sizes", nargs="+", type=int, default=list(range(1, 19)))
    parser.add_argument("--no-all-panel", action="store_true")
    parser.add_argument("--run-enrichment", action="store_true")
    parser.add_argument("--run-robustness", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def resolve_cancer(args: argparse.Namespace) -> str:
    if args.cancer is not None:
        return args.cancer
    if args.index is not None:
        return slug_from_index(args.index)
    raise ValueError("No cancer selected.")


def main() -> None:
    args = parse_args()
    if args.list_cancers:
        print(cancer_menu_text())
        if args.cancer is None and args.index is None:
            return

    try:
        cancer_slug = resolve_cancer(args)
    except ValueError:
        print(cancer_menu_text())
        print("\nRun one cancer with either:")
        print("  python revise_plan/scripts/run_single_cancer.py --index 8")
        print("  python revise_plan/scripts/run_single_cancer.py --cancer lungc")
        return

    result = run_single_cancer_pipeline(
        cancer_slug=cancer_slug,
        input_path=args.input,
        class_column=args.class_column,
        sample_id_column=args.sample_id_column,
        seed=args.seed,
        test_size=args.test_size,
        imputer=args.imputer,
        knn_neighbors=args.knn_neighbors,
        panel_sizes=args.panel_sizes,
        include_all_panel=not args.no_all_panel,
        run_enrichment=args.run_enrichment,
        run_robustness=args.run_robustness,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(f"Completed revised single-cancer run: {result['label']} ({result['slug']})")
    print(f"Output folder: {result['paths'].root}")
    print(f"Best panel AUC: {result['aggregate_row']['best_panel_test_auc']}")


if __name__ == "__main__":
    main()
