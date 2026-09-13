#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    CANCER_SLUGS,
    SOURCE_INPUT,
    aggregate_pan_cancer_outputs,
    cancer_menu_text,
    pan_cancer_paths,
    run_single_cancer_pipeline,
    slugs_from_indexes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the revised independent target-vs-rest workflow across all cancers."
    )
    parser.add_argument("--cancers", nargs="+", choices=CANCER_SLUGS, default=None)
    parser.add_argument("--indexes", nargs="+", type=int, default=None, help="Cancer indexes from the numbered menu.")
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


def resolve_cancers(args: argparse.Namespace) -> list[str]:
    if args.cancers:
        return args.cancers
    if args.indexes:
        return slugs_from_indexes(args.indexes)
    return list(CANCER_SLUGS)


def main() -> None:
    args = parse_args()
    if args.list_cancers:
        print(cancer_menu_text())
        if args.cancers is None and args.indexes is None:
            return

    cancer_slugs = resolve_cancers(args)
    results = []
    for cancer_slug in cancer_slugs:
        print(f"[run] {cancer_slug}")
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
        results.append(result)

    aggregate_pan_cancer_outputs(results)
    pan_paths = pan_cancer_paths()
    print("Completed revised pan-cancer run.")
    print(f"Cancers processed: {', '.join(cancer_slugs)}")
    print(f"Pan-cancer output folder: {pan_paths['root']}")


if __name__ == "__main__":
    main()
