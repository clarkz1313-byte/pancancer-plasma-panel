"""Assay-background over-representation analysis for all manuscript panels."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "filtered_pancancer_data.csv"
MULTICLASS = (
    ROOT
    / "revise_plan"
    / "external_validation_roadmap"
    / "tables"
    / "pathway_input_part_b_25_panel.csv"
)
SINGLE = (
    ROOT
    / "revise_plan"
    / "part_a_single_seed52_aligned"
    / "single_panel_seed52_aligned_performance.csv"
)
GENE_SETS = {
    "GO_BP_2023": ROOT / "references" / "gene_sets" / "GO_Biological_Process_2023.gmt",
    "KEGG_2021_Human": (
        ROOT
        / "revise_plan"
        / "gsea_analysis"
        / "results"
        / "aml"
        / "KEGG_2021_Human"
        / "gene_sets.gmt"
    ),
    "Reactome_2022": ROOT / "references" / "gene_sets" / "Reactome_2022.gmt",
}
OUT = ROOT / "revision_package_2026-09-04" / "05_data_audit" / "ora_assay_background_2026-09-05"
METADATA = {"Sample_ID", "Cancer", "protein_count"}


def read_gmt(path: Path) -> dict[str, set[str]]:
    terms: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 3:
                terms[row[0]] = {gene.strip().upper() for gene in row[2:] if gene.strip()}
    return terms


def bh_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def load_panels() -> dict[str, set[str]]:
    panels: dict[str, set[str]] = {
        "multiclass_25": set(pd.read_csv(MULTICLASS)["gene_symbol"].str.upper())
    }
    single = pd.read_csv(SINGLE)
    for row in single.itertuples(index=False):
        display_class = "DLBCL" if row.target_class == "LYMPH" else row.target_class
        panels[f"single_{display_class}"] = {
            value.strip().upper() for value in str(row.proteins).split(";") if value.strip()
        }
    panels["single_pooled_unique"] = set().union(
        *(genes for name, genes in panels.items() if name.startswith("single_") and name != "single_pooled_unique")
    )
    return panels


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    header = pd.read_csv(DATA, nrows=0).columns
    universe = {str(column).upper() for column in header if column not in METADATA}
    panels = load_panels()

    pd.DataFrame({"assayed_symbol": sorted(universe)}).to_csv(OUT / "assay_universe_1463.csv", index=False)
    panel_rows = [
        {"panel": panel, "protein": protein}
        for panel, proteins in panels.items()
        for protein in sorted(proteins)
    ]
    pd.DataFrame(panel_rows).to_csv(OUT / "panel_memberships.csv", index=False)

    all_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for database, path in GENE_SETS.items():
        terms = read_gmt(path)
        annotated_universe = universe.intersection(set().union(*terms.values()))
        tested_terms = {
            term: genes.intersection(annotated_universe)
            for term, genes in terms.items()
            if 2 <= len(genes.intersection(annotated_universe)) <= 500
        }
        for panel, raw_genes in panels.items():
            query = raw_genes.intersection(annotated_universe)
            coverage_rows.append(
                {
                    "panel": panel,
                    "database": database,
                    "intended_panel_size": len(raw_genes),
                    "annotated_panel_size": len(query),
                    "assay_universe_size": len(universe),
                    "annotated_universe_size": len(annotated_universe),
                    "tested_term_count": len(tested_terms),
                    "unmapped_panel_symbols": ";".join(sorted(raw_genes - query)),
                }
            )
            rows: list[dict[str, object]] = []
            for term, term_genes in tested_terms.items():
                overlap = query.intersection(term_genes)
                p_value = float(
                    hypergeom.sf(
                        len(overlap) - 1,
                        len(annotated_universe),
                        len(term_genes),
                        len(query),
                    )
                )
                rows.append(
                    {
                        "panel": panel,
                        "database": database,
                        "term": term,
                        "overlap_count": len(overlap),
                        "panel_annotated_n": len(query),
                        "term_assay_n": len(term_genes),
                        "assay_annotated_n": len(annotated_universe),
                        "overlap_genes": ";".join(sorted(overlap)),
                        "p_value": p_value,
                    }
                )
            p_values = np.asarray([row["p_value"] for row in rows], dtype=float)
            q_values = bh_adjust(p_values)
            for row, q_value in zip(rows, q_values, strict=True):
                row["p_adjust_bh"] = float(q_value)
            all_rows.extend(rows)

    results = pd.DataFrame(all_rows).sort_values(["panel", "database", "p_adjust_bh", "p_value"])
    results.to_csv(OUT / "ora_all_terms.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(OUT / "ora_mapping_and_test_counts.csv", index=False)

    observed = results.loc[results["overlap_count"] > 0].copy()
    observed.to_csv(OUT / "ora_terms_with_panel_overlap.csv", index=False)
    significant = results.loc[results["p_adjust_bh"] < 0.05].copy()
    significant.to_csv(OUT / "ora_significant_bh_0.05.csv", index=False)

    summary = (
        results.groupby(["panel", "database"], as_index=False)
        .agg(
            tested_terms=("term", "size"),
            terms_with_overlap=("overlap_count", lambda values: int((values > 0).sum())),
            minimum_p=("p_value", "min"),
            minimum_bh_q=("p_adjust_bh", "min"),
            significant_terms_bh_0_05=("p_adjust_bh", lambda values: int((values < 0.05).sum())),
        )
    )
    summary.to_csv(OUT / "ora_summary.csv", index=False)


if __name__ == "__main__":
    main()
