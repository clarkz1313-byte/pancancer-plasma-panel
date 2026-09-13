#!/usr/bin/env python3
"""Descriptive panel correlation analysis with an assay-matched reference.

The analysis excludes all metadata, including protein_count. It reports
overall and within-cancer mean absolute correlations. Empirical p-values are
post-selection descriptions, not confirmatory tests of panel efficiency.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


ROOT = Path("e:/Proteomics")
HERE = ROOT / "revise_plan" / "ppi_string"
MATRIX = ROOT / "data" / "processed" / "filtered_pancancer_data.csv"
PANELS_CSV = HERE / "disease_specific_panels.csv"
LOCKED25_CSV = ROOT / "revise_plan" / "part_b_multiclass" / "vRSX_v11_locked_reproducer" / "tables" / "locked_selected_features.csv"
OUT_V2 = HERE / "panel_redundancy_results_v2.csv"
OUT_CANONICAL = HERE / "panel_redundancy_results.csv"
N_PERM = 10_000
SEED = 42
METADATA_COLUMNS = {"Sample_ID", "Cancer", "protein_count"}


def load_panels() -> dict[str, list[str]]:
    panels: dict[str, list[str]] = {}
    with PANELS_CSV.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            panels[row["Panel"]] = [g.strip() for g in row["Genes"].split(";") if g.strip()]
    locked = pd.read_csv(LOCKED25_CSV)["protein"].astype(str).tolist()
    return {"LOCKED25": locked, **panels}


def mean_abs_offdiag(correlation: np.ndarray, indices: np.ndarray) -> float:
    sub = np.abs(correlation[np.ix_(indices, indices)])
    upper = sub[np.triu_indices(len(indices), k=1)]
    return float(upper.mean())


def bh_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def binomial_ci(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def main() -> int:
    data = pd.read_csv(MATRIX)
    cancer = data["Cancer"].to_numpy()
    protein_columns = [column for column in data.columns if column not in METADATA_COLUMNS]
    if len(protein_columns) != 1463:
        raise RuntimeError(f"Expected 1,463 proteins, found {len(protein_columns)}")
    values = data[protein_columns].to_numpy(dtype=float)
    if np.isnan(values).any():
        means = np.nanmean(values, axis=0)
        missing = np.where(np.isnan(values))
        values[missing] = np.take(means, missing[1])

    within = values.copy()
    for label in np.unique(cancer):
        rows = cancer == label
        within[rows] -= within[rows].mean(axis=0, keepdims=True)

    correlations = {
        "overall": np.corrcoef(values, rowvar=False),
        "within_cancer": np.corrcoef(within, rowvar=False),
    }
    index = {protein: i for i, protein in enumerate(protein_columns)}
    panels = load_panels()
    jobs = [(panel, view) for panel in panels for view in correlations]
    seeds = np.random.SeedSequence(SEED).spawn(len(jobs))
    rows = []
    for (panel, view), job_seed in zip(jobs, seeds):
        members = panels[panel]
        mapped = [gene for gene in members if gene in index]
        missing = [gene for gene in members if gene not in index]
        indices = np.asarray([index[gene] for gene in mapped], dtype=int)
        if len(indices) < 2:
            continue
        matrix = correlations[view]
        observed = mean_abs_offdiag(matrix, indices)
        rng = np.random.default_rng(job_seed)
        null = np.empty(N_PERM)
        for iteration in range(N_PERM):
            selected = rng.choice(len(protein_columns), size=len(indices), replace=False)
            null[iteration] = mean_abs_offdiag(matrix, selected)
        exceedances = int((null <= observed).sum())
        empirical_p = (exceedances + 1) / (N_PERM + 1)
        ci_low, ci_high = binomial_ci(exceedances, N_PERM)
        rows.append(
            {
                "panel": panel,
                "display_panel": "DLBCL" if panel == "LYMPH" else panel,
                "view": view,
                "n_proteins": len(mapped),
                "unmapped": ";".join(missing),
                "observed_mean_abs_r": observed,
                "assay_null_mean_abs_r": float(null.mean()),
                "assay_null_sd_abs_r": float(null.std(ddof=1)),
                "exceedances_less_or_equal": exceedances,
                "permutations": N_PERM,
                "descriptive_empirical_p_unadjusted": empirical_p,
                "mc_ci_lower": ci_low,
                "mc_ci_upper": ci_high,
                "inference_status": "descriptive_post_selection_not_confirmatory",
            }
        )

    output = pd.DataFrame(rows)
    p_values = output["descriptive_empirical_p_unadjusted"].to_numpy(float)
    output["bh_q_across_26_tests"] = bh_adjust(p_values)
    output["bonferroni_p_across_26_tests"] = np.minimum(p_values * len(output), 1.0)
    output.to_csv(OUT_V2, index=False, encoding="utf-8-sig")
    output.to_csv(OUT_CANONICAL, index=False, encoding="utf-8-sig")
    print(output.sort_values("descriptive_empirical_p_unadjusted").to_string(index=False))
    print(f"Wrote {OUT_V2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
