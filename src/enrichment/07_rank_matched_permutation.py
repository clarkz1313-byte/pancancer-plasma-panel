#!/usr/bin/env python3
"""Direction-preserving GSEA leading-edge permutation audit.

Single-cancer panel genes are matched to genes at a similar signed position
in the same cancer ranking. Positive and negative tails are never folded
together. Sampling is exact and without replacement.

The locked-25 arm is descriptive. A candidate-bank sensitivity analysis is
included, but it is not a selection-aware null because the full deterministic
quota, coverage, ranking, and resize pipeline is not repeated in each draw.
"""
from __future__ import annotations

import csv
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


ROOT = Path("e:/Proteomics")
GSEA = ROOT / "revise_plan" / "gsea_analysis"
EV = ROOT / "revise_plan" / "external_validation_roadmap"
VRSX = ROOT / "revise_plan" / "part_b_multiclass" / "vRSX_v11_locked_reproducer" / "tables"
OUT = GSEA / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FDR_THRESHOLD = 0.25
SEED = 52
B = 5_000
PRIMARY_TOLERANCE = 2.0
TOLERANCES = (1.0, 2.0, 3.0, 5.0, 10.0)


def load_significant_cells() -> pd.DataFrame:
    results = pd.read_csv(GSEA / "gsea_all_results.csv")
    hallmark = results.loc[results["gene_set_collection"] == "MSigDB_Hallmark_2020"].copy()
    significant = hallmark.loc[hallmark["FDR q-val"] < FDR_THRESHOLD].copy()
    significant["lead_set"] = significant["Lead_genes"].fillna("").apply(
        lambda value: {gene for gene in value.split(";") if gene}
    )
    significant["nes_direction"] = np.where(significant["NES"] >= 0, "positive", "negative")
    return significant


def load_single_panels() -> dict[str, list[str]]:
    panels: dict[str, list[str]] = {}
    pattern = EV / "tables" / "single_panel_validation_panels" / "*_single_panel.csv"
    for filename in glob.glob(str(pattern)):
        cancer = os.path.basename(filename).replace("_single_panel.csv", "").upper()
        panels[cancer] = pd.read_csv(filename)["gene_symbol"].astype(str).tolist()
    return panels


def load_rank(cancer: str) -> pd.DataFrame:
    filename = GSEA / "ranked_lists" / f"{cancer.lower()}_ranked.rnk"
    rank = pd.read_csv(filename, sep="\t", header=None, names=["gene", "score"])
    if rank["gene"].duplicated().any():
        raise RuntimeError(f"Duplicate genes in {filename}")
    rank["signed_percentile"] = 100.0 * np.arange(len(rank)) / max(len(rank) - 1, 1)
    return rank


def overlap_count(genes: set[str] | list[str], lead_sets: list[set[str]]) -> int:
    members = set(genes)
    return sum(bool(members & leading_edge) for leading_edge in lead_sets)


def matched_candidates(rank: pd.DataFrame, genes: list[str], tolerance: float) -> tuple[list[np.ndarray], np.ndarray]:
    percentile = rank.set_index("gene")["signed_percentile"]
    pool_genes = rank["gene"].to_numpy()
    pool_percentiles = rank["signed_percentile"].to_numpy()
    target_percentiles = percentile.loc[genes].to_numpy(float)
    candidates = []
    for target in target_percentiles:
        mask = np.abs(pool_percentiles - target) <= tolerance
        if mask.sum() < 5:
            raise RuntimeError(f"Only {mask.sum()} candidates within tolerance {tolerance} at percentile {target}")
        candidates.append(pool_genes[mask])
    return candidates, target_percentiles


def exact_sample(candidates: list[np.ndarray], rng: np.random.Generator) -> set[str]:
    for _ in range(100):
        selected: set[str] = set()
        success = True
        for index in rng.permutation(len(candidates)):
            available = candidates[index][~np.isin(candidates[index], list(selected))]
            if len(available) == 0:
                success = False
                break
            selected.add(str(available[rng.integers(len(available))]))
        if success and len(selected) == len(candidates):
            return selected
    raise RuntimeError("Could not draw an exact matched set without replacement")


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


def run_single_panels(
    significant: pd.DataFrame,
    panels: dict[str, list[str]],
    tolerance: float,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray]:
    cancers = sorted(panels)
    seeds = np.random.SeedSequence(seed).spawn(len(cancers))
    per_cancer_rows = []
    null_by_cancer = []
    observed_total = 0
    for cancer, cancer_seed in zip(cancers, seeds):
        cells = significant.loc[significant["cancer_type"] == cancer, "lead_set"].tolist()
        rank = load_rank(cancer)
        rank_index = set(rank["gene"])
        genes = [gene for gene in panels[cancer] if gene in rank_index]
        missing = [gene for gene in panels[cancer] if gene not in rank_index]
        observed = overlap_count(genes, cells)
        observed_total += observed
        if not cells:
            per_cancer_rows.append(
                {
                    "cancer": cancer,
                    "display_cancer": "DLBCL" if cancer == "LYMPH" else cancer,
                    "panel_size": len(genes),
                    "unmapped": ";".join(missing),
                    "n_significant_cells": 0,
                    "n_positive_cells": 0,
                    "n_negative_cells": 0,
                    "observed_overlap": 0,
                    "null_mean": np.nan,
                    "empirical_p": np.nan,
                    "mc_ci_lower": np.nan,
                    "mc_ci_upper": np.nan,
                    "mean_signed_percentile": float(rank.set_index("gene").loc[genes, "signed_percentile"].mean()),
                    "tolerance_pct_points": tolerance,
                }
            )
            continue

        candidates, target_percentiles = matched_candidates(rank, genes, tolerance)
        rng = np.random.default_rng(cancer_seed)
        null = np.empty(B, dtype=np.int32)
        for iteration in range(B):
            selected = exact_sample(candidates, rng)
            null[iteration] = overlap_count(selected, cells)
        null_by_cancer.append(null)
        exceedances = int((null >= observed).sum())
        empirical_p = (exceedances + 1) / (B + 1)
        ci_low, ci_high = binomial_ci(exceedances, B)
        cell_subset = significant.loc[significant["cancer_type"] == cancer]
        per_cancer_rows.append(
            {
                "cancer": cancer,
                "display_cancer": "DLBCL" if cancer == "LYMPH" else cancer,
                "panel_size": len(genes),
                "unmapped": ";".join(missing),
                "n_significant_cells": len(cells),
                "n_positive_cells": int((cell_subset["NES"] >= 0).sum()),
                "n_negative_cells": int((cell_subset["NES"] < 0).sum()),
                "observed_overlap": observed,
                "null_mean": float(null.mean()),
                "empirical_p": empirical_p,
                "mc_ci_lower": ci_low,
                "mc_ci_upper": ci_high,
                "mean_signed_percentile": float(target_percentiles.mean()),
                "tolerance_pct_points": tolerance,
            }
        )

    null_total = np.sum(np.vstack(null_by_cancer), axis=0)
    total_exceedances = int((null_total >= observed_total).sum())
    total_p = (total_exceedances + 1) / (B + 1)
    ci_low, ci_high = binomial_ci(total_exceedances, B)
    summary = {
        "test": "single_panels_direction_preserving_rank_matched",
        "observed": observed_total,
        "null_mean": float(null_total.mean()),
        "null_sd": float(null_total.std(ddof=1)),
        "exceedances": total_exceedances,
        "permutations": B,
        "empirical_p": total_p,
        "mc_ci_lower": ci_low,
        "mc_ci_upper": ci_high,
        "tolerance_pct_points": tolerance,
        "tested_cancers_with_significant_cells": len(null_by_cancer),
        "total_panel_cancers": len(cancers),
        "matching_definition": "same cancer and signed rank percentile, exact draw without replacement",
    }
    per_cancer = pd.DataFrame(per_cancer_rows)
    tested = per_cancer["empirical_p"].notna()
    p_values = per_cancer.loc[tested, "empirical_p"].to_numpy(float)
    per_cancer.loc[tested, "bh_q_across_11_testable_cancers"] = bh_adjust(p_values)
    per_cancer.loc[tested, "bonferroni_p_across_11_testable_cancers"] = np.minimum(p_values * len(p_values), 1.0)
    return summary, per_cancer, null_total


def locked25_descriptive(significant: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    locked = pd.read_csv(VRSX / "locked_selected_features.csv")["protein"].astype(str).tolist()
    candidate_bank = pd.read_csv(VRSX / "train_only_candidate_feature_bank.csv")["protein"].astype(str).tolist()
    all_leading_edges = significant["lead_set"].tolist()
    observed = overlap_count(locked, all_leading_edges)
    if not set(locked).issubset(candidate_bank):
        raise RuntimeError("The locked panel is not fully contained in the candidate bank")
    rng = np.random.default_rng(seed)
    null = np.empty(B, dtype=np.int32)
    for iteration in range(B):
        selected = set(rng.choice(candidate_bank, size=len(locked), replace=False))
        null[iteration] = overlap_count(selected, all_leading_edges)
    exceedances = int((null >= observed).sum())
    empirical_p = (exceedances + 1) / (B + 1)
    ci_low, ci_high = binomial_ci(exceedances, B)
    row = {
        "test": "locked25_candidate_bank_sensitivity_not_selection_aware",
        "observed": observed,
        "candidate_bank_size": len(candidate_bank),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "exceedances": exceedances,
        "permutations": B,
        "sensitivity_empirical_p": empirical_p,
        "mc_ci_lower": ci_low,
        "mc_ci_upper": ci_high,
        "inference_status": "descriptive_only_full_selection_pipeline_not_repeated",
    }
    return pd.DataFrame([row]), null


def hit_details(significant: pd.DataFrame, panels: dict[str, list[str]]) -> pd.DataFrame:
    locked = set(pd.read_csv(VRSX / "locked_selected_features.csv")["protein"].astype(str))
    rows = []
    for _, cell in significant.iterrows():
        cancer = cell["cancer_type"]
        leading = cell["lead_set"]
        single_hits = sorted(set(panels.get(cancer, [])) & leading)
        locked_hits = sorted(locked & leading)
        if single_hits or locked_hits:
            rows.append(
                {
                    "cancer": cancer,
                    "display_cancer": "DLBCL" if cancer == "LYMPH" else cancer,
                    "term": cell["Term"],
                    "NES": cell["NES"],
                    "FDR_q": cell["FDR q-val"],
                    "nes_direction": cell["nes_direction"],
                    "single_panel_hits": ";".join(single_hits),
                    "locked25_hits": ";".join(locked_hits),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    significant = load_significant_cells()
    panels = load_single_panels()
    tolerance_rows = []
    primary_per_cancer = None
    primary_null = None
    for index, tolerance in enumerate(TOLERANCES):
        summary, per_cancer, null_total = run_single_panels(
            significant,
            panels,
            tolerance,
            SEED + index * 1000,
        )
        tolerance_rows.append(summary)
        print(
            f"tolerance={tolerance:4.1f}: observed={summary['observed']} "
            f"null={summary['null_mean']:.3f} p={summary['empirical_p']:.4f}"
        )
        if tolerance == PRIMARY_TOLERANCE:
            primary_per_cancer = per_cancer
            primary_null = null_total

    if primary_per_cancer is None or primary_null is None:
        raise RuntimeError("Primary tolerance was not run")
    primary_summary = pd.DataFrame(
        [row for row in tolerance_rows if row["tolerance_pct_points"] == PRIMARY_TOLERANCE]
    )
    locked_summary, locked_null = locked25_descriptive(significant, SEED + 9000)
    details = hit_details(significant, panels)

    primary_summary.to_csv(OUT / "direction_matched_permutation_summary_v2.csv", index=False)
    primary_summary.to_csv(OUT / "rank_matched_permutation_summary.csv", index=False)
    primary_per_cancer.to_csv(OUT / "direction_matched_permutation_per_cancer_v2.csv", index=False)
    primary_per_cancer.to_csv(OUT / "rank_matched_permutation_per_cancer.csv", index=False)
    pd.DataFrame(tolerance_rows).to_csv(OUT / "direction_matched_tolerance_sensitivity_v2.csv", index=False)
    pd.DataFrame({"overlap_count": primary_null}).to_csv(OUT / "direction_matched_null_single_v2.csv", index=False)
    locked_summary.to_csv(OUT / "locked25_pathway_overlap_descriptive_v2.csv", index=False)
    pd.DataFrame({"overlap_count": locked_null}).to_csv(OUT / "locked25_candidate_bank_null_v2.csv", index=False)
    details.to_csv(OUT / "panel_pathway_hits_detail_v2.csv", index=False)
    print(locked_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
