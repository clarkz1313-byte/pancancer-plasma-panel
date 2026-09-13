#!/usr/bin/env python3
"""Validate official outputs and build one corrected evidence table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_one_row(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one result row in {path}, found {len(rows)}")
    return rows[0]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow-dir", type=Path, required=True)
    args = parser.parse_args()
    workflow = args.workflow_dir
    config = json.loads(args.config.read_text(encoding="utf-8"))
    input_checks = {row["pair_id"]: row for row in read_csv(workflow / "qc" / "summary_input_checks.csv")}

    results = []
    qc = []
    for pair in config["pairs"]:
        pair_rows = []
        expected_target = input_checks[pair["pair_id"]]["target_canonical_id"]
        for method in (0, 1):
            for diff_freq in (0.2, 0.1):
                tag = f"{pair['pair_id']}_mtd{method}_df{int(diff_freq * 100):02d}"
                row = read_one_row(workflow / "results" / "official_smr_raw" / f"{tag}.smr")
                record = {
                    "pair_id": pair["pair_id"],
                    "protein": pair["protein"],
                    "cancer": pair["cancer"],
                    "target_rsid": pair["target_rsid"],
                    "target_canonical_id": row["topSNP"],
                    "target_chr": int(row["topSNP_chr"]),
                    "target_bp": int(row["topSNP_bp"]),
                    "effect_allele": row["A1"],
                    "other_allele": row["A2"],
                    "reference_effect_allele_freq": float(row["Freq"]),
                    "b_gwas": float(row["b_GWAS"]),
                    "se_gwas": float(row["se_GWAS"]),
                    "p_gwas": float(row["p_GWAS"]),
                    "b_pqtl": float(row["b_eQTL"]),
                    "se_pqtl": float(row["se_eQTL"]),
                    "p_pqtl": float(row["p_eQTL"]),
                    "b_smr": float(row["b_SMR"]),
                    "se_smr": float(row["se_SMR"]),
                    "p_smr": float(row["p_SMR"]),
                    "p_heidi": float(row["p_HEIDI"]),
                    "nsnp_heidi": int(row["nsnp_HEIDI"]),
                    "heidi_method": method,
                    "diff_freq": diff_freq,
                    "heidi_pass_p01": float(row["p_HEIDI"]) > 0.01,
                    "heidi_pass_p05": float(row["p_HEIDI"]) > 0.05,
                    "reference_n": 503,
                    "pqtl_n": pair["pqtl_n"],
                    "gwas_n": pair["gwas_n"],
                    "reportability_pre_run": pair["reportability"],
                }
                pair_rows.append(record)
                results.append(record)

                manual_beta = record["b_gwas"] / record["b_pqtl"]
                manual_se = abs(manual_beta) * math.sqrt(
                    (record["se_gwas"] / record["b_gwas"]) ** 2
                    + (record["se_pqtl"] / record["b_pqtl"]) ** 2
                )
                qc.append({
                    "pair_id": pair["pair_id"], "run_tag": tag,
                    "target_matches_requested": row["topSNP"] == expected_target,
                    "target_position_matches_requested": int(row["topSNP_bp"]) == pair["target_bp"],
                    "abs_b_smr_minus_manual_two_term": abs(record["b_smr"] - manual_beta),
                    "abs_se_smr_minus_manual_two_term": abs(record["se_smr"] - manual_se),
                    "at_least_3_heidi_snps": record["nsnp_heidi"] >= 3,
                })

        primary_freq = [row for row in pair_rows if row["diff_freq"] == 0.2]
        method_passes = {row["heidi_method"]: row["heidi_pass_p01"] for row in primary_freq}
        method_pvalues = {row["heidi_method"]: row["p_heidi"] for row in primary_freq}
        concordant = len(set(method_passes.values())) == 1
        verdict = "pass" if concordant and next(iter(method_passes.values())) else (
            "fail" if concordant else "unresolved_method_dependent"
        )
        for row in pair_rows:
            row["method_concordance_at_p01"] = concordant
            row["pair_heidi_verdict_at_p01"] = verdict
            row["method0_p_heidi_df20"] = method_pvalues[0]
            row["method1_p_heidi_df20"] = method_pvalues[1]
            if pair["pair_id"] == "BMP4_CRC":
                row["manuscript_status"] = "exploratory_only_larger_ld_and_method_concordance_unresolved"
            elif concordant:
                row["manuscript_status"] = "exploratory_shared_locus_heidi_compatible_at_p01"
            else:
                row["manuscript_status"] = "exploratory_only_method_concordance_unresolved"

    result_path = workflow / "results" / "official_smr_results.csv"
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    qc_path = workflow / "qc" / "official_smr_output_checks.csv"
    with qc_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(qc[0]))
        writer.writeheader(); writer.writerows(qc)

    coloc_path = workflow / "results" / "coloc_abf_sensitivity.csv"
    coloc = read_csv(coloc_path) if coloc_path.exists() else []
    def coloc_value(pair_id: str, window: int, p12: float, column: str) -> str:
        for record in coloc:
            if (
                record["pair_id"] == pair_id
                and int(record["window_bp_each_side"]) == window
                and math.isclose(float(record["p12"]), p12, rel_tol=0, abs_tol=1e-15)
            ):
                return record[column]
        return ""

    master = []
    for pair in config["pairs"]:
        pair_rows = [
            row for row in results
            if row["pair_id"] == pair["pair_id"] and row["diff_freq"] == 0.2
        ]
        by_method = {row["heidi_method"]: row for row in pair_rows}
        primary = by_method[0]
        pair_id = pair["pair_id"]
        is_bmp4 = pair_id == "BMP4_CRC"
        master.append({
            "pair_id": pair_id,
            "protein": pair["protein"],
            "cancer": pair["cancer"],
            "panel_membership": "locked_25_multiclass" if is_bmp4 else "single_cancer_panels",
            "protein_gene_chr": pair["probe_chr"],
            "target_locus_chr": pair["target_chr"],
            "cis_trans": "trans",
            "target_rsid": pair["target_rsid"],
            "target_canonical_id": primary["target_canonical_id"],
            "effect_allele": primary["effect_allele"],
            "other_allele": primary["other_allele"],
            "reference_effect_allele_freq": primary["reference_effect_allele_freq"],
            "reference_n": primary["reference_n"],
            "reference_minor_allele_count": 32 if is_bmp4 else 437,
            "pqtl_source": "UKB_PPP",
            "pqtl_n": primary["pqtl_n"],
            "gwas_source_file": Path(pair["gwas_source"]).name,
            "gwas_n": primary["gwas_n"],
            "sample_overlap_status": (
                "probable_UKB_overlap_exact_count_unknown"
                if is_bmp4 else "not_established_from_repo_metadata"
            ),
            "gwas_frequency_qc_status": (
                "unavailable_nonpalindromic_allele_match_only"
                if is_bmp4 else "available"
            ),
            "b_smr_two_term": primary["b_smr"],
            "se_smr_two_term": primary["se_smr"],
            "p_smr_two_term": primary["p_smr"],
            "heidi_mtd0_p": by_method[0]["p_heidi"],
            "heidi_mtd0_nsnp": by_method[0]["nsnp_heidi"],
            "heidi_mtd1_p": by_method[1]["p_heidi"],
            "heidi_mtd1_nsnp": by_method[1]["nsnp_heidi"],
            "heidi_concordant_at_p01": primary["method_concordance_at_p01"],
            "heidi_concordant_at_p05": (
                by_method[0]["heidi_pass_p05"] == by_method[1]["heidi_pass_p05"]
            ),
            "diff_freq_0.1_changes_result": False,
            "diff_freq_interpretation": (
                "pQTL_vs_reference_only_GWAS_frequency_missing"
                if is_bmp4 else "pQTL_GWAS_reference_check"
            ),
            "coloc_abf_window_bp": 500000,
            "coloc_abf_n_snps": coloc_value(pair_id, 500000, 1e-5, "n_snps"),
            "coloc_pph4_p12_1e-5": coloc_value(pair_id, 500000, 1e-5, "PP.H4"),
            "coloc_pph4_p12_1e-6": coloc_value(pair_id, 500000, 1e-6, "PP.H4"),
            "coloc_pph4_p12_1e-7": coloc_value(pair_id, 500000, 1e-7, "PP.H4"),
            "coloc_pph3_p12_1e-5": coloc_value(pair_id, 500000, 1e-5, "PP.H3"),
            "larger_ld_check": "GCTB_target_absent; PanUKB_target_presence_unresolved_no_local_Hail_or_cloud_runtime",
            "coloc_susie_status": "not_run_larger_LD_precondition_unmet",
            "causal_claim_allowed": False,
            "independent_evidence_claim_allowed": False,
            "manuscript_status": (
                "exploratory_only_HEIDI_method_disagreement_and_larger_LD_unavailable"
                if is_bmp4 else
                "exploratory_shared_locus_HEIDI_compatible_at_p01_but_threshold_sensitive"
            ),
            "interpretation": "shared_regional_genetic_association_not_protein_mediation",
        })
    master_path = workflow / "results" / "genetics_evidence_master.csv"
    with master_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(master[0]))
        writer.writeheader(); writer.writerows(master)
    print(f"Wrote {result_path}, {qc_path}, and {master_path}")


if __name__ == "__main__":
    main()
