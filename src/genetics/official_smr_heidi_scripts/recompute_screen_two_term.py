#!/usr/bin/env python3
"""Supersede the NOME Wald columns with the two-term Zhu variance.

This is a descriptive screening statistic. It is not independent evidence
from the GWAS or colocalization inputs and is not used as a causal claim.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path("E:/Proteomics")
SOURCE = ROOT / "revise_plan/smr_coloc/results/tier1_combined_316.csv"
OUT_DIR = ROOT / "revise_plan/smr_coloc/official_smr_heidi/screen"

LEGACY_COLUMNS = {
    "b_wald_pkg": "legacy_b_wald_nome",
    "se_wald_pkg": "legacy_se_wald_nome",
    "p_wald_pkg": "legacy_p_wald_nome_equals_gwas_p",
    "p_heidi_rho0": "withdrawn_custom_p_heterogeneity_rho0",
    "bonferroni_threshold": "legacy_bonferroni_threshold",
    "passes_wald_bonferroni": "legacy_passes_nome_bonferroni",
    "heidi_descriptive_pass_rho0": "withdrawn_custom_heterogeneity_pass_rho0",
    "final_hit": "legacy_final_hit_nome_plus_custom_test",
}


def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def main() -> None:
    rows = []
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        for source_row in csv.DictReader(handle):
            row = {
                LEGACY_COLUMNS.get(column, column): value
                for column, value in source_row.items()
            }
            bx = float(row["b_pqtl_lead"]) if row.get("b_pqtl_lead") else math.nan
            sx = float(row["se_pqtl_lead"]) if row.get("se_pqtl_lead") else math.nan
            by = float(row["b_gwas_lead"]) if row.get("b_gwas_lead") else math.nan
            sy = float(row["se_gwas_lead"]) if row.get("se_gwas_lead") else math.nan
            if all(math.isfinite(value) for value in (bx, sx, by, sy)) and bx != 0:
                beta = by / bx
                variance = (sy * sy / (bx * bx)) + (
                    by * by * sx * sx / (bx * bx * bx * bx)
                )
                se = math.sqrt(variance)
                z = beta / se
                p = normal_two_sided_p(z)
            else:
                beta = se = z = p = math.nan
            row["b_wald_two_term"] = beta
            row["se_wald_two_term"] = se
            row["z_wald_two_term"] = z
            row["p_wald_two_term"] = p
            row["wald_variance_definition"] = "Zhu2016_two_term_delta"
            row["wald_role"] = "descriptive_not_independent_gate"
            rows.append(row)

    bonferroni = 0.05 / len(rows)
    for row in rows:
        p = row["p_wald_two_term"]
        row["bonferroni_threshold_316"] = bonferroni
        row["passes_two_term_bonferroni_descriptive"] = math.isfinite(p) and p < bonferroni

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / "tier1_combined_316_two_term.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
