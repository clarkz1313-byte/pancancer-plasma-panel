#!/usr/bin/env python3
"""Run acceptance checks for the official SMR and HEIDI patch."""

from __future__ import annotations

import csv
import json
from pathlib import Path


WORKFLOW = Path("E:/Proteomics/revise_plan/smr_coloc/official_smr_heidi")
ROOT = Path("E:/Proteomics")


def read_csv(path: Path, delimiter: str = ",") -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def add(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": passed, "detail": detail})


def main() -> None:
    checks: list[dict] = []
    run = json.loads((WORKFLOW / "provenance/official_smr_run.json").read_text(encoding="utf-8"))
    commands = run["commands"]
    add(checks, "all_official_commands_exit_zero", all(item["exit_code"] == 0 for item in commands), f"n={len(commands)}")

    official = read_csv(WORKFLOW / "results/official_smr_results.csv")
    add(checks, "eight_official_sensitivity_rows", len(official) == 8, f"n={len(official)}")
    add(checks, "forced_targets_retained", {row["target_canonical_id"] for row in official} == {"11:74716876:C:T", "16:53772541:A:G"}, "two exact targets")

    output_qc = read_csv(WORKFLOW / "qc/official_smr_output_checks.csv")
    truth_columns = ["target_matches_requested", "target_position_matches_requested", "at_least_3_heidi_snps"]
    qc_pass = all(row[column].lower() == "true" for row in output_qc for column in truth_columns)
    numeric_pass = all(float(row["abs_b_smr_minus_manual_two_term"]) < 1e-5 and float(row["abs_se_smr_minus_manual_two_term"]) < 1e-5 for row in output_qc)
    add(checks, "official_output_qc", qc_pass and numeric_pass, "target, position, SNP count, and manual two-term agreement")

    master = read_csv(WORKFLOW / "results/genetics_evidence_master.csv")
    add(checks, "one_master_row_per_pair", len(master) == 2 and {row["pair_id"] for row in master} == {"BMP4_CRC", "LEP_BRC"}, f"n={len(master)}")
    status_ok = (
        next(row for row in master if row["pair_id"] == "BMP4_CRC")["heidi_concordant_at_p01"].lower() == "false"
        and next(row for row in master if row["pair_id"] == "LEP_BRC")["heidi_concordant_at_p01"].lower() == "true"
    )
    add(checks, "method_concordance_classification", status_ok, "BMP4 unresolved; LEP compatible at 0.01")

    screen = read_csv(WORKFLOW / "screen/tier1_combined_316_two_term.csv")
    screen_fields = set(screen[0])
    old_fields = {"b_wald_pkg", "se_wald_pkg", "p_wald_pkg", "p_heidi_rho0", "final_hit"}
    add(checks, "screen_rows_and_column_names", len(screen) == 316 and not (screen_fields & old_fields), f"n={len(screen)}; unprefixed legacy fields={sorted(screen_fields & old_fields)}")
    descriptive_pass = {(row["protein"], row["cancer"]) for row in screen if row["passes_two_term_bonferroni_descriptive"].lower() == "true"}
    add(checks, "screen_descriptive_pass_set", descriptive_pass == {("BMP4", "CRC"), ("MSMB", "PRC"), ("LEP", "BRC")}, str(sorted(descriptive_pass)))

    expected_counts = {
        "BMP4_CRC": ("11:74716876:C:T", 32),
        "LEP_BRC": ("16:53772541:A:G", 437),
    }
    count_ok = True
    details = []
    for pair_id, (target, expected_alt_count) in expected_counts.items():
        rows = read_csv(WORKFLOW / f"reference_1000g_eur/{pair_id}/{pair_id}.acount", delimiter="\t")
        row = next(record for record in rows if record["ID"] == target)
        observed = int(row["ALT_CTS"])
        count_ok &= observed == expected_alt_count and int(row["OBS_CT"]) == 1006
        details.append(f"{pair_id}:{observed}/1006")
    add(checks, "lead_allele_counts", count_ok, "; ".join(details))

    banner_files = {
        ROOT / "MANUSCRIPT_NARRATIVE_DRAFT.md": "Status 2026-09-04",
        ROOT / "figurev5/fig_interpretation.md": "WITHDRAWN FROM MANUSCRIPT ASSEMBLY, 2026-09-04",
        ROOT / "wrap_project.md": "official SMR and HEIDI audit patch",
    }
    banners_ok = all(marker in path.read_text(encoding="utf-8") for path, marker in banner_files.items())
    add(checks, "handoff_banners_present", banners_ok, "manuscript narrative, figure guide, and root handoff")

    output = WORKFLOW / "qc/patch_acceptance_checks.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "detail"])
        writer.writeheader()
        writer.writerows(checks)
    failed = [row for row in checks if not row["passed"]]
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed; wrote {output}")
    if failed:
        for row in failed:
            print(f"FAILED: {row['check']}: {row['detail']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
