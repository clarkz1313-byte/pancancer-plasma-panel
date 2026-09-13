#!/usr/bin/env python3
"""Create strict ESD and GWAS MA files for the two promoted loci."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


PALINDROMIC = {frozenset(("A", "T")), frozenset(("C", "G"))}


def canonical_id(chrom: int, pos: int, ref: str, alt: str) -> str:
    return f"{chrom}:{pos}:{ref}:{alt}"


def load_reference(bim: Path) -> tuple[dict[int, list[dict]], set[str]]:
    by_pos = defaultdict(list)
    ids = set()
    with bim.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) == 1:
                row = row[0].split()
            chrom, snp, _, pos, allele1, allele2 = row[:6]
            ref, alt = snp.split(":")[-2:]
            item = {
                "chrom": int(chrom),
                "pos": int(pos),
                "snp": snp,
                "ref": ref.upper(),
                "alt": alt.upper(),
                "plink_a1": allele1.upper(),
                "plink_a2": allele2.upper(),
            }
            by_pos[int(pos)].append(item)
            ids.add(snp)
    return by_pos, ids


def choose_reference(by_pos: dict[int, list[dict]], pos: int, a1: str, a2: str):
    allele_set = {a1.upper(), a2.upper()}
    matches = [row for row in by_pos.get(pos, []) if {row["ref"], row["alt"]} == allele_set]
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, "allele_set_absent_from_reference"
    return None, "multiple_reference_variants_same_position_and_alleles"


def parse_pqtl(pair: dict, by_pos: dict[int, list[dict]], out_dir: Path):
    accepted = []
    rejected = []
    n_values = Counter()
    with gzip.open(pair["pqtl_source"], "rt", encoding="utf-8") as handle:
        header = handle.readline().split()
        for line in handle:
            fields = line.split()
            if len(fields) != len(header):
                continue
            row = dict(zip(header, fields))
            if row["CHROM"] != str(pair["target_chr"]):
                continue
            pos = int(row["GENPOS"])
            if pos < pair["region_start"] or pos > pair["region_end"]:
                continue
            a1, a2 = row["ALLELE1"].upper(), row["ALLELE0"].upper()
            reference, reason = choose_reference(by_pos, pos, a1, a2)
            if reference is None:
                rejected.append(("pQTL", pos, row["ID"], a1, a2, reason))
                continue
            try:
                beta = float(row["BETA"])
                se = float(row["SE"])
                freq = float(row["A1FREQ"])
                n = int(float(row["N"]))
                p = 10.0 ** (-float(row["LOG10P"]))
            except (ValueError, OverflowError):
                rejected.append(("pQTL", pos, row["ID"], a1, a2, "invalid_numeric_field"))
                continue
            if not all(map(math.isfinite, (beta, se, freq))) or se <= 0 or not 0 <= freq <= 1:
                rejected.append(("pQTL", pos, row["ID"], a1, a2, "invalid_numeric_range"))
                continue
            n_values[n] += 1
            accepted.append(
                {
                    "Chr": pair["target_chr"],
                    "SNP": reference["snp"],
                    "Bp": pos,
                    "A1": a1,
                    "A2": a2,
                    "Freq": freq,
                    "Beta": beta,
                    "se": se,
                    "p": max(p, 1e-300),
                    "source_id": row["ID"],
                    "source_n": n,
                }
            )
    if not accepted:
        raise RuntimeError(f"No pQTL rows accepted for {pair['pair_id']}")
    if set(n_values) != {pair["pqtl_n"]}:
        raise RuntimeError(f"Unexpected pQTL N values for {pair['pair_id']}: {dict(n_values)}")
    accepted.sort(key=lambda row: (row["Bp"], row["SNP"]))
    esd = out_dir / "pqtl_esd" / f"{pair['protein']}.esd"
    esd.parent.mkdir(parents=True, exist_ok=True)
    with esd.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Chr", "SNP", "Bp", "A1", "A2", "Freq", "Beta", "se", "p"], delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(accepted)
    source_rows = out_dir / "qc" / f"{pair['pair_id']}_pqtl_accepted.csv"
    source_rows.parent.mkdir(parents=True, exist_ok=True)
    with source_rows.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(accepted[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(accepted)
    return accepted, rejected, esd


def parse_gwas(pair: dict, by_pos: dict[int, list[dict]], out_dir: Path):
    accepted = []
    rejected = []
    with gzip.open(pair["gwas_source"], "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if pair["gwas_parser"] == "raw":
                chrom = row["chromosome"].replace("chr", "")
                pos_text = row["base_pair_location"]
                a1, a2 = row["effect_allele"], row["other_allele"]
                beta, se, p = row["beta"], row["standard_error"], row["p_value"]
                freq = row.get("effect_allele_frequency", "NA")
                source_id = row.get("rsid") or row.get("variant_id")
                hm_code = row.get("hm_code", "")
                if hm_code not in ("", "NA", "5", "10", "11"):
                    continue
            else:
                chrom = row["hm_chrom"]
                pos_text = row["hm_pos"]
                a1, a2 = row["hm_effect_allele"], row["hm_other_allele"]
                beta, se, p = row["hm_beta"], row["standard_error"], row["p_value"]
                freq = row.get("hm_effect_allele_frequency", "NA")
                source_id = row.get("hm_rsid") or row.get("variant_id")
            if chrom != str(pair["target_chr"]) or not pos_text or pos_text == "NA":
                continue
            pos = int(pos_text)
            if pos < pair["region_start"] or pos > pair["region_end"]:
                continue
            a1, a2 = (a1 or "").upper(), (a2 or "").upper()
            reference, reason = choose_reference(by_pos, pos, a1, a2)
            if reference is None:
                rejected.append(("GWAS", pos, source_id, a1, a2, reason))
                continue
            try:
                beta_f, se_f, p_f = float(beta), float(se), float(p)
                freq_f = float(freq) if freq not in (None, "", "NA") else None
            except ValueError:
                rejected.append(("GWAS", pos, source_id, a1, a2, "invalid_numeric_field"))
                continue
            if not math.isfinite(beta_f) or not math.isfinite(se_f) or se_f <= 0 or not 0 <= p_f <= 1:
                rejected.append(("GWAS", pos, source_id, a1, a2, "invalid_numeric_range"))
                continue
            if frozenset((a1, a2)) in PALINDROMIC and freq_f is None:
                rejected.append(("GWAS", pos, source_id, a1, a2, "palindromic_without_gwas_frequency"))
                continue
            accepted.append(
                {
                    "SNP": reference["snp"],
                    "A1": a1,
                    "A2": a2,
                    "freq": freq_f if freq_f is not None else "NA",
                    "b": beta_f,
                    "se": se_f,
                    "p": max(p_f, 1e-300),
                    "n": pair["gwas_n"],
                    "source_id": source_id,
                    "source_pos": pos,
                }
            )
    deduplicated = {}
    for row in accepted:
        current = deduplicated.get(row["SNP"])
        if current is None or row["p"] < current["p"]:
            if current is not None:
                rejected.append(("GWAS", row["source_pos"], row["source_id"], row["A1"], row["A2"], "duplicate_canonical_id_higher_p"))
            deduplicated[row["SNP"]] = row
        else:
            rejected.append(("GWAS", row["source_pos"], row["source_id"], row["A1"], row["A2"], "duplicate_canonical_id_higher_p"))
    accepted = sorted(deduplicated.values(), key=lambda row: (row["source_pos"], row["SNP"]))
    if not accepted:
        raise RuntimeError(f"No GWAS rows accepted for {pair['pair_id']}")
    ma = out_dir / "gwas_ma" / f"{pair['cancer']}.ma"
    ma.parent.mkdir(parents=True, exist_ok=True)
    with ma.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["SNP", "A1", "A2", "freq", "b", "se", "p", "n"], delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(accepted)
    source_rows = out_dir / "qc" / f"{pair['pair_id']}_gwas_accepted.csv"
    source_rows.parent.mkdir(parents=True, exist_ok=True)
    with source_rows.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(accepted[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(accepted)
    return accepted, rejected, ma


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    all_rejected = []
    summaries = []
    for pair in config["pairs"]:
        bim = args.workflow_dir / "reference_1000g_eur" / pair["pair_id"] / pair["pair_id"]
        by_pos, _ = load_reference(bim.with_suffix(".bim"))
        pqtl, rejected_pqtl, esd = parse_pqtl(pair, by_pos, args.workflow_dir)
        gwas, rejected_gwas, ma = parse_gwas(pair, by_pos, args.workflow_dir)
        all_rejected.extend((pair["pair_id"],) + row for row in rejected_pqtl + rejected_gwas)

        target = canonical_id(pair["target_chr"], pair["target_bp"],
                              by_pos[pair["target_bp"]][0]["ref"], by_pos[pair["target_bp"]][0]["alt"])
        target_pqtl = [row for row in pqtl if row["SNP"] == target]
        target_gwas = [row for row in gwas if row["SNP"] == target]
        if len(target_pqtl) != 1 or len(target_gwas) != 1:
            raise RuntimeError(f"Target {target} missing or duplicated for {pair['pair_id']}")

        flist = args.workflow_dir / "pqtl_esd" / f"{pair['protein']}.flist"
        with flist.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["Chr", "ProbeID", "GeneticDistance", "ProbeBp", "Gene", "Orientation", "PathOfEsd"])
            writer.writerow([pair["probe_chr"], pair["protein"], 0, pair["probe_bp"], pair["protein"], pair["probe_orientation"], str(esd.resolve()).replace("\\", "/")])
        target_file = args.workflow_dir / "target_pairs" / f"{pair['pair_id']}.txt"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(f"{target}\t{pair['protein']}\n", encoding="utf-8")
        summaries.append({
            "pair_id": pair["pair_id"],
            "target_rsid": pair["target_rsid"],
            "target_canonical_id": target,
            "pqtl_rows": len(pqtl),
            "gwas_rows": len(gwas),
            "pqtl_target_beta": target_pqtl[0]["Beta"],
            "pqtl_target_se": target_pqtl[0]["se"],
            "pqtl_target_freq": target_pqtl[0]["Freq"],
            "pqtl_target_p": target_pqtl[0]["p"],
            "pqtl_n": pair["pqtl_n"],
            "gwas_target_beta": target_gwas[0]["b"],
            "gwas_target_se": target_gwas[0]["se"],
            "gwas_target_freq": target_gwas[0]["freq"],
            "gwas_target_p": target_gwas[0]["p"],
            "gwas_n": pair["gwas_n"],
        })

    reject_path = args.workflow_dir / "qc" / "summary_input_rejections.csv"
    with reject_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_id", "source", "position", "source_id", "a1", "a2", "reason"])
        writer.writerows(all_rejected)
    summary_path = args.workflow_dir / "qc" / "summary_input_checks.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
