#!/usr/bin/env python3
"""Build BESD inputs and run official SMR 1.3.1 for promoted loci."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log_path: Path) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    ended = datetime.now(timezone.utc).isoformat()
    log_path.write_text(
        "COMMAND\n" + subprocess.list2cmdline(command) + "\n\nSTDOUT\n"
        + process.stdout + "\nSTDERR\n" + process.stderr,
        encoding="utf-8",
    )
    record = {
        "command": command,
        "started_utc": started,
        "ended_utc": ended,
        "exit_code": process.returncode,
        "log": str(log_path),
    }
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {log_path}")
    return record


def make_fixture(workflow: Path, smr: Path, pair: dict) -> list[dict]:
    fixture = workflow / "fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    source = workflow / "pqtl_esd" / f"{pair['protein']}.esd"
    target_id = (workflow / "target_pairs" / f"{pair['pair_id']}.txt").read_text().split()[0]
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = rows[:25]
    target = [row for row in rows if row["SNP"] == target_id]
    if len(target) != 1:
        raise RuntimeError("Fixture target was absent or duplicated")
    if target[0] not in selected:
        selected.append(target[0])
    esd = fixture / "fixture.esd"
    with esd.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(selected)
    flist = fixture / "fixture.flist"
    with flist.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["Chr", "ProbeID", "GeneticDistance", "ProbeBp", "Gene", "Orientation", "PathOfEsd"])
        writer.writerow([pair["probe_chr"], "FIXTURE", 0, pair["probe_bp"], "FIXTURE", pair["probe_orientation"], str(esd.resolve()).replace("\\", "/")])
    prefix = fixture / "fixture_besd"
    records = [run([
        str(smr), "--eqtl-flist", str(flist), "--add-n", str(pair["pqtl_n"]),
        "--make-besd-dense", "--out", str(prefix)
    ], workflow / "logs" / "fixture_make_besd.log")]
    records.append(run([
        str(smr), "--beqtl-summary", str(prefix), "--show-n"
    ], workflow / "logs" / "fixture_show_n.log"))
    for suffix in (".besd", ".esi", ".epi"):
        if not prefix.with_suffix(suffix).exists():
            raise RuntimeError(f"Fixture did not create {prefix.with_suffix(suffix)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    workflow = args.workflow_dir.resolve()
    smr = Path(config["smr_executable"])
    if not smr.exists():
        raise FileNotFoundError(smr)

    records = make_fixture(workflow, smr, config["pairs"][0])
    for pair in config["pairs"]:
        flist = workflow / "pqtl_esd" / f"{pair['protein']}.flist"
        besd = workflow / "pqtl_besd" / pair["protein"]
        besd.parent.mkdir(parents=True, exist_ok=True)
        records.append(run([
            str(smr), "--eqtl-flist", str(flist), "--add-n", str(pair["pqtl_n"]),
            "--make-besd-dense", "--out", str(besd)
        ], workflow / "logs" / f"{pair['pair_id']}_make_besd.log"))
        records.append(run([
            str(smr), "--beqtl-summary", str(besd), "--show-n"
        ], workflow / "logs" / f"{pair['pair_id']}_show_n.log"))
        query_out = workflow / "query" / pair["pair_id"]
        query_out.parent.mkdir(parents=True, exist_ok=True)
        records.append(run([
            str(smr), "--beqtl-summary", str(besd), "--query", "1",
            "--probe", pair["protein"], "--out", str(query_out)
        ], workflow / "logs" / f"{pair['pair_id']}_query_besd.log"))

        bfile = workflow / "reference_1000g_eur" / pair["pair_id"] / pair["pair_id"]
        gwas = workflow / "gwas_ma" / f"{pair['cancer']}.ma"
        target = workflow / "target_pairs" / f"{pair['pair_id']}.txt"
        for method in (0, 1):
            for diff_freq in (0.2, 0.1):
                tag = f"{pair['pair_id']}_mtd{method}_df{int(diff_freq * 100):02d}"
                out = workflow / "results" / "official_smr_raw" / tag
                out.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    str(smr),
                    "--bfile", str(bfile),
                    "--gwas-summary", str(gwas),
                    "--beqtl-summary", str(besd),
                    "--extract-target-snp-probe", str(target),
                    "--trans",
                    "--trans-wind", "500",
                    "--heidi-mtd", str(method),
                    "--peqtl-heidi", "1.5654e-3",
                    "--ld-lower-limit", "0.05",
                    "--ld-upper-limit", "0.9",
                    "--heidi-min-m", "3",
                    "--heidi-max-m", "20",
                    "--maf", "0.01",
                    "--diff-freq", str(diff_freq),
                    "--out", str(out),
                ]
                records.append(run(command, workflow / "logs" / f"{tag}.log"))

    provenance = {
        "workflow_version": config["workflow_version"],
        "smr_executable": str(smr),
        "smr_sha256": sha256(smr),
        "commands": records,
    }
    path = workflow / "provenance" / "official_smr_run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {len(records)} official commands; provenance: {path}")


if __name__ == "__main__":
    main()
