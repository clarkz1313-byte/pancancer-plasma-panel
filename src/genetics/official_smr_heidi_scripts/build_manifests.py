#!/usr/bin/env python3
"""Create source, software, and generated-file manifests for this workflow."""

from __future__ import annotations

import csv
import hashlib
import platform
from pathlib import Path


WORKFLOW = Path("E:/Proteomics/revise_plan/smr_coloc/official_smr_heidi")
ROOT = Path("E:/Proteomics")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    provenance = WORKFLOW / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)

    source_specs = [
        (Path("E:/smr_loco/pqtl_ukbpp/BMP4_pqtl.tsv.gz"), "BMP4 pQTL summary", "UKB-PPP", "GRCh38"),
        (Path("E:/smr_loco/pqtl_ukbpp/LEP_pqtl.tsv.gz"), "LEP pQTL summary", "UKB-PPP", "GRCh38"),
        (Path("E:/smr_loco/cancer_gwas/GCST90255675.h.tsv.gz"), "CRC GWAS summary", "GCST90255675", "GRCh38 harmonized"),
        (Path("E:/smr_loco/cancer_gwas/29059683-GCST004988-EFO_0000305.h.tsv.gz"), "BRC GWAS summary", "GCST004988", "GRCh38 harmonized"),
        (Path("E:/Proteomics/revise_plan/smr_coloc/_ARCHIVE_2026-09-04_pre_official_smr/ld_derived_data/integrated_call_samples_v3.20130502.ALL.panel"), "1000 Genomes population panel", "1000G", "not applicable"),
        (Path("E:/Proteomics/revise_plan/smr_coloc/_ARCHIVE_2026-09-04_pre_official_smr/ld_derived_data/chr11_grch38.vcf.gz.tbi"), "chromosome 11 source index", "1000G NYGC 30x", "GRCh38"),
        (Path("E:/Proteomics/revise_plan/smr_coloc/_ARCHIVE_2026-09-04_pre_official_smr/ld_derived_data/chr16_grch38.vcf.gz.tbi"), "chromosome 16 source index", "1000G NYGC 30x", "GRCh38"),
    ]
    source_rows = []
    for path, role, accession, build in source_specs:
        source_rows.append({
            "path": str(path),
            "role": role,
            "accession_or_release": accession,
            "genome_build": build,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    write_csv(provenance / "source_data_manifest.csv", source_rows)

    smr = WORKFLOW / "tools/smr-1.3.1-win-x86_64/smr-1.3.1-win.exe"
    plink = WORKFLOW / "tools/plink2_20260818_alpha7/plink2.exe"
    software_rows = [
        {"software": "SMR", "version": "1.3.1", "path_or_source": str(smr), "sha256_or_commit": sha256(smr)},
        {"software": "PLINK 2", "version": "2.0.0-a.7.4 AVX2, 2026-08-18", "path_or_source": str(plink), "sha256_or_commit": sha256(plink)},
        {"software": "Python", "version": platform.python_version(), "path_or_source": platform.python_implementation(), "sha256_or_commit": "not applicable"},
        {"software": "R", "version": "4.4.2", "path_or_source": "C:/Program Files/R/R-4.4.2/bin/Rscript.exe", "sha256_or_commit": "not recorded"},
        {"software": "R coloc", "version": "5.2.3", "path_or_source": "revise_plan/smr_coloc/r_libs", "sha256_or_commit": "not applicable"},
        {"software": "GCTB source", "version": "checked 2026-09-04", "path_or_source": "https://github.com/jianzeng/GCTB", "sha256_or_commit": "cc7fa7d765c83a89c6375946cf77fe50ba1a317e"},
        {"software": "SMR source", "version": "checked 2026-09-04", "path_or_source": "https://github.com/jianyangqt/SMR", "sha256_or_commit": "8ab72a9ebfa5ab3bb27a5d667bbae1e797eefdda"},
    ]
    write_csv(provenance / "software_manifest.csv", software_rows)

    patched_documents = [
        "MANUSCRIPT_NARRATIVE_DRAFT.md",
        "wrap_project.md",
        "SMR_COLOC_BMP4_HANDOFF_2026-08-18.md",
        "figurev5/fig_interpretation.md",
        "figurev5/fig_v5_changelog.md",
        "figurev5/GENETICS_PIPELINE_DIAGNOSIS_AND_PLAN_2026-09-03.md",
        "figurev5/GENETICS_STORY_PLAIN_2026-09-03.md",
        "figurev5/REAL_LD_VALIDATION_2026-09-03.md",
        "figurev5/PARTA_COLOC_RESULT_2026-09-03.md",
        "figurev5/OVERLEAF2_FIX_LIST_2026-09-03.md",
        "figurev5/BMP4_STORY_STATE_2026-09-01.md",
        "figurev5/SLIDE_13_REBUILD_2026-08-31.md",
        "revise_plan/smr_coloc/SMR_COLOC_BMP4_AUDIT_2026-08-18.md",
        "revise_plan/smr_coloc/COLLAB_README.md",
        "revise_plan/smr_coloc/26_jun_smr_coloc.md",
        "revise_plan/smr_coloc/28_jun_11cancers_smr_coloc.md",
        "revise_plan/smr_coloc/28_jun_easier_to_read_11cancers_smr_coloc.md",
        "revise_plan/smr_coloc/29_jun_essential_figs_plan.md",
        "revise_plan/smr_coloc/CLAUDE_NOTICE_11cancers_update.md",
    ]
    document_rows = []
    for relative_path in patched_documents:
        path = ROOT / relative_path
        document_rows.append({
            "relative_path": relative_path,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    write_csv(provenance / "patched_document_sha256.csv", document_rows)

    include_dirs = [
        "fixture", "gwas_ma", "logs", "pqtl_besd", "pqtl_esd", "qc",
        "query", "reference_1000g_eur", "reference_vcf", "results", "screen",
        "scripts", "target_pairs", "larger_ld", "tools",
    ]
    generated_rows = []
    top_level_files = [
        "config.json",
        "README.md",
        "IMPLEMENTATION_REPORT_2026-09-04.md",
        "PATCH_EXECUTION_LOG_2026-09-04.md",
    ]
    for filename in top_level_files:
        path = WORKFLOW / filename
        generated_rows.append({
            "relative_path": path.relative_to(WORKFLOW).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    for dirname in include_dirs:
        for path in sorted((WORKFLOW / dirname).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            generated_rows.append({
                "relative_path": path.relative_to(WORKFLOW).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    write_csv(provenance / "generated_file_sha256.csv", generated_rows)
    print(
        f"Recorded {len(source_rows)} sources, {len(software_rows)} tools, "
        f"{len(document_rows)} patched documents, and {len(generated_rows)} generated files"
    )


if __name__ == "__main__":
    main()
