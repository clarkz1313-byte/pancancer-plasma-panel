#!/usr/bin/env python3
"""Prepare GSE4290 GPL570 expression data for locked-panel transcript proxy.

Inputs:
- GSE4290_series_matrix.txt.gz from GEO.
- GPL570.annot.gz from GEO platform annotation.

The output is a sample x gene matrix for the locked panel genes and a metadata
file with binary labels. This is transcript-level proxy evidence, not protein
validation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SERIES = ROOT / "revise_plan/external_validation_roadmap/gse4290/data/source/GSE4290_series_matrix.txt.gz"
DEFAULT_ANNOT = ROOT / "revise_plan/external_validation_roadmap/gse4290/data/source/GPL570.annot.gz"
DEFAULT_PANEL = ROOT / "revise_plan/external_validation_roadmap/tables/pathway_input_part_b_25_panel.csv"
DEFAULT_OUT = ROOT / "revise_plan/external_validation_roadmap/gse4290/data"
DEFAULT_RESULTS = ROOT / "revise_plan/external_validation_roadmap/gse4290/results"


def clean(value: str) -> str:
    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    return value


def parse_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_rows: dict[str, list[str]] = {}
    table_lines: list[str] = []
    in_table = False
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                table_lines.append(line)
            elif line.startswith("!Sample_geo_accession") or line.startswith("!Sample_title") or line.startswith("!Sample_source_name_ch1") or line.startswith("!Sample_characteristics_ch1"):
                parts = next(csv.reader([line], delimiter="\t"))
                meta_rows[parts[0]] = [clean(x) for x in parts[1:]]

    if not table_lines:
        raise ValueError(f"No expression table found in {path}")
    table = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="\t")
    table.columns = [clean(c) for c in table.columns]
    table = table.rename(columns={"ID_REF": "probe_id"})
    table["probe_id"] = table["probe_id"].astype(str)

    sample_ids = meta_rows["!Sample_geo_accession"]
    diagnoses = [x.replace("Histopathological diagnostic:", "").strip().lower() for x in meta_rows["!Sample_characteristics_ch1"]]
    titles = meta_rows.get("!Sample_title", sample_ids)
    sources = meta_rows.get("!Sample_source_name_ch1", [""] * len(sample_ids))
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "sample_title": titles,
            "source_name": sources,
            "diagnosis": diagnoses,
        }
    )
    metadata["diagnosis"] = metadata["diagnosis"].replace("", "missing")
    return table, metadata


def parse_gpl570_annotation(path: Path, panel_symbols: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing GPL570 annotation: {path}\n"
            "Download with:\n"
            "curl -L --fail --retry 3 --retry-delay 2 "
            "-o revise_plan/external_validation_roadmap/gse4290/data/source/GPL570.annot.gz "
            "'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz'"
        )

    header = None
    in_platform_table = False
    records = []
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!platform_table_begin"):
                in_platform_table = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if not in_platform_table:
                continue
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = parts
                continue
            row = dict(zip(header, parts))
            probe = row.get("ID") or row.get("ID_REF")
            symbol_text = row.get("Gene symbol") or row.get("Gene Symbol") or row.get("gene_assignment") or ""
            if not probe or not symbol_text:
                continue
            symbols = []
            for chunk in symbol_text.replace("///", ";").replace(",", ";").split(";"):
                symbol = chunk.strip().upper()
                if symbol in panel_symbols:
                    symbols.append(symbol)
            for symbol in sorted(set(symbols)):
                records.append({"probe_id": probe, "gene_symbol": symbol})
    return pd.DataFrame(records)


def load_panel(path: Path) -> list[str]:
    panel = pd.read_csv(path)
    col = "gene_symbol" if "gene_symbol" in panel.columns else "protein"
    return [str(x).strip().upper() for x in panel[col].dropna()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=Path, default=DEFAULT_SERIES)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOT)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel(args.panel)
    expression, metadata = parse_series_matrix(args.series)
    annot = parse_gpl570_annotation(args.annotation, set(panel))
    if annot.empty:
        raise ValueError("GPL570 annotation did not map any locked-panel genes")

    merged = annot.merge(expression, on="probe_id", how="inner")
    sample_cols = [c for c in expression.columns if c != "probe_id"]
    gene_values = {}
    source_rows = []
    for gene, group in merged.groupby("gene_symbol", sort=False):
        numeric = group[sample_cols].apply(pd.to_numeric, errors="coerce")
        variances = numeric.var(axis=1, skipna=True)
        idx = int(variances.fillna(-1).to_numpy().argmax())
        selected_probe = group.iloc[idx]["probe_id"]
        values = numeric.iloc[idx]
        if values.median(skipna=True) > 100:
            values = np.log2(values + 1.0)
        gene_values[gene] = values.to_numpy(dtype=float)
        source_rows.append({"gene_symbol": gene, "selected_probe_id": selected_probe, "n_candidate_probes": len(group)})

    matrix = pd.DataFrame({"sample_id": sample_cols})
    for gene in panel:
        if gene in gene_values:
            matrix[gene] = gene_values[gene]

    metadata_all = metadata.copy()
    metadata_all["is_missing_diagnosis"] = metadata_all["diagnosis"].eq("missing")
    metadata_all.to_csv(args.out_dir / "sample_metadata_all_samples.csv", index=False)

    glioma_meta = metadata.loc[~metadata["diagnosis"].isin(["missing"])].copy()
    glioma_meta["label"] = np.where(glioma_meta["diagnosis"].eq("non-tumor"), 0, 1)
    glioma_meta["label_name"] = np.where(glioma_meta["label"].eq(1), "glioma_tumor", "non_tumor")
    glioma_meta.to_csv(args.out_dir / "sample_metadata_glioma_vs_nontumor.csv", index=False)

    gbm_meta = metadata.loc[metadata["diagnosis"].isin(["glioblastoma, grade 4", "non-tumor"])].copy()
    gbm_meta["label"] = np.where(gbm_meta["diagnosis"].eq("glioblastoma, grade 4"), 1, 0)
    gbm_meta["label_name"] = np.where(gbm_meta["label"].eq(1), "gbm_tumor", "non_tumor")
    gbm_meta.to_csv(args.out_dir / "sample_metadata_gbm_vs_nontumor.csv", index=False)

    matrix.to_csv(args.out_dir / "processed_matrix.csv", index=False)
    pd.DataFrame(source_rows).to_csv(args.out_dir / "probe_mapping_selected.csv", index=False)

    audit = []
    for gene in panel:
        audit.append(
            {
                "panel_symbol": gene,
                "mapping_status": "present" if gene in gene_values else "missing",
                "selected_probe_id": next((r["selected_probe_id"] for r in source_rows if r["gene_symbol"] == gene), ""),
                "transcript_proxy_only": True,
            }
        )
    pd.DataFrame(audit).to_csv(args.results_dir / "locked25_transcript_probe_coverage_audit.csv", index=False)

    print(f"Wrote {args.out_dir / 'processed_matrix.csv'} with shape {matrix.shape}")
    print(metadata["diagnosis"].value_counts(dropna=False).to_string())
    print(pd.DataFrame(audit).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
