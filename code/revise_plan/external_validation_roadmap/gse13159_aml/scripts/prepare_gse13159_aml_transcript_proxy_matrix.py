#!/usr/bin/env python3
"""Prepare GSE13159 MILE AML transcript-proxy data for locked-panel validation.

Inputs:
- GSE13159_series_matrix.txt.gz from GEO.
- GPL570.annot.gz from GEO platform annotation.

The output is a sample x gene matrix for locked-panel genes and an AML vs
non-leukemia healthy bone marrow metadata table. This is transcript-level proxy
evidence, not protein validation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SERIES = ROOT / "revise_plan/external_validation_roadmap/gse13159_aml/data/source/GSE13159_series_matrix.txt.gz"
DEFAULT_ANNOT = ROOT / "revise_plan/external_validation_roadmap/gse13159_aml/data/source/GPL570.annot.gz"
DEFAULT_PANEL = ROOT / "revise_plan/external_validation_roadmap/tables/pathway_input_part_b_25_panel.csv"
DEFAULT_OUT = ROOT / "revise_plan/external_validation_roadmap/gse13159_aml/data"
DEFAULT_RESULTS = ROOT / "revise_plan/external_validation_roadmap/gse13159_aml/results"


def clean(value: str) -> str:
    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    return value


def load_panel(path: Path) -> list[str]:
    panel = pd.read_csv(path)
    col = "gene_symbol" if "gene_symbol" in panel.columns else "protein"
    return [str(x).strip().upper() for x in panel[col].dropna() if str(x).strip()]


def parse_gpl570_annotation(path: Path, panel_symbols: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing GPL570 annotation: {path}\n"
            "Run:\n"
            "revise_plan/external_validation_roadmap/gse13159_aml/scripts/download_gse13159_aml_inputs.sh"
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
            if not in_platform_table or line.startswith("#") or not line.strip():
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


def parse_characteristics(characteristics_rows: list[list[str]], n_samples: int) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = defaultdict(lambda: [""] * n_samples)
    for row in characteristics_rows:
        if len(row) != n_samples:
            continue
        for idx, value in enumerate(row):
            text = clean(value)
            if ":" not in text:
                continue
            key, val = text.split(":", 1)
            parsed[key.strip().lower().replace(" ", "_")][idx] = val.strip()
    return dict(parsed)


def parse_series_matrix(path: Path, candidate_probe_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing GSE13159 series matrix: {path}\n"
            "Run:\n"
            "revise_plan/external_validation_roadmap/gse13159_aml/scripts/download_gse13159_aml_inputs.sh"
        )
    gzip_test = subprocess.run(["gzip", "-t", str(path)], check=False, capture_output=True, text=True)
    if gzip_test.returncode != 0:
        raise ValueError(
            f"GSE13159 series matrix is incomplete or not a valid gzip file: {path}\n"
            "Remove it and rerun:\n"
            "rm -f revise_plan/external_validation_roadmap/gse13159_aml/data/source/GSE13159_series_matrix.txt.gz\n"
            "revise_plan/external_validation_roadmap/gse13159_aml/scripts/download_gse13159_aml_inputs.sh"
        )

    meta_rows: dict[str, list[list[str]]] = defaultdict(list)
    selected_rows: list[list[str]] = []
    sample_cols: list[str] | None = None
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
                parts = [clean(x) for x in next(csv.reader([line], delimiter="\t"))]
                if sample_cols is None:
                    if parts[0] != "ID_REF":
                        raise ValueError(f"Unexpected expression header first column: {parts[0]}")
                    sample_cols = parts[1:]
                    continue
                if parts and parts[0] in candidate_probe_ids:
                    selected_rows.append(parts)
            elif line.startswith("!Sample_geo_accession") or line.startswith("!Sample_title") or line.startswith("!Sample_source_name_ch1") or line.startswith("!Sample_characteristics_ch1"):
                parts = next(csv.reader([line], delimiter="\t"))
                meta_rows[parts[0]].append([clean(x) for x in parts[1:]])

    if sample_cols is None:
        raise ValueError(f"No expression table found in {path}")
    if not selected_rows:
        raise ValueError("No locked-panel candidate probe rows were found in GSE13159")

    sample_ids = meta_rows["!Sample_geo_accession"][0]
    if sample_cols != sample_ids:
        raise ValueError("Expression columns do not match GEO sample accessions")
    characteristics = parse_characteristics(meta_rows.get("!Sample_characteristics_ch1", []), len(sample_ids))
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "sample_title": meta_rows.get("!Sample_title", [sample_ids])[0],
            "source_name": meta_rows.get("!Sample_source_name_ch1", [[""] * len(sample_ids)])[0],
            "sample_type": characteristics.get("sample_type", [""] * len(sample_ids)),
            "leukemia_class": characteristics.get("leukemia_class", [""] * len(sample_ids)),
        }
    )
    metadata["leukemia_class_clean"] = metadata["leukemia_class"].str.strip().str.lower()

    expression = pd.DataFrame(selected_rows, columns=["probe_id"] + sample_cols)
    return expression, metadata


def make_aml_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    cls = metadata["leukemia_class_clean"].fillna("")
    is_aml = cls.str.startswith("aml") | cls.str.contains("acute myeloid leukemia", regex=False)
    is_control = cls.str.contains("non-leukemia", regex=False) & cls.str.contains("healthy", regex=False)
    kept = metadata.loc[is_aml | is_control].copy()
    kept["label"] = np.where(is_aml.loc[kept.index], 1, 0)
    kept["label_name"] = np.where(kept["label"].eq(1), "aml", "non_leukemia_healthy_bone_marrow")
    if kept["label"].nunique() != 2:
        counts = metadata["leukemia_class"].value_counts(dropna=False).head(30).to_string()
        raise ValueError(f"Could not infer both AML and healthy controls from leukemia_class. Top classes:\n{counts}")
    return kept


def build_gene_matrix(expression: pd.DataFrame, annot: pd.DataFrame, panel: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return matrix, pd.DataFrame(source_rows)


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
    annot = parse_gpl570_annotation(args.annotation, set(panel))
    if annot.empty:
        raise ValueError("GPL570 annotation did not map any locked-panel genes")

    expression, metadata = parse_series_matrix(args.series, set(annot["probe_id"]))
    aml_metadata = make_aml_metadata(metadata)
    matrix, source = build_gene_matrix(expression, annot, panel)

    metadata.to_csv(args.out_dir / "sample_metadata_all_samples.csv", index=False)
    aml_metadata.to_csv(args.out_dir / "sample_metadata_aml_vs_control.csv", index=False)
    matrix.to_csv(args.out_dir / "processed_matrix.csv", index=False)
    source.to_csv(args.out_dir / "probe_mapping_selected.csv", index=False)

    audit = []
    source_records = source.to_dict("records")
    for gene in panel:
        hit = next((row for row in source_records if row["gene_symbol"] == gene), None)
        audit.append(
            {
                "panel_symbol": gene,
                "mapping_status": "present" if hit else "missing",
                "selected_probe_id": hit["selected_probe_id"] if hit else "",
                "transcript_proxy_only": True,
            }
        )
    pd.DataFrame(audit).to_csv(args.results_dir / "locked25_transcript_probe_coverage_audit.csv", index=False)

    print(f"Wrote {args.out_dir / 'processed_matrix.csv'} with shape {matrix.shape}")
    print("AML-vs-control cohort:")
    print(aml_metadata["label_name"].value_counts(dropna=False).to_string())
    print("Top leukemia classes:")
    print(metadata["leukemia_class"].value_counts(dropna=False).head(20).to_string())
    print(pd.DataFrame(audit).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
