#!/usr/bin/env python3
"""Prepare a GEO GPL570 series matrix for locked-panel transcript-proxy validation.

This is a generic helper for tumor/control GEO expression arrays. It maps GPL570
probes to the locked panel genes, selects one probe per gene by variance, labels
samples by regex over GEO metadata text, and writes a sample x gene matrix plus
binary metadata.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PANEL = ROOT / "revise_plan/external_validation_roadmap/tables/pathway_input_part_b_25_panel.csv"
DEFAULT_ANNOT = ROOT / "revise_plan/external_validation_roadmap/gse4290/data/source/GPL570.annot.gz"


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
        raise FileNotFoundError(f"Missing GPL570 annotation: {path}")
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
            for chunk in symbol_text.replace("///", ";").replace(",", ";").split(";"):
                symbol = chunk.strip().upper()
                if symbol in panel_symbols:
                    records.append({"probe_id": probe, "gene_symbol": symbol})
    return pd.DataFrame(records).drop_duplicates()


def parse_series_matrix(path: Path, candidate_probe_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
    except (OSError, EOFError) as exc:
        raise ValueError(f"Invalid or incomplete gzip file: {path}") from exc

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
                if line.startswith('"ID_REF"') or line.startswith("ID_REF"):
                    table_lines.append(line)
                    continue
                probe = clean(line.split("\t", 1)[0])
                if probe in candidate_probe_ids:
                    table_lines.append(line)
            elif line.startswith("!Sample_"):
                parts = next(csv.reader([line], delimiter="\t"))
                key = parts[0]
                values = [clean(x) for x in parts[1:]]
                if key in meta_rows:
                    key = f"{key}__{sum(k.startswith(parts[0]) for k in meta_rows)}"
                meta_rows[key] = values

    if not table_lines:
        raise ValueError(f"No selected probe expression table found in {path}")
    expression = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="\t")
    expression.columns = [clean(c) for c in expression.columns]
    expression = expression.rename(columns={"ID_REF": "probe_id"})
    expression["probe_id"] = expression["probe_id"].astype(str)

    sample_cols = [c for c in expression.columns if c != "probe_id"]
    metadata = pd.DataFrame({"sample_id": sample_cols})
    for key, values in meta_rows.items():
        if len(values) == len(sample_cols):
            metadata[key.replace("!", "").replace("Sample_", "").lower()] = values
    text_cols = [c for c in metadata.columns if c != "sample_id"]
    metadata["metadata_text"] = metadata[text_cols].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
    return expression, metadata


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


def label_metadata(
    metadata: pd.DataFrame,
    positive_regex: str,
    negative_regex: str,
    positive_name: str,
    negative_name: str,
    label_columns: list[str] | None = None,
) -> pd.DataFrame:
    pos_re = re.compile(positive_regex, re.IGNORECASE)
    neg_re = re.compile(negative_regex, re.IGNORECASE)
    if label_columns:
        missing_cols = [col for col in label_columns if col not in metadata.columns]
        if missing_cols:
            raise ValueError(f"Requested label columns not found: {missing_cols}")
        text = metadata[label_columns].fillna("").astype(str).agg(" | ".join, axis=1)
    else:
        text = metadata["metadata_text"].fillna("").astype(str)
    is_pos = text.map(lambda x: bool(pos_re.search(x)))
    is_neg = text.map(lambda x: bool(neg_re.search(x)))
    overlap = metadata.loc[is_pos & is_neg, ["sample_id", "metadata_text"]]
    if not overlap.empty:
        raise ValueError(f"Positive/negative label regex overlap for {len(overlap)} samples")
    kept = metadata.loc[is_pos | is_neg].copy()
    kept["label"] = np.where(is_pos.loc[kept.index], 1, 0)
    kept["label_name"] = np.where(kept["label"].eq(1), positive_name, negative_name)
    if kept["label"].nunique() != 2:
        top = metadata["metadata_text"].head(20).to_string()
        raise ValueError(f"Could not infer both classes. First metadata rows:\n{top}")
    return kept


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--positive-regex", required=True)
    parser.add_argument("--negative-regex", required=True)
    parser.add_argument("--positive-name", required=True)
    parser.add_argument("--negative-name", required=True)
    parser.add_argument("--label-columns", default="", help="Comma-separated metadata columns for labeling; defaults to all metadata text.")
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOT)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel(args.panel)
    annot = parse_gpl570_annotation(args.annotation, set(panel))
    expression, metadata = parse_series_matrix(args.series, set(annot["probe_id"]))
    matrix, source = build_gene_matrix(expression, annot, panel)
    label_columns = [col.strip() for col in args.label_columns.split(",") if col.strip()]
    labeled = label_metadata(metadata, args.positive_regex, args.negative_regex, args.positive_name, args.negative_name, label_columns or None)

    matrix.to_csv(args.out_dir / "processed_matrix.csv", index=False)
    metadata.to_csv(args.out_dir / "sample_metadata_all_samples.csv", index=False)
    labeled.to_csv(args.out_dir / "sample_metadata_binary.csv", index=False)
    source.to_csv(args.out_dir / "probe_mapping_selected.csv", index=False)

    source_records = source.to_dict("records")
    audit = []
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
    print(labeled["label_name"].value_counts().to_string())
    print(pd.DataFrame(audit).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
