#!/usr/bin/env python3
"""Create small auxiliary metadata tables for single-panel validation runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def add_gse19804_patient_groups(metadata: Path, out: Path) -> None:
    meta = pd.read_csv(metadata)
    patient = meta["title"].astype(str).str.extract(r"(\d+)\s*[TN]", flags=re.I)[0]
    meta["patient_id"] = patient.fillna(meta["sample_id"].astype(str))
    out.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out, index=False)
    print(meta[["sample_id", "label", "label_name", "patient_id"]].head().to_string(index=False))


def make_gse13159_cll(metadata_all: Path, out: Path) -> None:
    metadata = pd.read_csv(metadata_all)
    cls = metadata["leukemia_class_clean"].fillna("")
    is_cll = cls.eq("cll")
    is_control = cls.str.contains("non-leukemia", regex=False) & cls.str.contains("healthy", regex=False)
    kept = metadata.loc[is_cll | is_control].copy()
    kept["label"] = np.where(is_cll.loc[kept.index], 1, 0)
    kept["label_name"] = np.where(kept["label"].eq(1), "cll", "non_leukemia_healthy_bone_marrow")
    if kept["label"].nunique() != 2:
        raise SystemExit("Could not infer both CLL and non-leukemia healthy controls")
    out.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out, index=False)
    print(kept["label_name"].value_counts().to_string())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["gse19804_patient_groups", "gse13159_cll"])
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "gse19804_patient_groups":
        add_gse19804_patient_groups(args.metadata, args.out)
    elif args.mode == "gse13159_cll":
        make_gse13159_cll(args.metadata, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
