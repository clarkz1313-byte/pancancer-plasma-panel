"""Mixed-cancer pilot D2: GDC-harmonised STAR TPM (UCSC Xena GDC hub) for the TCGA projects of the study
cancers plus MMRF-CoMMpass (myeloma). Panel genes only; values are log2(TPM + 1) as served by the hub.
Samples: TCGA primary tumour (type 01; 03 for LAML), MMRF baseline bone-marrow CD138+ (visit 1); one per
patient. Analysis by script 85 (protocol in revise_plan/external_mixed_cancer_pilot).
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xenaPython as xena

ROOT = PROJECT_ROOT
OUT = ROOT / "revise_plan/external_mixed_cancer_pilot"
RAW = OUT / "raw"
CACHE = RAW / "D2_gdc_star_tpm_panel_genes.csv"
HUB = "https://gdc.xenahubs.net"
PROJECTS = {"LAML": "AML", "BRCA": "BRC", "COAD": "CRC", "READ": "CRC", "CESC": "CVX", "DLBC": "DLBCL",
            "UCEC": "ENDC", "GBM": "GLIOM", "LGG": "GLIOM", "LUAD": "LUNGC", "LUSC": "LUNGC", "OV": "OVC",
            "PRAD": "PRC"}

spec = importlib.util.spec_from_file_location("core", ROOT / "revise_plan/scripts/85_oct_v65_mixed_cancer_core_2026-10-15.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def fetch(dataset, genes, label):
    samples = xena.dataset_samples(HUB, dataset, None)
    rows = []
    for i in range(0, len(samples), 400):
        chunk = samples[i:i + 400]
        # gene-labelled call: dataset_gene_probes_values drops unknown genes silently, which misaligned the
        # columns in the first run (INVALID_misaligned files); dataset_gene_probe_avg returns one row per gene
        res = xena.dataset_gene_probe_avg(HUB, dataset, chunk, genes)
        cols = {}
        for r in res:
            sc = r.get("scores") or [[]]
            vals = sc[0] if sc and len(sc[0]) == len(chunk) else [np.nan] * len(chunk)
            cols[r["gene"]] = [np.nan if v in ("NaN", None) else float(v) for v in vals]
        df = pd.DataFrame({g: cols.get(g, [np.nan] * len(chunk)) for g in genes}, index=chunk)
        rows.append(df)
    df = pd.concat(rows)
    df.insert(0, "label", label)
    df.insert(0, "dataset", dataset)
    return df


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    genes = core.all_panel_genes()
    if CACHE.exists():
        raw = pd.read_csv(CACHE, index_col=0)
    else:
        frames = [fetch(f"TCGA-{p}.star_tpm.tsv", genes, c) for p, c in PROJECTS.items()]
        frames.append(fetch("MMRF-COMMPASS.star_tpm.tsv", genes, "MYEL"))
        raw = pd.concat(frames)
        raw.to_csv(CACHE)
    keep = []
    for s, row in raw.iterrows():
        if row["dataset"].startswith("TCGA-"):
            code = s[13:15]
            if code == ("03" if row["dataset"].startswith("TCGA-LAML") else "01"):
                keep.append((s[:12], s))
        elif "_1_BM_CD138pos" in s:
            keep.append(("_".join(s.split("_")[:2]), s))
    first = {}
    for pid, s in sorted(keep, key=lambda x: x[1]):
        first.setdefault(pid, s)
    sel = raw.loc[list(first.values())]
    X = sel[genes].copy()
    X.index = list(first.keys())
    y = pd.Series(sel["label"].to_numpy(), index=X.index)
    res = core.run_dataset(X, y, "D2_gdc_tcga_mmrf")
    json.dump(res, open(OUT / "summary_D2_gdc_tcga_mmrf.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
