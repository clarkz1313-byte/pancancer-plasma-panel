"""Source data for the v7 biology figure (Fig 7), genetics part: genome-wide Manhattan inputs for the two GWAS behind
the two loci that pass both the Wald and colocalisation rules (CRC: BMP4; BRC: LEP). Every variant with p < 1e-3 is
kept; 3% of the rest is kept at random (seed 20261016) for the background. Positions are GRCh38 (GWAS Catalog
harmonised file for BRC; the CRC file is already GRCh38 and uses the raw columns, as in B10f).
Outputs: figures/build/fig7/manhattan_CRC.csv, manhattan_BRC.csv
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[3]))
SMR_DATA_ROOT = _Path(_os.environ.get("SMR_DATA_ROOT", PROJECT_ROOT / "smr_data"))
import numpy as np
import pandas as pd

import v7_style as S

G = str(SMR_DATA_ROOT / "cancer_gwas")
BUILD = S.V7 / "figures/build/fig7"
GW = {"CRC": ("GCST90255675.h.tsv.gz", "raw"),
      "BRC": ("29059683-GCST004988-EFO_0000305.h.tsv.gz", "hm")}


def read_gwas(name):
    f, kind = GW[name]
    cols = (["chromosome", "base_pair_location", "p_value", "hm_code"] if kind == "raw"
            else ["hm_chrom", "hm_pos", "p_value"])
    parts = []
    for ch in pd.read_csv(G + f, sep="\t", usecols=cols, dtype=str, chunksize=2_000_000):
        if kind == "raw":
            ch = ch[ch.hm_code.isna() | ch.hm_code.isin(["5", "10", "11"])]
            ch = ch.rename(columns={"chromosome": "chr", "base_pair_location": "bp", "p_value": "p"})
        else:
            ch = ch.rename(columns={"hm_chrom": "chr", "hm_pos": "bp", "p_value": "p"})
        ch = ch[["chr", "bp", "p"]].dropna()
        ch = ch[ch["chr"].isin([str(c) for c in range(1, 23)])]
        parts.append(ch.astype({"chr": int, "bp": int, "p": float}))
    return pd.concat(parts, ignore_index=True)


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261016)
    for name in GW:
        d = read_gwas(name)
        keep = (d.p < 1e-3) | (rng.random(len(d)) < 0.03)
        d[keep].to_csv(BUILD / f"manhattan_{name}.csv", index=False)
        print(name, len(d), "variants;", int(keep.sum()), "kept; min p", d.p.min())


if __name__ == "__main__":
    main()
