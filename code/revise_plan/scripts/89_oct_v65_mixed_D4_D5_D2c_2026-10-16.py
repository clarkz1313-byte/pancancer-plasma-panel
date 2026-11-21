"""Mixed-cancer pilot, addendum A (PROTOCOL_2026-10-15.md): D4 PCAWG RNA-seq, D5 CCLE cell lines, D2c TCGA Toil.
Gene-labelled Xena fetching only (dataset_gene_probe_avg), with older-symbol fallback; marker sanity check
written next to each summary. Analysis by script 85.
Usage: python 89_... D4|D5|D2c
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import collections
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xenaPython as xena

ROOT = PROJECT_ROOT
OUT = ROOT / "revise_plan/external_mixed_cancer_pilot"
RAW = OUT / "raw"
spec = importlib.util.spec_from_file_location("core", ROOT / "revise_plan/scripts/85_oct_v65_mixed_cancer_core_2026-10-15.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
ALIASES = {"CXCL8": "IL8", "CDHR2": "PCDH24"}
MARKERS = {"MSMB": "PRC", "GFAP": "GLIOM", "FLT3": "AML", "SLAMF7": "MYEL", "CEACAM5": "CRC", "SFTPD": "LUNGC",
           "FCER2": "CLL", "TCL1A": "CLL", "LGALS4": "CRC"}


def fetch(hub, ds, samples, genes):
    frames = []
    for i in range(0, len(samples), 300):
        chunk = samples[i:i + 300]
        cols = {}
        for r in xena.dataset_gene_probe_avg(hub, ds, chunk, genes):
            sc = r.get("scores") or [[]]
            if sc and len(sc[0]) == len(chunk):
                cols[r["gene"]] = [np.nan if v in ("NaN", None) else float(v) for v in sc[0]]
        missing = [g for g in genes if g not in cols and g in ALIASES]
        if missing:
            for r in xena.dataset_gene_probe_avg(hub, ds, chunk, [ALIASES[g] for g in missing]):
                sc = r.get("scores") or [[]]
                back = {v: k for k, v in ALIASES.items()}[r["gene"]]
                if sc and len(sc[0]) == len(chunk):
                    cols[back] = [np.nan if v in ("NaN", None) else float(v) for v in sc[0]]
        frames.append(pd.DataFrame({g: cols.get(g, [np.nan] * len(chunk)) for g in genes}, index=chunk))
    return pd.concat(frames)


def decode(hub, ds, samples, fields):
    vals = xena.dataset_fetch(hub, ds, samples, fields)
    codes = {c["name"]: c["code"].split("\t") for c in xena.field_codes(hub, ds, fields) if c.get("code")}
    out = {}
    for f, v in zip(fields, vals):
        dec = []
        for x in v:
            try:
                dec.append(codes[f][int(x)] if f in codes else x)
            except (ValueError, TypeError, IndexError):
                dec.append(None)
        out[f] = dec
    return pd.DataFrame(out, index=samples)


def labels_D4():
    hub, hd = "https://pcawg.xenahubs.net", "pcawg_specimen_histology_August2016_v9_donor"
    h = decode(hub, hd, xena.dataset_samples(hub, hd, None), ["histology_abbreviation", "histology_tier4"])
    m = {"Breast-AdenoCa": "BRC", "Breast-LobularCA": "BRC", "ColoRect-AdenoCA": "CRC", "Cervix-SCC": "CVX",
         "Cervix-AdenoCA": "CVX", "Uterus-AdenoCA": "ENDC", "CNS-GBM": "GLIOM", "CNS-Oligo": "GLIOM",
         "Lung-AdenoCA": "LUNGC", "Lung-SCC": "LUNGC", "Ovary-AdenoCA": "OVC", "Prost-AdenoCA": "PRC",
         "Myeloid-AML": "AML", "Lymph-CLL": "CLL"}
    lab = {}
    for d, r in h.iterrows():
        a = r["histology_abbreviation"]
        if a in m:
            lab[d] = m[a]
        elif a == "Lymph-BNHL" and (r["histology_tier4"] or "").startswith("Diffse large B-cell"):
            lab[d] = "DLBCL"
    rna = set(xena.dataset_samples(hub, "tophat_star_fpkm_uq.v2_aliquot_gl.donor.log", None))
    lab = {d: c for d, c in lab.items() if d in rna}  # donors with RNA-seq only
    return hub, "tophat_star_fpkm_uq.v2_aliquot_gl.donor.log", lab, False


def labels_D5():
    hub, ds = "https://ucscpublic.xenahubs.net", "ccle/CCLE_DepMap_18Q2_RNAseq_RPKM_20180502"
    info = "ccle/CCLE_sample_info_file_2012-10-18.txt"
    hi = decode(hub, info, xena.dataset_samples(hub, info, None), ["Histology", "Hist Subtype1"])
    site_map = {"BREAST": "BRC", "LARGE_INTESTINE": "CRC", "CERVIX": "CVX", "ENDOMETRIUM": "ENDC", "LUNG": "LUNGC",
                "OVARY": "OVC", "PROSTATE": "PRC"}
    lab = {}
    for s in xena.dataset_samples(hub, ds, None):
        site = s.split("_", 1)[1] if "_" in s else ""
        hist = (hi.loc[s, "Histology"] if s in hi.index else None) or ""
        sub = (hi.loc[s, "Hist Subtype1"] if s in hi.index else None) or ""
        if site in site_map:
            lab[s] = site_map[site]
        elif site == "CENTRAL_NERVOUS_SYSTEM" and "glioma" in hist.lower():
            lab[s] = "GLIOM"
        elif site == "HAEMATOPOIETIC_AND_LYMPHOID_TISSUE":
            sl = sub.lower()
            if "acute_myeloid" in sl:
                lab[s] = "AML"
            elif "chronic_lymphocytic" in sl:
                lab[s] = "CLL"
            elif "plasma_cell_myeloma" in sl:
                lab[s] = "MYEL"
            elif "diffuse_large_b_cell" in sl:
                lab[s] = "DLBCL"
    return hub, ds, lab, True  # RPKM: log2(x + 1)


def labels_D2c():
    hub, ds = "https://toil.xenahubs.net", "TcgaTargetGtex_rsem_gene_tpm"
    ph = "TcgaTargetGTEX_phenotype.txt"
    samples = [s for s in xena.dataset_samples(hub, ds, None) if s.startswith("TCGA-")]
    p = decode(hub, ph, samples, ["primary disease or tissue", "_sample_type"]).rename(
        columns={"primary disease or tissue": "_primary_disease"})
    dis = {"breast invasive carcinoma": "BRC", "colon adenocarcinoma": "CRC", "rectum adenocarcinoma": "CRC",
           "cervical & endocervical cancer": "CVX", "diffuse large b-cell lymphoma": "DLBCL",
           "uterine corpus endometrioid carcinoma": "ENDC", "glioblastoma multiforme": "GLIOM",
           "brain lower grade glioma": "GLIOM", "lung adenocarcinoma": "LUNGC", "lung squamous cell carcinoma": "LUNGC",
           "ovarian serous cystadenocarcinoma": "OVC", "prostate adenocarcinoma": "PRC", "acute myeloid leukemia": "AML"}
    lab, seen = {}, set()
    for s, r in p.sort_index().iterrows():
        c = dis.get((r["_primary_disease"] or "").lower())
        code = s[13:15]
        if c and code == ("03" if c == "AML" else "01") and s[:12] not in seen:
            lab[s] = c
            seen.add(s[:12])
    return hub, ds, lab, False


def main(which):
    hub, ds, lab, log1p = {"D4": labels_D4, "D5": labels_D5, "D2c": labels_D2c}[which]()
    cache = RAW / f"{which}_panel_genes.csv"
    genes = core.all_panel_genes()
    if cache.exists():
        X = pd.read_csv(cache, index_col=0)
    else:
        X = fetch(hub, ds, sorted(lab), genes)
        X.to_csv(cache)
    if log1p:
        X = np.log2(X.clip(lower=0) + 1)
    y = pd.Series([lab[s] for s in X.index], index=X.index)
    sanity = {}
    for g, expect in MARKERS.items():
        if g in X.columns and X[g].notna().any() and expect in set(y):
            sanity[g] = {"expected": expect, "highest": X.groupby(y)[g].mean().idxmax()}
    res = core.run_dataset(X, y, which)
    res["label_counts_all"] = dict(collections.Counter(y))
    res["marker_sanity"] = sanity
    json.dump(res, open(OUT / f"summary_{which}.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k in ("n_patients", "n_per_cancer", "coverage", "missing_features", "marker_sanity")}, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
