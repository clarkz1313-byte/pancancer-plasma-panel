"""Mixed-cancer pilot D1: GSE2109 (expO, Affymetrix U133 Plus 2.0). Sample mapping and expression handling as
fixed in the protocol deviations log (revise_plan/external_mixed_cancer_pilot/PROTOCOL_2026-10-15.md).
Analysis by script 85.
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import gzip
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = PROJECT_ROOT
OUT = ROOT / "revise_plan/external_mixed_cancer_pilot"
RAW = OUT / "raw/D1"
spec = importlib.util.spec_from_file_location("core", ROOT / "revise_plan/scripts/85_oct_v65_mixed_cancer_core_2026-10-15.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

EXCLUDE = re.compile(r"sarcoma|leiomyo|granulosa|teratoma|germ cell|sex cord", re.I)


def site_to_cancer(site, histology):
    s, h = site.lower(), histology.lower()
    if s.startswith("breast"):
        return "BRC"
    if s.startswith(("colon", "rectum", "rectosigmoid")):
        return "CRC"
    if s.startswith("cervix"):
        return "CVX"
    if s.startswith(("corpus uteri", "endometrium")):
        return "ENDC"
    if "lung" in s or s.startswith("bronchus"):
        return "LUNGC"
    if s.startswith("ovary"):
        return "OVC"
    if s.startswith("prostate"):
        return "PRC"
    if s.startswith("brain") and re.search(r"glio|astrocyt|oligodend", h):
        return "GLIOM"
    if "large b" in h:
        return "DLBCL"
    return None


def read_series():
    meta = {"title": None, "source": None}
    desc = []
    with gzip.open(RAW / "GSE2109_series_matrix.txt.gz", "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                gsm = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_source_name_ch1"):
                meta["source"] = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_description"):
                desc.append([x.strip('"') for x in line.rstrip("\n").split("\t")[1:]])
            elif line.startswith("!series_matrix_table_begin"):
                break
        primary, hist = [""] * len(gsm), [""] * len(gsm)
        for row in desc:
            for i, v in enumerate(row):
                if v.startswith("Primary Site:") and not primary[i]:
                    primary[i] = v.split(":", 1)[1].strip()
                if v.startswith("Histology:") and not hist[i]:
                    hist[i] = v.split(":", 1)[1].strip()
        probes = probe_map()
        header = next(f).rstrip("\n").split("\t")
        cols = [h.strip('"') for h in header[1:]]
        assert cols == gsm
        data = {}
        for line in f:
            if line.startswith("!series_matrix_table_end"):
                break
            parts = line.rstrip("\n").split("\t")
            pid = parts[0].strip('"')
            if pid in probes:
                data[pid] = [float(x) if x not in ("", "null", "NA") else np.nan for x in parts[1:]]
    expr = pd.DataFrame(data, index=gsm)
    return gsm, meta["source"], primary, hist, expr, probes


def probe_map():
    genes = set(core.all_panel_genes())
    out = {}
    with gzip.open(RAW / "GPL570.annot.gz", "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!platform_table_begin"):
                hdr = next(f).rstrip("\n").split("\t")
                break
        ix = hdr.index("Gene symbol")
        for line in f:
            if line.startswith("!platform_table_end"):
                break
            parts = line.rstrip("\n").split("\t")
            sym = parts[ix].strip()
            if sym in genes:  # single-gene probe sets only
                out[parts[0]] = sym
    return out


def main():
    gsm, source, primary, hist, expr, probes = read_series()
    labels, keep = {}, []
    for i, g in enumerate(gsm):
        if EXCLUDE.search(hist[i] or ""):
            continue
        c = site_to_cancer(primary[i], hist[i])
        if c and site_to_cancer(source[i], hist[i]) == c:
            labels[g] = c
            keep.append(g)
    e = expr.loc[keep]
    if np.nanmax(e.to_numpy()) > 50:  # not yet on a log scale
        e = np.log2(e.clip(lower=1))
    # probe set with the highest mean per gene
    best = {}
    for pid, sym in probes.items():
        if pid in e.columns:
            m = e[pid].mean()
            if sym not in best or m > best[sym][1]:
                best[sym] = (pid, m)
    X = pd.DataFrame({sym: e[pid] for sym, (pid, _) in best.items()})
    y = pd.Series([labels[g] for g in X.index], index=X.index)
    pd.Series(labels).to_csv(OUT / "D1_sample_labels.csv", header=["cancer"])
    res = core.run_dataset(X, y, "D1_gse2109_expo")
    res["arrays_total"] = len(gsm)
    json.dump(res, open(OUT / "summary_D1_gse2109_expo.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
