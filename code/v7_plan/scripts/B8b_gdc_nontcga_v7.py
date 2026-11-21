"""v7 B8b: mixed-cancer Addendum B (results/B8_mixed_cancer/PROTOCOL_ADDENDUM_B.md), TCGA-independent GDC series.
D6 = CPTAC-2 + CPTAC-3 (6 cancers); D7 = D6 + BEATAML, NCICCR-DLBCL, MMRF, CGCI-HTMCP-CC, CMI-MPC (11 cancers).
Final v7 panels; analysis by revise_plan script 85 (unchanged), gene-labelled Xena fetching (script 89 fetch).
D8 = D7 + CLL-map CLL (Addendum C): raw TPM (RNA-SeQC, GENCODE19), log2(TPM + 1), 603 treatment-naive samples.
Usage: python B8b_gdc_nontcga_v7.py D6 D7 D8
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import collections
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xenaPython as xena

warnings.filterwarnings("ignore")
ROOT = PROJECT_ROOT
V7 = ROOT / "v7_plan"
OUT = V7 / "results/B8_mixed_cancer"
RAW = OUT / "raw"
HUB = "https://gdc.xenahubs.net"
DEVONLY = ROOT / "revise_plan/pilot_ds_dev_only_size/panel_memberships_devonly.csv"


def load(n, f):
    s = importlib.util.spec_from_file_location(n, ROOT / "revise_plan/scripts" / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


d89 = load("d89", "89_oct_v65_mixed_D4_D5_D2c_2026-10-16.py")
core = d89.core
core.PANELS, core.OUT = DEVONLY, OUT
d89.ALIASES.update({"ELOA": "TCEB3"})

# panels used by the core must equal the frozen v7 panels
_v7 = pd.read_csv(V7 / "panels/panel_memberships_v7.csv")
_mc, _ds = core.load_panels()
assert set(_mc["26"]) == set(_v7[_v7.panel == "multiclass_26"].protein)
for _c, _g in _ds.items():
    assert set(_g) == set(_v7[_v7.panel == f"single_{_c}"].protein), _c

SOLID = {"Primary Tumor", "Tumor"}
BLOOD = ["Primary Blood Derived Cancer - Bone Marrow", "Primary Blood Derived Cancer - Peripheral Blood"]


def lab_cptac2(dx, site):
    if "Breast" in site:
        return "BRC"
    if site.startswith(("Colon", "Rectum")) and "Adenocarcinoma" in dx:
        return "CRC"
    if "Serous" in dx and site.startswith(("Ovary", "Fallopian", "Peritoneum")):
        return "OVC"


def lab_cptac3(dx, site):
    if dx.startswith("Endometrioid adenocarcinoma") and site.startswith("Uterus"):
        return "ENDC"
    if dx == "Glioblastoma" and site.startswith("Brain"):
        return "GLIOM"
    if site.startswith("Lung") and (dx.startswith("Adenocarcinoma") or dx.startswith("Squamous cell carcinoma")):
        return "LUNGC"


PROJ = {
    "CPTAC-2": (lab_cptac2, SOLID),
    "CPTAC-3": (lab_cptac3, SOLID),
    "BEATAML1.0-COHORT": (lambda dx, s: "AML" if ("myeloid leukemia" in dx.lower() or "therapy related myeloid" in dx.lower()) else None, BLOOD),
    "NCICCR-DLBCL": (lambda dx, s: "DLBCL", SOLID),
    "MMRF-COMMPASS": (lambda dx, s: "MYEL" if dx == "Multiple myeloma" else None, BLOOD),
    "CGCI-HTMCP-CC": (lambda dx, s: "CVX", SOLID),
    "CMI-MPC": (lambda dx, s: "PRC", SOLID),
}
SETS = {"D6": ["CPTAC-2", "CPTAC-3"], "D7": list(PROJ), "D8": list(PROJ) + ["CLLMAP"]}
CLL_DROP = {"CRC-0007", "CRC-0011", "CRC-0028", "CRC-0033", "DFCI-5053", "JB-0010", "GCLL-0136"}


def get_cllmap(genes):
    d = RAW / "cllmap"
    naive = pd.read_csv(d / "cllmap_rnaseq_tpms_603samps_deseqLog10_batchcorr.tsv.gz", sep="	", nrows=0).columns[2:]
    keep = [s for s in naive if s not in CLL_DROP]
    t = pd.read_csv(d / "cllmap_rnaseq_tpms_full.tsv.gz", sep="	", usecols=["Name", "Description", *keep])
    names = {g: g for g in genes} | {v: k for k, v in d89.ALIASES.items() if k in genes}
    t = t[t.Description.isin(names)]
    t = t.assign(mu=t[keep].mean(1)).sort_values("mu", ascending=False).drop_duplicates("Description")
    X = np.log2(t.set_index(t.Description.map(names))[keep].T + 1).reindex(columns=genes)
    X.index = [f"CLLMAP|{s}" for s in X.index]
    return X, pd.Series("CLL", index=X.index)


def clinical(project):
    cl = f"{project}.clinical.tsv"
    expr = set(xena.dataset_samples(HUB, f"{project}.star_tpm.tsv", None))
    ss = sorted(s for s in xena.dataset_samples(HUB, cl, None) if s in expr)
    f = ["case_id", "primary_diagnosis.diagnoses", "tissue_or_organ_of_origin.diagnoses", "sample_type.samples"]
    return d89.decode(HUB, cl, ss, f)


def labels(project):
    fn, types = PROJ[project]
    cl = clinical(project)
    cand = {}
    for s, r in cl.iterrows():
        st = r["sample_type.samples"] or ""
        if st not in types:
            continue
        c = fn(str(r["primary_diagnosis.diagnoses"] or ""), str(r["tissue_or_organ_of_origin.diagnoses"] or ""))
        if c:
            # rank by preference of sample type (marrow before blood), then sample identifier
            cand.setdefault(r["case_id"], []).append((list(types).index(st) if isinstance(types, list) else 0, s, c))
    return {min(v)[1]: min(v)[2] for v in cand.values()}


def get(project, genes):
    cache = RAW / f"{project}_panel_genes.csv"
    lab = labels(project)
    if cache.exists():
        X = pd.read_csv(cache, index_col=0)
    else:
        RAW.mkdir(parents=True, exist_ok=True)
        X = d89.fetch(HUB, f"{project}.star_tpm.tsv", sorted(lab), genes)
        X.to_csv(cache)
    X.index = [f"{project}|{s}" for s in X.index]
    return X, pd.Series([lab[s.split("|", 1)[1]] for s in X.index], index=X.index)


def main(which):
    genes = core.all_panel_genes()
    parts = [get_cllmap(genes) if p == "CLLMAP" else get(p, genes) for p in SETS[which]]
    X = pd.concat([p[0] for p in parts])[genes]
    y = pd.concat([p[1] for p in parts])
    sanity = {g: {"expected": e, "highest": X.groupby(y)[g].mean().idxmax()}
              for g, e in d89.MARKERS.items() if g in X.columns and X[g].notna().any() and e in set(y)}
    res = core.run_dataset(X, y, f"{which}_gdc_nontcga")
    res["label_counts_all"] = dict(collections.Counter(y))
    res["source_projects"] = {p: dict(collections.Counter(pp[1])) for p, pp in zip(SETS[which], parts)}
    res["marker_sanity"] = sanity
    json.dump(res, open(OUT / f"summary_{which}_gdc_nontcga.json", "w"), indent=1, default=float)
    r = res.get("panel_26", {})
    print(which, json.dumps({"n": res["n_patients"], "per_cancer": res["n_per_cancer"], "coverage": res["coverage"],
                             "missing": res["missing_features"], "sanity": sanity,
                             "step1_acc": r.get("step1_accuracy"), "0.5": r.get("0.5"), "cmp": r.get("comparison")},
                            indent=1, default=float))


if __name__ == "__main__":
    for w in sys.argv[1:]:
        main(w)
