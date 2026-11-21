"""v7 B8c: mixed-cancer Addendum D (results/B8_mixed_cancer/PROTOCOL_ADDENDUM_B.md), tumour-educated platelet
RNA, GSE183635 (In 't Veld et al. 2022). D9a = HiSeq 2500 samples (GPL16791) only; D9b = both sequencers.
log2(CPM + 1) over all genes of the GEO count matrix; Ensembl IDs mapped to symbols through the GENCODE 19
annotation in the CLL-map file. Final v7 panels; analysis by revise_plan script 85 (unchanged).
Usage: python B8c_platelet_tep_v7.py D9a D9b
"""
import collections
import gzip
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import B8b_gdc_nontcga_v7 as b  # noqa: E402  (sets core.PANELS to the v7 panels and checks them)

RAW = b.RAW / "gse183635"
GROUPS = {"Non-Small-Cell Lung Cancer": "LUNGC", "Glioma": "GLIOM", "Breast Cancer": "BRC", "Ovarian Cancer": "OVC",
          "Colorectal Cancer": "CRC", "Multiple Myeloma": "MYEL", "Prostate Cancer": "PRC", "Endometrial Cancer": "ENDC"}


def sample_groups(platform):
    title, group = None, None
    with gzip.open(RAW / f"GSE183635-{platform}_series_matrix.txt.gz", "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            v = [x.strip('"') for x in line.rstrip("\n").split("\t")]
            if v[0] == "!Sample_title":
                title = v[1:]
            elif v[0] == "!Sample_characteristics_ch1" and v[1].startswith("patient group:"):
                group = [x.split(": ", 1)[1] for x in v[1:]]
    return dict(zip(title, group))


def main(which):
    genes = b.core.all_panel_genes()
    platforms = ["GPL16791"] if which == "D9a" else ["GPL16791", "GPL20301"]
    grp = {}
    for p in platforms:
        grp.update(sample_groups(p))
    counts = pd.read_csv(RAW / "tep_counts.csv.gz", index_col=0)
    counts.columns = [c.split("-", 1)[1] for c in counts.columns]
    cpm = np.log2(counts / counts.sum(0) * 1e6 + 1)
    ann = pd.read_csv(b.RAW / "cllmap/cllmap_rnaseq_tpms_full.tsv.gz", sep="\t", usecols=["Name", "Description"])
    ens2sym = dict(zip(ann.Name.str.split(".").str[0], ann.Description))
    names = {g: g for g in genes} | {v: k for k, v in b.d89.ALIASES.items() if k in genes}
    cpm = cpm[[ens2sym.get(e) in names for e in cpm.index]]
    cpm.index = [names[ens2sym[e]] for e in cpm.index]
    cpm = cpm.groupby(level=0).max()
    lab = {s: GROUPS[g] for s, g in grp.items() if g in GROUPS and s in cpm.columns}
    X = cpm[sorted(lab)].T.reindex(columns=genes)
    y = pd.Series([lab[s] for s in X.index], index=X.index)
    sanity = {g: {"expected": e, "highest": X.groupby(y)[g].mean().idxmax()}
              for g, e in b.d89.MARKERS.items() if g in X.columns and X[g].notna().any() and e in set(y)}
    res = b.core.run_dataset(X, y, f"{which}_platelet_tep")
    res["label_counts_all"] = dict(collections.Counter(y))
    res["marker_sanity"] = sanity
    json.dump(res, open(b.OUT / f"summary_{which}_platelet_tep.json", "w"), indent=1, default=float)
    r = res.get("panel_26", {})
    print(which, json.dumps({"n": res["n_patients"], "per_cancer": res["n_per_cancer"], "coverage": res["coverage"],
                             "error": res.get("error"), "sanity": sanity, "step1_acc": r.get("step1_accuracy"),
                             "bal": r.get("step1_balanced_accuracy"), "0.5": r.get("0.5"), "cmp": r.get("comparison")},
                            indent=1, default=float))


if __name__ == "__main__":
    for w in sys.argv[1:]:
        main(w)
