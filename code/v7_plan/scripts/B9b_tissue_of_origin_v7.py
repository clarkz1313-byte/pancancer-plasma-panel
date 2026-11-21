"""v7 B9b: tissue/cell-of-origin enrichment of the final panels (protocol results/B9_biology/tissue_origin/
PROTOCOL_B9b_tissue_of_origin.md). HPA 25.1 proteinatlas.tsv; background = the 1,463 measured proteins.
Outputs: results/B9_biology/tissue_origin/B9b_per_cancer.csv, B9b_wrong_tissue.csv, B9b_summary.json
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = PROJECT_ROOT
OUT = ROOT / "v7_plan/results/B9_biology/tissue_origin"
T, BL, SC = "RNA tissue specific nTPM", "RNA blood lineage specific nTPM", "RNA single cell type specific nCPM"
MAP = {"BRC": (T, ["breast"]), "CRC": (T, ["intestine"]), "CVX": (T, ["cervix"]), "ENDC": (T, ["endometrium 1"]),
       "GLIOM": (T, ["brain"]), "LUNGC": (T, ["lung"]), "OVC": (T, ["ovary"]), "PRC": (T, ["prostate"]),
       "AML": (BL, ["monocytes", "granulocytes", "dendritic cells"]), "CLL": (BL, ["B-cells"]),
       "DLBCL": (BL, ["B-cells"]), "MYEL": (SC, ["Plasma cells"])}


def labels(v):
    return set() if pd.isna(v) else {x.split(":")[0].strip() for x in str(v).split(";")}


def main():
    hpa = pd.read_csv(zipfile.ZipFile(OUT / "hpa/proteinatlas.tsv.zip").open("proteinatlas.tsv"), sep="\t",
                      usecols=["Gene", "Gene synonym", T, BL, SC]).drop_duplicates("Gene")
    df = pd.read_csv(ROOT / "data/processed/filtered_pancancer_data.csv", nrows=1)
    bg = [c for c in df.columns if c not in ("Sample_ID", "Cancer", "protein_count", "Label")]
    syn = {}
    for _, r in hpa.iterrows():
        for s in str(r["Gene synonym"]).split(","):
            syn.setdefault(s.strip(), r.Gene)
    genes = set(hpa.Gene)
    to_hpa = {p: (p if p in genes else syn.get(p)) for p in bg}
    unmapped = sorted(p for p, h in to_hpa.items() if h is None)
    bg = [p for p in bg if to_hpa[p]]
    H = hpa.set_index("Gene")
    elev = {c: {p for p in bg if labels(H.loc[to_hpa[p], col]) & set(tis)} for c, (col, tis) in MAP.items()}
    pan = pd.read_csv(ROOT / "v7_plan/panels/panel_memberships_v7.csv")
    ds = {c: [p for p in pan[pan.panel == f"single_{c}"].protein if p in to_hpa and to_hpa[p]] for c in MAP}
    N = len(bg)
    rows = []
    for c, P in ds.items():
        E = elev[c]
        a = len(set(P) & E); b = len(P) - a; cc = len(E) - a; d = N - len(P) - cc
        orr, p = fisher_exact([[a, b], [cc, d]], alternative="greater")
        rows.append({"cancer": c, "panel_size": len(P), "elevated_in_panel": a, "elevated_in_background": len(E),
                     "expected": len(P) * len(E) / N, "odds_ratio": orr, "p": p, "proteins": ";".join(sorted(set(P) & E))})
    r = pd.DataFrame(rows)
    r["q_bh"] = multipletests(r.p, method="fdr_bh")[1]
    r.to_csv(OUT / "B9b_per_cancer.csv", index=False)
    # pooled: random panels of the same sizes
    obs = int(r.elevated_in_panel.sum())
    rng = np.random.default_rng(20261015)
    bga = np.array(bg)
    null = np.array([sum(len(set(rng.choice(bga, len(P), replace=False)) & elev[c]) for c, P in ds.items()) for _ in range(10000)])
    pooled = {"observed": obs, "null_mean": float(null.mean()), "p_empirical": float((1 + (null >= obs).sum()) / 10001)}
    # multiclass 26: elevated in any of the 12 origins
    any_e = set().union(*elev.values())
    mc = [p for p in pan[pan.panel == "multiclass_26"].protein if to_hpa.get(p)]
    a = len(set(mc) & any_e)
    orr, p = fisher_exact([[a, len(mc) - a], [len(any_e) - a, N - len(mc) - len(any_e) + a]], alternative="greater")
    multiclass = {"panel_size": len(mc), "elevated_any_origin": a, "background_elevated": len(any_e),
                  "expected": len(mc) * len(any_e) / N, "odds_ratio": orr, "p": p, "proteins": sorted(set(mc) & any_e)}
    # wrong-tissue control
    wrong = []
    for c, P in ds.items():
        for o in MAP:
            if o == c or (c in ("CLL", "DLBCL") and o in ("CLL", "DLBCL")):
                continue
            E = elev[o]; a = len(set(P) & E)
            wrong.append({"panel": c, "origin_of": o, "elevated_in_panel": a, "expected": len(P) * len(E) / N,
                          "p": fisher_exact([[a, len(P) - a], [len(E) - a, N - len(P) - len(E) + a]], alternative="greater")[1]})
    w = pd.DataFrame(wrong)
    w.to_csv(OUT / "B9b_wrong_tissue.csv", index=False)
    summ = {"hpa_release": "25.1", "background": N, "unmapped": unmapped, "pooled": pooled, "multiclass_26": multiclass,
            "own_origin_hits_total": obs, "wrong_origin_mean_hits_per_panel_origin": float(w.elevated_in_panel.mean()),
            "wrong_origin_nominal_p_lt_0.05": int((w.p < 0.05).sum()), "wrong_origin_tests": len(w)}
    json.dump(summ, open(OUT / "B9b_summary.json", "w"), indent=1, default=float)
    print(r.drop(columns=["proteins"]).round(4).to_string(index=False))
    print(r[["cancer", "proteins"]].to_string(index=False))
    print(json.dumps({k: v for k, v in summ.items() if k != "unmapped"}, indent=1, default=float), "unmapped:", len(unmapped))


if __name__ == "__main__":
    main()
