"""B9f: Olink Explore sub-panel of the panel proteins (answers 'the assay already holds cancer markers').

Explore 1536 is made of four sub-panels (cardiometabolic, inflammation, neurology, oncology). The deposited matrix
gives each of the 1,463 proteins one sub-panel (proteins measured on several appear once). The share of panel
proteins from the oncology sub-panel is compared with its share of the measured proteins (one-sided
hypergeometric test, more oncology proteins than expected).
Output: v7_plan/results/B9_biology/olink_subpanel/B9f_summary.json
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import json
from pathlib import Path

import pandas as pd
from scipy.stats import hypergeom

R = PROJECT_ROOT
OUT = R / "v7_plan/results/B9_biology/olink_subpanel"
SUP = R / "overleaf-source-git/supplement"

raw = pd.read_csv(R / "data/raw/pancancer_olink_data_biostudies_v2.txt", sep=" ", usecols=["Assay", "Panel"])
panel_of = raw.drop_duplicates().drop_duplicates("Assay").set_index("Assay").Panel
mc = set(pd.read_csv(SUP / "Table_S1_multiclass26_panel_membership.csv").protein)
ds = set(pd.read_csv(SUP / "Table_S4_disease_specific_panel_membership.csv").protein)
N, K = len(panel_of), int((panel_of == "Oncology").sum())

res = {"measured": N, "oncology_measured": K, "oncology_share_measured": K / N,
       "measured_by_subpanel": panel_of.value_counts().to_dict()}
for name, s in [("multiclass_26", mc), ("disease_specific_81", ds), ("union_87", mc | ds)]:
    v = panel_of.reindex(sorted(s))
    k = int((v == "Oncology").sum())
    res[name] = {"n": len(s), "oncology": k, "share": k / len(s), "expected": len(s) * K / N,
                 "p_one_sided": float(hypergeom.sf(k - 1, N, K, len(s))), "by_subpanel": v.value_counts().to_dict()}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "B9f_summary.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: res[k] for k in ["oncology_share_measured", "multiclass_26", "union_87"]}, indent=1))
