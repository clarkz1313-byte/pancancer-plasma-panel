"""v7 B9e (post hoc sensitivity to B9b, author question 17 Oct): is the tissue-of-origin enrichment of the
disease-specific panels only a consequence of the selection pipeline?

B9b compared each panel with random panels drawn from all measured proteins. Here the random panels are drawn from
the pools the panels were actually selected from, per cancer and with the same sizes (10,000 draws, seed 20261017):
  pool "screen": proteins Bonferroni-significant for that cancer in the development-set one-vs-rest screen;
  pool "lasso":  proteins with a non-zero development-set LASSO coefficient for that cancer (the ranked list the
                 panel size was cut from).
Same HPA 25.1 definitions as B9b. Output: results/B9_biology/tissue_origin/B9e_selection_null.csv, B9e_summary.json
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = PROJECT_ROOT
OUT = ROOT / "v7_plan/results/B9_biology/tissue_origin"
PA = ROOT / "revise_plan/v5_seed52_common_split/part_a_single"
import importlib.util
spec = importlib.util.spec_from_file_location("b9b", ROOT / "v7_plan/scripts/B9b_tissue_of_origin_v7.py")
b9b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b9b)
SLUG = {"DLBCL": "lymph"}


def main():
    hpa = pd.read_csv(zipfile.ZipFile(OUT / "hpa/proteinatlas.tsv.zip").open("proteinatlas.tsv"), sep="\t",
                      usecols=["Gene", "Gene synonym", b9b.T, b9b.BL, b9b.SC]).drop_duplicates("Gene")
    bg = [c for c in pd.read_csv(ROOT / "data/processed/filtered_pancancer_data.csv", nrows=1).columns
          if c not in ("Sample_ID", "Cancer", "protein_count", "Label")]
    syn = {}
    for _, r in hpa.iterrows():
        for s in str(r["Gene synonym"]).split(","):
            syn.setdefault(s.strip(), r.Gene)
    genes = set(hpa.Gene)
    to_hpa = {p: (p if p in genes else syn.get(p)) for p in bg}
    bg = [p for p in bg if to_hpa[p]]
    H = hpa.set_index("Gene")
    elev = {c: {p for p in bg if b9b.labels(H.loc[to_hpa[p], col]) & set(t)} for c, (col, t) in b9b.MAP.items()}
    pan = pd.read_csv(ROOT / "v7_plan/panels/panel_memberships_v7.csv")
    rng = np.random.default_rng(20261017)
    rows, nulls = [], {"all": [], "screen": [], "lasso": []}
    for c in b9b.MAP:
        slug = SLUG.get(c, c.lower())
        P = [p for p in pan[pan.panel == f"single_{c}"].protein if to_hpa.get(p)]
        dea = pd.read_csv(PA / slug / "tables" / f"{slug}_dea.csv")
        rk = pd.read_csv(PA / slug / "tables" / f"{slug}_top_feature_ranking.csv")
        pools = {"all": bg,
                 "screen": [p for p in dea[dea.significant_bonf == True].protein if p in to_hpa and to_hpa[p]],
                 "lasso": [p for p in rk[rk.importance > 0].protein if p in to_hpa and to_hpa[p]]}
        row = {"cancer": c, "panel_size": len(P), "observed": len(set(P) & elev[c]),
               "panel_in_screen_pool": len(set(P) & set(pools["screen"])), "panel_in_lasso_pool": len(set(P) & set(pools["lasso"]))}
        for k, pool in pools.items():
            pool = np.array(pool)
            share = len(set(pool) & elev[c]) / len(pool)
            draws = np.array([len(set(rng.choice(pool, min(len(P), len(pool)), replace=False)) & elev[c]) for _ in range(10000)])
            nulls[k].append(draws)
            row.update({f"{k}_pool_n": len(pool), f"{k}_pool_share_elevated": round(share, 4),
                        f"{k}_expected": round(draws.mean(), 3),
                        f"{k}_p": (1 + (draws >= row["observed"]).sum()) / 10001})
        rows.append(row)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "B9e_selection_null.csv", index=False)
    obs = int(r.observed.sum())
    summ = {"observed_total": obs}
    for k, v in nulls.items():
        tot = np.sum(v, axis=0)
        summ[k] = {"mean": round(float(tot.mean()), 2), "q2.5_q97.5": [float(x) for x in np.percentile(tot, [2.5, 97.5])],
                   "max": int(tot.max()), "p_empirical": (1 + int((tot >= obs).sum())) / 10001}
    json.dump(summ, open(OUT / "B9e_summary.json", "w"), indent=1)
    print(r[["cancer", "panel_size", "observed", "all_expected", "screen_pool_n", "screen_pool_share_elevated",
             "screen_expected", "screen_p", "lasso_pool_n", "lasso_expected", "lasso_p"]].to_string(index=False))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
