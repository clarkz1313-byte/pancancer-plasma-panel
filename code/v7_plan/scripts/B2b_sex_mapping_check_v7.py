"""v7 B2b: biological check of the positional sex mapping used by B2 (audit SOL-B15).

The Bueno Alvez supplement (supp2.xlsx) has Cancer, Sex, Age, Stage, Grade but no sample ID; B2 assigns sex by
position within cancer (k-th metadata row = Sample_ID <cancer>_k). If the mapping is right, the plasma proteome
predicts sex well within each mixed-sex cancer; if wrong, prediction is at chance. Per mixed-sex cancer:
5-fold stratified CV (seed 52) of an L2 logistic regression (C = 0.1) on all proteins (median-imputed,
standardised within folds) -> AUC; null = the same with sex shuffled within cancer (20 shuffles). Also per-protein
AUCs of known sex-differing proteins. Output: results/B2_sex/B2b_sex_mapping_check.csv/.json
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = PROJECT_ROOT
OUT = ROOT / "v7_plan/results/B2_sex"
NAME = {"AML": "AML", "BRC": "Breast", "CLL": "CLL", "CRC": "Colorectal", "CVX": "Cervical", "ENDC": "Endometrial",
        "GLIOM": "Glioma", "LUNGC": "Lung", "LYMPH": "DLBCL", "MYEL": "Myeloma", "OVC": "Ovarian", "PRC": "Prostate"}


def main():
    df = pd.read_csv(ROOT / "data/processed/filtered_pancancer_data.csv")
    meta = pd.read_excel(ROOT / "references/alves_supplementary_data/supp2.xlsx").reset_index(drop=True)
    meta["_k"] = meta.groupby("Cancer").cumcount() + 1
    df["_c"] = df.Sample_ID.str.rsplit("_", n=1).str[0].map(NAME)
    df["_k"] = df.Sample_ID.str.rsplit("_", n=1).str[1].astype(int)
    df = df.merge(meta[["Cancer", "_k", "Sex"]], left_on=["_c", "_k"], right_on=["Cancer", "_k"], how="left")
    prot = [c for c in df.columns if c not in ("Sample_ID", "Cancer", "_c", "_k", "Cancer_x", "Cancer_y", "Sex")
            and pd.api.types.is_numeric_dtype(df[c])]
    rng = np.random.default_rng(52)
    rows = []
    for c, g in df.groupby("_c"):
        s = (g.Sex == "Male").astype(int).to_numpy()
        if s.min() == s.max() or min(s.sum(), len(s) - s.sum()) < 8:
            continue
        X = g[prot].to_numpy(float)
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        cv = StratifiedKFold(5, shuffle=True, random_state=52)
        auc = roc_auc_score(s, cross_val_predict(model, X, s, cv=cv, method="predict_proba")[:, 1])
        null = [roc_auc_score(p, cross_val_predict(model, X, p, cv=cv, method="predict_proba")[:, 1])
                for p in (rng.permutation(s) for _ in range(20))]
        single = {p: roc_auc_score(s, g[p].fillna(g[p].median())) for p in ("XG", "CGA", "PSPN", "LEP", "KLK3", "MSMB") if p in g}
        rows.append({"cancer": c, "n": len(s), "male": int(s.sum()), "cv_auc_sex": auc, "null_mean": float(np.mean(null)),
                     "null_max": float(np.max(null)), **{f"auc_{k}": v for k, v in single.items()}})
    r = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "B2b_sex_mapping_check.csv", index=False)
    json.dump({"rows": rows}, open(OUT / "B2b_sex_mapping_check.json", "w"), indent=1, default=float)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
