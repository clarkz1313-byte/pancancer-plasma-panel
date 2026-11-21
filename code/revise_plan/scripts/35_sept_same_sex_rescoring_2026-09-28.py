#!/usr/bin/env python3
"""
Same-sex rescoring of the five sex-restricted disease-specific panels
(BRC, CVX, ENDC, OVC, PRC) on the seed-52 held-out set.

WHY
---
Bueno Alvez et al. 2023 (Suppl. Data 4) scored each sex-restricted cancer
against other cancers FROM THE SAME SEX only. Table S9 scores ours against
all other cancers, so part of the separation may come from sex (e.g. prostate
versus every female patient). This script puts both designs side by side.
Closes changelog open item B1 (MC-20260927-19) and is TODO1 step 1.

DESIGN
------
Pipeline identical to 34_sept_single_panel_seed52_full_metrics_2026-09-12.py
(seed 52, stratified 70:30 on the 12-class label, all-column KNN(5) imputation
fitted on development, standardise, unregularised LR, frozen panels).

  S9   all-comparator model, scored on the full held-out set (reproduces S9;
       hard-fails if AUC differs from Table S9 by more than 5e-4)
  R    same model, RESCORED on held-out cases + same-sex held-out non-cases
  T    model RETRAINED with same-sex non-cases only, scored as in R
       (the full Alvez control design)
  X    sensitivity: as T, but imputation excludes the protein_count metadata
       column (script 34 left it in the KNN neighbour space)

AUC 95% CIs: 2,000 stratified bootstrap resamples of the scored set.
Alvez Suppl. Data 4 rows at 10 and 1,463 proteins are joined for comparison.

Output: revise_plan/part_a_single/same_sex_rescoring_2026-09-28/
"""
from __future__ import annotations
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))

from pathlib import Path
import csv
import json
import sys

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = PROJECT_ROOT
DATA = ROOT / "data" / "processed" / "filtered_pancancer_data.csv"
META = ROOT / "references" / "alves_supplementary_data" / "supp2.xlsx"
SUPP4 = ROOT / "references" / "alves_supplementary_data" / "supp4.xlsx"
PANELS = (ROOT / "revision_package_2026-09-04" / "05_data_audit"
          / "ora_assay_background_2026-09-05" / "panel_memberships.csv")
S9 = (ROOT / "revise_plan" / "part_a_single" / "seed52_full_metrics_2026-09-12"
      / "per_cancer_metrics_full.csv")
OUT = ROOT / "revise_plan" / "part_a_single" / "same_sex_rescoring_2026-09-28"

SEED, TEST_SIZE, N_BOOT = 52, 0.3, 2000
# code -> (panel key, sex of the cancer, Alvez supp4 name)
SEX_RESTRICTED = {
    "BRC": ("single_BRC", "Female", "Breast"),
    "CVX": ("single_CVX", "Female", "Cervical"),
    "ENDC": ("single_ENDC", "Female", "Endometrial"),
    "OVC": ("single_OVC", "Female", "Ovarian"),
    "PRC": ("single_PRC", "Male", "Prostate"),
}
MIXED_SEX = {
    "AML": ("single_AML", "AML"), "CLL": ("single_CLL", "CLL"),
    "CRC": ("single_CRC", "Colorectal"), "DLBCL": ("single_DLBCL", "DLBCL"),
    "GLIOM": ("single_GLIOM", "Glioma"), "LUNGC": ("single_LUNGC", "Lung"),
    "MYEL": ("single_MYEL", "Myeloma"),
}
ALVEZ_NAME = {**{k: v[2] for k, v in SEX_RESTRICTED.items()},
              **{k: v[1] for k, v in MIXED_SEX.items()}}
NAME_MAP = {"AML": "AML", "BRC": "Breast", "CLL": "CLL", "CRC": "Colorectal",
            "CVX": "Cervical", "ENDC": "Endometrial", "GLIOM": "Glioma",
            "LUNGC": "Lung", "LYMPH": "DLBCL", "MYEL": "Myeloma",
            "OVC": "Ovarian", "PRC": "Prostate"}


def boot_auc_ci(y, p, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = [roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                          np.r_[p[rng.choice(pos, len(pos))],
                                p[rng.choice(neg, len(neg))]])
            for _ in range(N_BOOT)]
    return np.percentile(aucs, [2.5, 97.5])


def score(y, p, rng):
    lo, hi = boot_auc_ci(y, p, rng)
    out = {"n_cases": int(y.sum()), "n_noncases": int((y == 0).sum()),
           "auc": float(roc_auc_score(y, p)), "auc_lo": lo, "auc_hi": hi}
    for thr in (0.50, 0.80):
        pred = p >= thr
        out[f"sens_{thr:.2f}"] = float(pred[y == 1].mean())
        out[f"spec_{thr:.2f}"] = float((~pred[y == 0]).mean())
    return out


def fit_predict(tr, te, feats, ytr, keep_tr):
    sc = StandardScaler().fit(tr.loc[keep_tr, feats])
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000,
                           random_state=SEED)
    m.fit(sc.transform(tr.loc[keep_tr, feats]), ytr[keep_tr])
    return m.predict_proba(sc.transform(te[feats]))[:, 1]


def impute(df, idx_tr, idx_te, cols):
    imp = KNNImputer(n_neighbors=5)
    tr = pd.DataFrame(imp.fit_transform(df.iloc[idx_tr][cols]), columns=cols)
    te = pd.DataFrame(imp.transform(df.iloc[idx_te][cols]), columns=cols)
    return tr, te


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    # every non-ID column, exactly as script 34 (this includes protein_count)
    cols34 = [c for c in df.columns if c not in ("Sample_ID", "Cancer")]
    cols_clean = [c for c in cols34 if c != "protein_count"]

    # sex: positional join within cancer group, validated as in script 30
    meta = pd.read_excel(META).reset_index(drop=True)
    meta["_k"] = meta.groupby("Cancer").cumcount() + 1
    key = df["Sample_ID"].str.rsplit("_", n=1)
    sex = (pd.DataFrame({"Cancer": key.str[0].map(NAME_MAP),
                         "_k": key.str[1].astype(int)})
           .merge(meta[["Cancer", "_k", "Sex"]], on=["Cancer", "_k"], how="left")
           ["Sex"].to_numpy())
    if pd.isna(sex).any():
        print("STOP: sex missing for some patients", file=sys.stderr)
        return 1
    for code, (_, s, _) in SEX_RESTRICTED.items():
        lab = df["Cancer"].to_numpy() == code
        if not (sex[lab] == s).all():
            print(f"STOP: {code} is not all {s}; positional join invalid",
                  file=sys.stderr)
            return 1

    panels = {}
    for r in csv.DictReader(open(PANELS, encoding="utf-8-sig")):
        v = list(r.values())
        panels.setdefault(v[0].strip(), []).append(v[1].strip())

    y_all = df["Cancer"].to_numpy()
    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=SEED, stratify=y_all)
    sex_tr, sex_te = sex[idx_tr], sex[idx_te]

    tr34, te34 = impute(df, idx_tr, idx_te, cols34)
    trX, teX = impute(df, idx_tr, idx_te, cols_clean)

    s9 = pd.read_csv(S9).drop_duplicates("cancer").set_index("cancer")
    s4 = pd.read_excel(SUPP4)

    rows = []
    for code, (pkey, s, aname) in SEX_RESTRICTED.items():
        rng = np.random.default_rng(SEED)
        feats = [p for p in panels[pkey] if p in cols_clean]
        ytr = (y_all[idx_tr] == code).astype(int)
        yte = (y_all[idx_te] == code).astype(int)
        all_tr = np.ones(len(ytr), bool)
        same_tr, same_te = sex_tr == s, sex_te == s

        p_all = fit_predict(tr34, te34, feats, ytr, all_tr)
        if abs(roc_auc_score(yte, p_all) - s9.loc[code, "auc"]) > 5e-4:
            print(f"STOP: {code} does not reproduce Table S9 AUC", file=sys.stderr)
            return 1
        p_same = fit_predict(tr34, te34, feats, ytr, same_tr)
        p_same_x = fit_predict(trX, teX, feats, ytr, same_tr)

        a4 = s4[s4.Cancer == aname].set_index("Num_proteins")["AUC"]
        for design, y, p in (
                ("S9_all_comparators", yte, p_all),
                ("R_rescored_same_sex", yte[same_te], p_all[same_te]),
                ("T_retrained_same_sex", yte[same_te], p_same[same_te]),
                ("X_retrained_same_sex_no_protein_count",
                 yte[same_te], p_same_x[same_te])):
            rows.append({"cancer": code, "sex": s, "panel_size": len(feats),
                         "design": design, **score(y, p, rng),
                         "alvez_auc_10p": float(a4.get(10, np.nan)),
                         "alvez_auc_1463p": float(a4.get(1463, np.nan))})

    # mixed-sex cancers: Table S9 design, same bootstrap, for manuscript Table 1
    for code, (pkey, aname) in MIXED_SEX.items():
        rng = np.random.default_rng(SEED)
        feats = [p for p in panels[pkey] if p in cols_clean]
        label = "LYMPH" if code == "DLBCL" else code
        ytr = (y_all[idx_tr] == label).astype(int)
        yte = (y_all[idx_te] == label).astype(int)
        p_all = fit_predict(tr34, te34, feats, ytr, np.ones(len(ytr), bool))
        if abs(roc_auc_score(yte, p_all) - s9.loc[code, "auc"]) > 5e-4:
            print(f"STOP: {code} does not reproduce Table S9 AUC", file=sys.stderr)
            return 1
        a4 = s4[s4.Cancer == aname].set_index("Num_proteins")["AUC"]
        rows.append({"cancer": code, "sex": "both", "panel_size": len(feats),
                     "design": "S9_all_comparators", **score(yte, p_all, rng),
                     "alvez_auc_10p": float(a4.get(10, np.nan)),
                     "alvez_auc_1463p": float(a4.get(1463, np.nan))})

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "same_sex_rescoring.csv", index=False)

    # Table 1: same-sex design where the cancer is sex-restricted, else Table S9
    t1 = out[(out.design == "T_retrained_same_sex")
             | ((out.sex == "both") & (out.design == "S9_all_comparators"))]
    for n in (3, 10, 50, 200, 1463):
        t1 = t1.assign(**{f"alvez_{n}p": [
            float(s4[(s4.Cancer == ALVEZ_NAME[c]) & (s4.Num_proteins == n)].AUC.iloc[0])
            for c in t1.cancer]})
    t1.drop(columns=["alvez_auc_10p", "alvez_auc_1463p"]).to_csv(
        OUT / "table1_same_cohort_comparison.csv", index=False)
    pd.set_option("display.width", 200)
    print(out[["cancer", "panel_size", "design", "n_cases", "n_noncases", "auc",
               "auc_lo", "auc_hi", "sens_0.50", "spec_0.50",
               "alvez_auc_10p", "alvez_auc_1463p"]].round(3).to_string(index=False))
    json.dump({"seed": SEED, "n_boot": N_BOOT,
               "n_test": int(len(idx_te)),
               "n_test_female": int((sex_te == "Female").sum()),
               "n_test_male": int((sex_te == "Male").sum())},
              open(OUT / "run_info.json", "w"), indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
