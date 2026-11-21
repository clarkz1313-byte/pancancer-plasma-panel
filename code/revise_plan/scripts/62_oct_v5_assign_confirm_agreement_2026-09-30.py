#!/usr/bin/env python3
"""
v5.0 exploratory analysis: agreement between the 25-protein multiclass model
and the matching disease-specific model ("assign, then confirm").

RULE, fixed before the analysis was run (2026-09-30, plan in the Discussion
rewrite, changelog MC-20260930-05):
  - Step 1: each test patient receives the multiclass top-1 label c.
  - Step 2: the label is "confirmed" when the disease-specific model for c
    gives predicted probability >= 0.50 (primary) or >= 0.80 (secondary).
  - Reported for each threshold: coverage (share confirmed), accuracy among
    confirmed and among unconfirmed patients, correct labels lost to
    non-confirmation, and wrong labels retained or flagged.
  - Also reported: top-1/2/3 multiclass accuracy.
No model, panel, threshold or rule is changed or tuned here. Both model
systems are the fixed v5 models; the test set is the same 413 patients that
were used to choose the panels, so the result is exploratory.

Inputs
  - multiclass test probabilities of the locked 25-protein model (seed 52),
    figures_nature_v3/production/01_study_cohort/source_data/
    locked_test_probability_summary.csv (top-1 accuracy 311/413 = 0.753,
    the reported value);
  - disease-specific models refitted exactly as in script 40 (same split,
    KNN imputation without protein_count, standardisation, unpenalised LR,
    frozen v5 panels).
Output: revise_plan/v5_seed52_common_split/internal/assign_confirm/
"""
from __future__ import annotations
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))

from pathlib import Path
import csv
import json

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = PROJECT_ROOT
DATA = ROOT / "data" / "processed" / "filtered_pancancer_data.csv"
PANELS = ROOT / "revise_plan/v5_seed52_common_split/panels/panel_memberships_v5.csv"
MC_PROBS = ROOT / ("figures_nature_v3/production/01_study_cohort/source_data/"
                   "locked_test_probability_summary.csv")
S9 = ROOT / "revise_plan/v5_seed52_common_split/internal/table_s9/per_cancer_metrics_full.csv"
OUT = ROOT / "revise_plan/v5_seed52_common_split/internal/assign_confirm"

SEED, TEST_SIZE = 52, 0.3
LABEL = {"DLBCL": "LYMPH"}  # historical label in the data file


def disease_specific_probs() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    proteins = [c for c in df.columns if c not in ("Sample_ID", "Cancer", "protein_count")]
    panels = {}
    for r in csv.DictReader(open(PANELS, encoding="utf-8-sig")):
        v = list(r.values())
        panels.setdefault(v[0].strip(), []).append(v[1].strip())

    y_all = df["Cancer"].to_numpy()
    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=SEED, stratify=y_all)
    imp = KNNImputer(n_neighbors=5)
    tr = pd.DataFrame(imp.fit_transform(df.iloc[idx_tr][proteins]), columns=proteins)
    te = pd.DataFrame(imp.transform(df.iloc[idx_te][proteins]), columns=proteins)

    out = pd.DataFrame({"Sample_ID": df.iloc[idx_te]["Sample_ID"].to_numpy(),
                        "true_class": y_all[idx_te]})
    for key, feats in panels.items():
        if not key.startswith("single_") or key == "single_pooled_unique":
            continue
        canc = key.replace("single_", "")
        lab = LABEL.get(canc, canc)
        feats = [p for p in feats if p in proteins]
        sc = StandardScaler().fit(tr[feats])
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, random_state=SEED)
        m.fit(sc.transform(tr[feats]), (y_all[idx_tr] == lab).astype(int))
        out[f"ds_{lab}"] = m.predict_proba(sc.transform(te[feats]))[:, 1]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = disease_specific_probs()
    mc = pd.read_csv(MC_PROBS)
    assert set(mc.Sample_ID) == set(ds.Sample_ID), "test IDs differ"
    d = mc.merge(ds.drop(columns="true_class"), on="Sample_ID")
    assert len(d) == 413

    # check the refit reproduces Table S9 AUCs
    s9 = pd.read_csv(S9).drop_duplicates("cancer").set_index("cancer")["auc"]
    for canc, auc in s9.items():
        lab = LABEL.get(canc, canc)
        got = roc_auc_score((d.true_class == lab).astype(int), d[f"ds_{lab}"])
        assert abs(got - auc) < 1e-9, (canc, got, auc)

    classes = [c.replace("score_", "") for c in d.columns if c.startswith("score_")]
    P = d[[f"score_{c}" for c in classes]].to_numpy()
    t = d.true_class.map({c: i for i, c in enumerate(classes)}).to_numpy()
    order = (-P).argsort(1)
    topk = {k: int(sum(t[i] in order[i, :k] for i in range(len(d)))) for k in (1, 2, 3)}

    d["top1"] = [classes[i] for i in order[:, 0]]
    d["correct"] = d.top1 == d.true_class
    d["ds_prob_top1"] = [d.loc[i, f"ds_{c}"] for i, c in zip(d.index, d.top1)]

    res = {"n": len(d), "topk_correct": topk,
           "topk_accuracy": {k: v / len(d) for k, v in topk.items()}}
    rows = []
    for thr in (0.50, 0.80):
        conf = d.ds_prob_top1 >= thr
        r = {
            "threshold": thr,
            "n_confirmed": int(conf.sum()),
            "coverage": float(conf.mean()),
            "accuracy_confirmed": float(d.correct[conf].mean()),
            "accuracy_unconfirmed": float(d.correct[~conf].mean()),
            "correct_total": int(d.correct.sum()),
            "correct_confirmed": int((d.correct & conf).sum()),
            "correct_lost": int((d.correct & ~conf).sum()),
            "wrong_total": int((~d.correct).sum()),
            "wrong_retained": int((~d.correct & conf).sum()),
            "wrong_flagged": int((~d.correct & ~conf).sum()),
        }
        res[str(thr)] = r
        for c in classes:
            sel = d.top1 == c
            rows.append({"threshold": thr, "assigned": c, "n_assigned": int(sel.sum()),
                         "n_confirmed": int((sel & conf).sum()),
                         "correct_confirmed": int((sel & conf & d.correct).sum()),
                         "wrong_confirmed": int((sel & conf & ~d.correct).sum()),
                         "correct_unconfirmed": int((sel & ~conf & d.correct).sum()),
                         "wrong_unconfirmed": int((sel & ~conf & ~d.correct).sum())})
    pd.DataFrame(rows).to_csv(OUT / "assign_confirm_by_assigned_cancer.csv", index=False)
    d[["Sample_ID", "true_class", "top1", "correct", "ds_prob_top1"]].to_csv(
        OUT / "assign_confirm_per_patient.csv", index=False)
    json.dump(res, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
