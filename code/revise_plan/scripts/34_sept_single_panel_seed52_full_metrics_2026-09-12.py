#!/usr/bin/env python3
"""
Single coherent seed-52 source for every disease-specific internal number.

Reproduces 33_sept_threshold_choice_2026-09-07.py exactly -- same seed, same
stratified 70:30 split on the 12-class label, same all-protein KNN(k=5)
imputation fitted on the development partition, same standardisation, same
unregularised logistic regression, same frozen panel memberships -- and adds
the two calibration metrics that script did not emit (expected calibration
error and Brier score).

Motivation: the manuscript was drawing 0.50-threshold numbers from one run
(part_a_single_seed52_aligned, L2 C=1.0) and 0.80-threshold numbers from
another (threshold_choice_2026-09-07, unregularised), which disagreed in the
third decimal. The author chose threshold_choice as the reference run, so
every reported figure now comes from it.

Output: revise_plan/part_a_single/seed52_full_metrics_2026-09-12/
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
from sklearn.metrics import brier_score_loss, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = PROJECT_ROOT
DATA = ROOT / "data" / "processed" / "filtered_pancancer_data.csv"
PANELS = (ROOT / "revision_package_2026-09-04" / "05_data_audit"
          / "ora_assay_background_2026-09-05" / "panel_memberships.csv")
OUT = ROOT / "revise_plan" / "part_a_single" / "seed52_full_metrics_2026-09-12"

SEED, TEST_SIZE = 52, 0.3
CANCERS = {
    "AML": "single_AML", "BRC": "single_BRC", "CLL": "single_CLL",
    "CRC": "single_CRC", "CVX": "single_CVX", "DLBCL": "single_DLBCL",
    "ENDC": "single_ENDC", "GLIOM": "single_GLIOM", "LUNGC": "single_LUNGC",
    "MYEL": "single_MYEL", "OVC": "single_OVC", "PRC": "single_PRC",
}


def expected_calibration_error(y_true, y_score, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (y_score > lo) & (y_score <= hi) if lo > 0 else (y_score >= lo) & (y_score <= hi)
        if not sel.any():
            continue
        ece += (sel.sum() / n) * abs(y_true[sel].mean() - y_score[sel].mean())
    return float(ece)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    proteins = [c for c in df.columns if c not in ("Sample_ID", "Cancer")]

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

    rows = []
    for canc, key in CANCERS.items():
        feats = [p for p in panels[key] if p in proteins]
        # the pipeline stores DLBCL under its historical LYMPH label
        ytr = (y_all[idx_tr] == ("LYMPH" if canc == "DLBCL" else canc)).astype(int)
        yte = (y_all[idx_te] == ("LYMPH" if canc == "DLBCL" else canc)).astype(int)
        sc = StandardScaler().fit(tr[feats])
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000,
                               random_state=SEED)
        m.fit(sc.transform(tr[feats]), ytr)
        p = m.predict_proba(sc.transform(te[feats]))[:, 1]

        base = dict(cancer=canc, panel_size=len(feats),
                    auc=float(roc_auc_score(yte, p)),
                    ece_10bin=expected_calibration_error(yte, p),
                    brier=float(brier_score_loss(yte, p)))
        for thr in (0.50, 0.80):
            pred = (p >= thr).astype(int)
            tp = int(((pred == 1) & (yte == 1)).sum())
            fn = int(((pred == 0) & (yte == 1)).sum())
            tn = int(((pred == 0) & (yte == 0)).sum())
            fp = int(((pred == 1) & (yte == 0)).sum())
            sens = tp / (tp + fn) if tp + fn else np.nan
            spec = tn / (tn + fp) if tn + fp else np.nan
            rows.append({**base, "threshold": thr, "sensitivity": sens,
                         "specificity": spec,
                         "balanced_accuracy": (sens + spec) / 2,
                         "mcc": float(matthews_corrcoef(yte, pred)),
                         "tp": tp, "fn": fn, "tn": tn, "fp": fp})

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "per_cancer_metrics_full.csv", index=False)

    summary = {}
    for thr in (0.50, 0.80):
        s = out[out.threshold == thr]
        summary[str(thr)] = {k: float(s[k].mean()) for k in
                             ("auc", "sensitivity", "specificity",
                              "balanced_accuracy", "ece_10bin", "brier")}
        summary[str(thr)]["sensitivity_min"] = float(s.sensitivity.min())
        summary[str(thr)]["sensitivity_min_cancer"] = s.loc[s.sensitivity.idxmin(), "cancer"]
        summary[str(thr)]["specificity_min"] = float(s.specificity.min())
        summary[str(thr)]["specificity_min_cancer"] = s.loc[s.specificity.idxmin(), "cancer"]
        summary[str(thr)]["auc_min"] = float(s.auc.min())
        summary[str(thr)]["auc_min_cancer"] = s.loc[s.auc.idxmin(), "cancer"]
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
