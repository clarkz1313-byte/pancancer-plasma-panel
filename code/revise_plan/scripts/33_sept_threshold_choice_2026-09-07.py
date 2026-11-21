#!/usr/bin/env python3
"""
Full metric comparison of the 0.50 and 0.80 operating points for the 12
disease-specific panels, to choose one convention for the manuscript.

Adds the metrics a confirmatory-stage decision actually turns on -- PPV, NPV,
balanced accuracy, F1, Youden's J, MCC -- which sensitivity and specificity
alone do not settle, plus the two-stage cascade sensitivity implied by each
threshold.

Also emits the per-class sample sizes (development / held-out / total), which
appear nowhere in the manuscript despite "natural class prevalence" and
"minimum class recall" carrying the argument.

Pipeline held fixed: seed 52, stratified 70/30, KNN k=5 imputation fitted on
the full 1,463-protein matrix in the development partition, standardisation,
unpenalised logistic regression.

Output: revise_plan/part_a_single/threshold_choice_2026-09-07/
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
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = PROJECT_ROOT
DATA = ROOT / "data" / "processed" / "filtered_pancancer_data.csv"
PANELS = (ROOT / "revision_package_2026-09-04" / "05_data_audit"
          / "ora_assay_background_2026-09-05" / "panel_memberships.csv")
OUT = ROOT / "revise_plan" / "part_a_single" / "threshold_choice_2026-09-07"

SEED, TEST_SIZE = 52, 0.3
STAGE1_MACRO_RECALL = 0.799  # locked-25 multiclass, held-out

CANCERS = {
    "AML": "single_AML", "BRC": "single_BRC", "CLL": "single_CLL",
    "CRC": "single_CRC", "CVX": "single_CVX", "LYMPH": "single_DLBCL",
    "ENDC": "single_ENDC", "GLIOM": "single_GLIOM", "LUNGC": "single_LUNGC",
    "MYEL": "single_MYEL", "OVC": "single_OVC", "PRC": "single_PRC",
}
FULL = {
    "AML": "acute myeloid leukaemia", "BRC": "breast",
    "CLL": "chronic lymphocytic leukaemia", "CRC": "colorectal",
    "CVX": "cervical", "LYMPH": "diffuse large B-cell lymphoma",
    "ENDC": "endometrial", "GLIOM": "glioma", "LUNGC": "lung",
    "MYEL": "myeloma", "OVC": "ovarian", "PRC": "prostate",
}


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
        np.arange(len(df)), test_size=TEST_SIZE, random_state=SEED,
        stratify=y_all)

    imp = KNNImputer(n_neighbors=5)
    tr = pd.DataFrame(imp.fit_transform(df.iloc[idx_tr][proteins]),
                      columns=proteins)
    te = pd.DataFrame(imp.transform(df.iloc[idx_te][proteins]),
                      columns=proteins)

    # ---------- per-class sample sizes ----------
    print("Per-class sample sizes\n" + "=" * 58)
    print(f"{'cancer':<8}{'total':>7}{'dev':>7}{'held-out':>10}{'% of cohort':>13}")
    sizes = []
    for c in sorted(CANCERS):
        n_tot = int((y_all == c).sum())
        n_dev = int((y_all[idx_tr] == c).sum())
        n_te = int((y_all[idx_te] == c).sum())
        sizes.append(dict(cancer=c, name=FULL[c], total=n_tot,
                          development=n_dev, held_out=n_te,
                          pct=100 * n_tot / len(y_all)))
        print(f"{c:<8}{n_tot:>7}{n_dev:>7}{n_te:>10}{100*n_tot/len(y_all):>12.1f}%")
    s = pd.DataFrame(sizes).sort_values("total")
    print(f"\ntotal {len(y_all)}; smallest {s.iloc[0].cancer} n={s.iloc[0].total}, "
          f"largest {s.iloc[-1].cancer} n={s.iloc[-1].total}, "
          f"ratio {s.iloc[-1].total / s.iloc[0].total:.1f}x")
    s.to_csv(OUT / "per_class_sizes.csv", index=False)

    # ---------- threshold grid ----------
    rows = []
    for canc, key in CANCERS.items():
        feats = [p for p in panels[key] if p in proteins]
        ytr = (y_all[idx_tr] == canc).astype(int)
        yte = (y_all[idx_te] == canc).astype(int)
        sc = StandardScaler().fit(tr[feats])
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000,
                               random_state=SEED)
        m.fit(sc.transform(tr[feats]), ytr)
        p = m.predict_proba(sc.transform(te[feats]))[:, 1]
        auc = float(roc_auc_score(yte, p))
        for thr in (0.50, 0.80):
            pred = (p >= thr).astype(int)
            tp = int(((pred == 1) & (yte == 1)).sum())
            fn = int(((pred == 0) & (yte == 1)).sum())
            tn = int(((pred == 0) & (yte == 0)).sum())
            fp = int(((pred == 1) & (yte == 0)).sum())
            sens = tp / (tp + fn) if tp + fn else np.nan
            spec = tn / (tn + fp) if tn + fp else np.nan
            ppv = tp / (tp + fp) if tp + fp else np.nan
            npv = tn / (tn + fn) if tn + fn else np.nan
            f1 = (2 * ppv * sens / (ppv + sens)
                  if ppv and sens and not np.isnan(ppv) else 0.0)
            rows.append(dict(
                cancer=canc, threshold=thr, auc=auc, sensitivity=sens,
                specificity=spec, ppv=ppv, npv=npv, f1=f1,
                balanced_accuracy=(sens + spec) / 2, youden_j=sens + spec - 1,
                mcc=float(matthews_corrcoef(yte, pred)),
                tp=tp, fn=fn, tn=tn, fp=fp))

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "per_cancer_metrics.csv", index=False)

    print("\n\nMean across the 12 panels, by threshold\n" + "=" * 58)
    cols = ["auc", "sensitivity", "specificity", "ppv", "npv", "f1",
            "balanced_accuracy", "youden_j", "mcc"]
    summ = r.groupby("threshold")[cols].mean().T
    print(summ.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nCascade sensitivity (stage 1 macro recall "
          f"{STAGE1_MACRO_RECALL} x stage 2 sensitivity)")
    for thr in (0.50, 0.80):
        s2 = r[r.threshold == thr].sensitivity.mean()
        print(f"  at {thr:.2f}: {STAGE1_MACRO_RECALL:.3f} x {s2:.3f} = "
              f"{STAGE1_MACRO_RECALL * s2:.3f}")

    print("\nPanels falling below 0.30 sensitivity")
    for thr in (0.50, 0.80):
        bad = r[(r.threshold == thr) & (r.sensitivity < 0.30)]
        names = ", ".join(f"{x.cancer} {x.sensitivity:.2f}"
                          for _, x in bad.iterrows()) or "none"
        print(f"  at {thr:.2f}: {len(bad)}/12  ({names})")

    summ.to_json(OUT / "threshold_summary.json", indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
