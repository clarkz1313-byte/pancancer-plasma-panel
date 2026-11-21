#!/usr/bin/env python3
"""
v5.0: "assign, then confirm" in an independent pan-cancer dataset (TCGA tumour
RNA-seq), using only the protein lists of the two plasma model systems.

DESIGN AND RULE, fixed before any TCGA data were loaded (2026-10-01,
changelog MC-20261001-01):
  - Data: UCSC Xena TCGA per-cohort HiSeqV2 (log2(RSEM normalised count + 1)),
    panel genes only, fetched from the Xena hub (older symbols IL8 = CXCL8,
    PCDH24 = CDHR2).
    Ten of the 12 cancers are available: AML (LAML), BRC (BRCA), CRC (COAD,
    READ), CVX (CESC), DLBCL (DLBC), ENDC (UCEC), GLIOM (GBM, LGG), LUNGC
    (LUAD, LUSC), OVC (OV), PRC (PRAD). CLL and myeloma are not in TCGA.
  - Samples: primary tumour (barcode type 01; 03 for LAML), one per patient.
  - Genes: panel proteins mapped by gene symbol; proteins absent from the data
    are left out and reported, as in the external analysis.
  - Resampling: 50 random stratified 75/25 splits (seed 20261001). In each split,
    genes are standardised on the training part, and
      step 1: multinomial L2 logistic regression (C = 0.1, balanced class
              weights) on the 25-protein list, as in the plasma model;
      step 2: for each cancer, binary L2 logistic regression (C = 1.0) of that
              cancer against the other nine, on its disease-specific list.
    A patient's probabilities are averaged over the splits in which the patient
    was in the testing part.
  - Rule (as in Appendix S1, Table A5): the step-1 assignment is the cancer with
    the highest mean probability; it is confirmed when the step-2 model of that
    cancer gives mean probability >= 0.50 (primary) or >= 0.80.
  - Reported: top-1/2/3 accuracy and balanced accuracy of step 1; for each
    threshold, the share confirmed, accuracy among confirmed and unconfirmed
    patients, wrong assignments flagged, correct assignments lost; the same by
    assigned cancer.
Output: revise_plan/tcga_two_step/results/
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = (PROJECT_ROOT / "revise_plan")
RAW = ROOT / "tcga_two_step/raw"
OUT = ROOT / "tcga_two_step/results"
PANELS = ROOT / "v5_seed52_common_split/panels/panel_memberships_v5.csv"
COHORTS = {"LAML": "AML", "BRCA": "BRC", "COAD": "CRC", "READ": "CRC",
           "CESC": "CVX", "DLBC": "DLBCL", "UCEC": "ENDC", "GBM": "GLIOM",
           "LGG": "GLIOM", "LUAD": "LUNGC", "LUSC": "LUNGC", "OV": "OVC",
           "PRAD": "PRC"}
N_SPLITS, TEST_SIZE, SEED = 50, 0.25, 20261001


def load_panels():
    panels = {}
    for r in csv.DictReader(open(PANELS, encoding="utf-8-sig")):
        v = list(r.values())
        panels.setdefault(v[0].strip(), []).append(v[1].strip())
    mc = panels["multiclass_25"]
    ds = {k.replace("single_", ""): v for k, v in panels.items()
          if k.startswith("single_") and k != "single_pooled_unique"}
    return mc, ds


ALIASES = {"CXCL8": "IL8", "CDHR2": "PCDH24"}  # older HUGO symbols in HiSeqV2
HUB = "https://tcga.xenahubs.net"
CACHE = RAW / "tcga_panel_genes_hiseqv2.csv"


def fetch(genes: list[str]) -> pd.DataFrame:
    """Download only the panel genes from the Xena hub (cached)."""
    import xenaPython as xena
    probes = [ALIASES.get(g, g) for g in genes]
    frames = []
    for cohort in COHORTS:
        ds = f"TCGA.{cohort}.sampleMap/HiSeqV2"
        samples = xena.dataset_samples(HUB, ds, None)
        vals = xena.dataset_probe_values(HUB, ds, samples, probes)[1]
        df = pd.DataFrame({g: [np.nan if v in ("NaN", None) else float(v) for v in col]
                           if col is not None else [np.nan] * len(samples)
                           for g, col in zip(genes, vals)}, index=samples)
        df.insert(0, "cohort", cohort)
        frames.append(df)
    out = pd.concat(frames)
    RAW.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE, index_label="sample")
    return out


def load_tcga(genes: set[str]) -> tuple[pd.DataFrame, pd.Series]:
    genes = sorted(genes)
    raw = pd.read_csv(CACHE, index_col="sample") if CACHE.exists() else fetch(genes)
    frames, labels = [], {}
    for cohort, cancer in COHORTS.items():
        df = raw[raw.cohort == cohort][genes]
        keep = [s for s in df.index
                if s[13:15] == ("03" if cohort == "LAML" else "01")]
        df = df.loc[sorted(keep)]
        df = df[~df.index.str[:12].duplicated()]
        df.index = df.index.str[:12]
        frames.append(df)
        for pt in df.index:
            labels[pt] = cancer
    X = pd.concat(frames, axis=0)
    X = X[~X.index.duplicated()]
    y = pd.Series(labels).loc[X.index]
    return X, y


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mc, ds = load_panels()
    all_genes = set(mc) | {g for v in ds.values() for g in v}
    X, y = load_tcga(all_genes)
    present = [g for g in X.columns if X[g].notna().all()]
    cancers = sorted(y.unique())
    ds = {c: v for c, v in ds.items() if c in cancers}
    mc_use = [g for g in mc if g in present]
    ds_use = {c: [g for g in v if g in present] for c, v in ds.items()}
    coverage = {"multiclass_25": f"{len(mc_use)}/{len(mc)}",
                **{c: f"{len(ds_use[c])}/{len(ds[c])}" for c in cancers},
                "missing": sorted(all_genes - set(present))}

    n = len(y)
    P1 = np.zeros((n, len(cancers))); P2 = np.zeros((n, len(cancers)))
    cnt = np.zeros(n)
    yv = y.to_numpy()
    sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)
    for tr, te in sss.split(np.zeros(n), yv):
        sc = StandardScaler().fit(X.iloc[tr][mc_use])
        m1 = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)
        m1.fit(sc.transform(X.iloc[tr][mc_use]), yv[tr])
        pr = m1.predict_proba(sc.transform(X.iloc[te][mc_use]))
        P1[te] += pr[:, [list(m1.classes_).index(c) for c in cancers]]
        for j, c in enumerate(cancers):
            f = ds_use[c]
            s2 = StandardScaler().fit(X.iloc[tr][f])
            m2 = LogisticRegression(C=1.0, max_iter=5000)
            m2.fit(s2.transform(X.iloc[tr][f]), (yv[tr] == c).astype(int))
            P2[te, j] += m2.predict_proba(s2.transform(X.iloc[te][f]))[:, 1]
        cnt[te] += 1
    ok = cnt > 0
    P1, P2, yv, pid = P1[ok] / cnt[ok, None], P2[ok] / cnt[ok, None], yv[ok], y.index[ok]
    t = np.array([cancers.index(c) for c in yv])
    order = (-P1).argsort(1)
    top1 = order[:, 0]
    correct = top1 == t
    res = {"n_patients": int(len(yv)), "n_per_cancer": pd.Series(yv).value_counts().to_dict(),
           "coverage": coverage,
           "topk_accuracy": {k: float(np.mean([t[i] in order[i, :k] for i in range(len(t))]))
                             for k in (1, 2, 3)},
           "balanced_accuracy_top1": float(balanced_accuracy_score(t, top1))}
    p_conf = P2[np.arange(len(t)), top1]
    rows = []
    for thr in (0.50, 0.80):
        conf = p_conf >= thr
        res[str(thr)] = {
            "n_confirmed": int(conf.sum()), "coverage": float(conf.mean()),
            "accuracy_confirmed": float(correct[conf].mean()),
            "accuracy_unconfirmed": float(correct[~conf].mean()) if (~conf).any() else None,
            "correct_total": int(correct.sum()), "correct_lost": int((correct & ~conf).sum()),
            "wrong_total": int((~correct).sum()), "wrong_flagged": int((~correct & ~conf).sum()),
            "wrong_retained": int((~correct & conf).sum())}
        for j, c in enumerate(cancers):
            sel = top1 == j
            rows.append({"threshold": thr, "assigned": c, "n_assigned": int(sel.sum()),
                         "n_confirmed": int((sel & conf).sum()),
                         "correct_confirmed": int((sel & conf & correct).sum()),
                         "wrong_confirmed": int((sel & conf & ~correct).sum()),
                         "correct_unconfirmed": int((sel & ~conf & correct).sum()),
                         "wrong_unconfirmed": int((sel & ~conf & ~correct).sum())})
    pd.DataFrame(rows).to_csv(OUT / "tcga_assign_confirm_by_assigned_cancer.csv", index=False)
    cm = pd.crosstab(pd.Series(yv, name="true"), pd.Series([cancers[i] for i in top1], name="assigned"))
    cm.to_csv(OUT / "tcga_step1_confusion.csv")
    pd.DataFrame({"patient": pid, "true": yv, "assigned": [cancers[i] for i in top1],
                  "correct": correct, "step2_prob_assigned": p_conf}).to_csv(
        OUT / "tcga_per_patient.csv", index=False)
    pd.DataFrame(P1, index=pid, columns=cancers).assign(true=yv).to_csv(OUT / "tcga_step1_probabilities.csv", index_label="patient")
    pd.DataFrame(P2, index=pid, columns=cancers).assign(true=yv).to_csv(OUT / "tcga_step2_probabilities.csv", index_label="patient")
    json.dump(res, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    print(cm.to_string())


if __name__ == "__main__":
    main()
