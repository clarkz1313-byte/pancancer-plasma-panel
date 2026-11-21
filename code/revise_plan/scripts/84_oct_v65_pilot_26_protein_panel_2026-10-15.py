"""Pilot (no manuscript change): what if the multiclass panel were the development-only winner?

The 25-protein panel was chosen among 540 shortlisted candidates by internal-test balanced accuracy.
The candidate ranked first by development cross-validation alone (candidate 2047, 26 proteins) contains
the same 25 proteins plus MSMB. This pilot refits both panels the same way and compares every
downstream analysis that can be rerun from saved inputs:

  A. internal multiclass performance (1,000 bootstrap resamples, seed 52024, as the locked reproducer)
  B. internal two-step rule at 0.50 and 0.80, accuracy gain and the stricter-cutoff comparison
     (2,000 paired bootstrap resamples, seed 20261003, as scripts 76, 81 and 82)
  C. TCGA two-step rule with the 26-protein list in step 1 (same 50 splits, seed 20261001; step 2 is
     unchanged because its lists and splits are unchanged, so the saved step-2 probabilities are reused)
  D. overlap and sex-dependence facts needed for the text

Not rerun here: single-cancer tissue refits of the 25-protein list, annotation/network tests and the
genetic screen for the ten new MSMB-cancer pairs (listed in the output as pending).
Output: revise_plan/pilot_26_protein_panel/
"""
from __future__ import annotations
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = PROJECT_ROOT
HERE = ROOT / "revise_plan/scripts"
OUT = ROOT / "revise_plan/pilot_26_protein_panel"
DATA = ROOT / "data/processed/filtered_pancancer_data.csv"
AUDIT = ROOT / "revise_plan/part_b_multiclass/v11_seed52_lr_l2_wide_compact_sensitivity/tables/stage2_test_audit_lr_l2_all_rows.csv"
MC_PROBS = ROOT / "figures_nature_v3/production/01_study_cohort/source_data/locked_test_probability_summary.csv"
TCGA_P2 = ROOT / "revise_plan/tcga_two_step/results/tcga_step2_probabilities.csv"
TCGA_SUMMARY = ROOT / "revise_plan/tcga_two_step/results/summary.json"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s62 = load("s62", "62_oct_v5_assign_confirm_agreement_2026-09-30.py")
s64 = load("s64", "64_oct_v5_tcga_assign_confirm_2026-10-01.py")
s76 = load("s76", "76_oct_v6_confidence_comparator_2026-10-03.py")


def panels_from_audit():
    d = pd.read_csv(AUDIT)
    chosen = d.loc[d.test_balanced_accuracy.idxmax()]
    devbest = d.loc[d.stage1_objective.idxmax()]
    assert int(chosen.feature_count) == 25 and int(devbest.feature_count) == 26
    p25, p26 = chosen.features.split(";"), devbest.features.split(";")
    assert set(p26) - set(p25) == {"MSMB"} and set(p25) <= set(p26)
    return p25, p26


def multiclass_test_probs(features):
    df = pd.read_csv(DATA)
    proteins = [c for c in df.columns if c not in ("Sample_ID", "Cancer", "protein_count")]
    y = df["Cancer"].to_numpy()
    tr, te = train_test_split(np.arange(len(df)), test_size=0.3, random_state=52, stratify=y)
    imp = KNNImputer(n_neighbors=5).fit(df.iloc[tr][proteins])
    xtr = pd.DataFrame(imp.transform(df.iloc[tr][proteins]), columns=proteins)[features]
    xte = pd.DataFrame(imp.transform(df.iloc[te][proteins]), columns=proteins)[features]
    sc = StandardScaler().fit(xtr)
    m = LogisticRegression(C=0.1, class_weight="balanced", solver="lbfgs", max_iter=4000)
    m.fit(sc.transform(xtr), y[tr])
    P = m.predict_proba(sc.transform(xte))
    return df.iloc[te]["Sample_ID"].to_numpy(), y[te], list(m.classes_), P


def metrics(y, pred, P, classes):
    rec = recall_score(y, pred, labels=classes, average=None)
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"),
            "macro_auc": roc_auc_score(y, P, multi_class="ovr", average="macro", labels=classes),
            "min_class_recall": float(rec.min()), **{f"recall_{c}": float(r) for c, r in zip(classes, rec)}}


def boot_ci(y, P, classes, B=1000, seed=52024):
    rng = np.random.default_rng(seed)
    pred = np.array(classes)[P.argmax(1)]
    vals = {k: [] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "min_class_recall")}
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < len(classes):
            continue
        m = metrics(y[i], pred[i], P[i], classes)
        for k in vals:
            vals[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items()}


def two_step(correct, conf, second, rng_seed=20261003, B=2000):
    """Accepted counts, accuracy gain and the stricter-cutoff comparison (scripts 76, 81, 82)."""
    out = {}
    for thr in (0.50, 0.80):
        acc = second >= thr
        out[str(thr)] = {"accepted": int(acc.sum()), "undecided": int((~acc).sum()),
                         "accepted_share": float(acc.mean()),
                         "correct_accepted": int((correct & acc).sum()),
                         "accuracy_accepted": float(correct[acc].mean()),
                         "wrong_left_undecided": int((~correct & ~acc).sum()),
                         "wrong_total": int((~correct).sum()),
                         "correct_left_undecided": int((correct & ~acc).sum())}
    n = len(correct)
    gate, keep, k, tie = s76.matched(conf, second, correct)
    wg, wc = int((~correct[gate]).sum()), int((~correct[keep]).sum())
    rng = np.random.default_rng(rng_seed)
    gain, rel, dacc = [], [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        cf, sd, cr = conf[i], second[i], correct[i]
        g = sd >= 0.50
        if g.sum() == 0:
            continue
        kp = s76.topk_mask(cf, int(g.sum()))
        gain.append(cr[g].mean() - cr.mean())
        dacc.append(cr[g].mean() - cr[kp].mean())
        wgb, wcb = (~cr[g]).sum(), (~cr[kp]).sum()
        rel.append(1 - wgb / wcb if wcb else np.nan)
    gain, rel, dacc = np.array(gain), np.array(rel), np.array(dacc)
    p_two = lambda a: float(min(1.0, 2 * min(np.mean(a <= 0), np.mean(a >= 0))))
    out["gain_points"] = 100 * (correct[second >= 0.5].mean() - correct.mean())
    out["gain_ci"] = (100 * np.percentile(gain, 2.5), 100 * np.percentile(gain, 97.5))
    out["stricter_cutoff"] = {"wrong_stricter": wc, "wrong_step2": wg, "shared_accepted": int((gate & keep).sum()),
                              "relative_reduction": 1 - wg / wc if wc else None,
                              "rel_ci": (float(np.nanpercentile(rel, 2.5)), float(np.nanpercentile(rel, 97.5))),
                              "accuracy_diff_points": 100 * (correct[gate].mean() - correct[keep].mean()),
                              "acc_ci": (100 * np.percentile(dacc, 2.5), 100 * np.percentile(dacc, 97.5)),
                              "p_two_sided": p_two(dacc)}  # same sign as the wrong-count difference at equal k
    return out


def internal(p25, p26):
    ds = s62.disease_specific_probs()
    res = {}
    ref = pd.read_csv(MC_PROBS)
    for name, feats in (("25", p25), ("26", p26)):
        ids, y, classes, P = multiclass_test_probs(feats)
        if name == "25":  # the refit must reproduce the locked 25-protein probabilities
            r = ref.set_index("Sample_ID").loc[ids, [f"score_{c}" for c in classes]].to_numpy()
            assert np.abs(r - P).max() < 1e-6, np.abs(r - P).max()
        pred = np.array(classes)[P.argmax(1)]
        d = pd.DataFrame({"Sample_ID": ids, "true": y, "pred": pred, "conf": P.max(1)})
        d = d.merge(ds.drop(columns="true_class"), on="Sample_ID")
        assert len(d) == 413
        lab = {c: c for c in classes}
        d["second"] = [d.loc[i, f"ds_{lab[c]}"] for i, c in zip(d.index, d.pred)]
        correct = (d.pred == d.true).to_numpy()
        order = (-P).argsort(1)
        t = np.array([classes.index(c) for c in y])
        res[name] = {"metrics": metrics(y, pred, P, classes), "ci": boot_ci(y, P, classes),
                     "top3": float(np.mean([t[i] in order[i, :3] for i in range(len(t))])),
                     "two_step": two_step(correct, d.conf.to_numpy(), d.second.to_numpy())}
        d.to_csv(OUT / f"internal_per_patient_{name}.csv", index=False)
    return res


def tcga(p25, p26):
    mc, ds = s64.load_panels()
    assert set(mc) == set(p25)
    all_genes = set(p26) | {g for v in ds.values() for g in v}
    X, y = s64.load_tcga(all_genes)
    present = [g for g in X.columns if X[g].notna().all()]
    cancers = sorted(y.unique())
    p2 = pd.read_csv(TCGA_P2, index_col="patient")
    res = {}
    for name, feats in (("25", p25), ("26", p26)):
        use = [g for g in feats if g in present]
        n, yv = len(y), y.to_numpy()
        P1, cnt = np.zeros((n, len(cancers))), np.zeros(n)
        sss = StratifiedShuffleSplit(n_splits=s64.N_SPLITS, test_size=s64.TEST_SIZE, random_state=s64.SEED)
        for tr, te in sss.split(np.zeros(n), yv):
            sc = StandardScaler().fit(X.iloc[tr][use])
            m1 = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)
            m1.fit(sc.transform(X.iloc[tr][use]), yv[tr])
            pr = m1.predict_proba(sc.transform(X.iloc[te][use]))
            P1[te] += pr[:, [list(m1.classes_).index(c) for c in cancers]]
            cnt[te] += 1
        ok = cnt > 0
        P1, yv2, pid = P1[ok] / cnt[ok, None], yv[ok], y.index[ok]
        top1 = P1.argmax(1)
        t = np.array([cancers.index(c) for c in yv2])
        correct = top1 == t
        P2 = p2.loc[pid, cancers].to_numpy()
        second = P2[np.arange(len(t)), top1]
        order = (-P1).argsort(1)
        gate, keep, _, _ = s76.matched(P1.max(1), second, correct)
        pd.DataFrame({"patient": pid, "true": yv2, "assigned": [cancers[i] for i in top1],
                      "correct": correct, "conf": P1.max(1), "second": second,
                      "accepted_step2": gate, "accepted_stricter": keep}).to_csv(
            OUT / f"tcga_per_patient_{name}.csv", index=False)
        res[name] = {"genes_used": f"{len(use)}/{len(feats)}", "n": int(len(t)),
                     "accuracy_step1": float(correct.mean()),
                     "balanced_accuracy_step1": float(balanced_accuracy_score(t, top1)),
                     "top3": float(np.mean([t[i] in order[i, :3] for i in range(len(t))])),
                     "two_step": two_step(correct, P1.max(1), second)}
    ref = json.load(open(TCGA_SUMMARY))
    assert abs(res["25"]["accuracy_step1"] - ref["topk_accuracy"]["1"]) < 1e-12, "TCGA 25 refit differs"
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p25, p26 = panels_from_audit()
    res = {"panels": {"25": p25, "26": p26, "added": "MSMB"},
           "internal": internal(p25, p26), "tcga": tcga(p25, p26),
           "pending_not_rerun": ["single-cancer tissue refits of the 25-protein list (add MSMB)",
                                 "ORA/STRING/Hallmark/correlation tests of the multiclass panel",
                                 "genetic screen: MSMB with the 10 other cancers (MSMB-prostate pair already tested)",
                                 "sex analysis (MSMB is a prostate protein)",
                                 "Figures 3D, 4A-B, 5, 7, 9, 10, 11, 12 and Tables S1, S2, S4, S6-S8, A2-A12"]}
    json.dump(res, open(OUT / "pilot_summary.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float)[:12000])


if __name__ == "__main__":
    main()
