"""Shared analysis for the mixed-cancer pilot (PROTOCOL_2026-10-15.md in revise_plan/external_mixed_cancer_pilot).

run_dataset(X, y, name) applies the fixed protocol to one dataset for both step-1 panels (25 proteins and the
development-only 26 = 25 + MSMB) and returns a summary dict. X: patients x features (gene symbols, log2 scale),
y: study-cancer label per patient. Nothing here is tuned to a dataset.
"""
from __future__ import annotations
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = PROJECT_ROOT
PANELS = ROOT / "revise_plan/v5_seed52_common_split/panels/panel_memberships_v5.csv"
OUT = ROOT / "revise_plan/external_mixed_cancer_pilot"
N_SPLITS, TEST_SIZE, SPLIT_SEED, BOOT_SEED, B = 50, 0.25, 20261001, 20261003, 2000
MIN_PATIENTS, MAX_MISSING = 20, 0.20

spec = importlib.util.spec_from_file_location("s76", ROOT / "revise_plan/scripts/76_oct_v6_confidence_comparator_2026-10-03.py")
s76 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s76)


def load_panels():
    panels = {}
    for r in csv.DictReader(open(PANELS, encoding="utf-8-sig")):
        v = list(r.values())
        panels.setdefault(v[0].strip(), []).append(v[1].strip())
    mc25 = panels["multiclass_25"]
    ds = {k.replace("single_", ""): v for k, v in panels.items()
          if k.startswith("single_") and k != "single_pooled_unique"}
    ds = {("DLBCL" if k == "LYMPH" else k): v for k, v in ds.items()}
    return {"25": mc25, "26": mc25 + ["MSMB"]}, ds


def all_panel_genes():
    mc, ds = load_panels()
    return sorted(set(mc["26"]) | {g for v in ds.values() for g in v})


def _fit_predict(Xtr, ytr, Xte, multinomial, C, balanced):
    med = np.nanmedian(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, class_weight="balanced" if balanced else None, max_iter=5000)
    m.fit(sc.transform(Xtr), ytr)
    return m, m.predict_proba(sc.transform(Xte))


def _boot(correct, conf, second):
    n = len(correct)
    gate, keep, k, _ = s76.matched(conf, second, correct)
    wg, wc = int((~correct[gate]).sum()), int((~correct[keep]).sum())
    rng = np.random.default_rng(BOOT_SEED)
    rel, dacc, gain = [], [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        cf, sd, cr = conf[i], second[i], correct[i]
        g = sd >= 0.50
        if g.sum() == 0:
            continue
        kp = s76.topk_mask(cf, int(g.sum()))
        wgb, wcb = (~cr[g]).sum(), (~cr[kp]).sum()
        rel.append(1 - wgb / wcb if wcb else np.nan)
        dacc.append(cr[g].mean() - cr[kp].mean())
        gain.append(cr[g].mean() - cr.mean())
    rel, dacc, gain = map(np.array, (rel, dacc, gain))
    p = float(min(1.0, 2 * min(np.mean(dacc <= 0), np.mean(dacc >= 0))))
    return {"wrong_stricter": wc, "wrong_step2": wg, "shared_accepted": int((gate & keep).sum()),
            "relative_reduction": (1 - wg / wc) if wc else None,
            "rel_ci": [float(np.nanpercentile(rel, 2.5)), float(np.nanpercentile(rel, 97.5))],
            "accuracy_diff_points": 100 * float(correct[gate].mean() - correct[keep].mean()),
            "acc_ci": [100 * float(np.percentile(dacc, 2.5)), 100 * float(np.percentile(dacc, 97.5))],
            "p_two_sided": p,
            "gain_points": 100 * float(correct[second >= 0.5].mean() - correct.mean()),
            "gain_ci": [100 * float(np.percentile(gain, 2.5)), 100 * float(np.percentile(gain, 97.5))]}


def run_dataset(X: pd.DataFrame, y: pd.Series, name: str) -> dict:
    mc, ds = load_panels()
    counts = y.value_counts()
    keep_c = sorted(counts[counts >= MIN_PATIENTS].index)
    sel = y.isin(keep_c)
    X, y = X.loc[sel], y.loc[sel]
    miss = X.isna().mean()
    usable = [g for g in X.columns if miss[g] <= MAX_MISSING]
    yv = y.to_numpy()
    cancers = keep_c
    ds_use = {c: [g for g in ds[c] if g in usable] for c in cancers}
    mc_use = {k: [g for g in v if g in usable] for k, v in mc.items()}
    res = {"dataset": name, "n_patients": int(len(yv)), "n_per_cancer": counts.loc[keep_c].astype(int).to_dict(),
           "excluded_cancers_lt20": counts[counts < MIN_PATIENTS].astype(int).to_dict(),
           "coverage": {"multiclass_25": f"{len(mc_use['25'])}/25", "multiclass_26": f"{len(mc_use['26'])}/26",
                        **{c: f"{len(ds_use[c])}/{len(ds[c])}" for c in cancers}},
           "missing_features": sorted(set(all_panel_genes()) - set(usable))}
    empty = [c for c in cancers if not ds_use[c]]
    if empty:
        res["error"] = f"no disease-specific features for {empty}"
        return res
    n = len(yv)
    sss = list(StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SPLIT_SEED).split(np.zeros(n), yv))
    # step 2 (independent of the step-1 panel)
    P2, cnt = np.zeros((n, len(cancers))), np.zeros(n)
    for tr, te in sss:
        for j, c in enumerate(cancers):
            f = ds_use[c]
            _, pr = _fit_predict(X.iloc[tr][f].to_numpy(float), (yv[tr] == c).astype(int),
                                 X.iloc[te][f].to_numpy(float), False, 1.0, False)
            P2[te, j] += pr[:, 1]
        cnt[te] += 1
    ok = cnt > 0
    P2 = P2 / np.maximum(cnt, 1)[:, None]
    t = np.array([cancers.index(c) for c in yv])
    per_patient = pd.DataFrame({"patient": y.index, "true": yv})
    for key in ("25", "26"):
        f = mc_use[key]
        P1 = np.zeros((n, len(cancers)))
        for tr, te in sss:
            m, pr = _fit_predict(X.iloc[tr][f].to_numpy(float), yv[tr], X.iloc[te][f].to_numpy(float), True, 0.1, True)
            P1[te] += pr[:, [list(m.classes_).index(c) for c in cancers]]
        P1 = P1 / np.maximum(cnt, 1)[:, None]
        top1 = P1.argmax(1)
        correct = (top1 == t)[ok]
        conf, second = P1.max(1)[ok], P2[np.arange(n), top1][ok]
        order = (-P1).argsort(1)
        r = {"step1_accuracy": float(correct.mean()),
             "step1_balanced_accuracy": float(balanced_accuracy_score(t[ok], top1[ok])),
             "top3": float(np.mean([t[i] in order[i, :3] for i in np.where(ok)[0]]))}
        for thr in (0.50, 0.80):
            acc = second >= thr
            r[str(thr)] = {"accepted": int(acc.sum()), "accepted_share": float(acc.mean()),
                           "accuracy_accepted": float(correct[acc].mean()) if acc.any() else None,
                           "wrong_left_undecided": int((~correct & ~acc).sum()), "wrong_total": int((~correct).sum()),
                           "correct_left_undecided": int((correct & ~acc).sum())}
        r["comparison"] = _boot(correct, conf, second)
        res[f"panel_{key}"] = r
        per_patient[f"pred_{key}"] = [cancers[i] for i in top1]
        per_patient[f"conf_{key}"] = P1.max(1)
        per_patient[f"second_{key}"] = P2[np.arange(n), top1]
    per_patient.to_csv(OUT / f"per_patient_{name}.csv", index=False)
    return res
