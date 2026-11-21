#!/usr/bin/env python3
"""
v5.1: the two-step rule evaluated as the yes/no test it is (Sol audit F2, F4, F5,
F7, F10), at three levels, with patient-grouped external predictions (F1).

FIXED BEFORE RUNNING (2026-10-01, changelog MC-20261001-05):
  Rules (unchanged from scripts 62-64):
    step 1   plasma/TCGA: the multiclass model assigns this cancer (top-1);
             external: 25-protein marker-set model >= 0.50
    step 2   disease-specific model of this cancer >= threshold;
             plasma/TCGA: 0.50 (primary) and 0.80; external: 0.80 (Table S10)
    two-step positive only when step 1 and step 2 are both positive.
  Per cancer (one cancer against the rest, or against non-malignant tissue
  externally) and rule: TP, FN, TN, FP; sensitivity and specificity with 95%
  Wilson intervals; balanced accuracy; LR+ and LR- (0.5 continuity correction
  when a cell is zero; flagged).
  Two-step versus each single step, on the same patients:
    false positives removed and true positives lost;
    break-even threshold probability pt* = TP lost / (TP lost + FP removed):
      the two-step rule has the higher net benefit for every threshold above pt*;
    net-benefit difference (two-step minus step) at pt = 0.50 and 0.80, with 95%
      percentile intervals from 2,000 patient-level bootstrap resamples (seed
      20261001; resampling unit = patient, so paired specimens stay together).
  McNemar tests are not used: under the AND rule every discordant pair moves in
  one direction by construction.
  Decision-curve data: net benefit of each yes/no rule for pt = 0.01-0.99.
  No summary pooled across cancers is used as evidence; counts of cancers are
  descriptive.
Inputs
  plasma: locked multiclass test probabilities (as script 62) and the fixed
          disease-specific models refitted by script 62's function;
  external: script-42 predictions, replaced by the patient-grouped reruns for
          LUNGC (script 68), CRC and PRC (script 69);
  TCGA: script-64 probabilities.
Output: revise_plan/two_step_binary_rule/
"""
from __future__ import annotations
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))

from pathlib import Path
import importlib.util
import math

import numpy as np
import pandas as pd

R = (PROJECT_ROOT / "revise_plan")
V5 = R / "v5_seed52_common_split"
OUT = R / "two_step_binary_rule"
SEED, NBOOT = 20261001, 2000
PT_REPORT = (0.50, 0.80)
TOL = 1e-9
PT_CURVE = np.round(np.arange(0.01, 0.991, 0.01), 2)
EXT = ["AML", "BRC", "CLL", "CRC", "CVX", "DLBCL", "ENDC", "GLIOM", "LUNGC", "MYEL", "OVC", "PRC"]
REGROUPED = {"LUNGC": V5 / "external_lung_patient_grouped/{v}/LUNGC/result/locked25_grouped_predictions.csv",
             "CRC": V5 / "external_patient_grouped/{v}/CRC/result/locked25_grouped_predictions.csv",
             "PRC": V5 / "external_patient_grouped/{v}/PRC/result/locked25_grouped_predictions.csv"}
PATIENT_META = {
    "LUNGC": (R / "external_validation_roadmap/cptac_icpc_luad_protein/data/sample_metadata_patient_grouped.csv", "patient_id"),
    "CRC": (V5 / "external_patient_grouped/multiclass_25/CRC/data/sample_metadata_binary_patient.csv", "patient_id"),
    "PRC": (V5 / "external_patient_grouped/multiclass_25/PRC/data/sample_metadata_binary_patient.csv", "patient_id")}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def counts(y, call):
    return (int((call & y).sum()), int((~call & y).sum()), int((~call & ~y).sum()), int((call & ~y).sum()))


def nb(tp, fp, n, pt):
    return tp / n - fp / n * pt / (1 - pt)


def rule_row(y, call):
    tp, fn, tn, fp = counts(y, call)
    zero = min(tp, fn, tn, fp) == 0
    a, b, c, d = (tp, fn, tn, fp) if not zero else (tp + .5, fn + .5, tn + .5, fp + .5)
    se, sp = tp / (tp + fn), tn / (tn + fp)
    sel, seh = wilson(tp, tp + fn); spl, sph = wilson(tn, tn + fp)
    return dict(tp=tp, fn=fn, tn=tn, fp=fp, sensitivity=se, sens_lo=sel, sens_hi=seh,
                specificity=sp, spec_lo=spl, spec_hi=sph, balanced_accuracy=(se + sp) / 2,
                lr_pos=(a / (a + b)) / (d / (c + d)), lr_neg=(b / (a + b)) / (c / (c + d)),
                lr_continuity_corrected=zero)


def compare(y, groups, c_step, c_two, rng):
    n = len(y)
    tp_s, _, _, fp_s = counts(y, c_step); tp_t, _, _, fp_t = counts(y, c_two)
    lost, removed = tp_s - tp_t, fp_s - fp_t
    out = dict(tp_lost=lost, fp_removed=removed,
               breakeven_pt=(lost / (lost + removed)) if (lost + removed) else np.nan)
    for pt in PT_REPORT:
        out[f"nb_diff_{pt:.2f}"] = nb(tp_t, fp_t, n, pt) - nb(tp_s, fp_s, n, pt)
    # patient-level bootstrap of the paired difference
    ug, inv = np.unique(groups, return_inverse=True)
    d_tp = (c_two & y).astype(float) - (c_step & y).astype(float)
    d_fp = (c_two & ~y).astype(float) - (c_step & ~y).astype(float)
    g_tp = np.bincount(inv, d_tp); g_fp = np.bincount(inv, d_fp); g_n = np.bincount(inv)
    boots = {pt: [] for pt in PT_REPORT}
    for _ in range(NBOOT):
        s = rng.integers(0, len(ug), len(ug))
        N = g_n[s].sum(); T = g_tp[s].sum(); F = g_fp[s].sum()
        for pt in PT_REPORT:
            boots[pt].append(T / N - F / N * pt / (1 - pt))
    for pt in PT_REPORT:
        lo, hi = np.percentile(boots[pt], [2.5, 97.5])
        out[f"nb_diff_{pt:.2f}_lo"], out[f"nb_diff_{pt:.2f}_hi"] = lo, hi
    return out


def curve(level, cancer, variant, y, calls):
    n = len(y); rows = []
    for name, c in calls.items():
        tp, _, _, fp = counts(y, c)
        for pt in PT_CURVE:
            rows.append(dict(level=level, cancer=cancer, variant=variant, rule=name, pt=pt,
                             net_benefit=nb(tp, fp, n, pt)))
    treat_all = y.mean()
    for pt in PT_CURVE:
        rows.append(dict(level=level, cancer=cancer, variant=variant, rule="treat_all", pt=pt,
                         net_benefit=treat_all - (1 - treat_all) * pt / (1 - pt)))
    return rows


def evaluate(level, cancer, variant, y, groups, c1, c2, rng, rows, comps, curves):
    y = np.asarray(y, bool); c1 = np.asarray(c1, bool); c2 = np.asarray(c2, bool)
    c12 = c1 & c2
    base = dict(level=level, cancer=cancer, variant=variant, n=len(y), cases=int(y.sum()))
    for name, c in (("step1", c1), ("step2", c2), ("two_step", c12)):
        rows.append({**base, "rule": name, **rule_row(y, c)})
    for ref, c in (("step1", c1), ("step2", c2)):
        comps.append({**base, "compared_with": ref, **compare(y, np.asarray(groups), c, c12, rng)})
    curves += curve(level, cancer, variant, y, {"step1": c1, "step2": c2, "two_step": c12})


def ext_scores(variant, cancer):
    p = (Path(str(REGROUPED[cancer]).format(v=variant)) if cancer in REGROUPED
         else V5 / "external" / variant / cancer / "result/locked25_grouped_predictions.csv")
    df = pd.read_csv(p)
    return df.groupby("sample_id").agg(y=("true_label", "first"), p=("predicted_probability_tumor", "mean"))


def plasma_frames():
    spec = importlib.util.spec_from_file_location(
        "s62", R / "scripts/62_oct_v5_assign_confirm_agreement_2026-09-30.py")
    s62 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s62)
    ds = s62.disease_specific_probs()
    mc = pd.read_csv(s62.MC_PROBS)
    d = mc.merge(ds.drop(columns="true_class"), on="Sample_ID")
    classes = [c.replace("score_", "") for c in d.columns if c.startswith("score_")]
    d["assigned"] = d[[f"score_{c}" for c in classes]].to_numpy().argmax(1)
    d["assigned"] = [classes[i] for i in d.assigned]
    return d, classes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, comps, curves = [], [], []

    d, classes = plasma_frames()
    for c in classes:
        y = d.true_class == c
        for thr in (0.50, 0.80):
            evaluate("plasma", "DLBCL" if c == "LYMPH" else c, f"step2>={thr:.2f}", y,
                     d.Sample_ID, d.assigned == c, d[f"ds_{c}"] >= thr, rng, rows, comps, curves)

    for c in EXT:
        a = ext_scores("multiclass_25", c); b = ext_scores("v5_disease_specific", c).loc[a.index]
        assert (a.y == b.y).all()
        if c in PATIENT_META:
            f, col = PATIENT_META[c]
            m = pd.read_csv(f).set_index("sample_id")[col]
            groups = m.loc[a.index].astype(str).to_numpy()
        else:
            groups = a.index.to_numpy()
        evaluate("external", c, "step2>=0.80", a.y.values == 1, groups,
                 a.p.values >= 0.50, b.p.values >= 0.80, rng, rows, comps, curves)

    t = R / "tcga_two_step/results"
    p1 = pd.read_csv(t / "tcga_step1_probabilities.csv", index_col="patient")
    p2 = pd.read_csv(t / "tcga_step2_probabilities.csv", index_col="patient").loc[p1.index]
    cancers = [c for c in p1.columns if c != "true"]
    assigned = p1[cancers].idxmax(axis=1)
    for c in cancers:
        for thr in (0.50, 0.80):
            evaluate("TCGA", c, f"step2>={thr:.2f}", (p1.true == c).values, p1.index.to_numpy(),
                     (assigned == c).values, (p2[c] >= thr).values, rng, rows, comps, curves)

    R1 = pd.DataFrame(rows); C = pd.DataFrame(comps); K = pd.DataFrame(curves)
    R1.to_csv(OUT / "binary_rule_metrics.csv", index=False)
    C.to_csv(OUT / "binary_rule_paired_comparisons.csv", index=False)
    K.to_csv(OUT / "binary_rule_decision_curves.csv", index=False)

    summ = []
    for (lev, var, ref), g in C.groupby(["level", "variant", "compared_with"]):
        s = dict(level=lev, variant=var, compared_with=ref, n_cancers=len(g),
                 fp_removed=int(g.fp_removed.sum()), tp_lost=int(g.tp_lost.sum()),
                 median_breakeven_pt=float(g.breakeven_pt.median()))
        for pt in PT_REPORT:
            k = f"nb_diff_{pt:.2f}"
            # tolerance: bootstrap bounds of 3e-18 are floating-point zeros
            s[f"{k}_ci_above_0"] = int((g[f"{k}_lo"] > TOL).sum())
            s[f"{k}_ci_below_0"] = int((g[f"{k}_hi"] < -TOL).sum())
            s[f"{k}_point_above_0"] = int((g[k] > TOL).sum())
        summ.append(s)
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "binary_rule_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
