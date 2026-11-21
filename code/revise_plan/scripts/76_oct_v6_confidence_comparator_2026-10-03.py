"""A1 for manuscript v6.0: does the second (disease-specific) model add anything beyond the first
(multiclass) model's own confidence?

Reads saved predictions only (no refits). Two settings:
  internal test   413 patients, second-model cutoff 0.50 (script 62 outputs + locked 25-protein
                  test-set probabilities)
  mixed-cancer    4,665 TCGA patients, second-model cutoff 0.50 (script 64 outputs)

Comparisons, both with paired patient-level bootstrap intervals (2,000 resamples):
  (a) matched coverage: the second model accepts k patients; the confidence-only rule keeps the k
      patients with the highest first-model probability. Difference in accuracy among accepted
      predictions (second model minus confidence-only). In each resample k is recomputed.
  (b) risk-coverage: area under the risk-coverage curve (AURC; mean error rate over all coverages)
      when patients are ranked by first-model confidence versus by the second model's probability.
      Difference = AURC(confidence) - AURC(second model); positive favours the second model.
No combined score is computed (it would be chosen after seeing the data).

Outputs: revise_plan/two_step_binary_rule/confidence_comparator.csv (summary) and
         confidence_comparator_coverage_grid.csv (accuracy at fixed coverages, descriptive).
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = PROJECT_ROOT
OUT = ROOT / "revise_plan/two_step_binary_rule"
B = 2000
SEED = 20261003
CUTOFF = 0.50


def load_internal():
    pp = pd.read_csv(ROOT / "revise_plan/v5_seed52_common_split/internal/assign_confirm/assign_confirm_per_patient.csv")
    mc = pd.read_csv(ROOT / "figures_nature_v3/production/01_study_cohort/source_data/locked_test_probability_summary.csv")
    assert pp.Sample_ID.is_unique and mc.Sample_ID.is_unique
    d = pp.merge(mc, on="Sample_ID", how="inner", suffixes=("", "_mc"), validate="one_to_one")
    assert len(d) == len(pp) == len(mc) == 413
    assert (d.true_class == d.true_class_mc).all(), "true labels disagree"
    sc = [c for c in mc.columns if c.startswith("score_")]
    probs = d[sc].to_numpy(float)
    assert np.allclose(probs.sum(1), 1.0, atol=1e-6), "probabilities do not sum to 1"
    classes = np.array([c.replace("score_", "") for c in sc])
    argmax = classes[probs.argmax(1)]
    assert (argmax == d.top1.to_numpy()).all() and (argmax == d.predicted_class.to_numpy()).all()
    correct = d.correct.astype(bool).to_numpy()
    assert (correct == (d.top1 == d.true_class).to_numpy()).all()
    second = d.ds_prob_top1.to_numpy(float)
    return probs.max(1), second, correct


def load_mixed():
    tp = pd.read_csv(ROOT / "revise_plan/tcga_two_step/results/tcga_per_patient.csv")
    t1 = pd.read_csv(ROOT / "revise_plan/tcga_two_step/results/tcga_step1_probabilities.csv")
    assert tp.patient.is_unique and t1.patient.is_unique
    e = tp.merge(t1, on="patient", how="inner", suffixes=("", "_p1"), validate="one_to_one")
    assert len(e) == len(tp) == len(t1) == 4665
    assert (e["true"] == e["true_p1"]).all(), "true labels disagree"
    cls = [c for c in t1.columns if c not in ("patient", "true")]
    probs = e[cls].to_numpy(float)
    assert np.allclose(probs.sum(1), 1.0, atol=1e-6), "probabilities do not sum to 1"
    argmax = np.array(cls)[probs.argmax(1)]
    assert (argmax == e.assigned.to_numpy()).all()
    correct = e.correct.astype(bool).to_numpy()
    assert (correct == (e.assigned == e["true"]).to_numpy()).all()
    return probs.max(1), e.step2_prob_assigned.to_numpy(float), correct


def topk_mask(score, k):
    order = np.argsort(-score, kind="stable")
    m = np.zeros(len(score), bool)
    m[order[:k]] = True
    return m


def matched(conf, second, correct):
    gate = second >= CUTOFF
    k = int(gate.sum())
    keep = topk_mask(conf, k)
    s = np.sort(conf)[::-1]
    tie = bool(k < len(s) and s[k - 1] == s[k])
    return gate, keep, k, tie


def aurc(score, correct):
    c = correct[np.argsort(-score, kind="stable")]
    risk = 1.0 - np.cumsum(c) / np.arange(1, len(c) + 1)
    return risk.mean()


def analyse(name, conf, second, correct, rng):
    n = len(correct)
    gate, keep, k, tie = matched(conf, second, correct)
    acc_g, acc_c = correct[gate].mean(), correct[keep].mean()
    a_c, a_g = aurc(conf, correct), aurc(second, correct)
    d_match, d_aurc = [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        cf, sd, cr = conf[i], second[i], correct[i]
        g = sd >= CUTOFF
        kb = int(g.sum())
        if kb == 0:
            continue
        kp = topk_mask(cf, kb)
        d_match.append(cr[g].mean() - cr[kp].mean())
        d_aurc.append(aurc(cf, cr) - aurc(sd, cr))
    d_match, d_aurc = np.array(d_match), np.array(d_aurc)
    row = {
        "setting": name, "patients": n, "accepted_k": k, "coverage": k / n,
        "second_model_correct": int(correct[gate].sum()), "confidence_only_correct": int(correct[keep].sum()),
        "overlap_accepted": int((gate & keep).sum()), "boundary_tie": tie,
        "second_model_accuracy": acc_g, "confidence_only_accuracy": acc_c,
        "matched_diff": acc_g - acc_c,
        "matched_diff_lo": np.percentile(d_match, 2.5), "matched_diff_hi": np.percentile(d_match, 97.5),
        "matched_share_diff_le0": float(np.mean(d_match <= 0)),
        "aurc_confidence": a_c, "aurc_second_model": a_g, "aurc_diff": a_c - a_g,
        "aurc_diff_lo": np.percentile(d_aurc, 2.5), "aurc_diff_hi": np.percentile(d_aurc, 97.5),
        "aurc_share_diff_le0": float(np.mean(d_aurc <= 0)), "bootstrap_resamples": len(d_match), "seed": SEED,
    }
    grid = []
    for cov in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        kk = int(round(cov * n))
        grid.append({"setting": name, "coverage": cov, "accepted": kk,
                     "confidence_accuracy": correct[topk_mask(conf, kk)].mean(),
                     "second_model_accuracy": correct[topk_mask(second, kk)].mean()})
    return row, grid


def main():
    rng = np.random.default_rng(SEED)
    rows, grids = [], []
    for name, loader, exp in (("internal test", load_internal, (243, 221, 218)),
                              ("mixed-cancer external test (TCGA)", load_mixed, (3572, 3522, 3489))):
        conf, second, correct = loader()
        row, grid = analyse(name, conf, second, correct, rng)
        assert (row["accepted_k"], row["second_model_correct"], row["confidence_only_correct"]) == exp, row
        rows.append(row); grids += grid
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "confidence_comparator.csv", index=False)
    pd.DataFrame(grids).to_csv(OUT / "confidence_comparator_coverage_grid.csv", index=False)
    pd.set_option("display.width", 220)
    show = out[["setting", "accepted_k", "second_model_correct", "confidence_only_correct", "overlap_accepted",
                "boundary_tie", "matched_diff", "matched_diff_lo", "matched_diff_hi", "aurc_confidence",
                "aurc_second_model", "aurc_diff", "aurc_diff_lo", "aurc_diff_hi"]]
    print(show.round(4).to_string(index=False))
    print(pd.DataFrame(grids).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
