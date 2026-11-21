"""v7 B1: internal-test description of the final 26-protein multiclass model (413 patients, scored once).
Per-cancer recall and precision, confusion matrix, calibration (top-label expected calibration error with 10 bins,
multiclass Brier score), mean predicted probability by true cancer, and decision curves (one cancer against the
rest: net benefit of treating patients whose predicted probability for that cancer exceeds the threshold, against
treat-all and treat-none), thresholds 0.01-0.99. Per-cancer AUC with 1,000 bootstrap intervals (seed 52024).
Outputs: results/B1_multiclass/
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score

ROOT = PROJECT_ROOT
V7 = ROOT / "v7_plan"
OUT = V7 / "results/B1_multiclass"
spec = importlib.util.spec_from_file_location("s84", ROOT / "revise_plan/scripts/84_oct_v65_pilot_26_protein_panel_2026-10-15.py")
s84 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s84)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pan = pd.read_csv(V7 / "panels/panel_memberships_v7.csv")
    ids, y, classes, P = s84.multiclass_test_probs(list(pan[pan.panel == "multiclass_26"].protein))
    pred = np.array(classes)[P.argmax(1)]
    assert abs((pred == y).mean() - 0.7457627) < 1e-6
    n, K = P.shape
    Y = np.array([[1 if yi == c else 0 for c in classes] for yi in y])
    conf, correct = P.max(1), pred == y
    edges = np.linspace(0, 1, 11)
    ece = sum(((conf > lo) & (conf <= hi)).sum() / n * abs(correct[(conf > lo) & (conf <= hi)].mean() - conf[(conf > lo) & (conf <= hi)].mean())
              for lo, hi in zip(edges[:-1], edges[1:]) if ((conf > lo) & (conf <= hi)).any())
    brier = float(((P - Y) ** 2).sum(1).mean())
    rng = np.random.default_rng(52024)
    boot = [rng.integers(0, n, n) for _ in range(1000)]
    per = []
    for k, c in enumerate(classes):
        a = [roc_auc_score(Y[i, k], P[i, k]) for i in boot if 0 < Y[i, k].sum() < n]
        per.append({"cancer": c, "n": int(Y[:, k].sum()), "recall": float(recall_score(y, pred, labels=[c], average=None)[0]),
                    "precision": float(precision_score(y, pred, labels=[c], average=None, zero_division=0)[0]),
                    "auc": float(roc_auc_score(Y[:, k], P[:, k])), "auc_lo": float(np.percentile(a, 2.5)), "auc_hi": float(np.percentile(a, 97.5)),
                    "mean_prob_true_class": float(P[Y[:, k] == 1, k].mean())})
    pd.DataFrame(per).to_csv(OUT / "B1_per_cancer.csv", index=False)
    pd.DataFrame(confusion_matrix(y, pred, labels=classes), index=classes, columns=classes).to_csv(OUT / "B1_confusion_matrix.csv")
    pd.DataFrame([[P[Y[:, j] == 1, k].mean() for k in range(K)] for j in range(K)], index=classes, columns=classes).to_csv(OUT / "B1_mean_probability_by_true_cancer.csv")
    rows = []
    for pt in np.round(np.arange(0.01, 1.0, 0.01), 2):
        w = pt / (1 - pt)
        for k, c in enumerate(classes):
            call = P[:, k] >= pt
            tp, fp = (call & (Y[:, k] == 1)).sum(), (call & (Y[:, k] == 0)).sum()
            prev = Y[:, k].mean()
            rows.append({"threshold": pt, "cancer": c, "nb_model": (tp - fp * w) / n, "nb_treat_all": prev - (1 - prev) * w, "nb_treat_none": 0.0})
    dc = pd.DataFrame(rows)
    dc.to_csv(OUT / "B1_decision_curves.csv", index=False)
    avg = dc.groupby("threshold")[["nb_model", "nb_treat_all"]].mean()
    above = avg[(avg.nb_model > avg.nb_treat_all.clip(lower=0))]
    summary = {"ece_top_label": float(ece), "brier_multiclass": brier,
               "decision_curve_mean_model_above_both_references": [float(above.index.min()), float(above.index.max())] if len(above) else None,
               "weakest_recall": min(per, key=lambda r: r["recall"])}
    json.dump(summary, open(OUT / "B1_summary.json", "w"), indent=1, default=float)
    print(pd.DataFrame(per).round(3).to_string(index=False)); print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
