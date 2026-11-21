"""v7 B5: internal two-step rule with the final panels (panels/panel_memberships_v7.csv).

Step 1: 26-protein multiclass ridge model (C = 0.1, balanced), trained on the 962 development patients.
Step 2: development-only disease-specific panels, unpenalised logistic regression on the development set.
Both scored once on the 413 internal-test patients (seed-52 70/30 split, KNN imputation fitted on development).
Outputs (results/B5_*): per-patient table; summary JSON with multiclass metrics (1,000 bootstrap, seed 52024),
two-step counts at 0.50 and 0.80, accuracy gain and stricter-cutoff comparison (2,000 paired bootstrap,
seed 20261003), top-3 retention of undecided patients, per-cancer net benefit at 0.50/0.80.
Reuses the verified functions of revise_plan scripts 84, 85 and 96.
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = PROJECT_ROOT
V7 = ROOT / "v7_plan"
PANELS = V7 / "panels/panel_memberships_v7.csv"
OUT = V7 / "results"


def load(n, f):
    s = importlib.util.spec_from_file_location(n, ROOT / "revise_plan/scripts" / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


s84 = load("s84", "84_oct_v65_pilot_26_protein_panel_2026-10-15.py")
core = load("core", "85_oct_v65_mixed_cancer_core_2026-10-15.py")
s96 = load("s96", "96_oct_v65_ds_devonly_net_benefit_2026-10-16.py")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pan = pd.read_csv(PANELS)
    mc = list(pan[pan.panel == "multiclass_26"].protein)
    ids, y, classes, P = s84.multiclass_test_probs(mc)
    pred = np.array(classes)[P.argmax(1)]
    metrics = s84.metrics(y, pred, P, classes)
    ci = s84.boot_ci(y, P, classes)
    df = pd.read_csv(ROOT / "data/processed/filtered_pancancer_data.csv")
    proteins = [c for c in df.columns if c not in ("Sample_ID", "Cancer", "protein_count")]
    yall = df["Cancer"].to_numpy()
    tr, te = train_test_split(np.arange(len(df)), test_size=0.3, random_state=52, stratify=yall)
    assert (df.iloc[te]["Sample_ID"].to_numpy() == ids).all()
    imp = KNNImputer(n_neighbors=5).fit(df.iloc[tr][proteins])
    Xtr = pd.DataFrame(imp.transform(df.iloc[tr][proteins]), columns=proteins)
    Xte = pd.DataFrame(imp.transform(df.iloc[te][proteins]), columns=proteins)
    probs = {}
    for key in pan.panel.unique():
        if not key.startswith("single_") or key == "single_pooled_unique":
            continue
        c = key.replace("single_", "")
        lab = "LYMPH" if c == "DLBCL" else c
        feats = list(pan[pan.panel == key].protein)
        sc = StandardScaler().fit(Xtr[feats])
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, random_state=52)
        m.fit(sc.transform(Xtr[feats]), (yall[tr] == lab).astype(int))
        probs[lab] = m.predict_proba(sc.transform(Xte[feats]))[:, 1]
    second = np.array([probs[c][i] for i, c in enumerate(pred)])
    correct = pred == y
    order = (-P).argsort(1)
    t = np.array([classes.index(c) for c in y])
    in_top3 = np.array([t[i] in order[i, :3] for i in range(len(t))])
    res = {"panels_sha256": __import__("hashlib").sha256(PANELS.read_bytes()).hexdigest(),
           "multiclass": {"metrics": metrics, "ci": ci, "top2": float(np.mean([t[i] in order[i, :2] for i in range(len(t))])),
                          "top3": float(in_top3.mean())}}
    for thr in (0.5, 0.8):
        acc = second >= thr
        res[f"two_step_{thr}"] = {"accepted": int(acc.sum()), "undecided": int((~acc).sum()), "accepted_share": float(acc.mean()),
                                  "correct_accepted": int((correct & acc).sum()), "accuracy_accepted": float(correct[acc].mean()),
                                  "wrong_total": int((~correct).sum()), "wrong_left_undecided": int((~correct & ~acc).sum()),
                                  "correct_left_undecided": int((correct & ~acc).sum()),
                                  "undecided_true_in_top3": int((~acc & in_top3).sum())}
    res["comparison_0.5"] = core._boot(correct, P.max(1), second)
    nb = pd.DataFrame(s96.nb_rows("plasma", "v7", y, pred, second, sorted(set(classes))))
    nb.to_csv(OUT / "B5_net_benefit_per_cancer.csv", index=False)
    res["net_benefit_0.8"] = {"higher_than_step1": int((nb["nb_diff_0.8"] > 0).sum()), "n": len(nb),
                              "interval_above_zero": int((nb["nb_lo_0.8"] > 0).sum()), "interval_below_zero": int((nb["nb_hi_0.8"] < 0).sum())}
    res["net_benefit_0.5"] = {"interval_above_zero": int((nb["nb_lo_0.5"] > 0).sum()), "interval_below_zero": int((nb["nb_hi_0.5"] < 0).sum())}
    pd.DataFrame({"Sample_ID": ids, "true": y, "pred": pred, "conf": P.max(1), "second": second, "correct": correct,
                  "true_in_top3": in_top3}).to_csv(OUT / "B5_internal_per_patient.csv", index=False)
    json.dump(res, open(OUT / "B5_internal_two_step_summary.json", "w"), indent=1, default=float)
    m, a = res["multiclass"]["metrics"], res["two_step_0.5"]
    c = res["comparison_0.5"]
    print(f"26-protein: acc {m['accuracy']:.3f} BA {m['balanced_accuracy']:.3f} F1 {m['macro_f1']:.3f} AUC {m['macro_auc']:.3f} min recall {m['min_class_recall']:.3f}; top3 {res['multiclass']['top3']:.3f}")
    print(f"two-step 0.50: accepted {a['accepted']} ({a['accepted_share']:.3f}), correct {a['correct_accepted']} ({a['accuracy_accepted']:.3f}); "
          f"wrong left undecided {a['wrong_left_undecided']}/{a['wrong_total']}; correct left undecided {a['correct_left_undecided']}; undecided with true in top 3 {a['undecided_true_in_top3']}/{a['undecided']}")
    print(f"gain {c['gain_points']:.1f} pts ({c['gain_ci'][0]:.1f}-{c['gain_ci'][1]:.1f}); stricter {c['wrong_stricter']} vs step2 {c['wrong_step2']}, rel {c['relative_reduction']:.3f} ({c['rel_ci'][0]:.3f} to {c['rel_ci'][1]:.3f}), p {c['p_two_sided']:.3f}")
    print("net benefit:", res["net_benefit_0.8"], res["net_benefit_0.5"])


if __name__ == "__main__":
    main()
