"""Downstream effects of the development-only disease-specific panels (script 91), part 5: per-cancer net benefit of
the two-step call against step 1 alone, internal test and TCGA (original Xena processing), 25-protein step 1.

Per cancer c: step-1 call = predicted type is c; two-step call = predicted c and step-2 probability of c >= 0.50.
TP lost = patients with c whose predicted c was not accepted; FP removed = patients without c whose predicted c
was not accepted. Net-benefit difference per patient at decision threshold pt:
    dNB = (FP removed * pt / (1 - pt) - TP lost) / N
with 95% intervals from 2,000 patient bootstrap resamples (seed 20261003). The published panels must reproduce
the published TP lost / FP removed counts (two_step_binary_rule/binary_rule_paired_comparisons.csv).
Output: revise_plan/pilot_ds_dev_only_size/net_benefit_published_vs_devonly.csv
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = PROJECT_ROOT
OUT = ROOT / "revise_plan/pilot_ds_dev_only_size"
PUBNB = ROOT / "revise_plan/two_step_binary_rule/binary_rule_paired_comparisons.csv"
PILOT = ROOT / "revise_plan/external_mixed_cancer_pilot"
spec = importlib.util.spec_from_file_location("s92", ROOT / "revise_plan/scripts/92_oct_v65_ds_devonly_downstream_internal_tcga_2026-10-16.py")
s92 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s92)


def nb_rows(level, version, true, pred, second, cancers):
    n = len(true)
    rng = np.random.default_rng(20261003)
    boot = [rng.integers(0, n, n) for _ in range(2000)]
    rows = []
    for c in cancers:
        rejected = (pred == c) & (second < 0.5)
        tp_lost = rejected & (true == c)
        fp_rem = rejected & (true != c)
        r = {"level": level, "version": version, "cancer": c, "cases": int((true == c).sum()),
             "tp_lost": int(tp_lost.sum()), "fp_removed": int(fp_rem.sum())}
        for pt in (0.5, 0.8):
            w = pt / (1 - pt)
            d = (fp_rem.sum() * w - tp_lost.sum()) / n
            bs = [(fp_rem[i].sum() * w - tp_lost[i].sum()) / n for i in boot]
            r[f"nb_diff_{pt}"] = 100 * d
            r[f"nb_lo_{pt}"], r[f"nb_hi_{pt}"] = 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)
        rows.append(r)
    return rows


def main():
    # internal: recompute both versions with the script-92 estimator
    df = pd.read_csv(ROOT / "data/processed/filtered_pancancer_data.csv")
    from sklearn.impute import KNNImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    proteins = [c for c in df.columns if c not in ("Sample_ID", "Cancer", "protein_count")]
    y = df["Cancer"].to_numpy()
    tr, te = train_test_split(np.arange(len(df)), test_size=0.3, random_state=52, stratify=y)
    imp = KNNImputer(n_neighbors=5).fit(df.iloc[tr][proteins])
    Xtr = pd.DataFrame(imp.transform(df.iloc[tr][proteins]), columns=proteins)
    Xte = pd.DataFrame(imp.transform(df.iloc[te][proteins]), columns=proteins)
    mc = pd.read_csv(s92.MC_PROBS).set_index("Sample_ID")
    classes = [c.replace("score_", "") for c in mc.columns if c.startswith("score_")]
    P = mc.loc[df.iloc[te]["Sample_ID"].to_numpy(), [f"score_{c}" for c in classes]].to_numpy()
    pred = np.array(classes)[P.argmax(1)]
    true = y[te]
    rows = []
    for version, path in s92.PANELS.items():
        probs = {}
        for c, feats in s92.ds_panels(path).items():
            lab = s92.LABEL.get(c, c)
            sc = StandardScaler().fit(Xtr[feats])
            m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, random_state=52)
            m.fit(sc.transform(Xtr[feats]), (y[tr] == lab).astype(int))
            probs[lab] = m.predict_proba(sc.transform(Xte[feats]))[:, 1]
        second = np.array([probs[c][i] for i, c in enumerate(pred)])
        rows += nb_rows("plasma", version, true, pred, second, sorted(set(classes)))
    # TCGA: per-patient files written by script 92 (core.run_dataset)
    for version in ("published", "dev_only"):
        d = pd.read_csv(PILOT / f"per_patient_tcga_xena_{version}.csv")
        rows += nb_rows("TCGA", version, d.true.to_numpy(), d.pred_25.to_numpy(), d.second_25.to_numpy(), sorted(d.true.unique()))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "net_benefit_published_vs_devonly.csv", index=False)
    # reproduce published counts
    pub = pd.read_csv(PUBNB)
    pub = pub[(pub.variant == "step2>=0.50") & (pub.compared_with == "step1")]
    for _, r in t[t.version == "published"].iterrows():
        q = pub[(pub.level.str.lower() == r.level.lower()) & (pub.cancer.replace({"LYMPH": "DLBCL"}) == r.cancer.replace("LYMPH", "DLBCL"))]
        if len(q):
            assert (int(q.tp_lost.iloc[0]), int(q.fp_removed.iloc[0])) == (r.tp_lost, r.fp_removed), (r.level, r.cancer, q[["tp_lost", "fp_removed"]].values, r.tp_lost, r.fp_removed)
    for (lvl, v), g in t.groupby(["level", "version"]):
        print(f"{lvl:6s} {v:9s}: at 0.80 higher than step 1 for {(g['nb_diff_0.8'] > 0).sum()} of {len(g)} "
              f"(interval above zero {(g['nb_lo_0.8'] > 0).sum()}, below zero {(g['nb_hi_0.8'] < 0).sum()}); "
              f"at 0.50 interval above zero {(g['nb_lo_0.5'] > 0).sum()}, below zero {(g['nb_hi_0.5'] < 0).sum()}")
    print(t.pivot_table(index=["level", "cancer"], columns="version", values="nb_diff_0.8").round(2).to_string())


if __name__ == "__main__":
    main()
