"""Downstream effects of the development-only disease-specific panels (script 91), part 1.

A. Internal test (413): per-cancer reported estimator (unpenalised logistic regression on the development set,
   Appendix S1.7) for published and development-only panels: AUC with 2,000-resample bootstrap interval,
   sensitivity/specificity at 0.50 and 0.80, ECE and Brier score.
B. Internal two-step rule (locked 25-protein step 1) and the stricter-cutoff comparison (script 85 recipe).
C. TCGA, original Xena legacy data (script 64 design), 25- and 26-protein step 1, with the
   development-only disease-specific lists in step 2.
Output: revise_plan/pilot_ds_dev_only_size/
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
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = PROJECT_ROOT
OUT = ROOT / "revise_plan/pilot_ds_dev_only_size"
PANELS = {"published": ROOT / "revise_plan/v5_seed52_common_split/panels/panel_memberships_v5.csv",
          "dev_only": OUT / "panel_memberships_devonly.csv"}
MC_PROBS = ROOT / "figures_nature_v3/production/01_study_cohort/source_data/locked_test_probability_summary.csv"
LABEL = {"DLBCL": "LYMPH"}


def load(n, f):
    s = importlib.util.spec_from_file_location(n, ROOT / "revise_plan/scripts" / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


core = load("core", "85_oct_v65_mixed_cancer_core_2026-10-15.py")
s64 = load("s64", "64_oct_v5_tcga_assign_confirm_2026-10-01.py")


def ds_panels(path):
    p = pd.read_csv(path)
    return {r.replace("single_", ""): list(p[p.panel == r].protein) for r in p.panel.unique()
            if r.startswith("single_") and r != "single_pooled_unique"}


def ece(y, p, bins=10):
    e, n = 0.0, len(y)
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if s.any():
            e += s.sum() / n * abs(y[s].mean() - p[s].mean())
    return float(e)


def internal():
    df = pd.read_csv(ROOT / "data/processed/filtered_pancancer_data.csv")
    proteins = [c for c in df.columns if c not in ("Sample_ID", "Cancer", "protein_count")]
    y = df["Cancer"].to_numpy()
    tr, te = train_test_split(np.arange(len(df)), test_size=0.3, random_state=52, stratify=y)
    imp = KNNImputer(n_neighbors=5).fit(df.iloc[tr][proteins])
    Xtr = pd.DataFrame(imp.transform(df.iloc[tr][proteins]), columns=proteins)
    Xte = pd.DataFrame(imp.transform(df.iloc[te][proteins]), columns=proteins)
    rng = np.random.default_rng(20261016)
    boot = [rng.integers(0, len(te), len(te)) for _ in range(2000)]
    rows, probs = [], {}
    for version, path in PANELS.items():
        probs[version] = {}
        for c, feats in ds_panels(path).items():
            lab = LABEL.get(c, c)
            sc = StandardScaler().fit(Xtr[feats])
            m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, random_state=52)
            m.fit(sc.transform(Xtr[feats]), (y[tr] == lab).astype(int))
            p = m.predict_proba(sc.transform(Xte[feats]))[:, 1]
            yt = (y[te] == lab).astype(int)
            aucs = [roc_auc_score(yt[i], p[i]) for i in boot if 0 < yt[i].sum() < len(i)]
            r = {"version": version, "cancer": c, "size": len(feats), "auc": roc_auc_score(yt, p),
                 "auc_lo": float(np.percentile(aucs, 2.5)), "auc_hi": float(np.percentile(aucs, 97.5)),
                 "ece": ece(yt, p), "brier": brier_score_loss(yt, p)}
            for thr in (0.5, 0.8):
                pr = p >= thr
                r[f"sens_{thr}"] = float(pr[yt == 1].mean())
                r[f"spec_{thr}"] = float((~pr)[yt == 0].mean())
            rows.append(r)
            probs[version][lab] = p
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "internal_per_cancer_published_vs_devonly.csv", index=False)
    pub = t[t.version == "published"]
    assert abs(pub.auc.mean() - 0.9412666845) < 1e-6, pub.auc.mean()  # reproduces Table S9
    mc = pd.read_csv(MC_PROBS).set_index("Sample_ID")
    ids = df.iloc[te]["Sample_ID"].to_numpy()
    classes = [c.replace("score_", "") for c in mc.columns if c.startswith("score_")]
    P = mc.loc[ids, [f"score_{c}" for c in classes]].to_numpy()
    top1 = np.array(classes)[P.argmax(1)]
    correct = top1 == y[te]
    two = {}
    for version in PANELS:
        second = np.array([probs[version][c][i] for i, c in enumerate(top1)])
        res = {}
        for thr in (0.5, 0.8):
            acc = second >= thr
            res[str(thr)] = {"accepted": int(acc.sum()), "accuracy_accepted": float(correct[acc].mean()),
                             "wrong_left_undecided": int((~correct & ~acc).sum()),
                             "correct_left_undecided": int((correct & ~acc).sum())}
        res["comparison"] = core._boot(correct, P.max(1), second)
        two[version] = res
    return t, two


def tcga_cache_with_new_genes():
    """Copy of the published TCGA cache plus the genes added by the development-only panels (same hub and
    datasets as script 64; probe-name requests return one entry per requested probe, so no misalignment)."""
    import xenaPython as xena
    merged = OUT / "tcga_panel_genes_hiseqv2_plus_devonly.csv"
    if merged.exists():
        return merged
    base = pd.read_csv(s64.CACHE, index_col="sample")
    core.PANELS = PANELS["dev_only"]
    extra = sorted(set(core.all_panel_genes()) - set(base.columns))
    alias = {"ELOA": "TCEB3"}
    frames = []
    for cohort in s64.COHORTS:
        ds = f"TCGA.{cohort}.sampleMap/HiSeqV2"
        samples = list(base[base.cohort == cohort].index)
        probes = [alias.get(g, g) for g in extra]
        vals = xena.dataset_probe_values(s64.HUB, ds, samples, probes)[1]
        assert len(vals) == len(extra)
        frames.append(pd.DataFrame({g: [np.nan if v in ("NaN", None) else float(v) for v in col]
                                    if col is not None else [np.nan] * len(samples)
                                    for g, col in zip(extra, vals)}, index=samples))
    add = pd.concat(frames)
    out = base.join(add)
    out.to_csv(merged, index_label="sample")
    return merged


def tcga():
    s64.CACHE = tcga_cache_with_new_genes()
    out = {}
    for version, path in PANELS.items():
        core.PANELS = path
        X, y = s64.load_tcga(set(core.all_panel_genes()))
        r = core.run_dataset(X, y, f"tcga_xena_{version}")
        out[version] = {k: {"step1_accuracy": r[k]["step1_accuracy"], "accepted": r[k]["0.5"]["accepted"],
                            "accuracy_accepted": r[k]["0.5"]["accuracy_accepted"], "comparison": r[k]["comparison"]}
                        for k in ("panel_25", "panel_26")}
        out[version]["coverage"] = r["coverage"]
    return out


def main():
    t, two = internal()
    tc = tcga()
    summary = {"internal_mean": {v: {m: float(t[t.version == v][m].mean()) for m in ("auc", "sens_0.5", "spec_0.5", "sens_0.8", "spec_0.8", "ece", "brier")} for v in PANELS},
               "internal_two_step": two, "tcga_xena": tc}
    json.dump(summary, open(OUT / "summary_downstream_internal_tcga.json", "w"), indent=1, default=float)
    print(t.pivot(index="cancer", columns="version", values=["size", "auc", "sens_0.5", "spec_0.5"]).round(3).to_string())
    print(json.dumps(summary["internal_mean"], indent=1, default=float))
    for v in PANELS:
        c = two[v]["comparison"]
        print(v, "internal two-step: accepted", two[v]["0.5"]["accepted"], "acc %.3f" % two[v]["0.5"]["accuracy_accepted"],
              "| stricter %d vs step2 %d (rel %.3f, p %.3f)" % (c["wrong_stricter"], c["wrong_step2"], c["relative_reduction"], c["p_two_sided"]))
        for k in ("panel_25", "panel_26"):
            x = tc[v][k]; c = x["comparison"]
            print("   TCGA", k, "accepted", x["accepted"], "acc %.3f" % x["accuracy_accepted"],
                  "| stricter %d vs step2 %d (rel %.3f, %.3f to %.3f, p %.3f)" % (c["wrong_stricter"], c["wrong_step2"], c["relative_reduction"], c["rel_ci"][0], c["rel_ci"][1], c["p_two_sided"]))


if __name__ == "__main__":
    main()
