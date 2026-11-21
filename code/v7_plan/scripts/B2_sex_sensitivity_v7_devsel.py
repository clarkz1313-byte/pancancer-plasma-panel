# v7 B2 corrected (audit SOL-B02): sex-associated proteins selected on the 962 development patients only; outputs results/B2_sex_devsel.
# v7 B2: AUTO-GENERATED fork of revise_plan/scripts/30_sept_sex_sensitivity_2026-09-06.py for the final 26-protein
# multiclass panel (variable name LOCKED_25 kept; it holds 26 proteins). Reproduction check uses the v7 B5 values.
#!/usr/bin/env python3
"""
Sex sensitivity analysis for the locked 25-protein multiclass panel.

WHY
---
The locked 25 contains at least five proteins with strong sex-associated
expression: PAEP (female reproductive secretory), WFDC2/HE4 (clinical ovarian
marker), CGA (glycoprotein hormone alpha subunit), KLK13 (kallikrein family)
and XG (X-linked blood group).

Alvez et al. selected their per-cancer markers against SEX-MATCHED controls
("For cancers consisting of male or female samples exclusively, only samples
from the same sex were used as controls"), which forces markers to be
biologically cancer-specific rather than sex-specific. This work did not do
that: the 12-class model sees all patients at natural class sizes.

That is a harder assignment task, but it opens a specific objection: the model
may be separating (say) prostate from ovarian partly on sex, which a clinician
already knows. This script measures how much of the reported performance
survives when sex cannot help.

DESIGN
------
Pipeline is identical to 02_may_vRSX_seed52_locked_25_lr_l2_reproducer.py:
seed 52, stratified 70/30, KNN(5) imputation fitted on train, standardise,
multinomial LR with L2, C=0.1, class_weight balanced.

  0. JOIN VALIDATION  -- metadata carries no Sample_ID, so sex is joined
     positionally within each cancer group. This is only trustworthy if the
     sex-restricted cancers come out single-sex. The script HARD FAILS if
     prostate is not all-male or the gynaecological cancers not all-female.
  A. baseline           -- reproduce the reported locked-25 numbers
  B. sex-only ceiling   -- what does sex ALONE predict? bounds the concern
  C. within-sex recall  -- baseline model scored inside each sex stratum,
                           where the sex shortcut is unavailable by design
  D. sex association   -- data-defined sex-associated panel proteins
  E. drop those, refit; F. drop the a priori list, for comparison
  G. coefficient audit  -- weight carried by each set

Output: revise_plan/part_b_multiclass/sex_sensitivity_2026-09-06/
"""
from __future__ import annotations
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = PROJECT_ROOT
DATA = ROOT / "data" / "processed" / "filtered_pancancer_data.csv"
META = ROOT / "references" / "alves_supplementary_data" / "supp2.xlsx"
OUT = ROOT / "v7_plan" / "results" / "B2_sex_devsel"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 52
TEST_SIZE = 0.3

LOCKED_25 = ['ADAMTS13', 'ADAMTS15', 'BMP4', 'CCDC80', 'CEACAM5', 'CGA',
             'CNTN1', 'CRTAC1', 'CXCL13', 'CXCL17', 'FCER2', 'FLT3', 'GFAP',
             'GLO1', 'KLK13', 'LTA4H', 'NEFL', 'PAEP', 'PRDX6', 'PSPN',
             'SLAMF7', 'TCL1A', 'TRAF2', 'WFDC2', 'XG', 'MSMB']

# Proteins flagged a priori as sex-related (literature); retained as a
# comparison. The analysed set is defined from the data in step D.
SEX_LINKED = ['PAEP', 'WFDC2', 'CGA', 'KLK13', 'XG']
# |median within-cancer male-vs-female AUC - 0.5| at or above this margin
# marks a panel protein as having sex-associated plasma levels.
SEX_ASSOC_MARGIN = 0.20

# Cancers that occur in one sex only (used to validate the positional join)
FEMALE_ONLY = ['Cervical', 'Endometrial', 'Ovarian']
MALE_ONLY = ['Prostate']


def impute_full(x_tr_full, x_te_full, all_proteins, seed=SEED):
    """Impute ONCE, in the full 1463-protein neighbour space, exactly as the
    source pipeline does (base.protein_columns_from_df feeds the full table
    to KNNImputer before any panel is selected -- panel selection happens
    AFTER imputation, not before). Fitting KNN on a 25-column subset instead
    puts every neighbour distance in the wrong space and was caught here by
    a min_class_recall mismatch against the reproducer's recorded value
    (0.619) that random-seed noise could not explain."""
    imp = KNNImputer(n_neighbors=5)
    tr = pd.DataFrame(imp.fit_transform(x_tr_full[all_proteins]),
                      columns=all_proteins, index=x_tr_full.index)
    te = pd.DataFrame(imp.transform(x_te_full[all_proteins]),
                      columns=all_proteins, index=x_te_full.index)
    return tr, te


def fit_eval(tr_imputed, y_tr, te_imputed, y_te, features, seed=SEED):
    """Model fit on an ALREADY fully-imputed matrix, subset to `features`."""
    tr, te = tr_imputed[features], te_imputed[features]
    sc = StandardScaler().fit(tr)
    model = LogisticRegression(penalty="l2", C=0.1, solver="lbfgs",
                               max_iter=5000, class_weight="balanced",
                               random_state=seed)
    model.fit(sc.transform(tr), y_tr)
    proba = model.predict_proba(sc.transform(te))
    pred = model.predict(sc.transform(te))
    classes = list(model.classes_)
    return {
        "macro_auc": float(roc_auc_score(y_te, proba, labels=classes,
                                         multi_class="ovr", average="macro")),
        "accuracy": float(accuracy_score(y_te, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro")),
        "macro_recall": float(recall_score(y_te, pred, average="macro")),
        "min_class_recall": float(recall_score(y_te, pred, average=None).min()),
        "_pred": pred, "_true": np.asarray(y_te), "_classes": classes,
        "_model": model,
    }


def main():
    df = pd.read_csv(DATA)
    print(f"working table: {df.shape[0]} patients, {df.shape[1] - 2} proteins")

    # ---------- 0. sex join + validation ----------
    meta = pd.read_excel(META)
    # Sample_ID is "<Cancer>_<n>"; metadata is grouped by cancer in the same
    # order. Assign positionally WITHIN each cancer group.
    meta = meta.reset_index(drop=True)
    meta["_k"] = meta.groupby("Cancer").cumcount() + 1
    df["_cancer_raw"] = df["Sample_ID"].str.rsplit("_", n=1).str[0]
    df["_k"] = df["Sample_ID"].str.rsplit("_", n=1).str[1].astype(int)

    # metadata cancer names vs Sample_ID prefixes
    print("\nmetadata cancers :", sorted(meta.Cancer.unique()))
    print("sample prefixes  :", sorted(df._cancer_raw.unique()))

    name_map = {
        "AML": "AML", "BRC": "Breast", "CLL": "CLL", "CRC": "Colorectal",
        "CVX": "Cervical", "ENDC": "Endometrial", "GLIOM": "Glioma",
        "LUNGC": "Lung", "LYMPH": "DLBCL", "MYEL": "Myeloma",
        "OVC": "Ovarian", "PRC": "Prostate",
    }
    print("resolved mapping :", name_map)
    unresolved = set(df._cancer_raw.unique()) - set(name_map)
    if unresolved:
        print(f"\nSTOP: could not map {unresolved} to metadata names.",
              file=sys.stderr)
        return 1

    df["_meta_cancer"] = df["_cancer_raw"].map(name_map)
    merged = df.merge(meta[["Cancer", "_k", "Sex"]],
                      left_on=["_meta_cancer", "_k"],
                      right_on=["Cancer", "_k"], how="left",
                      suffixes=("", "_meta"))
    n_missing = merged.Sex.isna().sum()
    print(f"\nsex assigned: {len(merged) - n_missing}/{len(merged)} "
          f"({n_missing} unmatched)")

    # HARD validation: single-sex cancers must come out single-sex
    print("\n--- join validation (must be 100% single-sex) ---")
    ok = True
    for canc, expect in ([(c, "Female") for c in FEMALE_ONLY]
                         + [(c, "Male") for c in MALE_ONLY]):
        sub = merged[merged._meta_cancer == canc]
        if not len(sub):
            continue
        frac = (sub.Sex == expect).mean()
        flag = "OK " if frac == 1.0 else "FAIL"
        if frac != 1.0:
            ok = False
        print(f"  {flag} {canc:12s} {expect:6s} {frac:6.1%}  (n={len(sub)})")
    brc = merged[merged._meta_cancer == "Breast"]
    if len(brc):
        print(f"  --  Breast       Female {(brc.Sex=='Female').mean():6.1%}"
              f"  (n={len(brc)}; male breast cancer is expected to be rare)")

    if not ok:
        print("\nSTOP: positional sex join is NOT valid. Results would be "
              "meaningless. Need a keyed Sample_ID<->sex mapping.",
              file=sys.stderr)
        return 1
    print("\njoin validated: sex-restricted cancers are single-sex.\n")

    merged = merged[merged.Sex.notna()].copy()
    # protein_count and Label are metadata, excluded exactly as in
    # run_12class_ovr_pipeline.protein_columns_from_df; including protein_count
    # put an unscaled count into the KNN neighbour space.
    proteins = [c for c in df.columns
                if c not in {"Sample_ID", "Cancer", "_cancer_raw", "_k",
                             "_meta_cancer", "protein_count", "Label"}]
    missing = [p for p in LOCKED_25 if p not in proteins]
    if missing:
        print(f"STOP: locked-25 members absent from table: {missing}",
              file=sys.stderr)
        return 1

    X = merged[proteins]
    y = merged["Cancer"].to_numpy()
    sex = merged["Sex"].to_numpy()

    idx_tr, idx_te = train_test_split(
        np.arange(len(merged)), test_size=TEST_SIZE,
        random_state=SEED, stratify=y)
    x_tr, x_te = X.iloc[idx_tr], X.iloc[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    sex_te = sex[idx_te]

    print(f"\nimputing once across all {len(proteins)} measured proteins "
          f"(train-fit), matching the source pipeline's neighbour space "
          f"before any panel subset is taken ...")
    full_tr, full_te = impute_full(x_tr, x_te, proteins)

    results = {}

    # ---------- A. baseline ----------
    print("=" * 66)
    print("A. BASELINE -- locked 25, all patients")
    base = fit_eval(full_tr, y_tr, full_te, y_te, LOCKED_25)
    for k in ("macro_auc", "accuracy", "balanced_accuracy", "macro_f1",
              "macro_recall", "min_class_recall"):
        print(f"   {k:20s} {base[k]:.4f}")
    print("   [manuscript reports macro AUC 0.956, accuracy 0.753, "
          "min class recall 0.619]")
    for k, v in (("accuracy", 0.745763), ("macro_auc", 0.957298),
                 ("min_class_recall", 0.571429)):
        if abs(base[k] - v) > 5e-4:
            print(f"STOP: baseline {k}={base[k]:.4f} does not reproduce the "
                  f"locked model ({v})", file=sys.stderr)
            return 1
    results["A_baseline"] = {k: base[k] for k in base if not k.startswith("_")}

    # ---------- B. sex-only ceiling ----------
    print("\n" + "=" * 66)
    print("B. SEX-ONLY CEILING -- how much can sex alone predict?")
    s_tr = pd.DataFrame({"is_female": (sex[idx_tr] == "Female").astype(float)},
                        index=full_tr.index)
    s_te = pd.DataFrame({"is_female": (sex_te == "Female").astype(float)},
                        index=full_te.index)
    sx = fit_eval(s_tr, y_tr, s_te, y_te, ["is_female"])
    for k in ("macro_auc", "accuracy", "balanced_accuracy", "macro_recall"):
        print(f"   {k:20s} {sx[k]:.4f}")
    results["B_sex_only"] = {k: sx[k] for k in sx if not k.startswith("_")}

    # ---------- C. within-sex performance of the baseline model ----------
    print("\n" + "=" * 66)
    print("C. WITHIN-SEX -- baseline predictions scored inside each stratum")
    print("   (sex carries no information within a stratum)")
    strata = {}
    for s in ("Female", "Male"):
        m = sex_te == s
        if m.sum() == 0:
            continue
        t, p = base["_true"][m], base["_pred"][m]
        present = sorted(set(t))
        rec = recall_score(t, p, average=None, labels=present)
        strata[s] = {
            "n_test": int(m.sum()),
            "n_classes_present": len(present),
            "accuracy": float(accuracy_score(t, p)),
            "macro_recall": float(np.mean(rec)),
            "min_class_recall": float(rec.min()),
            "per_class_recall": {c: float(r) for c, r in zip(present, rec)},
        }
        print(f"\n   {s} (n={m.sum()}, {len(present)} classes present)")
        print(f"      accuracy         {strata[s]['accuracy']:.4f}")
        print(f"      macro recall     {strata[s]['macro_recall']:.4f}")
        print(f"      min class recall {strata[s]['min_class_recall']:.4f}")
        for c, r in sorted(strata[s]["per_class_recall"].items(),
                           key=lambda kv: kv[1]):
            print(f"        {c:14s} {r:.3f}")
    results["C_within_sex"] = strata

    # ---------- D. sex association of each panel protein ----------
    print("\n" + "=" * 66)
    print("D. SEX ASSOCIATION -- within-cancer male-vs-female AUC")
    # audit SOL-B02: the sex-associated list is chosen on the development rows only
    X_d, y_d, sex_d = X.iloc[idx_tr], y[idx_tr], sex[idx_tr]
    mixed = [c for c in sorted(set(y_d))
             if set(sex_d[y_d == c]) == {"Female", "Male"}]
    rows = []
    for p in LOCKED_25:
        aucs = []
        for c in mixed:
            m = y_d == c
            v = X_d.loc[m, p].to_numpy()
            s_ = sex_d[m]
            ok = ~np.isnan(v)
            if len(set(s_[ok])) == 2:
                aucs.append(roc_auc_score(s_[ok] == "Male", v[ok]))
        rows.append({"protein": p,
                     "median_male_vs_female_auc": float(np.median(aucs)),
                     "n_cancers": len(aucs)})
    assoc = pd.DataFrame(rows)
    assoc["distance_from_0.5"] = (assoc.median_male_vs_female_auc - 0.5).abs()
    assoc = assoc.sort_values("distance_from_0.5", ascending=False)
    sex_assoc = assoc.loc[assoc["distance_from_0.5"] >= SEX_ASSOC_MARGIN,
                          "protein"].tolist()
    print(f"   mixed-sex cancers: {mixed}")
    print(assoc.head(8).to_string(index=False))
    print(f"   sex-associated (margin {SEX_ASSOC_MARGIN}): {sex_assoc}")
    assoc.to_csv(OUT / "locked25_sex_association.csv", index=False)
    results["D_sex_association"] = {"mixed_sex_cancers": mixed,
                                    "margin": SEX_ASSOC_MARGIN,
                                    "sex_associated": sex_assoc}

    def drop_and_refit(label, dropped):
        keep = [p for p in LOCKED_25 if p not in dropped]
        r = fit_eval(full_tr, y_tr, full_te, y_te, keep)
        print("\n" + "=" * 66)
        print(f"{label} -- drop {dropped}, refit on {len(keep)} proteins")
        for k in ("macro_auc", "accuracy", "balanced_accuracy", "macro_f1",
                  "macro_recall", "min_class_recall"):
            print(f"   {k:20s} {r[k]:.4f}   ({r[k] - base[k]:+.4f} vs baseline)")
        return {"dropped": list(dropped), "features_kept": keep,
                **{k: r[k] for k in r if not k.startswith("_")}}

    # ---------- E. drop the data-defined sex-associated proteins ----------
    results["E_drop_sex_associated"] = drop_and_refit("E", sex_assoc)
    # ---------- F. a priori list, for comparison ----------
    results["F_drop_apriori_list"] = drop_and_refit("F", SEX_LINKED)

    # ---------- G. coefficient audit ----------
    print("\n" + "=" * 66)
    print("G. COEFFICIENT AUDIT -- share of |coef| mass")
    mdl = base["_model"]
    coefs = pd.DataFrame(np.abs(mdl.coef_), columns=LOCKED_25,
                         index=mdl.classes_)
    for name, grp in (("sex_associated", sex_assoc), ("apriori_list", SEX_LINKED)):
        share = float(coefs[grp].to_numpy().sum() / coefs.to_numpy().sum())
        results[f"G_coef_share_{name}"] = share
        print(f"   {name:15s} {grp}: {share:.1%}")
    coefs.to_csv(OUT / "locked25_abs_coefficients_by_class.csv")

    with open(OUT / "sex_sensitivity_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
