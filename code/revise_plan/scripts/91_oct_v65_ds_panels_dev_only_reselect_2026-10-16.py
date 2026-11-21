"""Re-select the 12 disease-specific panels using the development set only (v6.5 pilot).

Same ranking and nested panels as the published selection (part_a_single top-18 LASSO ranking from the
development set; script 39 rule), same per-size model as base.evaluate_panel: standardisation on the
development set, L1 logistic regression (liblinear) with C in {0.01, 0.1, 1, 10} chosen by 5-fold
GridSearchCV on AUC (default unshuffled stratified folds). New: for each size, 5-fold cross-validated
predictions of the chosen model give a cross-validated balanced accuracy at 0.50, so the development-only
rule mirrors the published one with CV in place of the internal test set:
    smallest size (1-18) with CV AUC within 0.02 of the best CV AUC and CV balanced accuracy within 0.05 of
    the best CV balanced accuracy; floor 2 if the top protein is Bonferroni-significant in > 1 cancer.
Check: the recomputed CV AUC must equal the saved CV_AUC of each size.
Output: revise_plan/pilot_ds_dev_only_size/panel_memberships_devonly.csv and size_sweep_devonly.csv
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = PROJECT_ROOT
SEL = ROOT / "revise_plan/v5_seed52_common_split/part_a_single"
PUB = ROOT / "revise_plan/v5_seed52_common_split/panels/panel_memberships_v5.csv"
OUT = ROOT / "revise_plan/pilot_ds_dev_only_size"
CODES = {"aml": "AML", "brc": "BRC", "cll": "CLL", "crc": "CRC", "cvx": "CVX", "endc": "ENDC", "gliom": "GLIOM",
         "lungc": "LUNGC", "lymph": "LYMPH", "myel": "MYEL", "ovc": "OVC", "prc": "PRC"}
SEED, CAP = 52, 18


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/processed/filtered_pancancer_data.csv")
    proteins = [c for c in df.columns if c not in ("Sample_ID", "Cancer", "protein_count")]
    y = df["Cancer"].to_numpy()
    tr, _ = train_test_split(np.arange(len(df)), test_size=0.3, random_state=SEED, stratify=y)
    imp = KNNImputer(n_neighbors=5).fit(df.iloc[tr][proteins])
    Xdev = pd.DataFrame(imp.transform(df.iloc[tr][proteins]), columns=proteins)
    ydev = y[tr]
    bonf = {c: set(pd.read_csv(SEL / s / "tables" / f"{s}_dea_bonf_significant.csv").protein) for s, c in CODES.items()}
    pub = pd.read_csv(PUB)
    sweep, members, audit = [], [], []
    for slug, code in CODES.items():
        t = SEL / slug / "tables"
        top = list(pd.read_csv(t / f"{slug}_top_feature_ranking.csv").protein)[:CAP]
        saved = pd.read_csv(t / f"{slug}_panel_performance.csv").set_index("Features")
        yb = (ydev == code).astype(int)
        for k in range(1, len(top) + 1):
            feats = top[:k]
            xs = StandardScaler().fit_transform(Xdev[feats])
            grid = GridSearchCV(LogisticRegression(random_state=SEED, max_iter=1000),
                                {"C": [0.01, 0.1, 1, 10], "penalty": ["l1"], "solver": ["liblinear"]},
                                cv=5, scoring="roc_auc", n_jobs=-1).fit(xs, yb)
            p = cross_val_predict(grid.best_estimator_, xs, yb, cv=5, method="predict_proba")[:, 1]
            cv_ba = balanced_accuracy_score(yb, (p >= 0.5).astype(int))
            saved_auc = float(saved.loc[k, "CV_AUC"]) if k in saved.index else np.nan
            sweep.append({"cancer": code, "size": k, "cv_auc": grid.best_score_, "saved_cv_auc": saved_auc,
                          "cv_balanced_accuracy": cv_ba, "best_C": grid.best_params_["C"],
                          "test_auc_saved": float(saved.loc[k, "Test_AUC"]) if k in saved.index else np.nan})
        s = pd.DataFrame([r for r in sweep if r["cancer"] == code])
        assert np.allclose(s.cv_auc, s.saved_cv_auc, atol=1e-9), (code, (s.cv_auc - s.saved_cv_auc).abs().max())
        keep = s[(s.cv_auc >= s.cv_auc.max() - 0.02) & (s.cv_balanced_accuracy >= s.cv_balanced_accuracy.max() - 0.05)]
        n = int(keep["size"].min())
        shared = sum(top[0] in b for b in bonf.values()) > 1
        n_final = max(n, 2) if (shared and n == 1) else n
        members += [{"panel": f"single_{'DLBCL' if code == 'LYMPH' else code}", "protein": p} for p in top[:n_final]]
        pub_n = int((pub.panel == f"single_{code}").sum())
        audit.append({"cancer": code, "published_size": pub_n, "dev_only_size": n_final, "floor_applied": n_final != n,
                      "dev_only_proteins": ";".join(top[:n_final])})
    pd.DataFrame(sweep).to_csv(OUT / "size_sweep_devonly.csv", index=False)
    mc = pub[pub.panel == "multiclass_25"]
    pooled = sorted({m["protein"] for m in members})
    hub = pd.concat([mc, pd.DataFrame(members), pd.DataFrame({"panel": "single_pooled_unique", "protein": pooled})])
    hub.to_csv(OUT / "panel_memberships_devonly.csv", index=False)
    a = pd.DataFrame(audit)
    a.to_csv(OUT / "panel_sizes_published_vs_devonly.csv", index=False)
    print(a[["cancer", "published_size", "dev_only_size", "floor_applied"]].to_string(index=False))
    pub_pool = set(pub[pub.panel == "single_pooled_unique"].protein)
    print(f"distinct disease-specific proteins: published {len(pub_pool)}, dev-only {len(pooled)}; "
          f"multiclass proteins inside disease-specific panels: published {len(set(mc.protein) & pub_pool)}, dev-only {len(set(mc.protein) & set(pooled))}; "
          f"union with multiclass: published {len(pub_pool | set(mc.protein))}, dev-only {len(set(pooled) | set(mc.protein))}")


if __name__ == "__main__":
    main()
