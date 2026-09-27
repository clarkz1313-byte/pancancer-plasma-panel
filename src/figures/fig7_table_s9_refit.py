"""Render the original disease-specific figure template with the Table S9 seed-52 refit.

Keeps the original panel and mosaic builders unchanged. Input CSVs are
redirected to this release's data directory.
"""
from pathlib import Path
import csv
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/filtered_pancancer_data.csv"
PANELS = ROOT / "data/panel_memberships.csv"
TABLE = ROOT / "data/Table_S9_disease_specific_panel_metrics.csv"
OUTPUT = ROOT / "figures/fig6_table_s9_refit.png"
sys.path.insert(0, str(ROOT / "src/figures"))
import fig7_panels as template
import build_slide_mosaics as mosaic


def render():
    raw = pd.read_csv(DATA)
    features = [c for c in raw.columns if c not in ("Sample_ID", "Cancer")]
    membership = {}
    with PANELS.open(encoding="utf-8-sig", newline="") as source:
        for record in csv.DictReader(source):
            values = list(record.values())
            membership.setdefault(values[0].strip(), []).append(values[1].strip())

    y_all = raw["Cancer"].to_numpy()
    train_idx, test_idx = train_test_split(
        np.arange(len(raw)), test_size=0.30, random_state=52, stratify=y_all
    )
    imputer = KNNImputer(n_neighbors=5)
    train = pd.DataFrame(imputer.fit_transform(raw.iloc[train_idx][features]), columns=features)
    test = pd.DataFrame(imputer.transform(raw.iloc[test_idx][features]), columns=features)
    published = pd.read_csv(TABLE)

    with tempfile.TemporaryDirectory(prefix="fig7_v44_") as location:
        work = Path(location)
        template.PARTA_DIR = work / "predictions"
        template.SUPP = work / "panel_sizes.csv"
        template.OUT = work / "panels"
        template.OUT.mkdir()
        template.DPI = 250
        mosaic.OUT = template.OUT

        # The historical builder seeded AUC bootstrap draws with Python's
        # process-randomised hash. A fixed seed makes the new figure repeatable.
        original_auc_ci = template._bootstrap_auc_ci
        def stable_auc_ci(y_true, y_score, n_boot=500, seed=0):
            return original_auc_ci(y_true, y_score, n_boot=n_boot, seed=52)
        template._bootstrap_auc_ci = stable_auc_ci

        sizes = []
        for code in template.CANCERS:
            display = "DLBCL" if code == "LYMPH" else code
            selected = [f for f in membership["single_" + display] if f in features]
            positive = (y_all[train_idx] == code).astype(int)
            observed = (y_all[test_idx] == code).astype(int)
            scaler = StandardScaler().fit(train[selected])
            model = LogisticRegression(
                penalty=None, solver="lbfgs", max_iter=5000, random_state=52
            )
            model.fit(scaler.transform(train[selected]), positive)
            score = model.predict_proba(scaler.transform(test[selected]))[:, 1]
            called = (score >= 0.50).astype(int)
            actual = {
                "tp": int(((observed == 1) & (called == 1)).sum()),
                "fn": int(((observed == 1) & (called == 0)).sum()),
                "tn": int(((observed == 0) & (called == 0)).sum()),
                "fp": int(((observed == 0) & (called == 1)).sum()),
            }
            table_row = published[
                (published["cancer"] == display) &
                (published["threshold"] == 0.50)
            ].iloc[0]
            expected = {k: int(table_row[k]) for k in actual}
            if actual != expected:
                raise ValueError(f"{display}: {actual} differs from Table S9 {expected}")

            part_dir = template.PARTA_DIR / code.lower() / "tables"
            part_dir.mkdir(parents=True)
            pd.DataFrame({
                "panel": [f"{code} Top {len(selected)}"] * len(observed),
                "partition": ["test"] * len(observed),
                "y_true": observed,
                "y_score": score,
                "y_pred": called,
            }).to_csv(part_dir / f"{code.lower()}_test_predictions.csv", index=False)
            sizes.append({"target_class": code, "deploy_panel_size": len(selected)})
        pd.DataFrame(sizes).to_csv(template.SUPP, index=False)

        template.panel_confusion_grid()
        template.panel_calibration()
        template.panel_decision_curve()
        stem = "fig7_slide7_singleclass_mosaic"
        mosaic.build(stem, mosaic.SLIDES[stem])
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(template.OUT / (stem + ".png"), OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    render()
