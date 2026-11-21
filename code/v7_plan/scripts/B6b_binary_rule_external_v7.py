"""v7 B6b: the two-step rule as a yes/no test in the 12 single-cancer tissue endpoints (external level of revise_plan
script 70, unchanged functions, run with the v7 B6 predictions). Step 1 = 26-protein marker-set model >= 0.50;
step 2 = disease-specific model >= 0.80; two-step positive when both are. Specimen score = mean over the splits in
which it was tested; patient-grouped bootstrap (2,000, seed 20261001) for LUNGC, CRC and PRC as in script 70.
Plasma and TCGA levels are not rerun here (they are B5 and B7). Output: results/B6_external/two_step_binary/
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = PROJECT_ROOT
B6 = ROOT / "v7_plan/results/B6_external"
OUT = B6 / "two_step_binary"
spec = importlib.util.spec_from_file_location("s70", ROOT / "revise_plan/scripts/70_oct_v51_two_step_binary_rule_2026-10-01.py")
s70 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s70)


def ext_scores(variant, cancer):
    p = B6 / variant / cancer / "result_patient_grouped/locked25_grouped_predictions.csv"
    if not p.exists():
        p = B6 / variant / cancer / "result/locked25_grouped_predictions.csv"
    df = pd.read_csv(p)
    return df.groupby("sample_id").agg(y=("true_label", "first"), p=("predicted_probability_tumor", "mean"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(s70.SEED)
    rows, comps, curves = [], [], []
    for c in s70.EXT:
        a = ext_scores("multiclass_26", c); b = ext_scores("disease_specific", c).loc[a.index]
        assert (a.y == b.y).all()
        if c in s70.PATIENT_META:
            f, col = s70.PATIENT_META[c]
            m = pd.read_csv(f).set_index("sample_id")[col]
            groups = m.loc[a.index].astype(str).to_numpy()
        else:
            groups = a.index.to_numpy()
        s70.evaluate("external", c, "step2>=0.80", a.y.values == 1, groups, a.p.values >= 0.50, b.p.values >= 0.80,
                     rng, rows, comps, curves)
    pd.DataFrame(rows).to_csv(OUT / "binary_rule_metrics.csv", index=False)
    pd.DataFrame(comps).to_csv(OUT / "binary_rule_paired_comparisons.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "binary_rule_decision_curves.csv", index=False)
    R = pd.DataFrame(rows)
    print(R[R.rule.isin(["step1", "two_step"])][["cancer", "rule", "sensitivity", "specificity"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
