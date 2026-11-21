"""Downstream effects of the development-only disease-specific panels (script 91), part 3: the 12 single-cancer
external endpoints (Table S10), disease-specific lists only (the 25-protein results are unchanged).

Wraps script 42 (same preparation scripts, runner, 50 grouped repeats, 100 permutations, seed 52, thresholds)
with three redirections so that no production file is touched: membership hub = development-only panels,
panel files and outputs under revise_plan/pilot_ds_dev_only_size/external/. LUNGC is grouped by patient as in
script 68, CRC and PRC by GEO patient id as in script 69.
Then the combined call (25-protein list >= 0.50 and disease-specific list >= 0.80, per-specimen mean
probabilities) is computed per endpoint; the same function is first applied to the published predictions and
must reproduce Table A11 (per_cancer_two_denominators_single_cancer.csv).
Usage: python 94_... [--permutations 100] [--only CODE ...] [--combine-only]
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = PROJECT_ROOT
V5 = ROOT / "revise_plan/v5_seed52_common_split"
PILOT = ROOT / "revise_plan/pilot_ds_dev_only_size"
OUT = PILOT / "external"
A11 = ROOT / "revise_plan/two_step_binary_rule/per_cancer_two_denominators_single_cancer.csv"
spec = importlib.util.spec_from_file_location("s42", ROOT / "revise_plan/scripts/42_oct_v5_external_2026-09-29.py")
s42 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s42)
s42.HUB = PILOT / "panel_memberships_devonly.csv"
s42.OUT = OUT
VAR = "v5_disease_specific"
FIELD = {"CRC": "characteristics_ch1__2", "PRC": "characteristics_ch1"}


def write_panels():
    hub = pd.read_csv(s42.HUB)
    d = OUT / "panel_files" / VAR
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    for code in s42.CODES:
        f = d / f"{code.lower()}_panel.csv"
        pd.DataFrame({"gene_symbol": hub[hub.panel == f"single_{code}"].protein.tolist()}).to_csv(f, index=False)
        paths[code] = f
    return paths


def grouped_runner(matrix, meta, res, panel, log):
    res.mkdir(parents=True, exist_ok=True)
    s42.LOG = log
    s42.run([s42.PY, s42.SC / "generic_locked_panel_external_validation_null50.py", "--matrix", matrix, "--metadata", meta,
             "--out-dir", res, "--panel", panel, "--sample-column", "sample_id", "--label-column", "label",
             "--positive-values", "1", "--repeats", "50", "--permutations", str(s42.PERMUTATIONS), "--seed", "52",
             "--group-column", "patient_id"])
    s42.run([s42.PY, s42.SC / "summarize_grouped_prediction_thresholds.py", "--predictions", res / "locked25_grouped_predictions.csv",
             "--output", res / "fixed_threshold_summary.csv", "--cohort", res.parent.name, "--thresholds", "0.50,0.80"])


def run_endpoint(code, panel):
    work = OUT / VAR / code
    s42.LOG = work / "run_log.txt"
    if code == "LUNGC":
        data = s42.EV / "cptac_icpc_luad_protein/data"
        grouped_runner(data / "processed_matrix.csv", data / "sample_metadata_patient_grouped.csv", work / "result", panel, s42.LOG)
        return work / "result"
    s42.endpoint(code, panel, work)
    if code in FIELD:
        meta = pd.read_csv(work / "data/sample_metadata_binary.csv")
        meta["patient_id"] = meta[FIELD[code]].str.replace("patient id: ", "", regex=False).str.strip().str.lstrip("0").str.upper()
        assert meta.patient_id.notna().all()
        mp = work / "data/sample_metadata_binary_patient.csv"
        meta.to_csv(mp, index=False)
        grouped_runner(work / "data/processed_matrix.csv", mp, work / "result_patient_grouped", panel, s42.LOG)
        return work / "result_patient_grouped"
    return work / "result"


def per_specimen(res_dir):
    p = pd.read_csv(res_dir / "locked25_grouped_predictions.csv")
    return p.groupby("sample_id").agg(label=("true_label", "first"), prob=("predicted_probability_tumor", "mean"))


def production_dir(variant, code):
    if code == "LUNGC":
        return V5 / "external_lung_patient_grouped" / variant / "LUNGC/result"
    if code in FIELD:
        return V5 / "external_patient_grouped" / variant / code / "result"
    return V5 / "external" / variant / code / "result"


def combined(mc_dir, ds_dir):
    a, b = per_specimen(mc_dir), per_specimen(ds_dir)
    d = a.join(b, rsuffix="_ds", how="inner")
    assert (d.label == d.label_ds).all()
    y = d.label.to_numpy() == 1
    out = {}
    for name, call in (("25", d.prob >= 0.5), ("ds", d.prob_ds >= 0.8), ("combined", (d.prob >= 0.5) & (d.prob_ds >= 0.8))):
        c = call.to_numpy()
        out[f"sens_{name}"] = float(c[y].mean())
        out[f"spec_{name}"] = float((~c)[~y].mean())
    out.update(cases=int(y.sum()), non_malignant=int((~y).sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=100)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--combine-only", action="store_true")
    a = ap.parse_args()
    s42.PERMUTATIONS = a.permutations
    codes = [c for c in s42.CODES if not a.only or c in a.only]
    rows = []
    if not a.combine_only:
        panels = write_panels()
        for code in codes:
            try:
                res = run_endpoint(code, panels[code])
                s = pd.read_csv(res / "fixed_threshold_summary.csv")
                c = pd.read_csv(res / "cohort_summary.csv").iloc[0]
                perm = pd.read_csv(res / "locked25_leakage_diagnostic_summary.csv").iloc[0]
                r = s[s.threshold == 0.8].iloc[0]
                rows.append(dict(cancer=code, markers_mapped=f"{int(c.mapped_panel_features)}/{int(c.total_panel_features)}",
                                 auc=r.auc_per_sample_mean_probability, sensitivity=r.sensitivity, specificity=r.specificity,
                                 permutation_auc_mean=perm.permutation_auc_mean, empirical_p=perm.empirical_p_value_ge_observed))
            except Exception as e:
                rows.append(dict(cancer=code, error=str(e)[:400]))
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"devonly_ds_external_metrics_perm{a.permutations}.csv", index=False)
    # combined call: reproduce the published Table A11 first, then the development-only version
    a11 = pd.read_csv(A11).set_index("cancer")
    comb = []
    for code in s42.CODES:
        mc = production_dir("multiclass_25", code)
        pub = combined(mc, production_dir(VAR, code))
        for k, col in (("sens_25", "sens_25protein"), ("spec_25", "spec_25protein"), ("sens_ds", "sens_disease_specific"),
                       ("spec_ds", "spec_disease_specific"), ("sens_combined", "sens_combined"), ("spec_combined", "spec_combined")):
            assert abs(pub[k] - a11.loc[code, col]) < 1e-9, (code, k, pub[k], a11.loc[code, col])
        dev_dir = OUT / VAR / code / ("result_patient_grouped" if code in FIELD else "result")
        if (dev_dir / "locked25_grouped_predictions.csv").exists():
            dev = combined(mc, dev_dir)
            comb.append({"cancer": code, **{f"published_{k}": v for k, v in pub.items()}, **{f"devonly_{k}": v for k, v in dev.items()}})
    if comb:
        t = pd.DataFrame(comb)
        t.to_csv(OUT / "combined_call_published_vs_devonly.csv", index=False)
        means = t[[c for c in t.columns if "sens" in c or "spec" in c]].mean().round(3)
        print(t[["cancer", "published_sens_ds", "devonly_sens_ds", "published_spec_ds", "devonly_spec_ds",
                 "published_sens_combined", "devonly_sens_combined", "published_spec_combined", "devonly_spec_combined"]].round(3).to_string(index=False))
        print(means.to_string())


if __name__ == "__main__":
    sys.exit(main())
