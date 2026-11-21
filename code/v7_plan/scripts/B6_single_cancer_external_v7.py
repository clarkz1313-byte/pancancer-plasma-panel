"""v7 B6: single-cancer tissue tests (12 endpoints, Table S10) with the final panels.

Variant multiclass_26 (26-protein list, cutoff 0.50) and disease_specific (final disease-specific lists, cutoff
0.80). Same preparation scripts, runner, 50 grouped repeats, 100 permutations, seed 52 and thresholds as
revise_plan script 42; LUNGC grouped by patient (script 68), CRC and PRC by GEO patient id (script 69). All
outputs under v7_plan/results/B6_external/. Afterwards the combined call (both lists required) is computed per
endpoint from per-specimen mean probabilities.
Usage: python B6_... [--variants multiclass_26 disease_specific] [--only CODE ...] [--permutations 100] [--combine-only]
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import argparse
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = PROJECT_ROOT
V7 = ROOT / "v7_plan"
OUT = V7 / "results/B6_external"
PANELS = V7 / "panels/panel_memberships_v7.csv"
spec = importlib.util.spec_from_file_location("s42", ROOT / "revise_plan/scripts/42_oct_v5_external_2026-09-29.py")
s42 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s42)
s42.HUB, s42.OUT = PANELS, OUT
spec94 = importlib.util.spec_from_file_location("s94", ROOT / "revise_plan/scripts/94_oct_v65_ds_devonly_single_cancer_external_2026-10-16.py")
s94 = importlib.util.module_from_spec(spec94)
spec94.loader.exec_module(s94)
THRESHOLD = {"multiclass_26": 0.5, "disease_specific": 0.8}
FIELD = s94.FIELD


def write_panels(variant):
    hub = pd.read_csv(PANELS)
    d = OUT / "panel_files" / variant
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    for code in s42.CODES:
        key = "multiclass_26" if variant == "multiclass_26" else f"single_{code}"
        genes = hub[hub.panel == key].protein.tolist()
        assert genes, (variant, code)
        f = d / f"{code.lower()}_panel.csv"
        pd.DataFrame({"gene_symbol": genes}).to_csv(f, index=False)
        paths[code] = f
    return paths


def run_endpoint(variant, code, panel):
    work = OUT / variant / code
    s42.LOG = work / "run_log.txt"
    if code == "LUNGC":
        data = s42.EV / "cptac_icpc_luad_protein/data"
        s94.grouped_runner(data / "processed_matrix.csv", data / "sample_metadata_patient_grouped.csv", work / "result", panel, s42.LOG)
        return work / "result"
    s42.endpoint(code, panel, work)
    if code in FIELD:
        meta = pd.read_csv(work / "data/sample_metadata_binary.csv")
        meta["patient_id"] = meta[FIELD[code]].str.replace("patient id: ", "", regex=False).str.strip().str.lstrip("0").str.upper()
        assert meta.patient_id.notna().all()
        mp = work / "data/sample_metadata_binary_patient.csv"
        meta.to_csv(mp, index=False)
        s94.grouped_runner(work / "data/processed_matrix.csv", mp, work / "result_patient_grouped", panel, s42.LOG)
        return work / "result_patient_grouped"
    return work / "result"


def result_dir(variant, code):
    return OUT / variant / code / ("result_patient_grouped" if code in FIELD else "result")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(THRESHOLD))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--permutations", type=int, default=100)
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--skip-done", action="store_true", help="reuse endpoints whose final result files already exist")
    a = ap.parse_args()
    s42.PERMUTATIONS = a.permutations
    if not a.combine_only:
        for v in a.variants:
            panels = write_panels(v)
            rows = []
            for code in [c for c in s42.CODES if not a.only or c in a.only]:
                try:
                    res = result_dir(v, code)
                    done = (res / "fixed_threshold_summary.csv").exists() and (res / "locked25_leakage_diagnostic_summary.csv").exists()
                    if not (a.skip_done and done):
                        res = run_endpoint(v, code, panels[code])
                    s = pd.read_csv(res / "fixed_threshold_summary.csv")
                    c = pd.read_csv(res / "cohort_summary.csv").iloc[0]
                    perm = pd.read_csv(res / "locked25_leakage_diagnostic_summary.csv").iloc[0]
                    r = s[s.threshold == THRESHOLD[v]].iloc[0]
                    rows.append(dict(variant=v, cancer=code, cases=int(c.n_tumor), controls=int(c.n_control),
                                     markers_mapped=f"{int(c.mapped_panel_features)}/{int(c.total_panel_features)}",
                                     threshold=THRESHOLD[v], auc=r.auc_per_sample_mean_probability, sensitivity=r.sensitivity,
                                     specificity=r.specificity, tp=int(r.tp), fn=int(r.fn), tn=int(r.tn), fp=int(r.fp),
                                     permutations=a.permutations, permutation_auc_mean=perm.permutation_auc_mean,
                                     empirical_p=perm.empirical_p_value_ge_observed))
                except Exception as e:
                    rows.append(dict(variant=v, cancer=code, error=str(e)[:400]))
                print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(OUT / f"B6_metrics_{v}{'_' + '_'.join(a.only) if a.only else ''}.csv", index=False)
    comb = []
    for code in s42.CODES:
        mc, ds = result_dir("multiclass_26", code), result_dir("disease_specific", code)
        if (mc / "locked25_grouped_predictions.csv").exists() and (ds / "locked25_grouped_predictions.csv").exists():
            comb.append({"cancer": code, **s94.combined(mc, ds)})
    if comb:
        t = pd.DataFrame(comb)
        t.to_csv(OUT / "B6_combined_call.csv", index=False)
        print(t.round(3).to_string(index=False))
        print(t[[c for c in t.columns if c.startswith(("sens", "spec"))]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
