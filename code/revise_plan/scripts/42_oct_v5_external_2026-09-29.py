#!/usr/bin/env python3
"""
v5.0 Phase 4 (FORK of script 38; the pilot driver is unchanged). Differences:
panels from the v5 canonical membership (disease-specific) and the locked 25
(multiclass); runner = generic_locked_panel_external_validation_null50.py
(100 permutations, each with the same 50 grouped splits as the observed AUC);
full commands/stderr kept in each endpoint's run_log.txt; empirical P
recorded; outputs under revise_plan/v5_seed52_common_split/external/.

Usage: python 42_oct_v5_external_2026-09-29.py [--permutations 100]
           [--only CODE ...] [--variants v5_disease_specific multiclass_25]

Original docstring (script 38) follows.
PILOT step 3 (2026-09-29): external evaluation of the disease-specific panel
variants from script 37 on the 12 Table S10 endpoints, with the production
preparation scripts and the production runner
(generic_locked_panel_external_validation.py; L2 1.0, gradient descent,
50 participant-grouped 75/25 splits, seed 52) and the production threshold
summary (per-participant mean probability; AUC, sensitivity/specificity at
0.80). Permutations are reduced to 1 because only AUC/sensitivity/
specificity are compared here.

The "current_seed42" variant must reproduce the Table S10 disease-specific
AUCs; that is the check that this driver matches production.

Everything is written under revise_plan/pilot_seed52_common_split_2026-09-29/
external/; no production result folder is touched.
"""
from __future__ import annotations
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = PROJECT_ROOT
EV = ROOT / "revise_plan/external_validation_roadmap"
SC = EV / "scripts"
V5 = ROOT / "revise_plan/v5_seed52_common_split"
OUT = V5 / "external"
HUB = V5 / "panels/panel_memberships_v5.csv"
PERMUTATIONS = 100
LOG = OUT / "run_log.txt"
PY = sys.executable
GPL570 = EV / "gse4290/data/source/GPL570.annot.gz"
GPL96 = EV / "gse6477_myel_gene/data/source/GPL96.annot.gz"

# endpoint -> (kind, settings). "geo" rows use prepare_geo_gpl570_panel_proxy.py
GEO = {
    "BRC": (EV / "gse42568_brc_gene/data/source/GSE42568_series_matrix.txt.gz", EV / "gse42568_brc_gene/data/source/GPL570.annot.gz",
            "^tissue: breast cancer$", "^tissue: normal breast$", "characteristics_ch1", None),
    "CRC": (EV / "geo_crc_GSE41258/data/source/GSE41258_series_matrix.txt.gz", GPL96,
            "^tissue: Primary Tumor$", "^tissue: Normal Colon$", "characteristics_ch1", None),
    "CVX": (EV / "candidate_cvx_gse9750/data/source/GSE9750_series_matrix.txt.gz", EV / "candidate_cvx_gse9750/data/source/GPL96.annot.gz",
            "^cervical (?:cancer|cacner)_(?!cell)", "^normal cervix(?!, commercial)", "title", None),
    "ENDC": (EV / "geo_endc_GSE17025/data/source/GSE17025_series_matrix.txt.gz", GPL570,
             "^(EE|PS)_", "^NL_", "source_name_ch1", None),
    "OVC": (EV / "geo_ovc_GSE18520/data/source/GSE18520_series_matrix.txt.gz", GPL570,
            "adenocarcinoma", "normal", "characteristics_ch1,source_name_ch1,title", None),
    "PRC": (EV / "geo_prc_GSE17951/data/source/GSE17951_series_matrix.txt.gz", GPL570,
            "sample type .*: T$", "sample type .*: (Biopsy|Ctrl)$", "characteristics_ch1__1", None),
    "DLBCL": (EV / "gse32018_lymph_gene/data/source/GSE32018_series_matrix.txt.gz", EV / "gse32018_lymph_gene/data/source/GPL6480.annot.gz",
              "Diffuse Large B cell Lymphoma", "^Lymph-node|Reactive Tonsils", "title,source_name_ch1", None),
    "MYEL": (EV / "gse6477_myel_gene/data/source/GSE6477_series_matrix.txt.gz", GPL96,
             "myeloma|MM_", "normal donor", "title,source_name_ch1",
             EV / "gse6477_myel_gene/data/sample_metadata_binary.csv"),
}


def run(cmd: list) -> None:
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(" ".join(str(c) for c in cmd) + "\n" + r.stdout[-4000:] + "\n" + r.stderr + "\n")
    if r.returncode:
        raise RuntimeError(f"{cmd[1]} failed (full stderr in {LOG}):\n{r.stderr[-2000:]}")


def runner(matrix, meta, out, panel, group=None):
    cmd = [PY, SC / "generic_locked_panel_external_validation_null50.py", "--matrix", matrix, "--metadata", meta,
           "--out-dir", out, "--panel", panel, "--sample-column", "sample_id", "--label-column", "label",
           "--positive-values", "1", "--repeats", "50", "--permutations", str(PERMUTATIONS), "--seed", "52"]
    if group:
        cmd += ["--group-column", group]
    out.mkdir(parents=True, exist_ok=True)
    run(cmd)
    run([PY, SC / "summarize_grouped_prediction_thresholds.py", "--predictions", out / "locked25_grouped_predictions.csv",
         "--output", out / "fixed_threshold_summary.csv", "--cohort", out.name, "--thresholds", "0.50,0.80"])


def endpoint(code: str, panel: Path, work: Path) -> None:
    data, res = work / "data", work / "result"
    data.mkdir(parents=True, exist_ok=True)
    if code in GEO:
        series, annot, pos, neg, cols, meta = GEO[code]
        run([PY, SC / "prepare_geo_gpl570_panel_proxy.py", "--series", series, "--annotation", annot, "--panel", panel,
             "--out-dir", data, "--results-dir", work / "prep", "--positive-regex", pos, "--negative-regex", neg,
             "--positive-name", code.lower(), "--negative-name", "control", "--label-columns", cols])
        runner(data / "processed_matrix.csv", meta or data / "sample_metadata_binary.csv", res, panel)
    elif code in ("AML", "CLL"):
        # gse13159_aml/data/source/GPL570.annot.gz is a 44-byte placeholder;
        # use the full GPL570 annotation stored with GSE4290.
        run([PY, EV / "gse13159_aml/scripts/prepare_gse13159_aml_transcript_proxy_matrix.py", "--panel", panel,
             "--annotation", GPL570, "--out-dir", data, "--results-dir", work / "prep"])
        if code == "AML":
            meta = data / "sample_metadata_aml_vs_control.csv"
        else:
            meta = data / "sample_metadata_cll_vs_control.csv"
            run([PY, SC / "make_single_panel_aux_metadata.py", "--mode", "gse13159_cll",
                 "--metadata", data / "sample_metadata_all_samples.csv", "--out", meta])
        runner(data / "processed_matrix.csv", meta, res, panel)
    elif code == "GLIOM":
        run([PY, EV / "gse4290/scripts/prepare_gse4290_transcript_proxy_matrix.py", "--panel", panel,
             "--out-dir", data, "--results-dir", work / "prep"])
        runner(data / "processed_matrix.csv", data / "sample_metadata_glioma_vs_nontumor.csv", res, panel)
    elif code == "LUNGC":
        runner(EV / "cptac_icpc_luad_protein/data/processed_matrix.csv",
               EV / "cptac_icpc_luad_protein/data/sample_metadata_patient_grouped.csv", res, panel, "group")


CODES = ["AML", "BRC", "CLL", "CRC", "CVX", "DLBCL", "ENDC", "GLIOM", "LUNGC", "MYEL", "OVC", "PRC"]
# Each system is scored at its manuscript threshold: 0.80 disease-specific, 0.50 multiclass.
THRESHOLD = {"v5_disease_specific": 0.8, "multiclass_25": 0.5}


def write_panels() -> dict:
    hub = pd.read_csv(HUB)
    paths = {}
    for variant in THRESHOLD:
        d = V5 / "panels" / "external_panel_files" / variant
        d.mkdir(parents=True, exist_ok=True)
        for code in CODES:
            key = f"single_{code}" if variant == "v5_disease_specific" else "multiclass_25"
            f = d / f"{code.lower()}_panel.csv"
            pd.DataFrame({"gene_symbol": hub[hub.panel == key].protein.tolist()}).to_csv(f, index=False)
            paths[(variant, code)] = f
    return paths


def main() -> int:
    global LOG, PERMUTATIONS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=100)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--variants", nargs="*", default=list(THRESHOLD))
    a = ap.parse_args()
    PERMUTATIONS = a.permutations
    panels = write_panels()
    rows = []
    for v in a.variants:
        for code in CODES:
            if a.only and code not in a.only:
                continue
            work = OUT / v / code
            LOG = work / "run_log.txt"
            try:
                endpoint(code, panels[(v, code)], work)
                s = pd.read_csv(work / "result/fixed_threshold_summary.csv")
                c = pd.read_csv(work / "result/cohort_summary.csv").iloc[0]
                perm = pd.read_csv(work / "result/locked25_leakage_diagnostic_summary.csv").iloc[0]
                r = s[s.threshold == THRESHOLD[v]].iloc[0]
                rows.append(dict(variant=v, cancer=code, cases=int(c.n_tumor), controls=int(c.n_control),
                                 markers_mapped=f"{int(c.mapped_panel_features)}/{int(c.total_panel_features)}",
                                 threshold=THRESHOLD[v], auc=r.auc_per_sample_mean_probability,
                                 sensitivity=r.sensitivity, specificity=r.specificity,
                                 tp=int(r.tp), fn=int(r.fn), tn=int(r.tn), fp=int(r.fp),
                                 permutations=PERMUTATIONS, permutation_auc_mean=perm.permutation_auc_mean,
                                 empirical_p=perm.empirical_p_value_ge_observed))
            except Exception as e:  # keep going; the full stderr is in run_log.txt
                rows.append(dict(variant=v, cancer=code, error=str(e)[:500]))
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    tag = "_".join(a.variants) + ("_" + "_".join(a.only) if a.only else "") + f"_perm{PERMUTATIONS}"
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / f"v5_external_metrics_{tag}.csv", index=False)
    print(out.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
