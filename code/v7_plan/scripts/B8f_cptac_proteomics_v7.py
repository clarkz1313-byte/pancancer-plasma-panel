"""v7 B8f: CPTAC tumour proteomics (D3) rerun with the final v7 panels (audit SOL-D08), as a low-coverage
sensitivity analysis outside the >= 7-cancer synthesis. Same data handling as revise_plan script 88 (cptac 1.5.14,
umich harmonised proteomics, tumour samples only, gene-level mean over identifiers); same protocol (script 85).
Run with the isolated cptac environment (scratchpad venv_cptac).
Output: v7_plan/results/B8_mixed_cancer/summary_D3_cptac_proteomics_v7.json, raw/D3_cptac_umich_v7_panel_genes.csv
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import importlib.util
import json
from pathlib import Path

import pandas as pd
import cptac

ROOT = PROJECT_ROOT
OUT = ROOT / "v7_plan/results/B8_mixed_cancer"
CACHE = OUT / "raw/D3_cptac_umich_v7_panel_genes.csv"
COHORTS = {"Brca": "BRC", "Coad": "CRC", "Gbm": "GLIOM", "Luad": "LUNGC", "Lscc": "LUNGC", "Ov": "OVC", "Ucec": "ENDC"}
spec = importlib.util.spec_from_file_location("core", ROOT / "revise_plan/scripts/85_oct_v65_mixed_cancer_core_2026-10-15.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
core.PANELS = ROOT / "revise_plan/pilot_ds_dev_only_size/panel_memberships_devonly.csv"  # = v7 panels (checked in B8b)
core.OUT = OUT


def main():
    genes = core.all_panel_genes()
    if CACHE.exists():
        raw = pd.read_csv(CACHE, index_col=0)
    else:
        frames = []
        for cls, label in COHORTS.items():
            prot = getattr(cptac, cls)().get_proteomics("umich")
            prot = prot[[not str(i).endswith(".N") for i in prot.index]]
            names = prot.columns.get_level_values(0) if isinstance(prot.columns, pd.MultiIndex) else prot.columns
            g = prot.T.groupby(list(names)).mean().T.reindex(columns=genes)
            g.index = [f"{cls}:{i}" for i in g.index]
            g.insert(0, "label", label)
            frames.append(g)
        raw = pd.concat(frames)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(CACHE)
    res = core.run_dataset(raw[genes], raw["label"], "D3_cptac_proteomics_v7")
    json.dump(res, open(OUT / "summary_D3_cptac_proteomics_v7.json", "w"), indent=1, default=float)
    r = res.get("panel_26", {})
    print(json.dumps({"n": res["n_patients"], "per_cancer": res["n_per_cancer"], "coverage": res["coverage"], "error": res.get("error"),
                      "step1": r.get("step1_accuracy"), "0.5": r.get("0.5"), "cmp": r.get("comparison")}, indent=1, default=float))


if __name__ == "__main__":
    main()
