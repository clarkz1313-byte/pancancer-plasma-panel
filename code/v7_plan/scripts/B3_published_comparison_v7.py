"""v7 B3: comparison with the published U-CAN models (Bueno Alvez et al. 2023).
Multiclass: the final 26-protein model (B5 metrics; macro recall = balanced accuracy) against the published
12-, 36-, 83- and 1,463-protein multiclass rows (Supplementary Data 6 macro averages, as Appendix Table A2).
Disease-specific: copies the pilot comparison for the final disease-specific panels (revise_plan script 95,
rule of script 55: smallest published model with at least as many proteins; same-sex scoring for the five
sex-restricted cancers; equal if |difference| < 0.005), which used exactly these panels.
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import json, shutil
from pathlib import Path
import pandas as pd
V7 = (PROJECT_ROOT / "v7_plan"); OUT = V7 / "results/B3_published_comparison"
m = json.load(open(V7 / "results/B5_internal_two_step_summary.json"))["multiclass"]["metrics"]
pub = pd.DataFrame([["Published, 12 proteins", 0.875, 0.565, 0.594, 0.333], ["Published, 36 proteins", 0.938, 0.669, 0.699, 0.452],
                    ["Published, 83 proteins", 0.962, 0.729, 0.775, 0.571], ["Published, 1,463 proteins", 0.979, 0.816, 0.849, 0.608]],
                   columns=["panel", "macro_auc", "macro_f1", "macro_recall", "min_class_recall"])
ours = pd.DataFrame([["This study, 26 proteins", m["macro_auc"], m["macro_f1"], m["balanced_accuracy"], m["min_class_recall"]]], columns=pub.columns)
t = pd.concat([pub, ours], ignore_index=True)
t.to_csv(OUT / "B3_multiclass_vs_published.csv", index=False)
src = (PROJECT_ROOT / "revise_plan/pilot_ds_dev_only_size/published_comparison_published_vs_devonly.csv")
ds = pd.read_csv(src); ds = ds[ds.version == "dev_only"].drop(columns="version")
ds.to_csv(OUT / "B3_disease_specific_vs_published.csv", index=False)
print(t.round(3).to_string(index=False)); print(ds.verdict.value_counts().to_dict())
