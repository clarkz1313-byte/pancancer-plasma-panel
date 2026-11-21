"""v7 B8e: sensitivity for audit SOL-D11. CPTAC-2 breast cases whose recorded diagnosis is "Not Reported" (16 of
133) are excluded, so breast = recorded breast carcinoma only, as the Addendum B text states; D6, D7 and D8 rerun
with otherwise identical rules (script B8b). Outputs summary_<D>_gdc_nontcga_strictBRC.json and per-patient files.
"""
import collections
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import B8b_gdc_nontcga_v7 as b  # noqa: E402

_old = b.lab_cptac2


def lab_cptac2_strict(dx, site):
    if "Breast" in site and (not dx or dx in ("Not Reported", "Unknown", "nan")):
        return None
    return _old(dx, site)


b.PROJ["CPTAC-2"] = (lab_cptac2_strict, b.SOLID)


def get(project, genes):
    lab = b.labels(project)
    X = pd.read_csv(b.RAW / f"{project}_panel_genes.csv", index_col=0)
    X = X.loc[[s for s in X.index if s in lab]]
    X.index = [f"{project}|{s}" for s in X.index]
    return X, pd.Series([lab[s.split("|", 1)[1]] for s in X.index], index=X.index)


def main(which):
    genes = b.core.all_panel_genes()
    parts = [b.get_cllmap(genes) if p == "CLLMAP" else get(p, genes) for p in b.SETS[which]]
    X = pd.concat([p[0] for p in parts])[genes]
    y = pd.concat([p[1] for p in parts])
    res = b.core.run_dataset(X, y, f"{which}_gdc_nontcga_strictBRC")
    res["label_counts_all"] = dict(collections.Counter(y))
    json.dump(res, open(b.OUT / f"summary_{which}_gdc_nontcga_strictBRC.json", "w"), indent=1, default=float)
    r = res["panel_26"]
    print(which, res["n_patients"], res["n_per_cancer"].get("BRC"), round(r["step1_accuracy"], 4), r["0.5"]["accepted"],
          round(r["0.5"]["accuracy_accepted"], 4), {k: r["comparison"][k] for k in ("wrong_stricter", "wrong_step2", "p_two_sided", "gain_points")})


if __name__ == "__main__":
    for w in sys.argv[1:]:
        main(w)
