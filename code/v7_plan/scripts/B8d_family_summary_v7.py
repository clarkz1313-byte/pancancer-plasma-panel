"""v7 B8d: overall summary of the mixed-cancer series by independent family (Addendum D rule, post hoc combination).
Collects every series run with the final panels (26-protein step 1), tags family and eligibility (>= 7 cancers,
informative), and combines the primary series of each family by DerSimonian-Laird random effects for
(a) gain: accuracy of accepted minus all predictions, and (b) accuracy difference between step 2 and the
stricter cutoff at equal coverage (percentage points; SE from the bootstrap 95% interval / 3.92).
Leave-one-family-out and substitution checks (each TCGA series in turn; D8 for D7).
Outputs: results/B8_mixed_cancer/B8_all_series.csv, B8_family_summary.json
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = PROJECT_ROOT
PIL = ROOT / "revise_plan/pilot_ds_dev_only_size"
OUT = ROOT / "v7_plan/results/B8_mixed_cancer"

SERIES = [  # id, label, family, file, primary
    ("TCGA_xena", "TCGA, original processing (Xena)", "TCGA", None, True),
    ("D2c", "TCGA, Toil processing", "TCGA", PIL / "summary_D2c_dsdevonly.json", False),
    ("D2b", "TCGA, GDC processing", "TCGA", PIL / "summary_D2b_dsdevonly.json", False),
    ("D2", "TCGA GDC + MMRF myeloma", "TCGA", PIL / "summary_D2_dsdevonly.json", False),
    ("D4", "PCAWG (mostly TCGA donors)", "TCGA", PIL / "summary_D4_dsdevonly.json", False),
    ("D1", "expO GSE2109 (microarray)", "expO", PIL / "summary_D1_dsdevonly.json", True),
    ("D7", "GDC non-TCGA, 11 cancers", "GDC non-TCGA", OUT / "summary_D7_gdc_nontcga.json", True),
    ("D8", "GDC non-TCGA + CLL-map, 12 cancers", "GDC non-TCGA", OUT / "summary_D8_gdc_nontcga.json", False),
    ("D6", "CPTAC RNA-seq, 6 cancers", "GDC non-TCGA", OUT / "summary_D6_gdc_nontcga.json", False),
    ("D5", "CCLE cell lines", "Cell lines", PIL / "summary_D5_dsdevonly.json", True),
    ("D9a", "Platelet RNA GSE183635, HiSeq 2500", "Platelet RNA", OUT / "summary_D9a_platelet_tep.json", True),
    ("D9b", "Platelet RNA GSE183635, both sequencers", "Platelet RNA", OUT / "summary_D9b_platelet_tep.json", False),
    ("D3", "CPTAC proteomics", "GDC non-TCGA", None, False),
]


def row(sid, label, fam, f, primary):
    base = {"id": sid, "series": label, "family": fam, "primary": primary}
    if sid == "TCGA_xena":
        r = json.load(open(PIL / "summary_downstream_internal_tcga.json"))["tcga_xena"]["dev_only"]["panel_26"]
        return base | {"n": 4665, "cancers": 10, "step1": r["step1_accuracy"], "accepted_share": r["accepted"] / 4665,
                       "acc_accepted": r["accuracy_accepted"], **r["comparison"]}
    if sid == "D3":
        return base | {"n": 730, "cancers": 6, "note": "uninformative: 7 of 26 step-1 proteins measurable"}
    r = json.load(open(f))
    if "panel_26" in r and "n_patients" in r:  # full run_dataset output
        p, n, k = r["panel_26"], r["n_patients"], len(r["n_per_cancer"])
        sh, aa = p["0.5"]["accepted_share"], p["0.5"]["accuracy_accepted"]
    elif "panel_26" not in r:
        return base | {"n": r["n_patients"], "cancers": len(r["n_per_cancer"]),
                       "note": f"uninformative: {r['coverage']['multiclass_26']} step-1 proteins measurable; {r.get('error')}"}
    else:  # compact dsdevonly summary
        p, n = r["panel_26"], r["n"]
        k = sum(1 for c in r["coverage"] if not c.startswith("multiclass"))
        sh, aa = p["accepted_share"], p["accuracy_accepted"]
    return base | {"n": n, "cancers": k, "step1": p["step1_accuracy"], "accepted_share": sh, "acc_accepted": aa,
                   **p["comparison"]}


def dl(est, se):
    est, se = np.asarray(est, float), np.asarray(se, float)
    w = 1 / se ** 2
    fixed = (w * est).sum() / w.sum()
    q = (w * (est - fixed) ** 2).sum()
    k = len(est)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())) if k > 1 else 0.0
    wr = 1 / (se ** 2 + tau2)
    m = (wr * est).sum() / wr.sum()
    s = np.sqrt(1 / wr.sum())
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return {"estimate": m, "ci": [m - 1.96 * s, m + 1.96 * s], "tau2": tau2, "I2": i2, "k": k}


def pool(df, ids):
    d = df.set_index("id").loc[ids]
    g_se = (d.gain_ci.str[1] - d.gain_ci.str[0]) / 3.92
    a_se = (d.acc_ci.str[1] - d.acc_ci.str[0]) / 3.92
    return {"series": ids, "gain_points": dl(d.gain_points, g_se), "accuracy_diff_points": dl(d.accuracy_diff_points, a_se)}


def main():
    df = pd.DataFrame([row(*s) for s in SERIES])
    df["eligible_7plus"] = (df.cancers >= 7) & df.note.isna()
    df.drop(columns=[c for c in ("rel_ci",) if c in df]).to_csv(OUT / "B8_all_series.csv", index=False)
    prim = list(df[df.primary & df.eligible_7plus].id)
    out = {"primary_per_family": prim, "main": pool(df, prim), "leave_one_family_out": {}, "substitutions": {}}
    for x in prim:
        out["leave_one_family_out"][x] = pool(df, [p for p in prim if p != x])
    for old, new in [("TCGA_xena", "D2c"), ("TCGA_xena", "D2b"), ("TCGA_xena", "D2"), ("TCGA_xena", "D4"), ("D7", "D8")]:
        out["substitutions"][f"{old}->{new}"] = pool(df, [new if p == old else p for p in prim])
    json.dump(out, open(OUT / "B8_family_summary.json", "w"), indent=1, default=float)
    cols = ["id", "family", "n", "cancers", "step1", "accepted_share", "acc_accepted", "wrong_stricter", "wrong_step2",
            "accuracy_diff_points", "p_two_sided", "gain_points"]
    print(df[cols + ["eligible_7plus", "primary"]].round(3).to_string(index=False))

    def fmt(r):
        return f"{r['estimate']:+.2f} ({r['ci'][0]:+.2f} to {r['ci'][1]:+.2f}), I2 {r['I2']:.2f}"
    print("\nmain", prim, "\n  gain", fmt(out["main"]["gain_points"]), "\n  acc diff", fmt(out["main"]["accuracy_diff_points"]))
    for k, v in {**out["leave_one_family_out"], **out["substitutions"]}.items():
        print(f"  {k:14s} gain {fmt(v['gain_points'])} | acc diff {fmt(v['accuracy_diff_points'])}")


if __name__ == "__main__":
    main()
