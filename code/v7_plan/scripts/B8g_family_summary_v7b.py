"""v7 B8g: revised summary of the mixed-cancer series after the independent audit (supersedes B8d's outputs for
reporting; B8d and its outputs are kept unchanged).

Changes (audit IDs):
- SOL-D09: the "TCGA GDC + MMRF instead of TCGA Xena" substitution is dropped from the pooled checks, because D7
  also contains the 762 MMRF patients (double counting); D2 stays as a descriptive TCGA-family series.
- SOL-D10: Hartung-Knapp-Sidik-Jonkman intervals (t with k - 1 df) are added next to DerSimonian-Laird.
- SOL-D08: CPTAC proteomics (D3) is reported with the final v7 panels (B8f), as a low-coverage series outside the
  >= 7-cancer synthesis.
- SOL-D11: strict-breast versions of D6-D8 (B8e) are listed as sensitivity rows.
- Bootstrap p of 0 is reported as "<0.001".
- Supplementary comparator (author decision, following audit SOL-D18): the stricter multiclass cutoff applied
  within each step-1 predicted cancer (same number accepted per predicted cancer as step 2), paired bootstrap
  2,000 resamples, seed 20261003; computed for the internal test set and every informative series.
Outputs: results/B8_mixed_cancer/B8_all_series_v7b.csv, B8_family_summary_v7b.json, B8_class_matched_comparator.csv
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = PROJECT_ROOT
OUT = ROOT / "v7_plan/results/B8_mixed_cancer"
PIL = OUT / "pilot_series"
B = 2000


def load_b8d():
    import importlib.util
    s = importlib.util.spec_from_file_location("b8d", ROOT / "v7_plan/scripts/B8d_family_summary_v7.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def dl_hk(est, se):
    est, se = np.asarray(est, float), np.asarray(se, float)
    k = len(est)
    w = 1 / se ** 2
    fixed = (w * est).sum() / w.sum()
    q = (w * (est - fixed) ** 2).sum()
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())) if k > 1 else 0.0
    wr = 1 / (se ** 2 + tau2)
    m = (wr * est).sum() / wr.sum()
    s_dl = np.sqrt(1 / wr.sum())
    s_hk = np.sqrt((wr * (est - m) ** 2).sum() / ((k - 1) * wr.sum())) if k > 1 else np.nan
    t = stats.t.ppf(0.975, k - 1) if k > 1 else np.nan
    return {"estimate": m, "dl_ci": [m - 1.96 * s_dl, m + 1.96 * s_dl], "hk_ci": [m - t * s_hk, m + t * s_hk],
            "tau2": tau2, "I2": max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0, "k": k}


def pool(df, ids):
    d = df.set_index("id").loc[ids]
    g_se = (d.gain_ci.str[1] - d.gain_ci.str[0]) / 3.92
    a_se = (d.acc_ci.str[1] - d.acc_ci.str[0]) / 3.92
    return {"series": ids, "gain_points": dl_hk(d.gain_points, g_se), "accuracy_diff_points": dl_hk(d.accuracy_diff_points, a_se)}


def keep_within_class(pred, conf, gate):
    keep = np.zeros(len(pred), bool)
    for c in np.unique(pred):
        ii = np.flatnonzero(pred == c)
        keep[ii[np.argsort(-conf[ii], kind="stable")[:int(gate[ii].sum())]]] = True
    return keep


def class_matched(f, sid):
    d = pd.read_csv(f)
    sfx = "_26" if "pred_26" in d else ""
    pred, conf, second = d[f"pred{sfx}"].to_numpy(), d[f"conf{sfx}"].to_numpy(), d[f"second{sfx}"].to_numpy()
    correct = pred == d["true"].to_numpy()
    gate = second >= 0.5
    keep = keep_within_class(pred, conf, gate)
    rng = np.random.default_rng(20261003)
    diff = []
    for _ in range(B):
        i = rng.integers(0, len(d), len(d))
        g = second[i] >= 0.5
        if not g.any():
            continue
        k = keep_within_class(pred[i], conf[i], g)
        diff.append(100 * (correct[i][g].mean() - correct[i][k].mean()))
    diff = np.asarray(diff)
    p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
    return {"id": sid, "n": len(d), "accepted": int(gate.sum()), "wrong_class_matched_cutoff": int((~correct[keep]).sum()),
            "wrong_step2": int((~correct[gate]).sum()), "difference_points": 100 * (correct[gate].mean() - correct[keep].mean()),
            "ci_lo": float(np.percentile(diff, 2.5)), "ci_hi": float(np.percentile(diff, 97.5)),
            "p": "<0.001" if p == 0 else round(p, 3)}


def main():
    b8d = load_b8d()
    df = pd.DataFrame([b8d.row(*s) for s in b8d.SERIES if s[0] != "D3"])
    extra = []
    r3 = json.load(open(OUT / "summary_D3_cptac_proteomics_v7.json"))
    p3 = r3["panel_26"]
    extra.append({"id": "D3", "series": "CPTAC proteomics, final panels (7 of 26 step-1 proteins)", "family": "GDC non-TCGA",
                  "primary": False, "n": r3["n_patients"], "cancers": len(r3["n_per_cancer"]), "step1": p3["step1_accuracy"],
                  "accepted_share": p3["0.5"]["accepted_share"], "acc_accepted": p3["0.5"]["accuracy_accepted"],
                  **p3["comparison"], "note": "low coverage; below the 7-cancer rule"})
    for w in ("D6", "D7", "D8"):
        r = json.load(open(OUT / f"summary_{w}_gdc_nontcga_strictBRC.json"))
        p = r["panel_26"]
        extra.append({"id": f"{w}_strictBRC", "series": f"{w}, breast restricted to recorded carcinoma", "family": "GDC non-TCGA",
                      "primary": False, "n": r["n_patients"], "cancers": len(r["n_per_cancer"]), "step1": p["step1_accuracy"],
                      "accepted_share": p["0.5"]["accepted_share"], "acc_accepted": p["0.5"]["accuracy_accepted"],
                      **p["comparison"], "note": "sensitivity (audit SOL-D11)" + ("; below the 7-cancer rule" if w == "D6" else "")})
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    df["eligible_7plus"] = (df.cancers >= 7) & ~df.id.isin(["D9a", "D9b"]) & df.step1.notna()
    df["p_report"] = df.p_two_sided.map(lambda x: "" if pd.isna(x) else ("<0.001" if x == 0 else f"{x:.3f}"))
    df.drop(columns=[c for c in ("rel_ci",) if c in df]).to_csv(OUT / "B8_all_series_v7b.csv", index=False)

    prim = ["TCGA_xena", "D1", "D7", "D5"]
    out = {"primary_per_family": prim, "main": pool(df, prim), "leave_one_family_out": {}, "substitutions": {},
           "dropped_substitution": "TCGA_xena->D2 (GDC + MMRF) shares 762 MMRF patients with D7 (audit SOL-D09)"}
    for x in prim:
        out["leave_one_family_out"][x] = pool(df, [p for p in prim if p != x])
    for old, new in [("TCGA_xena", "D2c"), ("TCGA_xena", "D2b"), ("TCGA_xena", "D4"), ("D7", "D8"), ("D7", "D7_strictBRC")]:
        out["substitutions"][f"{old}->{new}"] = pool(df, [new if p == old else p for p in prim])

    files = {"internal": ROOT / "v7_plan/results/B5_internal_per_patient.csv",
             "TCGA_xena": ROOT / "v7_plan/results/B7_tcga/per_patient_tcga_xena_dev_only.csv",
             "D2c": PIL / "per_patient_D2c_dsdevonly.csv", "D2b": PIL / "per_patient_D2b_dsdevonly.csv",
             "D2": PIL / "per_patient_D2_dsdevonly.csv", "D4": PIL / "per_patient_D4_dsdevonly.csv",
             "D1": PIL / "per_patient_D1_dsdevonly.csv", "D7": OUT / "per_patient_D7_gdc_nontcga.csv",
             "D8": OUT / "per_patient_D8_gdc_nontcga.csv", "D6": OUT / "per_patient_D6_gdc_nontcga.csv",
             "D5": PIL / "per_patient_D5_dsdevonly.csv", "D3": OUT / "per_patient_D3_cptac_proteomics_v7.csv"}
    cm = pd.DataFrame([class_matched(f, k) for k, f in files.items()])
    cm.to_csv(OUT / "B8_class_matched_comparator.csv", index=False)
    out["class_matched_comparator"] = cm.to_dict("records")
    json.dump(out, open(OUT / "B8_family_summary_v7b.json", "w"), indent=1, default=float)

    def fmt(r):
        return f"{r['estimate']:+.2f} DL ({r['dl_ci'][0]:+.2f} to {r['dl_ci'][1]:+.2f}) HK ({r['hk_ci'][0]:+.2f} to {r['hk_ci'][1]:+.2f})"
    print("main gain", fmt(out["main"]["gain_points"]), "\nmain diff", fmt(out["main"]["accuracy_diff_points"]))
    for k, v in {**out["leave_one_family_out"], **out["substitutions"]}.items():
        print(f"  {k:18s} diff {fmt(v['accuracy_diff_points'])}")
    print(cm.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
