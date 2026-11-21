"""Downstream effects of the development-only disease-specific panels (script 91), part 2: the mixed-cancer
pilot datasets (protocol revise_plan/external_mixed_cancer_pilot/PROTOCOL_2026-10-15.md), same rules, with the
development-only lists in step 2 and the 25- and 26-protein lists in step 1.
Existing caches are copied and extended with the newly required genes (gene-labelled Xena calls; older symbols
retried); nothing earlier is overwritten. Outputs carry the suffix _dsdevonly.
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xenaPython as xena

ROOT = PROJECT_ROOT
PILOT = ROOT / "revise_plan/external_mixed_cancer_pilot"
OUT = ROOT / "revise_plan/pilot_ds_dev_only_size"
DEVONLY = OUT / "panel_memberships_devonly.csv"


def load(n, f):
    s = importlib.util.spec_from_file_location(n, ROOT / "revise_plan/scripts" / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


core = load("core", "85_oct_v65_mixed_cancer_core_2026-10-15.py")
d1 = load("d1", "87_oct_v65_mixed_D1_gse2109_expo_2026-10-15.py")
d2 = load("d2", "86_oct_v65_mixed_D2_gdc_tcga_mmrf_2026-10-15.py")
d89 = load("d89", "89_oct_v65_mixed_D4_D5_D2c_2026-10-16.py")
for m in (core, d1.core, d2.core, d89.core):
    m.PANELS = DEVONLY
    m.OUT = OUT
d89.ALIASES.update({"ELOA": "TCEB3"})
GENES = core.all_panel_genes()


def extend(cache_in, fetch_fn):
    cache_out = OUT / f"raw_{cache_in.stem}_plus_devonly.csv"
    if cache_out.exists():
        return pd.read_csv(cache_out, index_col=0)
    base = pd.read_csv(cache_in, index_col=0)
    extra = [g for g in GENES if g not in base.columns]
    add = fetch_fn(list(base.index), extra)
    out = base.join(add[extra])
    out.to_csv(cache_out)
    return out


def summarise(res):
    return {k: {"step1_accuracy": res[k]["step1_accuracy"], "accepted_share": res[k]["0.5"]["accepted_share"],
                "accuracy_accepted": res[k]["0.5"]["accuracy_accepted"], "comparison": res[k]["comparison"]}
            for k in ("panel_25", "panel_26") if k in res} | {"n": res["n_patients"], "coverage": res["coverage"],
                                                              "missing": res["missing_features"], "error": res.get("error")}


def run_D1():
    gsm, source, primary, hist, expr, probes = d1.read_series()
    labels, keep = {}, []
    for i, g in enumerate(gsm):
        if d1.EXCLUDE.search(hist[i] or ""):
            continue
        c = d1.site_to_cancer(primary[i], hist[i])
        if c and d1.site_to_cancer(source[i], hist[i]) == c:
            labels[g] = c
            keep.append(g)
    e = expr.loc[keep]
    if np.nanmax(e.to_numpy()) > 50:
        e = np.log2(e.clip(lower=1))
    best = {}
    for pid, sym in probes.items():
        if pid in e.columns:
            mu = e[pid].mean()
            if sym not in best or mu > best[sym][1]:
                best[sym] = (pid, mu)
    X = pd.DataFrame({s: e[p] for s, (p, _) in best.items()})
    y = pd.Series([labels[g] for g in X.index], index=X.index)
    return core.run_dataset(X, y, "D1_dsdevonly")


def gdc_fetch(samples_by_dataset):
    def f(samples, extra):
        frames = []
        for ds, ss in samples_by_dataset(samples).items():
            df = d2.fetch(ds, extra, "x")
            frames.append(df.loc[[s for s in ss if s in df.index], extra])
        return pd.concat(frames)
    return f


def run_D2(include_mmrf):
    raw0 = pd.read_csv(d2.CACHE, index_col=0)

    def by_ds(samples):
        out = {}
        for s in samples:
            out.setdefault(raw0.loc[s, "dataset"], []).append(s)
        return out
    raw = extend(d2.CACHE, gdc_fetch(by_ds))
    keep = []
    for s, row in raw.iterrows():
        if row["dataset"].startswith("TCGA-"):
            if s[13:15] == ("03" if row["dataset"].startswith("TCGA-LAML") else "01"):
                keep.append((s[:12], s))
        elif include_mmrf and "_1_BM_CD138pos" in s:
            keep.append(("_".join(s.split("_")[:2]), s))
    first = {}
    for pid, s in sorted(keep, key=lambda x: x[1]):
        first.setdefault(pid, s)
    sel = raw.loc[list(first.values())]
    X = sel[GENES].copy()
    X.index = list(first.keys())
    y = pd.Series(sel["label"].to_numpy(), index=X.index)
    return core.run_dataset(X, y, "D2_dsdevonly" if include_mmrf else "D2b_dsdevonly")


def run_89(which):
    hub, ds, lab, log1p = {"D4": d89.labels_D4, "D5": d89.labels_D5, "D2c": d89.labels_D2c}[which]()
    raw = extend(PILOT / "raw" / f"{which}_panel_genes.csv", lambda samples, extra: d89.fetch(hub, ds, samples, extra))
    X = raw[GENES]
    if log1p:
        X = np.log2(X.clip(lower=0) + 1)
    y = pd.Series([lab[s] for s in X.index], index=X.index)
    return core.run_dataset(X, y, f"{which}_dsdevonly")


def main(which):
    res = {"D1": run_D1, "D2": lambda: run_D2(True), "D2b": lambda: run_D2(False),
           "D2c": lambda: run_89("D2c"), "D4": lambda: run_89("D4"), "D5": lambda: run_89("D5")}[which]()
    s = summarise(res)
    json.dump(s, open(OUT / f"summary_{which}_dsdevonly.json", "w"), indent=1, default=float)
    print(which, json.dumps(s, default=float)[:600])


if __name__ == "__main__":
    for w in sys.argv[1:]:
        main(w)
