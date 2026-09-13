#!/usr/bin/env python3
"""
Isolate WHY the over-representation result changed.

The corrected run (2026-09-05) differed from the withdrawn run in FOUR ways at
once, which makes the shift impossible to attribute from the two result tables
alone:

  1. background      genome (18,737 / 19,960) -> measured assay (1,463)
  2. GO scope        BP + MF + CC             -> BP only
  3. annotation src  org.Hs.eg.db / ReactomePA -> Enrichr GMT libraries
  4. implementation  clusterProfiler::enrichGO -> hand-rolled hypergeometric

This script holds 2, 3 and 4 fixed -- same GMT, same hypergeometric code --
and varies ONLY the background. That isolates the effect of the change that
was actually intended, and says how much of the observed shift it explains.

Output: ora_background_isolation_2026-09-05.csv  (+ console summary)
"""
from pathlib import Path
from math import comb
import csv
import sys

ROOT = Path("e:/Proteomics")
GMT_DIR = ROOT / "references" / "gene_sets"
ORA = ROOT / "revision_package_2026-09-04" / "05_data_audit" / "ora_assay_background_2026-09-05"
OUT = ORA / "ora_background_isolation_2026-09-05.csv"

GMTS = {
    "GO_BP_2023": GMT_DIR / "GO_Biological_Process_2023.gmt",
    "Reactome_2022": GMT_DIR / "Reactome_2022.gmt",
}


def read_gmt(path):
    sets = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = {g.strip().upper() for g in parts[2:] if g.strip()}
            if genes:
                sets[name] = genes
    return sets


def hypergeom_sf(k, N, K, n):
    """P(X >= k) for drawing n from N with K successes."""
    if k <= 0:
        return 1.0
    total = 0.0
    denom = comb(N, n)
    if denom == 0:
        return 1.0
    hi = min(K, n)
    for i in range(k, hi + 1):
        total += comb(K, i) * comb(N - K, n - i)
    return total / denom


def bh(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = min(prev, pvals[i] * m / (rank + 1))
        q[i] = val
        prev = val
    return q


def main():
    # assay universe and the 25-protein panel, as used by the corrected run
    universe = [r["protein"].strip().upper() if "protein" in r else
                list(r.values())[0].strip().upper()
                for r in csv.DictReader(open(ORA / "assay_universe_1463.csv", encoding="utf-8-sig"))]
    universe = {u for u in universe if u}

    panels = {}
    for r in csv.DictReader(open(ORA / "panel_memberships.csv", encoding="utf-8-sig")):
        p = r.get("panel") or list(r.values())[0]
        g = (r.get("protein") or list(r.values())[1]).strip().upper()
        panels.setdefault(p.strip(), set()).add(g)

    panel = panels.get("multiclass_25")
    if not panel:
        print("PANEL NOT FOUND; available:", list(panels)[:6], file=sys.stderr)
        return
    print(f"assay universe: {len(universe)}   multiclass_25: {len(panel)}\n")

    rows = []
    for db, path in GMTS.items():
        if not path.exists():
            print(f"skip {db}: {path} not found")
            continue
        sets = read_gmt(path)
        gmt_genes = set().union(*sets.values())

        # --- background A: measured assay (the corrected run) ---
        uni_a = universe & gmt_genes
        pan_a = panel & uni_a
        # --- background B: full annotation space (approximates the old run) ---
        uni_b = gmt_genes
        pan_b = panel & uni_b

        res_a, res_b = [], []
        for name, genes in sets.items():
            ka = len(pan_a & genes)
            if ka:
                Ka = len(genes & uni_a)
                res_a.append((name, ka, Ka,
                              hypergeom_sf(ka, len(uni_a), Ka, len(pan_a))))
            kb = len(pan_b & genes)
            if kb:
                Kb = len(genes & uni_b)
                res_b.append((name, kb, Kb,
                              hypergeom_sf(kb, len(uni_b), Kb, len(pan_b))))

        qa = bh([r[3] for r in res_a])
        qb = bh([r[3] for r in res_b])
        na = sum(1 for v in qa if v < 0.05)
        nb = sum(1 for v in qb if v < 0.05)

        print(f"=== {db} ===")
        print(f"  assay background   : universe {len(uni_a):5d}  panel {len(pan_a):2d}"
              f"  terms {len(res_a):5d}  min p {min(r[3] for r in res_a):.3e}"
              f"  min q {min(qa):.4f}  sig {na}")
        print(f"  full-annot backgrd : universe {len(uni_b):5d}  panel {len(pan_b):2d}"
              f"  terms {len(res_b):5d}  min p {min(r[3] for r in res_b):.3e}"
              f"  min q {min(qb):.4f}  sig {nb}")
        print()

        for (name, k, K, p), q in zip(res_a, qa):
            rows.append(dict(database=db, background="measured_assay", term=name,
                             overlap=k, term_size_in_background=K,
                             universe_size=len(uni_a), p_value=p, bh_q=q))
        for (name, k, K, p), q in zip(res_b, qb):
            rows.append(dict(database=db, background="full_annotation", term=name,
                             overlap=k, term_size_in_background=K,
                             universe_size=len(uni_b), p_value=p, bh_q=q))

    if rows:
        with open(OUT, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", OUT)


if __name__ == "__main__":
    main()
