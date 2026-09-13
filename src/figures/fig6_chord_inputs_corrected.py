#!/usr/bin/env python3
"""
Build the three CSVs that fig6_chord_network.py expects, from the CORRECTED
assay-background ORA rather than the withdrawn genome-background run.

The chord panel is a membership diagram: it draws which panel proteins belong
to which annotated terms. That is descriptive and survives the background
correction unchanged -- only the choice of WHICH terms to draw came from the
withdrawn statistics, so only the term list is rebuilt here.

Terms are now selected by nominal P within the corrected run (top 6 GO BP,
top 5 KEGG, top 5 Reactome, each requiring >= 2 panel proteins so a chord has
something to connect). No significance is claimed or implied by inclusion.

Output: figurev5/output/_chord_inputs_corrected/{go,kegg,reactome}.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path("e:/Proteomics")
ORA = ROOT / "revision_package_2026-09-04" / "05_data_audit" / "ora_assay_background_2026-09-05"
OUT = ROOT / "figurev5" / "output" / "_chord_inputs_corrected"
OUT.mkdir(parents=True, exist_ok=True)

t = pd.read_csv(ORA / "ora_terms_with_panel_overlap.csv")
t = t[(t.panel == "multiclass_25") & (t.overlap_count >= 2)].copy()


def clean(s):
    s = str(s).split(" R-HSA-")[0]
    return s.split(" (GO:")[0]


t["Description"] = t.term.map(clean)
t["geneID"] = t.overlap_genes.str.replace(";", "/", regex=False)
t["marker_symbols"] = t.overlap_genes.str.replace(";", ", ", regex=False)
t["Count"] = t.overlap_count
t["pvalue"] = t.p_value
t["p.adjust"] = t.p_adjust_bh
t["ID"] = t.term

sel = {
    "GO_BP_2023": ("go.csv", 6),
    "KEGG_2021_Human": ("kegg.csv", 5),
    "Reactome_2022": ("reactome.csv", 5),
}

for db, (fname, n) in sel.items():
    d = t[t.database == db].nsmallest(n, "pvalue")
    cols = ["ID", "Description", "pvalue", "p.adjust", "geneID",
            "marker_symbols", "Count"]
    d[cols].to_csv(OUT / fname, index=False)
    print(f"{db:18s} -> {fname}  ({len(d)} terms, "
          f"min p {d.pvalue.min():.2e})")

print("\nwrote", OUT)
