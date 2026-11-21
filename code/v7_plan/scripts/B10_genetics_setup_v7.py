"""v7 B10 set-up: genetics pair list and forks for the final v7 panels (originals unchanged).

Pairs (same rules as script 50): the 25 x 11 multiclass screen (257 analysable pairs, reused from
revise_plan/smr_coloc/results) plus, through the Part A pipeline (same method as the original screen):
  - MSMB, the 26th multiclass protein, against all 11 cancers with a GWAS (panel_source multiclass_26);
  - each disease-specific protein not in the 26, against its own cancer (CVX excluded: no compatible GWAS).
Forks (written next to the originals in revise_plan/smr_coloc because they source shared paths; outputs go to
v7_plan/results/B10_genetics):
  v5_run_parta_coloc.R       -> v7_run_parta_coloc.R       (pairs from v7_parta_pairs.csv)
  v5_run_parta_wald_heidi.R  -> v7_run_parta_wald_heidi.R  (joint Bonferroni = 257 + v7 Part A pairs)
  v5_run_tier1_combined.R    -> v7_run_tier1_combined.R    (Part A rows labelled by panel_source)
The genome-wide lead SNP and region of each protein do not depend on the panel, so the v5 lead checkpoint is
copied into the v7 folder (its region files are read from the v5 folder, read-only); only proteins without a
lead are scanned.
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
SMR_DATA_ROOT = _Path(_os.environ.get("SMR_DATA_ROOT", PROJECT_ROOT / "smr_data"))
import os
import shutil
from pathlib import Path

import pandas as pd

ROOT = PROJECT_ROOT
V7 = ROOT / "v7_plan"
OUTG = V7 / "results/B10_genetics"
SC = ROOT / "revise_plan/smr_coloc"
V5G = ROOT / "revise_plan/v5_seed52_common_split/genetics"
GWAS_CODES = ["AML", "BRC", "CLL", "CRC", "ENDC", "GLIOM", "LUNGC", "LYMPH", "MYEL", "OVC", "PRC"]
CODES = ["AML", "BRC", "CLL", "CRC", "CVX", "DLBCL", "ENDC", "GLIOM", "LUNGC", "MYEL", "OVC", "PRC"]
V7G = str(OUTG).replace("\\", "/")


def pairs():
    hub = pd.read_csv(V7 / "panels/panel_memberships_v7.csv")
    mc = set(hub[hub.panel == "multiclass_26"].protein)
    assert len(mc) == 26 and "MSMB" in mc
    rows = [("MSMB", c, "multiclass_26_new") for c in GWAS_CODES]
    for c in CODES:
        if c == "CVX":
            continue
        code = "LYMPH" if c == "DLBCL" else c
        rows += [(p, code, "disease_specific") for p in hub[hub.panel == f"single_{c}"].protein if p not in mc]
    df = pd.DataFrame(rows, columns=["protein", "cancer", "panel_source"]).drop_duplicates(["protein", "cancer"])
    OUTG.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTG / "v7_parta_pairs.csv", index=False)
    cached = {f.split("_pqtl")[0] for f in os.listdir(str(SMR_DATA_ROOT / "pqtl_ukbpp")) if f.endswith("_pqtl.tsv.gz")}
    missing = sorted(set(df.protein) - cached)
    print(f"v7 Part A pairs {len(df)} ({df.protein.nunique()} proteins): "
          f"{(df.panel_source == 'multiclass_26_new').sum()} MSMB, {(df.panel_source == 'disease_specific').sum()} disease-specific")
    print("pQTL missing:", missing)
    assert not missing
    return df


def fork(src, dst, subs, note):
    s = (SC / src).read_text(encoding="utf-8")
    for old, new in subs:
        assert s.count(old) >= 1, (src, old)
        s = s.replace(old, new)
    (SC / dst).write_text(f"# v7.0 FORK of {src} (original unchanged; written by v7_plan/scripts/B10_genetics_setup_v7.py). {note}\n" + s,
                          encoding="utf-8")
    print("wrote", dst)


def main():
    pairs()
    if not (OUTG / "parta_pqtl_leads_checkpoint.rds").exists():
        shutil.copy2(V5G / "parta_pqtl_leads_checkpoint.rds", OUTG / "parta_pqtl_leads_checkpoint.rds")
    v5g = str(V5G).replace("\\", "/")
    fork("v5_run_parta_coloc.R", "v7_run_parta_coloc.R", [
        (f'RESULT_DIR <- "{v5g}"', f'RESULT_DIR <- "{V7G}"'),
        (f'PAIRS <- read.csv("{v5g}/v5_disease_specific_pairs.csv", stringsAsFactors = FALSE)',
         f'PAIRS <- read.csv("{V7G}/v7_parta_pairs.csv", stringsAsFactors = FALSE)'),
    ], "Screens the v7 Part A pairs (MSMB x 11 + disease-specific); writes to v7_plan/results/B10_genetics.")
    fork("v5_run_parta_wald_heidi.R", "v7_run_parta_wald_heidi.R", [
        (f'RESULT_DIR <- "{v5g}"', f'RESULT_DIR <- "{V7G}"'),
    ], "Reads/writes the v7 genetics folder; joint Bonferroni = 257 + v7 Part A pairs.")
    fork("v5_run_tier1_combined.R", "v7_run_tier1_combined.R", [
        (f'V5G <- "{v5g}"', f'V5G <- "{V7G}"'),
        ('mutate(panel_source = "12_single_cancer_panels")',
         'left_join(read_csv(file.path(V5G, "v7_parta_pairs.csv"), show_col_types = FALSE), by = c("protein","cancer"))'),
        ('file.path(V5G, "tier1_combined_v5.csv")', 'file.path(V5G, "tier1_combined_v7.csv")'),
    ], "Combines the 257 multiclass pairs with the v7 Part A pairs; output tier1_combined_v7.csv.")


if __name__ == "__main__":
    main()
