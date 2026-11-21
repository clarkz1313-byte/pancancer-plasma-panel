"""v7 B10b: chromosome-X fix for the genetics screen (found 15 Oct 2026 during the author's check).

Bug: UKB-PPP pQTL files label chromosome X as "23"; the lung, breast and prostate GWAS (GWAS Catalog harmonised
files) label it "X". Regions were matched on the chromosome label, so every protein whose genome-wide lead SNP is
on chromosome X (XG, CGA, XPNPEP2) found no GWAS SNPs for these three cancers and was dropped as not analysable.
The same mismatch was present in the original 25 x 11 screen (hence 257 of 275 pairs). Glioma (FinnGen) and
ovarian use "23" and were matched correctly; AML, CLL, endometrial, lymphoma, myeloma and colorectal GWAS have no
chromosome-X SNPs (truly not analysable).

Fix: GWAS chromosome "X"/"chrX" recoded to "23" in every GWAS loader of the three v7 forks. Rerun into a new
folder (results/B10_genetics_v2; the first run in results/B10_genetics is kept unchanged for the audit trail):
the v7 Part A pairs plus the six multiclass pairs lost to the bug (XG and CGA x BRC, LUNGC, PRC; panel_source
multiclass_chrX_rescreen), screened with the same Part A method. Joint Bonferroni = 257 + Part A pairs.
Lead checkpoint copied from the first v7 run (leads do not depend on the GWAS).
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import shutil
from pathlib import Path

import pandas as pd

ROOT = PROJECT_ROOT
SC = ROOT / "revise_plan/smr_coloc"
G1 = ROOT / "v7_plan/results/B10_genetics"
G2 = ROOT / "v7_plan/results/B10_genetics_v2"
g1, g2 = (str(p).replace("\\", "/") for p in (G1, G2))
RECODE = [('as.character(hm_chrom)', 'sub("^(chr)?X$", "23", as.character(hm_chrom))'),
          ('as.character(chromosome)', 'sub("^(chr)?X$", "23", as.character(chromosome))')]


def fork(src, dst, subs, note):
    s = (SC / src).read_text(encoding="utf-8")
    for old, new in subs:
        assert s.count(old) == 1, (src, old, s.count(old))
        s = s.replace(old, new)
    (SC / dst).write_text(f"# v7.0b FORK of {src} (original unchanged; written by v7_plan/scripts/B10b_genetics_chrx_fix_v7.py). {note}\n" + s,
                          encoding="utf-8")
    print("wrote", dst)


def main():
    G2.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(G1 / "v7_parta_pairs.csv")
    extra = pd.DataFrame([(p, c, "multiclass_chrX_rescreen") for p in ("XG", "CGA") for c in ("BRC", "LUNGC", "PRC")],
                         columns=pairs.columns)
    pd.concat([pairs, extra]).to_csv(G2 / "v7_parta_pairs.csv", index=False)
    if not (G2 / "parta_pqtl_leads_checkpoint.rds").exists():
        shutil.copy2(G1 / "parta_pqtl_leads_checkpoint.rds", G2 / "parta_pqtl_leads_checkpoint.rds")
    note = "GWAS chromosome X recoded to 23 (UKB-PPP label); results in v7_plan/results/B10_genetics_v2."
    fork("v7_run_parta_coloc.R", "v7b_run_parta_coloc.R",
         [(f'RESULT_DIR <- "{g1}"', f'RESULT_DIR <- "{g2}"'),
          (f'read.csv("{g1}/v7_parta_pairs.csv"', f'read.csv("{g2}/v7_parta_pairs.csv"'), *RECODE], note)
    fork("v7_run_parta_wald_heidi.R", "v7b_run_parta_wald_heidi.R", [(f'RESULT_DIR <- "{g1}"', f'RESULT_DIR <- "{g2}"'), *RECODE], note)
    fork("v7_run_tier1_combined.R", "v7b_run_tier1_combined.R", [(f'V5G <- "{g1}"', f'V5G <- "{g2}"'), *RECODE], note)
    print(f"pairs: {len(pairs)} + {len(extra)} rescreen")


if __name__ == "__main__":
    main()
