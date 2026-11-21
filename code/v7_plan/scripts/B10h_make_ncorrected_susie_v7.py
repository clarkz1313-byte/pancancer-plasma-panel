"""v7 B10h: write sample-size-corrected copies of the SuSiE scripts B10d, B10e, B10f and B10g (originals unchanged).

Why (Sol recheck N4, 18 Oct): the originals pass
  - breast cancer GWAS N = 139,274 (76,192 cases), the GWAS Catalog discovery subset, whereas the beta/SE used are the
    combined OncoArray + iCOGS + GWAS overall columns (122,977 cases + 105,974 controls = 228,951);
  - pQTL N = 54,219 for every protein, whereas the UKB-PPP files used give N per protein (column N, constant within
    each file): LEP 33,693; BMP4 33,372; MSMB 33,529; CD74 33,962; IL17A 33,084.
N enters susie_rss/runsusie and the LD-mismatch diagnostic, not coloc.abf with beta/varbeta (so the single-signal
PP.H4 values cannot change).

Copies: B10h (= B10d), B10i (= B10e), B10j (= B10f), B10k (= B10g), all writing to results/B10_genetics_ncorrected/.
The LD matrices do not depend on N: the cached UKB and 1000 Genomes GRCh38 matrices are copied from
B10_genetics_final/ and reused (the copies only rebuild one if it is missing).
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
V7 = HERE.parent
OLD = V7 / "results/B10_genetics_final"
NEW = V7 / "results/B10_genetics_ncorrected"
PQTL = 'PQTL_NS <- c(LEP = 33693, BMP4 = 33372, MSMB = 33529, CD74 = 33962, IL17A = 33084)  # UKB-PPP file column N'
LOOP = "  prot <- pr[1]; cn <- pr[2]; cfg <- GW[[cn]]"
JOBS = {"B10d_susie_coloc_v7.R": ("B10h_susie_coloc_ncorr_v7.R", "B10d_susie_coloc.csv", "B10h_susie_coloc_ncorr.csv"),
        "B10e_susie_coloc_ukbld_v7.R": ("B10i_susie_coloc_ukbld_ncorr_v7.R", "B10e_susie_coloc_ukbld.csv", "B10i_susie_coloc_ukbld_ncorr.csv"),
        "B10f_susie_coloc_1kg38_v7.R": ("B10j_susie_coloc_1kg38_ncorr_v7.R", "B10f_susie_coloc_1kg38.csv", "B10j_susie_coloc_1kg38_ncorr.csv"),
        "B10g_susie_pips_1kg38_v7.R": ("B10k_susie_pips_1kg38_ncorr_v7.R", "B10g_susie_pips_1kg38.csv", "B10k_susie_pips_1kg38_ncorr.csv")}


def rep(s, old, new, n=1):
    assert s.count(old) == n, (old, s.count(old))
    return s.replace(old, new)


def main():
    NEW.mkdir(exist_ok=True)
    for d in ("ukbld_tmp", "kg38_tmp"):
        if not (NEW / d).exists():
            shutil.copytree(OLD / d, NEW / d)
    for src, (dst, out_old, out_new) in JOBS.items():
        s = (HERE / src).read_text(encoding="utf-8")
        s = rep(s, 'OUTD <- file.path(ROOT, "v7_plan/results/B10_genetics_final")',
                'OUTD <- file.path(ROOT, "v7_plan/results/B10_genetics_ncorrected")')
        s = rep(s, 'n = 139274, nc = 76192', 'n = 228951, nc = 122977')
        s = rep(s, "PQTL_N <- 54219", PQTL)
        s = rep(s, LOOP, LOOP + "; PQTL_N <- PQTL_NS[[prot]]")
        s = rep(s, f'"{out_old}"', f'"{out_new}"')
        if src == "B10g_susie_pips_1kg38_v7.R":
            s = rep(s, '"B10g_check.csv"', '"B10k_check.csv"')
        if "  rc <- system2(" in s:
            s = rep(s, "  rc <- system2(", '  rc <- character(0); if (!file.exists(file.path(TMP, paste0(prot, "_R.csv")))) rc <- system2(')
        head = (f"# v7 {dst[:4]}: sample-size-corrected copy of {src}, written by B10h_make_ncorrected_susie_v7.py "
                f"(original unchanged).\n# Changes: BRC GWAS N 228,951 (122,977 cases); pQTL N per protein from the UKB-PPP "
                f"files; output to results/B10_genetics_ncorrected/.\n")
        (HERE / dst).write_text(head + s, encoding="utf-8")
        print("wrote", dst)


if __name__ == "__main__":
    main()
