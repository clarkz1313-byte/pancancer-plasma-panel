# v7.0b FORK of v7_run_parta_wald_heidi.R (original unchanged; written by v7_plan/scripts/B10b_genetics_chrx_fix_v7.py). GWAS chromosome X recoded to 23 (UKB-PPP label); results in v7_plan/results/B10_genetics_v2.
# v7.0 FORK of v5_run_parta_wald_heidi.R (original unchanged; written by v7_plan/scripts/B10_genetics_setup_v7.py). Reads/writes the v7 genetics folder; joint Bonferroni = 257 + v7 Part A pairs.
# v5.0 FORK (original unchanged). Reads/writes the v5 genetics folder; joint Bonferroni = 257 + v5 pairs.
# run_parta_wald_heidi.R -- the two legs missing from run_parta_coloc.R
#
# WHY THIS EXISTS
# ----------------
# run_parta_coloc.R (2026-09-02/03) claimed to use "the exact same method"
# as the original 257-pair screen, but only replicated colocalization. The
# original screen's decision cascade for each pair is THREE tests, in this
# order (run_phase5_smr.R):
#   1. SMR (Wald ratio) at the lead pQTL SNP -- p_SMR < 0.05
#   2. HEIDI (Cochran's Q across the top 20 instruments) -- p_HEIDI > 0.05
#      (i.e. NOT rejecting the single-causal-variant model)
#   3. coloc.abf -- PP.H4 > 0.8, with the Wald p ALSO required to clear a
#      Bonferroni threshold before a pair is called a hit (§12.3 of the
#      manuscript: 0.05/257 for the original screen)
#
# This was caught 2026-09-03 when asked directly "did you check the SMR and
# HEIDI test?" -- the answer was no, and the earlier claim of an identical
# method was wrong for two of its three legs. This script fills the gap.
#
# WHAT IT REUSES, WHAT IT DOESN'T RE-DO
# --------------------------------------
# Reuses run_parta_coloc.R's saved checkpoint: `parta_pqtl_leads_checkpoint.rds`
# (each protein's genome-wide lead position + its saved +/-500kb pQTL region
# in `parta_pqtl_regions/`). Nothing is re-downloaded and no pQTL file is
# re-scanned. Each cancer's GWAS is reloaded (same parsers, same files, same
# N / n_cases as run_parta_coloc.R and run_phase4_11cancers_coloc.R) because
# it was not cached to disk after the coloc run.
#
# The Wald/HEIDI functions (smr_wald, heidi_Q) and the lead-SNP-by-|z|,
# top-20-by-|z| selection are copied verbatim from run_phase5_smr.R, not
# reimplemented, to avoid a second source of transcription error.
#
# Output: revise_plan/smr_coloc/results/parta_wald_heidi_results.csv
#         revise_plan/smr_coloc/results/parta_full_cascade.csv (merged with
#         the coloc results, one row per pair, all three tests + the
#         updated Bonferroni call at n=316 total pairs)
# Run:    Rscript run_parta_wald_heidi.R

.libPaths(c(file.path(Sys.getenv("PROTEOMICS_ROOT", "."), ".rlib"), .libPaths()))
suppressPackageStartupMessages({
  library(dplyr); library(readr); library(tibble)
})

GWAS_DIR   <- file.path(Sys.getenv("SMR_DATA_ROOT", "smr_data"), "cancer_gwas")
RESULT_DIR <- file.path(Sys.getenv("PROTEOMICS_ROOT", "."), "v7_plan/results/B10_genetics_v2")
REGION_DIR <- file.path(RESULT_DIR, "parta_pqtl_regions")
LEADS_RDS  <- file.path(RESULT_DIR, "parta_pqtl_leads_checkpoint.rds")
COLOC_CSV  <- file.path(RESULT_DIR, "parta_coloc_results.csv")
OUT_CSV    <- file.path(RESULT_DIR, "parta_wald_heidi_results.csv")
FULL_CSV   <- file.path(RESULT_DIR, "parta_full_cascade.csv")

stopifnot(file.exists(LEADS_RDS), file.exists(COLOC_CSV))
leads <- readRDS(LEADS_RDS)
coloc <- read_csv(COLOC_CSV, show_col_types = FALSE)
cat(sprintf("Loaded %d protein leads, %d coloc pairs\n", length(leads), nrow(coloc)))

PAIRS <- coloc %>% select(protein, cancer)

# ── GWAS config -- identical to run_parta_coloc.R ──────────────────────────
CANCER_CONFIG <- list(
  BRC = list(file = file.path(GWAS_DIR, "29059683-GCST004988-EFO_0000305.h.tsv.gz"), parser = "hm"),
  CRC = list(file = file.path(GWAS_DIR, "GCST90255675.h.tsv.gz"), parser = "raw"),
  LUNGC = list(file = file.path(GWAS_DIR, "28604730-GCST004748-EFO_0001071.h.tsv.gz"), parser = "hm"),
  PRC = list(file = file.path(GWAS_DIR, "29892016-GCST006085-EFO_0001663.h.tsv.gz"), parser = "hm"),
  OVC = list(file = file.path(GWAS_DIR, "OVC_GCST90455661.h.tsv.gz"), parser = "raw"),
  ENDC = list(file = file.path(GWAS_DIR, "ENDC_GCST90454186.h.tsv.gz"), parser = "raw"),
  MYEL = list(file = file.path(GWAS_DIR, "MYEL_GCST90624747.h.tsv.gz"), parser = "raw"),
  LYMPH = list(file = file.path(GWAS_DIR, "LYMPH_GCST90624739.h.tsv.gz"), parser = "raw"),
  CLL = list(file = file.path(GWAS_DIR, "CLL_GCST90624738.h.tsv.gz"), parser = "raw"),
  AML = list(file = file.path(GWAS_DIR, "AML_GCST90707271.h.tsv.gz"), parser = "raw"),
  GLIOM = list(file = file.path(GWAS_DIR, "GLIOM_finngen_R12_C3_BRAIN_EXALLC.gz"), parser = "finngen")
)

load_gwas_cache <- function(cfg) {
  if (cfg$parser == "hm") {
    raw <- read_tsv(cfg$file, col_select = any_of(c("hm_rsid","hm_chrom","hm_pos",
                    "hm_effect_allele","hm_effect_allele_frequency","hm_beta",
                    "standard_error","hm_ci_lower","hm_ci_upper","p_value")),
                    col_types = cols(.default = "c"), show_col_types = FALSE)
    se_col <- if ("standard_error" %in% names(raw)) suppressWarnings(as.numeric(raw$standard_error))
              else (as.numeric(raw$hm_ci_upper) - as.numeric(raw$hm_ci_lower)) / 3.92
    out <- raw %>% transmute(chrom = sub("^(chr)?X$", "23", as.character(hm_chrom)), pos = as.integer(hm_pos),
                             hm_effect_allele = toupper(hm_effect_allele),
                             beta = as.numeric(hm_beta), se = se_col) %>%
      filter(!is.na(beta), !is.na(se), se > 0)
    rm(raw); gc(); return(out)
  }
  if (cfg$parser == "raw") {
    raw <- read_tsv(cfg$file, col_select = any_of(c("chromosome","base_pair_location",
                    "effect_allele","beta","standard_error","p_value","hm_code")),
                    col_types = cols(.default = "c"), show_col_types = FALSE)
    out <- raw %>% filter(is.na(hm_code) | hm_code %in% c("5","10","11")) %>%
      transmute(chrom = sub("^(chr)?X$", "23", as.character(chromosome)), pos = as.integer(base_pair_location),
               hm_effect_allele = toupper(effect_allele),
               beta = as.numeric(beta), se = as.numeric(standard_error)) %>%
      filter(!is.na(chrom), !is.na(pos), !is.na(beta), !is.na(se), se > 0, is.finite(beta))
    rm(raw); gc(); return(out)
  }
  if (cfg$parser == "finngen") {
    raw <- read_tsv(cfg$file, col_select = any_of(c("#chrom","pos","alt","beta","sebeta","pval")),
                    col_types = cols(.default = "c"), show_col_types = FALSE)
    out <- raw %>% transmute(chrom = gsub("^chr","",as.character(`#chrom`),ignore.case=TRUE),
                             pos = as.integer(pos), hm_effect_allele = toupper(alt),
                             beta = as.numeric(beta), se = as.numeric(sebeta)) %>%
      filter(!is.na(chrom), !is.na(pos), !is.na(beta), !is.na(se), se > 0, is.finite(beta))
    rm(raw); gc(); return(out)
  }
}

# ── Wald + HEIDI, copied verbatim from run_phase5_smr.R ────────────────────
smr_wald <- function(b_pqtl, se_pqtl, b_gwas, se_gwas) {
  if (abs(b_pqtl) < 1e-10) return(list(b=NA, se=NA, z=NA, p=NA))
  b_smr  <- b_gwas / b_pqtl
  var_smr <- b_smr^2 * ((se_gwas/b_gwas)^2 + (se_pqtl/b_pqtl)^2)
  se_smr  <- sqrt(pmax(var_smr, 0))
  z_smr   <- b_smr / se_smr
  p_smr   <- 2 * pnorm(-abs(z_smr))
  list(b=b_smr, se=se_smr, z=z_smr, p=p_smr)
}

heidi_Q <- function(b_pqtl_vec, se_pqtl_vec, b_gwas_vec, se_gwas_vec, b_smr_lead) {
  K <- length(b_pqtl_vec)
  if (K < 2) return(NA_real_)
  b_i   <- b_gwas_vec / b_pqtl_vec
  var_i <- b_i^2 * ((se_gwas_vec/b_gwas_vec)^2 + (se_pqtl_vec/b_pqtl_vec)^2)
  keep  <- is.finite(b_i) & is.finite(var_i) & var_i > 0
  if (sum(keep) < 2) return(NA_real_)
  b_i   <- b_i[keep]; var_i <- var_i[keep]
  Q  <- sum((b_i - b_smr_lead)^2 / var_i)
  df <- sum(keep) - 1
  pchisq(Q, df=df, lower.tail=FALSE)
}

results <- list()
for (cancer in unique(PAIRS$cancer)) {
  cfg <- CANCER_CONFIG[[cancer]]
  if (is.null(cfg) || !file.exists(cfg$file)) { cat(sprintf("skip %s: no GWAS\n", cancer)); next }
  cat(sprintf("\n=== %s ===\n", cancer))
  gwas <- tryCatch(load_gwas_cache(cfg), error = function(e) { cat("ERROR:", conditionMessage(e), "\n"); NULL })
  if (is.null(gwas)) next
  cat(sprintf("  GWAS cached: %s SNPs\n", format(nrow(gwas), big.mark=",")))

  for (prot in PAIRS$protein[PAIRS$cancer == cancer]) {
    lead <- leads[[prot]]
    if (is.null(lead)) next
    reg <- tryCatch(readRDS(lead$region_file), error = function(e) NULL)
    if (is.null(reg)) next

    gwas_reg <- gwas %>% filter(chrom == as.character(lead$chr),
                                pos >= lead$pos - lead$window, pos <= lead$pos + lead$window)
    if (nrow(gwas_reg) < 5) next

    merged <- inner_join(
      reg %>% select(chrom, pos, a1, a0, beta, se) %>%
        rename(beta.pqtl=beta, se.pqtl=se, effect_allele.pqtl=a1, other_allele.pqtl=a0),
      gwas_reg %>% select(chrom, pos, hm_effect_allele, beta, se) %>%
        rename(beta.gwas=beta, se.gwas=se, effect_allele.gwas=hm_effect_allele),
      by = c("chrom","pos"), relationship = "many-to-many"
    ) %>%
      mutate(needs_flip = toupper(effect_allele.pqtl) != toupper(effect_allele.gwas),
             beta.gwas = ifelse(needs_flip, -beta.gwas, beta.gwas)) %>%
      filter(!is.na(beta.pqtl), !is.na(beta.gwas), se.pqtl > 0, se.gwas > 0) %>%
      distinct(chrom, pos, .keep_all = TRUE)

    if (nrow(merged) < 5) { cat(sprintf("  [%s x %s] too few SNPs (%d)\n", prot, cancer, nrow(merged))); next }

    lead_snp <- merged %>% slice_max(abs(beta.pqtl/se.pqtl), n=1, with_ties=FALSE)
    smr <- smr_wald(lead_snp$beta.pqtl, lead_snp$se.pqtl, lead_snp$beta.gwas, lead_snp$se.gwas)
    if (is.na(smr$p)) { cat(sprintf("  [%s x %s] b_pQTL ~0, skip\n", prot, cancer)); next }

    K <- min(20L, nrow(merged))
    top_k <- merged %>% slice_max(abs(beta.pqtl/se.pqtl), n=K, with_ties=FALSE) %>%
      filter(abs(beta.pqtl) > 1e-10, abs(beta.gwas) > 1e-10)
    p_heidi <- heidi_Q(top_k$beta.pqtl, top_k$se.pqtl, top_k$beta.gwas, top_k$se.gwas, smr$b)

    smr_passes   <- !is.na(smr$p) && smr$p < 0.05
    heidi_passes <- is.na(p_heidi) || p_heidi > 0.05

    cat(sprintf("  [%s x %s] n=%d b_SMR=%.4f p_SMR=%.2e p_HEIDI=%s\n",
                prot, cancer, nrow(merged), smr$b, smr$p,
                ifelse(is.na(p_heidi), "NA", sprintf("%.3g", p_heidi))))

    results[[paste0(prot,"_",cancer)]] <- tibble(
      protein = prot, cancer = cancer, n_merged_snps = nrow(merged), n_heidi_snps = nrow(top_k),
      b_pqtl_lead = lead_snp$beta.pqtl, se_pqtl_lead = lead_snp$se.pqtl,
      b_smr = smr$b, se_smr = smr$se, p_smr = smr$p, p_heidi = p_heidi,
      smr_nominal_pass = smr_passes, heidi_pass = heidi_passes
    )
  }
  rm(gwas); gc()
}

wh <- bind_rows(results)
write_csv(wh, OUT_CSV)
cat(sprintf("\nsaved -> %s (%d rows)\n", OUT_CSV, nrow(wh)))

# ── merge with coloc results into one cascade table ─────────────────────────
full <- coloc %>% left_join(wh, by = c("protein","cancer"))
n_total_pairs <- 257 + nrow(coloc)   # original screen + this one
bonf <- 0.05 / n_total_pairs
full <- full %>% mutate(
  bonferroni_threshold_joint = bonf,
  passes_bonferroni_joint = !is.na(p_smr) & p_smr < bonf,
  final_hit = passes_bonferroni_joint & heidi_pass & passes_h4_0.8
)
write_csv(full, FULL_CSV)
cat(sprintf("saved -> %s\n", FULL_CSV))
cat(sprintf("\nBonferroni threshold at n=%d total pairs: %.3e\n", n_total_pairs, bonf))
cat("\nFinal hits (Bonferroni-significant Wald AND HEIDI-pass AND PP.H4>0.8):\n")
print(full %>% filter(final_hit) %>% select(protein, cancer, p_smr, p_heidi, PP.H4))
cat("\nAll rows sorted by PP.H4, with Wald/HEIDI columns:\n")
print(full %>% arrange(desc(PP.H4)) %>%
        select(protein, cancer, PP.H4, p_smr, p_heidi, smr_nominal_pass, heidi_pass, passes_bonferroni_joint, final_hit) %>%
        head(15))
cat("PARTA_WALD_HEIDI_COMPLETE\n")
