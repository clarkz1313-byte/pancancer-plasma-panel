# v7.0 FORK of v5_run_tier1_combined.R (original unchanged; written by v7_plan/scripts/B10_genetics_setup_v7.py). Combines the 257 multiclass pairs with the v7 Part A pairs; output tier1_combined_v7.csv.
# v5.0 FORK of run_tier1_combined.R (original unchanged): disease-specific pairs,
# leads and cascade from the v5 genetics folder; 257 25-protein pairs unchanged.
# run_tier1_combined.R -- Tier 1 of the citation-clean genetics pipeline
# (see figurev5/GENETICS_PIPELINE_DIAGNOSIS_AND_PLAN_2026-09-03.md section 2).
#
# What changes vs. the existing result files, and what doesn't:
#   - coloc PP.H0-H4: REUSED as-is from phase4_coloc_local_results_11cancers.csv
#     (257) and parta_coloc_results.csv (59). Already coloc::coloc.abf(),
#     already citable (Giambartolomei 2014). Not recomputed.
#   - Wald ratio: RECOMPUTED via TwoSampleMR::mr_wald_ratio() (Hemani 2018),
#     using its default (simpler, single-term/NOME) SE convention -- the
#     user's explicit choice over our own two-term delta method, for
#     citability. Inputs (b_pqtl_lead, se_pqtl_lead, b_gwas_lead,
#     se_gwas_lead) are taken directly from the already-saved per-pair lead
#     SNP effect sizes (257: phase5_smr_results.csv, already has all four
#     columns; 59: re-extracted from the saved region checkpoints, since
#     parta_wald_heidi_results.csv only stored b_smr/se_smr, not the raw
#     b_gwas_lead/se_gwas_lead -- no new pQTL/GWAS files touched, same
#     checkpoints run_parta_wald_heidi.R already used).
#   - HEIDI-style Q at rho=0: REUSED as-is (same formula, same numbers) from
#     both existing result files. Only the citation label changes (Zhu et
#     al. 2016, "HEIDI statistic evaluated at rho=0", not Cochran 1954) --
#     that's a text change, not a recompute, so nothing here touches it
#     numerically. Still explicitly NOT a decision gate.
#
# Output: revise_plan/smr_coloc/results/tier1_combined_316.csv, one row per
# pair, both panel systems, ready to be the single reference table.

.libPaths(c(file.path(Sys.getenv("PROTEOMICS_ROOT", "."), ".rlib"), .libPaths()))
suppressPackageStartupMessages({
  library(dplyr); library(readr); library(tibble); library(TwoSampleMR)
})

RESULT_DIR <- file.path(Sys.getenv("PROTEOMICS_ROOT", "."), "revise_plan/smr_coloc/results")

# ── 1. The 257-pair screen (25-protein panel x 11 cancers) ─────────────────
phase5 <- read_csv(file.path(RESULT_DIR, "phase5_smr_results.csv"), show_col_types = FALSE)
coloc257 <- read_csv(file.path(RESULT_DIR, "phase4_coloc_local_results_11cancers.csv"), show_col_types = FALSE)

wald257 <- phase5 %>%
  rowwise() %>%
  mutate(
    .res = list(mr_wald_ratio(b_exp = b_pqtl_lead, b_out = b_gwas_lead,
                               se_exp = se_pqtl_lead, se_out = se_gwas_lead))
  ) %>%
  ungroup() %>%
  mutate(
    b_wald_pkg  = sapply(.res, function(x) x$b),
    se_wald_pkg = sapply(.res, function(x) x$se),
    p_wald_pkg  = sapply(.res, function(x) x$pval)
  ) %>%
  select(protein, cancer, lead_chr, lead_pos, n_merged_snps, n_heidi_snps,
         b_pqtl_lead, se_pqtl_lead, b_gwas_lead, se_gwas_lead,
         b_wald_pkg, se_wald_pkg, p_wald_pkg,
         p_heidi_rho0 = p_HEIDI)

combo257 <- coloc257 %>%
  select(protein, cancer, n_snps, lead_chr, lead_pos,
         PP.H0, PP.H1, PP.H2, PP.H3, PP.H4, passes_h4_0.8 = colocalises) %>%
  left_join(wald257 %>% select(-lead_chr, -lead_pos), by = c("protein","cancer")) %>%
  mutate(panel_source = "locked25_x_11cancers")

cat(sprintf("Tier 1, 25-protein panel screen: %d rows, %d with Wald recomputed\n",
            nrow(combo257), sum(!is.na(combo257$p_wald_pkg))))

# ── 2. The 59-pair screen (12 single-cancer panels, Part A) ────────────────
# b_gwas_lead/se_gwas_lead were computed but not saved by
# run_parta_wald_heidi.R -- re-extract from the same checkpoints it used,
# no new pQTL/GWAS files touched.
GWAS_DIR   <- file.path(Sys.getenv("SMR_DATA_ROOT", "smr_data"), "cancer_gwas")
V5G <- file.path(Sys.getenv("PROTEOMICS_ROOT", "."), "v7_plan/results/B10_genetics")
LEADS_RDS  <- file.path(V5G, "parta_pqtl_leads_checkpoint.rds")
coloc59    <- read_csv(file.path(V5G, "parta_coloc_results.csv"), show_col_types = FALSE)
leads      <- readRDS(LEADS_RDS)

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
    out <- raw %>% transmute(chrom = as.character(hm_chrom), pos = as.integer(hm_pos),
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
      transmute(chrom = as.character(chromosome), pos = as.integer(base_pair_location),
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

results59 <- list()
for (cancer in unique(coloc59$cancer)) {
  cfg <- CANCER_CONFIG[[cancer]]
  if (is.null(cfg) || !file.exists(cfg$file)) next
  cat(sprintf("=== Part A: %s ===\n", cancer))
  gwas <- tryCatch(load_gwas_cache(cfg), error = function(e) NULL)
  if (is.null(gwas)) next

  for (prot in coloc59$protein[coloc59$cancer == cancer]) {
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
    if (nrow(merged) < 5) next

    lead_snp <- merged %>% slice_max(abs(beta.pqtl/se.pqtl), n=1, with_ties=FALSE)
    if (abs(lead_snp$beta.pqtl) < 1e-10) next

    wr <- mr_wald_ratio(b_exp = lead_snp$beta.pqtl, b_out = lead_snp$beta.gwas,
                         se_exp = lead_snp$se.pqtl, se_out = lead_snp$se.gwas)

    results59[[paste0(prot,"_",cancer)]] <- tibble(
      protein = prot, cancer = cancer,
      lead_chr = lead$chr, lead_pos = lead_snp$pos,
      n_merged_snps = nrow(merged),
      b_pqtl_lead = lead_snp$beta.pqtl, se_pqtl_lead = lead_snp$se.pqtl,
      b_gwas_lead = lead_snp$beta.gwas, se_gwas_lead = lead_snp$se.gwas,
      b_wald_pkg = wr$b, se_wald_pkg = wr$se, p_wald_pkg = wr$pval
    )
  }
  rm(gwas); gc()
}
wald59 <- bind_rows(results59)
cat(sprintf("Tier 1, Part A screen: %d/%d pairs re-extracted\n", nrow(wald59), nrow(coloc59)))

# bring in reused (not recomputed) heidi p and coloc PP.H4 from the existing files
parta_full <- read_csv(file.path(V5G, "parta_full_cascade.csv"), show_col_types = FALSE)

combo59 <- coloc59 %>%
  select(protein, cancer, n_snps, PP.H0, PP.H1, PP.H2, PP.H3, PP.H4, passes_h4_0.8) %>%
  left_join(wald59, by = c("protein","cancer")) %>%
  left_join(parta_full %>% select(protein, cancer, p_heidi_rho0 = p_heidi, n_heidi_snps),
            by = c("protein","cancer")) %>%
  left_join(read_csv(file.path(V5G, "v7_parta_pairs.csv"), show_col_types = FALSE), by = c("protein","cancer"))

# ── 3. Combine, apply the ONLY formal decision rule (coloc + Wald, Bonferroni @ n=316) ──
common_cols <- c("protein","cancer","panel_source","lead_chr","lead_pos",
                  "n_snps","n_merged_snps","n_heidi_snps",
                  "PP.H0","PP.H1","PP.H2","PP.H3","PP.H4","passes_h4_0.8",
                  "b_pqtl_lead","se_pqtl_lead","b_gwas_lead","se_gwas_lead",
                  "b_wald_pkg","se_wald_pkg","p_wald_pkg","p_heidi_rho0")

all316 <- bind_rows(
  combo257 %>% mutate(n_snps = n_merged_snps, lead_chr = as.character(lead_chr)) %>%
    select(any_of(common_cols)),
  combo59  %>% mutate(lead_chr = as.character(lead_chr)) %>%
    select(any_of(common_cols))
)

n_total <- nrow(all316)
bonf <- 0.05 / n_total
all316 <- all316 %>%
  mutate(
    bonferroni_threshold = bonf,
    passes_wald_bonferroni = !is.na(p_wald_pkg) & p_wald_pkg < bonf,
    heidi_descriptive_pass_rho0 = !is.na(p_heidi_rho0) & p_heidi_rho0 > 0.05,
    final_hit = passes_wald_bonferroni & passes_h4_0.8   # HEIDI NOT a gate, matches §2.8
  )

OUT <- file.path(V5G, "tier1_combined_v7.csv")
write_csv(all316, OUT)
cat(sprintf("\nsaved -> %s (%d rows, %d pairs)\n", OUT, nrow(all316), n_total))
cat(sprintf("Bonferroni threshold at n=%d: %.3e\n", n_total, bonf))
cat("\nFinal hits (Wald-Bonferroni AND PP.H4>0.8; HEIDI reported, not gated):\n")
print(all316 %>% filter(final_hit) %>%
        select(protein, cancer, panel_source, PP.H4, p_wald_pkg, p_heidi_rho0, heidi_descriptive_pass_rho0))
cat("\nTop 10 by PP.H4:\n")
print(all316 %>% arrange(desc(PP.H4)) %>%
        select(protein, cancer, panel_source, PP.H4, p_wald_pkg, p_heidi_rho0) %>% head(10))
cat("TIER1_COMPLETE\n")
