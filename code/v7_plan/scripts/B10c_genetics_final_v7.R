# v7 B10c: canonical final genetics table (audit SOL-F03, F04, F05, F21).
#
# Input: v7_plan/results/B10_genetics_v2/tier1_combined_v7.csv (332 pairs; coloc PP and hit rule unchanged).
# For every pair:
#   - genome-wide pQTL lead (checkpoint) and the actual Wald instrument (the region SNP whose pQTL beta/se equal
#     the stored b_pqtl_lead/se_pqtl_lead: the strongest pQTL among SNPs shared with that GWAS) are reported
#     separately (SOL-F04/F05);
#   - the GWAS row at the instrument position is harmonised on both alleles (SOL-F03): direct, swapped (flip),
#     strand complement (complement first), palindromic A/T or C/G resolved by allele frequency when both
#     frequencies are available and the minor allele frequency is below 0.42, otherwise direction ambiguous
#     (Wald estimate reported without sign; p value unchanged, it is sign-invariant); incompatible alleles excluded;
#   - Wald ratio recomputed with TwoSampleMR::mr_wald_ratio on the harmonised effects;
#   - cis/trans: instrument within 1 Mb of the protein's gene (GENCODE v36 gene coordinates) = cis (SOL-F21).
# Hit rule (unchanged): PP.H4 > 0.8 and Wald p < 0.05 / number of analysable pairs.
# Output: v7_plan/results/B10_genetics_final/tier1_combined_v7_final.csv
.libPaths(c(file.path(Sys.getenv("PROTEOMICS_ROOT", "."), ".rlib"), .libPaths()))
suppressPackageStartupMessages({ library(dplyr); library(readr); library(TwoSampleMR) })

ROOT <- Sys.getenv("PROTEOMICS_ROOT", ".")
OUTD <- file.path(ROOT, "v7_plan/results/B10_genetics_final")
G <- file.path(Sys.getenv("SMR_DATA_ROOT", "smr_data"), "cancer_gwas")
t1 <- read_csv(file.path(ROOT, "v7_plan/results/B10_genetics_v2/tier1_combined_v7.csv"), show_col_types = FALSE)
ll0 <- readRDS(file.path(ROOT, "revise_plan/smr_coloc/results/pqtl_leads_checkpoint.rds"))
ll1 <- readRDS(file.path(ROOT, "v7_plan/results/B10_genetics_v2/parta_pqtl_leads_checkpoint.rds"))
pm <- read_tsv(file.path(OUTD, "ref/gencode.v36.probemap"), show_col_types = FALSE) %>%
  transmute(gene, gchr = sub("^chr", "", chrom), gstart = chromStart, gend = chromEnd) %>%
  mutate(gchr = ifelse(gchr == "X", "23", gchr)) %>% distinct(gene, .keep_all = TRUE)

CFG <- list(
  BRC = list(f = "29059683-GCST004988-EFO_0000305.h.tsv.gz", p = "hm"), LUNGC = list(f = "28604730-GCST004748-EFO_0001071.h.tsv.gz", p = "hm"),
  PRC = list(f = "29892016-GCST006085-EFO_0001663.h.tsv.gz", p = "hm"), CRC = list(f = "GCST90255675.h.tsv.gz", p = "raw"),
  OVC = list(f = "OVC_GCST90455661.h.tsv.gz", p = "raw"), ENDC = list(f = "ENDC_GCST90454186.h.tsv.gz", p = "raw"),
  MYEL = list(f = "MYEL_GCST90624747.h.tsv.gz", p = "raw"), LYMPH = list(f = "LYMPH_GCST90624739.h.tsv.gz", p = "raw"),
  CLL = list(f = "CLL_GCST90624738.h.tsv.gz", p = "raw"), AML = list(f = "AML_GCST90707271.h.tsv.gz", p = "raw"),
  GLIOM = list(f = "GLIOM_finngen_R12_C3_BRAIN_EXALLC.gz", p = "finngen"))

load_gwas <- function(cfg) {
  f <- file.path(G, cfg$f)
  if (cfg$p == "hm") {
    read_tsv(f, col_select = c("hm_rsid", "hm_chrom", "hm_pos", "hm_effect_allele", "hm_other_allele",
                               "hm_effect_allele_frequency", "hm_beta", "standard_error", "hm_ci_lower", "hm_ci_upper"),
             col_types = cols(.default = "c"), show_col_types = FALSE) %>%
      transmute(chrom = sub("^(chr)?X$", "23", hm_chrom), pos = as.integer(hm_pos), g_rsid = hm_rsid,
                g_ea = toupper(hm_effect_allele), g_oa = toupper(hm_other_allele),
                g_eaf = suppressWarnings(as.numeric(hm_effect_allele_frequency)), g_beta = as.numeric(hm_beta),
                g_se = ifelse(is.na(suppressWarnings(as.numeric(standard_error))),
                              (as.numeric(hm_ci_upper) - as.numeric(hm_ci_lower)) / 3.92, suppressWarnings(as.numeric(standard_error))))
  } else if (cfg$p == "raw") {
    read_tsv(f, col_select = any_of(c("chromosome", "base_pair_location", "effect_allele", "other_allele",
                                      "effect_allele_frequency", "beta", "standard_error", "rsid", "hm_code")),
             col_types = cols(.default = "c"), show_col_types = FALSE) %>%
      filter(is.na(hm_code) | hm_code %in% c("5", "10", "11")) %>%
      transmute(chrom = sub("^(chr)?X$", "23", chromosome), pos = as.integer(base_pair_location), g_rsid = rsid,
                g_ea = toupper(effect_allele), g_oa = toupper(other_allele),
                g_eaf = suppressWarnings(as.numeric(effect_allele_frequency)), g_beta = as.numeric(beta), g_se = as.numeric(standard_error))
  } else {
    read_tsv(f, col_select = c("#chrom", "pos", "ref", "alt", "rsids", "af_alt", "beta", "sebeta"),
             col_types = cols(.default = "c"), show_col_types = FALSE) %>%
      transmute(chrom = gsub("^chr", "", `#chrom`, ignore.case = TRUE), pos = as.integer(pos), g_rsid = rsids,
                g_ea = toupper(alt), g_oa = toupper(ref), g_eaf = as.numeric(af_alt), g_beta = as.numeric(beta), g_se = as.numeric(sebeta))
  }
}

comp <- function(a) chartr("ACGT", "TGCA", a)
classify <- function(pe, po, ge, go, peaf, geaf) {
  pal <- (pe == comp(po))
  if (pal && ge %in% c(pe, po) && go %in% c(pe, po)) {
    ok <- !is.na(peaf) && !is.na(geaf) && min(peaf, 1 - peaf) < 0.42 && min(geaf, 1 - geaf) < 0.42
    if (!ok) return(list(cls = "palindromic_ambiguous", flip = NA))
    same <- abs(peaf - geaf) < abs(peaf - (1 - geaf))
    # whichever way the labels read, the GWAS effect allele is the pQTL effect allele when the frequencies agree
    flip <- !same
    return(list(cls = "palindromic_resolved", flip = flip))
  }
  if (ge == pe && go == po) return(list(cls = "direct", flip = FALSE))
  if (ge == po && go == pe) return(list(cls = "swapped", flip = TRUE))
  if (ge == comp(pe) && go == comp(po)) return(list(cls = "strand_complement", flip = FALSE))
  if (ge == comp(po) && go == comp(pe)) return(list(cls = "strand_complement_swapped", flip = TRUE))
  list(cls = "incompatible", flip = NA)
}

# 1. instruments
regions <- list(); inst <- list()
for (i in seq_len(nrow(t1))) {
  r <- t1[i, ]; p <- r$protein
  l <- if (!is.null(ll0[[p]])) ll0[[p]] else ll1[[p]]
  rf <- l$region_file; if (!grepl("^[A-Za-z]:|^/", rf)) rf <- file.path(ROOT, rf)  # old checkpoint stores relative paths
  if (is.null(regions[[p]])) regions[[p]] <- readRDS(rf)
  d <- regions[[p]]
  q <- d[abs(d$beta - r$b_pqtl_lead) < 1e-10 & abs(d$se - r$se_pqtl_lead) < 1e-10, , drop = FALSE]
  q <- q[1, , drop = FALSE]  # tied identical-effect rows (BMP4) are compatible duplicates
  inst[[i]] <- tibble(protein = p, cancer = r$cancer, gw_lead_chr = as.character(l$chr), gw_lead_pos = as.integer(l$pos),
                      inst_chr = as.character(q$chrom), inst_pos = as.integer(q$pos), inst_id = if ("rsid" %in% names(q)) q$rsid else NA_character_,
                      p_ea = toupper(q$a1), p_oa = toupper(q$a0), p_eaf = q$eaf, n_inst_match = nrow(q))
}
inst <- bind_rows(inst)
stopifnot(all(!is.na(inst$inst_pos)))

# 2. GWAS rows at the instrument positions, harmonised
out <- list()
for (cn in unique(inst$cancer)) {
  cat("GWAS", cn, "\n")
  gw <- load_gwas(CFG[[cn]])
  sub <- inst %>% filter(cancer == cn)
  gw <- gw %>% filter(paste(chrom, pos) %in% paste(sub$inst_chr, sub$inst_pos))
  for (j in seq_len(nrow(sub))) {
    s <- sub[j, ]
    cand <- gw %>% filter(chrom == s$inst_chr, pos == s$inst_pos)
    best <- NULL
    for (k in seq_len(nrow(cand))) {
      cl <- classify(s$p_ea, s$p_oa, cand$g_ea[k], cand$g_oa[k], s$p_eaf, cand$g_eaf[k])
      if (cl$cls != "incompatible") { best <- c(as.list(cand[k, ]), cl); break }
    }
    if (is.null(best)) best <- list(g_rsid = NA, g_ea = paste(cand$g_ea, collapse = "/"), g_oa = paste(cand$g_oa, collapse = "/"),
                                    g_eaf = NA, g_beta = NA, g_se = NA, cls = if (nrow(cand)) "incompatible" else "absent", flip = NA)
    out[[length(out) + 1]] <- bind_cols(s, tibble(gwas_rsid = best$g_rsid, g_ea = best$g_ea, g_oa = best$g_oa, g_eaf = best$g_eaf,
                                                  g_beta_raw = best$g_beta, g_se = best$g_se, harmonisation = best$cls, flip = best$flip))
  }
  rm(gw); gc()
}
h <- bind_rows(out)

# 3. Wald on harmonised effects; join coloc and hit rule
h <- h %>% left_join(t1 %>% select(protein, cancer, panel_source, PP.H3, PP.H4, passes_h4_0.8, b_pqtl_lead, se_pqtl_lead,
                                   b_gwas_lead, p_wald_pkg, p_heidi_rho0), by = c("protein", "cancer")) %>%
  mutate(abs_beta_check = abs(abs(g_beta_raw) - abs(b_gwas_lead)) < 1e-8,
         b_gwas_harm = case_when(is.na(flip) ~ NA_real_, flip ~ -g_beta_raw, TRUE ~ g_beta_raw),
         old_flip_sign_ok = sign(b_gwas_harm) == sign(b_gwas_lead))
w <- mapply(function(be, bo, se, so) if (is.na(bo)) c(NA, NA, NA) else unlist(mr_wald_ratio(be, bo, se, so)[c("b", "se", "pval")]),
            h$b_pqtl_lead, h$b_gwas_harm, h$se_pqtl_lead, h$g_se)
h$b_wald_harm <- w[1, ]; h$se_wald_harm <- w[2, ]; h$p_wald_harm <- w[3, ]
h <- h %>% left_join(pm, by = c("protein" = "gene")) %>%
  mutate(cis_trans = ifelse(!is.na(gchr) & inst_chr == gchr & inst_pos >= gstart - 1e6 & inst_pos <= gend + 1e6, "cis", "trans"),
         n_pairs = n(), bonferroni = 0.05 / n_pairs,
         final_hit = passes_h4_0.8 & p_wald_pkg < bonferroni)
write_csv(h, file.path(OUTD, "tier1_combined_v7_final.csv"))
cat("pairs", nrow(h), "| |beta| check failures", sum(!h$abs_beta_check, na.rm = TRUE), "\n")
print(table(h$harmonisation, useNA = "ifany"))
cat("sign changed vs previous table:", sum(!h$old_flip_sign_ok, na.rm = TRUE), "\n")
print(h %>% filter(!old_flip_sign_ok | harmonisation %in% c("palindromic_ambiguous", "incompatible", "absent")) %>%
        select(protein, cancer, harmonisation, b_gwas_lead, b_gwas_harm, p_wald_pkg) %>% as.data.frame())
cat("hits:\n"); print(h %>% filter(final_hit) %>% select(protein, cancer, PP.H4, p_wald_pkg, b_wald_harm, cis_trans, inst_chr, inst_pos, gw_lead_pos) %>% as.data.frame())
cat("cis/trans:", paste(names(table(h$cis_trans)), table(h$cis_trans), collapse = "; "), "\n")
cat("instrument position differs from genome-wide lead:", sum(h$inst_pos != h$gw_lead_pos | h$inst_chr != h$gw_lead_chr), "pairs\n")
cat("B10C_COMPLETE\n")
