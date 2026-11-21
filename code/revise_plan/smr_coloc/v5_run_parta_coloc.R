# v5.0 FORK (original unchanged). Screens the 52 v5 disease-specific pairs; writes to the v5 genetics folder.
# run_parta_coloc.R -- genetic screen for the 51 Part A-only proteins
# (proteins in the 12 single-cancer confirmation panels that are NOT in the
# locked-25 and were therefore never colocalized against anything).
#
# WHY
# ---
# The locked-25 was fully screened: 257 protein-cancer pairs, one hit
# (BMP4 x CRC). Part A's 12 panels were never screened at all -- an obvious
# asymmetry a reviewer will ask about. This closes it: each of the 51 new
# proteins is tested against its OWN cancer only (the panel it was
# selected for), 59 pairs in total, using the exact same method as the
# original screen.
#
# METHOD -- identical to run_phase4_only.R / run_phase4_11cancers_coloc.R
# ------------------------------------------------------------------------
# Two passes per protein, same as the original 25-protein screen:
#   pass 1: scan the WHOLE pQTL file, find the genome-wide lead SNP
#           (no chromosome restriction -- same principle documented in
#           MANUSCRIPT_NARRATIVE_DRAFT.md Sec 12.3a)
#   pass 2: extract the +/-500kb region around that lead, save to disk
# Then coloc.abf against the protein's own cancer's GWAS, N=54219 (UKB-PPP),
# p12=1e-5. The 11-cancer GWAS configuration (CANCER_CONFIG, its three
# parsers, N and n_cases) is copied verbatim from
# run_phase4_11cancers_coloc.R -- not re-derived, so there is no risk of a
# transcription error changing which pairs are comparable to the original
# 257.
#
# PATH NOTE: local_data/ was moved to E:\smr_loco\ on 2026-08-18 (see
# revise_plan/smr_coloc/local_data/LOCAL_DATA_MOVED.md). This script points
# directly at E:\smr_loco\, unlike the two scripts above which still
# reference the old (now-empty) local_data/ path.
#
# CHECKPOINTING: writes its own checkpoint files, separate from the
# original run's pqtl_leads_checkpoint.rds / phase4_coloc_local_results*.
# Nothing from the original 257-pair run is read, modified, or overwritten.
#
# Output: revise_plan/smr_coloc/results/parta_coloc_results.csv
# Run:    Rscript run_parta_coloc.R

.libPaths(c(file.path(Sys.getenv("PROTEOMICS_ROOT", "."), ".rlib"), .libPaths()))
suppressPackageStartupMessages({
  library(dplyr); library(readr); library(coloc); library(tibble)
})

PQTL_DIR   <- file.path(Sys.getenv("SMR_DATA_ROOT", "smr_data"), "pqtl_ukbpp")
GWAS_DIR   <- file.path(Sys.getenv("SMR_DATA_ROOT", "smr_data"), "cancer_gwas")
RESULT_DIR <- file.path(Sys.getenv("PROTEOMICS_ROOT", "."), "revise_plan/v5_seed52_common_split/genetics")
dir.create(RESULT_DIR, showWarnings = FALSE, recursive = TRUE)
REGION_DIR <- file.path(RESULT_DIR, "parta_pqtl_regions")
LEADS_RDS  <- file.path(RESULT_DIR, "parta_pqtl_leads_checkpoint.rds")
OUT_CSV    <- file.path(RESULT_DIR, "parta_coloc_results.csv")
dir.create(REGION_DIR, showWarnings = FALSE, recursive = TRUE)

PQTL_N <- 54219L
P12    <- 1e-5
WINDOW <- 500000L

PAIRS_TSV <- "figurev5/scratch_parta_synapse_ids.tsv"  # not used for pairs; see below
# protein -> cancer pairs, written by the session that sized this screen
PAIRS <- read.csv(file.path(Sys.getenv("PROTEOMICS_ROOT", "."), "revise_plan/v5_seed52_common_split/genetics/v5_disease_specific_pairs.csv"), stringsAsFactors = FALSE)
cat(sprintf("Pairs to screen: %d (%d unique proteins, %d unique cancers)\n",
            nrow(PAIRS), length(unique(PAIRS$protein)), length(unique(PAIRS$cancer))))

# ── GWAS configuration -- copied verbatim from run_phase4_11cancers_coloc.R,
#    paths repointed to E:\smr_loco\cancer_gwas ─────────────────────────────
CANCER_CONFIG <- list(
  BRC = list(name = "Breast cancer",
             file = file.path(GWAS_DIR, "29059683-GCST004988-EFO_0000305.h.tsv.gz"),
             n = 139274, n_cases = 76192, parser = "hm"),
  CRC = list(name = "Colorectal cancer",
             file = file.path(GWAS_DIR, "GCST90255675.h.tsv.gz"),
             n = 185616, n_cases = 78473, parser = "raw"),
  LUNGC = list(name = "Lung cancer",
               file = file.path(GWAS_DIR, "28604730-GCST004748-EFO_0001071.h.tsv.gz"),
               n = 85716, n_cases = 29266, parser = "hm"),
  PRC = list(name = "Prostate cancer",
             file = file.path(GWAS_DIR, "29892016-GCST006085-EFO_0001663.h.tsv.gz"),
             n = 140254, n_cases = 79148, parser = "hm"),
  OVC = list(name = "Ovarian cancer",
             file = file.path(GWAS_DIR, "OVC_GCST90455661.h.tsv.gz"),
             n = 398238, n_cases = 19883, parser = "raw"),
  ENDC = list(name = "Endometrial cancer",
              file = file.path(GWAS_DIR, "ENDC_GCST90454186.h.tsv.gz"),
              n = 306458, n_cases = 17278, parser = "raw"),
  MYEL = list(name = "Multiple myeloma",
              file = file.path(GWAS_DIR, "MYEL_GCST90624747.h.tsv.gz"),
              n = 1109399, n_cases = 3991, parser = "raw"),
  LYMPH = list(name = "Lymphoma (DLBCL)",
               file = file.path(GWAS_DIR, "LYMPH_GCST90624739.h.tsv.gz"),
               n = 658535, n_cases = 2280, parser = "raw"),
  CLL = list(name = "Chronic lymphocytic leukemia",
             file = file.path(GWAS_DIR, "CLL_GCST90624738.h.tsv.gz"),
             n = 1108777, n_cases = 4442, parser = "raw"),
  AML = list(name = "Acute myeloid leukemia",
             file = file.path(GWAS_DIR, "AML_GCST90707271.h.tsv.gz"),
             n = 17648, n_cases = 4710, parser = "raw"),
  GLIOM = list(name = "Glioma",
               file = file.path(GWAS_DIR, "GLIOM_finngen_R12_C3_BRAIN_EXALLC.gz"),
               n = 380643, n_cases = 1894, parser = "finngen")
)

parse_ukbpp_pqtl <- function(filepath, chrom_filter = NULL, pos_min = NULL, pos_max = NULL) {
  d <- read_delim(filepath, delim = " ",
                  col_select = any_of(c("CHROM","GENPOS","ID","ALLELE0",
                                        "ALLELE1","A1FREQ","BETA","SE","LOG10P")),
                  col_types = cols(.default = "c"), show_col_types = FALSE)
  names(d) <- toupper(names(d))
  d <- d %>% mutate(chrom = as.character(CHROM), pos = as.integer(GENPOS),
                    rsid = ID, a1 = toupper(ALLELE1), a0 = toupper(ALLELE0),
                    beta = as.numeric(BETA), se = as.numeric(SE),
                    eaf = as.numeric(A1FREQ), log10p = as.numeric(LOG10P)) %>%
    filter(!is.na(beta), !is.na(se), se > 0, is.finite(beta))
  if (!is.null(chrom_filter)) {
    d <- d %>% filter(chrom == as.character(chrom_filter),
                      pos >= pos_min, pos <= pos_max)
  }
  d
}

load_gwas_cache <- function(cfg) {
  cat(sprintf("  Loading GWAS: %s\n", basename(cfg$file)))
  if (cfg$parser == "hm") {
    cols_needed <- c("hm_rsid","hm_chrom","hm_pos","hm_effect_allele",
                     "hm_effect_allele_frequency","hm_beta","standard_error",
                     "hm_ci_lower","hm_ci_upper","p_value")
    raw <- read_tsv(cfg$file, col_select = any_of(cols_needed),
                    col_types = cols(.default = "c"), show_col_types = FALSE)
    se_col <- if ("standard_error" %in% names(raw)) suppressWarnings(as.numeric(raw$standard_error))
              else (as.numeric(raw$hm_ci_upper) - as.numeric(raw$hm_ci_lower)) / 3.92
    out <- raw %>% transmute(chrom = as.character(hm_chrom), pos = as.integer(hm_pos),
                             hm_rsid = hm_rsid, hm_effect_allele = toupper(hm_effect_allele),
                             beta = as.numeric(hm_beta), se = se_col,
                             eaf = as.numeric(hm_effect_allele_frequency)) %>%
      filter(!is.na(beta), !is.na(se), se > 0)
    rm(raw); gc(); return(out)
  }
  if (cfg$parser == "raw") {
    cols_needed <- c("chromosome","base_pair_location","effect_allele","beta",
                     "standard_error","p_value","rsid","effect_allele_frequency","hm_code")
    raw <- read_tsv(cfg$file, col_select = any_of(cols_needed),
                    col_types = cols(.default = "c"), show_col_types = FALSE)
    out <- raw %>% filter(is.na(hm_code) | hm_code %in% c("5","10","11")) %>%
      transmute(chrom = as.character(chromosome), pos = as.integer(base_pair_location),
               hm_rsid = rsid, hm_effect_allele = toupper(effect_allele),
               beta = as.numeric(beta), se = as.numeric(standard_error),
               eaf = as.numeric(effect_allele_frequency)) %>%
      filter(!is.na(chrom), !is.na(pos), !is.na(beta), !is.na(se), se > 0, is.finite(beta))
    rm(raw); gc(); return(out)
  }
  if (cfg$parser == "finngen") {
    cols_needed <- c("#chrom","pos","alt","beta","sebeta","pval","rsids","af_alt")
    raw <- read_tsv(cfg$file, col_select = any_of(cols_needed),
                    col_types = cols(.default = "c"), show_col_types = FALSE)
    out <- raw %>% transmute(chrom = gsub("^chr","",as.character(`#chrom`),ignore.case=TRUE),
                             pos = as.integer(pos), hm_rsid = rsids,
                             hm_effect_allele = toupper(alt), beta = as.numeric(beta),
                             se = as.numeric(sebeta), eaf = as.numeric(af_alt)) %>%
      filter(!is.na(chrom), !is.na(pos), !is.na(beta), !is.na(se), se > 0, is.finite(beta))
    rm(raw); gc(); return(out)
  }
  stop(sprintf("Unknown parser: %s", cfg$parser))
}

# ── Step 1: genome-wide lead SNP + region, per protein (checkpointed) ──────
prots <- sort(unique(PAIRS$protein))
if (file.exists(LEADS_RDS)) {
  leads <- readRDS(LEADS_RDS)
  cat(sprintf("Checkpoint found: %d / %d proteins already have a lead.\n",
              length(leads), length(prots)))
} else {
  leads <- list()
}

for (prot in setdiff(prots, names(leads))) {
  f <- file.path(PQTL_DIR, paste0(prot, "_pqtl.tsv.gz"))
  if (!file.exists(f)) { cat(sprintf("[%s] MISSING pQTL file, skip\n", prot)); next }
  cat(sprintf("[%s] pass 1: genome-wide lead SNP...\n", prot))
  full <- tryCatch(parse_ukbpp_pqtl(f), error = function(e) { cat("  ERROR:", conditionMessage(e), "\n"); NULL })
  if (is.null(full) || nrow(full) == 0) next
  lead_row <- full %>% arrange(desc(log10p)) %>% slice(1)
  lead_chr <- lead_row$chrom[1]; lead_pos <- lead_row$pos[1]
  cat(sprintf("  lead: chr%s:%d (log10p=%.1f)\n", lead_chr, lead_pos, lead_row$log10p[1]))
  reg <- full %>% filter(chrom == lead_chr, pos >= lead_pos - WINDOW, pos <= lead_pos + WINDOW)
  rm(full); gc()
  if (nrow(reg) < 50) { cat("  SKIP: insufficient region SNPs\n"); next }
  region_file <- file.path(REGION_DIR, paste0(prot, ".rds"))
  saveRDS(reg, region_file)
  leads[[prot]] <- list(chr = lead_chr, pos = lead_pos, window = WINDOW, region_file = region_file)
  saveRDS(leads, LEADS_RDS)
  cat(sprintf("  checkpoint saved (%d / %d proteins)\n", length(leads), length(prots)))
}
cat(sprintf("\nLeads ready: %d / %d proteins\n", length(leads), length(prots)))

# ── Step 2: coloc, cancer-outer / protein-inner (cache GWAS per cancer) ────
results <- list()
for (cancer in unique(PAIRS$cancer)) {
  cfg <- CANCER_CONFIG[[cancer]]
  if (is.null(cfg) || !file.exists(cfg$file)) {
    cat(sprintf("\n=== %s: NO GWAS FILE, skip ===\n", cancer)); next
  }
  cat(sprintf("\n=== Cancer: %s (%s) ===\n", cancer, cfg$name))
  gwas <- tryCatch(load_gwas_cache(cfg), error = function(e) {
    cat("  ERROR loading GWAS:", conditionMessage(e), "\n"); NULL })
  if (is.null(gwas)) next
  cat(sprintf("  GWAS cached: %s SNPs\n", format(nrow(gwas), big.mark = ",")))

  prots_this <- PAIRS$protein[PAIRS$cancer == cancer]
  for (prot in prots_this) {
    lead <- leads[[prot]]
    if (is.null(lead)) { cat(sprintf("  [%s] no lead, skip\n", prot)); next }
    reg <- tryCatch(readRDS(lead$region_file), error = function(e) NULL)
    if (is.null(reg)) next
    cat(sprintf("  [%s x %s] chr%s:%d +/-%dkb\n", prot, cancer, lead$chr, lead$pos, WINDOW/1000))

    gwas_reg <- gwas %>% filter(chrom == as.character(lead$chr),
                                pos >= lead$pos - lead$window, pos <= lead$pos + lead$window)
    if (nrow(gwas_reg) < 50) { cat(sprintf("    insufficient GWAS SNPs: %d\n", nrow(gwas_reg))); next }

    merged <- inner_join(
      reg %>% select(chrom, pos, a1, a0, beta, se, eaf) %>%
        rename(beta.pqtl=beta, se.pqtl=se, eaf.pqtl=eaf,
               effect_allele.pqtl=a1, other_allele.pqtl=a0),
      gwas_reg %>% select(chrom, pos, hm_rsid, hm_effect_allele, beta, se) %>%
        rename(beta.gwas=beta, se.gwas=se, effect_allele.gwas=hm_effect_allele),
      by = c("chrom","pos"), relationship = "many-to-many"
    ) %>%
      mutate(needs_flip = !is.na(effect_allele.gwas) & !is.na(effect_allele.pqtl) &
               toupper(effect_allele.pqtl) != toupper(effect_allele.gwas),
             beta.gwas = ifelse(needs_flip, -beta.gwas, beta.gwas)) %>%
      filter(!is.na(beta.pqtl), !is.na(beta.gwas), se.pqtl > 0, se.gwas > 0) %>%
      arrange(desc(abs(beta.pqtl))) %>%
      distinct(chrom, pos, .keep_all = TRUE)

    cat(sprintf("    merged SNPs: %d\n", nrow(merged)))
    if (nrow(merged) < 50) next

    snp_ids <- paste0(merged$chrom, ":", merged$pos)
    co <- tryCatch(
      coloc.abf(
        dataset1 = list(beta = merged$beta.pqtl, varbeta = merged$se.pqtl^2,
                        N = PQTL_N, type = "quant", snp = snp_ids, MAF = merged$eaf.pqtl),
        dataset2 = list(beta = merged$beta.gwas, varbeta = merged$se.gwas^2,
                        N = cfg$n, s = cfg$n_cases / cfg$n, type = "cc", snp = snp_ids),
        p12 = P12
      ),
      error = function(e) { cat("    coloc.abf ERROR:", conditionMessage(e), "\n"); NULL }
    )
    if (is.null(co)) next
    s <- co$summary
    cat(sprintf("    PP.H4=%.4g PP.H3=%.4g Wald_p_at_lead=NA\n", s["PP.H4.abf"], s["PP.H3.abf"]))

    results[[paste(prot, cancer, sep="_")]] <- tibble(
      protein = prot, cancer = cancer, n_snps = nrow(merged),
      lead_chr = lead$chr, lead_pos = lead$pos,
      PP.H0 = unname(s["PP.H0.abf"]), PP.H1 = unname(s["PP.H1.abf"]),
      PP.H2 = unname(s["PP.H2.abf"]), PP.H3 = unname(s["PP.H3.abf"]),
      PP.H4 = unname(s["PP.H4.abf"]),
      dominant_hypothesis = c("H0","H1","H2","H3","H4")[which.max(c(
        s["PP.H0.abf"],s["PP.H1.abf"],s["PP.H2.abf"],s["PP.H3.abf"],s["PP.H4.abf"]))],
      passes_h4_0.8 = unname(s["PP.H4.abf"]) > 0.8
    )
  }
  rm(gwas); gc()
}

out <- bind_rows(results)
write_csv(out, OUT_CSV)
cat(sprintf("\n=== Done: %d / %d pairs produced a coloc result ===\n", nrow(out), nrow(PAIRS)))
cat(sprintf("saved -> %s\n", OUT_CSV))
if (nrow(out) > 0) {
  cat("\nAny PP.H4 > 0.8:\n")
  print(out %>% filter(PP.H4 > 0.8) %>% arrange(desc(PP.H4)))
  cat("\nTop 10 by PP.H4:\n")
  print(out %>% arrange(desc(PP.H4)) %>% head(10) %>%
          select(protein, cancer, n_snps, PP.H3, PP.H4, dominant_hypothesis))
}
cat("PARTA_COLOC_COMPLETE\n")
