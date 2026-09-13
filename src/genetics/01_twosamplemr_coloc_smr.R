# =============================================================================
# TwoSampleMR + Coloc + SMR/HEIDI — 25-Protein Pan-Cancer Panel
# Project: FASEB submission, Proteomics repo
# Date: 25 June 2026
# =============================================================================
#
# STRUCTURE:
#   PHASE 0  — Download links (read this section FIRST)
#   PHASE 1  — Package install + pQTL coverage check  [RUNS NOW ~5 min]
#   PHASE 2  — TwoSampleMR via OpenGWAS API           [RUNS NOW ~30-60 min]
#   PHASE 3  — Coloc via OpenGWAS API                 [RUNS NOW ~30-60 min]
#   PHASE 4  — Local coloc with downloaded files      [RUNS AFTER DOWNLOAD]
#   PHASE 5  — SMR + HEIDI with SMR binary            [RUNS AFTER DOWNLOAD]
#
# Run each PHASE in order. Each phase saves its outputs so later phases
# can resume without re-running earlier ones.
# =============================================================================


# =============================================================================
# PHASE 0 — DOWNLOAD LINKS
# =============================================================================
#
# Copy each link and paste into your browser or wget command.
# Sections marked [NEEDED FOR PHASE X] tell you when each file is required.
#
# ── TOOL: SMR binary (Windows) ────────────────────────────────────────────────
#
#   SMR download page (Yang Lab, Westlake University):
#   https://yanglab.westlake.edu.cn/software/smr/#Download
#
#   → Click "SMR for Windows" to download smr_win.zip (~5 MB)
#   → Extract smr.exe to:  revise_plan/smr_coloc/local_data/smr.exe
#   [NEEDED FOR PHASE 5]
#
#   Alternative (WSL2 / Linux binary):
#   → Click "SMR for Linux" on the same page
#   → Place binary at:  revise_plan/smr_coloc/local_data/smr
#   → chmod +x revise_plan/smr_coloc/local_data/smr
#
# ── LD REFERENCE PANEL (EUR 1000 Genomes, prepared for SMR) ───────────────────
#
#   NOTE: The Yang Lab SMR page does NOT provide an LD reference file.
#   Use CTG MAGMA 1000 Genomes EUR reference (chromosome-split PLINK binary format):
#   → Download: https://ctg.cncr.nl/software/MAGMA/ref_data/g1000_eur.zip  (~1 GB)
#   → Extract to:  revise_plan/smr_coloc/local_data/ld_ref/
#     (should produce: g1000_eur.bed / .bim / .fam — split by chromosome)
#   → Set LD_REF_PREFIX in Phase 5 to the extracted path prefix (e.g., "local_data/ld_ref/g1000_eur")
#   [NEEDED FOR PHASE 5]
#
# ── pQTL SUMMARY STATISTICS (Olink proteins, plasma) ─────────────────────────
#
#   OPTION A — UKB-PPP (Sun et al. 2023, Nature) — RECOMMENDED, n=54,219
#   Paper DOI:  https://doi.org/10.1038/s41586-023-06592-6
#   Data landing page on EMBL-EBI GWAS Catalog:
#   https://www.ebi.ac.uk/gwas/publications/36635386
#
#   → On that page, click each protein's GCST accession → "Download association data"
#   → You need summary stats for these 25 proteins:
#     FLT3, CNTN1, FCER2, PRDX6, LTA4H, XG, CCDC80, CXCL17, CXCL13, SLAMF7,
#     PAEP, PSPN, BMP4, WFDC2, TRAF2, KLK13, GLO1, GFAP, CEACAM5, CGA,
#     ADAMTS13, CRTAC1, TCL1A, ADAMTS15, NEFL
#   → Save each file to:  revise_plan/smr_coloc/local_data/pqtl_ukbpp/
#   → File naming convention to use: <PROTEIN>_pqtl.tsv.gz
#   [NEEDED FOR PHASE 4 and 5 — skip if using API-only coloc in Phase 3]
#
#   OPTION B — INTERVAL pQTL (Sun et al. 2018, Nature) — already in OpenGWAS API
#   DOI:  https://doi.org/10.1038/s41586-018-0175-2
#   Coverage: ~2,994 Olink proteins, n~3,301 — smaller but API accessible
#   No download needed — Phase 1 checks which of our 25 proteins are available.
#
#   OPTION C — deCODE pQTL (Ferkingstad et al. 2021, Nature Genetics)
#   DOI:  https://doi.org/10.1038/s41588-021-00978-w
#   Data: https://www.decode.com/summarydata/
#   → Requires free registration on the deCODE website
#   → Download protein-specific summary statistics
#   → SomaScan platform (different protein list — check coverage vs Olink)
#   [NEEDED FOR PHASE 4 as an alternative to UKB-PPP]
#
# ── CANCER GWAS SUMMARY STATISTICS ───────────────────────────────────────────
#
#   GWAS Catalog (landing page for all cancer GWAS):
#   https://www.ebi.ac.uk/gwas/
#
#   Harmonized summary statistics FTP root:
#   https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/harmonised/
#
#   Specific studies to download for PHASE 4/5:
#
#   BRC — Breast cancer (Michailidou et al. 2017, Nature):
#     GWAS Catalog: https://www.ebi.ac.uk/gwas/studies/GCST004988
#     DOI: https://doi.org/10.1038/nature24284
#     Direct FTP: https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST004001-GCST005000/GCST004988/harmonised/29059683-GCST004988-EFO_0000305.h.tsv.gz
#     (~2 GB, harmonised .h.tsv.gz confirmed)
#
#   CRC — Colorectal cancer (Fernandez-Rozadilla et al. 2023, Nature Genetics):
#     GWAS Catalog: https://www.ebi.ac.uk/gwas/studies/GCST90255675
#     DOI: https://doi.org/10.1038/s41588-022-01222-9
#     n = 185,616 (78,473 cases / 107,143 controls), European ancestry only
#     Direct FTP: https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90255001-GCST90256000/GCST90255675/harmonised/GCST90255675.h.tsv.gz
#     (~280 MB, harmonised .h.tsv.gz confirmed; NOTE: filename has no PMID prefix for this newer study)
#     License: Academic/non-commercial use only
#
#   LUNGC — Lung cancer (McKay et al. 2017, Nature Genetics):
#     GWAS Catalog: https://www.ebi.ac.uk/gwas/studies/GCST004748
#     DOI: https://doi.org/10.1038/ng.3875
#     n = 85,716 (29,266 cases / 56,450 controls), European ancestry
#     Direct FTP: https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST004001-GCST005000/GCST004748/harmonised/28604730-GCST004748-EFO_0001071.h.tsv.gz
#     (~451 MB, harmonised .h.tsv.gz confirmed)
#
#   OVC — Ovarian cancer (Phelan et al. 2017, Nature Genetics):
#     GWAS Catalog: https://www.ebi.ac.uk/gwas/studies/GCST004415
#     DOI: https://doi.org/10.1038/ng.3826
#     n = 63,347 (22,406 cases / 40,941 controls), invasive epithelial OVC
#     IMPORTANT: This study uses Phelan_Archive.zip (3.5 GB), NOT a standard .h.tsv.gz.
#     Archive download: https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST004001-GCST005000/GCST004415/Phelan_Archive.zip
#     After download: extract the archive; rename the all-OVC summary file to OVC_GCST004415.h.tsv.gz
#     and adjust parse_gwas_catalog_harmonised() column names to match the Phelan archive format.
#
#   PRC — Prostate cancer (Schumacher et al. 2018, Nature Genetics):
#     GWAS Catalog: https://www.ebi.ac.uk/gwas/studies/GCST006085
#     DOI: https://doi.org/10.1038/s41588-018-0142-8
#     Direct FTP: https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST006001-GCST007000/GCST006085/harmonised/29892016-GCST006085-EFO_0001663.h.tsv.gz
#     (~1.5 GB, harmonised .h.tsv.gz confirmed)
#
#   Save all cancer GWAS files to:  revise_plan/smr_coloc/local_data/cancer_gwas/
#   [NEEDED FOR PHASE 4 and 5 — Phase 2/3 uses OpenGWAS API instead]
#
# ── WGET COMMANDS (paste into bash terminal or WSL2) ─────────────────────────
#
# cd e:/Proteomics/revise_plan/smr_coloc/local_data/cancer_gwas
#
# wget "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST004001-GCST005000/GCST004988/harmonised/29059683-GCST004988-EFO_0000305.h.tsv.gz" -O BRC_GCST004988.h.tsv.gz
# wget "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90255001-GCST90256000/GCST90255675/harmonised/GCST90255675.h.tsv.gz" -O CRC_GCST90255675.h.tsv.gz
# wget "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST004001-GCST005000/GCST004748/harmonised/28604730-GCST004748-EFO_0001071.h.tsv.gz" -O LUNGC_GCST004748.h.tsv.gz
# wget "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST004001-GCST005000/GCST004415/Phelan_Archive.zip" -O OVC_Phelan_Archive.zip
# # After OVC download: unzip OVC_Phelan_Archive.zip; rename relevant file to OVC_GCST004415.h.tsv.gz
# wget "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST006001-GCST007000/GCST006085/harmonised/29892016-GCST006085-EFO_0001663.h.tsv.gz" -O PRC_GCST006085.h.tsv.gz
#
# TOTAL DOWNLOAD SIZE ESTIMATE:  ~8-12 GB for all 5 cancer GWAS
#                                 ~12-25 GB for 25 UKB-PPP protein files
#                                 ~1.3 GB for LD reference
#
# =============================================================================


# =============================================================================
# PHASE 1 — PACKAGE INSTALL + pQTL API COVERAGE CHECK  [RUNS NOW]
# =============================================================================
# Runtime: ~5 minutes (first install) or ~1 minute (if packages already present)
# Output:  results/phase1_pqtl_coverage.csv

cat("=== PHASE 1: Package install + pQTL coverage check ===\n")

# Suppress all interactive prompts for non-interactive / autonomous runs
options(repos = c(CRAN = "https://cloud.r-project.org"))
options(install.packages.compile.from.source = "never")
options(install.packages.ask = FALSE)
Sys.setenv(R_COMPILE_AND_INSTALL_PACKAGES = "never")


options(repos = c(CRAN = "https://cloud.r-project.org"))

LOCAL_R_LIB <- normalizePath("revise_plan/smr_coloc/r_libs", winslash = "/", mustWork = FALSE)
dir.create(LOCAL_R_LIB, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(LOCAL_R_LIB, .libPaths()))
OPENGWAS_JWT <- Sys.getenv("OPENGWAS_JWT", unset = "")

if (OPENGWAS_JWT == "") {
  cat("WARNING: OPENGWAS_JWT is not set. Phase 1 to 3 API calls will fail until a valid token is exported.\n")
}

# ── 1.1  Install packages ─────────────────────────────────────────────────────
if (!requireNamespace("rlang", quietly = TRUE) || packageVersion("rlang") < "1.1.7")
  install.packages("rlang")
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
if (!requireNamespace("TwoSampleMR", quietly = TRUE))
  remotes::install_github("MRCIEU/TwoSampleMR")
if (!requireNamespace("ieugwasr", quietly = TRUE))
  remotes::install_github("MRCIEU/ieugwasr")
if (!requireNamespace("coloc", quietly = TRUE))
  install.packages("coloc")
if (!requireNamespace("dplyr",   quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("readr",   quietly = TRUE)) install.packages("readr")
if (!requireNamespace("tibble",  quietly = TRUE)) install.packages("tibble")
if (!requireNamespace("tidyr",   quietly = TRUE)) install.packages("tidyr")

library(rlang, lib.loc = LOCAL_R_LIB)
library(TwoSampleMR)
library(ieugwasr)
library(coloc)
library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)

# ── 1.2  Define the 25 panel proteins ─────────────────────────────────────────
PANEL_25 <- c("FLT3", "CNTN1", "FCER2", "PRDX6", "LTA4H", "XG", "CCDC80",
              "CXCL17", "CXCL13", "SLAMF7", "PAEP", "PSPN", "BMP4", "WFDC2",
              "TRAF2", "KLK13", "GLO1", "GFAP", "CEACAM5", "CGA", "ADAMTS13",
              "CRTAC1", "TCL1A", "ADAMTS15", "NEFL")

# ── 1.3  Fetch all available pQTL datasets from OpenGWAS ──────────────────────
cat("Fetching OpenGWAS dataset catalogue (may take ~60 seconds)...\n")
all_gwas <- available_outcomes()

# Filter for pQTL / protein QTL datasets
pqtl_datasets <- all_gwas %>%
  filter(grepl("prot|pQTL|protein|plasma", trait, ignore.case = TRUE) |
           grepl("^prot-", id)) %>%
  select(id, trait, author, year, sample_size, nsnp, unit)

cat("Found", nrow(pqtl_datasets), "potential pQTL datasets in OpenGWAS.\n")

# ── 1.4  Match each panel protein to an OpenGWAS pQTL dataset ─────────────────
coverage <- lapply(PANEL_25, function(prot) {
  matches <- pqtl_datasets %>%
    filter(grepl(paste0("\\b", prot, "\\b"), trait, ignore.case = TRUE))

  if (nrow(matches) == 0) {
    # Fallback: partial match
    matches <- pqtl_datasets %>%
      filter(grepl(prot, trait, ignore.case = TRUE))
  }

  if (nrow(matches) > 0) {
    # Prefer INTERVAL (prot-a-*) or UKB-PPP if available; else take first hit
    best <- matches %>%
      arrange(desc(grepl("^prot-a-", id)),
              desc(sample_size)) %>%
      slice(1)
    data.frame(
      protein        = prot,
      found_in_api   = TRUE,
      gwas_id        = best$id,
      trait_label    = best$trait,
      sample_size    = best$sample_size,
      source         = ifelse(grepl("^prot-a-", best$id), "INTERVAL", "other"),
      n_matches      = nrow(matches),
      stringsAsFactors = FALSE
    )
  } else {
    data.frame(
      protein        = prot,
      found_in_api   = FALSE,
      gwas_id        = NA_character_,
      trait_label    = NA_character_,
      sample_size    = NA_integer_,
      source         = "not_found",
      n_matches      = 0L,
      stringsAsFactors = FALSE
    )
  }
})
coverage_df <- bind_rows(coverage)

n_found  <- sum(coverage_df$found_in_api)
n_total  <- length(PANEL_25)
cat(sprintf("\npQTL API coverage: %d / %d proteins found in OpenGWAS\n", n_found, n_total))
print(coverage_df[, c("protein", "found_in_api", "gwas_id", "sample_size", "source")])

dir.create("revise_plan/smr_coloc/results", showWarnings = FALSE, recursive = TRUE)
write_csv(coverage_df, "revise_plan/smr_coloc/results/phase1_pqtl_coverage.csv")
cat("Saved: revise_plan/smr_coloc/results/phase1_pqtl_coverage.csv\n\n")

# ── 1.5  Decision gate ────────────────────────────────────────────────────────
if (n_found < 5) {
  cat("WARNING: Fewer than 5 proteins found in OpenGWAS API.\n")
  cat("Phase 2 and 3 (API-based analyses) will be limited.\n")
  cat("Proceed to Phase 4 using local UKB-PPP downloads (see Phase 0 links).\n")
} else {
  cat(sprintf("PROCEED: %d proteins available via API. Running Phase 2 and 3.\n", n_found))
}

# Keep only proteins with API pQTL data for downstream phases
proteins_api <- coverage_df %>% filter(found_in_api)


# =============================================================================
# PHASE 2 — TwoSampleMR via OpenGWAS API  [RUNS NOW]
# =============================================================================
# Runtime: ~30-60 minutes (API calls for each protein-cancer pair)
# Output:  results/phase2_mr_results.csv
#          results/phase2_mr_ivw_fdr.csv
#          figures/phase2_mr_forest_plot.pdf

cat("\n=== PHASE 2: TwoSampleMR via OpenGWAS API ===\n")

# ── 2.1  Cancer outcome IDs ───────────────────────────────────────────────────
# These IDs are from IEU OpenGWAS. Verify them by running:
#   available_outcomes() %>% filter(grepl("breast cancer|colorectal|lung cancer|ovarian|prostate", trait, ignore.case=TRUE))
# before trusting them. They are the best estimates as of 2023-2024.

CANCER_OUTCOMES_API <- list(
  BRC   = list(id = "ebi-a-GCST90018799", name = "Breast cancer",     n = 257730, n_cases = 17389),
  CRC   = list(id = "ebi-a-GCST90018808", name = "Colorectal cancer", n = 470002, n_cases = 6581),
  LUNGC = list(id = "ebi-a-GCST90018875", name = "Lung cancer",       n = 492803, n_cases = 3791),
  OVC   = list(id = "ebi-a-GCST90018888", name = "Ovarian cancer",    n = 246520, n_cases = 1588),
  PRC   = list(id = "ebi-a-GCST90018905", name = "Prostate cancer",   n = 211227, n_cases = 11599)
)

CANCER_OUTCOMES_LOCAL <- list(
  BRC   = list(id = "GCST004988",   name = "Breast cancer",     n = 228951, n_cases = 122977),
  CRC   = list(id = "GCST90255675", name = "Colorectal cancer", n = 185616, n_cases = 78473),  # Fernandez-Rozadilla 2023, EUR-only
  LUNGC = list(id = "GCST004748",   name = "Lung cancer",       n = 85716,  n_cases = 29266),  # McKay 2017
  OVC   = list(id = "GCST004415",   name = "Ovarian cancer",    n = 63347,  n_cases = 22406),  # Phelan 2017, invasive epithelial OVC
  PRC   = list(id = "GCST006085",   name = "Prostate cancer",   n = 140254, n_cases = 79148)
)

CANCER_OUTCOMES <- CANCER_OUTCOMES_API

# ── 2.2  Verify outcome IDs are present in OpenGWAS ──────────────────────────
cat("Verifying cancer outcome IDs in OpenGWAS...\n")
for (cancer in names(CANCER_OUTCOMES_API)) {
  info <- all_gwas %>% filter(id == CANCER_OUTCOMES_API[[cancer]]$id)
  if (is.null(info) || nrow(info) == 0) {
    cat(sprintf("  %s: ID %s NOT found — search manually with available_outcomes()\n",
                cancer, CANCER_OUTCOMES_API[[cancer]]$id))
  } else {
    cat(sprintf("  %s: OK — %s\n", cancer, info$trait[1]))
  }
}

# ── 2.3  Run TwoSampleMR for each protein-cancer pair ────────────────────────
all_mr <- list()

for (i in seq_len(nrow(proteins_api))) {
  prot    <- proteins_api$protein[i]
  pqtl_id <- proteins_api$gwas_id[i]
  cat(sprintf("\nExtracting instruments for %s (OpenGWAS ID: %s)...\n", prot, pqtl_id))

  exp_dat <- tryCatch(
    extract_instruments(
      outcomes  = pqtl_id,
      p1        = 5e-8,
      clump     = TRUE,
      r2        = 0.001,
      kb        = 10000,
      opengwas_jwt = OPENGWAS_JWT
    ),
    error = function(e) { cat("  ERROR extracting instruments:", conditionMessage(e), "\n"); NULL }
  )

  if (is.null(exp_dat) || nrow(exp_dat) == 0) {
    cat(sprintf("  %s: no genome-wide significant instruments found.\n", prot))
    next
  }

  # Count cis vs trans instruments (cis = within 1 Mb of gene body)
  cat(sprintf("  %s: %d instruments extracted\n", prot, nrow(exp_dat)))

  for (cancer in names(CANCER_OUTCOMES)) {
    out_id <- CANCER_OUTCOMES[[cancer]]$id
    cat(sprintf("    -> %s ... ", cancer))

    out_dat <- tryCatch(
      extract_outcome_data(
        snps     = exp_dat$SNP,
        outcomes = out_id,
        proxies  = TRUE,
        rsq      = 0.8,
        opengwas_jwt = OPENGWAS_JWT
      ),
      error = function(e) { cat("ERROR\n"); NULL }
    )
    if (is.null(out_dat) || nrow(out_dat) == 0) { cat("no SNPs in outcome\n"); next }

    dat <- harmonise_data(exp_dat, out_dat, action = 2)
    if (nrow(dat) == 0) { cat("no harmonised SNPs\n"); next }

    res <- tryCatch(
      mr(dat, method_list = c("mr_ivw",
                               "mr_egger_regression",
                               "mr_weighted_median",
                               "mr_wald_ratio")),
      error = function(e) { cat("MR error\n"); NULL }
    )
    if (is.null(res)) next

    # Pleiotropy test (MR-Egger intercept)
    pleio <- tryCatch(mr_pleiotropy_test(dat), error = function(e) NULL)
    hetero <- tryCatch(mr_heterogeneity(dat), error = function(e) NULL)

    res$protein       <- prot
    res$cancer        <- cancer
    res$n_instruments <- sum(dat$mr_keep)
    if (!is.null(pleio))  res$egger_intercept_p  <- pleio$pval[1]
    if (!is.null(hetero)) res$heterogeneity_q_p  <- hetero$Q_pval[hetero$method == "Inverse variance weighted"][1]

    all_mr[[paste(prot, cancer, sep = "_")]] <- res
    cat(sprintf("OK (%d SNPs used)\n", res$n_instruments[1]))
  }
}

# ── 2.4  Save and summarise MR results ───────────────────────────────────────
if (length(all_mr) == 0) {
  cat("\nNo MR results produced. Check API connectivity and pQTL coverage.\n")
} else {
  mr_all <- bind_rows(all_mr)
  write_csv(mr_all, "revise_plan/smr_coloc/results/phase2_mr_results.csv")
  cat(sprintf("\nSaved %d MR estimates: revise_plan/smr_coloc/results/phase2_mr_results.csv\n",
              nrow(mr_all)))

  # FDR correction on IVW only
  ivw <- mr_all %>%
    filter(method == "Inverse variance weighted") %>%
    mutate(OR    = exp(b),
           OR_lo = exp(b - 1.96 * se),
           OR_hi = exp(b + 1.96 * se),
           fdr   = p.adjust(pval, method = "BH"))
  write_csv(ivw, "revise_plan/smr_coloc/results/phase2_mr_ivw_fdr.csv")

  n_nominal <- sum(ivw$pval < 0.05, na.rm = TRUE)
  n_fdr     <- sum(ivw$fdr  < 0.05, na.rm = TRUE)
  cat(sprintf("IVW summary: %d nominal (p<0.05), %d FDR-significant (FDR<0.05)\n",
              n_nominal, n_fdr))

  # Forest plot (all nominal IVW hits)
  plot_dat <- ivw %>% filter(pval < 0.05)
  if (nrow(plot_dat) > 0) {
    plot_dat$label <- paste0(plot_dat$protein, " → ", plot_dat$cancer)
    p <- ggplot(plot_dat,
                aes(x = OR, y = reorder(label, OR),
                    xmin = OR_lo, xmax = OR_hi, colour = cancer)) +
      geom_errorbarh(height = 0.25, linewidth = 0.7) +
      geom_point(size = 3) +
      geom_vline(xintercept = 1, linetype = "dashed", colour = "grey50") +
      scale_x_log10(breaks = c(0.5, 0.7, 1.0, 1.5, 2.0)) +
      scale_colour_brewer(palette = "Set2") +
      labs(x = "Odds Ratio per 1 SD increase in plasma protein level (IVW MR)",
           y = NULL,
           title = "TwoSampleMR: 25-protein panel vs 5 cancer GWAS",
           subtitle = sprintf("Showing %d nominal hits (p < 0.05) out of %d protein-cancer pairs tested",
                              nrow(plot_dat), nrow(ivw)),
           colour = "Cancer type") +
      theme_bw(base_size = 12) +
      theme(legend.position = "right")

    dir.create("revise_plan/smr_coloc/figures", showWarnings = FALSE, recursive = TRUE)
    ggsave("revise_plan/smr_coloc/figures/phase2_mr_forest_plot.pdf",
           p, width = 10, height = max(5, nrow(plot_dat) * 0.45))
    ggsave("revise_plan/smr_coloc/figures/phase2_mr_forest_plot.png",
           p, width = 10, height = max(5, nrow(plot_dat) * 0.45), dpi = 400)
    cat("Saved: revise_plan/smr_coloc/figures/phase2_mr_forest_plot.pdf\n")
  } else {
    cat("No nominal MR hits to plot. All results in phase2_mr_ivw_fdr.csv.\n")
  }
}


# =============================================================================
# PHASE 3 — COLOC VIA OPENGEWAS API  [RUNS NOW for proteins found in Phase 1]
# =============================================================================
# For each nominally significant MR pair: pull the regional summary stats
# (~500kb window around the cis-pQTL lead SNP) and run coloc.abf.
#
# PP.H4 = posterior probability that pQTL and cancer GWAS share a causal variant
# PP.H3 = posterior probability of two distinct causal variants (no colocalization)
# PP.H0 = neither signal in the region
#
# Runtime: ~2-5 minutes per protein-cancer pair
# Output:  results/phase3_coloc_results.csv
#          figures/phase3_coloc_ppH4_heatmap.pdf

cat("\n=== PHASE 3: Coloc via OpenGWAS API ===\n")

# Load MR results from Phase 2
mr_ivw_file <- "revise_plan/smr_coloc/results/phase2_mr_ivw_fdr.csv"
if (!file.exists(mr_ivw_file)) {
  cat("Phase 2 results not found. Run Phase 2 first.\n")
} else {
  ivw <- read_csv(mr_ivw_file, show_col_types = FALSE)

  # Run coloc on all pairs with p < 0.1 (broader net — coloc decides significance)
  pairs_to_test <- ivw %>%
    filter(pval < 0.1) %>%
    select(protein, cancer, pval, OR) %>%
    left_join(coverage_df %>% select(protein, gwas_id), by = "protein")

  cat(sprintf("Testing %d protein-cancer pairs with coloc (MR p < 0.1)\n",
              nrow(pairs_to_test)))

  coloc_results <- list()

  for (i in seq_len(nrow(pairs_to_test))) {
    prot      <- pairs_to_test$protein[i]
    cancer    <- pairs_to_test$cancer[i]
    pqtl_id   <- pairs_to_test$gwas_id[i]
    gwas_id   <- CANCER_OUTCOMES[[cancer]]$id
    gwas_n    <- CANCER_OUTCOMES[[cancer]]$n
    gwas_ncas <- CANCER_OUTCOMES[[cancer]]$n_cases

    cat(sprintf("  [%d/%d] %s - %s\n", i, nrow(pairs_to_test), prot, cancer))

    # Pull lead cis-pQTL SNP (most significant SNP for this protein)
    lead_snp_data <- tryCatch({
      top_assoc <- associations(
        id       = pqtl_id,
        proxies  = 0
      )
      if (is.null(top_assoc) || nrow(top_assoc) == 0) NULL
      else top_assoc %>% arrange(p) %>% slice(1)
    }, error = function(e) NULL)

    if (is.null(lead_snp_data)) {
      cat("    Could not retrieve lead pQTL SNP. Skipping.\n")
      next
    }

    lead_rsid <- lead_snp_data$rsid[1]
    cat(sprintf("    Lead SNP: %s\n", lead_rsid))

    # Pull regional pQTL summary stats (±500kb around lead SNP)
    pqtl_region <- tryCatch(
      phewas(
        variants = lead_rsid,
        pval     = 1,           # all variants returned
        id       = pqtl_id
      ),
      error = function(e) NULL
    )

    # Alternative: use associations() with a list of nearby SNPs
    if (is.null(pqtl_region) || nrow(pqtl_region) == 0) {
      cat("    phewas failed. Trying associations() approach...\n")
      # Get all variants from OpenGWAS for this dataset in the region
      # This requires knowing the chromosome and position of the lead SNP
      # If lead_snp_data has chr/position columns, use them
      if (all(c("chr", "position") %in% names(lead_snp_data))) {
        chr_num <- lead_snp_data$chr[1]
        pos     <- lead_snp_data$position[1]
        window  <- 500000  # 500kb
        cat(sprintf("    Trying regional query: chr%s:%d-%d\n",
                    chr_num, pos - window, pos + window))
      }
      cat("    Could not pull regional pQTL data. Skipping coloc.\n")
      next
    }

    # Pull matching GWAS summary stats for the same region SNPs
    gwas_region <- tryCatch(
      extract_outcome_data(
        snps     = pqtl_region$rsid,
        outcomes = gwas_id,
        proxies  = FALSE,
        opengwas_jwt = OPENGWAS_JWT
      ),
      error = function(e) NULL
    )
    if (is.null(gwas_region) || nrow(gwas_region) == 0) {
      cat("    No GWAS data for regional SNPs. Skipping.\n")
      next
    }

    # Harmonise SNP direction
    merged <- inner_join(
      pqtl_region  %>% select(rsid, beta, se, p, eaf) %>% rename(SNP=rsid, beta.pqtl=beta, se.pqtl=se, p.pqtl=p, eaf.pqtl=eaf),
      gwas_region  %>% select(SNP, beta.outcome, se.outcome) %>% rename(beta.gwas=beta.outcome, se.gwas=se.outcome),
      by = "SNP"
    ) %>% filter(!is.na(beta.pqtl), !is.na(beta.gwas),
                 !is.na(se.pqtl),   !is.na(se.gwas),
                 se.pqtl > 0,        se.gwas > 0)

    if (nrow(merged) < 50) {
      cat(sprintf("    Only %d overlapping SNPs after merge — too few for coloc. Skipping.\n", nrow(merged)))
      next
    }
    cat(sprintf("    Coloc input: %d SNPs\n", nrow(merged)))

    # Run coloc.abf
    coloc_out <- tryCatch(
      coloc.abf(
        dataset1 = list(
          beta    = merged$beta.pqtl,
          varbeta = merged$se.pqtl ^ 2,
          N       = proteins_api$sample_size[proteins_api$protein == prot],
          type    = "quant",
          snp     = merged$SNP,
          MAF     = merged$eaf.pqtl
        ),
        dataset2 = list(
          beta    = merged$beta.gwas,
          varbeta = merged$se.gwas ^ 2,
          N       = gwas_n,
          s       = gwas_ncas / gwas_n,
          type    = "cc",
          snp     = merged$SNP
        ),
        p12 = 1e-5   # prior probability of shared causal variant
      ),
      error = function(e) {
        cat("    coloc.abf ERROR:", conditionMessage(e), "\n")
        NULL
      }
    )
    if (is.null(coloc_out)) next

    summ <- coloc_out$summary
    cat(sprintf("    PP.H4 (shared causal variant) = %.3f\n", summ["PP.H4.abf"]))

    coloc_results[[paste(prot, cancer, sep = "_")]] <- data.frame(
      protein     = prot,
      cancer      = cancer,
      mr_p        = pairs_to_test$pval[i],
      mr_OR       = pairs_to_test$OR[i],
      n_snps      = nrow(merged),
      PP.H0       = summ["PP.H0.abf"],   # no signal in either
      PP.H1       = summ["PP.H1.abf"],   # pQTL only
      PP.H2       = summ["PP.H2.abf"],   # GWAS only
      PP.H3       = summ["PP.H3.abf"],   # two distinct causal variants
      PP.H4       = summ["PP.H4.abf"],   # shared causal variant (colocalization)
      colocalises = summ["PP.H4.abf"] > 0.8,
      stringsAsFactors = FALSE
    )
  }

  if (length(coloc_results) == 0) {
    cat("\nNo coloc results produced (no regional data available via API).\n")
    cat("Use Phase 4 (local files) for coloc after downloading pQTL summary stats.\n")
  } else {
    coloc_df <- bind_rows(coloc_results)
    write_csv(coloc_df, "revise_plan/smr_coloc/results/phase3_coloc_results.csv")

    n_coloc <- sum(coloc_df$colocalises, na.rm = TRUE)
    cat(sprintf("\nColoc complete: %d of %d pairs show PP.H4 > 0.80\n",
                n_coloc, nrow(coloc_df)))
    print(coloc_df %>% select(protein, cancer, mr_p, PP.H3, PP.H4, colocalises) %>%
            arrange(desc(PP.H4)))

    # PP.H4 heatmap
    if (nrow(coloc_df) >= 2) {
      hm_dat <- coloc_df %>%
        select(protein, cancer, PP.H4) %>%
        tidyr::pivot_wider(names_from = cancer, values_from = PP.H4)

      p_coloc <- ggplot(coloc_df, aes(x = cancer, y = protein, fill = PP.H4)) +
        geom_tile(colour = "white", linewidth = 0.5) +
        geom_text(aes(label = sprintf("%.2f", PP.H4)), size = 3) +
        scale_fill_gradient2(low = "white", mid = "#FEE08B", high = "#D53E4F",
                             midpoint = 0.5, limits = c(0, 1),
                             name = "PP.H4\n(colocalisation)") +
        labs(title = "Coloc posterior probability of shared causal variant",
             subtitle = "PP.H4 > 0.80 = strong evidence for colocalization",
             x = "Cancer type", y = "Protein") +
        theme_bw(base_size = 12) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))

      ggsave("revise_plan/smr_coloc/figures/phase3_coloc_ppH4_heatmap.pdf",
             p_coloc, width = 8, height = max(4, nrow(coloc_df) * 0.5))
      ggsave("revise_plan/smr_coloc/figures/phase3_coloc_ppH4_heatmap.png",
             p_coloc, width = 8, height = max(4, nrow(coloc_df) * 0.5), dpi = 400)
      cat("Saved: revise_plan/smr_coloc/figures/phase3_coloc_ppH4_heatmap.pdf\n")
    }
  }
}


# =============================================================================
# PHASE 4 — LOCAL COLOC WITH DOWNLOADED GWAS + pQTL FILES  [AFTER DOWNLOAD]
# =============================================================================
# This phase runs coloc using the full genome-wide files downloaded in Phase 0.
# It does NOT require the SMR binary — only the coloc R package.
#
# Prerequisites (see Phase 0 download links):
#   revise_plan/smr_coloc/local_data/cancer_gwas/BRC_GCST004988.h.tsv.gz
#   revise_plan/smr_coloc/local_data/cancer_gwas/CRC_GCST006405.h.tsv.gz
#   revise_plan/smr_coloc/local_data/cancer_gwas/LUNGC_GCST007088.h.tsv.gz
#   revise_plan/smr_coloc/local_data/cancer_gwas/OVC_GCST004656.h.tsv.gz
#   revise_plan/smr_coloc/local_data/cancer_gwas/PRC_GCST006085.h.tsv.gz
#   revise_plan/smr_coloc/local_data/pqtl_ukbpp/<PROTEIN>_pqtl.tsv.gz  (one per protein)
#
# Output:  results/phase4_coloc_local_results.csv

cat("\n=== PHASE 4: Local coloc with downloaded files ===\n")

CANCER_OUTCOMES <- CANCER_OUTCOMES_LOCAL

GWAS_LOCAL_DIR <- "revise_plan/smr_coloc/local_data/cancer_gwas"
PQTL_LOCAL_DIR <- "revise_plan/smr_coloc/local_data/pqtl_ukbpp"

# Actual filenames as downloaded (no renaming needed)
GWAS_FILES <- list(
  BRC   = file.path(GWAS_LOCAL_DIR, "29059683-GCST004988-EFO_0000305.h.tsv.gz"),
  CRC   = file.path(GWAS_LOCAL_DIR, "GCST90255675.h.tsv.gz"),
  LUNGC = file.path(GWAS_LOCAL_DIR, "28604730-GCST004748-EFO_0001071.h.tsv.gz"),
  OVC   = "PHELAN_CHR_SPLIT",   # OVC uses chr-split archive — see parse_phelan_ovc() below
  PRC   = file.path(GWAS_LOCAL_DIR, "29892016-GCST006085-EFO_0001663.h.tsv.gz")
)
OVC_ARCHIVE_DIR <- file.path(GWAS_LOCAL_DIR, "Phelan_Archive")

# ── Helper: standard column names for GWAS Catalog harmonised format ──────────
# GWAS Catalog harmonised format columns (adjust if different):
#   hm_rsid, hm_chrom, hm_pos, hm_effect_allele, hm_other_allele,
#   hm_beta, hm_se, hm_effect_allele_frequency, p_value
parse_gwas_catalog_harmonised <- function(filepath, chrom_filter, pos_min, pos_max) {
  cat(sprintf("    Loading GWAS region chr%s:%d-%d from %s...\n",
              chrom_filter, pos_min, pos_max, basename(filepath)))
  gwas_full <- read_tsv(filepath, col_types = cols(.default = "c"), show_col_types = FALSE)
  cn <- names(gwas_full)

  # Handle two GWAS Catalog harmonised formats:
  # Format A (older studies, e.g. BRC GCST004988):
  #   hm_chrom, hm_pos, hm_effect_allele, hm_other_allele, hm_beta, hm_se, hm_rsid
  # Format B (newer studies, e.g. CRC GCST90255675):
  #   chromosome, base_pair_location, effect_allele, other_allele, beta, standard_error, rsid
  if ("hm_pos" %in% cn) {
    gwas_full <- gwas_full %>%
      mutate(
        chrom            = as.character(hm_chrom),
        pos              = as.integer(hm_pos),
        hm_rsid          = if ("hm_rsid" %in% cn) hm_rsid else NA_character_,
        hm_effect_allele = hm_effect_allele,
        hm_other_allele  = hm_other_allele,
        beta             = as.numeric(hm_beta),
        se               = as.numeric(hm_se),
        p                = as.numeric(p_value),
        eaf              = as.numeric(if ("hm_effect_allele_frequency" %in% cn)
                                        hm_effect_allele_frequency else NA_real_)
      )
  } else {
    # Format B: use raw columns
    gwas_full <- gwas_full %>%
      mutate(
        chrom            = as.character(chromosome),
        pos              = as.integer(base_pair_location),
        hm_rsid          = if ("rsid" %in% cn) rsid else
                           if ("variant_id" %in% cn) variant_id else NA_character_,
        hm_effect_allele = effect_allele,
        hm_other_allele  = other_allele,
        beta             = as.numeric(beta),
        se               = as.numeric(standard_error),
        p                = as.numeric(p_value),
        eaf              = as.numeric(if ("effect_allele_frequency" %in% cn)
                                        effect_allele_frequency else NA_real_)
      )
  }
  gwas_full %>%
    filter(chrom == as.character(chrom_filter),
           pos   >= pos_min,
           pos   <= pos_max,
           !is.na(beta), !is.na(se), se > 0)
}

# ── Helper: read Phelan 2017 OVC archive (chr-split .txt files, GRCh37) ───────
# Columns in Summary_chr*.txt:
#   Chromosome, Position, 1000G_SNPname, Effect, Baseline, EAF
#   overall_OR, overall_SE, overall_pvalue  (all-OVC endpoint)
# NOTE: Positions are GRCh37/hg19. UKB-PPP pQTL data is GRCh38.
#       Coloc will fail silently if builds don't match — verify lead SNP rsid
#       matches between pQTL and OVC before interpreting PP.H4.
parse_phelan_ovc <- function(archive_dir, chrom_filter, pos_min, pos_max) {
  chr_file <- file.path(archive_dir, paste0("Summary_chr", chrom_filter, ".txt"))
  if (!file.exists(chr_file)) {
    cat(sprintf("    OVC chr file not found: %s\n", chr_file)); return(NULL)
  }
  cat(sprintf("    Loading OVC region chr%s:%d-%d from Phelan archive...\n",
              chrom_filter, pos_min, pos_max))
  ovc <- read_tsv(chr_file, col_types = cols(.default = "c"), show_col_types = FALSE)
  ovc %>%
    mutate(
      chrom            = as.character(Chromosome),
      pos              = as.integer(Position),
      rsid             = `1000G_SNPname`,
      hm_effect_allele = Effect,
      hm_other_allele  = Baseline,
      eaf              = as.numeric(EAF),
      beta             = log(as.numeric(overall_OR)),   # OR -> log-OR
      se               = as.numeric(overall_SE),
      p                = as.numeric(overall_pvalue)
    ) %>%
    filter(chrom == as.character(chrom_filter),
           pos   >= pos_min,
           pos   <= pos_max,
           !is.na(beta), !is.na(se), se > 0, is.finite(beta))
}

# ── Helper: read UKB-PPP pQTL file for a single protein ──────────────────────
# UKB-PPP regenie format: space-delimited (not TSV), columns:
#   CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ INFO N TEST BETA SE CHISQ LOG10P EXTRA
# ID is CHR:POS:REF:ALT:imp:v1 — NOT an rsID. Merge with GWAS is position-based.
parse_ukbpp_pqtl <- function(filepath, chrom_filter, pos_min, pos_max) {
  cat(sprintf("    Loading pQTL region from %s...\n", basename(filepath)))
  pqtl_full <- read_delim(filepath, delim = " ",
                          col_types = cols(.default = "c"), show_col_types = FALSE)
  names(pqtl_full) <- toupper(names(pqtl_full))
  pqtl_full %>%
    mutate(
      chrom = as.character(CHROM),
      pos   = as.integer(GENPOS),
      rsid  = ID,          # variant ID (CHR:POS:REF:ALT); used as fallback key
      a1    = toupper(ALLELE1),   # effect allele
      a0    = toupper(ALLELE0),   # other allele
      beta  = as.numeric(BETA),
      se    = as.numeric(SE),
      eaf   = as.numeric(A1FREQ),
      p     = 10 ^ (-as.numeric(LOG10P))
    ) %>%
    filter(chrom == as.character(chrom_filter),
           pos   >= pos_min,
           pos   <= pos_max,
           !is.na(beta), !is.na(se), se > 0, is.finite(beta))
}

# ── Check which files are present ────────────────────────────────────────────
gwas_ready <- sapply(names(GWAS_FILES), function(c) {
  if (c == "OVC") dir.exists(OVC_ARCHIVE_DIR) else file.exists(GWAS_FILES[[c]])
})
cat("Cancer GWAS files present:\n")
for (cancer in names(gwas_ready)) {
  status <- if (cancer == "OVC") "Phelan_Archive dir" else GWAS_FILES[[cancer]]
  cat(sprintf("  %s: %s  [%s]\n", cancer,
              ifelse(gwas_ready[cancer], "READY", "MISSING"),
              basename(status)))
}

pqtl_files_present <- file.exists(
  file.path(PQTL_LOCAL_DIR, paste0(PANEL_25, "_pqtl.tsv.gz"))
)
names(pqtl_files_present) <- PANEL_25
cat(sprintf("\npQTL files present: %d / %d\n",
            sum(pqtl_files_present), length(pqtl_files_present)))

if (!any(gwas_ready) || !any(pqtl_files_present)) {
  cat("\nPhase 4 skipped — download files listed in Phase 0 first.\n")
} else {
  # Phase 4 tests ALL 25 proteins x available cancers (not filtered by Phase 2 API results).
  # Phase 2 MR covered only 3/25 proteins via OpenGWAS; all 25 pQTL files are now local.
  # I/O strategy: read each pQTL file once (lead + region), read each GWAS file once per
  # cancer (cached in memory), then extract windows in-memory per protein. This reduces
  # GWAS I/O from 125 reads to 5 reads.

  prots_available   <- names(pqtl_files_present)[pqtl_files_present]
  cancers_available <- names(gwas_ready)[gwas_ready]

  # ── Step 1: Pre-compute lead pQTL positions and regional data for all proteins ─
  cat(sprintf("\nStep 1: Pre-computing lead pQTL positions for %d proteins...\n",
              length(prots_available)))
  pqtl_leads <- list()
  for (prot in prots_available) {
    pqtl_file <- file.path(PQTL_LOCAL_DIR, paste0(prot, "_pqtl.tsv.gz"))
    cat(sprintf("  [%s] Reading pQTL file...\n", prot))
    pqtl_all <- tryCatch(
      read_delim(pqtl_file, delim = " ",
                 col_types = cols(.default = "c"), show_col_types = FALSE),
      error = function(e) NULL
    )
    if (is.null(pqtl_all)) { cat(sprintf("  SKIP %s: read error\n", prot)); next }
    names(pqtl_all) <- toupper(names(pqtl_all))
    pqtl_all <- pqtl_all %>% mutate(LOG10P_val = as.numeric(LOG10P))
    lead_row <- pqtl_all %>% arrange(desc(LOG10P_val)) %>% slice(1)
    if (nrow(lead_row) == 0) { cat(sprintf("  SKIP %s: no lead SNP\n", prot)); next }
    lead_chr <- lead_row$CHROM[1]
    lead_pos <- as.integer(lead_row$GENPOS[1])
    window   <- 500000L
    cat(sprintf("    Lead: chr%s:%d (LOG10P=%.1f)\n", lead_chr, lead_pos, lead_row$LOG10P_val[1]))
    pqtl_reg <- tryCatch(
      parse_ukbpp_pqtl(pqtl_file, lead_chr, lead_pos - window, lead_pos + window),
      error = function(e) NULL
    )
    if (is.null(pqtl_reg) || nrow(pqtl_reg) < 50) {
      cat(sprintf("  SKIP %s: insufficient pQTL SNPs in region\n", prot)); next
    }
    pqtl_leads[[prot]] <- list(chr = lead_chr, pos = lead_pos, window = window,
                                pqtl_reg = pqtl_reg)
    rm(pqtl_all); gc()
  }
  cat(sprintf("Lead positions ready for %d / %d proteins.\n",
              length(pqtl_leads), length(prots_available)))

  # ── Step 2: Loop cancer-outer, protein-inner; cache full GWAS per cancer ──────
  cat(sprintf("\nStep 2: Running coloc for %d proteins x %d cancers...\n",
              length(pqtl_leads), length(cancers_available)))
  local_coloc_results <- list()

  for (cancer in cancers_available) {
    gwas_file <- GWAS_FILES[[cancer]]
    gwas_n    <- CANCER_OUTCOMES[[cancer]]$n
    gwas_ncas <- CANCER_OUTCOMES[[cancer]]$n_cases
    cat(sprintf("\n=== Cancer: %s (n=%d, n_cases=%d) ===\n", cancer, gwas_n, gwas_ncas))

    # Read full GWAS file ONCE per cancer then extract windows in-memory
    if (cancer == "OVC") {
      gwas_cache <- NULL   # OVC Phelan archive is chr-split; read per-protein below
    } else {
      cat(sprintf("  Loading GWAS: %s\n", basename(gwas_file)))
      gwas_raw <- tryCatch(
        read_tsv(gwas_file, col_types = cols(.default = "c"), show_col_types = FALSE),
        error = function(e) { cat("  ERROR loading GWAS file\n"); NULL }
      )
      if (is.null(gwas_raw)) next
      cn <- names(gwas_raw)
      if ("hm_pos" %in% cn) {
        gwas_cache <- gwas_raw %>% mutate(
          chrom            = as.character(hm_chrom),
          pos              = as.integer(hm_pos),
          hm_rsid          = if ("hm_rsid" %in% cn) hm_rsid else NA_character_,
          hm_effect_allele = hm_effect_allele,
          beta = as.numeric(hm_beta), se = as.numeric(hm_se),
          p    = as.numeric(p_value),
          eaf  = as.numeric(if ("hm_effect_allele_frequency" %in% cn)
                              hm_effect_allele_frequency else NA_real_)
        ) %>% filter(!is.na(beta), !is.na(se), se > 0)
      } else {
        gwas_cache <- gwas_raw %>% mutate(
          chrom            = as.character(chromosome),
          pos              = as.integer(base_pair_location),
          hm_rsid          = if ("rsid" %in% cn) rsid else
                             if ("variant_id" %in% cn) variant_id else NA_character_,
          hm_effect_allele = effect_allele,
          beta = as.numeric(beta), se = as.numeric(standard_error),
          p    = as.numeric(p_value),
          eaf  = as.numeric(if ("effect_allele_frequency" %in% cn)
                              effect_allele_frequency else NA_real_)
        ) %>% filter(!is.na(beta), !is.na(se), se > 0)
      }
      rm(gwas_raw); gc()
      cat(sprintf("  GWAS cached: %d SNPs\n", nrow(gwas_cache)))
    }

    for (prot in names(pqtl_leads)) {
      lead <- pqtl_leads[[prot]]
      cat(sprintf("\n  [%s x %s] chr%s:%d +/-500kb\n", prot, cancer, lead$chr, lead$pos))

      if (cancer == "OVC") {
        gwas_reg <- tryCatch(
          parse_phelan_ovc(OVC_ARCHIVE_DIR, lead$chr,
                           lead$pos - lead$window, lead$pos + lead$window),
          error = function(e) { cat("  ERROR reading OVC region\n"); NULL }
        )
      } else {
        gwas_reg <- gwas_cache %>%
          filter(chrom == as.character(lead$chr),
                 pos   >= lead$pos - lead$window,
                 pos   <= lead$pos + lead$window)
      }

      if (is.null(gwas_reg) || nrow(gwas_reg) < 50) {
        cat(sprintf("  Insufficient GWAS SNPs: %s\n",
                    ifelse(is.null(gwas_reg), "NULL", nrow(gwas_reg))))
        next
      }

      merged <- inner_join(
        lead$pqtl_reg %>%
          select(chrom, pos, a1, a0, beta, se, eaf) %>%
          rename(beta.pqtl = beta, se.pqtl = se, eaf.pqtl = eaf,
                 effect_allele.pqtl = a1, other_allele.pqtl = a0),
        gwas_reg %>%
          select(chrom, pos, hm_rsid, hm_effect_allele, beta, se) %>%
          rename(beta.gwas = beta, se.gwas = se,
                 effect_allele.gwas = hm_effect_allele),
        by = c("chrom", "pos")
      ) %>%
        mutate(
          needs_flip = toupper(effect_allele.pqtl) != toupper(effect_allele.gwas),
          beta.gwas  = ifelse(needs_flip, -beta.gwas, beta.gwas)
        ) %>%
        filter(!is.na(beta.pqtl), !is.na(beta.gwas), se.pqtl > 0, se.gwas > 0)

      cat(sprintf("  Merged SNPs: %d\n", nrow(merged)))
      if (nrow(merged) < 50) { cat("  Too few overlapping SNPs\n"); next }

      snp_ids <- if ("hm_rsid" %in% names(merged) && !all(is.na(merged$hm_rsid))) {
        merged$hm_rsid
      } else {
        paste0(merged$chrom, ":", merged$pos)
      }

      coloc_out <- tryCatch(
        coloc.abf(
          dataset1 = list(
            beta    = merged$beta.pqtl,
            varbeta = merged$se.pqtl ^ 2,
            N       = 54219,
            type    = "quant",
            snp     = snp_ids,
            MAF     = merged$eaf.pqtl
          ),
          dataset2 = list(
            beta    = merged$beta.gwas,
            varbeta = merged$se.gwas ^ 2,
            N       = gwas_n,
            s       = gwas_ncas / gwas_n,
            type    = "cc",
            snp     = snp_ids
          ),
          p12 = 1e-5
        ),
        error = function(e) { cat("  coloc.abf ERROR:", conditionMessage(e), "\n"); NULL }
      )
      if (is.null(coloc_out)) next

      summ <- coloc_out$summary
      cat(sprintf("  PP.H4 = %.3f | PP.H3 = %.3f\n",
                  summ["PP.H4.abf"], summ["PP.H3.abf"]))

      local_coloc_results[[paste(prot, cancer, sep = "_")]] <- data.frame(
        protein  = prot, cancer = cancer, n_snps = nrow(merged),
        lead_chr = lead$chr, lead_pos = lead$pos,
        PP.H0 = summ["PP.H0.abf"], PP.H1 = summ["PP.H1.abf"],
        PP.H2 = summ["PP.H2.abf"], PP.H3 = summ["PP.H3.abf"],
        PP.H4 = summ["PP.H4.abf"],
        colocalises = summ["PP.H4.abf"] > 0.8,
        stringsAsFactors = FALSE
      )
    }

    if (cancer != "OVC" && exists("gwas_cache")) { rm(gwas_cache); gc() }
  }

  if (length(local_coloc_results) > 0) {
    local_coloc_df <- bind_rows(local_coloc_results)
    write_csv(local_coloc_df, "revise_plan/smr_coloc/results/phase4_coloc_local_results.csv")
    cat(sprintf("\nPhase 4 complete. %d / %d pairs tested. %d colocalise (PP.H4 > 0.80).\n",
                nrow(local_coloc_df),
                length(pqtl_leads) * length(cancers_available),
                sum(local_coloc_df$colocalises, na.rm = TRUE)))
    cat("Saved: revise_plan/smr_coloc/results/phase4_coloc_local_results.csv\n")
  } else {
    cat("\nNo coloc results produced — check parsing logs above.\n")
  }
}


# =============================================================================
# PHASE 5 — SMR + HEIDI USING SMR BINARY  [AFTER DOWNLOAD + BINARY INSTALL]
# =============================================================================
# SMR tests whether the pQTL and GWAS signals are consistent with a single
# shared causal variant (pleiotropy) vs. close linkage (two separate variants).
# HEIDI test: p > 0.05 = consistent with pleiotropy (not linkage).
#
# This phase calls the SMR binary from R using system().
#
# Prerequisites:
#   1. SMR binary downloaded and placed at:
#      revise_plan/smr_coloc/local_data/smr.exe  (Windows)
#      OR revise_plan/smr_coloc/local_data/smr    (Linux/WSL2)
#   2. LD reference extracted to:
#      revise_plan/smr_coloc/local_data/ld_ref/EUR_10  (chr-split .bed/.bim/.fam)
#   3. pQTL files converted to BESD format (see helper below)
#   4. Cancer GWAS in SMR .ma format (see helper below)
#
# Output:  results/phase5_smr_results.csv

cat("\n=== PHASE 5: SMR + HEIDI using SMR binary ===\n")

SMR_BINARY <- ifelse(.Platform$OS.type == "windows",
                     "revise_plan/smr_coloc/local_data/cancer_gwas/smr-1.3.1-win-x86_64/smr-1.3.1-win.exe",
                     "revise_plan/smr_coloc/local_data/cancer_gwas/smr-1.4.1-linux-x86_64/smr")
# LD reference: download g1000_eur.zip from https://ctg.cncr.nl/software/MAGMA/ref_data/g1000_eur.zip
# If that site is down, try: https://data.broadinstitute.org/alkesgroup/FUSION/LDREF.tar.bz2
# Extract to local_data/ld_ref/ and set prefix below accordingly
LD_REF     <- "revise_plan/smr_coloc/local_data/ld_ref/LDREF/1000G.EUR"
SMR_PQTL   <- "revise_plan/smr_coloc/local_data/pqtl_besd"
SMR_GWAS   <- "revise_plan/smr_coloc/local_data/cancer_gwas_ma"
SMR_OUT    <- "revise_plan/smr_coloc/results/smr_raw"

dir.create(SMR_PQTL,  showWarnings = FALSE, recursive = TRUE)
dir.create(SMR_GWAS,  showWarnings = FALSE, recursive = TRUE)
dir.create(SMR_OUT,   showWarnings = FALSE, recursive = TRUE)

# ── 5.1  Check binary is present ─────────────────────────────────────────────
if (!file.exists(SMR_BINARY)) {
  cat("SMR binary not found at:", SMR_BINARY, "\n")
  cat("Download from: https://yanglab.westlake.edu.cn/software/smr/#Download\n")
  cat("Phase 5 skipped.\n")
} else {
  cat("SMR binary found:", SMR_BINARY, "\n")

  # ── 5.2  Convert UKB-PPP pQTL to SMR .ma format, then BESD ───────────────
  # SMR requires pQTL in BESD format. Conversion steps:
  # Step A: Convert flat TSV to .ma format (Tab-separated: SNP A1 A2 freq b se p N)
  # Step B: Use SMR --make-besd to produce .besd/.bim/.esi files

  convert_pqtl_to_ma <- function(pqtl_file, out_ma_file, n_samples = 54219) {
    cat(sprintf("  Converting %s to .ma format...\n", basename(pqtl_file)))
    pqtl <- read_tsv(pqtl_file, col_types = cols(.default = "c"), show_col_types = FALSE)
    names(pqtl) <- toupper(names(pqtl))
    ma <- pqtl %>%
      transmute(
        SNP  = ID,
        A1   = ALLELE1,        # effect allele
        A2   = ALLELE0,        # other allele
        freq = as.numeric(A1FREQ),
        b    = as.numeric(BETA),
        se   = as.numeric(SE),
        p    = 10 ^ (-as.numeric(LOG10P)),
        N    = n_samples
      ) %>%
      filter(!is.na(b), !is.na(se), se > 0)
    write_tsv(ma, out_ma_file)
    cat(sprintf("  Saved %d SNPs to %s\n", nrow(ma), out_ma_file))
  }

  convert_gwas_to_ma <- function(gwas_tsv_gz, out_ma_file, n_total, n_cases) {
    cat(sprintf("  Converting %s to .ma format...\n", basename(gwas_tsv_gz)))
    gwas <- read_tsv(gwas_tsv_gz, col_types = cols(.default = "c"), show_col_types = FALSE)
    ma <- gwas %>%
      transmute(
        SNP  = hm_rsid,
        A1   = hm_effect_allele,
        A2   = hm_other_allele,
        freq = as.numeric(hm_effect_allele_frequency),
        b    = as.numeric(hm_beta),
        se   = as.numeric(hm_se),
        p    = as.numeric(p_value),
        N    = n_total
      ) %>%
      filter(!is.na(SNP), !is.na(b), !is.na(se), se > 0)
    write_tsv(ma, out_ma_file)
    cat(sprintf("  Saved %d SNPs to %s\n", nrow(ma), out_ma_file))
  }

  # Convert pQTL files (for proteins with local UKB-PPP files)
  pqtl_local_present <- PANEL_25[file.exists(
    file.path(PQTL_LOCAL_DIR, paste0(PANEL_25, "_pqtl.tsv.gz"))
  )]
  for (prot in pqtl_local_present) {
    in_file  <- file.path(PQTL_LOCAL_DIR, paste0(prot, "_pqtl.tsv.gz"))
    out_file <- file.path(SMR_PQTL, paste0(prot, ".ma"))
    if (!file.exists(out_file))
      convert_pqtl_to_ma(in_file, out_file)

    # Convert .ma to BESD using SMR binary
    besd_prefix <- file.path(SMR_PQTL, prot)
    if (!file.exists(paste0(besd_prefix, ".besd"))) {
      cmd <- sprintf('"%s" --qfile %s --make-besd --out %s',
                     SMR_BINARY, out_file, besd_prefix)
      cat("  Running:", cmd, "\n")
      system(cmd)
    }
  }

  # Convert cancer GWAS files
  for (cancer in names(GWAS_FILES)) {
    in_file  <- GWAS_FILES[[cancer]]
    out_file <- file.path(SMR_GWAS, paste0(cancer, ".ma"))
    if (file.exists(in_file) && !file.exists(out_file)) {
      convert_gwas_to_ma(in_file, out_file,
                         CANCER_OUTCOMES[[cancer]]$n,
                         CANCER_OUTCOMES[[cancer]]$n_cases)
    }
  }

  # ── 5.3  Run SMR + HEIDI for each protein-cancer pair ────────────────────
  smr_results_files <- character(0)

  for (prot in pqtl_local_present) {
    besd_prefix <- file.path(SMR_PQTL, prot)
    if (!file.exists(paste0(besd_prefix, ".besd"))) next

    for (cancer in names(GWAS_FILES)) {
      gwas_ma <- file.path(SMR_GWAS, paste0(cancer, ".ma"))
      if (!file.exists(gwas_ma)) next

      out_prefix <- file.path(SMR_OUT, paste0(prot, "_", cancer))

      # Skip if already run
      if (file.exists(paste0(out_prefix, ".smr"))) {
        cat(sprintf("  %s x %s: already done (skipping)\n", prot, cancer))
        smr_results_files <- c(smr_results_files, paste0(out_prefix, ".smr"))
        next
      }

      cmd <- sprintf(
        '"%s" --bfile %s --gwas-summary %s --beqtl-summary %s --out %s --heidi-mtd 1 --thread-num 4',
        SMR_BINARY, LD_REF, gwas_ma, besd_prefix, out_prefix
      )
      cat(sprintf("  Running SMR: %s x %s\n", prot, cancer))
      exit_code <- system(cmd)
      if (exit_code == 0 && file.exists(paste0(out_prefix, ".smr"))) {
        smr_results_files <- c(smr_results_files, paste0(out_prefix, ".smr"))
        cat(sprintf("  Done: %s.smr\n", out_prefix))
      } else {
        cat(sprintf("  FAILED (exit code %d)\n", exit_code))
      }
    }
  }

  # ── 5.4  Parse and combine SMR output files ───────────────────────────────
  if (length(smr_results_files) > 0) {
    smr_parsed <- lapply(smr_results_files, function(f) {
      tryCatch({
        dat <- read_tsv(f, show_col_types = FALSE)
        # Add protein and cancer labels from filename
        base_name <- tools::file_path_sans_ext(basename(f))
        parts     <- strsplit(base_name, "_")[[1]]
        dat$protein <- parts[1]
        dat$cancer  <- parts[2]
        dat
      }, error = function(e) NULL)
    })
    smr_combined <- bind_rows(Filter(Negate(is.null), smr_parsed))

    # SMR columns: probeID, ProbeChr, Gene, Probe_bp, topSNP, topSNP_Chr,
    #              topSNP_bp, A1, A2, Freq, b_GWAS, se_GWAS, p_GWAS,
    #              b_eQTL, se_eQTL, p_eQTL, b_SMR, se_SMR, p_SMR, p_HEIDI, nsnp_HEIDI
    smr_combined <- smr_combined %>%
      mutate(
        smr_significant = p_SMR   < 8.4e-6,  # standard SMR threshold
        heidi_pass      = p_HEIDI > 0.05,     # not rejected by HEIDI
        passes_both     = smr_significant & heidi_pass
      )

    write_csv(smr_combined, "revise_plan/smr_coloc/results/phase5_smr_results.csv")
    n_pass <- sum(smr_combined$passes_both, na.rm = TRUE)
    cat(sprintf("\nSMR complete: %d protein-cancer pairs pass SMR + HEIDI\n", n_pass))
    print(smr_combined %>%
            filter(passes_both) %>%
            select(protein, cancer, Gene, b_SMR, se_SMR, p_SMR, p_HEIDI))
    cat("Saved: revise_plan/smr_coloc/results/phase5_smr_results.csv\n")
  }
}


# =============================================================================
# FINAL SUMMARY TABLE — combine all phases
# =============================================================================
cat("\n=== FINAL SUMMARY ===\n")

# Load whichever results exist
summary_parts <- list()

if (file.exists("revise_plan/smr_coloc/results/phase2_mr_ivw_fdr.csv")) {
  mr_sum <- read_csv("revise_plan/smr_coloc/results/phase2_mr_ivw_fdr.csv",
                     show_col_types = FALSE) %>%
    select(protein, cancer, mr_OR = OR, mr_p = pval, mr_fdr = fdr)
  summary_parts[["mr"]] <- mr_sum
}

if (file.exists("revise_plan/smr_coloc/results/phase3_coloc_results.csv")) {
  coloc_api <- read_csv("revise_plan/smr_coloc/results/phase3_coloc_results.csv",
                        show_col_types = FALSE) %>%
    select(protein, cancer, PP.H4_api = PP.H4, colocalises_api = colocalises)
  summary_parts[["coloc_api"]] <- coloc_api
}

if (file.exists("revise_plan/smr_coloc/results/phase4_coloc_local_results.csv")) {
  coloc_loc <- read_csv("revise_plan/smr_coloc/results/phase4_coloc_local_results.csv",
                        show_col_types = FALSE) %>%
    select(protein, cancer, PP.H4_local = PP.H4, colocalises_local = colocalises)
  summary_parts[["coloc_local"]] <- coloc_loc
}

if (file.exists("revise_plan/smr_coloc/results/phase5_smr_results.csv")) {
  smr_sum <- read_csv("revise_plan/smr_coloc/results/phase5_smr_results.csv",
                      show_col_types = FALSE) %>%
    select(protein, cancer, b_SMR, p_SMR, p_HEIDI, passes_both)
  summary_parts[["smr"]] <- smr_sum
}

if (length(summary_parts) > 0) {
  final_table <- Reduce(
    function(a, b) full_join(a, b, by = c("protein", "cancer")),
    summary_parts
  )
  write_csv(final_table, "revise_plan/smr_coloc/results/FINAL_combined_summary.csv")
  cat("\nFinal combined summary saved: revise_plan/smr_coloc/results/FINAL_combined_summary.csv\n")
  cat("Columns:", paste(names(final_table), collapse = " | "), "\n\n")
  print(final_table)
} else {
  cat("No results yet. Run Phase 1 first.\n")
}

cat("\n=== SCRIPT COMPLETE ===\n")
cat("Check revise_plan/smr_coloc/results/ for all output CSV files.\n")
cat("Check revise_plan/smr_coloc/figures/ for all plots.\n")
