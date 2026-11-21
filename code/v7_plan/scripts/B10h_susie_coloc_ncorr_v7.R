# v7 B10h: sample-size-corrected copy of B10d_susie_coloc_v7.R, written by B10h_make_ncorrected_susie_v7.py (original unchanged).
# Changes: BRC GWAS N 228,951 (122,977 cases); pQTL N per protein from the UKB-PPP files; output to results/B10_genetics_ncorrected/.
# v7 B10d: multi-signal colocalisation sensitivity (coloc.susie) for every pair whose Wald test passes the joint
# Bonferroni threshold (author request after audit SOL-F05; same policy for all such loci, not MSMB alone):
# BMP4-CRC, LEP-BRC (hits), MSMB-PRC, CD74-LYMPH, IL17A-CRC (Wald-significant, PP.H3 > 0.98).
#
# Region: the same +/-500 kb pQTL region used by the main screen; SNPs shared with the GWAS (GRCh38 position),
# harmonised on both alleles (palindromic SNPs dropped), then matched to the 1000 Genomes European LD reference
# ($SMR_DATA_ROOT/ld_ref/LDREF, 489 individuals, GRCh37, HapMap3 SNPs) through the GRCh37 position in the UKB-PPP
# variant ID and both alleles. Effects aligned to the reference A1 allele; LD = Pearson correlation of A1 dosages.
# coloc::runsusie on each trait (pQTL N 54,219; GWAS N and case fraction as in the main screen), coloc.susie
# between all credible-set pairs. Reported: SNPs used, credible sets per trait, max PP.H4 over credible-set pairs,
# and the single-signal coloc.abf on the same reduced SNP set for comparison.
# Diagnostics: SNPs shared before/after LD matching, whether each trait's top shared SNP survives the LD match,
# and susieR::estimate_s_rss (LD mismatch between summary statistics and reference; values near 0 are good).
# Output: v7_plan/results/B10_genetics_final/B10d_susie_coloc.csv
.libPaths(c(file.path(Sys.getenv("PROTEOMICS_ROOT", "."), ".rlib"), .libPaths()))
suppressPackageStartupMessages({ library(dplyr); library(readr); library(coloc); library(susieR) })

ROOT <- Sys.getenv("PROTEOMICS_ROOT", "."); G <- file.path(Sys.getenv("SMR_DATA_ROOT", "smr_data"), "cancer_gwas"); LD <- file.path(Sys.getenv("SMR_DATA_ROOT", "smr_data"), "ld_ref/LDREF")
OUTD <- file.path(ROOT, "v7_plan/results/B10_genetics_ncorrected")
ll0 <- readRDS(file.path(ROOT, "revise_plan/smr_coloc/results/pqtl_leads_checkpoint.rds"))
ll1 <- readRDS(file.path(ROOT, "v7_plan/results/B10_genetics_v2/parta_pqtl_leads_checkpoint.rds"))
PAIRS <- list(c("BMP4", "CRC"), c("LEP", "BRC"), c("MSMB", "PRC"), c("CD74", "LYMPH"), c("IL17A", "CRC"))
GW <- list(CRC = list(f = "GCST90255675.h.tsv.gz", p = "raw", n = 185616, nc = 78473),
           BRC = list(f = "29059683-GCST004988-EFO_0000305.h.tsv.gz", p = "hm", n = 228951, nc = 122977),
           PRC = list(f = "29892016-GCST006085-EFO_0001663.h.tsv.gz", p = "hm", n = 140254, nc = 79148),
           LYMPH = list(f = "LYMPH_GCST90624739.h.tsv.gz", p = "raw", n = 658535, nc = 2280))
PQTL_NS <- c(LEP = 33693, BMP4 = 33372, MSMB = 33529, CD74 = 33962, IL17A = 33084)  # UKB-PPP file column N

comp <- function(a) chartr("ACGT", "TGCA", a)
pal <- function(a, b) a == comp(b)

load_gwas <- function(cfg, chr, lo, hi) {
  f <- file.path(G, cfg$f)
  if (cfg$p == "hm") {
    d <- read_tsv(f, col_select = c("hm_chrom", "hm_pos", "hm_effect_allele", "hm_other_allele", "hm_beta",
                                    "standard_error", "hm_ci_lower", "hm_ci_upper"), col_types = cols(.default = "c"), show_col_types = FALSE)
    d %>% filter(hm_chrom == chr) %>% transmute(pos = as.integer(hm_pos), g_ea = toupper(hm_effect_allele), g_oa = toupper(hm_other_allele),
      g_b = as.numeric(hm_beta), g_se = ifelse(is.na(suppressWarnings(as.numeric(standard_error))),
        (as.numeric(hm_ci_upper) - as.numeric(hm_ci_lower)) / 3.92, suppressWarnings(as.numeric(standard_error)))) %>%
      filter(pos >= lo, pos <= hi, !is.na(g_b), g_se > 0)
  } else {
    d <- read_tsv(f, col_select = any_of(c("chromosome", "base_pair_location", "effect_allele", "other_allele", "beta", "standard_error", "hm_code")),
                  col_types = cols(.default = "c"), show_col_types = FALSE)
    d %>% filter(chromosome == chr, is.na(hm_code) | hm_code %in% c("5", "10", "11")) %>%
      transmute(pos = as.integer(base_pair_location), g_ea = toupper(effect_allele), g_oa = toupper(other_allele),
                g_b = as.numeric(beta), g_se = as.numeric(standard_error)) %>% filter(pos >= lo, pos <= hi, !is.na(g_b), g_se > 0)
  }
}

read_bed <- function(chr, idx, n) {  # PLINK SNP-major; returns A1 dosage matrix (n x length(idx))
  bpr <- ceiling(n / 4); con <- file(file.path(LD, sprintf("1000G.EUR.%s.bed", chr)), "rb"); on.exit(close(con))
  out <- matrix(NA_real_, n, length(idx))
  map <- c(2, NA, 1, 0)  # 00 hom A1, 01 missing, 10 het, 11 hom A2
  for (j in seq_along(idx)) {
    seek(con, 3 + (idx[j] - 1) * bpr)
    by <- as.integer(readBin(con, "raw", bpr))
    codes <- as.vector(rbind(by %% 4, (by %/% 4) %% 4, (by %/% 16) %% 4, (by %/% 64) %% 4))[1:n]
    out[, j] <- map[codes + 1]
  }
  out
}

res <- list()
for (pr in PAIRS) {
  prot <- pr[1]; cn <- pr[2]; cfg <- GW[[cn]]; PQTL_N <- PQTL_NS[[prot]]
  l <- if (!is.null(ll0[[prot]])) ll0[[prot]] else ll1[[prot]]
  rf <- l$region_file; if (!grepl("^[A-Za-z]:|^/", rf)) rf <- file.path(ROOT, rf)
  q <- readRDS(rf); if (!"rsid" %in% names(q)) q$rsid <- q$ID
  q <- q %>% mutate(pos37 = as.integer(sapply(strsplit(rsid, ":"), `[`, 2)), a1 = toupper(a1), a0 = toupper(a0))
  chr <- as.character(l$chr)
  g <- load_gwas(cfg, chr, l$pos - l$window, l$pos + l$window)
  m <- inner_join(q, g, by = "pos", relationship = "many-to-many") %>%
    filter(!pal(a1, a0)) %>%
    mutate(gflip = case_when(g_ea == a1 & g_oa == a0 ~ 1, g_ea == a0 & g_oa == a1 ~ -1,
                             g_ea == comp(a1) & g_oa == comp(a0) ~ 1, g_ea == comp(a0) & g_oa == comp(a1) ~ -1, TRUE ~ NA_real_)) %>%
    filter(!is.na(gflip)) %>% mutate(g_b = g_b * gflip) %>% distinct(pos, .keep_all = TRUE)
  top_q <- m$pos[which.max(abs(m$beta / m$se))]; top_g <- m$pos[which.max(abs(m$g_b / m$g_se))]; n_shared <- nrow(m)
  bim <- read_tsv(file.path(LD, sprintf("1000G.EUR.%s.bim", chr)), col_names = c("chr", "snp", "cm", "bp", "A1", "A2"),
                  col_types = "ccdicc", show_col_types = FALSE) %>% mutate(idx = row_number())
  fam_n <- length(readLines(file.path(LD, sprintf("1000G.EUR.%s.fam", chr))))
  m <- inner_join(m, bim, by = c("pos37" = "bp")) %>%
    mutate(rflip = case_when(A1 == a1 & A2 == a0 ~ 1, A1 == a0 & A2 == a1 ~ -1,
                             A1 == comp(a1) & A2 == comp(a0) ~ 1, A1 == comp(a0) & A2 == comp(a1) ~ -1, TRUE ~ NA_real_)) %>%
    filter(!is.na(rflip)) %>% distinct(pos, .keep_all = TRUE) %>% arrange(pos)
  X <- read_bed(chr, m$idx, fam_n)
  keep <- apply(X, 2, function(v) sd(v, na.rm = TRUE) > 0)
  m <- m[keep, ]; X <- X[, keep, drop = FALSE]
  R <- suppressWarnings(cor(X, use = "pairwise.complete.obs")); R[is.na(R)] <- 0
  snp <- paste0("chr", chr, ":", m$pos)
  dimnames(R) <- list(snp, snp)
  b1 <- m$beta * m$rflip; b2 <- m$g_b * m$rflip
  D1 <- list(beta = setNames(b1, snp), varbeta = setNames(m$se^2, snp), N = PQTL_N, type = "quant", sdY = 1, snp = snp, LD = R, position = m$pos)
  D2 <- list(beta = setNames(b2, snp), varbeta = setNames(m$g_se^2, snp), N = cfg$n, s = cfg$nc / cfg$n, type = "cc", snp = snp, LD = R, position = m$pos)
  abf <- coloc.abf(D1[c("beta", "varbeta", "N", "type", "sdY", "snp")], D2[c("beta", "varbeta", "N", "s", "type", "snp")], p12 = 1e-5)$summary
  S1 <- tryCatch(runsusie(D1, repeat_until_convergence = FALSE, maxit = 1000), error = function(e) NULL)
  S2 <- tryCatch(runsusie(D2, repeat_until_convergence = FALSE, maxit = 1000), error = function(e) NULL)
  s_q <- tryCatch(estimate_s_rss(b1 / m$se, R, n = PQTL_N), error = function(e) NA)
  s_g <- tryCatch(estimate_s_rss(b2 / m$g_se, R, n = cfg$n), error = function(e) NA)
  ncs <- function(S) if (is.null(S) || is.null(S$sets$cs)) 0L else length(S$sets$cs)
  out <- tibble(protein = prot, cancer = cn, n_snps_shared = n_shared, n_snps_ld = nrow(m), top_pqtl_shared = top_q, top_pqtl_in_ld = top_q %in% m$pos,
                top_gwas_shared = top_g, top_gwas_in_ld = top_g %in% m$pos, ld_mismatch_s_pqtl = s_q, ld_mismatch_s_gwas = s_g, abf_PP.H3 = abf["PP.H3.abf"], abf_PP.H4 = abf["PP.H4.abf"],
                cs_pqtl = ncs(S1), cs_gwas = ncs(S2), susie_max_PP.H4 = NA_real_, susie_max_PP.H3 = NA_real_, note = "")
  if (ncs(S1) > 0 && ncs(S2) > 0) {
    cs <- coloc.susie(S1, S2, p12 = 1e-5)$summary
    out$susie_max_PP.H4 <- max(cs$PP.H4.abf); out$susie_max_PP.H3 <- cs$PP.H3.abf[which.max(cs$PP.H4.abf)]
    out$note <- paste0("lead pairs (pQTL/GWAS): ", paste(cs$hit1, cs$hit2, sprintf("H4=%.3f", cs$PP.H4.abf), collapse = "; "))
  } else out$note <- "no credible set in at least one trait: coloc.susie not applicable"
  cat(sprintf("%s-%s: %d SNPs | abf H4 %.3g H3 %.3g | CS pQTL %d GWAS %d | susie max H4 %s\n", prot, cn, nrow(m),
              abf["PP.H4.abf"], abf["PP.H3.abf"], ncs(S1), ncs(S2), format(out$susie_max_PP.H4, digits = 3)))
  res[[length(res) + 1]] <- out
}
r <- bind_rows(res)
write_csv(r, file.path(OUTD, "B10h_susie_coloc_ncorr.csv"))
print(as.data.frame(r))
cat("B10D_COMPLETE\n")
