# make_coloc_figures.R
# Generates publication-quality coloc figures for Phase 4 results
# Run from e:/Proteomics/  →  Rscript revise_plan/smr_coloc/make_coloc_figures.R

setwd("e:/Proteomics")
LOCAL_R_LIB <- normalizePath("revise_plan/smr_coloc/r_libs", winslash = "/", mustWork = FALSE)
.libPaths(c(LOCAL_R_LIB, .libPaths()))

for (pkg in c("ggplot2","dplyr","readr","tidyr","scales","ggrepel","patchwork","RColorBrewer")) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr); library(tidyr)
  library(scales);  library(ggrepel); library(patchwork); library(RColorBrewer)
})

FIG_DIR    <- "revise_plan/smr_coloc/figures"
RESULTS_CSV <- "revise_plan/smr_coloc/results/phase4_coloc_local_results.csv"
CRC_GWAS   <- "revise_plan/smr_coloc/local_data/cancer_gwas/GCST90255675.h.tsv.gz"
BMP4_RDS   <- "revise_plan/smr_coloc/results/pqtl_regions/BMP4.rds"
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

cat("Loading results...\n")
res <- read_csv(RESULTS_CSV, show_col_types = FALSE)

cancer_labels <- c(BRC = "Breast", CRC = "Colorectal", LUNGC = "Lung", PRC = "Prostate")
res <- res %>%
  mutate(cancer_label = cancer_labels[cancer],
         cancer_label = factor(cancer_label, levels = c("Breast","Colorectal","Lung","Prostate")))

# ── Protein order: by max PP.H4 descending ─────────────────────────────────
prot_order <- res %>%
  group_by(protein) %>%
  summarise(max_h4 = max(PP.H4, na.rm = TRUE)) %>%
  arrange(max_h4) %>%
  pull(protein)
res <- res %>% mutate(protein = factor(protein, levels = prot_order))

# ===========================================================================
# FIGURE 1 — PP.H4 heatmap across all protein × cancer pairs
# ===========================================================================
cat("Figure 1: PP.H4 heatmap...\n")

fig1 <- ggplot(res, aes(x = cancer_label, y = protein, fill = PP.H4)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(data = res %>% filter(PP.H4 >= 0.1),
            aes(label = sprintf("%.2f", PP.H4)), size = 2.8,
            color = "white", fontface = "bold") +
  geom_text(data = res %>% filter(PP.H4 < 0.1 & PP.H4 >= 0.05),
            aes(label = sprintf("%.2f", PP.H4)), size = 2.5, color = "grey20") +
  scale_fill_gradientn(
    colors = c("#f7f7f7","#fee8c8","#fdd49e","#fc8d59","#e34a33","#b30000"),
    values = c(0, 0.05, 0.10, 0.20, 0.50, 1.0),
    name   = "PP.H4\n(Coloc)",
    limits = c(0, 1),
    breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0)
  ) +
  labs(
    title    = "Bayesian Colocalization: Plasma Proteins × Cancer GWAS",
    subtitle = "Posterior probability of shared causal variant (PP.H4)",
    x        = "Cancer type",
    y        = "Plasma protein (UKB-PPP pQTL)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid     = element_blank(),
    axis.text.y    = element_text(size = 9, face = "italic"),
    axis.text.x    = element_text(size = 10, angle = 0),
    plot.title     = element_text(size = 12, face = "bold"),
    plot.subtitle  = element_text(size = 9, color = "grey40"),
    legend.position = "right",
    legend.key.height = unit(1.5, "cm")
  )

ggsave(file.path(FIG_DIR, "fig1_coloc_heatmap.pdf"), fig1,
       width = 7, height = 8, device = cairo_pdf)
ggsave(file.path(FIG_DIR, "fig1_coloc_heatmap.png"), fig1,
       width = 7, height = 8, dpi = 300)
cat("  Saved fig1_coloc_heatmap\n")

# ===========================================================================
# FIGURE 2 — Posterior probability profile: BMP4 × CRC
# ===========================================================================
cat("Figure 2: BMP4 x CRC posterior probabilities...\n")

bmp4_crc <- res %>% filter(protein == "BMP4", cancer == "CRC")

pp_data <- tibble(
  Hypothesis = factor(
    c("H0\n(no signal)", "H1\n(pQTL only)", "H2\n(GWAS only)",
      "H3\n(distinct variants)", "H4\n(shared variant)"),
    levels = c("H0\n(no signal)", "H1\n(pQTL only)", "H2\n(GWAS only)",
               "H3\n(distinct variants)", "H4\n(shared variant)")
  ),
  PP = c(bmp4_crc$PP.H0, bmp4_crc$PP.H1, bmp4_crc$PP.H2,
         bmp4_crc$PP.H3, bmp4_crc$PP.H4),
  highlight = c(FALSE, FALSE, FALSE, FALSE, TRUE)
)

fig2 <- ggplot(pp_data, aes(x = Hypothesis, y = PP, fill = highlight)) +
  geom_col(width = 0.65, color = "white", linewidth = 0.5) +
  geom_text(aes(label = ifelse(PP > 0.005, sprintf("%.3f", PP), "")),
            vjust = -0.4, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("grey75", "#b30000"), guide = "none") +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1.08),
                     breaks = c(0, 0.25, 0.5, 0.75, 1.0)) +
  labs(
    title    = "BMP4 × Colorectal Cancer — Colocalization Posterior Probabilities",
    subtitle = sprintf("PP.H4 = %.3f | Lead pQTL: chr11:%s | %d overlapping SNPs",
                       bmp4_crc$PP.H4, format(bmp4_crc$lead_pos, big.mark=","), bmp4_crc$n_snps),
    x        = "Colocalization hypothesis",
    y        = "Posterior probability"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    plot.title    = element_text(size = 12, face = "bold"),
    plot.subtitle = element_text(size = 9, color = "grey40"),
    axis.text.x   = element_text(size = 9)
  )

ggsave(file.path(FIG_DIR, "fig2_BMP4_CRC_posterior.pdf"), fig2,
       width = 7, height = 5, device = cairo_pdf)
ggsave(file.path(FIG_DIR, "fig2_BMP4_CRC_posterior.png"), fig2,
       width = 7, height = 5, dpi = 300)
cat("  Saved fig2_BMP4_CRC_posterior\n")

# ===========================================================================
# FIGURE 3 — BMP4 × CRC regional locus plot
# ===========================================================================
cat("Figure 3: BMP4 x CRC locus plot...\n")

LEAD_CHR <- "11"
LEAD_POS <- 74721142
WINDOW   <- 500000
POS_MIN  <- LEAD_POS - WINDOW
POS_MAX  <- LEAD_POS + WINDOW

# Load pQTL region
cat("  Loading BMP4 pQTL region...\n")
pqtl_reg <- readRDS(BMP4_RDS)
pqtl_loc <- pqtl_reg %>%
  filter(chrom == LEAD_CHR, pos >= POS_MIN, pos <= POS_MAX) %>%
  mutate(neglog10p = -log10(p), source = "pQTL (BMP4)")
cat(sprintf("  pQTL SNPs in region: %d\n", nrow(pqtl_loc)))

# Load CRC GWAS region
cat("  Loading CRC GWAS region (chr11)...\n")
gwas_raw <- read_tsv(CRC_GWAS, col_types = cols(.default = "c"), show_col_types = FALSE)
cn <- names(gwas_raw)

# CRC GWAS uses direct columns: chromosome, base_pair_location, beta, standard_error, p_value
if ("base_pair_location" %in% cn) {
  gwas_loc <- gwas_raw %>%
    mutate(
      chrom     = as.character(chromosome),
      pos       = as.integer(base_pair_location),
      beta      = as.numeric(beta),
      se        = as.numeric(standard_error),
      p_raw     = as.numeric(p_value)
    ) %>%
    filter(chrom == LEAD_CHR, pos >= POS_MIN, pos <= POS_MAX,
           !is.na(beta), !is.na(se), se > 0) %>%
    mutate(
      neglog10p  = -log10(pmax(p_raw, 1e-300)),
      source     = "GWAS (CRC)"
    )
} else if ("hm_pos" %in% cn) {
  gwas_loc <- gwas_raw %>%
    mutate(
      chrom = as.character(hm_chrom),
      pos   = as.integer(hm_pos),
      beta  = as.numeric(hm_beta),
      se    = as.numeric(standard_error)
    ) %>%
    filter(chrom == LEAD_CHR, pos >= POS_MIN, pos <= POS_MAX,
           !is.na(beta), !is.na(se), se > 0) %>%
    mutate(
      z          = beta / se,
      p_raw      = 2 * pnorm(-abs(z)),
      neglog10p  = -log10(pmax(p_raw, 1e-300)),
      source     = "GWAS (CRC)"
    )
} else {
  stop(paste("Unknown CRC GWAS column format. Columns:", paste(cn, collapse=", ")))
}
cat(sprintf("  GWAS SNPs in region: %d\n", nrow(gwas_loc)))

# Find lead SNP in pQTL (highest -log10p)
lead_pqtl <- pqtl_loc %>% slice_max(neglog10p, n = 1)
cat(sprintf("  pQTL lead: chr%s:%d  -log10p=%.1f\n",
            lead_pqtl$chrom, lead_pqtl$pos, lead_pqtl$neglog10p))

# Find lead SNP in GWAS
lead_gwas <- gwas_loc %>% slice_max(neglog10p, n = 1)
cat(sprintf("  GWAS lead: chr%s:%d  -log10p=%.1f\n",
            lead_gwas$chrom, lead_gwas$pos, lead_gwas$neglog10p))

# LD proxy: distance-based colour (approximation without actual LD data)
# Color by distance from pQTL lead (as LD proxy)
pqtl_plot <- pqtl_loc %>%
  mutate(dist_kb = abs(pos - lead_pqtl$pos) / 1000,
         ld_proxy = cut(dist_kb,
                        breaks = c(0, 50, 100, 200, 350, 500),
                        labels = c(">0.8","0.6–0.8","0.4–0.6","0.2–0.4","<0.2"),
                        include.lowest = TRUE))

gwas_plot <- gwas_loc %>%
  mutate(dist_kb = abs(pos - lead_pqtl$pos) / 1000,
         ld_proxy = cut(dist_kb,
                        breaks = c(0, 50, 100, 200, 350, 500),
                        labels = c(">0.8","0.6–0.8","0.4–0.6","0.2–0.4","<0.2"),
                        include.lowest = TRUE))

ld_colors <- c(">0.8"="#d7191c","0.6–0.8"="#fdae61","0.4–0.6"="#a6d96a",
               "0.2–0.4"="#74add1","<0.2"="#313695")

xlab <- sprintf("Chromosome 11 position (Mb) [±500 kb around %s Mb]",
                format(round(LEAD_POS/1e6, 3), nsmall = 3))

p_pqtl <- ggplot(pqtl_plot, aes(x = pos/1e6, y = neglog10p, color = ld_proxy)) +
  geom_point(size = 1.2, alpha = 0.7) +
  geom_point(data = lead_pqtl %>% mutate(pos = pos),
             aes(x = pos/1e6, y = neglog10p),
             color = "purple", size = 3.5, shape = 18, inherit.aes = FALSE) +
  scale_color_manual(values = ld_colors, name = "r² (approx)",
                     drop = FALSE, guide = guide_legend(override.aes = list(size=2.5))) +
  scale_x_continuous(labels = function(x) sprintf("%.2f", x)) +
  labs(y = expression(-log[10](italic(p))),
       title = "pQTL: BMP4 plasma levels (UKB-PPP)") +
  theme_bw(base_size = 10) +
  theme(plot.title = element_text(size = 10, face = "bold"),
        legend.position = "none",
        axis.title.x = element_blank(),
        panel.grid.minor = element_blank())

p_gwas <- ggplot(gwas_plot, aes(x = pos/1e6, y = neglog10p, color = ld_proxy)) +
  geom_point(size = 1.2, alpha = 0.7) +
  geom_point(data = lead_gwas %>% mutate(pos = pos),
             aes(x = pos/1e6, y = neglog10p),
             color = "purple", size = 3.5, shape = 18, inherit.aes = FALSE) +
  scale_color_manual(values = ld_colors, name = "r² (approx)",
                     drop = FALSE, guide = guide_legend(override.aes = list(size=2.5))) +
  scale_x_continuous(labels = function(x) sprintf("%.2f", x)) +
  labs(x = xlab, y = expression(-log[10](italic(p))),
       title = "GWAS: Colorectal Cancer (GCST90255675, n=185,616)") +
  theme_bw(base_size = 10) +
  theme(plot.title = element_text(size = 10, face = "bold"),
        legend.position = "right",
        panel.grid.minor = element_blank())

fig3 <- p_pqtl / p_gwas +
  plot_annotation(
    title    = "BMP4 × Colorectal Cancer: Regional Colocalization (PP.H4 = 0.994)",
    subtitle = sprintf("chr11:%s–%s (1 Mb window)",
                       format(POS_MIN, big.mark=","), format(POS_MAX, big.mark=",")),
    theme    = theme(plot.title = element_text(size = 12, face = "bold"),
                     plot.subtitle = element_text(size = 9, color = "grey40"))
  )

ggsave(file.path(FIG_DIR, "fig3_BMP4_CRC_locus.pdf"), fig3,
       width = 8, height = 7, device = cairo_pdf)
ggsave(file.path(FIG_DIR, "fig3_BMP4_CRC_locus.png"), fig3,
       width = 8, height = 7, dpi = 300)
cat("  Saved fig3_BMP4_CRC_locus\n")

# ===========================================================================
# FIGURE 4 — PP.H3 vs PP.H4 scatter (all pairs), highlighting key proteins
# ===========================================================================
cat("Figure 4: PP.H3 vs PP.H4 scatter...\n")

highlight_prots <- c("BMP4","FLT3","CCDC80","PRDX6","CEACAM5","ADAMTS13")
res_annot <- res %>%
  mutate(label = ifelse(protein %in% highlight_prots | PP.H4 > 0.05,
                        paste0(as.character(protein), "\n(", cancer, ")"), NA))

fig4 <- ggplot(res, aes(x = PP.H3, y = PP.H4, color = cancer_label)) +
  geom_point(aes(size = n_snps), alpha = 0.6) +
  geom_hline(yintercept = 0.80, linetype = "dashed", color = "red", linewidth = 0.6) +
  geom_label_repel(data = res %>% filter(PP.H4 > 0.05 | (protein == "BMP4")),
                   aes(label = paste0(as.character(protein), " × ", cancer)),
                   size = 2.8, max.overlaps = 20, min.segment.length = 0,
                   box.padding = 0.4, fontface = "italic") +
  scale_color_brewer(palette = "Set2", name = "Cancer") +
  scale_size_continuous(name = "Overlapping\nSNPs", range = c(1.5, 5),
                        breaks = c(2000, 4000, 6000)) +
  scale_x_continuous(labels = percent_format(), limits = c(0, 1.02)) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1.05)) +
  annotate("text", x = 0.98, y = 0.84, label = "PP.H4 = 0.80 threshold",
           hjust = 1, size = 3, color = "red") +
  labs(
    title    = "Colocalization Evidence: Shared vs. Distinct Causal Variants",
    subtitle = "All 92 protein × cancer pairs. PP.H3 = distinct variants; PP.H4 = shared variant",
    x        = "PP.H3 (distinct causal variants)",
    y        = "PP.H4 (shared causal variant)"
  ) +
  theme_bw(base_size = 11) +
  theme(plot.title    = element_text(size = 12, face = "bold"),
        plot.subtitle = element_text(size = 9, color = "grey40"),
        panel.grid.minor = element_blank(),
        legend.position = "right")

ggsave(file.path(FIG_DIR, "fig4_H3vsH4_scatter.pdf"), fig4,
       width = 8, height = 6.5, device = cairo_pdf)
ggsave(file.path(FIG_DIR, "fig4_H3vsH4_scatter.png"), fig4,
       width = 8, height = 6.5, dpi = 300)
cat("  Saved fig4_H3vsH4_scatter\n")

# ===========================================================================
# FIGURE 5 — Top 10 pairs by PP.H4 (ranked bar)
# ===========================================================================
cat("Figure 5: Top pairs by PP.H4...\n")

top_pairs <- res %>%
  arrange(desc(PP.H4)) %>%
  head(12) %>%
  mutate(pair = paste0(as.character(protein), " × ", cancer_label),
         pair = factor(pair, levels = rev(pair)))

fig5 <- ggplot(top_pairs, aes(x = PP.H4, y = pair,
                               fill = PP.H4 > 0.80)) +
  geom_col(width = 0.7) +
  geom_vline(xintercept = 0.80, linetype = "dashed", color = "red", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.3f", PP.H4), x = PP.H4 + 0.01),
            hjust = 0, size = 3.2) +
  scale_fill_manual(values = c("grey60", "#b30000"), guide = "none") +
  scale_x_continuous(labels = percent_format(), limits = c(0, 1.12)) +
  labs(
    title    = "Top 12 Protein × Cancer Pairs by Colocalization Probability",
    subtitle = "Red dashed line = PP.H4 = 0.80 (colocalization threshold)",
    x        = "PP.H4 (posterior probability of shared causal variant)",
    y        = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    axis.text.y  = element_text(size = 9.5, face = "italic"),
    plot.title   = element_text(size = 12, face = "bold"),
    plot.subtitle = element_text(size = 9, color = "grey40")
  )

ggsave(file.path(FIG_DIR, "fig5_top_pairs_ranked.pdf"), fig5,
       width = 8, height = 5.5, device = cairo_pdf)
ggsave(file.path(FIG_DIR, "fig5_top_pairs_ranked.png"), fig5,
       width = 8, height = 5.5, dpi = 300)
cat("  Saved fig5_top_pairs_ranked\n")

cat("\n=== All 5 figures saved to", FIG_DIR, "===\n")
cat("Files:\n")
list.files(FIG_DIR, pattern = "\\.(pdf|png)$", full.names = FALSE) |>
  sort() |> cat(sep = "\n")
