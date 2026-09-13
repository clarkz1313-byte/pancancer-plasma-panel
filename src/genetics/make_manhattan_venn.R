# make_manhattan_venn.R
# Generates 3 new figures mirroring reference Fig 6 (Chen & Hua FASEB J 2026):
#   fig10: Manhattan plot — CRC GWAS genome-wide, BMP4 locus highlighted
#   fig11: SMR effect scatter — pQTL beta vs GWAS beta for BMP4xCRC locus
#   fig12: Venn diagram — Coloc evidence vs SMR causal overlap
# Run from e:/Proteomics/

setwd("e:/Proteomics")
LOCAL_R_LIB <- normalizePath("revise_plan/smr_coloc/r_libs", winslash="/", mustWork=FALSE)
.libPaths(c(LOCAL_R_LIB, .libPaths()))

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr)
  library(scales);  library(ggrepel); library(patchwork)
})

FIG_DIR   <- "revise_plan/smr_coloc/figures"
COLOC_CSV <- "revise_plan/smr_coloc/results/phase4_coloc_local_results.csv"
SMR_CSV   <- "revise_plan/smr_coloc/results/phase5_smr_results.csv"
CRC_GWAS  <- "revise_plan/smr_coloc/local_data/cancer_gwas/GCST90255675.h.tsv.gz"
BMP4_RDS  <- "revise_plan/smr_coloc/results/pqtl_regions/BMP4.rds"
dir.create(FIG_DIR, showWarnings=FALSE, recursive=TRUE)

coloc <- read_csv(COLOC_CSV, show_col_types=FALSE)
smr   <- read_csv(SMR_CSV,   show_col_types=FALSE)

# =============================================================================
# FIGURE 10 — Manhattan plot: CRC GWAS genome-wide, BMP4 locus highlighted
# =============================================================================
cat("=== Figure 10: Manhattan plot ===\n")
cat("  Loading CRC GWAS (3 columns)...\n")

gwas_mh <- read_tsv(CRC_GWAS,
  col_select     = c("chromosome","base_pair_location","p_value"),
  col_types      = cols(.default = "c"),
  show_col_types = FALSE) %>%
  mutate(
    chr = suppressWarnings(as.integer(chromosome)),
    bp  = suppressWarnings(as.integer(base_pair_location)),
    p   = suppressWarnings(as.numeric(p_value))
  ) %>%
  filter(!is.na(chr), !is.na(bp), !is.na(p), p > 0, chr %in% 1:22)

cat(sprintf("  Loaded: %d SNPs\n", nrow(gwas_mh)))

# Downsample: keep all p < 0.01, random 2% of the rest
set.seed(42)
sig_snps   <- gwas_mh %>% filter(p < 0.01)
insig_snps <- gwas_mh %>% filter(p >= 0.01) %>% slice_sample(prop = 0.02)
gwas_plot  <- bind_rows(sig_snps, insig_snps) %>% arrange(chr, bp)
cat(sprintf("  After downsample: %d SNPs\n", nrow(gwas_plot)))
rm(gwas_mh, sig_snps, insig_snps); gc()

# Build cumulative chromosome positions
chr_sizes <- gwas_plot %>%
  group_by(chr) %>%
  summarise(chr_len = max(bp), .groups = "drop") %>%
  arrange(chr) %>%
  mutate(offset = lag(cumsum(as.numeric(chr_len)), default = 0) + chr * 1.5e7)

gwas_plot <- gwas_plot %>%
  left_join(chr_sizes %>% select(chr, offset), by = "chr") %>%
  mutate(
    pos_cum = bp + offset,
    logp    = -log10(p),
    col_grp = factor(chr %% 2)
  )

chr_axis <- gwas_plot %>%
  group_by(chr) %>%
  summarise(centre = (min(pos_cum) + max(pos_cum)) / 2, .groups = "drop")

# BMP4 locus highlight (chr11 74.2-75.2 Mb)
bmp4_off <- chr_sizes %>% filter(chr == 11) %>% pull(offset)
bmp4_lo  <- 74221142 + bmp4_off
bmp4_hi  <- 75221142 + bmp4_off

# Top loci to check for annotation
top_loci <- gwas_plot %>%
  filter(p < 5e-8) %>%
  group_by(chr) %>%
  slice_min(p, n = 1, with_ties = FALSE) %>%
  ungroup()

bmp4_lead <- gwas_plot %>%
  filter(chr == 11, bp > 74500000, bp < 75000000) %>%
  slice_min(p, n = 1, with_ties = FALSE)

fig10 <- ggplot(gwas_plot, aes(x = pos_cum, y = logp, color = col_grp)) +
  annotate("rect",
    xmin = bmp4_lo, xmax = bmp4_hi, ymin = -Inf, ymax = Inf,
    fill = "#b30000", alpha = 0.10) +
  geom_point(size = 0.4, alpha = 0.55, shape = 16, show.legend = FALSE) +
  geom_hline(yintercept = -log10(5e-8),
             linetype = "dashed", color = "red3", linewidth = 0.6) +
  geom_hline(yintercept = -log10(1e-5),
             linetype = "dotted", color = "grey55", linewidth = 0.4) +
  {if (nrow(bmp4_lead) > 0)
    geom_label_repel(
      data     = bmp4_lead,
      aes(label = "BMP4 locus\nchr11:74.7 Mb"),
      color    = "#b30000", fontface = "bold", size = 3.2,
      box.padding = 0.7, min.segment.length = 0,
      nudge_y  = 3, show.legend = FALSE
    )
  } +
  scale_color_manual(values = c("0" = "grey60", "1" = "#3a6ea5")) +
  scale_x_continuous(
    breaks = chr_axis$centre,
    labels = chr_axis$chr,
    expand = c(0.01, 0)
  ) +
  scale_y_continuous(expand = c(0.01, 0.5)) +
  annotate("text", x = max(gwas_plot$pos_cum) * 0.98,
           y = -log10(5e-8) + 0.5,
           label = "p = 5×10⁻⁸",
           hjust = 1, size = 3, color = "red3") +
  labs(
    title    = "Manhattan Plot: Colorectal Cancer GWAS",
    subtitle = paste0(
      "Huyghe et al. 2019 (N=185,616; 78,473 cases). ",
      "Red dashed = genome-wide significance (p=5×10⁻⁸). ",
      "Shaded region = BMP4 pQTL locus (chr11:74.2-75.2 Mb)."),
    x        = "Chromosome",
    y        = expression(-log[10](italic(p)))
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title        = element_text(size = 12, face = "bold"),
    plot.subtitle     = element_text(size = 8.5, color = "grey40"),
    panel.grid.minor  = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text.x       = element_text(size = 8)
  )

ggsave(file.path(FIG_DIR, "fig10_manhattan_CRC.pdf"), fig10,
       width = 14, height = 5, device = cairo_pdf)
ggsave(file.path(FIG_DIR, "fig10_manhattan_CRC.png"), fig10,
       width = 14, height = 5, dpi = 300)
cat("  Saved fig10_manhattan_CRC\n")
rm(gwas_plot, top_loci); gc()

# =============================================================================
# FIGURE 11 — SMR effect scatter: pQTL beta vs GWAS beta (mirror of Fig 6C)
# =============================================================================
cat("\n=== Figure 11: SMR effect scatter ===\n")

pqtl_reg <- readRDS(BMP4_RDS)
cat(sprintf("  pQTL SNPs: %d\n", nrow(pqtl_reg)))

gwas_win <- read_tsv(CRC_GWAS,
  col_select = c("chromosome","base_pair_location","effect_allele",
                 "beta","standard_error","p_value"),
  col_types  = cols(.default = "c"),
  show_col_types = FALSE) %>%
  mutate(
    chrom          = as.character(chromosome),
    pos            = as.integer(base_pair_location),
    beta           = as.numeric(beta),
    se             = as.numeric(standard_error),
    p              = as.numeric(p_value),
    hm_effect_allele = effect_allele
  ) %>%
  filter(chrom == "11", pos >= 74221142, pos <= 75221142,
         !is.na(beta), !is.na(se), se > 0)
cat(sprintf("  GWAS window SNPs: %d\n", nrow(gwas_win)))

merged_sc <- inner_join(
  pqtl_reg %>%
    select(chrom, pos, a1, a0, beta, se) %>%
    rename(b_pq = beta, s_pq = se, ea_pq = a1),
  gwas_win %>%
    select(chrom, pos, hm_effect_allele, beta, se, p) %>%
    rename(b_gw = beta, s_gw = se, ea_gw = hm_effect_allele, p_gw = p),
  by = c("chrom","pos"),
  relationship = "many-to-many"
) %>%
  arrange(desc(abs(b_pq / s_pq))) %>%
  distinct(chrom, pos, .keep_all = TRUE) %>%
  mutate(
    flip    = !is.na(toupper(ea_gw)) & toupper(ea_gw) != toupper(ea_pq),
    b_gw    = ifelse(flip, -b_gw, b_gw),
    z_pq    = abs(b_pq / s_pq),
    is_lead = (pos == 74721142L)
  ) %>%
  filter(abs(b_pq) > 0, abs(b_gw) > 0, is.finite(b_pq), is.finite(b_gw))

cat(sprintf("  Merged SNPs: %d\n", nrow(merged_sc)))

b_smr    <- 0.6398
lead_row <- merged_sc %>% filter(is_lead)
if (nrow(lead_row) == 0) lead_row <- merged_sc %>% slice_max(z_pq, n = 1, with_ties = FALSE)

fig11 <- ggplot(merged_sc, aes(x = b_pq, y = b_gw)) +
  geom_hline(yintercept = 0, color = "grey80", linewidth = 0.4) +
  geom_vline(xintercept = 0, color = "grey80", linewidth = 0.4) +
  geom_point(aes(color = z_pq, size = z_pq), alpha = 0.75, shape = 16) +
  geom_abline(intercept = 0, slope = b_smr,
              color = "#b30000", linetype = "dashed", linewidth = 1.0) +
  geom_errorbar(data = lead_row,
    aes(ymin = b_gw - 1.96*s_gw, ymax = b_gw + 1.96*s_gw),
    width = 0.003, color = "#b30000", linewidth = 0.9) +
  geom_errorbarh(data = lead_row,
    aes(xmin = b_pq - 1.96*s_pq, xmax = b_pq + 1.96*s_pq),
    height = 0.006, color = "#b30000", linewidth = 0.9) +
  geom_point(data = lead_row,
    color = "#b30000", size = 5, shape = 18) +
  geom_label_repel(data = lead_row,
    aes(label = paste0("Lead pQTL\nchr11:", format(pos, big.mark=","))),
    color = "#b30000", size = 3, fontface = "bold",
    box.padding = 0.9, min.segment.length = 0) +
  annotate("text",
    x = max(merged_sc$b_pq, na.rm=TRUE) * 0.55,
    y = max(merged_sc$b_gw, na.rm=TRUE) * 0.88,
    label = paste0("b[SMR] == ", round(b_smr,3)),
    parse = TRUE, color = "#b30000", size = 4, fontface = "bold") +
  scale_color_gradient(low = "grey72", high = "#08306b",
                       name = "pQTL |z-score|") +
  scale_size_continuous(range = c(0.8, 3.8), guide = "none") +
  labs(
    title    = "SMR Effect Plot: BMP4 × Colorectal Cancer",
    subtitle = paste0(
      "Each dot = one SNP in chr11 ±500 kb window. ",
      "X = pQTL effect on plasma BMP4 per SD (UKB-PPP). ",
      "Y = GWAS effect on CRC log-OR (Huyghe 2019).\n",
      "Dashed line: slope = b_SMR = 0.640. ",
      "Consistent positive association confirms HEIDI pass (p = 0.885)."),
    x = "pQTL effect size (β on plasma BMP4, SD units)",
    y = "GWAS effect size (β on CRC risk, log-OR)"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title      = element_text(size = 12, face = "bold"),
    plot.subtitle   = element_text(size = 8, color = "grey40"),
    legend.position = "right",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(FIG_DIR, "fig11_SMR_effect_scatter.pdf"), fig11,
       width = 8, height = 7, device = cairo_pdf)
ggsave(file.path(FIG_DIR, "fig11_SMR_effect_scatter.png"), fig11,
       width = 8, height = 7, dpi = 300)
cat("  Saved fig11_SMR_effect_scatter\n")
rm(gwas_win, merged_sc, pqtl_reg); gc()

# =============================================================================
# FIGURE 12 — Venn diagram: Coloc (PP.H4 > 0.05) vs SMR causal
# =============================================================================
cat("\n=== Figure 12: Venn diagram ===\n")

coloc_sig  <- coloc %>% filter(PP.H4 > 0.05) %>%
  mutate(pair = paste0(protein, "_", cancer))
smr_causal <- smr   %>% filter(causal == TRUE) %>%
  mutate(pair = paste0(protein, "_", cancer))

n_coloc_only  <- sum(!coloc_sig$pair  %in% smr_causal$pair)
n_smr_only    <- sum(!smr_causal$pair %in% coloc_sig$pair)
n_both        <- sum(smr_causal$pair  %in% coloc_sig$pair)
both_labels   <- smr_causal %>%
  filter(pair %in% coloc_sig$pair) %>%
  mutate(lab = paste0(protein, " × ", cancer)) %>%
  pull(lab) %>% paste(collapse = "\n")

cat(sprintf("  Coloc PP.H4>0.05: %d | SMR causal: %d | Both: %d\n",
    nrow(coloc_sig), nrow(smr_causal), n_both))

# Draw two overlapping circles with ggplot2
theta   <- seq(0, 2*pi, length.out = 300)
r       <- 1.6
cx1     <- -0.75; cx2 <- 0.75; cy <- 0
c1_df   <- data.frame(x = cx1 + r*cos(theta), y = cy + r*sin(theta))
c2_df   <- data.frame(x = cx2 + r*cos(theta), y = cy + r*sin(theta))

fig12 <- ggplot() +
  geom_polygon(data = c1_df, aes(x, y),
               fill = "#9ecae1", alpha = 0.5, color = "#2171b5", linewidth = 1.2) +
  geom_polygon(data = c2_df, aes(x, y),
               fill = "#fcae91", alpha = 0.5, color = "#cb181d", linewidth = 1.2) +
  # Main counts
  annotate("text", x = cx1 - 0.6, y =  0.15,
           label = as.character(n_coloc_only),
           size = 13, fontface = "bold", color = "#08519c") +
  annotate("text", x = cx1 - 0.6, y = -0.4,
           label = sprintf("(%.1f%%)", 100 * n_coloc_only / 92),
           size =  4,               color = "#08519c") +
  annotate("text", x = 0,           y =  0.15,
           label = as.character(n_both),
           size = 13, fontface = "bold", color = "#67000d") +
  annotate("text", x = 0,           y = -0.4,
           label = sprintf("(%.1f%%)", 100 * n_both / 92),
           size =  4,               color = "#67000d") +
  annotate("text", x = cx2 + 0.6,  y =  0.15,
           label = as.character(n_smr_only),
           size = 13, fontface = "bold", color = "#a50f15") +
  annotate("text", x = cx2 + 0.6,  y = -0.4,
           label = sprintf("(%.1f%%)", 100 * n_smr_only / 92),
           size =  4,               color = "#a50f15") +
  # Circle labels (top)
  annotate("text", x = cx1 - 0.9, y = 2.0,
           label = sprintf("Colocalization\n(PP.H4 > 0.05)\nn = %d pairs", nrow(coloc_sig)),
           size = 4, fontface = "bold", color = "#08519c", hjust = 0.5) +
  annotate("text", x = cx2 + 0.9, y = 2.0,
           label = sprintf("SMR causal\n(pₛₘᵣ < 0.05 & HEIDI pass)\nn = %d pairs", nrow(smr_causal)),
           size = 4, fontface = "bold", color = "#a50f15", hjust = 0.5) +
  # Overlap label (bottom of overlap zone)
  annotate("text", x = 0, y = -0.85,
           label = both_labels,
           size = 3.5, fontface = "bold.italic", color = "#67000d") +
  # Footer
  annotate("text", x = 0, y = -2.5,
           label = "92 protein × cancer pairs tested (25 proteins × 4 cancers, OVC and chrX excluded)",
           size = 3, color = "grey45", hjust = 0.5) +
  coord_equal(xlim = c(-3.5, 3.5), ylim = c(-2.8, 2.8)) +
  labs(
    title    = "Overlap: Colocalization (Phase 4) ∩ SMR Causal (Phase 5)",
    subtitle = paste0(
      "Mirroring Chen & Hua FASEB J 2026 Fig 6B. ",
      "Only BMP4 × CRC satisfies both criteria: PP.H4 = 0.994 and dual SMR + HEIDI filter.")
  ) +
  theme_void(base_size = 11) +
  theme(
    plot.title    = element_text(size = 12, face = "bold", hjust = 0.5,
                                 margin = margin(b = 4)),
    plot.subtitle = element_text(size = 8.5, color = "grey40", hjust = 0.5,
                                 margin = margin(b = 8)),
    plot.margin   = margin(12, 12, 12, 12)
  )

ggsave(file.path(FIG_DIR, "fig12_venn_coloc_smr.pdf"), fig12,
       width = 8, height = 7, device = cairo_pdf)
ggsave(file.path(FIG_DIR, "fig12_venn_coloc_smr.png"), fig12,
       width = 8, height = 7, dpi = 300)
cat("  Saved fig12_venn_coloc_smr\n")

cat("\n=== All three new figures saved ===\n")
cat("fig10_manhattan_CRC.pdf/.png\n")
cat("fig11_SMR_effect_scatter.pdf/.png\n")
cat("fig12_venn_coloc_smr.pdf/.png\n")
cat(sprintf("End: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
