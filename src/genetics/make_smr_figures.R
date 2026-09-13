# make_smr_figures.R — Phase 5 SMR + Phase 4 coloc linked figures
# Run after phase5_smr_results.csv exists
# Generates: forest plot, SMR-coloc linkage scatter, HEIDI instrument plot, combined panel

setwd("e:/Proteomics")
LOCAL_R_LIB <- normalizePath("revise_plan/smr_coloc/r_libs", winslash="/", mustWork=FALSE)
.libPaths(c(LOCAL_R_LIB, .libPaths()))

for (pkg in c("ggplot2","dplyr","readr","tidyr","scales","ggrepel","patchwork","RColorBrewer")) {
  if (!requireNamespace(pkg, quietly=TRUE)) install.packages(pkg)
}
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr); library(tidyr)
  library(scales);  library(ggrepel); library(patchwork); library(RColorBrewer)
})

FIG_DIR    <- "revise_plan/smr_coloc/figures"
SMR_CSV    <- "revise_plan/smr_coloc/results/phase5_smr_results.csv"
COLOC_CSV  <- "revise_plan/smr_coloc/results/phase4_coloc_local_results.csv"
dir.create(FIG_DIR, showWarnings=FALSE, recursive=TRUE)

cat("Loading results...\n")
smr   <- read_csv(SMR_CSV,   show_col_types=FALSE)
coloc <- read_csv(COLOC_CSV, show_col_types=FALSE)

cancer_labels <- c(BRC="Breast", CRC="Colorectal", LUNGC="Lung", PRC="Prostate")
smr   <- smr   %>% mutate(cancer_label = cancer_labels[cancer],
                           cancer_label = factor(cancer_label, c("Breast","Colorectal","Lung","Prostate")),
                           ci_lo = b_SMR - 1.96*se_SMR,
                           ci_hi = b_SMR + 1.96*se_SMR,
                           neg_log10_p = -log10(pmax(p_SMR, 1e-300)),
                           label = paste0(protein, " × ", cancer_label))
coloc <- coloc %>% mutate(cancer_label = cancer_labels[cancer],
                           cancer_label = factor(cancer_label, c("Breast","Colorectal","Lung","Prostate")))

# Joined table for linkage plots
linked <- inner_join(
  smr   %>% select(protein, cancer, b_SMR, se_SMR, p_SMR, p_HEIDI, causal, neg_log10_p, cancer_label, ci_lo, ci_hi),
  coloc %>% select(protein, cancer, PP.H4, PP.H3),
  by=c("protein","cancer")
)
cat(sprintf("Linked pairs: %d\n", nrow(linked)))
cat(sprintf("Causal (SMR+HEIDI): %d\n", sum(smr$causal, na.rm=TRUE)))

# ============================================================================
# FIGURE 6 — SMR forest plot: all pairs, grouped by cancer, highlight causal
# ============================================================================
cat("Figure 6: SMR forest plot...\n")

# Order proteins by b_SMR within cancer, show top-10 per cancer by |z|
top_forest <- smr %>%
  filter(!is.na(b_SMR), is.finite(b_SMR), abs(b_SMR) < 5) %>%
  group_by(cancer) %>%
  slice_max(abs(b_SMR/se_SMR), n=10, with_ties=FALSE) %>%
  ungroup() %>%
  arrange(cancer_label, b_SMR) %>%
  mutate(pair = factor(paste0(protein, "\n", cancer_label),
                       levels=paste0(protein, "\n", cancer_label)))

fig6 <- ggplot(top_forest, aes(x=b_SMR, y=pair, color=causal, shape=causal)) +
  geom_vline(xintercept=0, linetype="dashed", color="grey50", linewidth=0.5) +
  geom_errorbarh(aes(xmin=ci_lo, xmax=ci_hi), height=0.3, linewidth=0.5, alpha=0.7) +
  geom_point(size=2.5) +
  facet_wrap(~cancer_label, scales="free_y", ncol=2) +
  scale_color_manual(values=c("FALSE"="grey55","TRUE"="#b30000"),
                     labels=c("FALSE"="Not causal","TRUE"="Causal (p_SMR<0.05, HEIDI pass)"),
                     name="SMR result") +
  scale_shape_manual(values=c("FALSE"=16,"TRUE"=18), guide="none") +
  labs(
    title    = "SMR Effect Estimates: Plasma Protein → Cancer Risk",
    subtitle = "Top 10 pairs per cancer by |z_SMR|. Bars = 95% CI. Red = passes SMR + HEIDI dual filter",
    x        = "b_SMR (Wald ratio: effect on cancer log-odds per SD protein)",
    y        = NULL
  ) +
  theme_bw(base_size=10) +
  theme(
    plot.title    = element_text(size=11, face="bold"),
    plot.subtitle = element_text(size=8, color="grey40"),
    strip.text    = element_text(size=10, face="bold"),
    axis.text.y   = element_text(size=7.5, face="italic"),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(FIG_DIR,"fig6_smr_forest.pdf"), fig6, width=10, height=10, device=cairo_pdf)
ggsave(file.path(FIG_DIR,"fig6_smr_forest.png"), fig6, width=10, height=10, dpi=300)
cat("  Saved fig6_smr_forest\n")

# ============================================================================
# FIGURE 7 — SMR × Coloc linkage: PP.H4 vs -log10(p_SMR) scatter
# ============================================================================
cat("Figure 7: SMR-Coloc linkage scatter...\n")

linked_clean <- linked %>% filter(!is.na(b_SMR), is.finite(b_SMR), !is.na(PP.H4))
highlight_pairs <- linked_clean %>% filter(causal | PP.H4 > 0.10 | neg_log10_p > 3)

fig7 <- ggplot(linked_clean, aes(x=PP.H4, y=neg_log10_p, color=cancer_label)) +
  geom_hline(yintercept=-log10(0.05), linetype="dashed", color="steelblue", linewidth=0.6) +
  geom_vline(xintercept=0.80, linetype="dashed", color="red",      linewidth=0.6) +
  geom_point(aes(size=abs(b_SMR)), alpha=0.65) +
  geom_label_repel(
    data=highlight_pairs,
    aes(label=paste0(protein, "\n×", cancer)),
    size=2.5, max.overlaps=20, min.segment.length=0,
    box.padding=0.45, label.padding=0.2,
    fontface="italic", fill=alpha("white", 0.8)
  ) +
  annotate("text", x=0.02, y=-log10(0.05)+0.12, label="p_SMR = 0.05",
           hjust=0, size=3, color="steelblue") +
  annotate("text", x=0.82, y=0.3, label="PP.H4 = 0.80",
           hjust=0, size=3, color="red") +
  annotate("rect", xmin=0.80, xmax=1.02, ymin=-log10(0.05), ymax=Inf,
           fill="red", alpha=0.06) +
  scale_color_brewer(palette="Set2", name="Cancer") +
  scale_size_continuous(name="|b_SMR|", range=c(1.2,5),
                        breaks=c(0.1,0.3,0.5,0.7)) +
  scale_x_continuous(labels=percent_format(), limits=c(0,1.02)) +
  scale_y_continuous(limits=c(0, NA)) +
  labs(
    title    = "Triangulating Causal Evidence: Colocalization × SMR",
    subtitle = "X-axis: PP.H4 (shared causal variant, Phase 4)  |  Y-axis: SMR significance (Phase 5)\nTop-right quadrant = strong causal evidence from both methods",
    x        = "PP.H4 — posterior probability of shared causal variant (coloc)",
    y        = expression(-log[10](italic(p)[SMR]))
  ) +
  theme_bw(base_size=11) +
  theme(
    plot.title    = element_text(size=12, face="bold"),
    plot.subtitle = element_text(size=8.5, color="grey40"),
    legend.position = "right",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(FIG_DIR,"fig7_smr_coloc_linkage.pdf"), fig7, width=9, height=7, device=cairo_pdf)
ggsave(file.path(FIG_DIR,"fig7_smr_coloc_linkage.png"), fig7, width=9, height=7, dpi=300)
cat("  Saved fig7_smr_coloc_linkage\n")

# ============================================================================
# FIGURE 8 — BMP4 × CRC: HEIDI instrument plot (Wald ratios across top-K SNPs)
# ============================================================================
cat("Figure 8: BMP4 x CRC HEIDI instrument plot...\n")

# Reconstruct the Wald ratios for BMP4 x CRC from the RDS data
bmp4_rds <- "revise_plan/smr_coloc/results/pqtl_regions/BMP4.rds"
crc_gwas  <- "revise_plan/smr_coloc/local_data/cancer_gwas/GCST90255675.h.tsv.gz"

pqtl_reg <- readRDS(bmp4_rds)
cat("  BMP4 pQTL loaded:", nrow(pqtl_reg), "SNPs\n")

# Load CRC GWAS region
gwas_raw <- read_tsv(crc_gwas,
  col_select=c("chromosome","base_pair_location","effect_allele","beta","standard_error","p_value"),
  col_types=cols(.default="c"), show_col_types=FALSE)
gwas_reg <- gwas_raw %>%
  mutate(chrom=as.character(chromosome), pos=as.integer(base_pair_location),
         beta=as.numeric(beta), se=as.numeric(standard_error),
         hm_effect_allele=effect_allele) %>%
  filter(chrom=="11", pos>=74221142, pos<=75221142, !is.na(beta), !is.na(se), se>0)
rm(gwas_raw); gc()
cat("  CRC GWAS region:", nrow(gwas_reg), "SNPs\n")

merged <- inner_join(
  pqtl_reg %>% select(chrom,pos,a1,a0,beta,se) %>%
    rename(b.pq=beta, s.pq=se, ea.pq=a1),
  gwas_reg %>% select(chrom,pos,hm_effect_allele,beta,se) %>%
    rename(b.gw=beta, s.gw=se, ea.gw=hm_effect_allele),
  by=c("chrom","pos"), relationship="many-to-many"
) %>%
  arrange(desc(abs(b.pq/s.pq))) %>%
  distinct(chrom, pos, .keep_all=TRUE) %>%
  mutate(flip = toupper(ea.gw) != toupper(ea.pq),
         b.gw = ifelse(flip, -b.gw, b.gw)) %>%
  filter(abs(b.pq)>0, abs(b.gw)>0, is.finite(b.pq), is.finite(b.gw))

# Lead SMR estimate
lead_snp <- merged %>% slice_max(abs(b.pq/s.pq), n=1, with_ties=FALSE)
b_smr <- lead_snp$b.gw / lead_snp$b.pq
cat(sprintf("  Lead b_SMR = %.4f at chr11:%d\n", b_smr, lead_snp$pos))

# Top 20 instruments
instruments <- merged %>%
  slice_max(abs(b.pq/s.pq), n=20, with_ties=FALSE) %>%
  mutate(
    wald_ratio = b.gw / b.pq,
    var_wald   = wald_ratio^2 * ((s.gw/b.gw)^2 + (s.pq/b.pq)^2),
    se_wald    = sqrt(pmax(var_wald, 0)),
    ci_lo      = wald_ratio - 1.96*se_wald,
    ci_hi      = wald_ratio + 1.96*se_wald,
    rank       = row_number(),
    snp_label  = paste0("chr11:", format(pos, big.mark=",")),
    is_lead    = (pos == lead_snp$pos)
  )

fig8 <- ggplot(instruments, aes(y=reorder(snp_label, -rank), x=wald_ratio,
                                 color=is_lead, shape=is_lead)) +
  geom_vline(xintercept=b_smr, linetype="dashed", color="#b30000", linewidth=0.7) +
  geom_errorbarh(aes(xmin=ci_lo, xmax=ci_hi), height=0.35, linewidth=0.5) +
  geom_point(size=2.8) +
  scale_color_manual(values=c("FALSE"="grey55","TRUE"="#b30000"),
                     labels=c("FALSE"="Instrument","TRUE"="Lead pQTL SNP"),
                     name="") +
  scale_shape_manual(values=c("FALSE"=16,"TRUE"=18), guide="none") +
  labs(
    title    = "BMP4 × CRC: HEIDI Test — Instrument Consistency",
    subtitle = sprintf("Top 20 pQTL instruments. Red dashed = b_SMR lead (%.3f). Consistent Wald ratios → HEIDI passes (p=0.885)", b_smr),
    x        = "Wald ratio (b_GWAS / b_pQTL per instrument)",
    y        = "Instrument SNP (ranked by pQTL z-score)"
  ) +
  theme_bw(base_size=10) +
  theme(
    plot.title    = element_text(size=11, face="bold"),
    plot.subtitle = element_text(size=8, color="grey40"),
    axis.text.y   = element_text(size=8),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(FIG_DIR,"fig8_BMP4_CRC_HEIDI.pdf"), fig8, width=8, height=7, device=cairo_pdf)
ggsave(file.path(FIG_DIR,"fig8_BMP4_CRC_HEIDI.png"), fig8, width=8, height=7, dpi=300)
cat("  Saved fig8_BMP4_CRC_HEIDI\n")

# ============================================================================
# FIGURE 9 — Combined evidence panel: BMP4 × CRC summary
# ============================================================================
cat("Figure 9: BMP4 x CRC combined evidence panel...\n")

# Panel A: posterior probability bars (replicate fig2 with SMR added)
bmp4_crc_coloc <- coloc %>% filter(protein=="BMP4", cancer=="CRC")
bmp4_crc_smr   <- smr   %>% filter(protein=="BMP4", cancer=="CRC")

pp_data <- tibble(
  Hypothesis = factor(
    c("H0\n(no signal)","H1\n(pQTL only)","H2\n(GWAS only)",
      "H3\n(distinct)","H4\n(shared)"),
    levels=c("H0\n(no signal)","H1\n(pQTL only)","H2\n(GWAS only)",
             "H3\n(distinct)","H4\n(shared)")
  ),
  PP = c(bmp4_crc_coloc$PP.H0, bmp4_crc_coloc$PP.H1, bmp4_crc_coloc$PP.H2,
         bmp4_crc_coloc$PP.H3, bmp4_crc_coloc$PP.H4),
  coloc=c(FALSE,FALSE,FALSE,FALSE,TRUE)
)

pA <- ggplot(pp_data, aes(x=Hypothesis, y=PP, fill=coloc)) +
  geom_col(width=0.7, color="white") +
  geom_text(aes(label=ifelse(PP>0.005, sprintf("%.3f",PP),"")),
            vjust=-0.3, size=3.2, fontface="bold") +
  scale_fill_manual(values=c("grey75","#b30000"), guide="none") +
  scale_y_continuous(labels=percent_format(), limits=c(0,1.1)) +
  labs(title="A. Colocalization (Phase 4)",
       subtitle="PP.H4 = 0.994",
       x="Hypothesis", y="Posterior probability") +
  theme_minimal(base_size=10) +
  theme(panel.grid.major.x=element_blank(), panel.grid.minor=element_blank(),
        plot.title=element_text(size=10,face="bold"),
        plot.subtitle=element_text(size=8.5,color="#b30000",face="bold"))

# Panel B: SMR effect estimate
smr_row <- bmp4_crc_smr
pB_data <- tibble(
  label="BMP4 → CRC",
  b=smr_row$b_SMR, lo=smr_row$ci_lo, hi=smr_row$ci_hi,
  pval=smr_row$p_SMR, pheidi=smr_row$p_HEIDI
)

pB <- ggplot(pB_data, aes(y=label, x=b)) +
  geom_vline(xintercept=0, linetype="dashed", color="grey50") +
  geom_errorbarh(aes(xmin=lo, xmax=hi), height=0.25, color="#b30000", linewidth=1.2) +
  geom_point(color="#b30000", size=5, shape=18) +
  geom_text(aes(label=sprintf("b = %.3f\np_SMR = %.1e\np_HEIDI = %.3f", b, pval, pheidi)),
            hjust=-0.1, vjust=0.5, size=3.2, color="grey20") +
  scale_x_continuous(limits=c(-0.2, 1.4)) +
  labs(title="B. SMR Effect Estimate (Phase 5)",
       subtitle="Wald ratio at lead pQTL SNP",
       x="b_SMR (effect per SD plasma BMP4 on CRC log-odds)", y=NULL) +
  theme_minimal(base_size=10) +
  theme(panel.grid.minor=element_blank(),
        axis.text.y=element_blank(),
        plot.title=element_text(size=10,face="bold"),
        plot.subtitle=element_text(size=8.5,color="grey40"))

fig9 <- pA + pB +
  plot_annotation(
    title="BMP4 × Colorectal Cancer: Multi-Method Causal Evidence",
    subtitle="Phase 4 (Coloc) + Phase 5 (SMR) converge on the same causal variant at chr11:74,721,142",
    theme=theme(plot.title=element_text(size=12,face="bold"),
                plot.subtitle=element_text(size=9,color="grey40"))
  )

ggsave(file.path(FIG_DIR,"fig9_BMP4_CRC_combined.pdf"), fig9, width=10, height=5.5, device=cairo_pdf)
ggsave(file.path(FIG_DIR,"fig9_BMP4_CRC_combined.png"), fig9, width=10, height=5.5, dpi=300)
cat("  Saved fig9_BMP4_CRC_combined\n")

cat(sprintf("\n=== All Phase 5 figures saved to %s ===\n", FIG_DIR))
list.files(FIG_DIR, pattern="fig[6-9].*\\.(pdf|png)$") |> sort() |> cat(sep="\n")
