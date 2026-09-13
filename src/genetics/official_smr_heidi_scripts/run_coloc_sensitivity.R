# Rerun coloc.abf on the strictly matched promoted loci.

.libPaths(c("E:/Proteomics/revise_plan/smr_coloc/r_libs", "E:/Proteomics/.rlib", .libPaths()))
suppressPackageStartupMessages({
  library(coloc)
  library(readr)
  library(dplyr)
})

workflow <- "E:/Proteomics/revise_plan/smr_coloc/official_smr_heidi"
configs <- list(
  list(pair_id="BMP4_CRC", protein="BMP4", cancer="CRC", center=74716876L,
       pqtl_n=33372L, gwas_n=185616L, cases=78473L),
  list(pair_id="LEP_BRC", protein="LEP", cancer="BRC", center=53772541L,
       pqtl_n=33693L, gwas_n=139274L, cases=76192L)
)

results <- list()
for (cfg in configs) {
  pqtl <- read_csv(file.path(workflow, "qc", paste0(cfg$pair_id, "_pqtl_accepted.csv")),
                   show_col_types=FALSE) %>%
    transmute(SNP, pos=Bp, beta_p=Beta, se_p=se, maf=Freq)
  gwas <- read_csv(file.path(workflow, "qc", paste0(cfg$pair_id, "_gwas_accepted.csv")),
                   show_col_types=FALSE) %>%
    transmute(SNP, beta_g=b, se_g=se)
  merged <- inner_join(pqtl, gwas, by="SNP") %>% distinct(SNP, .keep_all=TRUE)

  for (window in c(250000L, 500000L)) {
    subset <- merged %>% filter(abs(pos - cfg$center) <= window)
    for (p12 in c(1e-5, 1e-6, 1e-7)) {
      fit <- coloc.abf(
        dataset1=list(beta=subset$beta_p, varbeta=subset$se_p^2,
                      N=cfg$pqtl_n, type="quant", snp=subset$SNP, MAF=subset$maf),
        dataset2=list(beta=subset$beta_g, varbeta=subset$se_g^2,
                      N=cfg$gwas_n, s=cfg$cases/cfg$gwas_n,
                      type="cc", snp=subset$SNP),
        p12=p12
      )
      s <- fit$summary
      results[[length(results) + 1L]] <- data.frame(
        pair_id=cfg$pair_id, window_bp_each_side=window, p12=p12,
        n_snps=nrow(subset), pqtl_n=cfg$pqtl_n, gwas_n=cfg$gwas_n,
        PP.H0=unname(s["PP.H0.abf"]), PP.H1=unname(s["PP.H1.abf"]),
        PP.H2=unname(s["PP.H2.abf"]), PP.H3=unname(s["PP.H3.abf"]),
        PP.H4=unname(s["PP.H4.abf"])
      )
    }
  }
}

out <- bind_rows(results)
write_csv(out, file.path(workflow, "results", "coloc_abf_sensitivity.csv"))
print(out)
