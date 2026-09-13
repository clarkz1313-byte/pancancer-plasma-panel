# Independent table and lead-position checks for the genetics handoff.

result_dir <- "E:/Proteomics/revise_plan/smr_coloc/results"

phase5 <- read.csv(file.path(result_dir, "phase5_smr_results.csv"),
                   check.names = FALSE)
leads <- readRDS(file.path(result_dir, "pqtl_leads_checkpoint.rds"))

position_rows <- lapply(seq_len(nrow(phase5)), function(i) {
  row <- phase5[i, ]
  lead <- leads[[row$protein]]
  region <- readRDS(lead$region_file)
  matched <- region[
    abs(region$beta - row$b_pqtl_lead) < 1e-12 &
      abs(region$se - row$se_pqtl_lead) < 1e-12,
  ]
  data.frame(
    protein = row$protein,
    cancer = row$cancer,
    stored_pos = row$lead_pos,
    actual_pos = if (nrow(matched) == 1) matched$pos else NA_integer_,
    n_matches = nrow(matched)
  )
})
positions <- do.call(rbind, position_rows)
write.csv(
  positions,
  file.path(result_dir, "independent_audit_lead_positions.csv"),
  row.names = FALSE
)

tier <- read.csv(file.path(result_dir, "tier1_combined_316.csv"),
                 check.names = FALSE)
gwas_p <- 2 * pnorm(abs(tier$b_gwas_lead / tier$se_gwas_lead),
                    lower.tail = FALSE)

single <- read.csv(
  "E:/Proteomics/paper/supplementary_tables/Supplementary_Table_S1_single_panel_protein_summary.csv",
  check.names = FALSE
)
members <- strsplit(single$deploy_proteins, ";", fixed = TRUE)
non_cvx <- single$target_class != "CVX"
locked <- unique(phase5$protein)
expected_parta <- unique(do.call(rbind, lapply(which(non_cvx), function(i) {
  protein <- members[[i]]
  protein <- protein[!protein %in% locked]
  if (length(protein) == 0) return(NULL)
  data.frame(protein = protein, cancer = single$target_class[i])
})))
observed_parta <- unique(tier[
  tier$panel_source == "12_single_cancer_panels", c("protein", "cancer")
])
parta_keys_expected <- paste(expected_parta$protein, expected_parta$cancer)
parta_keys_observed <- paste(observed_parta$protein, observed_parta$cancer)

checks <- data.frame(
  check = c(
    "tier_rows",
    "tier_duplicate_protein_cancer_pairs",
    "tier_missing_cells",
    "max_abs_log10_difference_wald_p_vs_gwas_p",
    "wald_bonferroni_passes",
    "coloc_h4_gt_0.8",
    "formal_final_hits",
    "locked25_observed_pairs",
    "locked25_intended_pairs_25x11",
    "locked25_unanalyzed_pairs",
    "parta_expected_pairs",
    "parta_observed_pairs",
    "parta_missing_pairs",
    "parta_extra_pairs",
    "phase5_unique_lead_position_matches",
    "phase5_ambiguous_lead_position_matches",
    "phase5_stored_position_mismatches"
  ),
  value = c(
    nrow(tier),
    sum(duplicated(tier[c("protein", "cancer")])),
    sum(is.na(tier)),
    max(abs(log10(tier$p_wald_pkg) - log10(gwas_p))),
    sum(tier$p_wald_pkg < 0.05 / nrow(tier)),
    sum(tier$PP.H4 > 0.8),
    sum(tier$final_hit),
    sum(tier$panel_source == "locked25_x_11cancers"),
    25 * 11,
    25 * 11 - sum(tier$panel_source == "locked25_x_11cancers"),
    nrow(expected_parta),
    nrow(observed_parta),
    length(setdiff(parta_keys_expected, parta_keys_observed)),
    length(setdiff(parta_keys_observed, parta_keys_expected)),
    sum(positions$n_matches == 1),
    sum(positions$n_matches != 1),
    sum(!is.na(positions$actual_pos) & positions$stored_pos != positions$actual_pos)
  )
)
write.csv(
  checks,
  file.path(result_dir, "independent_audit_table_checks.csv"),
  row.names = FALSE
)

print(checks, row.names = FALSE)
