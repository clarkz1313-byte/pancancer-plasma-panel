# Release v1.0.2 manifest

Paths are relative to `code/`, which mirrors the project folder used for the manuscript. Result folders are written by
the scripts under `code/` (or under `PROTEOMICS_ROOT` if set); they are not shipped. The final result tables are in
the Supporting Information archive of the manuscript.

## Tables and figures (Table S30 of the Supporting Information)

| Item | Result files | Scripts |
|---|---|---|
| Figure 1 | `v7_plan/panels/panel_memberships_v7.csv` | `v7_plan/figures/scripts/fig1_v7b_methods.py` |
| Figure 2 | `data/processed/filtered_pancancer_data.csv` | `v7_plan/figures/scripts/fig2_v7d_cohort.py` |
| Figure 3; Tables S3, S5 | `revise_plan/part_b_multiclass/v11_seed52_lr_l2_wide_compact_sensitivity/tables/stage1_cv_results.csv`; `revise_plan/pilot_ds_dev_only_size/size_sweep_devonly.csv` | `revise_plan/scripts/02_may_v11_seed52_lr_l2_wide_compact_sensitivity.py`; `revise_plan/scripts/91_oct_v65_ds_panels_dev_only_reselect_2026-10-16.py`; `v7_plan/figures/scripts/fig3_v7h_layout.py` |
| Figure 4; Tables S6-S9, A2, A4 | `v7_plan/results/B1_multiclass/`; `v7_plan/results/B4_internal_disease_specific/`; `v7_plan/results/B3_published_comparison/` | `v7_plan/scripts/B1_multiclass_internal_v7.py`; `revise_plan/scripts/92_oct_v65_ds_devonly_downstream_internal_tcga_2026-10-16.py`; `v7_plan/scripts/B3_published_comparison_v7.py`; `v7_plan/figures/scripts/fig4_v7l_internal.py` |
| Figure 5; Figure S3; Table S10 | `v7_plan/results/B6_external/` | `v7_plan/scripts/B6_single_cancer_external_v7.py`; `v7_plan/figures/scripts/fig5_v7g_tissue_mosaic13.py` |
| Figure 6; Table 2; Tables A5-A9, S24-S26; Figure S4 | `v7_plan/results/B5_internal_*`; `v7_plan/results/B6_external/two_step_binary/`; `v7_plan/results/B7_tcga/`; `v7_plan/results/B8_mixed_cancer/` | `v7_plan/scripts/B5_internal_two_step_v7.py`; `v7_plan/scripts/B6b_binary_rule_external_v7.py`; `revise_plan/scripts/85_oct_v65_mixed_cancer_core_2026-10-15.py`, `87_oct_v65_mixed_D1_gse2109_expo_2026-10-15.py`, `89_oct_v65_mixed_D4_D5_D2c_2026-10-16.py`, `92_oct_v65_ds_devonly_downstream_internal_tcga_2026-10-16.py`, `93_oct_v65_ds_devonly_mixed_datasets_2026-10-16.py`; `v7_plan/scripts/B8b_gdc_nontcga_v7.py` to `B8g_family_summary_v7b.py`; `v7_plan/figures/scripts/fig6_v7m_two_step.py`; `v7_plan/figures/scripts/fig9c_v7_mixed_forest.py` |
| Figure 7; Tables S11-S18, S12b | `v7_plan/results/B9_biology/` | `v7_plan/scripts/B9_biology_v7.py`, `B9c_string_figures_v7.py`, `B9d_string_multiclass26_v7.py`; `v7_plan/figures/scripts/fig8_v7j_biology_annotation.py` |
| Figure S5; Table S28 | `v7_plan/results/B9_biology/tissue_origin/` | `v7_plan/scripts/B9b_tissue_of_origin_v7.py`, `B9e_tissue_origin_selection_null_v7.py`; `v7_plan/figures/scripts/figS_v7_tissue_origin.py` |
| Figure 8; Tables S20-S23, S27 | `v7_plan/results/B10_genetics_final/`; `v7_plan/results/B10_genetics_ncorrected/` | `v7_plan/scripts/B10_genetics_setup_v7.py`, then `B10b_genetics_chrx_fix_v7.py`, then `B10c_genetics_final_v7.R` and `B10d` to `B10g`; `B10h_make_ncorrected_susie_v7.py`, then `B10h` to `B10k` (R); `v7_plan/figures/scripts/fig7_v7n_genetics.py` |
| Table A3 | `v7_plan/results/B2_sex_devsel/` | `v7_plan/scripts/B2_sex_sensitivity_v7_devsel.py`; `v7_plan/scripts/B2b_sex_mapping_check_v7.py` |
| Table 1 | literature; `v7_plan/results/B5_internal_two_step_summary.json` | manuscript text |
| Tables S1, S4 | `v7_plan/panels/panel_memberships_v7.csv` | `overleaf-tools/build_supplement_v7.py` |

## Run order

1. **Panels.** `revise_plan/scripts/02_may_v11_seed52_lr_l2_wide_compact_sensitivity.py` (multiclass panel search) and
   `revise_plan/scripts/91_oct_v65_ds_panels_dev_only_reselect_2026-10-16.py` (cancer-specific panels). The locked
   panels are shipped in `v7_plan/panels/`, so the later steps can start from them.
2. **Internal validation.** `B1`, `B3`, `B2`, `B2b` and `B5` in `v7_plan/scripts/`;
   `revise_plan/scripts/92_oct_v65_ds_devonly_downstream_internal_tcga_2026-10-16.py`.
3. **Single-cancer datasets.** `v7_plan/scripts/B6_single_cancer_external_v7.py`, then `B6b_binary_rule_external_v7.py`.
   B6 calls the preparation scripts and the refit runner in `revise_plan/external_validation_roadmap/` as
   subprocesses.
4. **Pan-cancer datasets.** `revise_plan/scripts/85`, `87`, `89`, `93`; then `v7_plan/scripts/B8b` to `B8g`.
5. **Biology.** `v7_plan/scripts/B9`, `B9b`, `B9c`, `B9d`, `B9e`.
6. **Genetics.** `v7_plan/scripts/B10`, `B10b` (these run the R pipelines in `revise_plan/smr_coloc/`), then the R
   scripts `B10c` to `B10k` (needs `SMR_DATA_ROOT`).
7. **Figures and supplement.** The `v7_plan/figures/scripts/` builders listed above, then
   `overleaf-tools/build_supplement_v7.py`.

Scripts that are not listed above are helpers loaded by the listed ones (shared settings, earlier panel-search
stages, figure styles); they are shipped so that every import resolves.

## All shipped files


### `data/processed/` (1)

- `filtered_pancancer_data.csv`

### `data/` (1)

- `README.md`

### `figurev6/scripts/` (29)

- `build_ppi_slides.py`
- `build_slide_mosaics.py`
- `fig10_5_ppi_combined.py`
- `fig10_panels.py`
- `fig11_panels.py`
- `fig11c_pooled12panels_network.py`
- `fig11d_pooled12panels_official.py`
- `fig13_regional_evidence_2026-09-05.py`
- `fig2a_cohort.py`
- `fig2b_npx_heatmap.py`
- `fig2b_pooled_distributions.py`
- `fig2f_umap.py`
- `fig3_5_and_fig4_5_panel_size_selection.py`
- `fig3_6_structure_with_sweep.py`
- `fig3_panels.py`
- `fig3_slide3_mosaic.py`
- `fig4_7_sweep_structure_and_pairs.py`
- `fig4_panels.py`
- `fig6_chord_network.py`
- `fig6_panels.py`
- `fig6_parta_sankey_bubble.py`
- `fig6_pathway_enrichment_v2.py`
- `fig89_common.py`
- `fig89_stacked_mosaic.py`
- `fig8_panels.py`
- `fig9_panels.py`
- `fig_candidate_common.py`
- `figure_labels.py`
- `publication_style.py`

### `overleaf-tools/` (21)

- `build_supplement_v7.py`
- `make_fig10_v50_ppi.py`
- `make_fig10_v5110_two_step_mosaic.py`
- `make_fig10_v517_two_step_mosaic.py`
- `make_fig10_v518_two_step_mosaic.py`
- `make_fig10_v519_two_step_mosaic.py`
- `make_fig10_v51d_two_step_mosaic.py`
- `make_fig10_v520_two_step_mosaic.py`
- `make_fig10_v521_two_step_mosaic.py`
- `make_fig10_v522_two_step_mosaic.py`
- `make_fig10_v524_two_step_mosaic.py`
- `make_fig10_v53_two_step_mosaic.py`
- `make_fig1_v52_methods.py`
- `make_fig2_v50_cohort.py`
- `make_fig2_v51_cohort.py`
- `make_fig3_v50_derivation.py`
- `make_fig7_v50_external_disease_specific.py`
- `make_fig7_v51_external_disease_specific.py`
- `make_fig8_v50_external_multiclass.py`
- `make_fig8_v51_external_multiclass.py`
- `make_fig9_v50_pathway.py`

### `revise_plan/external_validation_roadmap/gse13159_aml/scripts/` (1)

- `prepare_gse13159_aml_transcript_proxy_matrix.py`

### `revise_plan/external_validation_roadmap/gse4290/scripts/` (1)

- `prepare_gse4290_transcript_proxy_matrix.py`

### `revise_plan/external_validation_roadmap/scripts/` (5)

- `generic_locked_panel_external_validation.py`
- `generic_locked_panel_external_validation_null50.py`
- `make_single_panel_aux_metadata.py`
- `prepare_geo_gpl570_panel_proxy.py`
- `summarize_grouped_prediction_thresholds.py`

### `revise_plan/ppi_string/` (4)

- `batch_string_export.py`
- `download_STRING_analysis_12_panels.py`
- `fetch_locked25_string.py`
- `fetch_pooled12panels_string.py`

### `revise_plan/scripts/` (36)

- `02_may_v10_seed52_lr_l2_compact_tournament.py`
- `02_may_v11_seed52_lr_l2_wide_compact_sensitivity.py`
- `02_may_v8_final_reproducibility_tournament.py`
- `02_may_v9_seed52_compact_tournament.py`
- `02_may_vRSX_seed52_locked_25_lr_l2_reproducer.py`
- `27_apr_multiclass_compact_tournament.py`
- `27_apr_multiclass_compact_tournament_v2.py`
- `30_sept_sex_sensitivity_2026-09-06.py`
- `33_sept_threshold_choice_2026-09-07.py`
- `34_sept_single_panel_seed52_full_metrics_2026-09-12.py`
- `35_sept_same_sex_rescoring_2026-09-28.py`
- `42_oct_v5_external_2026-09-29.py`
- `45_oct_v5_ora_assay_background_2026-09-29.py`
- `46_oct_v5_ora_background_isolation_2026-09-29.py`
- `47_oct_v5_ppi_background_permutation_2026-09-29.py`
- `48_oct_v5_hallmark_rank_matched_2026-09-29.py`
- `49_oct_v5_panel_redundancy_2026-09-29.py`
- `62_oct_v5_assign_confirm_agreement_2026-09-30.py`
- `64_oct_v5_tcga_assign_confirm_2026-10-01.py`
- `70_oct_v51_two_step_binary_rule_2026-10-01.py`
- `76_oct_v6_confidence_comparator_2026-10-03.py`
- `84_oct_v65_pilot_26_protein_panel_2026-10-15.py`
- `85_oct_v65_mixed_cancer_core_2026-10-15.py`
- `86_oct_v65_mixed_D2_gdc_tcga_mmrf_2026-10-15.py`
- `87_oct_v65_mixed_D1_gse2109_expo_2026-10-15.py`
- `89_oct_v65_mixed_D4_D5_D2c_2026-10-16.py`
- `91_oct_v65_ds_panels_dev_only_reselect_2026-10-16.py`
- `92_oct_v65_ds_devonly_downstream_internal_tcga_2026-10-16.py`
- `93_oct_v65_ds_devonly_mixed_datasets_2026-10-16.py`
- `94_oct_v65_ds_devonly_single_cancer_external_2026-10-16.py`
- `96_oct_v65_ds_devonly_net_benefit_2026-10-16.py`
- `common.py`
- `run_12class_ovr_pipeline.py`
- `run_dea.py`
- `run_enrichment.py`
- `run_pan_feature_multiclass.py`

### `revise_plan/smr_coloc/` (9)

- `v5_run_parta_coloc.R`
- `v5_run_parta_wald_heidi.R`
- `v5_run_tier1_combined.R`
- `v7_run_parta_coloc.R`
- `v7_run_parta_wald_heidi.R`
- `v7_run_tier1_combined.R`
- `v7b_run_parta_coloc.R`
- `v7b_run_parta_wald_heidi.R`
- `v7b_run_tier1_combined.R`

### `scripts/` (3)

- `run_12class_ovr_pipeline.py`
- `run_dea.py`
- `run_enrichment.py`

### `v7_plan/figures/scripts/` (28)

- `fig1_v7_methods.py`
- `fig1_v7b_methods.py`
- `fig2_v7_cohort.py`
- `fig2_v7d_cohort.py`
- `fig3_v7_derivation.py`
- `fig3_v7h_derivation.py`
- `fig3_v7h_layout.py`
- `fig3de_v7_dev_selection.py`
- `fig4_v7d_internal.py`
- `fig4_v7f_internal.py`
- `fig4_v7l_internal.py`
- `fig5_v7c_two_step.py`
- `fig5_v7g_tissue_mosaic13.py`
- `fig6_v7m_two_step.py`
- `fig7_v7_genetics_data.py`
- `fig7_v7c_biology.py`
- `fig7_v7n_genetics.py`
- `fig8_v7_external_tissue.py`
- `fig8_v7j_biology_annotation.py`
- `fig9ab_v7_two_step_tissue.py`
- `fig9c_v7_mixed_forest.py`
- `figS_v7_pathway.py`
- `figS_v7_ppi.py`
- `figS_v7_tissue_origin.py`
- `figS_v7c_pathway.py`
- `figS_v7d_biology_annotation.py`
- `make_fig1_v7_fork.py`
- `v7_style.py`

### `v7_plan/panels/` (2)

- `panel_manifest_v7.json`
- `panel_memberships_v7.csv`

### `v7_plan/scripts/` (33)

- `B10_genetics_setup_v7.py`
- `B10b_genetics_chrx_fix_v7.py`
- `B10c_genetics_final_v7.R`
- `B10d_susie_coloc_v7.R`
- `B10e_susie_coloc_ukbld_v7.R`
- `B10e_ukbb_ld_extract.py`
- `B10f_1kg38_region_ld.py`
- `B10f_susie_coloc_1kg38_v7.R`
- `B10g_susie_pips_1kg38_v7.R`
- `B10h_make_ncorrected_susie_v7.py`
- `B10h_susie_coloc_ncorr_v7.R`
- `B10i_susie_coloc_ukbld_ncorr_v7.R`
- `B10j_susie_coloc_1kg38_ncorr_v7.R`
- `B10k_susie_pips_1kg38_ncorr_v7.R`
- `B1_multiclass_internal_v7.py`
- `B2_sex_sensitivity_v7_devsel.py`
- `B2b_sex_mapping_check_v7.py`
- `B3_published_comparison_v7.py`
- `B5_internal_two_step_v7.py`
- `B6_single_cancer_external_v7.py`
- `B6b_binary_rule_external_v7.py`
- `B8b_gdc_nontcga_v7.py`
- `B8c_platelet_tep_v7.py`
- `B8d_family_summary_v7.py`
- `B8e_strict_breast_v7.py`
- `B8f_cptac_proteomics_v7.py`
- `B8g_family_summary_v7b.py`
- `B9_biology_v7.py`
- `B9b_tissue_of_origin_v7.py`
- `B9c_string_figures_v7.py`
- `B9d_string_multiclass26_v7.py`
- `B9e_tissue_origin_selection_null_v7.py`
- `B9f_olink_subpanel_origin_v7.py`
