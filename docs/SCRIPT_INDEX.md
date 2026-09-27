# Reproduction entry points

This index identifies the scripts to start from for the v4.4 manuscript.
The `src/figures/` directory also retains earlier visual designs and
component builders; running every file in that directory is not a step in
the analysis.

| Manuscript analysis | Entry point | Inputs needed |
|---|---|---|
| Locked 25-protein discovery model and held-out metrics | `src/panels/02_may_vRSX_seed52_locked_25_lr_l2_reproducer.py` | Included discovery matrix; writes selected-feature and metric tables under `revise_plan/part_b_multiclass/`. |
| Disease-specific panel derivation | `src/panels/run_pan_cancer_loop.py` | Included discovery matrix; writes outputs under `revise_plan/part_a_pan_cancer/`. |
| Held-out disease-specific Figure 6 | `src/figures/fig7_table_s9_refit.py` | Included discovery matrix, `data/panel_memberships.csv`, and Table S9 audit counts. This script renders the original three-panel layout via `fig7_panels.py` and `build_slide_mosaics.py`. |
| External marker-set refits (Figures 7 and 8) | `src/external/generic_locked_panel_external_validation.py` | Processed sample-by-feature matrix and binary metadata for each source accession in `data/README.md`; these matrices are not included. `fig8_panels.py` and `fig9_panels.py` render the original external result-tree tables and cannot run from this archive alone. |
| External estimator comparison | `src/external/generic_locked_panel_external_validation_sklearn.py` | The same prepared external inputs. This is a sensitivity fit, not the source of Table S10's reported calls. |
| Functional and interaction analyses (Figures 9 and 10) | `src/enrichment/` plus corresponding `src/figures/` builders | STRING, Enrichr, and other specified resources plus intermediate analysis tables in the original `revise_plan/` tree. These outputs are not packaged here. |
| Genetic evidence (Figure 11) | `src/genetics/official_smr_heidi_README.md` and its scripts | UKB-PPP pQTL approval, GWAS and LD downloads, external SMR/PLINK binaries, and path configuration. `src/figures/fig13_regional_evidence_2026-09-05.py` also expects intermediate genetic tables that are not packaged here. |

The included figure scripts provide source provenance. Figure 6 has been
verified as directly rebuildable from the packaged inputs. External and
genetic methods are documented, but their full
source-data processing and every plot dependency have not been reproduced
from this archive alone.
