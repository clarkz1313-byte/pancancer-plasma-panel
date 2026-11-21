# Computational environment (release v1.0.2)

Versions match Appendix S2 of the Supporting Information. The analyses were run across more than one machine, and
BLAS/LAPACK builds and thread settings were not recorded for every result, so these pins are a reproducibility
target rather than a certified inventory of every producing machine.

## Python 3.11.9 (64-bit CPython)

Pinned in `requirements.txt`: scikit-learn 1.6.1, SciPy 1.15.2, statsmodels 0.14.5, NumPy 2.2.3, pandas 2.2.3,
gseapy 1.3.1, requests 2.34.2, umap-learn 0.5.12, networkx 3.4.2, matplotlib 3.10.1, seaborn 0.13.2, Pillow 11.1.0,
xenaPython 1.0.14. CPTAC tumor proteomics used cptac 1.5.14 in a separate environment.

## R 4.4.2 (genetic analyses only)

| Package | Version |
|---|---|
| coloc | 5.2.3 |
| susieR | 0.14.2 |
| TwoSampleMR | 0.7.9 |
| ieugwasr | 1.1.0 |
| dplyr, readr, data.table | 1.2.1, 2.2.0, 1.18.2.1 |

The R scripts add `$PROTEOMICS_ROOT/.rlib` to the library path, so packages can be installed into a project-local
library. External binaries: SMR 1.3.1 and PLINK 2.0.0-a.7.4.

## Fixed analysis settings

```
input             code/data/processed/filtered_pancancer_data.csv
class column      Cancer
sample id column  Sample_ID
split             stratified 70:30, seed 52 (962 development, 413 internal validation patients)
imputation        KNN, k = 5, fitted on the development set only
scaling           z-scores with the mean and SD of each model's training patients
screen            Mann-Whitney U, one cancer against the other 11, Bonferroni over 1,463 proteins
```

**Cancer-specific panels.** L1-penalized logistic regression (C=0.1) of each cancer against the other 11 ranks the
proteins; the top 18 are kept. Panels of the top 1 to 18 are scored by five-fold cross-validation in the development
set; the smallest panel within 0.02 AUC and 0.05 balanced accuracy of the best is kept (at least two proteins when
the top protein is significant for another cancer). The final model of each cancer is an unpenalized logistic
regression on its 2 to 12 proteins, fitted once on all 962 development patients.

**Multiclass panel.** Pool of 801 proteins (14 most significant per cancer plus proteins significant for two or more
cancers); candidate bank of the 180 highest on 0.45 × cancers significant + 0.35 × top-14 lists entered (both scaled
0 to 1) + 0.20 × 1/(1 + best rank); ten rankings; 3,000 candidate panels of 12 to 30 proteins; each fitted with
ridge multinomial logistic regression (C=0.1, balanced class weights) and scored by four-fold cross-validation on a
combined objective. Selected panel: 26 proteins (`code/v7_plan/panels/panel_manifest_v7.json`).

**Two-step model.** Step 1 names the most probable cancer; step 2 accepts it when the cancer-specific model of
that cancer gives a probability of at least 0.50. Comparison with the same number of most-confident predictions uses
2,000 bootstrap resamples of patients.

**External refits.** Protein lists frozen. Single-cancer datasets: binary L2-penalized logistic regression fitted
by the production runner
(`revise_plan/external_validation_roadmap/scripts/generic_locked_panel_external_validation_null50.py`; penalty 1.0,
fixed-step gradient descent with step size 0.05 and 2,500 steps) on 50 patient-grouped 75:25 splits. Pan-cancer
datasets: 50 stratified 75:25 splits; multiclass model refitted with its development settings and cancer-specific
models as binary L2-penalized scikit-learn regressions (C=1.0).

## Comparing a rerun with the manuscript

First fix the code commit, input CSV content and row order, class labels, sample grouping, panel membership, split
seed and thresholds. Use seed 52 for the reported partition; other seeds are useful for split sensitivity and may
give slightly different estimates. Verify the input checksum before running (see `code/data/README.md`).

Small floating-point differences in scores or continuous metrics may occur with a different operating system,
numerical library, thread count or package build. A changed feature list, split or integer confusion count needs
investigation; do not assign it to numerical noise without examining the scores near the threshold and the run
provenance. Figure pixels may also vary with fonts and rendering backends.

For a comparison, record the code commit, source-data checksums, Python and R versions, full `pip freeze --all` or R
`sessionInfo()`, OS and architecture, BLAS/LAPACK implementation and thread settings, and report the observed
differences in features, splits, predictions and metrics.
