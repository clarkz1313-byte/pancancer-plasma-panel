# Computational environment

## Python 3.11.9

Pinned packages (`requirements.txt`):

| Package | Version |
|---|---|
| pandas | 2.3.3 |
| numpy | 2.4.3 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| statsmodels | 0.14.6 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| xgboost | 3.2.0 |
| shap | 0.51.0 |
| mygene | 3.2.2 |
| requests | 2.32.5 |
| tqdm | 4.67.3 |

Also required by figure and enrichment scripts, unpinned:
`umap-learn`, `optuna`, `gseapy`, `networkx`, `Pillow`, `adjustText`, `plotly`.

## R 4.4.2

Used by `src/genetics/` only.

| Package | Version |
|---|---|
| coloc | 5.2.3 |
| TwoSampleMR | 0.7.9 |
| ieugwasr | 1.1.0 |
| susieR | present, version matched to coloc's requirement |
| rlang | >= 1.1.7 (install in the project-local library, not the user library) |
| dplyr | 1.2.1 |
| ggplot2 | 4.0.3 |
| tibble | 3.3.1 |
| tidyr | 1.3.2 |
| readr | 2.2.0 |
| purrr | 1.2.2 |
| ggrepel | 0.9.6 |
| patchwork | 1.3.0 |
| scales | 1.4.0 |
| RColorBrewer | 1.1.3 |

External binaries: SMR 1.3.1, PLINK 2.

## Fixed analysis settings

```
input             data/filtered_pancancer_data.csv
class_column      Cancer
sample_id_column  Sample_ID
seed              52
test_size         0.3
imputer           KNN, k = 5, fit on the development partition only
```

**Disease-specific panels**: L1-penalised logistic regression ranks proteins
within each cancer's Bonferroni-significant pool; panel size set by the
smallest panel within 0.02 AUC and 0.05 balanced accuracy of that cancer's
best. Final panels refit unpenalised with standardisation.

**25-protein multiclass panel**: candidate bank of 180 proteins (per-cancer
quota + coverage rule), 3,000 candidate panels sizes 12–30, scored by
cross-validated composite objective, size fixed at 25. Final model:
class-balanced L2-penalised multinomial logistic regression (C=0.1, lbfgs).
Bootstrap seed 52024, 1,000 resamples.

**External refit**: marker membership frozen, L2-logistic model (C=1.0,
lbfgs) refit within each cohort over 50 repeated participant-grouped 75/25
splits.

**Colocalisation**: approximate Bayes factors at 250 kb / 500 kb windows,
cross-trait prior swept across 1e-5, 1e-6, 1e-7; promotion at 1e-5. SMR/HEIDI
run in both the current 20-variant-capped mode and the uncapped mode.

## Verification

- Feature selections, sample partitions, class labels, and discrete
  predictions: exact match.
- Continuous metrics: absolute tolerance 1e-6.
- Values rounded to four decimal places: exact match.

Verify the input checksum before running anything (see `data/README.md`).
Record the Python/R versions, `pip freeze --all`, and BLAS/LAPACK backend
actually used for any run you intend to compare against the values above.
