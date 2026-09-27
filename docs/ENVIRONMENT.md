# Reference computational environment

This is the tested setup for rerunning the packaged code. The study's
analyses and figures were produced across more than one machine. Historical
Python package builds, BLAS/LAPACK libraries, and thread settings were not
recorded for every result, so these pins are a reproducibility target rather
than a certified inventory of every producing machine.

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

Versions of the ancillary packages in the successful 2026-09-27 audit are in
`requirements.txt` and `AUDIT_PYTHON_FREEZE_2026-09-27.txt`. Their historical
versions on other producing machines are not known.

### Tested local rerun, 2026-09-27

- 64-bit CPython 3.11.9 on Windows 11 build 26100, x86-64.
- NumPy 2.4.3 and SciPy 1.17.1 with OpenBLAS 0.3.31.dev and 0.3.30,
  respectively; both reported 16 BLAS threads during the audit. The
  scikit-learn OpenMP runtime reported 16 threads.
- `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` were unset;
  library defaults applied. This is a record of this audit, not a claim about
  the historical training machines.
- The exact installed Python package list is in
  `AUDIT_PYTHON_FREEZE_2026-09-27.txt`.

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
The locus-specific follow-up's recorded software manifest names R 4.4.2,
coloc 5.2.3, SMR 1.3.1, and PLINK 2.0.0-a.7.4; it is copied to
`GENETICS_SOFTWARE_MANIFEST.csv`. Its absolute paths record the producing
machine and must be changed for a new machine.

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

**External refit**: marker membership frozen, L2-logistic model fitted by the
included production script with penalty coefficient 1.0 and fixed-step gradient
descent (step size 0.05, 2,500 steps) over 50 repeated participant-grouped
75/25 splits. The scikit-learn `C=1.0, lbfgs` variant is an estimator
sensitivity analysis, not the source of the reported external counts.

**Colocalisation**: approximate Bayes factors at 250 kb / 500 kb windows,
cross-trait prior swept across 1e-5, 1e-6, 1e-7; promotion at 1e-5. SMR/HEIDI
run in both the current 20-variant-capped mode and the uncapped mode.

## Comparing a rerun with the manuscript

First fix the code commit, input CSV content and row order, class labels,
sample grouping, panel membership, split seed, and thresholds. Use seed 52
for the reported participant partition. Other seeds are useful for split
sensitivity checks and may yield slightly different estimates. Verify the
input checksum before running (see `data/README.md`).

The `1e-6` absolute tolerance and four-decimal agreement used in earlier
internal audits were working comparison criteria for a specified run, not a
guarantee across machines. Our 2026-09-27 locked-panel and Figure 6 reruns
passed their code assertions in the tested environment above. Small
floating-point differences in scores or continuous metrics may occur with a
different operating system, numerical library, thread count, or package
build. A changed feature list, split, or integer confusion count needs
investigation; do not assign it to numerical noise without examining the
individual scores near the threshold and the run provenance. Figure pixels
may also vary with fonts and rendering backends.

For a comparison, record the code commit, source-data checksums, Python/R
versions, full `pip freeze --all` or R `sessionInfo()`, OS and architecture,
BLAS/LAPACK implementation, and thread settings. Report the observed
differences in features, splits, predictions, and metrics rather than only
whether an arbitrary tolerance passed.
