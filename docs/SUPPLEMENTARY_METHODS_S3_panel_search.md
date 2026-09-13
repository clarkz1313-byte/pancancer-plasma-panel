# Supplementary Methods S3: panel search specification

**Manuscript:** A 25-protein plasma panel discriminates 12 cancer types, and a
paired disease-specific panel set confirms them
**Compiled:** 2026-09-12

This file specifies the multiclass panel search that Materials and methods
summarises, and the size rule applied to the disease-specific panels. Every
value below is taken from the code that produced the reported results;
the implementing file is named against each item.

---

## S3.1 Candidate bank

`revise_plan/scripts/run_pan_feature_multiclass.py` —
`collect_train_only_feature_sources`, `build_candidate_feature_set`

Built from development-partition data only, after the one-versus-rest
Mann–Whitney screen described in Materials and methods.

| Rule | Definition |
|---|---|
| Per-cancer quota | The 10 most significant proteins for each of the 12 one-versus-rest contrasts, ordered by Bonferroni-adjusted *P*. Guarantees every cancer a contribution to the bank. |
| Coverage | Any protein reaching Bonferroni significance in at least 2 of the 12 contrasts. |
| Bank | Union of the two, giving **180 proteins** (`train_only_candidate_feature_bank.csv`). |

## S3.2 The ten ranking strategies

`revise_plan/scripts/27_apr_multiclass_compact_tournament_v2.py` —
`build_rankings_v2`; ranker names are the `SEARCH_RANKERS` tuple in
`02_may_v8_final_reproducibility_tournament.py`.

Each strategy returns a complete ordering of the 180-protein bank.

| Name in code | What it ranks by |
|---|---|
| `LOGISTIC_COEF` | Mean absolute coefficient across the 12 rows of a penalised multinomial logistic fit. |
| `RF_IMPORTANCE` | Random-forest impurity importance. |
| `UNIVAR_OVR_AUC` | Univariate one-versus-rest AUC, averaged over classes. |
| `OVR_STABILITY` | How consistently a protein is retained across resampled one-versus-rest fits. |
| `OVR_MINMAX` | A protein's value to its *worst-served* class in the one-versus-rest contrasts, rather than its average value. |
| `PAIRWISE_STABILITY` | The stability measure recomputed over all pairwise class contrasts rather than one-versus-rest. |
| `PAIRWISE_MINMAX` | The worst-class measure recomputed over all pairwise class contrasts. |
| `COVERAGE_IMPORTANCE` | 0.50 × normalised coefficient magnitude + 0.35 × normalised number of cancers in which the protein is significant + 0.15 × their product. |
| `OVR_BONF_QUOTA` | The per-cancer quota order of S3.1, taken directly as a ranking. |
| `CONSENSUS_SOFT_VOTE` | Reciprocal-rank vote over the other nine, with fixed weights: PAIRWISE_STABILITY 1.30, OVR_STABILITY 1.25, PAIRWISE_MINMAX 1.20, LOGISTIC_COEF 1.15, OVR_MINMAX 1.10, COVERAGE_IMPORTANCE 1.00, UNIVAR_OVR_AUC 0.90, OVR_BONF_QUOTA 0.85, RF_IMPORTANCE 0.75. |

The two `MINMAX` rankings and the quota exist for the same reason: on a cohort
whose smallest class holds 29 patients, a ranking that optimises average
behaviour will drop the markers that the rarest cancers depend on.

## S3.3 Candidate panel construction

`02_may_v8_final_reproducibility_tournament.py` — `generate_panel_candidates`

For every size in 12–30:

| Origin | Construction | Count |
|---|---|---|
| `structural` | A per-class quota block (minimum 0, 1 or 2 markers per class), then a block of proteins significant in several cancers (0, 4, 8, 12, 16 or the full size), then the ranking's own order, truncated or extended to the target size. Repeated for each of the ten rankings. | 686 |
| `weighted_random` | Quota positions locked, remaining slots sampled without replacement with probability proportional to consensus priority. | 2,314 |
| | **Total distinct panels** | **3,000** |

Recorded in `stage0_panel_candidates_prescreen.csv`.

## S3.4 Scoring objective

`27_apr_multiclass_compact_tournament_v2.py` — `compact_objective`

Every candidate was scored by stratified five-fold cross-validation inside the
development partition:

```
0.25 × balanced accuracy
+ 0.25 × macro F1
+ 0.20 × macro one-versus-rest AUC
+ 0.20 × minimum class recall
− 0.20 × (panel size / largest size searched)
− 0.10 × standard deviation of macro F1 across folds
```

Two terms make this a compactness search rather than an accuracy search: the
explicit size penalty, and the minimum-class-recall term that prevents a panel
from buying aggregate accuracy at the expense of its worst class. The
cross-fold standard deviation penalises panels whose advantage does not
reproduce across folds.

The 540 best candidates, spread across the size range, were then scored once on
the held-out partition (`stage1_selected_for_test.csv`). Panel size was fixed
at 25 on that evidence; the sweep is plotted as Figure 3-5.

## S3.5 Final multiclass estimator

`27_apr_multiclass_compact_tournament_v2.py` — `make_estimator_v2`, `LR_L2`

Standardisation followed by multinomial logistic regression, L2 penalty,
`C = 0.1`, `class_weight="balanced"`, `lbfgs` solver, intercept unpenalised
(scikit-learn default), `max_iter = 4000`.

## S3.6 Disease-specific panel size rule

`revise_plan/scripts/common.py` — `build_panel_review_tables`

Applied independently for each cancer to nested panels of the top 1–18 ranked
proteins plus the full Bonferroni pool:

1. Keep every panel whose held-out AUC is within **0.02** of that cancer's best
   and whose held-out balanced accuracy is within **0.05** of that cancer's
   best.
2. Take the **smallest** survivor (`n_stat_min`).
3. Raise it to **2 proteins** where the single top-ranked marker is shared with
   another cancer, so that no panel ships as a one-protein test resting on a
   non-specific marker (`n_clinical_min`). This raised only CLL, from 1 to 2.

The rule returned panels of 2 to 13 proteins. It is drawn, with its tolerance
band, as Figure 4-5.

## S3.7 Reported estimator for the disease-specific panels

Membership frozen, then standardisation and logistic regression **without
regularisation** (`penalty=None`, `lbfgs`, `max_iter = 5000`), fitted on the
development partition and scored at a fixed 0.50 threshold, with the 0.80
threshold reported alongside for the confirmation role.
Source run: `revise_plan/part_a_single/seed52_full_metrics_2026-09-12/`,
which reproduces `threshold_choice_2026-09-07` and adds expected calibration
error and Brier score, so that every disease-specific number in the manuscript
comes from a single seed-52 fit.
