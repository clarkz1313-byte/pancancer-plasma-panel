# Supplementary: TRIPOD checklist

**Manuscript:** A 25-protein plasma panel discriminates 12 cancer types, and a
paired disease-specific panel set confirms them
**Guideline:** TRIPOD — Transparent Reporting of a multivariable prediction
model for Individual Prognosis Or Diagnosis (Collins GS, Reitsma JB, Altman DG,
Moons KGM. *Ann Intern Med* 2015;162:55–63. doi:10.7326/M14-0697)
**Completed:** 2026-09-12

---

## Study type under TRIPOD

This is a **TRIPOD type 2a** study: a single dataset was split at random into a
development partition (962 patients) and a held-out partition (413 patients),
the models were developed in the first and evaluated in the second.

The external work reported in the manuscript is **not** a TRIPOD type 4
external validation and is not claimed as one. The frozen quantity across
cohorts is panel membership; the scaler, coefficients and intercept were
re-estimated inside each external cohort, because eleven of the twelve external
endpoints measure tissue transcript abundance rather than plasma NPX and no
common calibration exists between the two scales. Under TRIPOD's vocabulary
this is closest to developing a new model from a predefined predictor set in
each cohort. It tests whether the markers retain discriminative information
after a change of platform and tissue; it cannot show that the discovery model
transports.

Two further deviations are declared here rather than buried:

- **Held-out data informed one design choice.** Multiclass panel size (25) was
  chosen with reference to held-out performance across sizes 12–30. Internal
  multiclass estimates are therefore optimistic relative to a fully nested
  procedure, and the manuscript says so in Methods, Discussion and Limitations.
- **One panel deviates from the common stopping rule.** The BRC
  disease-specific panel was compacted from 17 to 13 markers by manual review
  rather than by the size rule applied to the other 11 panels. Recorded in
  Limitations.

Item numbering follows TRIPOD 2015. `D` = development only, `V` = validation
only, `D;V` = both. Page numbers should be inserted against the typeset proof;
section names are given here because they are stable across versions.

---

## Title and abstract

| Item | Type | Checklist item | Reported in |
|---|---|---|---|
| 1 | D;V | Identify the study as developing and/or validating a multivariable prediction model, the target population, and the outcome to be predicted. | Title; Abstract |
| 2 | D;V | Provide a summary of objectives, study design, setting, participants, sample size, predictors, outcome, statistical analysis, results, and conclusions. | Abstract |

## Introduction

| Item | Type | Checklist item | Reported in |
|---|---|---|---|
| 3a | D;V | Explain the medical context and rationale for developing or validating the model, including references to existing models. | Introduction (paragraphs 1–3; the published size series for this cohort is the existing comparator) |
| 3b | D;V | Specify the objectives, including whether the study describes the development or validation of the model, or both. | Introduction, final paragraph; Methods §Study cohort and design |

## Methods

| Item | Type | Checklist item | Reported in |
|---|---|---|---|
| 4a | D;V | Describe the study design or source of data, separately for the development and validation data sets, if applicable. | Methods §Study cohort and design; Methods §External marker-set evaluation; Table 4 |
| 4b | D;V | Specify the key study dates, including start of accrual, end of accrual, and, if applicable, end of follow-up. | **Not reported.** The discovery cohort accrual dates are those of the source publication and are not restated here. No follow-up was analysed; every outcome is a concurrent diagnostic label. |
| 5a | D;V | Specify key elements of the study setting, including number and location of centres. | Methods §Study cohort and design (secondary analysis of a published multi-site resource); external cohorts in Table 4 |
| 5b | D;V | Describe eligibility criteria for participants. | Methods §Study cohort and design; inclusion criteria for external cohorts in Methods §External marker-set evaluation |
| 5c | D;V | Give details of any treatments received, if relevant. | **Not applicable.** Treatment data were not available in the discovery resource and no treatment effect is modelled. |
| 6a | D;V | Clearly define the outcome that is predicted by the prediction model, including how and when assessed. | Methods §Study cohort and design. The outcome is the recorded cancer group, one of 12, assigned at diagnosis in the source cohort. |
| 6b | D;V | Report any actions to blind assessment of the outcome to be predicted. | **Not applicable.** Outcome labels were fixed in the source data before this analysis began; no outcome was adjudicated here. |
| 7a | D;V | Clearly define all predictors used in developing or validating the model, including how and when they were measured. | Methods §Study cohort and design (Olink Explore 1536 PEA, NPX, 1,463 analytes); external platforms in Table 4 |
| 7b | D;V | Report any actions to blind assessment of predictors for the outcome and other predictors. | **Not applicable.** Predictors are assay readouts generated before and independently of this analysis. |
| 8 | D;V | Explain how the study size was arrived at. | Methods §Study cohort and design. No sample-size calculation was performed; the analysis used the full available published cohort. The 70:30 split ratio was fixed in advance. |
| 9 | D;V | Describe how missing data were handled, with details of any imputation method. | Methods §Panel development, paragraph 1. Five-nearest-neighbour imputation fitted on the development partition across all 1,463 proteins and applied to both partitions; median imputation reported as a sensitivity analysis. |
| 10a | D | Describe how predictors were handled in the analysis. | Methods §Panel development. NPX values used on their native log2 scale, standardised to zero mean and unit variance using development statistics inside every model fit; no categorisation, no transformation. |
| 10b | D | Specify type of model, all model-building procedures (including any predictor selection), and method for internal validation. | Methods §Panel development, paragraphs 1–4. Bonferroni-filtered Mann–Whitney screen; 180-protein candidate bank; ten rankings; 3,000 candidate panels over sizes 12–30; five-fold cross-validated composite objective in the development partition; L2 multinomial logistic regression (balanced, C = 0.1) for the multiclass panel and L1-ranked, L2-refitted logistic regression for the 12 disease-specific panels. |
| 10c | V | For validation, describe how the predictions were calculated. | Methods §External marker-set evaluation. Coefficients were re-estimated inside each cohort over 50 repeated participant-grouped splits; predictions are the mean held-out probability per participant. Discovery coefficients were not transported — see "Study type under TRIPOD" above. |
| 10d | D;V | Specify all measures used to assess model performance and, if relevant, to compare multiple models. | Methods §Internal evaluation on held-out patients; Methods §External marker-set evaluation. Discrimination (one-versus-rest AUC per class and macro, accuracy, balanced accuracy, macro F1, minimum class recall, sensitivity, specificity), calibration (top-label and classwise ECE, Brier score, reliability curves) and net benefit. |
| 10e | V | Describe any model updating arising from the validation, if done. | Methods §External marker-set evaluation. Coefficients were refitted per cohort by design; no updated model is proposed for use. |
| 11 | D;V | Provide details on how risk groups were created, if done. | **Not applicable.** No risk groups were formed. Thresholds (0.50 screening, 0.80 confirmation) define calls, not risk strata. |
| 12 | V | For validation, identify any differences from the development data in setting, eligibility criteria, outcome, and predictors. | Methods §External marker-set evaluation; Table 4; Discussion §Limitations. Differences are substantial and are stated: tissue rather than plasma, transcript rather than protein for 11 of 12 endpoints, non-malignant controls rather than other cancers, case-enriched designs, incomplete marker coverage for four endpoints. |

## Results

| Item | Type | Checklist item | Reported in |
|---|---|---|---|
| 13a | D;V | Describe the flow of participants, including the number of participants with and without the outcome and, if applicable, a summary of the follow-up time. A diagram may be helpful. | Methods §Study cohort and design; Table 1; Figure 1B; Figure 2A. 1,375 patients, 962 development and 413 held out, per-group counts in Table 1. |
| 13b | D;V | Describe the characteristics of the participants (basic demographics, clinical features, available predictors), including the number of participants with missing data for predictors and outcome. | Table 1 and Figure 2 give group sizes and the NPX distributions. **Partly reported:** age and sex distributions per group are not tabulated; the discovery resource's own publication carries them. Missingness is summarised in Methods §Panel development rather than per predictor. |
| 13c | V | For validation, show a comparison with the development data of the distribution of important variables (demographics, predictors and outcome). | **Not reported, and not meaningful as posed.** The external cohorts measure a different analyte class on different platforms, so predictor distributions are not comparable to discovery NPX. Case and control counts and marker coverage are given per endpoint in Table 4. |
| 14a | D | Specify the number of participants and outcome events in each analysis. | Table 1; Methods §Study cohort and design; Table 4 for each external endpoint |
| 14b | D | If done, report the unadjusted association between each candidate predictor and outcome. | Figure 3A and 3B (per-cancer volcano plots and counts of significantly increased and decreased proteins, from the unadjusted Mann–Whitney statistics with adjusted significance marked). Full per-protein tables accompany the code release. |
| 15a | D | Present the full prediction model to allow predictions for individuals (i.e. all regression coefficients, and model intercept or baseline survival at a given time point). | **Partly reported.** The 25 panel members are named in Figure 3D and Figure 4A and the estimator is fully specified in Methods. The fitted coefficient matrix and intercepts are deposited with the code release rather than printed, being a 25 × 12 matrix. Disease-specific panel membership is given in Figure 4C. |
| 15b | D | Explain how to use the prediction model. | Methods §Panel development and §Internal evaluation; Discussion §Two operating points. Inputs are standardised NPX for the panel members; output is a probability per cancer group; the call is the maximum-probability class for the multiclass model and a fixed threshold for each disease-specific panel. |
| 16 | D;V | Report performance measures (with CIs) for the prediction model. | Results §The 25-protein model…; Results §Disease-specific models…; Results §Predefined markers retained discrimination…; Figures 5, 6, 7, 8; Tables 2 and 4. Bootstrap intervals for internal multiclass metrics, Wilson intervals for external proportions. |
| 17 | V | If done, report the results from any model updating (i.e. model specification, model performance). | Methods and Results §Predefined markers retained discrimination after external refitting. Refitting is the design; no updated coefficients are proposed for use. |

## Discussion

| Item | Type | Checklist item | Reported in |
|---|---|---|---|
| 18 | D;V | Discuss any limitations of the study (such as non-representative sample, few events per predictor, missing data). | Discussion §Limitations. Cancer-only cohort, held-out data informing panel size, non-independent re-evaluation of the disease-specific panels, transcript proxies, case-enriched external cohorts, shared AML/CLL controls, single split, BRC compaction. |
| 19a | V | For validation, discuss the results with reference to performance in the development data, and any other validation data. | Discussion §Compact panels…; Discussion §Two operating points…; Results §Predefined markers… (internal and external figures reported side by side) |
| 19b | D;V | Give an overall interpretation of the results, considering objectives, limitations, results from similar studies, and other relevant evidence. | Discussion, throughout; Table 2 (published size series on the same cohort) and Table 3 (other tissue-of-origin systems) |
| 20 | D;V | Discuss the potential clinical use of the model and implications for future research. | Discussion §Two operating points…; Discussion §Next steps |

## Other information

| Item | Type | Checklist item | Reported in |
|---|---|---|---|
| 21 | D;V | Provide information about the availability of supplementary resources, such as study protocol, web calculator, and data sets. | Declarations §Data availability; §Code availability; this checklist; Supplementary Table S2 (GWAS sources); Supplementary computational-environment specification. **No prespecified protocol exists** — the analysis was not registered, and the manuscript does not claim prespecification except for the items explicitly named as fixed in advance (split ratio, seed, decision thresholds, HEIDI threshold). |
| 22 | D;V | Give the source of funding and the role of the funders for the present study. | Declarations §Funding — **to be completed by the authors before submission.** |

---

## Summary of items not fully met

| Item | Status | Reason |
|---|---|---|
| 4b | Not reported | Accrual dates belong to the source cohort publication; no follow-up analysed. |
| 5c | Not applicable | No treatment data. |
| 6b, 7b | Not applicable | Secondary analysis of pre-existing labels and assay readouts. |
| 11 | Not applicable | No risk groups formed. |
| 13b | Partly reported | Per-group demographics not tabulated. |
| 13c | Not reported | Predictor distributions are not comparable across platforms. |
| 15a | Partly reported | Coefficient matrix deposited rather than printed. |
| 21 | Reported with a caveat | No registered protocol. |
| 22 | Outstanding | Funding statement to be completed. |
