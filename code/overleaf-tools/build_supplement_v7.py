"""Build the v7 Supporting Information from the v7 result files.

Writes into overleaf-source-git/supplement/:
  - CSV tables S1-S9, S11-S18, S20-S28, S30 (from v7_plan/results and the upstream files they were built from);
  - Table_S10_external_datasets.md (generated from the B6 metrics);
  - Appendix_S1_tables.md (Tables A1-A9, generated);
  - figures/FigS1-S5 (copies of v7_plan/figures/output/supplement);
  - Supporting_Information.md and .tex (pandoc), Supporting_Tables_S1-S30.zip.
Static sources kept by hand in the same folder: _SI_front.md, Appendix_S1_methods.md, Table_S19_gwas_sources.md,
Table_S29_single_cancer_pea_panels.md, Appendix_S2_computational_environment.md, Appendix_S3_TRIPOD_AI_checklist.md.
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[1]))
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

R = PROJECT_ROOT
RES = R / "v7_plan/results"
RP = R / "revise_plan"
SUP = R / "overleaf-source-git/supplement"
ORDER = ["AML", "BRC", "CLL", "CRC", "CVX", "DLBCL", "ENDC", "GLIOM", "LUNGC", "MYEL", "OVC", "PRC"]
NAME = {"AML": "Acute myeloid leukaemia", "BRC": "Breast", "CLL": "Chronic lymphocytic leukaemia", "CRC": "Colorectal",
        "CVX": "Cervical", "DLBCL": "Diffuse large B-cell lymphoma", "ENDC": "Endometrial", "GLIOM": "Glioma",
        "LUNGC": "Lung", "MYEL": "Myeloma", "OVC": "Ovarian", "PRC": "Prostate"}
fix = lambda c: "DLBCL" if c == "LYMPH" else c
written = {}


def csv(name, df):
    df.to_csv(SUP / name, index=False)
    written[name] = len(df)


def md_table(df, align=None):
    cols = list(df.columns)
    al = align or ["---"] * len(cols)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(al) + "|"]
    rows += ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"
             for r in df.itertuples(index=False)]
    return "\n".join(rows)


def f3(x):
    return f"{x:.3f}"


def fx(df, nd=3):
    """Fixed decimals for float columns (keeps trailing zeros); integers unchanged."""
    return df.apply(lambda col: col.map(lambda v: f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else v))


# ---------------------------------------------------------------- panels and names
pan = pd.read_csv(R / "v7_plan/panels/panel_memberships_v7.csv")
names = pd.read_csv(RP / "v5_seed52_common_split/panels/protein_names_v5.csv").rename(columns={"gene_symbol": "protein"})
s1old = pd.read_csv(R / "overleaf-source-git/archive/supplement/v4.9_root/Table_S1_locked25_panel_membership.csv")
names = pd.concat([names, s1old[["protein", "uniprot", "protein_name"]]]).drop_duplicates("protein")
import zipfile as _z
hpa = pd.read_csv(_z.ZipFile(RES / "B9_biology/tissue_origin/hpa/proteinatlas.tsv.zip").open("proteinatlas.tsv"), sep="\t",
                  usecols=lambda c: c in ("Gene", "Gene description", "Uniprot"))
hpa = hpa.rename(columns={"Gene": "protein", "Uniprot": "uniprot", "Gene description": "protein_name"})
missing = sorted(set(pan.protein) - set(names.protein))
extra = hpa[hpa.protein.isin(missing)].drop_duplicates("protein").copy()
extra["protein_name"] = extra["protein_name"] + " †"
names = pd.concat([names, extra]).drop_duplicates("protein").set_index("protein")
assert set(pan.protein) <= set(names.index), set(pan.protein) - set(names.index)

mc = pan[pan.panel == "multiclass_26"].protein.tolist()
csv("Table_S1_multiclass26_panel_membership.csv",
    pd.DataFrame({"protein": mc, "uniprot": [names.loc[p, "uniprot"] for p in mc],
                  "protein_name": [names.loc[p, "protein_name"] for p in mc]}))
ds = pan[pan.panel.str.startswith("single_") & (pan.panel != "single_pooled_unique")].copy()
ds["cancer"] = ds.panel.str.replace("single_", "")
ds["rank"] = ds.groupby("cancer").cumcount() + 1
ds["in_multiclass_26"] = ds.protein.isin(mc)
ds["uniprot"] = [names.loc[p, "uniprot"] for p in ds.protein]
ds["protein_name"] = [names.loc[p, "protein_name"] for p in ds.protein]
ds["cancer"] = pd.Categorical(ds.cancer, ORDER)
csv("Table_S4_disease_specific_panel_membership.csv",
    ds.sort_values(["cancer", "rank"])[["cancer", "rank", "protein", "uniprot", "protein_name", "in_multiclass_26"]])

# ---------------------------------------------------------------- selection
shutil.copy(R / "overleaf-source-git/archive/supplement/v4.9_root/Table_S3_candidate_bank_180.csv", SUP / "Table_S2_candidate_bank_180.csv")
written["Table_S2_candidate_bank_180.csv"] = len(pd.read_csv(SUP / "Table_S2_candidate_bank_180.csv"))
cv = pd.read_csv(RP / "part_b_multiclass/v11_seed52_lr_l2_wide_compact_sensitivity/tables/stage1_cv_results.csv")
cv = cv.sort_values("stage1_objective", ascending=False)
cv.insert(0, "rank_by_objective", range(1, len(cv) + 1))
cv["chosen"] = cv.candidate_id == 2047
csv("Table_S3_multiclass_candidates_development_cv.csv", cv)
sw = pd.read_csv(RP / "pilot_ds_dev_only_size/size_sweep_devonly.csv")
sw["cancer"] = sw.cancer.map(fix)
csv("Table_S5_disease_specific_size_sweep_development_cv.csv", sw)

# ---------------------------------------------------------------- internal multiclass
b5 = json.load(open(RES / "B5_internal_two_step_summary.json"))
m = b5["multiclass"]["metrics"]
pp = pd.read_csv(RES / "B5_internal_per_patient.csv")
pp["true"], pp["pred"] = pp["true"].map(fix), pp["pred"].map(fix)
per = pd.read_csv(RES / "B1_multiclass/B1_per_cancer.csv")
per.iloc[:, 0] = per.iloc[:, 0].map(fix)
csv("Table_S6_multiclass26_per_cancer.csv", per)
ci = b5["multiclass"]["ci"]
csv("Table_S7_multiclass26_bootstrap_intervals.csv",
    pd.DataFrame([{"metric": k, "estimate": m[k], "ci_low": ci[k][0], "ci_high": ci[k][1]} for k in ci]))
cm = pd.crosstab(pd.Categorical(pp.true, ORDER), pd.Categorical(pp.pred, ORDER), rownames=["true"], colnames=["predicted"], dropna=False)
csv("Table_S8_multiclass26_confusion_matrix.csv", cm.reset_index())
b4 = pd.read_csv(RES / "B4_internal_disease_specific/internal_per_cancer_published_vs_devonly.csv")
b4 = b4[b4.version == "dev_only"].drop(columns="version")
b4["cancer"] = b4.cancer.map(fix)
csv("Table_S9_disease_specific_internal_metrics.csv", b4)

# ---------------------------------------------------------------- biology
B9 = RES / "B9_biology"
for src, dst in [("ora/ora_summary.csv", "Table_S11_ora_summary_42_tests.csv"),
                 ("ora/ora_terms_with_panel_overlap.csv", "Table_S12_ora_terms_with_panel_overlap.csv"),
                 ("ora/ora_background_isolation_2026-09-05.csv", "Table_S13_ora_background_isolation.csv"),
                 ("string/ppi_reaudit_combined_results_v2.csv", "Table_S14_string_ppi_enrichment.csv"),
                 ("hallmark/direction_matched_permutation_summary_v2.csv", "Table_S16_hallmark_permutation_summary.csv"),
                 ("hallmark/direction_matched_permutation_per_cancer_v2.csv", "Table_S17_hallmark_permutation_per_cancer.csv")]:
    csv(dst, pd.read_csv(B9 / src))
csv("Table_S18_panel_redundancy.csv", pd.read_csv(B9 / "redundancy/panel_redundancy_results_v2.csv"))
sys.path[:0] = [str(R / "v7_plan/figures/scripts")]
import figS_v7d_biology_annotation as BA  # noqa: E402
fm = BA.func_map()
pooled = sorted(set(ds.protein))
csv("Table_S12b_functional_categories.csv",
    pd.DataFrame({"protein": pooled, "category": [fm.get(p, "") for p in pooled],
                  "panels": [";".join(sorted(ds[ds.protein == p].cancer.astype(str))) for p in pooled]}))
to = B9 / "tissue_origin"
t28 = pd.read_csv(to / "B9b_per_cancer.csv"); t28["cancer"] = t28.cancer.map(fix)
csv("Table_S28_tissue_of_origin_per_cancer.csv", t28)
e = json.load(open(to / "B9e_summary.json"))
csv("Table_S28b_tissue_of_origin_nulls.csv",
    pd.DataFrame([{"pool": k, "observed": e["observed_total"], "null_mean": v["mean"], "null_2.5%": v["q2.5_q97.5"][0],
                   "null_97.5%": v["q2.5_q97.5"][1], "null_max": v["max"], "p_empirical": v["p_empirical"]}
                  for k, v in e.items() if isinstance(v, dict) and "mean" in v]))

# ---------------------------------------------------------------- genetics
G = RES / "B10_genetics_final"
csv("Table_S20_pqtl_gwas_screen_332_pairs.csv", pd.read_csv(G / "tier1_combined_v7_final.csv"))
old = R / "overleaf-source-git/archive/supplement/v4.9_root"
for n in ["Table_S21_smr_heidi_official_results.csv", "Table_S22_coloc_prior_sensitivity.csv", "Table_S23_genetics_evidence_master.csv"]:
    shutil.copy(old / n, SUP / n); written[n] = len(pd.read_csv(SUP / n))
NC = RES / "B10_genetics_ncorrected"
ms = pd.concat([pd.read_csv(NC / "B10i_susie_coloc_ukbld_ncorr.csv").assign(ld_reference="UK Biobank (GRCh37)"),
                pd.read_csv(NC / "B10j_susie_coloc_1kg38_ncorr.csv").assign(ld_reference="1000 Genomes high coverage (GRCh38)")])
csv("Table_S27_multisignal_colocalisation.csv", ms)

# ---------------------------------------------------------------- tissue two-step, mixed cancer
TB = RES / "B6_external/two_step_binary"
t24 = pd.read_csv(TB / "binary_rule_metrics.csv"); t24["cancer"] = t24.cancer.map(fix)
t24["rule"] = t24.rule.map({"step1": "26-protein list (0.50)", "step2": "disease-specific list (0.80)", "two_step": "both positive"})
csv("Table_S24_tissue_two_step_binary_rule.csv", t24)
t24b = pd.read_csv(TB / "binary_rule_paired_comparisons.csv"); t24b["cancer"] = t24b.cancer.map(fix)
csv("Table_S24b_tissue_net_benefit_differences.csv", t24b)
M8 = RES / "B8_mixed_cancer"
csv("Table_S25_mixed_cancer_all_series.csv", pd.read_csv(M8 / "B8_all_series_v7b.csv"))
shutil.copy(M8 / "B8_family_summary_v7b.json", SUP / "Table_S25b_mixed_cancer_family_summary.json")
csv("Table_S26_class_matched_comparator.csv", pd.read_csv(M8 / "B8_class_matched_comparator.csv"))

# ---------------------------------------------------------------- manifest
V = "v7_plan/results/"
man = [
    ("Figure 1", "panels/panel_memberships_v7.csv", "figures/scripts/fig1_v7b_methods.py"),
    ("Figure 2", "data/processed/filtered_pancancer_data.csv", "figures/scripts/fig2_v7d_cohort.py"),
    ("Figure 3; Tables S3, S5", "revise_plan/part_b_multiclass/v11_seed52_lr_l2_wide_compact_sensitivity/tables/stage1_cv_results.csv; revise_plan/pilot_ds_dev_only_size/size_sweep_devonly.csv", "revise_plan/scripts/02_may_v11_seed52_lr_l2_wide_compact_sensitivity.py; revise_plan/scripts/91_oct_v65_ds_panels_dev_only_reselect_2026-10-16.py; figures/scripts/fig3_v7h_layout.py"),
    ("Figure 4; Tables S6-S9, A2, A4", V + "B1_multiclass/; " + V + "B4_internal_disease_specific/; " + V + "B3_published_comparison/", "v7_plan/scripts/B1_multiclass_internal_v7.py; revise_plan/scripts/92_oct_v65_ds_devonly_downstream_internal_tcga_2026-10-16.py; v7_plan/scripts/B3_published_comparison_v7.py; figures/scripts/fig4_v7l_internal.py"),
    ("Figure 5; Figure S3; Table S10", V + "B6_external/", "v7_plan/scripts/B6_single_cancer_external_v7.py; figures/scripts/fig5_v7g_tissue_mosaic13.py"),
    ("Figure 6; Table 2; Tables A5-A9, S24-S26; Figure S4", V + "B5_internal_*; " + V + "B6_external/two_step_binary/; " + V + "B7_tcga/; " + V + "B8_mixed_cancer/", "v7_plan/scripts/B5_internal_two_step_v7.py; B6b_binary_rule_external_v7.py; revise_plan/scripts/85, 87, 89, 92, 93; v7_plan/scripts/B8b-B8g; figures/scripts/fig6_v7m_two_step.py; fig9c_v7_mixed_forest.py"),
    ("Figure 7; Tables S11-S18, S12b", V + "B9_biology/", "v7_plan/scripts/B9_biology_v7.py; B9c, B9d; figures/scripts/fig8_v7j_biology_annotation.py"),
    ("Figure S5; Table S28", V + "B9_biology/tissue_origin/", "v7_plan/scripts/B9b_tissue_of_origin_v7.py; B9e_tissue_origin_selection_null_v7.py; figures/scripts/figS_v7_tissue_origin.py"),
    ("Figure 8; Tables S20-S23, S27", V + "B10_genetics_final/; " + V + "B10_genetics_ncorrected/", "v7_plan/scripts/B10_genetics_setup_v7.py -> B10b -> B10c; B10h_make_ncorrected_susie_v7.py (B10h-k); figures/scripts/fig7_v7n_genetics.py"),
    ("Table A3", V + "B2_sex_devsel/", "v7_plan/scripts/B2_sex_sensitivity_v7_devsel.py; B2b_sex_mapping_check_v7.py"),
    ("Table 1", "literature; " + V + "B5_internal_two_step_summary.json", "manuscript text"),
    ("Tables S1, S4", "panels/panel_memberships_v7.csv", "overleaf-tools/build_supplement_v7.py"),
]
csv("Table_S30_source_manifest.csv", pd.DataFrame(man, columns=["item", "source files", "scripts"]))

# ---------------------------------------------------------------- Table S10 (text)
mc6 = pd.read_csv(RES / "B6_external/B6_metrics_multiclass_26.csv").set_index("cancer")
ds6 = pd.read_csv(RES / "B6_external/B6_metrics_disease_specific.csv").set_index("cancer")
DSET = {"AML": "GSE13159 (Haferlach et al. 2010)", "BRC": "GSE42568 (Clarke et al. 2013)", "CLL": "GSE13159 (Haferlach et al. 2010)",
        "CRC": "GSE41258 (Sheffer et al. 2009)", "CVX": "GSE9750 (Scotto et al. 2008)", "ENDC": "GSE17025 ^a^",
        "GLIOM": "GSE4290 (Sun et al. 2006)", "LUNGC": "CPTAC–ICPC LUAD proteomics (Satpathy et al. 2025)",
        "DLBCL": "GSE32018 (Gómez-Abad et al. 2011)", "MYEL": "GSE6477 (Chng et al. 2007)", "OVC": "GSE18520 (Mok et al. 2009)",
        "PRC": "GSE17951 (Jia et al. 2011)"}
rows = [{"Endpoint": c, "Dataset": DSET[c], "Cases": int(mc6.loc[c, "cases"]), "Controls": int(mc6.loc[c, "controls"]),
         "DS cov.": ds6.loc[c, "markers_mapped"], "MC cov.": mc6.loc[c, "markers_mapped"],
         "DS AUC": f3(ds6.loc[c, "auc"]), "MC AUC": f3(mc6.loc[c, "auc"])} for c in ORDER]
s10 = ("# Table S10: Single-cancer tissue datasets and marker coverage\n\n"
       "Coverage is measured markers / intended markers. DS, disease-specific panel; MC, 26-protein multiclass list. "
       "Marker membership was frozen and an L2-penalised logistic model refitted within each dataset; no "
       "discovery-cohort scaler, coefficient or intercept was transported.\n\n"
       + md_table(pd.DataFrame(rows), ["---", "---", "---:", "---:", ":---:", ":---:", "---:", "---:"])
       + "\n\n^a^Gene Expression Omnibus accession GSE17025; no associated primary publication was identified.\n\n"
       "Splits were grouped by patient. In the LUNGC (794 specimens from 581 patients), CRC (240 from 191) and PRC (154 "
       "from 123) endpoints, several specimens came from one patient, and the published patient identifiers kept them in "
       "the same part of every split. Scores and counts are per specimen. The AML and CLL rows draw their cases and the "
       "same control group from one dataset (GSE13159) and are not independent of each other. Eleven endpoints use "
       "gene-expression measurements, predominantly tumour-tissue microarrays; the lung endpoint uses CPTAC–ICPC lung "
       "adenocarcinoma proteomics. The comparison group is non-malignant tissue at every endpoint.\n")
(SUP / "Table_S10_external_datasets.md").write_text(s10, encoding="utf-8")

# ---------------------------------------------------------------- Appendix S1 tables A1-A9
A = ["# Appendix S1: Cohort, appendix tables and supplementary methods\n",
     "Tables A1 to A9 give the cohort and per-cancer results behind the main text; sections S1.1 to S1.13 specify the "
     "panel search, the size rule and the implementation of the external, sequential, genetic and tissue-of-origin "
     "analyses.\n"]
a1 = (R / "overleaf-source-git/archive/supplement/v6.1/Appendix_S1_panel_search.md").read_text(encoding="utf-8")
a1 = a1[a1.index("| Code | Cancer group"):a1.index("\\clearpage", a1.index("| Code | Cancer group"))].strip()
a1 = a1.replace("| Development | Held-out |", "| Development | Internal validation |")
A += ["## Table A1. Discovery cohort by cancer group\n", a1, "\n"]
b3m = pd.read_csv(RES / "B3_published_comparison/B3_multiclass_vs_published.csv")
A += ["## Table A2. Multiclass models of the U-CAN resource and the present panel\n",
      md_table(fx(b3m).rename(columns={"panel": "Model", "macro_auc": "Macro AUC", "macro_f1": "Macro F1",
                                            "macro_recall": "Macro recall", "min_class_recall": "Min. class recall"})),
      "\nPublished values from Bueno Álvez et al. 2023 (test set of 30% of 1,477 patients); present panel in the internal "
      "validation set (30% of 1,375). The splits differ, so the comparison is descriptive.\n"]
sx = json.load(open(RES / "B2_sex_devsel/sex_sensitivity_results.json"))
rows = [("All patients (26 proteins)", sx["A_baseline"]["accuracy"], sx["A_baseline"]["macro_auc"]),
        ("Sex as the only predictor", sx["B_sex_only"]["accuracy"], sx["B_sex_only"]["macro_auc"])]
for k, lab in (("Female", "Women"), ("Male", "Men")):
    rows.append((f"{lab} (n = {sx['C_within_sex'][k]['n_test']})", sx["C_within_sex"][k]["accuracy"], np.nan))
dm = sx["E_drop_sex_associated"]
rows.append(("Without " + ", ".join(dm["dropped"]), dm["accuracy"], dm["macro_auc"]))
A += ["## Table A3. Sex analysis of the 26-protein model in the internal validation set\n",
      md_table(fx(pd.DataFrame(rows, columns=["Analysis", "Accuracy", "Macro AUC"]))),
      "\nThe three sex-associated proteins were identified in the development set. Five cancers occur in one sex only.\n"]
b3d = pd.read_csv(RES / "B3_published_comparison/B3_disease_specific_vs_published.csv")
b3d["Scored"] = np.where(b3d.cancer.isin(["BRC", "CVX", "ENDC", "OVC", "PRC"]), "same sex", "all")
b3d["cancer"] = b3d.cancer.map(fix)
A += ["## Table A4. Disease-specific AUC and the comparable published model\n",
      md_table(fx(b3d).rename(columns={"cancer": "Cancer", "panel_size": "Proteins", "auc": "AUC", "published_model_size": "Published model proteins",
                                   "published_auc": "Published AUC", "verdict": "Present panel"})),
      "\nEach panel is compared with the smallest published model with at least as many proteins; differences below 0.005 "
      "count as equal.\n"]
rows = []
for c in ORDER:
    sub = pp[pp.pred == c]
    r = {"Predicted cancer": c, "Predictions": len(sub), "Patients with the cancer": int((pp.true == c).sum())}
    for thr in (0.5, 0.8):
        acc = sub[sub.second >= thr]
        r[f"Accepted at {thr:.2f}"] = len(acc)
        r[f"Correct among accepted at {thr:.2f}"] = int((acc.true == c).sum())
    rows.append(r)
a5 = pd.DataFrame(rows)
tot = {k: (a5[k].sum() if k != "Predicted cancer" else "Total") for k in a5.columns}
a5 = pd.concat([a5, pd.DataFrame([tot])])
A += ["## Table A5. Two-step rule in the internal validation set, by predicted cancer\n", md_table(a5),
      f"\nStep 2 at 0.50 left {b5['two_step_0.5']['undecided']} predictions undecided; for "
      f"{b5['two_step_0.5']['undecided_true_in_top3']} of them the correct cancer was among the three most probable. "
      f"At 0.80, {b5['two_step_0.8']['undecided']} were undecided and {b5['two_step_0.8']['undecided_true_in_top3']} "
      "had the correct cancer among the three most probable.\n"]
nb = pd.read_csv(RES / "B5_net_benefit_per_cancer.csv"); nb["cancer"] = nb.cancer.map(fix)
nbt = nb[["cancer", "cases", "tp_lost", "fp_removed", "nb_diff_0.5", "nb_lo_0.5", "nb_hi_0.5", "nb_diff_0.8", "nb_lo_0.8", "nb_hi_0.8"]].copy()
A += ["## Table A6. Adding step 2 to step 1 in the internal validation set: per-cancer trade-off\n",
      md_table(fx(nbt, 2).rename(columns={"cancer": "Cancer", "cases": "Patients", "tp_lost": "Correct calls lost",
                                            "fp_removed": "Wrong calls removed", "nb_diff_0.5": "Net benefit change at 0.50",
                                            "nb_lo_0.5": "low", "nb_hi_0.5": "high", "nb_diff_0.8": "Net benefit change at 0.80",
                                            "nb_lo_0.8": "low ", "nb_hi_0.8": "high "})),
      "\nNet benefit changes are net true positives per 100 patients with 95% bootstrap intervals; Table S24b gives the "
      "same comparison in the single-cancer datasets.\n"]
tc = pd.read_csv(RES / "B7_tcga/per_patient_tcga_xena_dev_only.csv")
rows = []
for c in sorted(tc.true.unique(), key=lambda x: ORDER.index(fix(x)) if fix(x) in ORDER else 99):
    sub = tc[tc.pred_26 == c]; acc = sub[sub.second_26 >= 0.5]
    rows.append({"Predicted cancer": fix(c), "Predictions": len(sub), "Accepted": len(acc),
                 "Correct among accepted": int((acc.true == c).sum()), "Patients with the cancer": int((tc.true == c).sum())})
A += ["## Table A7. Two-step rule in TCGA, by predicted cancer\n", md_table(pd.DataFrame(rows)),
      "\nTCGA has no CLL or myeloma. Disease-specific models were trained without class weights.\n"]
s8 = pd.read_csv(M8 / "B8_all_series_v7b.csv"); cmc = pd.read_csv(M8 / "B8_class_matched_comparator.csv").set_index("id")
rows = [{"Setting": "Internal validation", "Accepted": b5["two_step_0.5"]["accepted"],
         "Wrong accepted, stricter cutoff": b5["comparison_0.5"]["wrong_stricter"],
         "Wrong accepted, step 2": b5["comparison_0.5"]["wrong_step2"], "p": f"{b5['comparison_0.5']['p_two_sided']:.2f}",
         "Class-matched cutoff": int(cmc.loc["internal", "wrong_class_matched_cutoff"]), "p (class-matched)": cmc.loc["internal", "p"]}]
LAB = {"TCGA_xena": "TCGA", "D1": "expO", "D7": "Other GDC cohorts", "D5": "CCLE"}
for _, r in s8[s8.id.isin(LAB)].iterrows():
    rows.append({"Setting": LAB[r.id], "Accepted": int(round(r.accepted_share * r.n)), "Wrong accepted, stricter cutoff": int(r.wrong_stricter),
                 "Wrong accepted, step 2": int(r.wrong_step2), "p": r.p_report,
                 "Class-matched cutoff": int(cmc.loc[r.id, "wrong_class_matched_cutoff"]), "p (class-matched)": cmc.loc[r.id, "p"]})
A += ["## Table A8. Step 2 against the stricter multiclass cutoff\n", md_table(pd.DataFrame(rows)),
      "\nThe global cutoff keeps the same number of predictions with the highest multiclass probability; the class-matched "
      "cutoff keeps the same number for each predicted cancer. Paired bootstrap p-values (2,000 resamples). All series "
      "are in Tables S25 and S26.\n"]
w = t24.pivot_table(index="cancer", columns="rule", values=["sensitivity", "specificity"])
w.columns = [f"{a} {b}" for a, b in w.columns]
w = w.reindex(ORDER).reset_index().rename(columns={"cancer": "Endpoint"})
A += ["## Table A9. Single-cancer datasets: each binary model and both positive\n", md_table(fx(w)),
      "\nPer-specimen sensitivity and specificity; counts and Wilson intervals in Table S24.\n"]
(SUP / "Appendix_S1_tables.md").write_text("\n".join(A), encoding="utf-8")

# ---------------------------------------------------------------- figures, assembly, pandoc, zip
fd = SUP / "figures"; fd.mkdir(exist_ok=True)
FIGS = {"FigS1_internal_multiclass_full.png": "FigS_internal_multiclass_full.png",
        "FigS2_internal_disease_specific_full.png": "FigS_internal_disease_specific_full.png",
        "FigS3_tissue_multiclass26_full.png": "FigS_tissue_multiclass26_full.png",
        "FigS4_mixed_cancer_forest.png": "FigS_mixed_cancer_forest.png",
        "FigS5_tissue_of_origin.png": "FigS_tissue_of_origin.png"}
for dst, src in FIGS.items():
    shutil.copy2(R / "v7_plan/figures/output/supplement" / src, fd / dst)
parts = ["_SI_front.md", "Table_S10_external_datasets.md", "Table_S19_gwas_sources.md", "Table_S29_single_cancer_pea_panels.md",
         "Appendix_S1_tables.md", "Appendix_S1_methods.md", "Appendix_S2_computational_environment.md",
         "Appendix_S3_TRIPOD_AI_checklist.md"]
text = "\n\n\\clearpage\n\n".join((SUP / p).read_text(encoding="utf-8").rstrip() for p in parts)
(SUP / "Supporting_Information.md").write_text(text + "\n", encoding="utf-8")
subprocess.run(["pandoc", "Supporting_Information.md", "-f", "markdown+raw_tex", "-t", "latex", "-s",
                "--include-in-header", "_SI_header.tex", "-V", "geometry:margin=1in", "-V", "fontsize:11pt",
                "-o", "Supporting_Information.tex"], cwd=SUP, check=True)
tex = (SUP / "Supporting_Information.tex").read_text(encoding="utf-8")
tex = tex.replace(r"\_", r"\_\allowbreak{}").replace("≤", r"$\le$").replace("≥", r"$\ge$")
(SUP / "Supporting_Information.tex").write_text(tex, encoding="utf-8")
with zipfile.ZipFile(SUP / "Supporting_Tables_S1-S30.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(SUP.glob("Table_S*.csv")) + [SUP / "Table_S25b_mixed_cancer_family_summary.json"]:
        z.write(f, f.name)
for k, v in sorted(written.items()):
    print(f"{v:6d}  {k}")
print("wrote Supporting_Information.tex and the zip")
