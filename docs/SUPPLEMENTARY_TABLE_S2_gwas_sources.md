# Supplementary Table S2: genetic summary-statistic sources

**Manuscript:** A 25-protein plasma panel discriminates 12 cancer types, and a
paired disease-specific panel set confirms them
**Compiled:** 2026-09-12
**Source of record:** `revise_plan/smr_coloc/28_jun_11cancers_smr_coloc.md`,
`revise_plan/smr_coloc/download_gwas.sh`,
`revise_plan/smr_coloc/official_smr_heidi/provenance/source_data_manifest.csv`

---

## S2a. Protein quantitative trait loci

| Source | Release | Participants | Ancestry | Build |
|---|---|---|---|---|
| UK Biobank Pharma Proteomics Project (UKB-PPP), Sun et al. 2023 | Public release | 34,557 | European | GRCh38 |

One *cis*/*trans* lead variant was taken per protein as the top variant by
absolute pQTL *z* statistic within a ±500 kb window.

## S2b. Cancer GWAS summary statistics

Eleven of the twelve cancer groups had a usable GWAS. All eleven were analysed
against both panel systems.

| Cancer | Code | Accession / release | Study | Cases | Total *n* | Column format |
|---|---|---|---|---|---|---|
| Breast | BRC | GCST004988 | Michailidou et al. 2017 | 76,192 † | 139,274 † | harmonised `hm_` |
| Colorectal | CRC | GCST90255675 | Fernandez-Rozadilla et al. 2023 (European-only) | 78,473 | 185,616 | `chromosome` / `base_pair_location` / `beta` |
| Lung | LUNGC | GCST004748 | McKay et al. 2017 | 29,266 | 85,716 | harmonised `hm_` |
| Prostate | PRC | GCST006085 | Schumacher et al. 2018 | 79,148 | 140,254 | harmonised `hm_` |
| Ovarian | OVC | GCST90455661 | Barnes et al. 2025 | 19,883 | 398,238 | `chromosome` / `base_pair_location` / `beta` |
| Endometrial | ENDC | GCST90454186 | Ramachandran et al. 2025 | 17,278 | 306,458 | `chromosome` / `base_pair_location` / `beta` |
| Myeloma | MYEL | GCST90624747 | Güler et al. 2025 (UKB + FinnGen) | 3,991 | 1,109,399 | `chromosome` / `base_pair_location` / `beta` |
| Diffuse large B-cell lymphoma | DLBCL | GCST90624739 | Güler et al. 2025 | 2,280 | 658,535 | `chromosome` / `base_pair_location` / `beta` |
| Chronic lymphocytic leukaemia | CLL | GCST90624738 | Güler et al. 2025 | 4,442 | 1,108,777 | `chromosome` / `base_pair_location` / `beta` |
| Acute myeloid leukaemia | AML | GCST90707271 | Ranasinghe et al. 2026 | 4,710 | 17,648 | `chromosome` / `base_pair_location` / `beta` |
| Glioma ‡ | GLIOM | FinnGen R12, endpoint `C3_BRAIN_EXALLC` | FinnGen | 1,894 | 380,643 | `#chrom` / `pos` / `beta` / `sebeta`, effect allele = ALT |

All GWAS Catalog files were retrieved from the EBI summary-statistics FTP
service in their harmonised GRCh38 form. Harmonisation codes 5, 10 and 11 were
retained and code 6 (ambiguous palindromic) was excluded.

**†** The breast-cancer case and control counts differ between two internal
records: 76,192 / 139,274 in the 11-cancer analysis report, and 122,977 /
228,951 in the analysis configuration file. Verify against GWAS Catalog
GCST004988 and reconcile before submission. The accession itself is consistent
across every record and is the quantity that determined which file was analysed.

**‡** Glioma is exploratory. The best-powered study (Melin et al. 2017,
GCST004347, 12,469 cases) is under EGA controlled access and no open-access
alternative of comparable size exists, so FinnGen R12 was used with its case
count stated wherever a glioma result appears.

## S2c. Cancer groups without a usable GWAS

| Cancer | Code | Reason for exclusion |
|---|---|---|
| Cervical | CVX | Koel et al. 2023 (GCST90246358 / GCST90246359) reports MR-MEGA meta-regression output and supplies no pooled effect estimate, which the Wald ratio requires. FinnGen R12 `C3_CERVIX_UTERI` carries roughly 470 cases, too few for this analysis. |

## S2d. Linkage-disequilibrium reference

| Source | Subset | Build | Tool |
|---|---|---|---|
| 1000 Genomes Project, high-coverage NYGC 30× callset (Byrska-Bishop et al. 2022) | 503 European participants | GRCh38 | PLINK 2 (Chang et al. 2015) |

## S2e. Pair counts

| Panel system | Proteins | Cancers | Intended pairs | Analysable pairs |
|---|---|---|---|---|
| Locked 25-protein multiclass panel | 25 | 11 | 275 | 257 |
| 12 disease-specific panels | 51 distinct | 11 | 59 | 59 |
| **Total** | **76 distinct** | **11** | **334** | **316** |

The 18 unanalysable pairs all had pQTL lead variants on chromosome X, for which
the relevant GWAS supplied no compatible results. The Bonferroni threshold for
the association screen was 0.05 / 316 = 1.58 × 10⁻⁴.
