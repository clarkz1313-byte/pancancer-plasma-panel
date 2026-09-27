# Pan-cancer plasma protein panels (12 cancers)

Code and data release for:

> **A 25-protein plasma panel discriminates 12 cancer types with complementary
> disease-specific marker sets.**

Assigning which cancer a patient has is a different problem from detecting
that a cancer is present. This project asks how small a plasma protein panel
can be and still make that assignment among patients already known to have
cancer, using Olink Explore 1536 measurements from 1,375 patients across 12
cancers. A staged search over 3,000 candidate panels returned a locked
**25-protein multiclass panel** (one-vs-rest AUC 0.956, minimum class recall
0.619 on held-out patients); 12 one-vs-rest workflows separately returned
**disease-specific panels** of 2–13 proteins each (mean specificity 0.985 at a
fixed threshold). With marker membership frozen and coefficients refitted per
cohort, both panels retained discrimination across 12 endpoints from 11
external datasets (macro AUC 0.963 and 0.946 respectively). Pathway, protein–protein
interaction, and pQTL–GWAS analyses then ask whether panel membership carries
biological structure beyond the statistics that selected it.

This repository holds the analysis code and processed discovery-cohort matrix.
The internal locked model and disease-specific Figure 6 can be rebuilt from
this archive. External cohort and genetic analyses require the source datasets
listed in `data/README.md`; several additional figure builders require
intermediate result tables from the original analysis directory.

## Citation

The citation metadata for this software is in `CITATION.cff`. Cite the
version-specific Zenodo DOI for the code snapshot and the associated manuscript
separately when it is published. The first archived release, `v1.0.0`, is
[doi:10.5281/zenodo.22986391](https://doi.org/10.5281/zenodo.22986391).

## Repository layout

```
data/                    processed discovery matrix + download instructions for external data
src/panels/              derivation of both marker systems (Part A single-cancer, Part B multiclass)
src/external/            refitting frozen marker sets in independent cohorts
src/enrichment/          pathway over-representation, PPI enrichment, redundancy, Hallmark permutation
src/genetics/            pQTL–GWAS, colocalisation, SMR/HEIDI (R + supporting Python tooling)
src/figures/             figure-generation scripts
docs/                    environment spec, TRIPOD checklist, GWAS source table, panel-search methods
requirements.txt         pinned Python packages
```

## Environment

**Python 3.11.9**

```bash
python -m pip install -r requirements.txt
```

**R 4.4.2**, used by `src/genetics/` only (colocalisation, the TwoSampleMR/
ieugwasr cross-check, and driving the external SMR and PLINK 2 binaries).
Install into a project-local library:

```r
dir.create("r_libs")
install.packages(c("rlang", "coloc", "TwoSampleMR", "ieugwasr", "susieR",
                    "data.table", "dplyr", "ggplot2", "readr", "tidyr",
                    "ggrepel", "patchwork", "scales", "RColorBrewer"),
                  lib = "r_libs")
```

Several genetics scripts retain absolute paths from the original analysis
directory. Set those paths and the local R library for your environment before
running them. External binaries:
[SMR 1.3.1](https://yanglab.westlake.edu.cn/software/smr/) and
[PLINK 2](https://www.cog-genomics.org/plink/2.0/).

Full pinned versions for both languages: `docs/ENVIRONMENT.md`.

## Data

`data/filtered_pancancer_data.csv` (15 MB, checksummed) is the discovery
cohort. External validation cohorts, GWAS summary statistics, and pathway
resources are available separately; UKB-PPP pQTL data require approved access.
See `data/README.md` for accessions and links.

## Reproduction scope

| Analysis | What is in this archive | Reviewer action |
|---|---|---|
| Locked 25-protein model | Discovery matrix and reproducer | Run the second command below; it checks the selected proteins and held-out metrics. |
| Disease-specific models and Figure 6 | Discovery matrix, panel membership, Table S9 counts, figure source | Run the Figure 6 command below to check all 12 matrices and recreate the manuscript bitmap. The panel-search command is also below. |
| External marker-set refits | Marker membership and original refit script | Obtain and process each public tissue dataset in `data/README.md`, then supply sample-by-feature and metadata tables to `src/external/generic_locked_panel_external_validation.py`. The archive does not contain those processed matrices. |
| Genetic and functional analyses | Analysis scripts and source identifiers | Obtain the stated pQTL, GWAS, LD, and annotation resources; UKB-PPP requires approved access. Some scripts need original project paths changed. |
| Other manuscript figures | Figure-generation scripts | Several builders expect intermediate result tables that are not included in this archive. |

See `docs/SCRIPT_INDEX.md` for the primary entry points and the status of
figure builders in this release.

## Running the pipeline

```powershell
# Disease-specific panels (12 one-vs-rest models)
python src/panels/run_pan_cancer_loop.py --seed 52

# 25-protein multiclass panel (locked reproducer)
python src/panels/02_may_vRSX_seed52_locked_25_lr_l2_reproducer.py
```

`docs/ENVIRONMENT.md` gives the tested reproduction environment and explains
how to compare a rerun across machines. Its package pins are a practical
target, not proof that every historical result used an identical machine or
library build.

The `--panel-name` option on the external refit script selects
`multiclass_25` or a `single_<CANCER>` set from
`data/panel_memberships.csv`. Use the production script for the reported
external endpoints; the `_sklearn.py` script is an estimator sensitivity
analysis and can give different fixed-threshold calls.

### Disease-specific held-out figure

The corrected disease-specific figure can be rebuilt from the included
discovery matrix, frozen panel memberships, and Table S9 audit counts:

```bash
python src/figures/fig7_table_s9_refit.py
```

The script checks each of the 12 confusion matrices against
`data/Table_S9_disease_specific_panel_metrics.csv` and writes
`figures/fig6_table_s9_refit.png`. It reuses the original three-panel figure
template with AUC confidence intervals, Wilson intervals for sensitivity and
specificity, reliability curves, and decision curves.

## Licence

Code: MIT (see `LICENSE`). The discovery-cohort data is CC BY 4.0 from its
original publication — see `data/README.md`.
