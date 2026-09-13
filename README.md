# Pan-cancer plasma protein panels (12 cancers)

Code and data release for:

> **A 25-protein plasma panel discriminates 12 cancer types, and a paired
> disease-specific panel set confirms them.**
> Manuscript in preparation for *The FASEB Journal*.

Assigning which cancer a patient has is a different problem from detecting
that a cancer is present. This project asks how small a plasma protein panel
can be and still make that assignment among patients already known to have
cancer, using Olink Explore 1536 measurements from 1,375 patients across 12
cancers. A staged search over 3,000 candidate panels returned a locked
**25-protein multiclass panel** (one-vs-rest AUC 0.956, minimum class recall
0.619 on held-out patients); 12 one-vs-rest workflows separately returned
**disease-specific panels** of 2–13 proteins each (mean specificity 0.985 at a
fixed threshold). With marker membership frozen and coefficients refitted per
cohort, both panels retained discrimination in 12 independent external
cohorts (macro AUC 0.963 and 0.946 respectively). Pathway, protein–protein
interaction, and pQTL–GWAS analyses then ask whether panel membership carries
biological structure beyond the statistics that selected it.

This repository holds the analysis code, the processed discovery-cohort
matrix, and the documentation needed to run it. It does not include the
manuscript text or high-resolution figures — those stay with the authors
until the paper is accepted.

## Citation

A citation entry will be added on submission/acceptance (see `CITATION.cff`).
If you use this code before then, please cite the repository URL and note
the manuscript's working title above.

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

Scripts in `src/genetics/` set `.libPaths()` to this folder themselves, so no
further configuration is needed. External binaries:
[SMR 1.3.1](https://yanglab.westlake.edu.cn/software/smr/) and
[PLINK 2](https://www.cog-genomics.org/plink/2.0/).

Full pinned versions for both languages: `docs/ENVIRONMENT.md`.

## Data

`data/filtered_pancancer_data.csv` (15 MB, checksummed) is the discovery
cohort. External validation cohorts, GWAS summary statistics, pQTLs, and
pathway/interaction resources are public but not redistributed here — see
`data/README.md` for accessions and download links.

## Running the pipeline

```powershell
# Disease-specific panels (12 one-vs-rest models)
python src/panels/run_pan_cancer_loop.py --seed 52

# 25-protein multiclass panel (locked reproducer)
python src/panels/02_may_vRSX_seed52_locked_25_lr_l2_reproducer.py
```

Reproduction follows the tolerance in `docs/ENVIRONMENT.md`: exact matches
for feature selections, partitions, and discrete predictions; agreement
within 1e-6 for continuous metrics, given the pinned environment there.

## Licence

Code: MIT (see `LICENSE`). The discovery-cohort data is CC BY 4.0 from its
original publication — see `data/README.md`.
