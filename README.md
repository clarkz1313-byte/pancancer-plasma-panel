# Pan-cancer plasma protein panels (12 cancers)

Analysis code for the manuscript:

> **Compact plasma protein panels classify cancer type, and a disease-specific second step trades coverage for
> accuracy.** Manuscript in preparation.

Release v1.0.2 holds the code for the panels and analyses reported in the manuscript. Release v1.0.1 holds the code
of an earlier version of this work (a 25-protein panel) and stays available under its tag and Zenodo version.

## What the study did

Olink Explore 1536 measured 1,463 plasma proteins in 1,375 patients with 12 cancers (development set 962, internal
validation set 413). A 26-protein multiclass model names the cancer type (step 1), and the disease-specific model of
that cancer, with 2 to 12 proteins, accepts the prediction or leaves it undecided (step 2). In the internal
validation set, step 1 was correct for 74.6% of patients (macro AUC 0.957); step 2 accepted 56.7% of predictions,
and 90.2% of those were correct. The protein lists were then refitted in 12 single-cancer tissue endpoints and four
pan-cancer datasets (TCGA, expO, other GDC projects, CCLE), and examined with pathway, tissue-of-origin and genetic
analyses.

## Layout

```
code/                     a copy of the project folder with the scripts used for the manuscript
  data/processed/         the processed U-CAN plasma matrix (included, checksummed; see code/data/README.md)
  v7_plan/panels/         the locked multiclass and disease-specific panels
  revise_plan/scripts/    panel search, disease-specific panels, pan-cancer datasets
  v7_plan/scripts/        internal validation, single-cancer and pan-cancer analyses, biology, genetics (Python, R)
  v7_plan/figures/scripts/, figurev6/scripts/   figure builders and their shared helpers
  overleaf-tools/         supplement builder and its helpers
docs/MANIFEST_v1.0.2.md   which script produces each table and figure (Table S30), run order, full file list
docs/ENVIRONMENT.md       software versions and fixed analysis settings
requirements.txt          pinned Python packages
```

## Setup

Python 3.11.9:

```bash
python -m pip install -r requirements.txt
```

R 4.4.2 is needed for the genetic analyses only (coloc, susieR, TwoSampleMR, ieugwasr, dplyr, readr, data.table),
with the external binaries [SMR 1.3.1](https://yanglab.westlake.edu.cn/software/smr/) and
[PLINK 2](https://www.cog-genomics.org/plink/2.0/). The R scripts look for packages in `$PROTEOMICS_ROOT/.rlib`
first.

Two settings replace the fixed local paths of the original project:

| Setting | Meaning | Default |
|---|---|---|
| `PROTEOMICS_ROOT` | folder that holds `data/`, the script folders and the result folders | the `code/` folder |
| `SMR_DATA_ROOT` | GWAS, pQTL and LD reference files for the genetic analyses (SI Table S19) | `code/smr_data` |

Three figure builders (`fig1_v7_methods.py`, `fig1_v7b_methods.py`, `overleaf-tools/make_fig1_v52_methods.py`) and
`figurev6/scripts/fig11_panels.py` render SVG with headless Chrome at its default Windows location; edit that path
if Chrome is installed elsewhere.

## Running

Run the scripts in the order given in `docs/MANIFEST_v1.0.2.md`. Paths resolve from `PROTEOMICS_ROOT`, so the
working folder does not matter. For example, internal validation of the multiclass model:

```bash
python code/v7_plan/scripts/B1_multiclass_internal_v7.py
```

Each script writes its results under `code/` (or `PROTEOMICS_ROOT`) at the paths named in the manifest. The
intermediate results of the full project (about 10 GB) and the external datasets are not shipped: the external
datasets keep their own licences and are downloaded as described in `code/data/README.md`. The split seed is 52
throughout. Reproduction tolerances are given in `docs/ENVIRONMENT.md`.

## Citation

Please cite the manuscript and the archived code (Zenodo concept DOI
[10.5281/zenodo.22986768](https://doi.org/10.5281/zenodo.22986768)); see `CITATION.cff`.

## Licence

Code: MIT (see `LICENSE`). The processed U-CAN matrix is derived from BioStudies S-BSST935 (CC0); see
`code/data/README.md`.
