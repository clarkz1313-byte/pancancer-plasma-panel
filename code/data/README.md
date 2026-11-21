# Data

## `processed/filtered_pancancer_data.csv` (included, 15 MB)

1,375 patients, 12 cancer groups, 1,463 Olink Explore 1536 protein measurements (log2-scale normalized protein
expression, NPX), plus `Sample_ID`, `Cancer` and `protein_count`.

```
SHA-256: 7825207BD0502EB7C8C414E980DD9F015A96E9001491F1C500C417A819AA252F
```

Verify before running anything:

```powershell
Get-FileHash .\code\data\processed\filtered_pancancer_data.csv -Algorithm SHA256
```

Do not reorder rows, rename columns or change the missing-value encoding: the scripts assume this layout.

## Provenance and licence

The file is derived from the pan-cancer plasma proteomics resource of Bueno Álvez et al., *Nat Commun*
14:4308 (2023), [doi:10.1038/s41467-023-39765-y](https://doi.org/10.1038/s41467-023-39765-y). The source data are
deposited in BioStudies under accession [S-BSST935](https://www.ebi.ac.uk/biostudies/studies/S-BSST935) under CC0.
Only the cancer cohort is used; the source study's healthy (Wellness) cohort is not included.

## External datasets (not included)

None of the datasets below is redistributed here; each keeps its own licence and access route. Exact versions,
processing and sample counts are in the Supporting Information of the manuscript.

### Single-cancer datasets (tumor against non-malignant tissue; SI Table S10)

| Cancer | Accession | Repository |
|---|---|---|
| AML, CLL | GSE13159 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159) |
| BRC | GSE42568 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42568) |
| CRC | GSE41258 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE41258) |
| CVX | GSE9750 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE9750) |
| DLBCL | GSE32018 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32018) |
| ENDC | GSE17025 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE17025) |
| GLIOM | GSE4290 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE4290) |
| LUNGC | CPTAC–ICPC LUAD proteomics | [PDC](https://proteomic.datacommons.cancer.gov/) |
| MYEL | GSE6477 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE6477) |
| OVC | GSE18520 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE18520) |
| PRC | GSE17951 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE17951) |

### Pan-cancer datasets (two-step model)

| Dataset | Source |
|---|---|
| TCGA | UCSC Xena hubs (`tcga`, `toil`), downloaded by the scripts with `xenaPython` |
| expO | GEO [GSE2109](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2109) |
| Other GDC projects (BEATAML1.0, CGCI-HTMCP-CC, CMI-MPC, CPTAC-2, CPTAC-3, MMRF-COMMPASS, NCICCR-DLBCL) | [GDC portal](https://portal.gdc.cancer.gov) and the UCSC Xena GDC hub |
| CCLE | UCSC Xena public hub (DepMap release, CC BY 4.0) |

### Genetic analyses (SI Table S19)

pQTL summary statistics (UK Biobank Pharma Proteomics Project), cancer GWAS summary statistics (GWAS Catalog and
FinnGen) and the LD references are listed with accessions in SI Table S19. The genetics scripts read them from
`SMR_DATA_ROOT` (see the top-level README).

### Pathway and interaction resources

| Resource | Identifier | Access |
|---|---|---|
| Protein interactions | STRING v12, `ppi_enrichment` endpoint | [string-db.org](https://string-db.org/) |
| Gene sets | Enrichr `GO_Biological_Process_2023`, `KEGG_2021_Human`, `Reactome_2022` | [maayanlab.cloud/Enrichr](https://maayanlab.cloud/Enrichr/) |
| Tissue and cell-type specificity | Human Protein Atlas release 25.1 | [proteinatlas.org](https://www.proteinatlas.org/) |
