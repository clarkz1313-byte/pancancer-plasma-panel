# Data

## `filtered_pancancer_data.csv` (included, 15 MB)

1,375 patients, 12 cancer groups, 1,463 Olink Explore 1536 protein measurements
(log2-scale normalised protein expression, NPX), plus `Sample_ID`, `Cancer`,
`protein_count`.

```
SHA-256: 7825207BD0502EB7C8C414E980DD9F015A96E9001491F1C500C417A819AA252F
```

Verify before running anything:

```powershell
Get-FileHash .\data\filtered_pancancer_data.csv -Algorithm SHA256
```

Do not reorder rows, rename columns, or change missing-value encoding —
several analysis scripts assume this exact layout.

## Provenance and licence

This file is derived from the pan-cancer plasma proteomics resource of
Bueno Álvez et al., *Nat Commun* 14:4308 (2023),
[doi:10.1038/s41467-023-39765-y](https://doi.org/10.1038/s41467-023-39765-y).
The source data are openly deposited in BioStudies under accession
[S-BSST935](https://www.ebi.ac.uk/biostudies/studies/S-BSST935) and licensed
CC BY 4.0. This is the cancer-only cohort; the source study's Wellness healthy
cohort is not used here and is not included (it is separately access-controlled
through the Swedish National Data Service — irrelevant to this project, which
has no healthy-control arm).

## External datasets (not included — download separately)

None of the datasets below are redistributed in this repository; each has its
own licence and access route.

### External validation cohorts (public, no application needed)

| Cancer | Accession | Repository |
|---|---|---|
| AML, CLL | GSE13159 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159) |
| BRC | GSE42568 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42568) |
| CRC | GSE41258 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE41258) |
| CVX | GSE9750, GSE63514 | [GEO](https://www.ncbi.nlm.nih.gov/geo/) |
| ENDC | GSE17025 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE17025) |
| GLIOM | GSE4290 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE4290) |
| LUNGC | CPTAC–ICPC LUAD | [PDC](https://proteomic.datacommons.cancer.gov/) |
| DLBCL | GSE32018 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32018) |
| MYEL | GSE6477 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE6477) |
| OVC | GSE18520 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE18520) |
| PRC | GSE17951 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE17951) |

### Genetics arm (public; UKB-PPP needs an approved application)

| Resource | Identifier | Access |
|---|---|---|
| pQTLs | UK Biobank Pharma Proteomics Project, 34,557 participants, GRCh38 | [ukbiobank.ac.uk](https://www.ukbiobank.ac.uk/) — **requires an approved application, not instant download** |
| Cancer GWAS (10 of 11) | GCST004988 (BRC), GCST90255675 (CRC), GCST004748 (LUNGC), GCST006085 (PRC), GCST90455661 (OVC), GCST90454186 (ENDC), GCST90624747 (MYEL), GCST90624739 (DLBCL), GCST90624738 (CLL), GCST90707271 (AML) | [GWAS Catalog](https://www.ebi.ac.uk/gwas/), harmonised GRCh38 |
| Glioma GWAS | FinnGen R12, endpoint `C3_BRAIN_EXALLC` | [finngen.fi](https://www.finngen.fi/) |
| LD reference | 1000 Genomes high-coverage GRCh38, 503 European participants | [internationalgenome.org](https://www.internationalgenome.org/) |

Full accession table with case/control counts: `docs/SUPPLEMENTARY_TABLE_S2_gwas_sources.md`.

### Pathway / interaction resources

| Resource | Identifier | Access |
|---|---|---|
| Protein interactions | STRING v12, `ppi_enrichment` endpoint | [string-db.org](https://string-db.org/) |
| Gene sets | Enrichr `GO_Biological_Process_2023`, `KEGG_2021_Human`, `Reactome_2022` | [maayanlab.cloud/Enrichr](https://maayanlab.cloud/Enrichr/) |
| Hallmark gene sets | MSigDB `Hallmark_2020` (version-frozen; checksum below) | [gsea-msigdb.org](https://www.gsea-msigdb.org/) |

`MSigDB_Hallmark_2020_frozen.gmt` SHA-256:
`6D4141849364D2C93453CA37B6963158B7127B8739FBC3050AFE419AE523BEA0`
