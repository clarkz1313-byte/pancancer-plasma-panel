# Official SMR/HEIDI scripts

Supporting Python tooling for the SMR and HEIDI analysis of the two promoted
protein–cancer pairs (BMP4–CRC, LEP–BRC): preparing summary-statistic inputs,
running the official SMR binary, summarising its output, and cross-checking
with a two-term Wald computation and `coloc`-based sensitivity analysis
(`run_coloc_sensitivity.R`).

These scripts operate on the pQTL, GWAS, and LD-reference files described in
`data/README.md` and the Methods section on colocalisation and SMR — that raw
data is not included in this repository. Run each script with `--help` for
its exact inputs.

| Script | Purpose |
|---|---|
| `fetch_region_subset.py` | extract a regional slice of the LD reference |
| `prepare_summary_inputs.py` | convert pQTL/GWAS summary statistics to SMR's input formats |
| `run_official_smr.py` | drive the SMR 1.3.1 binary |
| `summarize_official_results.py` | parse SMR/HEIDI output |
| `recompute_screen_two_term.py` | independent two-term Wald-ratio cross-check |
| `run_coloc_sensitivity.R` | colocalisation sensitivity across the swept priors |
| `build_manifests.py` | provenance manifest (software versions, source and output hashes) |
| `validate_patch.py` | consistency checks over the assembled result tables |

The reported result for both pairs is exploratory: the manuscript reports
the SMR and colocalisation estimates alongside their HEIDI heterogeneity
tests at both the current 20-variant-capped mode and the older uncapped
mode, and neither pair supports a claim that the measured protein mediates
cancer risk. See Results and Methods in the manuscript for the full
reporting.
