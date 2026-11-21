# v7 FORK of revise_plan/ppi_string/fetch_locked25_string.py (original unchanged): the 26-protein list (25 + MSMB); output under v7_plan/results/B9_biology/string_figures. File names keep the LOCKED25 prefix so the unchanged figure code reads them.
#!/usr/bin/env python3
r"""
Fetch official STRING v12.0 data for the locked 25-protein multiclass panel
(Part B), filling the gap left by the original batch: that batch covered
only the 12 disease-specific panels (`disease_specific_panels.csv`) and
never ran `locked_25_multiclass_panel.csv` through the pipeline.

Written 2026-08-23 to mirror the exact API parameters used by the two
sibling scripts in this folder (`download_STRING_analysis_12_panels.py`
and `batch_string_export.py`), so the locked-25 numbers are directly
comparable to the 12 disease-panel numbers -- same STRING version, same
confidence threshold, same network type/flavor, same two modes.

Fetches, for both `query_only` (add_nodes=0, the statistically honest
mode) and `first_shell_max5` (add_nodes=5, exploratory context only):
  - gene -> STRING ID mapping                  (00_mapping.tsv)
  - network edges                              (network.tsv)
  - PPI enrichment statistics (nodes/edges/p)   (ppi_enrichment.tsv)
  - functional enrichment terms (all)           (functional_enrichment.tsv)
  - official STRING network image (PNG)         (LOCKED25_<mode>_official_STRING_v12.png)

Does not fetch the enrichmentfigure GO/KEGG bubble plots (the 12-panel
batch's "official figures" step) -- not needed for the fig10 mosaic, and
skipping it keeps this run short and API-polite for a single 25-protein
panel.

Run:
    python fetch_locked25_string.py
Output:
    STRING_analysis_locked25/{query_only,first_shell_max5}/*
    STRING_analysis_locked25/LOCKED25_ppi_statistics.csv
"""
from __future__ import annotations
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))

import csv
import io
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

STRING_VERSION = "12.0"
API_BASE = "https://version-12-0.string-db.org/api"
SPECIES = 9606
REQUIRED_SCORE = 400
CALLER_IDENTITY = "proteomics_locked25_ppi_pipeline"
DELAY_SECONDS = 1.1

LOCKED25 = [
    "FLT3", "CNTN1", "FCER2", "PRDX6", "LTA4H", "XG", "CCDC80", "CXCL17",
    "CXCL13", "SLAMF7", "PAEP", "PSPN", "BMP4", "WFDC2", "TRAF2", "KLK13",
    "GLO1", "GFAP", "CEACAM5", "CGA", "ADAMTS13", "CRTAC1", "TCL1A",
    "ADAMTS15", "NEFL", "MSMB",
]

OUT = (PROJECT_ROOT / "v7_plan/results/B9_biology/string_figures/STRING_analysis_multiclass26")
MODES = {"query_only": 0, "first_shell_max5": 5}

_last_request = 0.0


def _wait() -> None:
    global _last_request
    elapsed = time.monotonic() - _last_request
    if elapsed < DELAY_SECONDS:
        time.sleep(DELAY_SECONDS - elapsed)
    _last_request = time.monotonic()


def get_tsv(method: str, params: dict) -> str:
    clean = {k: v for k, v in params.items() if v is not None}
    url = f"{API_BASE}/tsv/{method}?{urllib.parse.urlencode(clean)}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "locked25-ppi-fetch/1.0", "Accept": "text/tab-separated-values"}
    )
    _wait()
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read().decode("utf-8", errors="replace")


def get_image(params: dict) -> bytes:
    clean = {k: v for k, v in params.items() if v is not None}
    url = f"{API_BASE}/highres_image/network?{urllib.parse.urlencode(clean)}"
    request = urllib.request.Request(url, headers={"User-Agent": "locked25-ppi-fetch/1.0", "Accept": "*/*"})
    _wait()
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def parse_tsv(text: str) -> list[dict]:
    if not text.strip():
        return []
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    id_value = "\r".join(LOCKED25)

    print(f"[LOCKED25] mapping {len(LOCKED25)} proteins ...")
    mapping_text = get_tsv(
        "get_string_ids",
        {"identifiers": id_value, "species": SPECIES, "limit": 1, "echo_query": 1,
         "caller_identity": CALLER_IDENTITY},
    )
    mapping_rows = parse_tsv(mapping_text)
    missing = [g for g in LOCKED25 if g.upper() not in {r.get("queryItem", "").upper() for r in mapping_rows}]
    if missing:
        raise SystemExit(f"Unmapped protein(s): {missing}")
    print(f"   mapped {len(mapping_rows)}/{len(LOCKED25)} exactly")

    stats_rows = []
    for mode, add_nodes in MODES.items():
        mode_dir = OUT / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        print(f"[LOCKED25] {mode} (add_nodes={add_nodes}) ...")

        (mode_dir / "00_mapping.tsv").write_text(mapping_text, encoding="utf-8")

        network_text = get_tsv(
            "network",
            {"identifiers": id_value, "species": SPECIES, "required_score": REQUIRED_SCORE,
             "network_type": "functional", "add_nodes": add_nodes, "caller_identity": CALLER_IDENTITY},
        )
        (mode_dir / "network.tsv").write_text(network_text, encoding="utf-8")

        ppi_text = get_tsv(
            "ppi_enrichment",
            {"identifiers": id_value, "species": SPECIES, "required_score": REQUIRED_SCORE,
             "network_type": "functional", "add_nodes": add_nodes, "caller_identity": CALLER_IDENTITY},
        )
        (mode_dir / "ppi_enrichment.tsv").write_text(ppi_text, encoding="utf-8")
        ppi_row = parse_tsv(ppi_text)[0] if parse_tsv(ppi_text) else {}

        enrichment_text = get_tsv(
            "enrichment",
            {"identifiers": id_value, "species": SPECIES, "caller_identity": CALLER_IDENTITY},
        )
        (mode_dir / "functional_enrichment.tsv").write_text(enrichment_text, encoding="utf-8")
        n_terms = len(parse_tsv(enrichment_text))

        png_bytes = get_image({
            "identifiers": id_value, "species": SPECIES, "required_score": REQUIRED_SCORE,
            "network_type": "functional", "network_flavor": "evidence",
            "add_color_nodes": add_nodes, "add_white_nodes": 0,
            "hide_disconnected_nodes": 0, "hide_node_labels": 0,
            "show_query_node_labels": 1, "caller_identity": CALLER_IDENTITY,
        })
        png_path = mode_dir / f"LOCKED25_{mode}_official_STRING_v12.png"
        png_path.write_bytes(png_bytes)
        if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise SystemExit(f"{mode}: image response was not a valid PNG")

        row = {
            "panel": "LOCKED25", "cancer": "Locked 25-protein multiclass panel (Part B)",
            "mode": mode, "original_input_count": len(LOCKED25),
            "original_input_proteins": ";".join(LOCKED25),
            "first_shell_maximum": add_nodes,
            "number_of_nodes": ppi_row.get("number_of_nodes", ""),
            "number_of_edges": ppi_row.get("number_of_edges", ""),
            "average_node_degree": ppi_row.get("average_node_degree", ""),
            "local_clustering_coefficient": ppi_row.get("local_clustering_coefficient", ""),
            "expected_number_of_edges": ppi_row.get("expected_number_of_edges", ""),
            "ppi_enrichment_p_value": ppi_row.get("p_value", ""),
            "significant_functional_terms": n_terms,
        }
        stats_rows.append(row)
        print(f"   nodes={row['number_of_nodes']} edges={row['number_of_edges']} "
              f"p={row['ppi_enrichment_p_value']} terms={n_terms}")

    stats_path = OUT / "LOCKED25_ppi_statistics.csv"
    with stats_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)
    print(f"\nwrote {stats_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
