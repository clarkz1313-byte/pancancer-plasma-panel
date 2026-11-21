#!/usr/bin/env python3
r"""
Fetch a single, POOLED STRING v12.0 network across all 12 disease-specific
panels' own proteins combined (query_only, add_nodes=0 -- no STRING-added
neighbours).

Why this exists: `download_STRING_analysis_12_panels.py` (the collaborator's
original batch) and `fig11_panels.py` both analyse each of the 12 disease
panels SEPARATELY -- 12 independent STRING queries. That never asks STRING
"does protein X from panel A interact with protein Y from panel B" because
X and Y were never submitted in the same query together. This script asks
that question directly: submit all 12 panels' proteins (deduplicated) as
ONE query, so any real STRING-documented edge between two different panels'
proteins actually shows up.

15 of the 77 unique proteins across the 12 panels are already known to be
shared BY NAME (e.g. LEP appears in BRC, ENDC, and PRC's own gene list) --
that overlap is visible just from `disease_specific_panels.csv` with no API
call. This fetch additionally finds any FUNCTIONAL/PPI edges STRING knows
about between proteins that are NOT the same gene but are still linked --
the kind of cross-panel connection you cannot see from the panel-by-panel
figures at all.

Mode: query_only only (add_nodes=0) -- deliberately no first_shell_max5
here. Adding STRING-picked neighbours on top of a pool already drawn from
12 different panels would make it impossible to tell whether any given
connection reflects the panels' own proteins or an artificial STRING
addition; the user asked for "no extended" for exactly this reason.

Run:
    python fetch_pooled12panels_string.py
Output:
    STRING_analysis_pooled12panels/
        00_mapping.tsv, network.tsv, ppi_enrichment.tsv,
        functional_enrichment.tsv, gene_panel_membership.csv,
        POOLED12_query_only_official_STRING_v12.png
"""
from __future__ import annotations

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
CALLER_IDENTITY = "proteomics_pooled12panels_ppi_pipeline"
DELAY_SECONDS = 1.1

HERE = Path(__file__).resolve().parent
PANELS_CSV = HERE / "disease_specific_panels.csv"
OUT = HERE / "STRING_analysis_pooled12panels"

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
        url, headers={"User-Agent": "pooled12panels-ppi-fetch/1.0", "Accept": "text/tab-separated-values"}
    )
    _wait()
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read().decode("utf-8", errors="replace")


def get_image(params: dict) -> bytes:
    clean = {k: v for k, v in params.items() if v is not None}
    url = f"{API_BASE}/highres_image/network?{urllib.parse.urlencode(clean)}"
    request = urllib.request.Request(url, headers={"User-Agent": "pooled12panels-ppi-fetch/1.0", "Accept": "*/*"})
    _wait()
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def get_svg(params: dict) -> bytes:
    """SVG export of the same network. Fetched in addition to the PNG so
    fig11c_pooled12panels_network.py can reuse STRING's own node
    coordinates (embedded as data-x_pos/data-y_pos attributes) instead of
    computing an independent layout -- lets the panel-coloured figure and
    the official-render figure show nodes in the same positions, telling
    different stories from the same layout rather than two unrelated ones."""
    clean = {k: v for k, v in params.items() if v is not None}
    url = f"{API_BASE}/svg/network?{urllib.parse.urlencode(clean)}"
    request = urllib.request.Request(url, headers={"User-Agent": "pooled12panels-ppi-fetch/1.0", "Accept": "*/*"})
    _wait()
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def parse_tsv(text: str) -> list[dict]:
    if not text.strip():
        return []
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def load_panels() -> dict[str, list[str]]:
    panels: dict[str, list[str]] = {}
    with PANELS_CSV.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            panels[row["Panel"]] = [g.strip() for g in row["Genes"].split(";") if g.strip()]
    return panels


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panels = load_panels()

    gene_to_panels: dict[str, list[str]] = {}
    for panel, genes in panels.items():
        for g in genes:
            gene_to_panels.setdefault(g, []).append(panel)
    pooled_genes = sorted(gene_to_panels)

    membership_path = OUT / "gene_panel_membership.csv"
    with membership_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gene", "n_panels", "panels"])
        for g in pooled_genes:
            ps = gene_to_panels[g]
            writer.writerow([g, len(ps), ";".join(sorted(ps))])
    print(f"[POOLED12] {len(pooled_genes)} unique proteins across {len(panels)} panels "
          f"({sum(1 for g in pooled_genes if len(gene_to_panels[g]) > 1)} shared by name)")
    print(f"   wrote {membership_path}")

    id_value = "\r".join(pooled_genes)

    print(f"[POOLED12] mapping {len(pooled_genes)} proteins ...")
    mapping_text = get_tsv(
        "get_string_ids",
        {"identifiers": id_value, "species": SPECIES, "limit": 1, "echo_query": 1,
         "caller_identity": CALLER_IDENTITY},
    )
    mapping_rows = parse_tsv(mapping_text)
    missing = [g for g in pooled_genes if g.upper() not in {r.get("queryItem", "").upper() for r in mapping_rows}]
    if missing:
        print(f"   WARNING -- unmapped: {missing}")
    print(f"   mapped {len(mapping_rows)}/{len(pooled_genes)}")
    (OUT / "00_mapping.tsv").write_text(mapping_text, encoding="utf-8")

    print("[POOLED12] query_only (add_nodes=0) ...")
    network_text = get_tsv(
        "network",
        {"identifiers": id_value, "species": SPECIES, "required_score": REQUIRED_SCORE,
         "network_type": "functional", "add_nodes": 0, "caller_identity": CALLER_IDENTITY},
    )
    (OUT / "network.tsv").write_text(network_text, encoding="utf-8")
    n_edges = len(parse_tsv(network_text))

    ppi_text = get_tsv(
        "ppi_enrichment",
        {"identifiers": id_value, "species": SPECIES, "required_score": REQUIRED_SCORE,
         "network_type": "functional", "add_nodes": 0, "caller_identity": CALLER_IDENTITY},
    )
    (OUT / "ppi_enrichment.tsv").write_text(ppi_text, encoding="utf-8")
    ppi_row = parse_tsv(ppi_text)[0] if parse_tsv(ppi_text) else {}

    enrichment_text = get_tsv(
        "enrichment",
        {"identifiers": id_value, "species": SPECIES, "caller_identity": CALLER_IDENTITY},
    )
    (OUT / "functional_enrichment.tsv").write_text(enrichment_text, encoding="utf-8")
    n_terms = len(parse_tsv(enrichment_text))

    image_params = {
        "identifiers": id_value, "species": SPECIES, "required_score": REQUIRED_SCORE,
        "network_type": "functional", "network_flavor": "evidence",
        "add_color_nodes": 0, "add_white_nodes": 0,
        "hide_disconnected_nodes": 0, "hide_node_labels": 0,
        "show_query_node_labels": 1, "caller_identity": CALLER_IDENTITY,
    }
    png_bytes = get_image(image_params)
    png_path = OUT / "POOLED12_query_only_official_STRING_v12.png"
    png_path.write_bytes(png_bytes)
    if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise SystemExit("image response was not a valid PNG")

    svg_bytes = get_svg(image_params)
    svg_path = OUT / "POOLED12_query_only_official_STRING_v12.svg"
    svg_path.write_bytes(svg_bytes)
    if not svg_bytes.lstrip().startswith(b"<svg"):
        raise SystemExit("svg response was not valid SVG")

    print(f"   nodes={ppi_row.get('number_of_nodes')} edges={n_edges} "
          f"expected={ppi_row.get('expected_number_of_edges')} "
          f"p={ppi_row.get('p_value')} terms={n_terms}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
