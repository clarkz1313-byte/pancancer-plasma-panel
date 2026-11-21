#!/usr/bin/env python3
"""Batch export official STRING v12.0 results for 12 disease-specific panels.

The script reads ``disease_specific_panels.csv`` with two columns:

    Panel,Genes
    AML,"FLT3;CD244;PGLYRP1"

It produces two strictly separated analyses for every validated panel:

1. Query-only (primary): submitted proteins only; no added interactors.
2. First-shell max 5 (exploratory): up to five STRING-added interactors.

The locked 25-marker multiclass panel is deliberately not read or processed.
Official network PNG/SVG files are downloaded directly from STRING; this
script never redraws or recolors them.
"""

from __future__ import print_function

import argparse
import csv
import html
import os
import socket
import sys
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

API_BASE = "https://version-12-0.string-db.org/api"
STRING_VERSION = "12.0"
SPECIES = 9606
REQUIRED_SCORE = 400
FIRST_SHELL_MAX = 5
CALLER_IDENTITY = "proteomics_disease_specific_ppi_pipeline"

EXPECTED_COUNTS = OrderedDict(
    [
        ("AML", 3),
        ("BRC", 13),
        ("CLL", 2),
        ("CRC", 10),
        ("CVX", 11),
        ("ENDC", 10),
        ("GLIOM", 3),
        ("LUNGC", 6),
        ("LYMPH", 5),
        ("MYEL", 8),
        ("OVC", 9),
        ("PRC", 13),
    ]
)

# Conservative display names. Panel codes remain the authoritative identifiers.
DISPLAY_NAMES = {
    "AML": "acute myeloid leukemia",
    "BRC": "breast cancer",
    "CLL": "chronic lymphocytic leukemia",
    "CRC": "colorectal cancer",
    "CVX": "cervical cancer",
    "ENDC": "endometrial cancer",
    "GLIOM": "glioma",
    "LUNGC": "lung cancer",
    "LYMPH": "lymphoma",
    "MYEL": "multiple myeloma",
    "OVC": "ovarian cancer",
    "PRC": "prostate cancer",
}

TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


EVIDENCE_LEGEND = """# STRING evidence-view legend

## Nodes

- Each node represents all proteins produced by a single protein-coding gene locus.
- Node colors do not represent cancer type, biological function, pathway membership, or biomarker importance; they are primarily visual aids.
- Colored nodes represent submitted query proteins and first-shell interactors.
- White or grey nodes represent second-shell interactors.
- Filled nodes indicate that a known or predicted three-dimensional protein structure is available.
- Empty nodes indicate that structural information is unavailable.
- In an expanded network, original query proteins and added first-shell interactors must be distinguished using the accompanying added-interactors table, not by node color alone.

## Edges

- Edges represent protein-protein functional associations and do not necessarily indicate direct physical binding.
- In evidence view, one interaction may contain multiple colored lines because several evidence sources may support the same association.

## Evidence-view edge colors

- Red: gene fusion evidence
- Green: gene neighborhood evidence
- Blue: gene co-occurrence evidence
- Purple or pink: experimental evidence
- Yellow: text-mining evidence
- Light blue or cyan: curated database evidence
- Black: co-expression evidence

## Analysis settings

- STRING version 12.0
- Homo sapiens, taxon ID 9606
- Full functional STRING network
- Evidence view
- All seven evidence sources enabled
- Medium-confidence threshold: combined score >= 0.400
"""


class PipelineError(Exception):
    """Base pipeline error."""


class ApiError(PipelineError):
    """STRING request failure with details for the error report."""

    def __init__(self, endpoint, message, status="", retry_count=0):
        super().__init__(message)
        self.endpoint = endpoint
        self.status = status
        self.retry_count = retry_count


class HttpResponse:
    """Small response wrapper used to avoid third-party dependencies."""

    def __init__(self, status_code, headers, content):
        self.status_code = status_code
        self.headers = headers
        self.content = content

    @property
    def text(self):
        return self.content.decode("utf-8", errors="replace")


class StringClient:
    """Rate-limited STRING v12.0 HTTP client with bounded retries."""

    def __init__(self, delay=1.05, max_retries=4, timeout=120):
        self.delay = max(float(delay), 1.0)
        self.max_retries = max(int(max_retries), 0)
        self.timeout = int(timeout)
        self.last_request_at = 0.0

    def _rate_limit(self):
        elapsed = time.monotonic() - self.last_request_at
        wait_for = self.delay - elapsed
        if wait_for > 0:
            time.sleep(wait_for)

    def post(self, output_format, method, params):
        endpoint = "%s/%s/%s" % (API_BASE, output_format, method)
        data = dict(params)
        data.setdefault("caller_identity", CALLER_IDENTITY)
        last_message = ""
        last_status = ""

        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(min(2 ** attempt, 20))
            self._rate_limit()
            try:
                encoded = urllib_parse.urlencode(data).encode("utf-8")
                request = urllib_request.Request(
                    endpoint,
                    data=encoded,
                    headers={
                        "User-Agent": CALLER_IDENTITY,
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    method="POST",
                )
                with urllib_request.urlopen(request, timeout=self.timeout) as raw_response:
                    content = raw_response.read()
                    response = HttpResponse(
                        status_code=raw_response.getcode(),
                        headers=raw_response.headers,
                        content=content,
                    )
                self.last_request_at = time.monotonic()
                return response
            except urllib_error.HTTPError as exc:
                self.last_request_at = time.monotonic()
                last_status = exc.code
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body = ""
                last_message = "HTTP %s: %s" % (
                    exc.code,
                    body[:300].replace("\n", " "),
                )
                if exc.code not in TRANSIENT_HTTP_CODES:
                    break
            except (urllib_error.URLError, TimeoutError, socket.timeout, OSError) as exc:
                self.last_request_at = time.monotonic()
                last_message = "%s: %s" % (type(exc).__name__, exc)

        raise ApiError(
            endpoint=endpoint,
            message=last_message or "Unknown STRING API error",
            status=last_status,
            retry_count=self.max_retries,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export official STRING v12.0 outputs for 12 disease-specific panels."
    )
    parser.add_argument(
        "--panels",
        default="disease_specific_panels.csv",
        help="Input CSV with Panel and semicolon-delimited Genes columns.",
    )
    parser.add_argument(
        "--output",
        default="string_outputs_disease_specific",
        help="Disease-specific output directory.",
    )
    parser.add_argument(
        "--panel",
        choices=list(EXPECTED_COUNTS),
        help="Optionally process one validated panel only (for testing).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the input CSV without sending any STRING requests.",
    )
    parser.add_argument("--delay", type=float, default=1.05)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--zip-name",
        default="string_outputs_disease_specific_official_figures.zip",
    )
    parser.add_argument("--no-zip", action="store_true")
    return parser.parse_args(argv)


def read_and_validate_panels(path):
    path = Path(path)
    if not path.is_file():
        raise PipelineError("Input file not found: %s" % path)

    panels = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Panel" not in reader.fieldnames or "Genes" not in reader.fieldnames:
            raise PipelineError(
                "Input CSV must contain exactly named columns Panel and Genes."
            )
        for line_number, row in enumerate(reader, start=2):
            panel = (row.get("Panel") or "").strip()
            raw_genes = (row.get("Genes") or "").strip()
            if not panel and not raw_genes:
                continue
            if not panel or not raw_genes:
                raise PipelineError("Incomplete panel row at CSV line %d." % line_number)
            if panel in panels:
                raise PipelineError("Duplicate panel in input: %s" % panel)
            genes = [item.strip() for item in raw_genes.split(";") if item.strip()]
            if len(genes) != len(set(genes)):
                raise PipelineError("Duplicate protein within panel %s." % panel)
            panels[panel] = genes

    expected = set(EXPECTED_COUNTS)
    observed = set(panels)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise PipelineError(
            "Panel set mismatch. Missing: %s; unexpected: %s"
            % (";".join(missing) or "none", ";".join(extra) or "none")
        )

    ordered = OrderedDict()
    for panel, expected_count in EXPECTED_COUNTS.items():
        genes = panels[panel]
        if len(genes) != expected_count:
            raise PipelineError(
                "%s: expected %d proteins, found %d"
                % (panel, expected_count, len(genes))
            )
        ordered[panel] = genes
    return ordered


def print_panel_validation(panels):
    print("Validated disease-specific panels:")
    for panel, genes in panels.items():
        print("  %-6s expected %2d, found %2d: %s" % (
            panel,
            EXPECTED_COUNTS[panel],
            len(genes),
            ";".join(genes),
        ))
    print("All 12 disease-specific panel counts are correct.")
    print("Locked 25-marker multiclass panel will not be processed.")


def atomic_write_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
    os.replace(str(temporary), str(path))


def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode("utf-8"))


def is_html_error(content):
    prefix = content.lstrip()[:200].lower()
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")


def valid_png(path):
    try:
        with Path(path).open("rb") as handle:
            return handle.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE
    except OSError:
        return False


def valid_svg(path):
    try:
        content = Path(path).read_bytes()
        return bool(content) and not is_html_error(content) and b"<svg" in content[:5000].lower()
    except OSError:
        return False


def valid_text(path):
    try:
        content = Path(path).read_bytes()
        return bool(content) and not is_html_error(content)
    except OSError:
        return False


def valid_url_file(path):
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
        return value.startswith("https://") and "string-db.org" in value
    except OSError:
        return False


def record_error(errors, panel, analysis_type, requested_file, exc):
    errors.append(
        {
            "panel": panel,
            "analysis_type": analysis_type,
            "requested_file": str(requested_file),
            "endpoint": getattr(exc, "endpoint", "local validation"),
            "HTTP_status": getattr(exc, "status", ""),
            "error_message": str(exc),
            "retry_count": getattr(exc, "retry_count", 0),
            "final_status": "failed",
        }
    )
    print("  FAILED %s: %s" % (requested_file, exc))


def fetch_text(client, panel, analysis_type, output_format, method, params, path, errors):
    path = Path(path)
    if valid_text(path):
        print("  Reusing valid file: %s" % path)
        return path.read_text(encoding="utf-8")
    try:
        response = client.post(output_format, method, params)
        content = response.content
        if not content or is_html_error(content):
            raise ApiError(
                "%s/%s/%s" % (API_BASE, output_format, method),
                "STRING returned empty or HTML content instead of %s" % output_format,
                response.status_code,
            )
        text = content.decode("utf-8")
        atomic_write_text(path, text)
        print("  Saved: %s" % path)
        return text
    except (ApiError, UnicodeDecodeError, OSError) as exc:
        record_error(errors, panel, analysis_type, path, exc)
        return None


def fetch_image(client, panel, analysis_type, output_format, params, path, errors):
    path = Path(path)
    validator = valid_png if output_format == "highres_image" else valid_svg
    if validator(path):
        print("  Reusing valid file: %s" % path)
        return True
    try:
        response = client.post(output_format, "network", params)
        content = response.content
        content_type = (response.headers.get("Content-Type") or "").lower()
        if not content or is_html_error(content):
            raise ApiError(
                "%s/%s/network" % (API_BASE, output_format),
                "STRING returned empty or HTML content instead of an image",
                response.status_code,
            )
        if output_format == "highres_image":
            if not content.startswith(PNG_SIGNATURE):
                raise ApiError(
                    "%s/%s/network" % (API_BASE, output_format),
                    "Invalid PNG signature (Content-Type: %s)" % content_type,
                    response.status_code,
                )
        elif b"<svg" not in content[:5000].lower():
            raise ApiError(
                "%s/%s/network" % (API_BASE, output_format),
                "Invalid SVG content (Content-Type: %s)" % content_type,
                response.status_code,
            )
        atomic_write_bytes(path, content)
        print("  Saved official STRING image: %s" % path)
        return True
    except (ApiError, OSError) as exc:
        record_error(errors, panel, analysis_type, path, exc)
        return False


def parse_tsv(text):
    if not text or not text.strip():
        return []
    return list(csv.DictReader(text.splitlines(), delimiter="\t"))


def fetch_stable_link(client, panel, analysis_type, params, path, errors):
    path = Path(path)
    if valid_url_file(path):
        print("  Reusing valid link: %s" % path)
        return path.read_text(encoding="utf-8").strip()
    try:
        response = client.post("tsv", "get_link", params)
        rows = parse_tsv(response.text)
        url = (rows[0].get("url") if rows else "") or ""
        url = url.strip()
        if not url.startswith("https://") or "string-db.org" not in url:
            raise ApiError(
                "%s/tsv/get_link" % API_BASE,
                "STRING did not return a valid stable URL",
                response.status_code,
            )
        atomic_write_text(path, url + "\n")
        print("  Saved stable STRING link: %s" % path)
        return url
    except (ApiError, OSError) as exc:
        record_error(errors, panel, analysis_type, path, exc)
        return ""


def parse_mapping(mapping_text, submitted_genes):
    rows = parse_tsv(mapping_text)
    by_query = {}
    for row in rows:
        query_item = (row.get("queryItem") or "").strip()
        if query_item:
            by_query[query_item.upper()] = row

    mapped = []
    unmapped = []
    problems = []
    for gene in submitted_genes:
        row = by_query.get(gene.upper())
        if not row or not (row.get("stringId") or "").strip():
            unmapped.append(gene)
            continue
        taxon = str(row.get("ncbiTaxonId") or "").strip()
        if taxon != str(SPECIES):
            problems.append("%s mapped to taxon %s" % (gene, taxon or "unknown"))
            continue
        mapped.append(
            {
                "query": gene,
                "string_id": row["stringId"].strip(),
                "preferred_name": (row.get("preferredName") or gene).strip(),
                "taxon": taxon,
            }
        )
    return mapped, unmapped, problems


def parse_network_nodes(network_text):
    nodes = OrderedDict()
    for row in parse_tsv(network_text):
        for suffix in ("A", "B"):
            string_id = (row.get("stringId_%s" % suffix) or "").strip()
            preferred = (row.get("preferredName_%s" % suffix) or string_id).strip()
            if string_id:
                nodes[string_id] = preferred
    return nodes


def parse_ppi_stats(ppi_text):
    rows = parse_tsv(ppi_text)
    row = rows[0] if rows else {}
    return {
        "number_of_nodes": row.get("number_of_nodes", ""),
        "number_of_edges": row.get("number_of_edges", ""),
        "expected_number_of_edges": row.get("expected_number_of_edges", ""),
        "average_node_degree": row.get("average_node_degree", ""),
        "local_clustering_coefficient": row.get("local_clustering_coefficient", ""),
        "ppi_enrichment_p_value": row.get("p_value", ""),
    }


def image_params(identifiers, add_color_nodes):
    return {
        "identifiers": "\r".join(identifiers),
        "species": SPECIES,
        "required_score": REQUIRED_SCORE,
        "network_type": "functional",
        "network_flavor": "evidence",
        "add_color_nodes": add_color_nodes,
        "add_white_nodes": 0,
        "hide_disconnected_nodes": 0,
        "hide_node_labels": 0,
        "show_query_node_labels": 1,
    }


def network_params(identifiers, add_nodes):
    return {
        "identifiers": "\r".join(identifiers),
        "species": SPECIES,
        "required_score": REQUIRED_SCORE,
        "network_type": "functional",
        "add_nodes": add_nodes,
        # Keep canonical preferred gene names in TSV columns; the official
        # image/link requests separately preserve submitted gene labels.
        "show_query_node_labels": 0,
    }


def enrichment_params(identifiers, include_score=False):
    params = {"identifiers": "\r".join(identifiers), "species": SPECIES}
    if include_score:
        params["required_score"] = REQUIRED_SCORE
    return params


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def process_panel(client, panel, genes, output_root, errors):
    figures_root = output_root / "official_figures"
    query_dir = figures_root / "query_only" / panel
    expanded_dir = figures_root / "first_shell_max5" / panel
    query_dir.mkdir(parents=True, exist_ok=True)
    expanded_dir.mkdir(parents=True, exist_ok=True)

    print("\n[%s] Mapping %d submitted proteins..." % (panel, len(genes)))
    mapping_path = query_dir / (panel + "_query_only_mapping.tsv")
    mapping_text = fetch_text(
        client,
        panel,
        "mapping",
        "tsv",
        "get_string_ids",
        {
            "identifiers": "\r".join(genes),
            "species": SPECIES,
            "echo_query": 1,
            "limit": 1,
        },
        mapping_path,
        errors,
    )
    if mapping_text is None:
        return None

    mapped, unmapped, mapping_problems = parse_mapping(mapping_text, genes)
    if mapping_problems:
        message = "; ".join(mapping_problems)
        record_error(
            errors,
            panel,
            "mapping",
            mapping_path,
            PipelineError(message),
        )
        return None
    if unmapped:
        print("  WARNING unmapped proteins: %s" % ";".join(unmapped))
    else:
        print("  All proteins mapped to Homo sapiens.")
    if not mapped:
        record_error(
            errors,
            panel,
            "mapping",
            mapping_path,
            PipelineError("No proteins mapped; panel cannot be processed."),
        )
        return None

    mapped_ids = [item["string_id"] for item in mapped]
    mapped_names = [item["preferred_name"] for item in mapped]

    # -------------------- Query-only primary analysis --------------------
    print("[%s] Query-only official STRING outputs..." % panel)
    q_network_path = query_dir / (panel + "_query_only_network.tsv")
    q_network_text = fetch_text(
        client,
        panel,
        "query_only",
        "tsv",
        "network",
        network_params(mapped_ids, 0),
        q_network_path,
        errors,
    )

    q_ppi_path = query_dir / (panel + "_query_only_ppi_enrichment.tsv")
    q_ppi_text = fetch_text(
        client,
        panel,
        "query_only",
        "tsv",
        "ppi_enrichment",
        enrichment_params(mapped_ids, include_score=True),
        q_ppi_path,
        errors,
    )

    q_enrichment_path = query_dir / (panel + "_query_only_functional_enrichment.tsv")
    fetch_text(
        client,
        panel,
        "query_only",
        "tsv",
        "enrichment",
        enrichment_params(mapped_ids),
        q_enrichment_path,
        errors,
    )

    # Preferred names preserve human-readable gene labels in official figures,
    # while the data/statistics calls above continue to use unambiguous IDs.
    q_image_parameters = image_params(mapped_names, 0)
    q_png = query_dir / (panel + "_query_only_official_STRING_v12.png")
    q_svg = query_dir / (panel + "_query_only_official_STRING_v12.svg")
    fetch_image(client, panel, "query_only", "highres_image", q_image_parameters, q_png, errors)
    fetch_image(client, panel, "query_only", "svg", q_image_parameters, q_svg, errors)
    q_link = fetch_stable_link(
        client,
        panel,
        "query_only",
        q_image_parameters,
        query_dir / (panel + "_query_only_official_STRING_link.txt"),
        errors,
    )

    q_stats = parse_ppi_stats(q_ppi_text or "")
    q_summary = {
        "panel": panel,
        "submitted_protein_count": len(genes),
        "submitted_protein_names": ";".join(genes),
        "mapped_protein_count": len(mapped),
        "mapped_protein_names": ";".join(mapped_names),
        "unmapped_protein_count": len(unmapped),
        "unmapped_protein_names": ";".join(unmapped),
    }
    q_summary.update(q_stats)

    # ---------------- Exploratory first-shell max 5 analysis -------------
    print("[%s] First-shell-expanded official STRING outputs (maximum 5)..." % panel)
    expanded_network_path = expanded_dir / (panel + "_first_shell_max5_network.tsv")
    expanded_network_text = fetch_text(
        client,
        panel,
        "first_shell_max5",
        "tsv",
        "network",
        network_params(mapped_ids, FIRST_SHELL_MAX),
        expanded_network_path,
        errors,
    )

    network_nodes = parse_network_nodes(expanded_network_text or "")
    query_ids = set(mapped_ids)
    added_ids = [string_id for string_id in network_nodes if string_id not in query_ids]
    if len(added_ids) > FIRST_SHELL_MAX:
        record_error(
            errors,
            panel,
            "first_shell_max5",
            expanded_network_path,
            PipelineError(
                "Expanded network contained %d non-query nodes; expected no more than %d."
                % (len(added_ids), FIRST_SHELL_MAX)
            ),
        )
    added_names = [network_nodes[string_id] for string_id in added_ids]
    expanded_ids = list(mapped_ids) + added_ids

    added_rows = []
    for item in mapped:
        added_rows.append(
            {
                "panel": panel,
                "protein_name": item["preferred_name"],
                "string_id": item["string_id"],
                "query_or_added": "query",
                "requested_first_shell_maximum": FIRST_SHELL_MAX,
                "actual_added_interactor_count": len(added_ids),
            }
        )
    for string_id in added_ids:
        added_rows.append(
            {
                "panel": panel,
                "protein_name": network_nodes[string_id],
                "string_id": string_id,
                "query_or_added": "added_first_shell",
                "requested_first_shell_maximum": FIRST_SHELL_MAX,
                "actual_added_interactor_count": len(added_ids),
            }
        )
    added_table_path = expanded_dir / (panel + "_first_shell_added_interactors.tsv")
    added_table_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_added = added_table_path.with_name(added_table_path.name + ".tmp")
    with temporary_added.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "panel",
            "protein_name",
            "string_id",
            "query_or_added",
            "requested_first_shell_maximum",
            "actual_added_interactor_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(added_rows)
    os.replace(str(temporary_added), str(added_table_path))

    e_ppi_path = expanded_dir / (panel + "_first_shell_max5_ppi_enrichment.tsv")
    e_ppi_text = fetch_text(
        client,
        panel,
        "first_shell_max5",
        "tsv",
        "ppi_enrichment",
        enrichment_params(expanded_ids, include_score=True),
        e_ppi_path,
        errors,
    )
    fetch_text(
        client,
        panel,
        "first_shell_max5",
        "tsv",
        "enrichment",
        enrichment_params(expanded_ids),
        expanded_dir / (panel + "_first_shell_max5_functional_enrichment.tsv"),
        errors,
    )

    e_image_parameters = image_params(mapped_names, FIRST_SHELL_MAX)
    e_png = expanded_dir / (panel + "_first_shell_max5_official_STRING_v12.png")
    e_svg = expanded_dir / (panel + "_first_shell_max5_official_STRING_v12.svg")
    fetch_image(client, panel, "first_shell_max5", "highres_image", e_image_parameters, e_png, errors)
    fetch_image(client, panel, "first_shell_max5", "svg", e_image_parameters, e_svg, errors)
    e_link = fetch_stable_link(
        client,
        panel,
        "first_shell_max5",
        e_image_parameters,
        expanded_dir / (panel + "_first_shell_max5_official_STRING_link.txt"),
        errors,
    )

    e_stats = parse_ppi_stats(e_ppi_text or "")
    e_summary = {
        "panel": panel,
        "original_query_count": len(mapped),
        "original_query_proteins": ";".join(mapped_names),
        "requested_first_shell_maximum": FIRST_SHELL_MAX,
        "actual_added_interactor_count": len(added_ids),
        "added_interactor_names": ";".join(added_names),
        "total_expanded_node_count": len(expanded_ids),
    }
    e_summary.update(e_stats)

    links_row = {
        "panel": panel,
        "submitted_protein_count": len(genes),
        "mapped_protein_count": len(mapped),
        "query_only_STRING_url": q_link,
        "requested_first_shell_maximum": FIRST_SHELL_MAX,
        "actual_added_interactor_count": len(added_ids),
        "first_shell_max5_STRING_url": e_link,
    }

    return {
        "panel": panel,
        "genes": genes,
        "mapped_names": mapped_names,
        "unmapped": unmapped,
        "added_names": added_names,
        "query_summary": q_summary,
        "expanded_summary": e_summary,
        "links": links_row,
        "query_png": q_png,
        "expanded_png": e_png,
        "query_link": q_link,
        "expanded_link": e_link,
        "query_network_available": q_network_text is not None,
    }


def write_combined_outputs(output_root, results, errors):
    figures_root = output_root / "official_figures"
    figures_root.mkdir(parents=True, exist_ok=True)

    q_fields = [
        "panel",
        "submitted_protein_count",
        "submitted_protein_names",
        "mapped_protein_count",
        "mapped_protein_names",
        "unmapped_protein_count",
        "unmapped_protein_names",
        "number_of_nodes",
        "number_of_edges",
        "expected_number_of_edges",
        "average_node_degree",
        "local_clustering_coefficient",
        "ppi_enrichment_p_value",
    ]
    e_fields = [
        "panel",
        "original_query_count",
        "original_query_proteins",
        "requested_first_shell_maximum",
        "actual_added_interactor_count",
        "added_interactor_names",
        "total_expanded_node_count",
        "number_of_nodes",
        "number_of_edges",
        "expected_number_of_edges",
        "average_node_degree",
        "local_clustering_coefficient",
        "ppi_enrichment_p_value",
    ]
    link_fields = [
        "panel",
        "submitted_protein_count",
        "mapped_protein_count",
        "query_only_STRING_url",
        "requested_first_shell_maximum",
        "actual_added_interactor_count",
        "first_shell_max5_STRING_url",
    ]
    error_fields = [
        "panel",
        "analysis_type",
        "requested_file",
        "endpoint",
        "HTTP_status",
        "error_message",
        "retry_count",
        "final_status",
    ]

    write_csv(
        output_root / "ppi_summary_query_only_all_12.csv",
        q_fields,
        [result["query_summary"] for result in results],
    )
    write_csv(
        output_root / "ppi_summary_first_shell_max5_all_12.csv",
        e_fields,
        [result["expanded_summary"] for result in results],
    )
    write_csv(
        figures_root / "STRING_network_links_all_12.csv",
        link_fields,
        [result["links"] for result in results],
    )
    # Provide the requested true TSV error report.
    error_tsv_path = figures_root / "STRING_export_errors.tsv"
    temporary = error_tsv_path.with_name(error_tsv_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=error_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(errors)
    os.replace(str(temporary), str(error_tsv_path))

    atomic_write_text(figures_root / "STRING_evidence_view_legend.md", EVIDENCE_LEGEND)
    write_figure_legends(figures_root / "figure_legends_all_12_panels.txt", results)
    write_html_index(figures_root / "STRING_official_networks_all_12.html", results)


def write_figure_legends(path, results):
    blocks = []
    for result in results:
        panel = result["panel"]
        disease = DISPLAY_NAMES.get(panel, panel)
        genes = result["genes"]
        added_names = result["added_names"]
        query_legend = (
            "[%s query-only] Protein-protein association network of the %s "
            "disease-specific panel. The network was generated using STRING v12.0 "
            "for Homo sapiens (taxon ID 9606), using the full functional association "
            "network, evidence view, all seven evidence channels, and a medium-confidence "
            "interaction threshold of 0.400. The submitted panel contained %d proteins: "
            "%s. No first- or second-shell interactors were added, and disconnected "
            "query proteins were retained. Nodes represent protein-coding gene loci, "
            "whereas edges represent functional associations and do not necessarily imply "
            "direct physical binding. Edge colors denote the supporting evidence channels: "
            "red, gene fusion; green, gene neighborhood; blue, gene co-occurrence; "
            "purple/pink, experiments; yellow, text mining; light blue/cyan, curated "
            "databases; and black, co-expression. Node colors are visual aids and do not "
            "indicate biological categories."
        ) % (panel, disease, len(genes), ", ".join(genes))

        expanded_legend = (
            "[%s first-shell-expanded] Exploratory STRING interaction-neighborhood "
            "network of the %s disease-specific panel. The %d model-selected query "
            "proteins were expanded by requesting up to five highest-scoring first-shell "
            "interactors, with no second-shell interactors added. STRING added %d "
            "first-shell interactors: %s. All other STRING settings were identical to "
            "those of the query-only analysis. Added interactors provide exploratory "
            "functional context but were not selected by the predictive model and are not "
            "members of the disease-specific biomarker panel."
        ) % (
            panel,
            disease,
            len(genes),
            len(added_names),
            ", ".join(added_names) if added_names else "none",
        )
        blocks.extend([query_legend, "", expanded_legend, "", "=" * 80, ""])
    atomic_write_text(path, "\n".join(blocks).rstrip() + "\n")


def write_html_index(path, results):
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>Official STRING v12.0 networks - 12 disease-specific panels</title>",
        "<style>body{font-family:Arial,sans-serif;margin:2rem;line-height:1.45;color:#222}",
        ".panel{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1.5rem 0}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}",
        "img{max-width:100%;height:auto;border:1px solid #eee;background:#fafafa}",
        "a.button{display:inline-block;padding:.55rem .8rem;background:#4257b2;color:#fff;text-decoration:none;border-radius:4px}",
        ".legend{background:#f5f5f5;padding:1rem;border-left:4px solid #4257b2}</style></head><body>",
        "<h1>Official STRING v12.0 disease-specific networks</h1>",
        "<p>Primary query-only networks are separated from exploratory first-shell-expanded networks.</p>",
    ]
    for result in results:
        panel = result["panel"]
        q_rel = "query_only/%s/%s_query_only_official_STRING_v12.png" % (panel, panel)
        e_rel = "first_shell_max5/%s/%s_first_shell_max5_official_STRING_v12.png" % (panel, panel)
        parts.extend(
            [
                '<section class="panel">',
                "<h2>%s - %s</h2>" % (html.escape(panel), html.escape(DISPLAY_NAMES.get(panel, panel))),
                "<p><strong>Submitted proteins (%d):</strong> %s</p>" % (
                    len(result["genes"]), html.escape(", ".join(result["genes"]))
                ),
                '<div class="grid"><div><h3>Query-only (primary)</h3>',
                '<img src="%s" alt="%s query-only official STRING network">' % (html.escape(q_rel), html.escape(panel)),
                '<p><a class="button" href="%s">Open query-only network in STRING</a></p></div>' % html.escape(result["query_link"] or "#"),
                '<div><h3>First-shell maximum 5 (exploratory)</h3>',
                '<img src="%s" alt="%s expanded official STRING network">' % (html.escape(e_rel), html.escape(panel)),
                "<p><strong>Added interactors (%d):</strong> %s</p>" % (
                    len(result["added_names"]),
                    html.escape(", ".join(result["added_names"]) or "none"),
                ),
                '<p><a class="button" href="%s">Open expanded network in STRING</a></p></div></div>' % html.escape(result["expanded_link"] or "#"),
                "</section>",
            ]
        )
    parts.extend(
        [
            '<section class="legend"><h2>Evidence-view legend</h2>',
            "<p>Red: gene fusion; green: gene neighborhood; blue: gene co-occurrence; "
            "purple/pink: experiments; yellow: text mining; light blue/cyan: curated "
            "databases; black: co-expression.</p>",
            "<p>Edges are functional associations and do not necessarily represent direct physical binding. "
            "Node colors are visual aids and do not indicate biological categories.</p></section>",
            "</body></html>",
        ]
    )
    atomic_write_text(path, "\n".join(parts) + "\n")


def write_validated_panel_snapshot(path, panels):
    rows = [
        {"Panel": panel, "Genes": ";".join(genes)}
        for panel, genes in panels.items()
    ]
    write_csv(path, ["Panel", "Genes"], rows)


def create_zip(output_root, zip_path):
    output_root = output_root.resolve()
    zip_path = zip_path.resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = zip_path.with_name(zip_path.name + ".tmp")
    allowed_root_files = {
        "validated_disease_specific_panels.csv",
        "ppi_summary_query_only_all_12.csv",
        "ppi_summary_first_shell_max5_all_12.csv",
    }
    candidates = []
    for file_path in sorted(output_root.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(output_root)
        if relative.parts[0] == "official_figures" or relative.name in allowed_root_files:
            candidates.append((file_path, relative))
    with zipfile.ZipFile(str(temporary), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path, relative in candidates:
            archive.write(str(file_path), arcname=str(relative))
    os.replace(str(temporary), str(zip_path))


def main(argv=None):
    args = parse_args(argv)
    try:
        panels = read_and_validate_panels(args.panels)
        print_panel_validation(panels)
    except PipelineError as exc:
        print("VALIDATION FAILED: %s" % exc, file=sys.stderr)
        return 2

    if args.validate_only:
        print("Validation-only mode complete. No STRING API request was sent.")
        return 0

    if args.panel:
        panels = OrderedDict([(args.panel, panels[args.panel])])
        print("Test mode: processing only %s." % args.panel)

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_validated_panel_snapshot(
        output_root / "validated_disease_specific_panels.csv", panels
    )
    client = StringClient(args.delay, args.max_retries, args.timeout)
    errors = []
    results = []

    for panel, genes in panels.items():
        try:
            result = process_panel(client, panel, genes, output_root, errors)
            if result is not None:
                results.append(result)
        except Exception as exc:  # keep other panels running; report unexpected failures
            record_error(errors, panel, "panel", panel, exc)

    write_combined_outputs(output_root, results, errors)

    zip_path = Path(args.zip_name)
    if not zip_path.is_absolute():
        zip_path = output_root.parent / zip_path
    if not args.no_zip:
        create_zip(output_root, zip_path)

    print("\nCompletion report")
    print("  Requested panels: %d" % len(panels))
    print("  Completed panel records: %d" % len(results))
    for result in results:
        print(
            "  %s: mapped %d/%d; unmapped %s; added first-shell %d"
            % (
                result["panel"],
                len(result["mapped_names"]),
                len(result["genes"]),
                ";".join(result["unmapped"]) or "none",
                len(result["added_names"]),
            )
        )
    print("  Failed requests/files: %d" % len(errors))
    print("  Query-only summary: %s" % (output_root / "ppi_summary_query_only_all_12.csv"))
    print("  Expanded summary: %s" % (output_root / "ppi_summary_first_shell_max5_all_12.csv"))
    print("  Links: %s" % (output_root / "official_figures" / "STRING_network_links_all_12.csv"))
    print("  HTML index: %s" % (output_root / "official_figures" / "STRING_official_networks_all_12.html"))
    if not args.no_zip:
        print("  ZIP: %s" % zip_path)

    if len(results) != len(panels) or errors:
        print("Pipeline finished with incomplete or failed outputs; review STRING_export_errors.tsv.")
        return 1
    print("Pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
