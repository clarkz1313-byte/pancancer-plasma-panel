#!/usr/bin/env python3
"""Assay-background STRING PPI analysis for the locked and single panels.

The primary analysis calls STRING v12 ppi_enrichment with the mapped
1,463-protein assay as the explicit background. A local 10,000-draw edge-count
permutation is retained as a sensitivity analysis. The two p-values are not
assumed to be numerically identical because STRING and the local permutation
need not use the same tail model.
"""
from __future__ import annotations

import csv
import io
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


ROOT = Path("e:/Proteomics")
HERE = ROOT / "revise_plan" / "ppi_string"
MATRIX = ROOT / "data" / "processed" / "filtered_pancancer_data.csv"
PANELS_CSV = HERE / "disease_specific_panels.csv"
LOCKED25_CSV = ROOT / "revise_plan" / "part_b_multiclass" / "vRSX_v11_locked_reproducer" / "tables" / "locked_selected_features.csv"

STRING_VERSION = "12.0"
API_BASE = "https://version-12-0.string-db.org/api"
SPECIES = 9606
REQUIRED_SCORE = 400
NETWORK_TYPE = "functional"
CALLER_IDENTITY = "proteomics_ppi_reaudit_2026_09_04"
N_PERM = 10_000
SEED = 42
METADATA_COLUMNS = {"Sample_ID", "Cancer", "protein_count"}

MAP_CSV = HERE / "string_identifier_map_1463.csv"
NETWORK_TSV = HERE / "background_network_1463_string_ids.tsv"
CUSTOM_V2 = HERE / "ppi_background_permutation_results_v2.csv"
OFFICIAL_V2 = HERE / "ppi_official_assay_background_results_v2.csv"
COMBINED_V2 = HERE / "ppi_reaudit_combined_results_v2.csv"
CANONICAL_CUSTOM = HERE / "ppi_background_permutation_results.csv"


def api_post(method: str, payload: dict[str, object], timeout: int = 600) -> str:
    data = urllib.parse.urlencode(payload).encode()
    req = urllib.request.Request(
        f"{API_BASE}/tsv/{method}",
        data=data,
        headers={"User-Agent": f"{CALLER_IDENTITY}/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"STRING API call failed for {method}: {last_error}")


def assay_symbols() -> list[str]:
    header = pd.read_csv(MATRIX, nrows=0).columns.tolist()
    symbols = sorted({column for column in header if column not in METADATA_COLUMNS})
    if len(symbols) != 1463:
        raise RuntimeError(f"Expected 1,463 assay proteins, found {len(symbols)}")
    return symbols


def load_panels() -> dict[str, list[str]]:
    panels: dict[str, list[str]] = {}
    with PANELS_CSV.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            panels[row["Panel"]] = [g.strip() for g in row["Genes"].split(";") if g.strip()]
    locked = pd.read_csv(LOCKED25_CSV)["protein"].astype(str).tolist()
    pooled = sorted({gene for members in panels.values() for gene in members})
    targets: dict[str, list[str]] = {"LOCKED25": locked, "POOLED12": pooled}
    targets.update(panels)
    return targets


def map_symbols(symbols: list[str]) -> pd.DataFrame:
    if MAP_CSV.exists():
        mapped = pd.read_csv(MAP_CSV)
        if set(mapped["query_symbol"]) == set(symbols):
            return mapped

    text = api_post(
        "get_string_ids",
        {
            "identifiers": "\r".join(symbols),
            "species": SPECIES,
            "limit": 1,
            "echo_query": 1,
            "caller_identity": CALLER_IDENTITY,
        },
    )
    response = pd.read_csv(io.StringIO(text), sep="\t")
    if response.empty:
        raise RuntimeError("STRING returned no identifier mappings")
    response = response.sort_values(["queryItem", "queryIndex", "stringId"])
    response = response.drop_duplicates("queryItem", keep="first")
    by_query = response.set_index("queryItem")
    rows = []
    for symbol in symbols:
        if symbol in by_query.index:
            row = by_query.loc[symbol]
            rows.append(
                {
                    "query_symbol": symbol,
                    "string_id": row["stringId"],
                    "preferred_name": row.get("preferredName", ""),
                    "mapping_status": "mapped",
                }
            )
        else:
            rows.append(
                {
                    "query_symbol": symbol,
                    "string_id": "",
                    "preferred_name": "",
                    "mapping_status": "unmapped",
                }
            )
    mapped = pd.DataFrame(rows)
    mapped.to_csv(MAP_CSV, index=False, encoding="utf-8-sig")
    return mapped


def fetch_background_network(background_ids: list[str]) -> pd.DataFrame:
    if NETWORK_TSV.exists():
        return pd.read_csv(NETWORK_TSV, sep="\t")
    text = api_post(
        "network",
        {
            "identifiers": "\r".join(background_ids),
            "species": SPECIES,
            "required_score": REQUIRED_SCORE,
            "network_type": NETWORK_TYPE,
            "caller_identity": CALLER_IDENTITY,
        },
    )
    network = pd.read_csv(io.StringIO(text), sep="\t")
    network.to_csv(NETWORK_TSV, sep="\t", index=False)
    return network


def bh_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def binomial_ci(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def official_enrichment(target_ids: list[str], background_ids: list[str]) -> dict[str, object]:
    text = api_post(
        "ppi_enrichment",
        {
            "identifiers": "\r".join(target_ids),
            "species": SPECIES,
            "background_string_identifiers": "\r".join(background_ids),
            "caller_identity": CALLER_IDENTITY,
        },
    )
    result = pd.read_csv(io.StringIO(text), sep="\t")
    if result.empty:
        raise RuntimeError("STRING returned an empty ppi_enrichment response")
    return result.iloc[0].to_dict()


def main() -> int:
    symbols = assay_symbols()
    mapping = map_symbols(symbols)
    mapping_ok = mapping.loc[mapping["mapping_status"] == "mapped"].copy()
    symbol_to_id = dict(zip(mapping_ok["query_symbol"], mapping_ok["string_id"]))
    background_ids = sorted(set(symbol_to_id.values()))
    if len(background_ids) != len(mapping_ok):
        raise RuntimeError("Multiple assay symbols mapped to the same STRING identifier")

    print(f"Assay symbols: {len(symbols)}; mapped STRING IDs: {len(background_ids)}")
    network = fetch_background_network(background_ids)
    background_set = set(background_ids)
    edge_pairs = {
        tuple(sorted((a, b)))
        for a, b in zip(network["stringId_A"], network["stringId_B"])
        if a != b and a in background_set and b in background_set
    }
    id_to_index = {identifier: index for index, identifier in enumerate(background_ids)}
    adjacency = np.zeros((len(background_ids), len(background_ids)), dtype=bool)
    for a, b in edge_pairs:
        i, j = id_to_index[a], id_to_index[b]
        adjacency[i, j] = adjacency[j, i] = True

    custom_rows = []
    official_rows = []
    targets = load_panels()
    seeds = np.random.SeedSequence(SEED).spawn(len(targets))
    for (panel, members), panel_seed in zip(targets.items(), seeds):
        mapped_ids = [symbol_to_id[gene] for gene in members if gene in symbol_to_id]
        missing = [gene for gene in members if gene not in symbol_to_id]
        indices = [id_to_index[identifier] for identifier in mapped_ids]
        observed = int(adjacency[np.ix_(indices, indices)].sum() // 2)

        rng = np.random.default_rng(panel_seed)
        null = np.empty(N_PERM, dtype=np.int32)
        for iteration in range(N_PERM):
            selected = rng.choice(len(background_ids), size=len(indices), replace=False)
            null[iteration] = int(adjacency[np.ix_(selected, selected)].sum() // 2)
        exceedances = int((null >= observed).sum())
        empirical_p = (exceedances + 1) / (N_PERM + 1)
        ci_low, ci_high = binomial_ci(exceedances, N_PERM)
        custom_rows.append(
            {
                "panel": panel,
                "display_panel": "DLBCL" if panel == "LYMPH" else panel,
                "n_submitted": len(members),
                "n_mapped_to_string": len(mapped_ids),
                "unmapped": ";".join(missing),
                "observed_edges_score_ge_400": observed,
                "custom_null_mean_edges": float(null.mean()),
                "custom_null_p95_edges": float(np.percentile(null, 95)),
                "custom_exceedances": exceedances,
                "custom_permutations": N_PERM,
                "custom_empirical_p": empirical_p,
                "custom_mc_ci_lower": ci_low,
                "custom_mc_ci_upper": ci_high,
                "custom_null_definition": "uniform size-matched draw from mapped assay network at score>=400",
            }
        )

        official = official_enrichment(mapped_ids, background_ids)
        official_rows.append(
            {
                "panel": panel,
                "display_panel": "DLBCL" if panel == "LYMPH" else panel,
                "n_submitted": len(members),
                "n_mapped_to_string": len(mapped_ids),
                "official_number_of_nodes": official.get("number_of_nodes"),
                "official_number_of_edges": official.get("number_of_edges"),
                "official_expected_number_of_edges": official.get("expected_number_of_edges"),
                "official_p": official.get("p_value"),
                "official_background_n_string_ids": len(background_ids),
                "string_version": STRING_VERSION,
                "official_endpoint": f"{API_BASE}/tsv/ppi_enrichment",
            }
        )
        print(f"{panel:9s} mapped={len(mapped_ids):2d} custom edges={observed:3d} p={empirical_p:.4g}")

    custom_df = pd.DataFrame(custom_rows)
    custom_p = custom_df["custom_empirical_p"].to_numpy(float)
    custom_df["custom_bh_q_14"] = bh_adjust(custom_p)
    custom_df["custom_bonferroni_p_14"] = np.minimum(custom_p * len(custom_df), 1.0)

    official_df = pd.DataFrame(official_rows)
    official_p = pd.to_numeric(official_df["official_p"], errors="raise").to_numpy(float)
    official_df["official_bh_q_14"] = bh_adjust(official_p)
    official_df["official_bonferroni_p_14"] = np.minimum(official_p * len(official_df), 1.0)

    combined = official_df.merge(custom_df, on=["panel", "display_panel", "n_submitted", "n_mapped_to_string"])
    custom_df.to_csv(CUSTOM_V2, index=False, encoding="utf-8-sig")
    custom_df.to_csv(CANONICAL_CUSTOM, index=False, encoding="utf-8-sig")
    official_df.to_csv(OFFICIAL_V2, index=False, encoding="utf-8-sig")
    combined.to_csv(COMBINED_V2, index=False, encoding="utf-8-sig")
    print(f"Wrote {COMBINED_V2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
