from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mygene
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
ENRICHMENT_TABLE_DIR = ROOT / "results" / "tables" / "enrichment"
ENRICHMENT_FIGURE_DIR = ROOT / "results" / "figures" / "enrichment"
GENE_SET_DIR = ROOT / "references" / "gene_sets"
MAPPING_CACHE = ENRICHMENT_TABLE_DIR / "protein_id_mapping.csv"
DEFAULT_DATABASES = ["go_bp", "reactome"]
GENE_LIKE_PATTERN = re.compile(r"^[A-Z0-9\-]+$")
MIN_GSEA_TERM_SIZE = 5
MAX_GSEA_TERM_SIZE = 500


COHORT_DEA = {
    "lung": ROOT / "results" / "tables" / "lung_differential_expression_results.csv",
    "glioma": ROOT / "results" / "tables" / "glioma_differential_expression_results.csv",
}

LIBRARIES = {
    "go_bp": "GO_Biological_Process_2023",
    "reactome": "Reactome_2022",
    "kegg": "KEGG_2021_Human",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GO and pathway enrichment with cached Enrichr libraries.")
    parser.add_argument("--cohort", choices=sorted(COHORT_DEA), required=True)
    parser.add_argument("--dea", type=Path, default=None)
    parser.add_argument("--databases", nargs="+", default=DEFAULT_DATABASES, choices=sorted(LIBRARIES))
    parser.add_argument("--top-terms", type=int, default=15)
    parser.add_argument("--gsea-permutations", type=int, default=250)
    return parser.parse_args()


def load_dea_table(path: Path) -> pd.DataFrame:
    de_df = pd.read_csv(path)
    if "lung_mean" in de_df.columns:
        mean_column = "lung_mean"
    elif "glioma_mean" in de_df.columns:
        mean_column = "glioma_mean"
    else:
        raise ValueError("DEA table must contain either lung_mean or glioma_mean.")
    de_df["mean_column"] = mean_column
    return de_df


def looks_like_gene_symbol(protein: str) -> bool:
    return bool(GENE_LIKE_PATTERN.match(protein))


def query_mygene(proteins: list[str]) -> pd.DataFrame:
    mg = mygene.MyGeneInfo()
    records = mg.querymany(
        proteins,
        scopes="symbol,alias",
        fields="symbol,name,entrezgene",
        species="human",
        as_dataframe=False,
        returnall=False,
        verbose=False,
    )
    mapping_rows = []
    for record in records:
        protein = record.get("query")
        if record.get("notfound"):
            if looks_like_gene_symbol(protein):
                mapping_rows.append(
                    {
                        "protein": protein,
                        "mapped_gene": protein,
                        "mapping_status": "identity_fallback",
                        "mapping_source": "identity_fallback",
                        "match_name": "",
                        "entrezgene": "",
                    }
                )
            else:
                mapping_rows.append(
                    {
                        "protein": protein,
                        "mapped_gene": "",
                        "mapping_status": "unmapped",
                        "mapping_source": "none",
                        "match_name": "",
                        "entrezgene": "",
                    }
                )
            continue

        mapping_rows.append(
            {
                "protein": protein,
                "mapped_gene": record.get("symbol", protein),
                "mapping_status": "mapped",
                "mapping_source": "mygene",
                "match_name": record.get("name", ""),
                "entrezgene": record.get("entrezgene", ""),
            }
        )
    return pd.DataFrame(mapping_rows)


def load_or_build_mapping(proteins: list[str]) -> pd.DataFrame:
    ENRICHMENT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if MAPPING_CACHE.exists():
        cache_df = pd.read_csv(MAPPING_CACHE)
    else:
        cache_df = pd.DataFrame(columns=["protein", "mapped_gene", "mapping_status", "mapping_source", "match_name", "entrezgene"])

    cached_proteins = set(cache_df["protein"].tolist()) if not cache_df.empty else set()
    missing = sorted(set(proteins) - cached_proteins)
    if missing:
        queried_df = query_mygene(missing)
        cache_df = pd.concat([cache_df, queried_df], ignore_index=True)
        cache_df = cache_df.drop_duplicates(subset=["protein"], keep="last").sort_values("protein").reset_index(drop=True)
        cache_df.to_csv(MAPPING_CACHE, index=False)

    return cache_df[cache_df["protein"].isin(proteins)].copy()


def fetch_library_text(library_name: str) -> str:
    url = f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={library_name}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def parse_library_text(raw_text: str) -> dict[str, set[str]]:
    gene_sets: dict[str, set[str]] = {}
    for line in raw_text.splitlines():
        parts = line.rstrip().split("\t")
        if not parts or not parts[0]:
            continue
        term = parts[0]
        genes = {gene for gene in parts[1:] if gene}
        if genes:
            gene_sets[term] = genes
    return gene_sets


def load_or_fetch_library(database: str) -> dict[str, set[str]]:
    library_name = LIBRARIES[database]
    library_path = GENE_SET_DIR / f"{library_name}.gmt"
    GENE_SET_DIR.mkdir(parents=True, exist_ok=True)
    if library_path.exists():
        raw_text = library_path.read_text(encoding="utf-8")
    else:
        raw_text = fetch_library_text(library_name)
        library_path.write_text(raw_text, encoding="utf-8")
    return parse_library_text(raw_text)


def collapse_to_genes(de_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    merged = de_df.merge(mapping_df, on="protein", how="left")
    merged = merged[merged["mapping_status"].isin(["mapped", "identity_fallback"])].copy()
    merged["mapped_gene"] = merged["mapped_gene"].fillna(merged["protein"])
    merged["abs_fold_change"] = merged["fold_change"].abs()
    merged = merged.sort_values(["p_adjusted_bonf", "abs_fold_change"], ascending=[True, False])
    collapsed = merged.drop_duplicates(subset=["mapped_gene"], keep="first").reset_index(drop=True)
    return collapsed


def ora(
    selected_genes: set[str],
    background_genes: set[str],
    gene_sets: dict[str, set[str]],
    database: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected_count = len(selected_genes)
    background_count = len(background_genes)

    for term, genes in gene_sets.items():
        term_genes = genes & background_genes
        if len(term_genes) < 2:
            continue

        overlap = selected_genes & term_genes
        if not overlap:
            continue

        a = len(overlap)
        b = selected_count - a
        c = len(term_genes) - a
        d = background_count - a - b - c
        if min(a, b, c, d) < 0:
            continue

        _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append(
            {
                "database": database,
                "term_name": term,
                "overlap_count": a,
                "selected_gene_count": selected_count,
                "term_size": len(term_genes),
                "background_size": background_count,
                "gene_ratio": a / selected_count,
                "p_value": p_value,
                "overlapping_genes": ";".join(sorted(overlap)),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["database", "term_name", "overlap_count", "selected_gene_count", "term_size", "background_size", "gene_ratio", "p_value", "p_adjusted_bh", "neg_log10_fdr", "overlapping_genes"])

    results = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    results["p_adjusted_bh"] = multipletests(results["p_value"], method="fdr_bh")[1]
    results["neg_log10_fdr"] = -np.log10(results["p_adjusted_bh"].clip(lower=np.finfo(float).tiny))
    return results.sort_values(["p_adjusted_bh", "p_value", "overlap_count"], ascending=[True, True, False]).reset_index(drop=True)


def save_dotplot(results: pd.DataFrame, title: str, output_path: Path, top_terms: int) -> None:
    top_df = results.head(top_terms).copy()
    if top_df.empty:
        return
    plot_height = max(4, min(12, 0.45 * len(top_df)))
    fig, ax = plt.subplots(figsize=(10, plot_height))
    sns.scatterplot(
        data=top_df,
        x="gene_ratio",
        y="term_name",
        size="overlap_count",
        hue="neg_log10_fdr",
        palette="viridis",
        sizes=(50, 250),
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Overlap / selected genes")
    ax.set_ylabel("")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def enrichment_score(hit_indices: np.ndarray, hit_weights: np.ndarray, n_genes: int) -> tuple[float, np.ndarray, int]:
    if len(hit_indices) == 0 or n_genes <= len(hit_indices):
        return math.nan, np.array([]), -1

    increments = np.full(n_genes, -1.0 / (n_genes - len(hit_indices)), dtype=float)
    weight_sum = hit_weights.sum()
    if weight_sum <= 0:
        increments[hit_indices] = 1.0 / len(hit_indices)
    else:
        increments[hit_indices] = hit_weights / weight_sum

    running = np.cumsum(increments)
    max_idx = int(np.argmax(running))
    min_idx = int(np.argmin(running))
    max_es = float(running[max_idx])
    min_es = float(running[min_idx])
    if abs(max_es) >= abs(min_es):
        return max_es, running, max_idx
    return min_es, running, min_idx


def leading_edge_genes(ranked_genes: np.ndarray, hit_indices: np.ndarray, peak_index: int, es: float) -> list[str]:
    if peak_index < 0:
        return []
    if es >= 0:
        selected = hit_indices[hit_indices <= peak_index]
    else:
        selected = hit_indices[hit_indices >= peak_index]
    return ranked_genes[selected].tolist()


def preranked_gsea(
    collapsed_df: pd.DataFrame,
    gene_sets: dict[str, set[str]],
    database: str,
    permutations: int,
) -> pd.DataFrame:
    ranked_df = collapsed_df[["mapped_gene", "fold_change"]].copy()
    ranked_df = ranked_df.sort_values("fold_change", ascending=False).reset_index(drop=True)
    ranked_genes = ranked_df["mapped_gene"].to_numpy()
    weight_vector = ranked_df["fold_change"].abs().to_numpy(dtype=float)
    gene_to_index = {gene: idx for idx, gene in enumerate(ranked_genes)}
    background_genes = set(ranked_genes.tolist())
    rng = np.random.default_rng(42)
    n_genes = len(ranked_genes)
    null_cache: dict[int, np.ndarray] = {}
    rows: list[dict[str, object]] = []

    def null_distribution(size: int) -> np.ndarray:
        if size not in null_cache:
            null_scores = np.empty(permutations, dtype=float)
            for i in range(permutations):
                random_hits = np.sort(rng.choice(n_genes, size=size, replace=False))
                random_weights = weight_vector[random_hits]
                null_scores[i], _, _ = enrichment_score(random_hits, random_weights, n_genes)
            null_cache[size] = null_scores
        return null_cache[size]

    for term, genes in gene_sets.items():
        term_genes = sorted((genes & background_genes))
        size = len(term_genes)
        if size < MIN_GSEA_TERM_SIZE or size > MAX_GSEA_TERM_SIZE:
            continue

        hit_indices = np.array(sorted(gene_to_index[gene] for gene in term_genes), dtype=int)
        hit_weights = weight_vector[hit_indices]
        es, _, peak_index = enrichment_score(hit_indices, hit_weights, n_genes)
        if math.isnan(es):
            continue

        null_scores = null_distribution(size)
        p_value = (1.0 + float(np.sum(np.abs(null_scores) >= abs(es)))) / (len(null_scores) + 1.0)
        if es >= 0:
            same_sign = np.abs(null_scores[null_scores >= 0])
        else:
            same_sign = np.abs(null_scores[null_scores < 0])
        if same_sign.size == 0:
            same_sign = np.abs(null_scores)
        nes = es / float(np.mean(same_sign)) if same_sign.size else math.nan
        leading_edge = leading_edge_genes(ranked_genes, hit_indices, peak_index, es)

        rows.append(
            {
                "database": database,
                "term_name": term,
                "term_size": size,
                "enrichment_score": es,
                "nes": nes,
                "p_value": p_value,
                "leading_edge_size": len(leading_edge),
                "leading_edge_genes": ";".join(leading_edge),
                "direction": "positive" if es >= 0 else "negative",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "database",
                "term_name",
                "term_size",
                "enrichment_score",
                "nes",
                "p_value",
                "p_adjusted_bh",
                "neg_log10_fdr",
                "leading_edge_size",
                "leading_edge_genes",
                "direction",
            ]
        )

    results = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    results["p_adjusted_bh"] = multipletests(results["p_value"], method="fdr_bh")[1]
    results["neg_log10_fdr"] = -np.log10(results["p_adjusted_bh"].clip(lower=np.finfo(float).tiny))
    return results.sort_values(["p_adjusted_bh", "p_value", "nes"], ascending=[True, True, False]).reset_index(drop=True)


def save_preranked_barplot(results: pd.DataFrame, title: str, output_path: Path, top_terms: int) -> None:
    if results.empty:
        return

    positive = results[results["direction"] == "positive"].head(max(1, top_terms // 2))
    negative = results[results["direction"] == "negative"].head(max(1, top_terms // 2))
    plot_df = pd.concat([positive, negative], ignore_index=True)
    if plot_df.empty:
        plot_df = results.head(top_terms).copy()
    plot_df = plot_df.sort_values("nes")

    fig_height = max(4, min(12, 0.45 * len(plot_df)))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    colors = plot_df["direction"].map({"positive": "#1f77b4", "negative": "#d62728"}).fillna("#7f7f7f")
    ax.barh(plot_df["term_name"], plot_df["nes"], color=colors)
    ax.axvline(0, color="black", linewidth=1, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Normalized enrichment score (NES)")
    ax.set_ylabel("")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dea_path = args.dea or COHORT_DEA[args.cohort]
    de_df = load_dea_table(dea_path)
    proteins = de_df["protein"].tolist()
    mapping_df = load_or_build_mapping(proteins)
    collapsed_df = collapse_to_genes(de_df, mapping_df)

    selected_df = collapsed_df[collapsed_df["significant_bonf"]].copy()
    selected_genes = set(selected_df["mapped_gene"])
    background_genes = set(collapsed_df["mapped_gene"])

    ENRICHMENT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ENRICHMENT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    ranked_df = collapsed_df[["mapped_gene", "fold_change", "p_adjusted_bonf", "p_adjusted_bh"]].copy()
    ranked_df = ranked_df.rename(columns={"mapped_gene": "gene_symbol"})
    ranked_df.to_csv(ENRICHMENT_TABLE_DIR / f"{args.cohort}_ranked_gene_list.csv", index=False)

    for database in args.databases:
        gene_sets = load_or_fetch_library(database)
        results = ora(selected_genes, background_genes, gene_sets, database)
        preranked_results = preranked_gsea(collapsed_df, gene_sets, database, args.gsea_permutations)
        full_output = ENRICHMENT_TABLE_DIR / f"{args.cohort}_{database}_ora.csv"
        top_output = ENRICHMENT_TABLE_DIR / f"{args.cohort}_{database}_top_terms.csv"
        plot_output = ENRICHMENT_FIGURE_DIR / f"{args.cohort}_{database}_dotplot.png"
        preranked_output = ENRICHMENT_TABLE_DIR / f"{args.cohort}_{database}_preranked_gsea.csv"
        preranked_top_output = ENRICHMENT_TABLE_DIR / f"{args.cohort}_{database}_preranked_top_terms.csv"
        preranked_plot_output = ENRICHMENT_FIGURE_DIR / f"{args.cohort}_{database}_preranked_barplot.png"

        results.to_csv(full_output, index=False)
        results.head(args.top_terms).to_csv(top_output, index=False)
        preranked_results.to_csv(preranked_output, index=False)
        preranked_results.head(args.top_terms).to_csv(preranked_top_output, index=False)
        save_dotplot(results, f"{args.cohort.title()} {database.replace('_', ' ').upper()} enrichment", plot_output, args.top_terms)
        save_preranked_barplot(
            preranked_results,
            f"{args.cohort.title()} {database.replace('_', ' ').upper()} preranked enrichment",
            preranked_plot_output,
            args.top_terms,
        )

        print(f"{database}: {len(results)} enriched terms written to {full_output}")
        print(f"{database}: {len(preranked_results)} preranked terms written to {preranked_output}")

    selected_df[["protein", "mapped_gene", "mapping_status", "mapping_source", "p_adjusted_bonf", "fold_change"]].to_csv(
        ENRICHMENT_TABLE_DIR / f"{args.cohort}_selected_gene_mapping.csv", index=False
    )
    print(f"Selected genes for {args.cohort}: {len(selected_genes)}")
    print(f"Background genes for {args.cohort}: {len(background_genes)}")
    print(f"Mapping cache: {MAPPING_CACHE}")


if __name__ == "__main__":
    main()
