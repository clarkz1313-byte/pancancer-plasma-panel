from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 52
TEST_SIZE = 0.3
MIN_GROUP_SIZE = 5
KNN_NEIGHBORS = 5
IMPUTER_METHODS = ("iterative", "knn", "median")
METADATA_COLUMNS = {"Sample_ID", "Cancer", "Label", "protein_count"}


COHORT_CONFIG = {
    "lung": {
        "input": ROOT / "data" / "processed" / "balanced_lung_dataset.csv",
        "train_output": ROOT / "data" / "processed" / "lung_train_imputed.csv",
        "test_output": ROOT / "data" / "processed" / "lung_test_imputed.csv",
        "dea_output": ROOT / "results" / "tables" / "lung_differential_expression_results.csv",
        "case_mean_column": "lung_mean",
        "case_n_column": "lung_n",
        "control_n_column": "control_n",
    },
    "glioma": {
        "input": ROOT / "data" / "processed" / "balanced_glioma_dataset.csv",
        "train_output": ROOT / "data" / "processed" / "glioma_train_imputed.csv",
        "test_output": ROOT / "data" / "processed" / "glioma_test_imputed.csv",
        "dea_output": ROOT / "results" / "tables" / "glioma_differential_expression_results.csv",
        "case_mean_column": "glioma_mean",
        "case_n_column": "glioma_n",
        "control_n_column": "control_n",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run train/test split, KNN imputation, and Mann-Whitney DEA for a cohort."
    )
    parser.add_argument("--cohort", choices=sorted(COHORT_CONFIG), required=True)
    parser.add_argument("--input", type=Path, default=None, help="Override the default balanced cohort input.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--knn-neighbors", type=int, default=KNN_NEIGHBORS)
    parser.add_argument("--imputer", choices=sorted(IMPUTER_METHODS), default="knn")
    return parser.parse_args()


def get_protein_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in METADATA_COLUMNS]


def build_imputer(method: str, knn_neighbors: int, seed: int) -> KNNImputer | SimpleImputer | IterativeImputer:
    if method == "knn":
        return KNNImputer(n_neighbors=knn_neighbors)
    if method == "median":
        return SimpleImputer(strategy="median")
    return IterativeImputer(
        initial_strategy="median",
        max_iter=15,
        n_nearest_features=50,
        random_state=seed,
        sample_posterior=False,
        skip_complete=True,
    )


def split_and_impute(
    df: pd.DataFrame,
    protein_columns: list[str],
    seed: int,
    test_size: float,
    knn_neighbors: int,
    imputer_method: str = "knn",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df[protein_columns].copy()
    y = df["Label"].copy()
    metadata = df[["Sample_ID", "Cancer"]].copy()

    x_train, x_test, y_train, y_test, meta_train, meta_test = train_test_split(
        x,
        y,
        metadata,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    imputer = build_imputer(imputer_method, knn_neighbors, seed)
    imputer.fit(x_train)

    x_train_imputed = pd.DataFrame(
        imputer.transform(x_train),
        columns=protein_columns,
        index=x_train.index,
    )
    x_test_imputed = pd.DataFrame(
        imputer.transform(x_test),
        columns=protein_columns,
        index=x_test.index,
    )

    train_df = pd.concat(
        [
            meta_train.reset_index(drop=True),
            y_train.reset_index(drop=True).rename("Label"),
            x_train_imputed.reset_index(drop=True),
        ],
        axis=1,
    )
    test_df = pd.concat(
        [
            meta_test.reset_index(drop=True),
            y_test.reset_index(drop=True).rename("Label"),
            x_test_imputed.reset_index(drop=True),
        ],
        axis=1,
    )
    return train_df, test_df


def differential_expression(
    train_df: pd.DataFrame,
    protein_columns: list[str],
    case_mean_column: str,
    case_n_column: str,
    control_n_column: str,
) -> pd.DataFrame:
    case_group = train_df[train_df["Label"] == 1]
    control_group = train_df[train_df["Label"] == 0]

    results: list[dict[str, float | int | str | bool]] = []
    for protein in protein_columns:
        case_values = case_group[protein].dropna()
        control_values = control_group[protein].dropna()

        if len(case_values) < MIN_GROUP_SIZE or len(control_values) < MIN_GROUP_SIZE:
            continue

        statistic, p_value = mannwhitneyu(case_values, control_values, alternative="two-sided")
        n_case = len(case_values)
        n_control = len(control_values)
        effect_size = 1 - (2 * statistic) / (n_case * n_control)

        results.append(
            {
                "protein": protein,
                case_mean_column: case_values.mean(),
                "control_mean": control_values.mean(),
                "fold_change": case_values.mean() - control_values.mean(),
                "p_value": p_value,
                "effect_size": effect_size,
                case_n_column: n_case,
                control_n_column: n_control,
            }
        )

    de_df = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)
    bh_corrected = multipletests(de_df["p_value"], method="fdr_bh", alpha=0.05)
    bonf_corrected = multipletests(de_df["p_value"], method="bonferroni", alpha=0.05)

    de_df["p_adjusted_bh"] = bh_corrected[1]
    de_df["significant_bh"] = bh_corrected[0]
    de_df["p_adjusted_bonf"] = bonf_corrected[1]
    de_df["significant_bonf"] = bonf_corrected[0]
    return de_df


def main() -> None:
    args = parse_args()
    config = COHORT_CONFIG[args.cohort]
    input_path = args.input or config["input"]

    df = pd.read_csv(input_path)
    protein_columns = get_protein_columns(df)
    train_df, test_df = split_and_impute(
        df=df,
        protein_columns=protein_columns,
        seed=args.seed,
        test_size=args.test_size,
        knn_neighbors=args.knn_neighbors,
        imputer_method=args.imputer,
    )
    de_df = differential_expression(
        train_df=train_df,
        protein_columns=protein_columns,
        case_mean_column=config["case_mean_column"],
        case_n_column=config["case_n_column"],
        control_n_column=config["control_n_column"],
    )

    config["train_output"].parent.mkdir(parents=True, exist_ok=True)
    config["dea_output"].parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config["train_output"], index=False)
    test_df.to_csv(config["test_output"], index=False)
    de_df.to_csv(config["dea_output"], index=False)

    bonf_threshold = 0.05 / len(de_df)
    print(f"Cohort: {args.cohort}")
    print(f"Input: {input_path}")
    print(f"Imputer: {args.imputer}")
    print(f"Proteins tested: {len(de_df)}")
    print(f"Bonferroni threshold: {bonf_threshold:.6g}")
    print(f"BH-significant proteins: {int(de_df['significant_bh'].sum())}")
    print(f"Bonferroni-significant proteins: {int(de_df['significant_bonf'].sum())}")
    print(f"Saved train imputed: {config['train_output']}")
    print(f"Saved test imputed: {config['test_output']}")
    print(f"Saved DEA table: {config['dea_output']}")


if __name__ == "__main__":
    main()
