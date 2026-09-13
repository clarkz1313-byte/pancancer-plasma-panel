from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
ROADMAP_ROOT = ROOT / "revise_plan" / "external_validation_roadmap"
VRSX_ROOT = ROOT / "revise_plan" / "part_b_multiclass" / "vRSX_v11_locked_reproducer"
V11_ROOT = ROOT / "revise_plan" / "part_b_multiclass" / "v11_seed52_lr_l2_wide_compact_sensitivity"
PART_A_ROOT = ROOT / "revise_plan" / "part_a_pan_cancer"
PART_A_SINGLE_ROOT = ROOT / "revise_plan" / "part_a_single"
VRSX_SCRIPT = SCRIPT_DIR / "02_may_vRSX_seed52_locked_25_lr_l2_reproducer.py"


def load_vrsx_module():
    spec = importlib.util.spec_from_file_location("vrsx_locked", VRSX_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def roadmap_paths() -> dict[str, Path]:
    paths = {
        "root": ROADMAP_ROOT,
        "tables": ROADMAP_ROOT / "tables",
        "figures": ROADMAP_ROOT / "figures",
        "docs": ROADMAP_ROOT / "docs",
        "manifests": ROADMAP_ROOT / "manifests",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def locked_features() -> list[str]:
    path = VRSX_ROOT / "tables" / "locked_selected_features.csv"
    if path.exists():
        return read_required(path).sort_values("rank")["protein"].astype(str).tolist()
    return list(load_vrsx_module().EXPECTED_FEATURES)


def locked_summary() -> pd.Series:
    return read_required(VRSX_ROOT / "tables" / "locked_reproduction_summary.csv").iloc[0]


def reconstruct_locked_model():
    """Rebuild the locked vRSX LR_L2 model without changing selection logic."""
    vrsx = load_vrsx_module()
    full_df = pd.read_csv(vrsx.INPUT)
    protein_columns = vrsx.v8.base.protein_columns_from_df(full_df, vrsx.CLASS_COLUMN, vrsx.SAMPLE_ID_COLUMN)
    classes = sorted(full_df[vrsx.CLASS_COLUMN].dropna().unique().tolist())
    train_df, test_df = vrsx.v8.v1.split_and_impute(
        full_df=full_df,
        protein_columns=protein_columns,
        class_column=vrsx.CLASS_COLUMN,
        sample_id_column=vrsx.SAMPLE_ID_COLUMN,
        seed=vrsx.SEED,
        test_size=vrsx.TEST_SIZE,
        imputer_method=vrsx.IMPUTER,
        knn_neighbors=vrsx.KNN_NEIGHBORS,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    features = locked_features()
    if features != list(vrsx.EXPECTED_FEATURES):
        raise RuntimeError("Locked feature list drifted from vRSX EXPECTED_FEATURES.")

    estimator = vrsx.v8.v2.make_estimator_v2(vrsx.COACH, vrsx.SEED, {})
    estimator.fit(train_df[features], train_df[vrsx.CLASS_COLUMN])
    return {
        "vrsx": vrsx,
        "full_df": full_df,
        "train_df": train_df,
        "test_df": test_df,
        "features": features,
        "classes": classes,
        "estimator": estimator,
    }


def estimator_artifacts(model_bundle: dict[str, object]) -> dict[str, object]:
    estimator = model_bundle["estimator"]
    features = list(model_bundle["features"])
    classes = list(model_bundle["classes"])
    scaler = estimator.named_steps["scale"]
    model = estimator.named_steps["model"]
    model_classes = [str(value) for value in model.classes_.tolist()]
    if model_classes != classes:
        raise RuntimeError(f"Class order drifted: model={model_classes}, expected={classes}")
    coef = pd.DataFrame(model.coef_, index=classes, columns=features)
    intercept = pd.Series(model.intercept_, index=classes, name="intercept")
    scaler_df = pd.DataFrame(
        {
            "feature_order": range(1, len(features) + 1),
            "protein": features,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        }
    )
    return {"coef": coef, "intercept": intercept, "scaler": scaler_df}


def save_locked_model_artifacts(model_bundle: dict[str, object], tables_dir: Path) -> None:
    artifacts = estimator_artifacts(model_bundle)
    coef = artifacts["coef"]
    coef_long = (
        coef.reset_index(names="class_label")
        .melt(id_vars="class_label", var_name="protein", value_name="coefficient")
        .sort_values(["class_label", "coefficient"], ascending=[True, False])
        .reset_index(drop=True)
    )
    coef_long["abs_coefficient"] = coef_long["coefficient"].abs()
    coef_long["within_class_abs_rank"] = coef_long.groupby("class_label")["abs_coefficient"].rank(ascending=False, method="first").astype(int)
    write_csv(coef_long, tables_dir / "v11_lr_l2_class_coefficient_audit.csv")
    write_csv(coef, tables_dir / "v11_lr_l2_class_coefficient_matrix.csv")
    write_csv(artifacts["scaler"], tables_dir / "v11_lr_l2_scaler_parameters.csv")
    write_csv(artifacts["intercept"].reset_index().rename(columns={"index": "class_label"}), tables_dir / "v11_lr_l2_intercepts.csv")


def gene_symbol_table(proteins: list[str], evidence_type: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protein": proteins,
            "gene_symbol": proteins,
            "mapping_status": "identity_gene_symbol",
            "evidence_type": evidence_type,
        }
    )


def simple_hypergeom_enrichment(selected: set[str], background: set[str], gene_sets: dict[str, set[str]], database: str) -> pd.DataFrame:
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    rows: list[dict[str, object]] = []
    for term, genes in gene_sets.items():
        term_genes = set(genes) & background
        if len(term_genes) < 2:
            continue
        overlap = selected & term_genes
        if not overlap:
            continue
        a = len(overlap)
        b = len(selected) - a
        c = len(term_genes) - a
        d = len(background) - a - b - c
        if min(a, b, c, d) < 0:
            continue
        _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append(
            {
                "database": database,
                "term_name": term,
                "overlap_count": a,
                "selected_gene_count": len(selected),
                "term_size": len(term_genes),
                "background_size": len(background),
                "p_value": p_value,
                "overlapping_genes": ";".join(sorted(overlap)),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "database",
                "term_name",
                "overlap_count",
                "selected_gene_count",
                "term_size",
                "background_size",
                "p_value",
                "p_adjusted_bh",
                "overlapping_genes",
            ]
        )
    df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    df["p_adjusted_bh"] = multipletests(df["p_value"], method="fdr_bh")[1]
    return df


def write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def feature_signature(features: list[str]) -> str:
    return ";".join(features)


def max_abs_coefficient_by_protein(coef_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protein in coef_df.columns:
        values = coef_df[protein]
        best_class = values.abs().idxmax()
        rows.append(
            {
                "protein": protein,
                "max_abs_coefficient": float(values.abs().max()),
                "max_abs_coefficient_class": str(best_class),
                "signed_coefficient_at_max_abs": float(values.loc[best_class]),
            }
        )
    return pd.DataFrame(rows)
