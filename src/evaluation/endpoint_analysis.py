"""Endpoint Analysis and Label Diagnostics Framework.

Student 4 Module: Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks.

This module provides comprehensive diagnostic analysis for Tox21 toxicity endpoints:
1. Endpoint-Level Label Statistics:
   - Valid sample counts, missing counts, missingness percentage.
   - Positive/negative class counts, positive prevalence, and class imbalance ratios.
2. Partition Breakdown:
   - Evaluates label distributions across Train, Validation, and Test scaffold splits.
   - Detects class diversity issues (single-class partitions, extreme sparsity).
   - Audits prevalence drift between scaffold partitions.
3. Safe Handling:
   - Treats missing labels strictly as unknown (NaN) without zero-imputation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard 12 Tox21 toxicity endpoints
TOX21_ENDPOINTS: tuple[str, ...] = (
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)


def compute_single_endpoint_diagnostics(
    labels: Sequence[float | int | None] | np.ndarray,
    endpoint_name: str = "",
    min_minority_samples: int = 5,
) -> dict[str, Any]:
    """Compute exhaustive label distribution diagnostics for a single endpoint.

    Args:
        labels: 1D array of labels (0.0, 1.0, or NaN).
        endpoint_name: Name of the endpoint.
        min_minority_samples: Minimum count in minority class for sufficient diversity.

    Returns:
        Dictionary with sample counts, missingness, prevalence, imbalance ratio, and diversity status.
    """
    arr = np.asarray(labels, dtype=np.float64).flatten()
    total_samples = len(arr)

    valid_mask = ~np.isnan(arr)
    observed = int(np.sum(valid_mask))
    missing = total_samples - observed
    missing_pct = float((missing / total_samples) * 100.0) if total_samples > 0 else 0.0

    if observed == 0:
        return {
            "endpoint": endpoint_name,
            "total_samples": total_samples,
            "observed_samples": 0,
            "missing_samples": missing,
            "missing_percentage": missing_pct,
            "positive_count": 0,
            "negative_count": 0,
            "prevalence": float(np.nan),
            "imbalance_ratio": float(np.nan),
            "diversity_status": "unobserved",
        }

    observed_vals = arr[valid_mask]
    n_pos = int(np.sum(observed_vals == 1.0))
    n_neg = int(np.sum(observed_vals == 0.0))

    # Validate binary values
    if n_pos + n_neg != observed:
        unique_vals = set(np.unique(observed_vals))
        raise ValueError(f"Endpoint {endpoint_name} contains non-binary values: {unique_vals}")

    prevalence = float(n_pos / observed) if observed > 0 else 0.0
    imbalance_ratio = float(n_neg / n_pos) if n_pos > 0 else float(np.nan)

    if n_pos == 0 or n_neg == 0:
        diversity_status = "single_class"
    elif min(n_pos, n_neg) < min_minority_samples:
        diversity_status = "critically_imbalanced"
    elif imbalance_ratio > 10.0:
        diversity_status = "highly_imbalanced"
    else:
        diversity_status = "sufficient"

    return {
        "endpoint": endpoint_name,
        "total_samples": total_samples,
        "observed_samples": observed,
        "missing_samples": missing,
        "missing_percentage": round(missing_pct, 2),
        "positive_count": n_pos,
        "negative_count": n_neg,
        "prevalence": round(prevalence, 4),
        "imbalance_ratio": round(imbalance_ratio, 2) if not np.isnan(imbalance_ratio) else None,
        "diversity_status": diversity_status,
    }


def diagnose_endpoint_labels(
    data: pd.DataFrame | np.ndarray,
    endpoint_names: Sequence[str] = TOX21_ENDPOINTS,
    split_indices: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Audit label distributions and partition statistics across all endpoints.

    Args:
        data: DataFrame containing endpoint columns, or 2D array of shape (N, n_endpoints).
        endpoint_names: List of endpoint names matching columns or array axis 1.
        split_indices: Optional mapping {'train': [...], 'val': [...], 'test': [...]}.

    Returns:
        Structured dictionary with global and partition-level diagnostics.
    """
    if isinstance(data, pd.DataFrame):
        for ep in endpoint_names:
            if ep not in data.columns:
                raise KeyError(f"Endpoint '{ep}' not found in DataFrame columns")
        mat = data[list(endpoint_names)].to_numpy(dtype=np.float64)
    else:
        mat = np.asarray(data, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[1] != len(endpoint_names):
            raise ValueError(f"Expected shape (N, {len(endpoint_names)}), got {mat.shape}")

    n_samples, n_endpoints = mat.shape

    # Overall dataset diagnostics
    overall_diagnostics: dict[str, dict[str, Any]] = {}
    for i, ep in enumerate(endpoint_names):
        overall_diagnostics[ep] = compute_single_endpoint_diagnostics(mat[:, i], endpoint_name=ep)

    result: dict[str, Any] = {
        "n_samples": n_samples,
        "n_endpoints": n_endpoints,
        "endpoints": list(endpoint_names),
        "overall": overall_diagnostics,
    }

    # Partition-specific breakdown if split indices provided
    if split_indices is not None:
        partition_diagnostics: dict[str, dict[str, dict[str, Any]]] = {}
        prevalence_by_partition: dict[str, dict[str, float]] = {ep: {} for ep in endpoint_names}

        for part_name, idxs in split_indices.items():
            part_mat = mat[idxs, :]
            part_results = {}
            for i, ep in enumerate(endpoint_names):
                diag = compute_single_endpoint_diagnostics(part_mat[:, i], endpoint_name=ep)
                part_results[ep] = diag
                prevalence_by_partition[ep][part_name] = diag["prevalence"]
            partition_diagnostics[part_name] = part_results

        # Audit prevalence drift and single-class risks across partitions
        drift_analysis: dict[str, dict[str, Any]] = {}
        for ep in endpoint_names:
            p_train = prevalence_by_partition[ep].get("train", float(np.nan))
            p_val = prevalence_by_partition[ep].get("val", float(np.nan))
            p_test = prevalence_by_partition[ep].get("test", float(np.nan))

            has_single_class_partition = any(
                partition_diagnostics[p][ep]["diversity_status"] == "single_class"
                for p in split_indices.keys()
            )

            max_drift = float(np.nanmax([
                abs(p_train - p_val) if not np.isnan(p_train) and not np.isnan(p_val) else 0.0,
                abs(p_train - p_test) if not np.isnan(p_train) and not np.isnan(p_test) else 0.0,
            ]))

            drift_analysis[ep] = {
                "train_prevalence": p_train,
                "val_prevalence": p_val,
                "test_prevalence": p_test,
                "max_drift_from_train": round(max_drift, 4),
                "has_single_class_partition": has_single_class_partition,
            }

        result["partitions"] = partition_diagnostics
        result["partition_drift"] = drift_analysis

    return result


def format_diagnostics_markdown_table(diagnostics: dict[str, Any]) -> str:
    """Format endpoint diagnostics into a clean markdown table.

    Args:
        diagnostics: Output from diagnose_endpoint_labels.

    Returns:
        Markdown table string.
    """
    lines = [
        "| Endpoint | Total | Observed | Missing (%) | Positives | Negatives | Prevalence | Imbalance | Diversity Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for ep, d in diagnostics["overall"].items():
        imb_str = f"{d['imbalance_ratio']}:1" if d["imbalance_ratio"] is not None else "N/A"
        lines.append(
            f"| `{ep}` | {d['total_samples']} | {d['observed_samples']} | "
            f"{d['missing_samples']} ({d['missing_percentage']}%) | {d['positive_count']} | "
            f"{d['negative_count']} | {d['prevalence']:.4f} | {imb_str} | `{d['diversity_status']}` |"
        )

    return "\n".join(lines)


def run_label_diagnostics_cli(
    data_path: str = "data/processed/tox21_cleaned.csv",
    split_path: str = "data/processed/scaffold_split_indices.json",
    output_path: str = "results/metrics/endpoint_label_diagnostics.json",
) -> dict[str, Any]:
    """Execute diagnostics from CLI and save results."""
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {data_file}")

    df = pd.read_csv(data_file)

    split_indices = None
    split_file = Path(split_path)
    if split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
            split_indices = split_data.get("indices", split_data)

    diagnostics = diagnose_endpoint_labels(df, split_indices=split_indices)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    logger.info("Saved endpoint label diagnostics to %s", out_file)
    return diagnostics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Tox21 Endpoint Label Diagnostics")
    parser.add_argument("--data", default="data/processed/tox21_cleaned.csv", help="Path to cleaned CSV")
    parser.add_argument("--split", default="data/processed/scaffold_split_indices.json", help="Path to split indices JSON")
    parser.add_argument("--output", default="results/metrics/endpoint_label_diagnostics.json", help="Output JSON path")
    args = parser.parse_args()

    run_label_diagnostics_cli(data_path=args.data, split_path=args.split, output_path=args.output)
