"""Unit tests for endpoint label diagnostics.

Student 4 Module: Validates sample counts, missingness, prevalence,
imbalance ratios, and partition-level diversity across toxicity endpoints.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.endpoint_analysis import (
    TOX21_ENDPOINTS,
    compute_single_endpoint_diagnostics,
    diagnose_endpoint_labels,
    format_diagnostics_markdown_table,
)


def test_single_endpoint_diagnostics_normal():
    """Verify diagnostics for a well-behaved balanced endpoint."""
    labels = np.array([1, 0, 1, 0, np.nan, 0, 1, 0, 0, 0])
    diag = compute_single_endpoint_diagnostics(labels, endpoint_name="TEST_EP")

    assert diag["endpoint"] == "TEST_EP"
    assert diag["total_samples"] == 10
    assert diag["observed_samples"] == 9
    assert diag["missing_samples"] == 1
    assert diag["missing_percentage"] == 10.0
    assert diag["positive_count"] == 3
    assert diag["negative_count"] == 6
    assert abs(diag["prevalence"] - (3 / 9)) < 1e-4
    assert abs(diag["imbalance_ratio"] - 2.0) < 1e-4
    assert diag["diversity_status"] == "critically_imbalanced" or diag["diversity_status"] == "sufficient"


def test_single_endpoint_diagnostics_all_missing():
    """Verify handling when all values in an endpoint are NaN."""
    labels = np.array([np.nan, np.nan, np.nan])
    diag = compute_single_endpoint_diagnostics(labels, endpoint_name="ALL_NAN")

    assert diag["total_samples"] == 3
    assert diag["observed_samples"] == 0
    assert diag["missing_samples"] == 3
    assert diag["missing_percentage"] == 100.0
    assert diag["positive_count"] == 0
    assert diag["negative_count"] == 0
    assert np.isnan(diag["prevalence"])
    assert diag["diversity_status"] == "unobserved"


def test_single_endpoint_diagnostics_single_class():
    """Verify single class detection (all negative or all positive)."""
    all_neg = np.array([0, 0, 0, np.nan, 0])
    diag_neg = compute_single_endpoint_diagnostics(all_neg, endpoint_name="ALL_NEG")
    assert diag_neg["positive_count"] == 0
    assert diag_neg["negative_count"] == 4
    assert diag_neg["diversity_status"] == "single_class"

    all_pos = np.array([1, 1, 1, 1])
    diag_pos = compute_single_endpoint_diagnostics(all_pos, endpoint_name="ALL_POS")
    assert diag_pos["positive_count"] == 4
    assert diag_pos["negative_count"] == 0
    assert diag_pos["diversity_status"] == "single_class"


def test_single_endpoint_non_binary_rejection():
    """Verify non-binary values (e.g. 2 or -1) raise ValueError."""
    invalid_labels = np.array([0, 1, 2, 0])
    with pytest.raises(ValueError, match="non-binary values"):
        compute_single_endpoint_diagnostics(invalid_labels, endpoint_name="BAD")


def test_diagnose_endpoint_labels_multitask_and_splits():
    """Verify multitask diagnosis with train/val/test split breakdown."""
    n_samples = 100
    n_eps = len(TOX21_ENDPOINTS)

    # Synthetic matrix with ~10% positives and ~15% NaNs
    rng = np.random.RandomState(42)
    mat = rng.choice([0.0, 1.0, np.nan], size=(n_samples, n_eps), p=[0.75, 0.10, 0.15])

    splits = {
        "train": list(range(0, 80)),
        "val": list(range(80, 90)),
        "test": list(range(90, 100)),
    }

    result = diagnose_endpoint_labels(mat, endpoint_names=TOX21_ENDPOINTS, split_indices=splits)

    assert result["n_samples"] == 100
    assert result["n_endpoints"] == 12
    assert "overall" in result
    assert "partitions" in result
    assert "train" in result["partitions"]
    assert "val" in result["partitions"]
    assert "test" in result["partitions"]
    assert "partition_drift" in result

    # Check table formatter
    table_str = format_diagnostics_markdown_table(result)
    assert "| `NR-AR` |" in table_str
    assert "| `SR-p53` |" in table_str


def test_real_dataset_endpoint_label_diagnostics():
    """Verify label diagnostics on real cleaned dataset if available."""
    data_path = Path("data/processed/tox21_cleaned.csv")
    split_path = Path("data/processed/scaffold_split_indices.json")

    if not data_path.exists():
        pytest.skip("Real cleaned data not available")

    df = pd.read_csv(data_path)
    split_indices = None
    if split_path.exists():
        import json
        with open(split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
            split_indices = split_data.get("indices", split_data)

    diag = diagnose_endpoint_labels(df, split_indices=split_indices)
    assert diag["n_samples"] == 7823
    assert diag["n_endpoints"] == 12

    # Check known Tox21 properties from cleaned dataset (accounting for 8 excluded structures)
    nr_ar = diag["overall"]["NR-AR"]
    assert nr_ar["positive_count"] == 308
    assert nr_ar["negative_count"] == 6950
    assert nr_ar["observed_samples"] == 7258
    assert nr_ar["missing_samples"] == 565

    if split_indices:
        # Check that no partition in the real split has single class for any endpoint
        for ep in TOX21_ENDPOINTS:
            for part in ["train", "val", "test"]:
                status = diag["partitions"][part][ep]["diversity_status"]
                assert status != "single_class", f"{part} {ep} has single_class!"
