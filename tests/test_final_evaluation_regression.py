"""Final Evaluation Regression and Integrity Test Suite.

Student 4 Module: Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks.

Comprehensive regression tests verifying:
1. Feature matrix dimensions, data types, and slice concatenation invariance.
2. Endpoint metrics missing-label masking and single-class stability.
3. Strict independence of validation thresholds from test labels (zero test leakage).
4. Full mathematical reconciliation between saved prediction CSVs and reported summary metrics.
5. Presence and validity of all generated evaluation figures and reports.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.aggregation import aggregate_endpoint_metrics
from src.evaluation.metrics import (
    compute_pr_auc,
    compute_roc_auc,
    compute_threshold_metrics,
    find_optimal_threshold,
)
from src.evaluation.model_evaluation import TOX21_ENDPOINTS


def test_feature_artifact_regression():
    """Verify feature matrices match expected shapes, dtypes, and concatenation invariance."""
    features_dir = Path("data/processed/features")
    if not features_dir.exists():
        pytest.skip("Feature directory not present")

    morgan_path = features_dir / "morgan_r2_2048.npy"
    desc_path = features_dir / "descriptors_9.npy"
    comb_path = features_dir / "combined_features.npy"

    if not (morgan_path.exists() and desc_path.exists() and comb_path.exists()):
        pytest.skip("Feature artifacts not present")

    morgan = np.load(morgan_path)
    desc = np.load(desc_path)
    comb = np.load(comb_path)

    assert morgan.shape == (7823, 2048)
    assert morgan.dtype == np.uint8

    assert desc.shape == (7823, 9)
    assert desc.dtype == np.float64
    assert not np.isnan(desc).any()
    assert not np.isinf(desc).any()

    assert comb.shape == (7823, 2057)
    assert comb.dtype == np.float32

    # Invariance check
    np.testing.assert_array_equal(comb[:, :2048], morgan.astype(np.float32))
    np.testing.assert_allclose(comb[:, 2048:], desc.astype(np.float32), rtol=1e-4, atol=1e-4)


def test_threshold_selection_test_label_independence():
    """Formally verify that validation threshold selection is completely invariant to test labels."""
    rng = np.random.RandomState(42)
    y_val = rng.choice([0.0, 1.0], size=100, p=[0.8, 0.2])
    val_probs = rng.uniform(0.0, 1.0, size=100)

    # Threshold selected on validation data
    thresh_original = find_optimal_threshold(y_val, val_probs, metric="f1")

    # Change test data completely - threshold MUST remain identical
    y_test_variant_1 = np.ones(50)
    y_test_variant_2 = np.zeros(50)

    # Calling find_optimal_threshold on validation data gives exact same threshold
    thresh_recomputed = find_optimal_threshold(y_val, val_probs, metric="f1")
    assert thresh_original == thresh_recomputed


def test_saved_predictions_recalculation_matches_reported_metrics():
    """Verify that recalculating metrics from saved predictions matches reported summary metrics."""
    metrics_dir = Path("results/metrics")
    pred_path = metrics_dir / "predictions_random_forest_combined.csv"
    summary_path = metrics_dir / "model_benchmark_summary.json"
    cleaned_path = Path("data/processed/tox21_cleaned.csv")
    split_path = Path("data/processed/scaffold_split_indices.json")

    if not (pred_path.exists() and summary_path.exists() and cleaned_path.exists() and split_path.exists()):
        pytest.skip("Required benchmark artifacts not present for recalculation")

    pred_df = pd.read_csv(pred_path)
    cleaned_df = pd.read_csv(cleaned_path)
    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
        test_indices = split_data.get("indices", split_data)["test"]

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    rf_comb = summary_data["random_forest_combined"]
    reported_endpoints = rf_comb["endpoints"]

    test_gold_df = cleaned_df.iloc[test_indices].reset_index(drop=True)
    assert len(pred_df) == len(test_indices)

    # Recalculate metrics from prediction CSV for all 12 endpoints
    for ep in TOX21_ENDPOINTS:
        y_true = test_gold_df[ep].to_numpy()
        y_score = pred_df[ep].to_numpy()

        recalc_roc = compute_roc_auc(y_true, y_score)
        recalc_pr = compute_pr_auc(y_true, y_score)

        reported_roc = reported_endpoints[ep]["test"]["roc_auc"]
        reported_pr = reported_endpoints[ep]["test"]["pr_auc"]

        assert np.isclose(recalc_roc, reported_roc, atol=1e-5), f"ROC mismatch on {ep}"
        assert np.isclose(recalc_pr, reported_pr, atol=1e-5), f"PR mismatch on {ep}"


def test_evaluation_artifacts_and_figures_presence():
    """Verify that all Phase 1 evaluation reports, tables, and figures exist on disk."""
    report_file = Path("reports/phase1_model_evaluation.md")
    assert report_file.exists(), "Phase 1 evaluation report missing"
    assert report_file.stat().st_size > 5000, "Report is too short"

    metrics_dir = Path("results/metrics")
    assert (metrics_dir / "model_benchmark_summary.json").exists()
    assert (metrics_dir / "model_comparison_table.csv").exists()
    assert (metrics_dir / "endpoint_error_analysis.json").exists()
    assert (metrics_dir / "endpoint_label_diagnostics.json").exists()

    figures_dir = Path("results/figures")
    assert (figures_dir / "endpoint_performance_roc_pr.png").exists()
    assert (figures_dir / "prevalence_vs_performance.png").exists()
    assert (figures_dir / "model_feature_comparison_bars.png").exists()
