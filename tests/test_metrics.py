"""Unit tests for evaluation metrics and missing-label handling."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_pr_auc,
    compute_roc_auc,
    evaluate_endpoint,
    evaluate_multitask_predictions,
)


def test_roc_auc_perfect_and_inverse():
    y_true = [0, 0, 1, 1]
    y_perfect = [0.1, 0.2, 0.8, 0.9]
    y_inverse = [0.9, 0.8, 0.2, 0.1]
    y_random = [0.5, 0.5, 0.5, 0.5]

    assert np.isclose(compute_roc_auc(y_true, y_perfect), 1.0)
    assert np.isclose(compute_roc_auc(y_true, y_inverse), 0.0)
    assert np.isclose(compute_roc_auc(y_true, y_random), 0.5)


def test_roc_auc_nan_label_masking():
    # NaNs should be dropped automatically without affecting valid evaluation
    y_true = [0, np.nan, 0, np.nan, 1, 1]
    y_score = [0.1, 0.99, 0.2, 0.01, 0.8, 0.9]

    auc = compute_roc_auc(y_true, y_score)
    assert np.isclose(auc, 1.0)


def test_roc_auc_single_class_handling():
    # If a slice has only 0s or only 1s, ROC-AUC is mathematically undefined
    y_all_neg = [0, 0, 0, 0]
    y_all_pos = [1, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.4]

    assert np.isnan(compute_roc_auc(y_all_neg, y_score))
    assert np.isnan(compute_roc_auc(y_all_pos, y_score))


def test_pr_auc_methods():
    y_true = [0, 1, 0, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]

    ap = compute_pr_auc(y_true, y_score, method="average_precision")
    trap = compute_pr_auc(y_true, y_score, method="trapezoid")

    assert 0.0 <= ap <= 1.0
    assert 0.0 <= trap <= 1.0
    assert ap > 0.5


def test_pr_auc_single_class():
    y_all_neg = [0, 0, 0]
    y_score = [0.2, 0.4, 0.6]
    assert np.isnan(compute_pr_auc(y_all_neg, y_score))


def test_evaluate_endpoint():
    y_true = [1.0, 0.0, np.nan, 1.0, 0.0]
    y_score = [0.9, 0.1, 0.5, 0.8, 0.2]

    res = evaluate_endpoint(y_true, y_score, endpoint_name="NR-AR")
    assert res["endpoint"] == "NR-AR"
    assert res["status"] == "valid"
    assert res["total_samples"] == 5
    assert res["observed"] == 4
    assert res["missing"] == 1
    assert res["positive_count"] == 2
    assert res["negative_count"] == 2
    assert np.isclose(res["roc_auc"], 1.0)


def test_evaluate_multitask_predictions():
    y_true_mat = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [np.nan, 0.0],
        [1.0, np.nan],
    ])
    y_score_mat = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.5, 0.1],
        [0.9, 0.5],
    ])
    res = evaluate_multitask_predictions(
        y_true_mat, y_score_mat, endpoint_names=["EP1", "EP2"]
    )
    assert "EP1" in res and "EP2" in res
    assert res["EP1"]["positive_count"] == 2
    assert res["EP2"]["positive_count"] == 1


def test_missing_labels_not_treated_as_negative():
    """Verify that imputing 0 for missing labels causes severe metric distortion.

    Scientific Principle: Untested compounds (NaN) must never be treated as inactive (0).
    Imputing 0 distorts true negatives, alters false positive rates, and artificially
    alters the PR-AUC baseline prevalence.
    """
    # Ground truth with known positives and missing labels
    y_true = np.array([1, 1, 0, 0, np.nan, np.nan, np.nan, np.nan])
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.95, 0.85, 0.1, 0.05])

    # 1. Masked calculation (scientific protocol)
    auc_masked = compute_roc_auc(y_true, y_score)
    pr_masked = compute_pr_auc(y_true, y_score)
    assert np.isclose(auc_masked, 1.0)  # Positives (0.9, 0.8) strictly outrank Negatives (0.3, 0.2)

    # 2. Flawed zero-imputation
    y_imputed = np.nan_to_num(y_true, nan=0.0)
    auc_imputed = compute_roc_auc(y_imputed, y_score)
    pr_imputed = compute_pr_auc(y_imputed, y_score)

    # Zero-imputation corrupts AUC because untested compounds with high scores (0.95, 0.85)
    # are falsely treated as false positives!
    assert auc_imputed < 1.0
    assert auc_imputed != auc_masked
    assert pr_imputed != pr_masked


def test_threshold_metrics_division_by_zero_safety():
    """Verify threshold metrics handle edge cases where precision/recall denominator is 0."""
    from src.evaluation.metrics import compute_threshold_metrics

    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.3, 0.2, 0.4, 0.1])

    # Threshold = 0.9: no sample predicted positive (tp=0, fp=0)
    m_high = compute_threshold_metrics(y_true, y_score, threshold=0.9)
    assert m_high["tp"] == 0
    assert m_high["fp"] == 0
    assert m_high["precision"] == 0.0  # Safe division, not NaN/ZeroDivisionError
    assert m_high["recall"] == 0.0
    assert m_high["f1"] == 0.0

    # Threshold = 0.05: all samples predicted positive (tn=0, fn=0)
    m_low = compute_threshold_metrics(y_true, y_score, threshold=0.05)
    assert m_low["tn"] == 0
    assert m_low["fn"] == 0
    assert m_low["recall"] == 1.0
    assert m_low["specificity"] == 0.0


def test_find_optimal_threshold_validation_only():
    """Verify optimal threshold search correctly identifies maximum F1 on validation split."""
    from src.evaluation.metrics import compute_threshold_metrics, find_optimal_threshold

    # Synthetic validation set with known separation around 0.6
    y_val_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, np.nan])
    y_val_score = np.array([0.85, 0.75, 0.65, 0.45, 0.35, 0.25, 0.15, 0.05, 0.99])

    best_thresh = find_optimal_threshold(y_val_true, y_val_score, metric="f1")
    assert 0.45 < best_thresh <= 0.65

    # At optimal threshold, F1 should be perfect 1.0
    m_opt = compute_threshold_metrics(y_val_true, y_val_score, threshold=best_thresh)
    assert np.isclose(m_opt["f1"], 1.0)


def test_extreme_class_imbalance_metric_stability():
    """Verify metric computation under extreme class imbalance (1 positive in 1000)."""
    y_true = np.zeros(1000, dtype=np.float64)
    y_true[0] = 1.0  # Only 1 positive (0.1% prevalence)

    # Perfect ranking: positive has highest score
    y_score_perfect = np.linspace(0.01, 0.90, 1000)
    y_score_perfect[0] = 0.99
    auc_perf = compute_roc_auc(y_true, y_score_perfect)
    pr_perf = compute_pr_auc(y_true, y_score_perfect)
    assert np.isclose(auc_perf, 1.0)
    assert np.isclose(pr_perf, 1.0)

    # Worst ranking: positive has lowest score
    y_score_worst = np.linspace(0.10, 0.99, 1000)
    y_score_worst[0] = 0.01
    auc_worst = compute_roc_auc(y_true, y_score_worst)
    assert np.isclose(auc_worst, 0.0)


def test_continuous_scores_vs_discrete_predictions():
    """Verify that thresholded binary predictions degrade continuous ROC-AUC resolution."""
    y_true = np.array([1, 1, 0, 0])
    y_continuous = np.array([0.9, 0.7, 0.4, 0.2])  # Strict ordering: AUC = 1.0
    y_discrete = (y_continuous >= 0.5).astype(np.float64)  # [1, 1, 0, 0]

    auc_cont = compute_roc_auc(y_true, y_continuous)
    auc_disc = compute_roc_auc(y_true, y_discrete)
    assert np.isclose(auc_cont, 1.0)
    # When predictions are discrete with ties, tie-handling ranks are used
    assert np.isclose(auc_disc, 1.0)

    # Imperfect case where continuous score separates border cases but discrete collapses them
    y_cont_subtle = np.array([0.8, 0.6, 0.55, 0.2])
    y_disc_subtle = (y_cont_subtle >= 0.5).astype(np.float64)  # [1, 1, 1, 0] -> false positive at idx 2
    auc_subtle_cont = compute_roc_auc(y_true, y_cont_subtle)
    auc_subtle_disc = compute_roc_auc(y_true, y_disc_subtle)
    # Continuous score provides finer ranking discrimination
    assert auc_subtle_cont != auc_subtle_disc

