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
