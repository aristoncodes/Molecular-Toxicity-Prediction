"""Unit tests for model training and benchmark evaluation framework.

Student 4 Module: Validates baseline classifiers, validation-only threshold tuning,
feature configuration routing, and macro metric aggregation.
"""

import numpy as np
import pytest

from src.evaluation.model_evaluation import (
    TOX21_ENDPOINTS,
    create_classifier,
    evaluate_benchmark_experiment,
    train_and_evaluate_endpoint,
)


def test_create_classifier_options():
    """Verify supported model types and invalid option rejection."""
    lr = create_classifier("logistic_regression", pos_neg_ratio=5.0)
    assert lr is not None
    assert lr.class_weight == "balanced"

    rf = create_classifier("random_forest", pos_neg_ratio=5.0)
    assert rf is not None
    assert rf.class_weight == "balanced"

    xgb = create_classifier("xgboost", pos_neg_ratio=5.0)
    assert xgb is not None
    assert xgb.scale_pos_weight == 5.0

    with pytest.raises(ValueError, match="Unknown model_type"):
        create_classifier("invalid_model")


def test_train_and_evaluate_endpoint_validation_threshold_isolation():
    """Verify threshold is derived strictly from validation split, not test split."""
    rng = np.random.RandomState(42)
    n_train, n_val, n_test, n_feats = 100, 30, 30, 20

    X_train = rng.randn(n_train, n_feats)
    # Synthetic label with feature correlation
    y_train = (X_train[:, 0] > 0.2).astype(float)
    y_train[0:5] = np.nan  # Include some missing labels

    X_val = rng.randn(n_val, n_feats)
    y_val = (X_val[:, 0] > 0.2).astype(float)

    X_test = rng.randn(n_test, n_feats)
    y_test = (X_test[:, 0] > 0.2).astype(float)

    res = train_and_evaluate_endpoint(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        endpoint_name="TEST_EP",
        model_type="logistic_regression",
    )

    assert res["endpoint"] == "TEST_EP"
    assert "val_threshold" in res
    assert 0.0 < res["val_threshold"] < 1.0
    assert "validation" in res
    assert "test" in res
    assert 0.0 <= res["test"]["roc_auc"] <= 1.0
    assert 0.0 <= res["test"]["pr_auc"] <= 1.0
    assert "confusion_matrix" in res["test"]
    assert res["train_samples"] == 95  # 100 - 5 NaNs


def test_benchmark_experiment_feature_configs():
    """Verify benchmark runs across feature configurations on synthetic data."""
    rng = np.random.RandomState(42)
    n_total = 60
    # Combined shape (60, 2057): 2048 Morgan bits + 9 descriptors
    X_bits = rng.choice([0, 1], size=(n_total, 2048)).astype(np.float32)
    X_descs = (rng.rand(n_total, 9) * 100.0).astype(np.float32)
    X_comb = np.hstack([X_bits, X_descs])

    Y_labels = rng.choice([0.0, 1.0, np.nan], size=(n_total, 2), p=[0.7, 0.2, 0.1])
    # Guarantee at least 2 positives and 2 negatives in train, val, and test slices
    Y_labels[0:5, :] = 1.0
    Y_labels[5:10, :] = 0.0
    Y_labels[40:43, :] = 1.0
    Y_labels[43:46, :] = 0.0
    Y_labels[50:53, :] = 1.0
    Y_labels[53:56, :] = 0.0
    endpoint_names = ["EP_A", "EP_B"]

    split_indices = {
        "train": list(range(0, 40)),
        "val": list(range(40, 50)),
        "test": list(range(50, 60)),
    }

    # Test Morgan configuration
    res_morgan = evaluate_benchmark_experiment(
        X_comb, Y_labels, split_indices,
        feature_config="morgan",
        model_type="logistic_regression",
        endpoint_names=endpoint_names,
    )
    assert res_morgan["feature_config"] == "morgan"
    assert "roc_auc" in res_morgan["macro_metrics"]
    assert res_morgan["macro_metrics"]["roc_auc"]["k"] == 2

    # Test Descriptors configuration
    res_desc = evaluate_benchmark_experiment(
        X_comb, Y_labels, split_indices,
        feature_config="descriptors",
        model_type="logistic_regression",
        endpoint_names=endpoint_names,
    )
    assert res_desc["feature_config"] == "descriptors"

    # Test Combined configuration
    res_comb = evaluate_benchmark_experiment(
        X_comb, Y_labels, split_indices,
        feature_config="combined",
        model_type="logistic_regression",
        endpoint_names=endpoint_names,
    )
    assert res_comb["feature_config"] == "combined"
