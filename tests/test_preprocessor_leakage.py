"""Unit tests for leakage-safe feature preprocessing and scaling."""

from pathlib import Path
import json
import pytest
import numpy as np

from src.features.preprocessor import (
    LeakageSafeScaler,
    PartitionPreprocessor,
    audit_for_preprocessing_leakage,
)


def test_leakage_safe_scaler_basic():
    X_train = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    scaler = LeakageSafeScaler()
    scaler.fit(X_train)

    assert np.allclose(scaler.mean_, [3.0, 20.0])
    # Population variance: [ (4+0+4)/3 = 8/3, (100+0+100)/3 = 200/3 ]
    # std = [sqrt(8/3) ~ 1.63299, sqrt(200/3) ~ 8.16496]
    X_tr_scaled = scaler.transform(X_train)
    assert np.allclose(np.mean(X_tr_scaled, axis=0), [0.0, 0.0])
    assert np.allclose(np.std(X_tr_scaled, axis=0), [1.0, 1.0])

    # Out of sample transform does not alter fitted mean/scale
    X_test = np.array([[7.0, 40.0]])
    X_te_scaled = scaler.transform(X_test)
    assert np.allclose(scaler.mean_, [3.0, 20.0])
    assert np.allclose(X_te_scaled, [[(7 - 3) / scaler.scale_[0], (40 - 20) / scaler.scale_[1]]])


def test_constant_feature_handling():
    X = np.array([[1.0, 5.0], [1.0, 15.0]])  # First column is constant
    scaler = LeakageSafeScaler()
    X_scaled = scaler.fit_transform(X)

    # Constant feature scale defaults to 1.0, preventing NaN/inf
    assert np.allclose(scaler.scale_[0], 1.0)
    assert np.allclose(X_scaled[:, 0], [0.0, 0.0])
    assert not np.isnan(X_scaled).any()


def test_partition_preprocessor_preserves_morgan_bits():
    n_samples = 10
    n_bits = 16
    n_desc = 3

    np.random.seed(42)
    morgan_bits = np.random.choice([0.0, 1.0], size=(n_samples, n_bits)).astype(np.float32)
    descriptors = np.random.randn(n_samples, n_desc).astype(np.float32) * 50 + 100
    X_combined = np.hstack([morgan_bits, descriptors])

    train_indices = [0, 1, 2, 3, 4, 5]
    test_indices = [6, 7, 8, 9]

    preprocessor = PartitionPreprocessor(n_morgan_bits=n_bits, n_descriptors=n_desc)
    preprocessor.fit(X_combined, train_indices=train_indices)
    X_proc = preprocessor.transform(X_combined)

    # 1. Assert Morgan bits are bit-for-bit identical to original
    np.testing.assert_array_equal(X_proc[:, :n_bits], morgan_bits)

    # 2. Assert train descriptors are centered near 0
    train_desc_scaled = X_proc[train_indices, n_bits:]
    assert np.allclose(np.mean(train_desc_scaled, axis=0), 0.0, atol=1e-5)
    assert np.allclose(np.std(train_desc_scaled, axis=0), 1.0, atol=1e-5)

    # 3. Assert test descriptors are NOT strictly mean 0 (shows true out-of-sample scaling)
    test_desc_scaled = X_proc[test_indices, n_bits:]
    assert not np.allclose(np.mean(test_desc_scaled, axis=0), 0.0, atol=1e-2)


def test_audit_for_preprocessing_leakage():
    # Construct synthetic data where test set has a mean shift
    n_bits = 4
    n_desc = 2
    X_train = np.hstack([np.zeros((10, n_bits)), np.ones((10, n_desc)) * 10])
    X_val = np.hstack([np.zeros((5, n_bits)), np.ones((5, n_desc)) * 20])
    X_test = np.hstack([np.zeros((5, n_bits)), np.ones((5, n_desc)) * 50])

    X_full = np.vstack([X_train, X_val, X_test])
    train_idx = list(range(10))
    val_idx = list(range(10, 15))
    test_idx = list(range(15, 20))

    audit_res = audit_for_preprocessing_leakage(
        X_full=X_full,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        n_morgan_bits=n_bits,
    )

    assert audit_res["status"] == "passed"
    assert audit_res["distributional_shift_present"] is True
    assert audit_res["max_mean_discrepancy_if_leaked"] > 0
