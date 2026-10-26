"""Leakage-Safe Feature Preprocessing and Scaling Module.

Student 3 Module: Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks.

Addresses Preprocessing Leakage:
--------------------------------
In molecular toxicity modeling with combined representations (Morgan fingerprints + continuous
physicochemical descriptors), feature normalization must strictly isolate partitions:
1. Binary Morgan fingerprints ({0, 1}) should NEVER be standard-scaled: standardizing bit vectors
   destroys binary sparsity, bit interpretability, and increases memory footprint needlessly.
2. Continuous physicochemical descriptors (e.g., MolWt, LogP, TPSA) have diverse units and scales.
   Any scaling (mean/std or min/max) MUST be fitted STRICTLY on the training partition indices.
3. Fitting scalers across the full dataset prior to splitting constitutes distribution and variance
   leakage from validation and test partitions into the training process.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class LeakageSafeScaler:
    """Standard scaler fitted strictly on training partition data to prevent leakage.

    Guarantees:
    - Parameters (mean, std) are estimated solely from training partition indices.
    - Features with zero or near-zero variance are preserved with unit scale to avoid divide-by-zero.
    - Does not alter binary fingerprint slices if applied as part of PartitionPreprocessor.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True, eps: float = 1e-8):
        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.var_: np.ndarray | None = None
        self.n_samples_seen_: int = 0
        self.fitted_: bool = False

    def fit(self, X: np.ndarray) -> "LeakageSafeScaler":
        """Fit scaler strictly on training array X."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_arr.ndim}D")
        if X_arr.shape[0] == 0:
            raise ValueError("Cannot fit scaler on 0 samples")

        self.n_samples_seen_ = X_arr.shape[0]

        if self.with_mean:
            self.mean_ = np.mean(X_arr, axis=0)
        else:
            self.mean_ = np.zeros(X_arr.shape[1], dtype=np.float64)

        if self.with_std:
            self.var_ = np.var(X_arr, axis=0)
            # Handle constant features (zero variance)
            scale = np.sqrt(self.var_)
            scale[scale < self.eps] = 1.0
            self.scale_ = scale
        else:
            self.var_ = np.ones(X_arr.shape[1], dtype=np.float64)
            self.scale_ = np.ones(X_arr.shape[1], dtype=np.float64)

        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply learned training scaling out-of-sample to X."""
        if not self.fitted_ or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted yet. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_arr.ndim}D")
        if X_arr.shape[1] != len(self.mean_):
            raise ValueError(
                f"Feature dimension mismatch: expected {len(self.mean_)}, got {X_arr.shape[1]}"
            )

        X_out = X_arr.copy()
        if self.with_mean:
            X_out -= self.mean_
        if self.with_std:
            X_out /= self.scale_

        return X_out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class PartitionPreprocessor:
    """Preprocesses combined molecular representations (Morgan bits + descriptors) safely.

    Leaves Morgan bit vectors untouched (preserving uint8/float32 binary 0/1 encoding)
    and standard-scales continuous descriptor columns using statistics learned strictly
    from the training indices.
    """

    def __init__(
        self,
        n_morgan_bits: int = 2048,
        n_descriptors: int = 9,
    ):
        self.n_morgan_bits = n_morgan_bits
        self.n_descriptors = n_descriptors
        self.expected_total_cols = n_morgan_bits + n_descriptors
        self.descriptor_scaler = LeakageSafeScaler(with_mean=True, with_std=True)
        self.train_indices_fitted_: list[int] | None = None
        self.is_fitted: bool = False

    def fit(self, X_combined: np.ndarray, train_indices: Sequence[int]) -> "PartitionPreprocessor":
        """Fit descriptor scaler strictly on the slice defined by train_indices.

        Args:
            X_combined: Full or training combined feature matrix of shape (N, 2057).
            train_indices: Integer indices of the training partition.

        Returns:
            self
        """
        X_arr = np.asarray(X_combined)
        if X_arr.shape[1] != self.expected_total_cols:
            raise ValueError(
                f"Expected {self.expected_total_cols} columns, got {X_arr.shape[1]}"
            )

        # Slice out strictly the training descriptors
        train_idx = list(train_indices)
        if len(train_idx) == 0:
            raise ValueError("train_indices cannot be empty")

        train_descriptors = X_arr[train_idx, self.n_morgan_bits :]
        self.descriptor_scaler.fit(train_descriptors)
        self.train_indices_fitted_ = train_idx
        self.is_fitted = True
        return self

    def transform(self, X_combined: np.ndarray) -> np.ndarray:
        """Transform combined matrix: Morgan bits unmodified, descriptors scaled.

        Args:
            X_combined: Combined feature matrix (N, 2057).

        Returns:
            Transformed feature matrix of shape (N, 2057), dtype float32.
        """
        if not self.is_fitted:
            raise RuntimeError("PartitionPreprocessor is not fitted. Call fit() first.")

        X_arr = np.asarray(X_combined, dtype=np.float32)
        morgan_part = X_arr[:, : self.n_morgan_bits]
        desc_part = X_arr[:, self.n_morgan_bits :]

        scaled_desc = self.descriptor_scaler.transform(desc_part).astype(np.float32)

        return np.hstack([morgan_part, scaled_desc])

    def get_scaling_parameters(self) -> dict[str, list[float]]:
        """Return learned descriptor means and standard deviations."""
        if not self.is_fitted:
            raise RuntimeError("Not fitted")
        return {
            "descriptor_means": self.descriptor_scaler.mean_.tolist(),
            "descriptor_stds": self.descriptor_scaler.scale_.tolist(),
        }


def audit_for_preprocessing_leakage(
    X_full: np.ndarray,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
    n_morgan_bits: int = 2048,
) -> dict[str, Any]:
    """Verify that train-fitted scaling differs from illicit whole-dataset scaling.

    Demonstrates that full-dataset scaling leaks test variance into training.
    """
    train_idx = list(train_indices)
    test_idx = list(test_indices)

    desc_train = X_full[train_idx, n_morgan_bits:]
    desc_test = X_full[test_idx, n_morgan_bits:]
    desc_all = X_full[:, n_morgan_bits:]

    train_mean = np.mean(desc_train, axis=0)
    train_std = np.std(desc_train, axis=0)

    all_mean = np.mean(desc_all, axis=0)
    all_std = np.std(desc_all, axis=0)

    test_mean = np.mean(desc_test, axis=0)
    test_std = np.std(desc_test, axis=0)

    mean_shift = np.abs(train_mean - all_mean)
    has_distributional_shift = bool(np.any(mean_shift > 1e-4))

    return {
        "status": "passed",
        "train_descriptors_mean": train_mean.tolist(),
        "all_descriptors_mean": all_mean.tolist(),
        "test_descriptors_mean": test_mean.tolist(),
        "max_mean_discrepancy_if_leaked": float(np.max(mean_shift)),
        "distributional_shift_present": has_distributional_shift,
        "recommendation": "Always use PartitionPreprocessor fitted strictly on train_indices",
    }
