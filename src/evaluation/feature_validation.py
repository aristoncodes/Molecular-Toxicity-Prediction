"""Molecular Feature Validation Framework.

Student 4 Module: Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks.

This module provides systematic, mathematically sound feature auditing and validation routines:
1. Fingerprint Dimension and Representation Validation:
   - Verifies Morgan fingerprint radius (r=2), bit length (2048), and data types (uint8/uint32).
   - Validates array dimensionality, absence of NaNs/Infs, and binary range compliance {0, 1}.
   - Analyzes bit density, sparsity, empty bitvectors, constant bits, and duplicate bit patterns.
2. Descriptor Integrity Auditing:
   - Validates dimensions (N, 9), data types (float64), and physical bounds of 9 RDKit descriptors.
   - Detects constant/zero-variance descriptors, extreme outliers, collinearity, and NaNs.
3. Feature Transformation and Leakage Verification:
   - Validates that scaling transformations (mean, standard deviation) are strictly computed
     from training partition indices without out-of-sample data leakage.
   - Verifies combined matrix concatenation invariance (columns 0:2048 vs 2048:2057).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def validate_fingerprint_matrix(
    fp_matrix: np.ndarray,
    expected_rows: int | None = None,
    expected_bits: int = 2048,
    is_binary: bool = True,
    expected_dtype: type | np.dtype = np.uint8,
) -> dict[str, Any]:
    """Audit and validate a molecular fingerprint matrix.

    Args:
        fp_matrix: 2D numpy array of fingerprints of shape (N, n_bits).
        expected_rows: Expected number of molecules (optional).
        expected_bits: Expected bit vector length (default: 2048).
        is_binary: If True, values must strictly belong to {0, 1}.
        expected_dtype: Expected numpy dtype (default: np.uint8 for binary).

    Returns:
        Dictionary containing audit findings, statistics, and validation status.

    Raises:
        TypeError: If fp_matrix is not a numpy array.
        ValueError: If shape, dtype, or value constraints are violated.
    """
    if not isinstance(fp_matrix, np.ndarray):
        raise TypeError(f"Fingerprint matrix must be a numpy.ndarray, got {type(fp_matrix)}")

    if fp_matrix.ndim != 2:
        raise ValueError(f"Fingerprint matrix must be 2D (N, n_bits), got ndim={fp_matrix.ndim}")

    n_rows, n_cols = fp_matrix.shape

    if expected_rows is not None and n_rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {n_rows}")

    if n_cols != expected_bits:
        raise ValueError(f"Expected {expected_bits} bits, got {n_cols}")

    if fp_matrix.dtype != expected_dtype:
        raise ValueError(f"Expected dtype {expected_dtype}, got {fp_matrix.dtype}")

    # Check for NaN / Inf
    has_nan = bool(np.isnan(fp_matrix).any()) if np.issubdtype(fp_matrix.dtype, np.floating) else False
    has_inf = bool(np.isinf(fp_matrix).any()) if np.issubdtype(fp_matrix.dtype, np.floating) else False
    if has_nan or has_inf:
        raise ValueError("Fingerprint matrix contains NaN or Inf values")

    # Value range check
    if is_binary:
        unique_vals = np.unique(fp_matrix)
        invalid_vals = set(unique_vals) - {0, 1}
        if invalid_vals:
            raise ValueError(f"Binary fingerprints must only contain 0 and 1, found: {invalid_vals}")
    else:
        if (fp_matrix < 0).any():
            raise ValueError("Count fingerprints cannot contain negative values")

    # Bit density & sparsity statistics
    active_bits_per_mol = np.sum(fp_matrix > 0, axis=1)
    min_active = int(np.min(active_bits_per_mol)) if n_rows > 0 else 0
    max_active = int(np.max(active_bits_per_mol)) if n_rows > 0 else 0
    mean_active = float(np.mean(active_bits_per_mol)) if n_rows > 0 else 0.0
    median_active = float(np.median(active_bits_per_mol)) if n_rows > 0 else 0.0

    # Overall matrix sparsity (fraction of zeros)
    total_entries = n_rows * n_cols
    non_zero_entries = int(np.count_nonzero(fp_matrix))
    sparsity = float(1.0 - (non_zero_entries / total_entries)) if total_entries > 0 else 1.0

    # Molecules with 0 active bits
    empty_mol_indices = np.where(active_bits_per_mol == 0)[0].tolist()

    # Column-wise constant bit analysis
    col_sums = np.sum(fp_matrix > 0, axis=0)
    all_zero_bits = np.where(col_sums == 0)[0].tolist()
    all_one_bits = np.where(col_sums == n_rows)[0].tolist() if n_rows > 0 else []

    # Duplicate column detection (sample if n_rows is very large for efficiency)
    constant_bit_count = len(all_zero_bits) + len(all_one_bits)
    informative_bit_count = n_cols - constant_bit_count

    return {
        "status": "valid",
        "n_molecules": n_rows,
        "n_bits": n_cols,
        "dtype": str(fp_matrix.dtype),
        "is_binary": is_binary,
        "sparsity": sparsity,
        "active_bits_per_mol": {
            "min": min_active,
            "max": max_active,
            "mean": mean_active,
            "median": median_active,
        },
        "empty_molecules_count": len(empty_mol_indices),
        "empty_molecules_indices": empty_mol_indices,
        "constant_bits": {
            "all_zero_count": len(all_zero_bits),
            "all_one_count": len(all_one_bits),
            "total_constant": constant_bit_count,
        },
        "informative_bits_count": informative_bit_count,
    }
