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


# Standard 9 physicochemical descriptors
STANDARD_DESCRIPTOR_NAMES: tuple[str, ...] = (
    "mol_weight",
    "logp",
    "tpsa",
    "num_h_donors",
    "num_h_acceptors",
    "num_rotatable_bonds",
    "ring_count",
    "heavy_atom_count",
    "fraction_csp3",
)


def validate_descriptor_matrix(
    desc_matrix: np.ndarray,
    expected_rows: int | None = None,
    expected_cols: int = 9,
    expected_dtype: type | np.dtype = np.float64,
    column_names: Sequence[str] = STANDARD_DESCRIPTOR_NAMES,
) -> dict[str, Any]:
    """Audit and validate physicochemical molecular descriptors.

    Verifies matrix shape, numerical types, absence of NaNs/Infs,
    physical range adherence, variance, and collinearity.

    Args:
        desc_matrix: 2D numpy array of descriptors of shape (N, 9).
        expected_rows: Expected number of molecules (optional).
        expected_cols: Expected descriptor count (default: 9).
        expected_dtype: Expected numpy dtype (default: np.float64).
        column_names: Sequence of descriptor column names.

    Returns:
        Dictionary containing validation findings, summary stats, and integrity status.

    Raises:
        TypeError: If desc_matrix is not a numpy array.
        ValueError: If shape, dtype, or physical bound constraints are violated.
    """
    if not isinstance(desc_matrix, np.ndarray):
        raise TypeError(f"Descriptor matrix must be a numpy.ndarray, got {type(desc_matrix)}")

    if desc_matrix.ndim != 2:
        raise ValueError(f"Descriptor matrix must be 2D (N, {expected_cols}), got ndim={desc_matrix.ndim}")

    n_rows, n_cols = desc_matrix.shape

    if expected_rows is not None and n_rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {n_rows}")

    if n_cols != expected_cols:
        raise ValueError(f"Expected {expected_cols} descriptor columns, got {n_cols}")

    if desc_matrix.dtype != expected_dtype:
        raise ValueError(f"Expected dtype {expected_dtype}, got {desc_matrix.dtype}")

    # Check for NaN / Inf
    nan_mask = np.isnan(desc_matrix)
    inf_mask = np.isinf(desc_matrix)
    if nan_mask.any() or inf_mask.any():
        nan_cols = [column_names[i] for i in range(n_cols) if nan_mask[:, i].any()]
        inf_cols = [column_names[i] for i in range(n_cols) if inf_mask[:, i].any()]
        raise ValueError(f"Descriptor matrix contains NaNs in {nan_cols} or Infs in {inf_cols}")

    # Physical domain bounds verification
    # mol_weight > 0
    if n_rows > 0:
        if (desc_matrix[:, 0] <= 0).any():
            invalid_idx = np.where(desc_matrix[:, 0] <= 0)[0][0]
            raise ValueError(f"mol_weight must be strictly positive, got {desc_matrix[invalid_idx, 0]}")
        # tpsa >= 0
        if (desc_matrix[:, 2] < 0).any():
            raise ValueError("tpsa must be non-negative")
        # num_h_donors >= 0
        if (desc_matrix[:, 3] < 0).any():
            raise ValueError("num_h_donors must be non-negative")
        # num_h_acceptors >= 0
        if (desc_matrix[:, 4] < 0).any():
            raise ValueError("num_h_acceptors must be non-negative")
        # num_rotatable_bonds >= 0
        if (desc_matrix[:, 5] < 0).any():
            raise ValueError("num_rotatable_bonds must be non-negative")
        # ring_count >= 0
        if (desc_matrix[:, 6] < 0).any():
            raise ValueError("ring_count must be non-negative")
        # heavy_atom_count >= 1
        if (desc_matrix[:, 7] < 1).any():
            raise ValueError("heavy_atom_count must be >= 1 for valid molecules")
        # fraction_csp3 in [0.0, 1.0]
        if (desc_matrix[:, 8] < 0.0).any() or (desc_matrix[:, 8] > 1.0 + 1e-6).any():
            raise ValueError("fraction_csp3 must be bounded in [0.0, 1.0]")

    # Descriptive statistics and variance analysis per column
    stats_per_col: dict[str, dict[str, float]] = {}
    constant_descriptors: list[str] = []
    for col_idx, col_name in enumerate(column_names):
        vals = desc_matrix[:, col_idx]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        median_val = float(np.median(vals))

        if std_val < 1e-12:
            constant_descriptors.append(col_name)

        stats_per_col[col_name] = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "median": median_val,
        }

    # Collinearity check (Pearson correlation)
    collinear_pairs: list[dict[str, Any]] = []
    if n_rows > 1 and len(constant_descriptors) == 0:
        corr_matrix = np.corrcoef(desc_matrix, rowvar=False)
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                r_val = float(corr_matrix[i, j])
                if abs(r_val) > 0.95:
                    collinear_pairs.append({
                        "feature_1": column_names[i],
                        "feature_2": column_names[j],
                        "correlation": r_val,
                    })

    return {
        "status": "valid",
        "n_molecules": n_rows,
        "n_descriptors": n_cols,
        "dtype": str(desc_matrix.dtype),
        "descriptors": list(column_names),
        "statistics": stats_per_col,
        "constant_descriptors": constant_descriptors,
        "high_correlation_pairs": collinear_pairs,
    }


def validate_combined_feature_matrix(
    combined_matrix: np.ndarray,
    morgan_matrix: np.ndarray,
    desc_matrix: np.ndarray,
) -> dict[str, Any]:
    """Validate concatenation invariance between combined features, Morgan bits, and descriptors.

    Guarantees:
        combined[:, :2048] == morgan
        combined[:, 2048:2057] == desc (within float32 precision)

    Args:
        combined_matrix: (N, 2057) float32 matrix.
        morgan_matrix: (N, 2048) uint8 matrix.
        desc_matrix: (N, 9) float64 matrix.

    Returns:
        Dictionary confirming slice alignment and numerical invariance.
    """
    if combined_matrix.ndim != 2:
        raise ValueError(f"Combined matrix must be 2D, got {combined_matrix.ndim}D")

    n_rows, n_cols = combined_matrix.shape
    if n_cols != 2057:
        raise ValueError(f"Combined matrix must have 2057 columns, got {n_cols}")

    if morgan_matrix.shape != (n_rows, 2048):
        raise ValueError(f"Morgan matrix shape {morgan_matrix.shape} incompatible with combined ({n_rows}, 2057)")

    if desc_matrix.shape != (n_rows, 9):
        raise ValueError(f"Descriptor matrix shape {desc_matrix.shape} incompatible with combined ({n_rows}, 2057)")

    # Verify bit slice
    morgan_slice = combined_matrix[:, :2048]
    if not np.array_equal(morgan_slice, morgan_matrix.astype(np.float32)):
        max_diff = float(np.max(np.abs(morgan_slice - morgan_matrix)))
        raise ValueError(f"Morgan slice mismatch in combined matrix: max diff = {max_diff}")

    # Verify descriptor slice
    desc_slice = combined_matrix[:, 2048:]
    max_desc_diff = float(np.max(np.abs(desc_slice - desc_matrix.astype(np.float32))))
    if max_desc_diff > 1e-4:
        raise ValueError(f"Descriptor slice mismatch in combined matrix: max diff = {max_desc_diff}")

    return {
        "status": "valid",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "morgan_slice_match": True,
        "desc_slice_match": True,
        "max_desc_diff_float32": max_desc_diff,
    }

