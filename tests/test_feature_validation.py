"""Unit tests for molecular fingerprint validation.

Student 4 Module: Validates fingerprint dimensions, bit counts, data types,
sparsity, constant bits, and representation edge cases.
"""

from pathlib import Path
import numpy as np
import pytest

from src.features.fingerprints import compute_morgan_fingerprints, compute_morgan_fingerprint
from src.evaluation.feature_validation import validate_fingerprint_matrix


@pytest.fixture
def diverse_molecules() -> list[str]:
    """Collection of chemically diverse molecules including edge cases."""
    return [
        "C",  # Methane (single carbon)
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene (aromatic)
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine (heterocycle)
        "CC(C)(C)OC(=O)N",  # Boc group
        "C1=CC=C(C=C1)C2=CC=CC=C2",  # Biphenyl
        "C1CCC(CC1)N",  # Cyclohexylamine
        "FC(F)(F)c1ccc(cc1)Cl",  # Halogenated aromatic
        "CC(=O)N[C@@H](C)C(=O)O",  # Chiral amino acid (Alanine)
    ]


def test_fingerprint_matrix_valid_dimensions(diverse_molecules):
    """Verify matrix validation succeeds on valid (N, 2048) uint8 array."""
    fps = compute_morgan_fingerprints(diverse_molecules, radius=2, n_bits=2048)
    audit = validate_fingerprint_matrix(fps, expected_rows=len(diverse_molecules), expected_bits=2048)

    assert audit["status"] == "valid"
    assert audit["n_molecules"] == len(diverse_molecules)
    assert audit["n_bits"] == 2048
    assert audit["dtype"] == "uint8"
    assert audit["is_binary"] is True
    assert 0.0 < audit["sparsity"] < 1.0
    assert audit["active_bits_per_mol"]["min"] > 0
    assert audit["informative_bits_count"] > 0


def test_fingerprint_matrix_invalid_inputs():
    """Verify validation raises appropriate exceptions on invalid inputs."""
    # Non-array input
    with pytest.raises(TypeError, match="must be a numpy.ndarray"):
        validate_fingerprint_matrix([[0, 1], [1, 0]])

    # 1D array instead of 2D
    with pytest.raises(ValueError, match="must be 2D"):
        validate_fingerprint_matrix(np.zeros(2048, dtype=np.uint8))

    # Row mismatch
    fps = np.zeros((5, 2048), dtype=np.uint8)
    with pytest.raises(ValueError, match="Expected 10 rows, got 5"):
        validate_fingerprint_matrix(fps, expected_rows=10)

    # Bit length mismatch
    with pytest.raises(ValueError, match="Expected 2048 bits, got 1024"):
        validate_fingerprint_matrix(np.zeros((5, 1024), dtype=np.uint8), expected_bits=2048)

    # Dtype mismatch
    with pytest.raises(ValueError, match="Expected dtype"):
        validate_fingerprint_matrix(np.zeros((5, 2048), dtype=np.float32))


def test_fingerprint_non_binary_values():
    """Verify error raised when binary array contains non-binary entries."""
    fps = np.zeros((3, 2048), dtype=np.uint8)
    fps[0, 5] = 2  # Non-binary count
    with pytest.raises(ValueError, match="must only contain 0 and 1"):
        validate_fingerprint_matrix(fps, is_binary=True)


def test_fingerprint_nan_inf_detection():
    """Verify floating point array with NaNs or Infs is caught."""
    fps = np.zeros((3, 2048), dtype=np.float32)
    fps[1, 10] = np.nan
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        validate_fingerprint_matrix(fps, expected_dtype=np.float32, is_binary=False)


def test_fingerprint_constant_and_empty_bits():
    """Verify accurate detection of constant zero and all-one columns."""
    fps = np.zeros((10, 2048), dtype=np.uint8)
    # Set bit 0 to 1 for all molecules
    fps[:, 0] = 1
    # Set bit 1 to 1 for 3 molecules
    fps[0:3, 1] = 1
    # Leave remaining 2046 bits as all-zero

    audit = validate_fingerprint_matrix(fps, expected_rows=10, expected_bits=2048)
    assert audit["constant_bits"]["all_one_count"] == 1
    assert audit["constant_bits"]["all_zero_count"] == 2046
    assert audit["informative_bits_count"] == 1
    assert audit["empty_molecules_count"] == 0


def test_real_dataset_fingerprint_artifact():
    """Verify existing Morgan fingerprint artifact if present on disk."""
    artifact_path = Path("data/processed/features/morgan_r2_2048.npy")
    if not artifact_path.exists():
        pytest.skip("Real data fingerprint artifact not present")

    fp_matrix = np.load(artifact_path)
    audit = validate_fingerprint_matrix(
        fp_matrix,
        expected_rows=7823,
        expected_bits=2048,
        expected_dtype=np.uint8,
    )
    assert audit["status"] == "valid"
    assert audit["n_molecules"] == 7823
    assert audit["n_bits"] == 2048
    # Tox21 dataset has high sparsity in 2048-bit Morgan space (~97-99%)
    assert 0.90 < audit["sparsity"] < 0.999
    assert audit["empty_molecules_count"] == 0


def test_descriptor_matrix_valid_dimensions(diverse_molecules):
    """Verify descriptor validation succeeds on valid (N, 9) float64 array."""
    from src.features.descriptors import compute_descriptors_matrix
    from src.evaluation.feature_validation import validate_descriptor_matrix

    descs = compute_descriptors_matrix(diverse_molecules)
    audit = validate_descriptor_matrix(descs, expected_rows=len(diverse_molecules), expected_cols=9)

    assert audit["status"] == "valid"
    assert audit["n_molecules"] == len(diverse_molecules)
    assert audit["n_descriptors"] == 9
    assert audit["dtype"] == "float64"
    assert len(audit["constant_descriptors"]) == 0
    # Check that statistics for each descriptor are calculated
    assert "mol_weight" in audit["statistics"]
    assert audit["statistics"]["mol_weight"]["min"] > 0


def test_descriptor_matrix_invalid_bounds():
    """Verify physical bounds violations in descriptors are caught."""
    from src.evaluation.feature_validation import validate_descriptor_matrix

    # Invalid negative molecular weight
    descs = np.zeros((3, 9), dtype=np.float64)
    descs[:, 0] = -10.0
    with pytest.raises(ValueError, match="mol_weight must be strictly positive"):
        validate_descriptor_matrix(descs)

    # Invalid negative TPSA
    descs = np.ones((3, 9), dtype=np.float64)
    descs[:, 2] = -5.0
    with pytest.raises(ValueError, match="tpsa must be non-negative"):
        validate_descriptor_matrix(descs)

    # Invalid fraction_csp3 (> 1.0)
    descs = np.ones((3, 9), dtype=np.float64)
    descs[:, 8] = 1.5
    with pytest.raises(ValueError, match="fraction_csp3 must be bounded"):
        validate_descriptor_matrix(descs)


def test_descriptor_matrix_nan_inf():
    """Verify descriptor validation rejects NaNs and Infs."""
    from src.evaluation.feature_validation import validate_descriptor_matrix

    descs = np.ones((3, 9), dtype=np.float64)
    descs[0, 1] = np.nan
    with pytest.raises(ValueError, match="contains NaNs"):
        validate_descriptor_matrix(descs)

    descs = np.ones((3, 9), dtype=np.float64)
    descs[1, 3] = np.inf
    with pytest.raises(ValueError, match="contains NaNs .* or Infs"):
        validate_descriptor_matrix(descs)


def test_combined_feature_matrix_invariance(diverse_molecules):
    """Verify horizontal concatenation invariance of Morgan bits and 9 descriptors."""
    from src.features.descriptors import compute_descriptors_matrix
    from src.evaluation.feature_validation import validate_combined_feature_matrix

    fps = compute_morgan_fingerprints(diverse_molecules, radius=2, n_bits=2048)
    descs = compute_descriptors_matrix(diverse_molecules)
    combined = np.hstack([fps.astype(np.float32), descs.astype(np.float32)])

    audit = validate_combined_feature_matrix(combined, fps, descs)
    assert audit["status"] == "valid"
    assert audit["morgan_slice_match"] is True
    assert audit["desc_slice_match"] is True
    assert audit["max_desc_diff_float32"] < 1e-4


def test_real_dataset_descriptors_and_combined_artifacts():
    """Verify real dataset descriptors and combined features artifacts if present."""
    from src.evaluation.feature_validation import (
        validate_descriptor_matrix,
        validate_combined_feature_matrix,
    )

    desc_path = Path("data/processed/features/descriptors_9.npy")
    combined_path = Path("data/processed/features/combined_features.npy")
    morgan_path = Path("data/processed/features/morgan_r2_2048.npy")

    if not (desc_path.exists() and combined_path.exists() and morgan_path.exists()):
        pytest.skip("Real data feature artifacts not present")

    desc_matrix = np.load(desc_path)
    combined_matrix = np.load(combined_path)
    morgan_matrix = np.load(morgan_path)

    desc_audit = validate_descriptor_matrix(desc_matrix, expected_rows=7823, expected_cols=9)
    assert desc_audit["status"] == "valid"
    assert desc_audit["n_molecules"] == 7823
    assert len(desc_audit["constant_descriptors"]) == 0

    combined_audit = validate_combined_feature_matrix(combined_matrix, morgan_matrix, desc_matrix)
    assert combined_audit["status"] == "valid"
    assert combined_audit["morgan_slice_match"] is True
    assert combined_audit["desc_slice_match"] is True

