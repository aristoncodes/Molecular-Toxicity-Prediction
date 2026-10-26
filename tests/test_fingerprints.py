"""Unit tests for Morgan fingerprint generation.

Verifies:
- Shape and bit length configurability
- Determinism and reproducibility
- Strict error raising on unparsable / empty SMILES with row index
- Row order preservation
- Binary vs count fingerprint representations
- Consistency between single-item and batch APIs
"""

import numpy as np
import pytest
from rdkit import Chem

from src.features.fingerprints import (
    DEFAULT_MORGAN_N_BITS,
    DEFAULT_MORGAN_RADIUS,
    compute_morgan_fingerprint,
    compute_morgan_fingerprints,
    get_morgan_generator,
)


@pytest.fixture
def sample_smiles() -> list[str]:
    """Diverse small synthetic molecules for testing."""
    return [
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)Nc1ccc(O)cc1",  # Paracetamol
    ]


def test_generator_creation():
    """Verify generator initialization and validation."""
    gen = get_morgan_generator(radius=2, n_bits=2048)
    assert gen is not None

    with pytest.raises(ValueError, match="Morgan radius must be >= 1"):
        get_morgan_generator(radius=0, n_bits=2048)

    with pytest.raises(ValueError, match="Fingerprint size .* must be >= 1"):
        get_morgan_generator(radius=2, n_bits=0)


def test_fingerprint_shape_and_dtype(sample_smiles):
    """Verify output shape and data types for default and custom bit lengths."""
    # Default 2048 bits
    fps_bin = compute_morgan_fingerprints(sample_smiles)
    assert isinstance(fps_bin, np.ndarray)
    assert fps_bin.shape == (len(sample_smiles), DEFAULT_MORGAN_N_BITS)
    assert fps_bin.dtype == np.uint8

    # Custom 1024 bits
    fps_1024 = compute_morgan_fingerprints(sample_smiles, n_bits=1024)
    assert fps_1024.shape == (len(sample_smiles), 1024)
    assert fps_1024.dtype == np.uint8

    # Custom 512 bits, radius 3
    fps_512_r3 = compute_morgan_fingerprints(sample_smiles, radius=3, n_bits=512)
    assert fps_512_r3.shape == (len(sample_smiles), 512)


def test_fingerprint_determinism(sample_smiles):
    """Verify fingerprints are strictly deterministic across calls."""
    fps_1 = compute_morgan_fingerprints(sample_smiles, radius=2, n_bits=2048)
    fps_2 = compute_morgan_fingerprints(sample_smiles, radius=2, n_bits=2048)
    np.testing.assert_array_equal(fps_1, fps_2)

    # Individual call determinism
    for s in sample_smiles:
        fp_a = compute_morgan_fingerprint(s, radius=2, n_bits=2048)
        fp_b = compute_morgan_fingerprint(s, radius=2, n_bits=2048)
        np.testing.assert_array_equal(fp_a, fp_b)


def test_single_vs_batch_consistency(sample_smiles):
    """Verify batch output matches individual molecule calculations row by row."""
    batch_fps = compute_morgan_fingerprints(sample_smiles, radius=2, n_bits=2048)
    for i, s in enumerate(sample_smiles):
        single_fp = compute_morgan_fingerprint(s, radius=2, n_bits=2048)
        np.testing.assert_array_equal(batch_fps[i], single_fp)


def test_row_order_preservation(sample_smiles):
    """Verify input row order is strictly preserved."""
    reversed_smiles = list(reversed(sample_smiles))
    fps_normal = compute_morgan_fingerprints(sample_smiles)
    fps_reversed = compute_morgan_fingerprints(reversed_smiles)

    np.testing.assert_array_equal(fps_normal[::-1], fps_reversed)


def test_binary_vs_count_fingerprints(sample_smiles):
    """Verify binary vs count representations."""
    fps_bin = compute_morgan_fingerprints(sample_smiles, count=False)
    fps_cnt = compute_morgan_fingerprints(sample_smiles, count=True)

    assert fps_bin.dtype == np.uint8
    assert fps_cnt.dtype == np.uint32

    # Binary must only contain 0 and 1
    unique_bin_vals = set(np.unique(fps_bin))
    assert unique_bin_vals.issubset({0, 1})

    # For any bit set in binary, count must be >= 1
    for i in range(len(sample_smiles)):
        bin_active = (fps_bin[i] == 1)
        assert np.all(fps_cnt[i][bin_active] >= 1)
        assert np.all(fps_cnt[i][~bin_active] == 0)


def test_invalid_smiles_rejected():
    """Verify clear ValueError raised for invalid SMILES with index reporting."""
    invalid_batch = ["CCO", "INVALID_SMILES_XYZ", "c1ccccc1"]

    with pytest.raises(ValueError) as excinfo:
        compute_morgan_fingerprints(invalid_batch)

    err_msg = str(excinfo.value)
    assert "row index 1" in err_msg
    assert "INVALID_SMILES_XYZ" in err_msg

    # Empty string
    with pytest.raises(ValueError, match="Empty or non-string structure"):
        compute_morgan_fingerprints(["CCO", "   ", "c1ccccc1"])

    # Single invalid molecule
    with pytest.raises(ValueError, match="failed to parse SMILES"):
        compute_morgan_fingerprint("INVALID_MOL")


def test_empty_list_rejected():
    """Verify empty input list raises ValueError."""
    with pytest.raises(ValueError, match="Cannot compute fingerprints for an empty list"):
        compute_morgan_fingerprints([])


def test_mol_object_support():
    """Verify function works with Chem.Mol objects directly."""
    mol = Chem.MolFromSmiles("c1ccccc1")
    fp = compute_morgan_fingerprint(mol)
    assert fp.shape == (DEFAULT_MORGAN_N_BITS,)
    assert fp.dtype == np.uint8
    assert np.sum(fp) > 0
