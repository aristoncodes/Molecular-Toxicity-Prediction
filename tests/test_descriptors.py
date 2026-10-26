"""Unit tests for physicochemical molecular descriptor calculation.

Verifies:
- Stable feature naming and column ordering (DESCRIPTOR_NAMES, 9 features)
- Numerical accuracy against independent reference calculations on known molecules
- Exact preservation of row sequence in batch calculations
- Handling of invalid SMILES (raising ValueError with index reporting)
- Detection and rejection of NaN/infinite descriptor values
- Single molecule vs batch consistency
"""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

from src.features.descriptors import (
    DESCRIPTOR_METADATA,
    DESCRIPTOR_NAMES,
    compute_descriptors_matrix,
    compute_molecular_descriptors,
)


@pytest.fixture
def known_molecules() -> list[tuple[str, str]]:
    """Known small molecules with common names."""
    return [
        ("Ethanol", "CCO"),
        ("Benzene", "c1ccccc1"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Alanine", "CC(N)C(=O)O"),
    ]


def test_descriptor_constants():
    """Verify descriptor names and metadata count and structure."""
    assert len(DESCRIPTOR_NAMES) == 9
    assert len(DESCRIPTOR_METADATA) == 9

    expected_names = [
        "mol_weight",
        "logp",
        "tpsa",
        "num_h_donors",
        "num_h_acceptors",
        "num_rotatable_bonds",
        "ring_count",
        "heavy_atom_count",
        "fraction_csp3",
    ]
    assert list(DESCRIPTOR_NAMES) == expected_names
    for name in DESCRIPTOR_NAMES:
        assert name in DESCRIPTOR_METADATA
        assert "rdkit_function" in DESCRIPTOR_METADATA[name]
        assert "unit" in DESCRIPTOR_METADATA[name]


def test_known_molecule_ethanol():
    """Independently verify descriptor values for ethanol (CCO)."""
    smiles = "CCO"
    vals = compute_molecular_descriptors(smiles)

    assert len(vals) == 9
    assert vals.dtype == np.float64

    # Independent calculations
    mol = Chem.MolFromSmiles(smiles)
    ref_mw = Descriptors.MolWt(mol)
    ref_logp = Crippen.MolLogP(mol)
    ref_tpsa = Descriptors.TPSA(mol)
    ref_hbd = Lipinski.NumHDonors(mol)
    ref_hba = Lipinski.NumHAcceptors(mol)
    ref_rot = Lipinski.NumRotatableBonds(mol)
    ref_rings = Lipinski.RingCount(mol)
    ref_heavy = Lipinski.HeavyAtomCount(mol)
    ref_fsp3 = Lipinski.FractionCSP3(mol)

    assert np.isclose(vals[0], ref_mw, atol=1e-3)
    assert np.isclose(vals[1], ref_logp, atol=1e-3)
    assert np.isclose(vals[2], ref_tpsa, atol=1e-3)
    assert vals[3] == ref_hbd == 1.0
    assert vals[4] == ref_hba == 1.0
    assert vals[5] == ref_rot == 0.0
    assert vals[6] == ref_rings == 0.0
    assert vals[7] == ref_heavy == 3.0
    assert np.isclose(vals[8], ref_fsp3, atol=1e-3)


def test_known_molecule_aspirin():
    """Independently verify descriptor values for aspirin (acetylsalicylic acid)."""
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    vals = compute_molecular_descriptors(smiles)

    mol = Chem.MolFromSmiles(smiles)
    ref_vals = np.array([
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Descriptors.TPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        Lipinski.RingCount(mol),
        Lipinski.HeavyAtomCount(mol),
        Lipinski.FractionCSP3(mol),
    ], dtype=np.float64)

    np.testing.assert_allclose(vals, ref_vals, rtol=1e-5, atol=1e-5)

    # Sanity checks on known chemistry of Aspirin (C9H8O4)
    assert np.isclose(vals[0], 180.159, atol=1e-2)  # MW ~ 180.16 g/mol
    assert vals[3] == 1.0  # 1 H-donor (carboxylic acid OH)
    assert vals[6] == 1.0  # 1 benzene ring
    assert vals[7] == 13.0  # 13 heavy atoms (9 C + 4 O)
    assert np.isclose(vals[8], 1.0 / 9.0, atol=1e-3)  # 1 sp3 carbon out of 9 total carbons


def test_batch_descriptors_shape_and_order(known_molecules):
    """Verify matrix dimensions, types, and sequence preservation."""
    smiles_list = [s for _, s in known_molecules]
    matrix = compute_descriptors_matrix(smiles_list)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(smiles_list), 9)
    assert matrix.dtype == np.float64
    assert not np.isnan(matrix).any()
    assert not np.isinf(matrix).any()

    # Compare row-by-row with single-molecule calculation
    for i, s in enumerate(smiles_list):
        single_row = compute_molecular_descriptors(s)
        np.testing.assert_allclose(matrix[i], single_row, rtol=1e-7)


def test_row_order_preservation(known_molecules):
    """Verify reversing input list reverses output matrix rows."""
    smiles_list = [s for _, s in known_molecules]
    reversed_list = list(reversed(smiles_list))

    mat_forward = compute_descriptors_matrix(smiles_list)
    mat_reversed = compute_descriptors_matrix(reversed_list)

    np.testing.assert_allclose(mat_forward[::-1], mat_reversed, rtol=1e-7)


def test_invalid_smiles_rejected():
    """Verify clear ValueError when invalid SMILES is encountered."""
    batch = ["CCO", "NOT_A_MOLECULE_123", "c1ccccc1"]
    with pytest.raises(ValueError) as excinfo:
        compute_descriptors_matrix(batch)

    err = str(excinfo.value)
    assert "row index 1" in err
    assert "NOT_A_MOLECULE_123" in err

    # Empty string
    with pytest.raises(ValueError, match="Empty or non-string structure"):
        compute_descriptors_matrix(["CCO", ""])

    # Single invalid molecule
    with pytest.raises(ValueError, match="failed to parse SMILES"):
        compute_molecular_descriptors("XYZ")


def test_empty_list_rejected():
    """Verify empty input raises ValueError."""
    with pytest.raises(ValueError, match="Cannot compute descriptors for an empty list"):
        compute_descriptors_matrix([])


def test_mol_object_support():
    """Verify compute_molecular_descriptors accepts Chem.Mol directly."""
    mol = Chem.MolFromSmiles("c1ccccc1")
    vals = compute_molecular_descriptors(mol)
    assert vals.shape == (9,)
    assert vals[6] == 1.0  # 1 ring
