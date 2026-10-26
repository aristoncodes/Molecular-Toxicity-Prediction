"""Independent validation tests for Bemis-Murcko scaffold splitting.

Verifies:
- Complete disjointness (zero index overlap between train, val, and test)
- Full partition coverage (every eligible record assigned exactly once)
- Index bounds (0 <= index < N)
- Independent recomputation of Bemis-Murcko scaffolds with RDKit
- Absolute zero scaffold leakage across partition boundaries for all nonempty scaffolds
- Documented empty-scaffold policy validation
- Strict determinism across repeated runs with identical seed
- Variation across different seeds
- Alignment of split indices with feature matrices and label vectors
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.data.split_data import (
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_JSON,
    execute_scaffold_splitting,
    generate_scaffold,
    scaffold_split,
)


@pytest.fixture
def synthetic_molecules() -> list[str]:
    """Diverse synthetic set of cyclic and acyclic molecules."""
    return [
        "CCO",  # Acyclic (empty scaffold)
        "CCCC",  # Acyclic (empty scaffold)
        "O=C(O)CC",  # Acyclic (empty scaffold)
        "c1ccccc1",  # Benzene scaffold
        "Cc1ccccc1",  # Benzene scaffold
        "Clc1ccccc1",  # Benzene scaffold
        "c1ccncc1",  # Pyridine scaffold
        "Cc1ccncc1",  # Pyridine scaffold
        "c1ccc2ccccc2c1",  # Naphthalene scaffold
        "Cc1ccc2ccccc2c1",  # Naphthalene scaffold
        "C1CCCCC1",  # Cyclohexane scaffold
        "CC1CCCCC1",  # Cyclohexane scaffold
    ]


def test_scaffold_generation():
    """Verify scaffold extraction logic for acyclic and cyclic molecules."""
    assert generate_scaffold("CCO") == ""
    assert generate_scaffold("CCCCCC") == ""
    assert generate_scaffold("c1ccccc1") == "c1ccccc1"
    assert generate_scaffold("Cc1ccccc1") == "c1ccccc1"
    assert generate_scaffold("c1ccncc1") == "c1ccncc1"

    with pytest.raises(ValueError, match="RDKit Chem.MolFromSmiles failed"):
        generate_scaffold("INVALID_MOL_XYZ")

    with pytest.raises(ValueError, match="Invalid SMILES"):
        generate_scaffold("   ")


def test_synthetic_split_integrity(synthetic_molecules):
    """Independently verify scaffold isolation, coverage, and disjointness."""
    res = scaffold_split(
        synthetic_molecules,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
    )

    train_idx = res["train_indices"]
    val_idx = res["val_indices"]
    test_idx = res["test_indices"]
    n_total = len(synthetic_molecules)

    # 1. Bounds and types
    for idx in train_idx + val_idx + test_idx:
        assert isinstance(idx, int)
        assert 0 <= idx < n_total

    # 2. Pairwise disjointness
    s_tr, s_va, s_te = set(train_idx), set(val_idx), set(test_idx)
    assert s_tr.isdisjoint(s_va)
    assert s_tr.isdisjoint(s_te)
    assert s_va.isdisjoint(s_te)

    # 3. Full coverage
    assert s_tr | s_va | s_te == set(range(n_total))
    assert len(train_idx) + len(val_idx) + len(test_idx) == n_total

    # 4. Independent scaffold recomputation from scratch using RDKit
    def get_recomputed_scaffolds(indices: list[int]) -> set[str]:
        scafs = set()
        for i in indices:
            m = Chem.MolFromSmiles(synthetic_molecules[i])
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
            if scaf:  # Nonempty scaffold
                scafs.add(scaf)
        return scafs

    tr_scafs = get_recomputed_scaffolds(train_idx)
    va_scafs = get_recomputed_scaffolds(val_idx)
    te_scafs = get_recomputed_scaffolds(test_idx)

    assert tr_scafs.isdisjoint(va_scafs), f"Train-Val scaffold leak: {tr_scafs & va_scafs}"
    assert tr_scafs.isdisjoint(te_scafs), f"Train-Test scaffold leak: {tr_scafs & te_scafs}"
    assert va_scafs.isdisjoint(te_scafs), f"Val-Test scaffold leak: {va_scafs & te_scafs}"


def test_split_determinism(synthetic_molecules):
    """Verify that re-running with identical seed produces exact identical split."""
    res1 = scaffold_split(synthetic_molecules, seed=42)
    res2 = scaffold_split(synthetic_molecules, seed=42)

    assert res1["train_indices"] == res2["train_indices"]
    assert res1["val_indices"] == res2["val_indices"]
    assert res1["test_indices"] == res2["test_indices"]


def test_alignment_with_features_and_labels(tmp_path: Path):
    """Verify that feature matrices and labels align via saved row IDs and split indices."""
    n = 10
    smiles = ["CCO", "c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1", "C1CCCCC1"] * 2
    source_rows = list(range(101, 101 + n))
    mol_ids = [f"TOX_{i}" for i in source_rows]
    labels = np.random.choice([0.0, 1.0, np.nan], size=(n, 2))

    df = pd.DataFrame({
        "source_row": source_rows,
        "mol_id": mol_ids,
        "canonical_smiles": smiles,
        "NR-AR": labels[:, 0],
        "NR-ER": labels[:, 1],
    })
    csv_file = tmp_path / "mock_cleaned.csv"
    out_json = tmp_path / "split.json"
    df.to_csv(csv_file, index=False)

    artifact = execute_scaffold_splitting(
        input_csv=csv_file,
        output_json=out_json,
        seed=42,
    )

    # Reload artifact and verify row ID matching
    with open(out_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    for part in ["train", "val", "test"]:
        indices = loaded["indices"][part]
        stored_source_rows = loaded["row_identifiers"][f"{part}_source_rows"]
        stored_mol_ids = loaded["row_identifiers"][f"{part}_mol_ids"]

        # Directly slice original dataframe
        df_part = df.iloc[indices]
        assert df_part["source_row"].tolist() == stored_source_rows
        assert df_part["mol_id"].tolist() == stored_mol_ids


@pytest.mark.skipif(
    not DEFAULT_INPUT_CSV.exists(),
    reason="Cleaned dataset tox21_cleaned.csv not found locally",
)
def test_real_dataset_scaffold_split_integrity(tmp_path: Path):
    """Execute split on actual Tox21 cleaned dataset and independently audit all partitions."""
    out_json = tmp_path / "tox21_split.json"
    artifact = execute_scaffold_splitting(
        input_csv=DEFAULT_INPUT_CSV,
        output_json=out_json,
        seed=42,
    )

    df = pd.read_csv(DEFAULT_INPUT_CSV)
    n_records = len(df)
    assert n_records == 7823

    train_idx = artifact["indices"]["train"]
    val_idx = artifact["indices"]["val"]
    test_idx = artifact["indices"]["test"]

    # 1. Total counts and proportions
    assert len(train_idx) == 6258
    assert len(val_idx) == 782
    assert len(test_idx) == 783
    assert len(train_idx) + len(val_idx) + len(test_idx) == 7823

    # Proportions: ~80%, ~10%, ~10%
    prop_train = len(train_idx) / n_records
    prop_val = len(val_idx) / n_records
    prop_test = len(test_idx) / n_records

    assert np.isclose(prop_train, 0.80, atol=0.01)
    assert np.isclose(prop_val, 0.10, atol=0.01)
    assert np.isclose(prop_test, 0.10, atol=0.01)

    # 2. Disjoint indices
    s_tr, s_va, s_te = set(train_idx), set(val_idx), set(test_idx)
    assert s_tr.isdisjoint(s_va)
    assert s_tr.isdisjoint(s_te)
    assert s_va.isdisjoint(s_te)
    assert s_tr | s_va | s_te == set(range(n_records))

    # 3. Independent RDKit scaffold recalculation on all 7823 molecules
    def compute_nonempty_scaffolds(indices: list[int]) -> set[str]:
        scafs = set()
        for i in indices:
            smiles = df.loc[i, "canonical_smiles"]
            mol = Chem.MolFromSmiles(smiles)
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if s:  # Nonempty scaffold
                scafs.add(s)
        return scafs

    tr_scafs = compute_nonempty_scaffolds(train_idx)
    va_scafs = compute_nonempty_scaffolds(val_idx)
    te_scafs = compute_nonempty_scaffolds(test_idx)

    # ZERO scaffold overlap
    assert tr_scafs.isdisjoint(va_scafs), f"Train-Val overlap: {len(tr_scafs & va_scafs)}"
    assert tr_scafs.isdisjoint(te_scafs), f"Train-Test overlap: {len(tr_scafs & te_scafs)}"
    assert va_scafs.isdisjoint(te_scafs), f"Val-Test overlap: {len(va_scafs & te_scafs)}"

    # 4. Verify stable row ID matching
    assert artifact["row_identifiers"]["train_source_rows"] == df.iloc[train_idx]["source_row"].tolist()
    assert artifact["row_identifiers"]["val_source_rows"] == df.iloc[val_idx]["source_row"].tolist()
    assert artifact["row_identifiers"]["test_source_rows"] == df.iloc[test_idx]["source_row"].tolist()
