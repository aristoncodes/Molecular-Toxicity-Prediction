"""Unit tests for split indices and scaffold isolation auditor."""

from pathlib import Path
import json
import pytest
import pandas as pd
import numpy as np

from src.audit.split_audit import (
    EXPECTED_ACYCLIC_COUNT,
    EXPECTED_TEST_COUNT,
    EXPECTED_TOTAL_RECORDS,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_VAL_COUNT,
    audit_acyclic_molecule_handling,
    audit_row_identifier_alignment,
    audit_scaffold_isolation,
    audit_split_disjointness_and_coverage,
    run_split_audit,
)


@pytest.fixture
def mock_split_setup(tmp_path: Path):
    clean_csv = tmp_path / "mock_clean.csv"
    split_json = tmp_path / "mock_split.json"

    # 4 molecules: 2 acyclic, 2 cyclic with different scaffolds
    smiles = ["CCO", "CCCC", "c1ccccc1", "c1ccncc1"]
    df = pd.DataFrame({
        "source_row": [1, 2, 3, 4],
        "mol_id": ["M1", "M2", "M3", "M4"],
        "canonical_smiles": smiles,
    })
    df.to_csv(clean_csv, index=False)

    # Train: 0, 1 (acyclics); Val: 2 (benzene); Test: 3 (pyridine)
    split_data = {
        "indices": {
            "train": [0, 1],
            "val": [2],
            "test": [3],
        },
        "row_identifiers": {
            "train_source_rows": [1, 2],
            "val_source_rows": [3],
            "test_source_rows": [4],
            "train_mol_ids": ["M1", "M2"],
            "val_mol_ids": ["M3"],
            "test_mol_ids": ["M4"],
        },
    }
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump(split_data, f)

    return clean_csv, split_json


def test_split_disjointness_and_coverage(mock_split_setup):
    _, split_json = mock_split_setup
    res = audit_split_disjointness_and_coverage(split_json, expected_total=4)
    assert res["status"] == "passed"
    assert res["disjoint"] is True
    assert res["complete_coverage"] is True
    assert res["train_count"] == 2
    assert res["val_count"] == 1
    assert res["test_count"] == 1


def test_split_disjointness_failure(tmp_path: Path):
    split_json = tmp_path / "overlapping_split.json"
    split_data = {
        "indices": {
            "train": [0, 1, 2],
            "val": [2, 3],  # 2 is in both train and val!
            "test": [4],
        }
    }
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump(split_data, f)

    with pytest.raises(ValueError, match="Index overlap between train and val"):
        audit_split_disjointness_and_coverage(split_json, expected_total=5)


def test_scaffold_isolation(mock_split_setup):
    clean_csv, split_json = mock_split_setup
    res = audit_scaffold_isolation(clean_csv, split_json)
    assert res["status"] == "passed"
    assert res["scaffold_leakage_free"] is True
    assert res["train_val_overlap"] == 0
    assert res["train_test_overlap"] == 0
    assert res["val_test_overlap"] == 0


def test_scaffold_leakage_detected(tmp_path: Path):
    clean_csv = tmp_path / "leaky_clean.csv"
    split_json = tmp_path / "leaky_split.json"

    # Both molecules have benzene scaffold!
    smiles = ["c1ccccc1", "Cc1ccccc1"]
    df = pd.DataFrame({
        "source_row": [1, 2],
        "mol_id": ["M1", "M2"],
        "canonical_smiles": smiles,
    })
    df.to_csv(clean_csv, index=False)

    # But split puts 0 in train and 1 in val!
    split_data = {
        "indices": {
            "train": [0],
            "val": [1],
            "test": [],
        }
    }
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump(split_data, f)

    with pytest.raises(ValueError, match="Bemis-Murcko scaffold leakage detected"):
        audit_scaffold_isolation(clean_csv, split_json)


def test_acyclic_policy(mock_split_setup):
    clean_csv, split_json = mock_split_setup
    res = audit_acyclic_molecule_handling(clean_csv, split_json)
    assert res["status"] == "passed"
    assert res["total_acyclic_molecules"] == 2
    assert res["acyclic_in_train"] == 2
    assert res["acyclic_in_val"] == 0
    assert res["acyclic_in_test"] == 0


def test_row_identifier_alignment(mock_split_setup):
    clean_csv, split_json = mock_split_setup
    res = audit_row_identifier_alignment(clean_csv, split_json)
    assert res["status"] == "passed"
    assert res["alignment_verified"] is True


@pytest.mark.skipif(
    not Path("data/processed/tox21_cleaned.csv").exists() or not Path("data/processed/scaffold_split_indices.json").exists(),
    reason="Real Tox21 cleaned or split artifacts not present",
)
def test_real_dataset_split_audit():
    res = run_split_audit()
    assert res["status"] == "passed"
    cov = res["coverage_and_disjointness"]
    assert cov["train_count"] == EXPECTED_TRAIN_COUNT
    assert cov["val_count"] == EXPECTED_VAL_COUNT
    assert cov["test_count"] == EXPECTED_TEST_COUNT
    assert cov["total_records"] == EXPECTED_TOTAL_RECORDS
    assert cov["disjoint"] is True
    assert cov["complete_coverage"] is True

    scaf = res["scaffold_isolation"]
    assert scaf["scaffold_leakage_free"] is True
    assert scaf["train_val_overlap"] == 0
    assert scaf["train_test_overlap"] == 0
    assert scaf["val_test_overlap"] == 0

    acyc = res["acyclic_policy"]
    assert acyc["total_acyclic_molecules"] == EXPECTED_ACYCLIC_COUNT
    assert acyc["acyclic_in_train"] == EXPECTED_ACYCLIC_COUNT
    assert acyc["acyclic_in_val"] == 0
    assert acyc["acyclic_in_test"] == 0
