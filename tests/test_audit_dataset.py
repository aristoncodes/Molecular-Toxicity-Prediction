"""Unit tests for independent dataset integrity auditor."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.audit.dataset_audit import (
    EXPECTED_EXCLUSIONS,
    EXPECTED_RAW_ROWS,
    EXPECTED_REFERENCE_SHA256,
    EXPECTED_VALID_RECORDS,
    TOX21_ENDPOINTS,
    audit_canonical_smiles_integrity,
    audit_dataset_provenance_and_checksum,
    audit_label_encodings_and_missingness,
    audit_raw_vs_cleaned_accounting,
    compute_file_sha256,
    run_dataset_audit,
)


@pytest.fixture
def mock_clean_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "mock_clean.csv"
    data = {
        "source_row": [1, 2, 3],
        "mol_id": ["M1", "M2", "M3"],
        "canonical_smiles": ["CCO", "c1ccccc1", "CC(=O)O"],
    }
    for ep in TOX21_ENDPOINTS:
        data[ep] = [1.0, 0.0, np.nan]
    pd.DataFrame(data).to_csv(csv_file, index=False)
    return csv_file


def test_canonical_smiles_integrity(mock_clean_csv: Path):
    res = audit_canonical_smiles_integrity(mock_clean_csv)
    assert res["status"] == "passed"
    assert res["total_molecules_checked"] == 3
    assert res["canonical_invariance_rate"] == 1.0
    assert res["unique_structures"] == 3
    assert not res["has_duplicates"]


def test_canonical_smiles_with_invalid(tmp_path: Path):
    csv_file = tmp_path / "invalid.csv"
    pd.DataFrame({
        "canonical_smiles": ["CCO", "INVALID_SMILES_STRING", "c1ccccc1"]
    }).to_csv(csv_file, index=False)
    with pytest.raises(ValueError, match="Found 1 invalid SMILES"):
        audit_canonical_smiles_integrity(csv_file)


def test_label_encodings_and_missingness(mock_clean_csv: Path):
    res = audit_label_encodings_and_missingness(mock_clean_csv)
    assert res["status"] == "passed"
    assert res["endpoints_evaluated"] == 12
    for ep in TOX21_ENDPOINTS:
        stats = res["endpoint_statistics"][ep]
        assert stats["observed"] == 2
        assert stats["positive"] == 1
        assert stats["negative"] == 1
        assert stats["missing"] == 1
        assert stats["prevalence"] == 0.5


def test_label_encodings_unexpected_values(tmp_path: Path):
    csv_file = tmp_path / "bad_labels.csv"
    data = {"canonical_smiles": ["CCO", "c1ccccc1"]}
    for ep in TOX21_ENDPOINTS:
        data[ep] = [1.0, 2.0]  # Invalid value 2.0
    pd.DataFrame(data).to_csv(csv_file, index=False)
    with pytest.raises(ValueError, match="contains unexpected values"):
        audit_label_encodings_and_missingness(csv_file)


@pytest.mark.skipif(
    not Path("data/raw/tox21.csv.gz").exists() or not Path("data/processed/tox21_cleaned.csv").exists(),
    reason="Real Tox21 artifacts not present",
)
def test_real_dataset_full_audit():
    res = run_dataset_audit()
    assert res["status"] == "passed"
    assert res["provenance_and_checksum"]["measured_sha256"] == EXPECTED_REFERENCE_SHA256
    assert res["accounting"]["cleaned_records"] == EXPECTED_VALID_RECORDS
    assert res["accounting"]["exclusions_count"] == EXPECTED_EXCLUSIONS
    assert res["accounting"]["total_accounted"] == EXPECTED_RAW_ROWS
    assert res["canonical_smiles"]["total_molecules_checked"] == EXPECTED_VALID_RECORDS
    assert res["canonical_smiles"]["canonical_invariance_rate"] == 1.0
    assert res["labels"]["endpoints_evaluated"] == 12
