"""Unit tests for feature representation assembly and combined representations.

Verifies:
- Extraction of all three feature representations (Morgan, Descriptors, Combined)
- Dimensionality consistency: (N, 2048), (N, 9), (N, 2057)
- Exact preservation of values across concatenation slices
- Preservation of row sequence and row identifier alignment (source_row, mol_id)
- Integrity and completeness of feature_manifest.json and feature_row_ids.json
- Proper error handling when input CSV or required columns are missing
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    build_feature_representations,
    compute_sha256,
    generate_feature_names,
)
from src.features.descriptors import DESCRIPTOR_NAMES


@pytest.fixture
def synthetic_dataset_csv(tmp_path: Path) -> Path:
    """Create a temporary synthetic cleaned CSV for unit testing."""
    csv_file = tmp_path / "synthetic_cleaned.csv"
    data = {
        "source_row": [10, 20, 30, 40],
        "mol_id": ["SYNTH_1", "SYNTH_2", "SYNTH_3", "SYNTH_4"],
        "canonical_smiles": [
            "CCO",  # Ethanol
            "c1ccccc1",  # Benzene
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        ],
        "NR-AR": [0.0, 1.0, np.nan, 0.0],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return csv_file


def test_generate_feature_names():
    """Verify feature name generation counts and naming scheme."""
    morgan_names, desc_names, combined_names = generate_feature_names(
        n_bits=2048,
        descriptor_names=DESCRIPTOR_NAMES,
    )
    assert len(morgan_names) == 2048
    assert morgan_names[0] == "morgan_bit_0"
    assert morgan_names[-1] == "morgan_bit_2047"
    assert len(desc_names) == 9
    assert desc_names == list(DESCRIPTOR_NAMES)
    assert len(combined_names) == 2048 + 9
    assert combined_names[:2048] == morgan_names
    assert combined_names[2048:] == desc_names


def test_build_feature_representations(synthetic_dataset_csv: Path, tmp_path: Path):
    """Verify complete feature assembly pipeline on synthetic dataset."""
    out_dir = tmp_path / "features_out"
    manifest = build_feature_representations(
        input_csv=synthetic_dataset_csv,
        output_dir=out_dir,
        radius=2,
        n_bits=2048,
        smiles_column="canonical_smiles",
    )

    # 1. Check generated files exist
    morgan_file = out_dir / "morgan_r2_2048.npy"
    desc_file = out_dir / "descriptors_9.npy"
    combined_file = out_dir / "combined_features.npy"
    row_ids_file = out_dir / "feature_row_ids.json"
    manifest_file = out_dir / "feature_manifest.json"

    assert morgan_file.exists()
    assert desc_file.exists()
    assert combined_file.exists()
    assert row_ids_file.exists()
    assert manifest_file.exists()

    # 2. Check array shapes and dtypes
    morgan = np.load(morgan_file)
    desc = np.load(desc_file)
    combined = np.load(combined_file)

    assert morgan.shape == (4, 2048)
    assert morgan.dtype == np.uint8

    assert desc.shape == (4, 9)
    assert desc.dtype == np.float64

    assert combined.shape == (4, 2057)
    assert combined.dtype == np.float32

    # 3. Check concatenation preservation
    np.testing.assert_array_equal(combined[:, :2048], morgan.astype(np.float32))
    np.testing.assert_allclose(combined[:, 2048:], desc.astype(np.float32), rtol=1e-5)

    # 4. Check row identifiers alignment
    with open(row_ids_file, "r", encoding="utf-8") as f:
        row_ids_data = json.load(f)

    assert row_ids_data["n_records"] == 4
    assert row_ids_data["source_rows"] == [10, 20, 30, 40]
    assert row_ids_data["mol_ids"] == ["SYNTH_1", "SYNTH_2", "SYNTH_3", "SYNTH_4"]

    # 5. Check manifest contents and checksums
    assert manifest["status"] == "success"
    assert manifest["artifacts"]["morgan_fingerprints"]["sha256"] == compute_sha256(morgan_file)
    assert manifest["artifacts"]["molecular_descriptors"]["sha256"] == compute_sha256(desc_file)
    assert manifest["artifacts"]["combined_features"]["sha256"] == compute_sha256(combined_file)


def test_build_feature_missing_file(tmp_path: Path):
    """Verify FileNotFoundError when input CSV is missing."""
    with pytest.raises(FileNotFoundError):
        build_feature_representations(
            input_csv=tmp_path / "non_existent.csv",
            output_dir=tmp_path / "features",
        )


def test_build_feature_missing_column(tmp_path: Path):
    """Verify KeyError when specified SMILES column is missing."""
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"wrong_col": ["CCO"]}).to_csv(bad_csv, index=False)

    with pytest.raises(KeyError, match="Required column 'canonical_smiles' missing"):
        build_feature_representations(
            input_csv=bad_csv,
            output_dir=tmp_path / "features",
            smiles_column="canonical_smiles",
        )
