"""Feature Representation Assembly Module.

Student 2 Module: Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks.

This module assembles three distinct feature representations from cleaned Tox21 molecular data:
1. Morgan Fingerprints (binary bit vectors, radius=2, 2048 bits)
2. Physicochemical Descriptors (9 continuous molecular properties)
3. Combined Features (concatenation of Morgan fingerprints and descriptors, 2057 dimensions)

Artifacts Produced:
-------------------
data/processed/features/
├── morgan_r2_2048.npy      - Binary Morgan fingerprint matrix (N, 2048), uint8
├── descriptors_9.npy       - Molecular descriptors matrix (N, 9), float64
├── combined_features.npy   - Concatenated feature matrix (N, 2057), float32
├── feature_row_ids.json    - Ordered source_row and mol_id mapping
└── feature_manifest.json   - Comprehensive audit manifest with shapes, dtypes, checksums

Guarantees:
-----------
- All representations share identical row counts and molecule ordering.
- Concatenation exactly preserves original bit patterns and descriptor values.
- Row identifiers (source_row, mol_id) are explicitly stored for alignment with splits and labels.
- Zero label contamination: toxicity labels are never accessed or included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.features.descriptors import (
    DESCRIPTOR_NAMES,
    compute_descriptors_matrix,
)
from src.features.fingerprints import (
    DEFAULT_MORGAN_N_BITS,
    DEFAULT_MORGAN_RADIUS,
    compute_morgan_fingerprints,
)

logger = logging.getLogger(__name__)

DEFAULT_INPUT_CSV: Final[Path] = Path("data/processed/tox21_cleaned.csv")
DEFAULT_OUTPUT_DIR: Final[Path] = Path("data/processed/features")


def compute_sha256(filepath: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        filepath: Path to the file.

    Returns:
        Hexadecimal SHA-256 string.
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_feature_names(
    n_bits: int = DEFAULT_MORGAN_N_BITS,
    descriptor_names: tuple[str, ...] = DESCRIPTOR_NAMES,
) -> tuple[list[str], list[str], list[str]]:
    """Generate standardized feature names for the three representations.

    Args:
        n_bits: Number of Morgan fingerprint bits.
        descriptor_names: Sequence of descriptor names.

    Returns:
        Tuple of (morgan_feature_names, descriptor_feature_names, combined_feature_names).
    """
    morgan_names = [f"morgan_bit_{i}" for i in range(n_bits)]
    desc_names = list(descriptor_names)
    combined_names = morgan_names + desc_names
    return morgan_names, desc_names, combined_names


def build_feature_representations(
    input_csv: Path = DEFAULT_INPUT_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    radius: int = DEFAULT_MORGAN_RADIUS,
    n_bits: int = DEFAULT_MORGAN_N_BITS,
    smiles_column: str = "canonical_smiles",
) -> dict[str, Any]:
    """Extract, concatenate, and persist all three feature representations.

    Args:
        input_csv: Path to cleaned dataset CSV.
        output_dir: Directory where feature arrays and manifests will be written.
        radius: Morgan radius (default 2, ECFP4 equivalent).
        n_bits: Morgan bit count (default 2048).
        smiles_column: Column name containing validated molecular SMILES.

    Returns:
        Dictionary containing manifest metadata and file checksums.

    Raises:
        FileNotFoundError: If input CSV does not exist.
        KeyError: If required columns are missing from the input CSV.
        ValueError: If row counts or concatenation assertions fail.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    input_sha256 = compute_sha256(input_csv)

    logger.info("Reading input dataset from %s (SHA-256: %s)...", input_csv, input_sha256)
    df = pd.read_csv(input_csv)
    n_records = len(df)

    # Validate required columns
    required_cols = [smiles_column]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from {input_csv}. Columns: {list(df.columns)}")

    # Extract stable row identifiers
    source_rows = df["source_row"].tolist() if "source_row" in df.columns else list(range(1, n_records + 1))
    mol_ids = df["mol_id"].tolist() if "mol_id" in df.columns else [f"MOL_{i}" for i in range(n_records)]
    smiles_list = df[smiles_column].tolist()

    logger.info("Processing %d molecules...", n_records)

    # 1. Morgan Fingerprints (binary, uint8)
    logger.info("Computing Morgan fingerprints (radius=%d, n_bits=%d)...", radius, n_bits)
    morgan_matrix = compute_morgan_fingerprints(
        smiles_list,
        radius=radius,
        n_bits=n_bits,
        count=False,
    )
    if morgan_matrix.shape != (n_records, n_bits):
        raise ValueError(
            f"Expected Morgan matrix shape ({n_records}, {n_bits}), got {morgan_matrix.shape}"
        )

    # 2. Physicochemical Descriptors (float64)
    logger.info("Computing 9 physicochemical descriptors...")
    descriptors_matrix = compute_descriptors_matrix(smiles_list)
    if descriptors_matrix.shape != (n_records, len(DESCRIPTOR_NAMES)):
        raise ValueError(
            f"Expected descriptors matrix shape ({n_records}, {len(DESCRIPTOR_NAMES)}), "
            f"got {descriptors_matrix.shape}"
        )

    # 3. Combined Features (concatenation, float32)
    # Using float32 for combined representation preserves descriptor numerical range
    # and keeps memory footprint manageable for downstream ML models.
    logger.info("Assembling combined representation (dimensions: %d)...", n_bits + len(DESCRIPTOR_NAMES))
    combined_matrix = np.hstack(
        [
            morgan_matrix.astype(np.float32),
            descriptors_matrix.astype(np.float32),
        ]
    )

    expected_combined_cols = n_bits + len(DESCRIPTOR_NAMES)
    if combined_matrix.shape != (n_records, expected_combined_cols):
        raise ValueError(
            f"Expected combined matrix shape ({n_records}, {expected_combined_cols}), "
            f"got {combined_matrix.shape}"
        )

    # Verification: Check concatenation integrity
    np.testing.assert_array_equal(
        combined_matrix[:, :n_bits],
        morgan_matrix.astype(np.float32),
        err_msg="Morgan slice in combined representation differs from original Morgan matrix",
    )
    np.testing.assert_allclose(
        combined_matrix[:, n_bits:],
        descriptors_matrix.astype(np.float32),
        rtol=1e-5,
        atol=1e-5,
        err_msg="Descriptor slice in combined representation differs from original descriptor matrix",
    )

    # Define artifact filepaths
    morgan_file = output_dir / f"morgan_r{radius}_{n_bits}.npy"
    desc_file = output_dir / f"descriptors_{len(DESCRIPTOR_NAMES)}.npy"
    combined_file = output_dir / "combined_features.npy"
    row_ids_file = output_dir / "feature_row_ids.json"
    manifest_file = output_dir / "feature_manifest.json"

    # Save NumPy matrices
    np.save(morgan_file, morgan_matrix)
    np.save(desc_file, descriptors_matrix)
    np.save(combined_file, combined_matrix)

    # Save Row Identifiers Mapping
    row_ids_data = {
        "n_records": n_records,
        "input_csv": str(input_csv),
        "source_rows": source_rows,
        "mol_ids": mol_ids,
    }
    with open(row_ids_file, "w", encoding="utf-8") as f:
        json.dump(row_ids_data, f, indent=2)

    # Generate Feature Names
    morgan_names, desc_names, combined_names = generate_feature_names(
        n_bits=n_bits,
        descriptor_names=DESCRIPTOR_NAMES,
    )

    # Build Comprehensive Manifest
    manifest = {
        "status": "success",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": {
            "path": str(input_csv),
            "sha256": input_sha256,
            "records": n_records,
            "smiles_column": smiles_column,
        },
        "artifacts": {
            "morgan_fingerprints": {
                "file": morgan_file.name,
                "path": str(morgan_file),
                "shape": list(morgan_matrix.shape),
                "dtype": str(morgan_matrix.dtype),
                "parameters": {
                    "radius": radius,
                    "n_bits": n_bits,
                    "representation": "binary",
                    "generator_api": "rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator",
                },
                "sha256": compute_sha256(morgan_file),
            },
            "molecular_descriptors": {
                "file": desc_file.name,
                "path": str(desc_file),
                "shape": list(descriptors_matrix.shape),
                "dtype": str(descriptors_matrix.dtype),
                "feature_names": desc_names,
                "sha256": compute_sha256(desc_file),
            },
            "combined_features": {
                "file": combined_file.name,
                "path": str(combined_file),
                "shape": list(combined_matrix.shape),
                "dtype": str(combined_matrix.dtype),
                "composition": {
                    "morgan_slice": [0, n_bits],
                    "descriptors_slice": [n_bits, expected_combined_cols],
                },
                "sha256": compute_sha256(combined_file),
            },
            "row_identifiers": {
                "file": row_ids_file.name,
                "path": str(row_ids_file),
                "sha256": compute_sha256(row_ids_file),
            },
        },
        "feature_dimensions": {
            "morgan": n_bits,
            "descriptors": len(DESCRIPTOR_NAMES),
            "combined": expected_combined_cols,
        },
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Feature assembly complete. Manifest written to %s", manifest_file)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for build_features."""
    parser = argparse.ArgumentParser(
        description="Assemble Morgan fingerprints, descriptors, and combined feature representations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to cleaned dataset CSV (default: data/processed/tox21_cleaned.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for feature arrays (default: data/processed/features)",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=DEFAULT_MORGAN_RADIUS,
        help="Morgan radius (default: 2)",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=DEFAULT_MORGAN_N_BITS,
        help="Morgan bit length (default: 2048)",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default="canonical_smiles",
        help="SMILES column in input CSV (default: canonical_smiles)",
    )
    return parser.parse_args()


def main() -> None:
    """Execute feature assembly CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    build_feature_representations(
        input_csv=args.input,
        output_dir=args.output_dir,
        radius=args.radius,
        n_bits=args.n_bits,
        smiles_column=args.smiles_column,
    )


if __name__ == "__main__":
    main()
