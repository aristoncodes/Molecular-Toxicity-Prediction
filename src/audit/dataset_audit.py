"""Independent Dataset Integrity and Provenance Auditor.

Student 3 Module: Independent audit of raw Tox21 data, exclusions, canonical structures,
and endpoint label distributions.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)

DEFAULT_RAW_PATH = Path("data/raw/tox21.csv.gz")
DEFAULT_PROCESSED_PATH = Path("data/processed/tox21_cleaned.csv")
DEFAULT_EXCLUSIONS_PATH = Path("results/experiment_logs/exclusions.csv")
DEFAULT_CONFIG_PATH = Path("configs/data_config.yaml")

EXPECTED_REFERENCE_SHA256 = "45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360"
EXPECTED_RAW_ROWS = 7831
EXPECTED_VALID_RECORDS = 7823
EXPECTED_EXCLUSIONS = 8

TOX21_ENDPOINTS: tuple[str, ...] = (
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_dataset_provenance_and_checksum(
    raw_path: Path = DEFAULT_RAW_PATH,
    expected_sha256: str = EXPECTED_REFERENCE_SHA256,
) -> dict[str, Any]:
    """Verify raw gzip archive integrity and SHA-256 checksum."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found: {raw_path}")

    # 1. Verify Gzip Magic Number
    with open(raw_path, "rb") as f:
        magic = f.read(2)
        if magic != b"\x1f\x8b":
            raise ValueError(f"Invalid gzip magic number for {raw_path}: {magic!r}")

    # 2. Decompress full archive to test CRC32 & trailer
    decompressed_bytes = 0
    raw_line_count = 0
    with gzip.open(raw_path, "rt", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        for _ in reader:
            raw_line_count += 1

    file_size = raw_path.stat().st_size
    measured_sha256 = compute_file_sha256(raw_path)

    if measured_sha256 != expected_sha256:
        raise ValueError(
            f"Raw SHA-256 mismatch: measured {measured_sha256}, expected {expected_sha256}"
        )

    if raw_line_count != EXPECTED_RAW_ROWS:
        raise ValueError(
            f"Raw data rows mismatch: counted {raw_line_count}, expected {EXPECTED_RAW_ROWS}"
        )

    return {
        "status": "passed",
        "raw_path": str(raw_path),
        "file_size_bytes": file_size,
        "measured_sha256": measured_sha256,
        "header_columns": header,
        "raw_record_count": raw_line_count,
    }


def audit_raw_vs_cleaned_accounting(
    raw_path: Path = DEFAULT_RAW_PATH,
    cleaned_path: Path = DEFAULT_PROCESSED_PATH,
    exclusions_path: Path = DEFAULT_EXCLUSIONS_PATH,
) -> dict[str, Any]:
    """Verify mathematical accounting: raw_records == cleaned_records + exclusions."""
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {cleaned_path}")
    if not exclusions_path.exists():
        raise FileNotFoundError(f"Exclusions CSV not found: {exclusions_path}")

    df_cleaned = pd.read_csv(cleaned_path)
    df_exclusions = pd.read_csv(exclusions_path)

    n_cleaned = len(df_cleaned)
    n_exclusions = len(df_exclusions)
    total_accounted = n_cleaned + n_exclusions

    if n_cleaned != EXPECTED_VALID_RECORDS:
        raise ValueError(f"Cleaned rows ({n_cleaned}) != expected ({EXPECTED_VALID_RECORDS})")
    if n_exclusions != EXPECTED_EXCLUSIONS:
        raise ValueError(f"Exclusions ({n_exclusions}) != expected ({EXPECTED_EXCLUSIONS})")
    if total_accounted != EXPECTED_RAW_ROWS:
        raise ValueError(
            f"Record accounting failed: cleaned ({n_cleaned}) + exclusions ({n_exclusions}) "
            f"= {total_accounted} != raw ({EXPECTED_RAW_ROWS})"
        )

    # Verify that every excluded structure actually fails RDKit MolFromSmiles
    unparsable_verified = 0
    for _, row in df_exclusions.iterrows():
        smiles = row["original_smiles"]
        mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
        if mol is None:
            unparsable_verified += 1
        else:
            raise ValueError(f"Excluded structure '{smiles}' surprisingly parsed in RDKit!")

    assert unparsable_verified == n_exclusions

    return {
        "status": "passed",
        "cleaned_records": n_cleaned,
        "exclusions_count": n_exclusions,
        "total_accounted": total_accounted,
        "raw_expected": EXPECTED_RAW_ROWS,
        "all_exclusions_reproduced": True,
    }


def audit_canonical_smiles_integrity(
    cleaned_path: Path = DEFAULT_PROCESSED_PATH,
    smiles_col: str = "canonical_smiles",
) -> dict[str, Any]:
    """Verify that 100% of canonical SMILES parse in RDKit, are invariant, and unique."""
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {cleaned_path}")

    df = pd.read_csv(cleaned_path)
    if smiles_col not in df.columns:
        raise KeyError(f"Column '{smiles_col}' not found in {cleaned_path}")

    smiles_list = df[smiles_col].tolist()
    n_total = len(smiles_list)

    re_canonicalized_matches = 0
    invalid_mols = []

    for idx, smi in enumerate(smiles_list):
        if not isinstance(smi, str) or not smi.strip():
            invalid_mols.append((idx, smi, "empty_or_non_string"))
            continue

        mol = Chem.MolFromSmiles(smi, sanitize=True)
        if mol is None:
            invalid_mols.append((idx, smi, "rdkit_parse_failed"))
            continue

        re_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if re_smi == smi:
            re_canonicalized_matches += 1
        else:
            invalid_mols.append((idx, smi, f"canonical_variance_mismatch: {re_smi}"))

    if invalid_mols:
        raise ValueError(f"Found {len(invalid_mols)} invalid SMILES: {invalid_mols[:5]}")

    unique_smiles_count = len(set(smiles_list))

    return {
        "status": "passed",
        "total_molecules_checked": n_total,
        "valid_rdkit_molecules": re_canonicalized_matches,
        "canonical_invariance_rate": re_canonicalized_matches / n_total,
        "unique_structures": unique_smiles_count,
        "has_duplicates": unique_smiles_count < n_total,
    }


def audit_label_encodings_and_missingness(
    cleaned_path: Path = DEFAULT_PROCESSED_PATH,
    endpoints: Sequence[str] = TOX21_ENDPOINTS,
) -> dict[str, Any]:
    """Verify strictly binary 0/1/NaN label encodings and compile per-endpoint statistics."""
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {cleaned_path}")

    df = pd.read_csv(cleaned_path)
    endpoint_stats: dict[str, dict[str, Any]] = {}

    for ep in endpoints:
        if ep not in df.columns:
            raise KeyError(f"Endpoint column '{ep}' not found in cleaned dataset")

        col = df[ep]
        observed = col.dropna()

        # Verify all non-null values are exactly 0.0 or 1.0
        unique_vals = set(observed.unique())
        if not unique_vals.issubset({0.0, 1.0, 0, 1}):
            raise ValueError(f"Endpoint '{ep}' contains unexpected values: {unique_vals}")

        n_pos = int((observed == 1.0).sum())
        n_neg = int((observed == 0.0).sum())
        n_missing = int(col.isna().sum())
        n_obs = n_pos + n_neg
        prevalence = n_pos / n_obs if n_obs > 0 else 0.0

        if n_pos == 0 or n_neg == 0:
            raise ValueError(f"Endpoint '{ep}' has single class! pos={n_pos}, neg={n_neg}")

        endpoint_stats[ep] = {
            "total_records": len(col),
            "observed": n_obs,
            "positive": n_pos,
            "negative": n_neg,
            "missing": n_missing,
            "prevalence": prevalence,
        }

    return {
        "status": "passed",
        "endpoints_evaluated": len(endpoints),
        "endpoint_statistics": endpoint_stats,
    }


def run_dataset_audit(
    raw_path: Path = DEFAULT_RAW_PATH,
    cleaned_path: Path = DEFAULT_PROCESSED_PATH,
    exclusions_path: Path = DEFAULT_EXCLUSIONS_PATH,
) -> dict[str, Any]:
    """Run all dataset integrity audits and return consolidated report."""
    logger.info("Running dataset provenance and checksum audit...")
    prov_res = audit_dataset_provenance_and_checksum(raw_path)

    logger.info("Running raw vs cleaned accounting audit...")
    acc_res = audit_raw_vs_cleaned_accounting(raw_path, cleaned_path, exclusions_path)

    logger.info("Running canonical SMILES integrity audit...")
    smi_res = audit_canonical_smiles_integrity(cleaned_path)

    logger.info("Running label encodings and missingness audit...")
    lbl_res = audit_label_encodings_and_missingness(cleaned_path)

    return {
        "status": "passed",
        "provenance_and_checksum": prov_res,
        "accounting": acc_res,
        "canonical_smiles": smi_res,
        "labels": lbl_res,
    }
