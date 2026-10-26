"""Independent Scaffold Splitting and Partition Integrity Auditor.

Student 3 Module: Independent verification of Bemis–Murcko scaffold isolation,
split disjointness, coverage, and acyclic molecule policy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)

DEFAULT_CLEANED_PATH = Path("data/processed/tox21_cleaned.csv")
DEFAULT_SPLIT_PATH = Path("data/processed/scaffold_split_indices.json")
DEFAULT_ROW_IDS_PATH = Path("data/processed/features/feature_row_ids.json")

EXPECTED_TRAIN_COUNT = 6258
EXPECTED_VAL_COUNT = 782
EXPECTED_TEST_COUNT = 783
EXPECTED_TOTAL_RECORDS = 7823
EXPECTED_ACYCLIC_COUNT = 1775


def audit_split_disjointness_and_coverage(
    split_path: Path = DEFAULT_SPLIT_PATH,
    expected_total: int = EXPECTED_TOTAL_RECORDS,
) -> dict[str, Any]:
    """Verify index disjointness, boundedness, and complete coverage."""
    if not split_path.exists():
        raise FileNotFoundError(f"Split indices JSON not found at {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    indices = split_data.get("indices", {})
    train_idx = indices.get("train", [])
    val_idx = indices.get("val", [])
    test_idx = indices.get("test", [])

    n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)
    total_split = n_train + n_val + n_test

    # 1. Boundedness & Types
    for name, idx_list in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        for i in idx_list:
            if not isinstance(i, int) or i < 0 or i >= expected_total:
                raise ValueError(f"Index {i} in {name} is out of bounds [0, {expected_total - 1}]")

    # 2. Cross-partition pairwise disjointness
    s_tr, s_va, s_te = set(train_idx), set(val_idx), set(test_idx)
    overlap_tr_va = s_tr & s_va
    overlap_tr_te = s_tr & s_te
    overlap_va_te = s_va & s_te

    if overlap_tr_va:
        raise ValueError(f"Index overlap between train and val: {len(overlap_tr_va)} items")
    if overlap_tr_te:
        raise ValueError(f"Index overlap between train and test: {len(overlap_tr_te)} items")
    if overlap_va_te:
        raise ValueError(f"Index overlap between val and test: {len(overlap_va_te)} items")

    # 3. Within-partition uniqueness
    if len(set(train_idx)) != n_train:
        raise ValueError("Duplicate indices found within train partition")
    if len(set(val_idx)) != n_val:
        raise ValueError("Duplicate indices found within val partition")
    if len(set(test_idx)) != n_test:
        raise ValueError("Duplicate indices found within test partition")

    # 4. Total record count check
    if total_split != expected_total:
        raise ValueError(f"Sum of split indices ({total_split}) != expected ({expected_total})")

    # 4. Complete coverage
    union_all = s_tr | s_va | s_te
    expected_all = set(range(expected_total))
    if union_all != expected_all:
        missing = expected_all - union_all
        raise ValueError(f"Partitions do not cover all indices. Missing {len(missing)} indices.")

    return {
        "status": "passed",
        "train_count": n_train,
        "val_count": n_val,
        "test_count": n_test,
        "train_proportion": n_train / expected_total,
        "val_proportion": n_val / expected_total,
        "test_proportion": n_test / expected_total,
        "total_records": expected_total,
        "disjoint": True,
        "complete_coverage": True,
    }


def audit_scaffold_isolation(
    cleaned_path: Path = DEFAULT_CLEANED_PATH,
    split_path: Path = DEFAULT_SPLIT_PATH,
    smiles_col: str = "canonical_smiles",
) -> dict[str, Any]:
    """Independently recompute Bemis-Murcko scaffolds and assert 0 scaffold leakage."""
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Split indices not found at {split_path}")

    df = pd.read_csv(cleaned_path)
    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    indices = split_data["indices"]
    train_idx = indices["train"]
    val_idx = indices["val"]
    test_idx = indices["test"]

    # Recompute Murcko scaffolds from scratch with RDKit
    scaffolds: list[str] = []
    for s in df[smiles_col]:
        mol = Chem.MolFromSmiles(s, sanitize=True)
        if mol is None:
            raise ValueError(f"SMILES failed parsing during scaffold recomputation: {s}")
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.append(scaf)

    # Scaffolds by partition (excluding empty scaffold for cyclic isolation check)
    tr_scafs = {scaffolds[i] for i in train_idx if scaffolds[i]}
    va_scafs = {scaffolds[i] for i in val_idx if scaffolds[i]}
    te_scafs = {scaffolds[i] for i in test_idx if scaffolds[i]}

    leak_tr_va = tr_scafs & va_scafs
    leak_tr_te = tr_scafs & te_scafs
    leak_va_te = va_scafs & te_scafs

    if leak_tr_va or leak_tr_te or leak_va_te:
        raise ValueError(
            f"Bemis-Murcko scaffold leakage detected! "
            f"Train-Val: {len(leak_tr_va)}, Train-Test: {len(leak_tr_te)}, Val-Test: {len(leak_va_te)}"
        )

    return {
        "status": "passed",
        "train_unique_scaffolds": len(tr_scafs),
        "val_unique_scaffolds": len(va_scafs),
        "test_unique_scaffolds": len(te_scafs),
        "train_val_overlap": 0,
        "train_test_overlap": 0,
        "val_test_overlap": 0,
        "scaffold_leakage_free": True,
    }


def audit_acyclic_molecule_handling(
    cleaned_path: Path = DEFAULT_CLEANED_PATH,
    split_path: Path = DEFAULT_SPLIT_PATH,
    smiles_col: str = "canonical_smiles",
) -> dict[str, Any]:
    """Verify that acyclic molecules (empty scaffold) are handled per single_group policy."""
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Split indices not found at {split_path}")

    df = pd.read_csv(cleaned_path)
    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    indices = split_data["indices"]
    train_set = set(indices["train"])
    val_set = set(indices["val"])
    test_set = set(indices["test"])

    acyclic_indices = []
    for idx, s in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(s, sanitize=True)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if scaf == "":
            acyclic_indices.append(idx)

    n_acyclic = len(acyclic_indices)
    acyclic_in_train = [i for i in acyclic_indices if i in train_set]
    acyclic_in_val = [i for i in acyclic_indices if i in val_set]
    acyclic_in_test = [i for i in acyclic_indices if i in test_set]

    # Under single_group policy, ALL acyclic molecules must reside in train
    if len(acyclic_in_val) > 0 or len(acyclic_in_test) > 0:
        raise ValueError(
            f"Acyclic policy violation: {len(acyclic_in_val)} acyclics in val, "
            f"{len(acyclic_in_test)} in test (single_group policy expects all in train)"
        )

    return {
        "status": "passed",
        "total_acyclic_molecules": n_acyclic,
        "acyclic_in_train": len(acyclic_in_train),
        "acyclic_in_val": len(acyclic_in_val),
        "acyclic_in_test": len(acyclic_in_test),
        "policy": "single_group (all acyclics assigned to train)",
    }


def audit_row_identifier_alignment(
    cleaned_path: Path = DEFAULT_CLEANED_PATH,
    split_path: Path = DEFAULT_SPLIT_PATH,
    row_ids_path: Path = DEFAULT_ROW_IDS_PATH,
) -> dict[str, Any]:
    """Verify alignment between split JSON row identifiers, cleaned CSV, and feature row IDs."""
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Split indices not found at {split_path}")

    df = pd.read_csv(cleaned_path)
    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    indices = split_data["indices"]
    row_ids = split_data.get("row_identifiers", {})

    for part in ["train", "val", "test"]:
        idx = indices[part]
        source_rows_key = f"{part}_source_rows"
        mol_ids_key = f"{part}_mol_ids"

        if source_rows_key in row_ids:
            expected_src = df.iloc[idx]["source_row"].tolist()
            stored_src = row_ids[source_rows_key]
            if stored_src != expected_src:
                raise ValueError(f"Mismatch in {source_rows_key}")

        if mol_ids_key in row_ids:
            expected_mol = df.iloc[idx]["mol_id"].tolist()
            stored_mol = row_ids[mol_ids_key]
            if stored_mol != expected_mol:
                raise ValueError(f"Mismatch in {mol_ids_key}")

    # If feature row IDs exist and correspond to this dataset, verify them as well
    if row_ids_path is not None and row_ids_path.exists():
        with open(row_ids_path, "r", encoding="utf-8") as f:
            feat_row_data = json.load(f)
        if len(feat_row_data.get("source_rows", [])) == len(df):
            if feat_row_data.get("source_rows") != df["source_row"].tolist():
                raise ValueError("Feature row IDs source_rows do not match cleaned CSV")
            if feat_row_data.get("mol_ids") != df["mol_id"].tolist():
                raise ValueError("Feature row IDs mol_ids do not match cleaned CSV")

    return {
        "status": "passed",
        "alignment_verified": True,
        "checked_partitions": ["train", "val", "test"],
    }


def run_split_audit(
    cleaned_path: Path = DEFAULT_CLEANED_PATH,
    split_path: Path = DEFAULT_SPLIT_PATH,
    row_ids_path: Path = DEFAULT_ROW_IDS_PATH,
) -> dict[str, Any]:
    """Run full split and scaffold isolation audit."""
    logger.info("Auditing split disjointness and coverage...")
    cov_res = audit_split_disjointness_and_coverage(split_path)

    logger.info("Auditing scaffold isolation and zero leakage...")
    scaf_res = audit_scaffold_isolation(cleaned_path, split_path)

    logger.info("Auditing acyclic molecule policy...")
    acyc_res = audit_acyclic_molecule_handling(cleaned_path, split_path)

    logger.info("Auditing row identifier alignment...")
    align_res = audit_row_identifier_alignment(cleaned_path, split_path, row_ids_path)

    return {
        "status": "passed",
        "coverage_and_disjointness": cov_res,
        "scaffold_isolation": scaf_res,
        "acyclic_policy": acyc_res,
        "alignment": align_res,
    }
