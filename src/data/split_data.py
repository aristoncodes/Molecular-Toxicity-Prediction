"""Scaffold-Based Train/Validation/Test Splitting Module.

Student 2 Module: Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks.

This module implements deterministic, leakage-free molecular dataset splitting based on
Bemis–Murcko molecular scaffolds (RDKit `MurckoScaffold.MurckoScaffoldSmiles`).

Scientific Motivation:
----------------------
Random splitting in molecular property prediction leads to overoptimistic performance estimates
due to "scaffold memorization" (analogous molecules with identical core scaffolds appearing
in both train and test sets). Bemis–Murcko scaffold splitting clusters molecules by their core
ring and linker architecture, assigning entire scaffold clusters exclusively to train, val, or test.
This evaluates models on true out-of-distribution generalization to structurally novel chemical series.

Empty Scaffold Policy (Acyclic Molecules):
------------------------------------------
Acyclic molecules (e.g., aliphatic chains like ethanol, hexane) possess no rings, yielding an empty
Murcko scaffold string `""`. In Tox21, 1,775 molecules (22.69%) are acyclic.
- Default policy: `single_group`. All acyclic molecules form a single scaffold group `""`.
  Sorting scaffold clusters by size descending assigns this group, along with the benzene scaffold
  (1,474 molecules), to the training partition (capacity ~80% = 6,258).
- The validation (~10%) and test (~10%) partitions receive distinct, non-overlapping cyclic scaffold
  clusters.
- No nonempty scaffold group is ever split across partitions.

Target Proportions:
-------------------
- Training: ~80% (0.80)
- Validation: ~10% (0.10)
- Test: ~10% (0.10)

Output Artifact:
----------------
data/processed/scaffold_split_indices.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Sequence

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)

DEFAULT_INPUT_CSV: Final[Path] = Path("data/processed/tox21_cleaned.csv")
DEFAULT_OUTPUT_JSON: Final[Path] = Path("data/processed/scaffold_split_indices.json")
DEFAULT_SPLIT_SEED: Final[int] = 42
DEFAULT_TRAIN_RATIO: Final[float] = 0.8
DEFAULT_VAL_RATIO: Final[float] = 0.1
DEFAULT_TEST_RATIO: Final[float] = 0.1


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Compute Bemis-Murcko scaffold SMILES for a given molecular SMILES.

    Args:
        smiles: Valid SMILES string.
        include_chirality: Whether to include stereochemistry in the scaffold.

    Returns:
        Canonical SMILES of the Bemis-Murcko scaffold, or empty string for acyclic molecules.

    Raises:
        ValueError: If RDKit fails to parse the SMILES string.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"Invalid SMILES string: '{smiles}'")

    mol = Chem.MolFromSmiles(smiles.strip(), sanitize=True)
    if mol is None:
        raise ValueError(f"RDKit Chem.MolFromSmiles failed for SMILES: '{smiles}'")

    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def scaffold_split(
    smiles_list: Sequence[str],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
    empty_scaffold_policy: str = "single_group",
    include_chirality: bool = False,
) -> dict[str, Any]:
    """Perform deterministic scaffold-based train/val/test splitting.

    Groups molecules by their Bemis-Murcko scaffold. Sorts groups by size descending,
    shuffling same-size groups with the specified random seed. Greedily fills
    the training partition up to approximately train_ratio, then validation up to
    val_ratio, placing the remainder into test.

    Args:
        smiles_list: Sequence of validated molecular SMILES strings.
        train_ratio: Target proportion for training set (~0.8).
        val_ratio: Target proportion for validation set (~0.1).
        test_ratio: Target proportion for test set (~0.1).
        seed: Random seed for deterministic group ordering.
        empty_scaffold_policy: How acyclic molecules are handled ('single_group').
        include_chirality: Whether scaffolds retain chiral specifications.

    Returns:
        Dict containing partition indices, counts, proportions, and scaffold accounting.

    Raises:
        ValueError: If ratios do not sum to ~1.0, list is empty, or policy is invalid.
    """
    n_total = len(smiles_list)
    if n_total == 0:
        raise ValueError("Cannot perform scaffold split on an empty list of SMILES")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if not (0.999 <= ratio_sum <= 1.001):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.4f}")

    if empty_scaffold_policy not in ("single_group",):
        raise ValueError(f"Unsupported empty_scaffold_policy: '{empty_scaffold_policy}'")

    # Group 0-based indices by Bemis-Murcko scaffold
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, s in enumerate(smiles_list):
        scaf = generate_scaffold(s, include_chirality=include_chirality)
        scaffold_to_indices[scaf].append(idx)

    total_scaffolds = len(scaffold_to_indices)
    acyclic_count = len(scaffold_to_indices.get("", []))

    # Deterministic grouping: sort by group size descending, tie-break by shuffled order
    rng = random.Random(seed)
    by_size: dict[int, list[tuple[str, list[int]]]] = defaultdict(list)
    for scaf, indices in scaffold_to_indices.items():
        by_size[len(indices)].append((scaf, indices))

    ordered_groups: list[tuple[str, list[int]]] = []
    for size in sorted(by_size.keys(), reverse=True):
        tier = by_size[size][:]
        # Sort stably by scaffold string first, then shuffle with seed for cross-platform determinism
        tier.sort(key=lambda item: item[0])
        rng.shuffle(tier)
        ordered_groups.extend(tier)

    # Target partition counts
    train_target = int(train_ratio * n_total)
    val_target = int(val_ratio * n_total)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    train_scaffolds: list[str] = []
    val_scaffolds: list[str] = []
    test_scaffolds: list[str] = []

    for scaf, indices in ordered_groups:
        if len(train_indices) + len(indices) <= train_target:
            train_indices.extend(indices)
            train_scaffolds.append(scaf)
        elif len(val_indices) + len(indices) <= val_target:
            val_indices.extend(indices)
            val_scaffolds.append(scaf)
        else:
            test_indices.extend(indices)
            test_scaffolds.append(scaf)

    # Sort indices ascending for clean, deterministic storage
    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    # --- Strict Integrity Assertions ---
    # 1. Bounds check
    all_indices = train_indices + val_indices + test_indices
    assert all(0 <= idx < n_total for idx in all_indices), "Index out of bounds"

    # 2. Uniqueness
    assert len(all_indices) == len(set(all_indices)), "Duplicate indices detected across partitions"

    # 3. Disjointness
    s_train, s_val, s_test = set(train_indices), set(val_indices), set(test_indices)
    assert s_train.isdisjoint(s_val), "Train and validation partitions overlap"
    assert s_train.isdisjoint(s_test), "Train and test partitions overlap"
    assert s_val.isdisjoint(s_test), "Validation and test partitions overlap"

    # 4. Full coverage
    assert len(s_train | s_val | s_test) == n_total, "Partitions do not cover all input records"

    # 5. Scaffold group isolation (no nonempty scaffold crosses partitions)
    scaf_train_set = {s for s in train_scaffolds if s}
    scaf_val_set = {s for s in val_scaffolds if s}
    scaf_test_set = {s for s in test_scaffolds if s}

    assert scaf_train_set.isdisjoint(scaf_val_set), "Nonempty scaffold leaked between train and val"
    assert scaf_train_set.isdisjoint(scaf_test_set), "Nonempty scaffold leaked between train and test"
    assert scaf_val_set.isdisjoint(scaf_test_set), "Nonempty scaffold leaked between val and test"

    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "partition_counts": {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices),
        },
        "partition_proportions": {
            "train": len(train_indices) / n_total,
            "val": len(val_indices) / n_total,
            "test": len(test_indices) / n_total,
        },
        "unique_scaffold_counts": {
            "train": len(train_scaffolds),
            "val": len(val_scaffolds),
            "test": len(test_scaffolds),
            "total": total_scaffolds,
            "acyclic_molecules": acyclic_count,
        },
    }


def execute_scaffold_splitting(
    input_csv: Path = DEFAULT_INPUT_CSV,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
    empty_scaffold_policy: str = "single_group",
    smiles_column: str = "canonical_smiles",
) -> dict[str, Any]:
    """Run scaffold splitting pipeline on cleaned dataset and persist output JSON.

    Args:
        input_csv: Path to cleaned dataset CSV.
        output_json: Destination for scaffold split JSON artifact.
        train_ratio: Target train fraction (default 0.8).
        val_ratio: Target val fraction (default 0.1).
        test_ratio: Target test fraction (default 0.1).
        seed: Random seed (default 42).
        empty_scaffold_policy: Policy for acyclic molecules (default 'single_group').
        smiles_column: Column name containing validated SMILES.

    Returns:
        Complete split metadata and audit dictionary.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Input dataset not found at {input_csv}")

    input_sha256 = compute_sha256(input_csv)
    logger.info("Reading dataset from %s (SHA-256: %s)...", input_csv, input_sha256)
    df = pd.read_csv(input_csv)
    n_total = len(df)

    if smiles_column not in df.columns:
        raise KeyError(f"Column '{smiles_column}' not found in {input_csv}")

    # Extract stable row identifiers
    source_rows = df["source_row"].tolist() if "source_row" in df.columns else list(range(1, n_total + 1))
    mol_ids = df["mol_id"].tolist() if "mol_id" in df.columns else [f"MOL_{i}" for i in range(n_total)]
    smiles_list = df[smiles_column].tolist()

    logger.info("Performing scaffold split on %d molecules (seed=%d)...", n_total, seed)
    split_res = scaffold_split(
        smiles_list,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        empty_scaffold_policy=empty_scaffold_policy,
    )

    train_idx = split_res["train_indices"]
    val_idx = split_res["val_indices"]
    test_idx = split_res["test_indices"]

    train_row_ids = [source_rows[i] for i in train_idx]
    val_row_ids = [source_rows[i] for i in val_idx]
    test_row_ids = [source_rows[i] for i in test_idx]

    train_mol_ids = [mol_ids[i] for i in train_idx]
    val_mol_ids = [mol_ids[i] for i in val_idx]
    test_mol_ids = [mol_ids[i] for i in test_idx]

    artifact: dict[str, Any] = {
        "status": "success",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_metadata": {
            "input_csv": str(input_csv),
            "sha256": input_sha256,
            "total_records": n_total,
            "smiles_column": smiles_column,
        },
        "split_configuration": {
            "method": "bemis_murcko_scaffold",
            "scaffold_algorithm": "rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(mol, includeChirality=False)",
            "empty_scaffold_policy": empty_scaffold_policy,
            "seed": seed,
            "target_proportions": {
                "train": train_ratio,
                "val": val_ratio,
                "test": test_ratio,
            },
        },
        "partition_counts": split_res["partition_counts"],
        "partition_proportions": split_res["partition_proportions"],
        "unique_scaffold_counts": split_res["unique_scaffold_counts"],
        "indices": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        },
        "row_identifiers": {
            "train_source_rows": train_row_ids,
            "val_source_rows": val_row_ids,
            "test_source_rows": test_row_ids,
            "train_mol_ids": train_mol_ids,
            "val_mol_ids": val_mol_ids,
            "test_mol_ids": test_mol_ids,
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    logger.info(
        "Scaffold split complete:\n"
        "  Train: %d (%.2f%%) - %d scaffolds\n"
        "  Val:   %d (%.2f%%) - %d scaffolds\n"
        "  Test:  %d (%.2f%%) - %d scaffolds\n"
        "Artifact saved to %s",
        split_res["partition_counts"]["train"],
        split_res["partition_proportions"]["train"] * 100,
        split_res["unique_scaffold_counts"]["train"],
        split_res["partition_counts"]["val"],
        split_res["partition_proportions"]["val"] * 100,
        split_res["unique_scaffold_counts"]["val"],
        split_res["partition_counts"]["test"],
        split_res["partition_proportions"]["test"] * 100,
        split_res["unique_scaffold_counts"]["test"],
        output_json,
    )
    return artifact


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for scaffold splitting."""
    parser = argparse.ArgumentParser(
        description="Perform Bemis-Murcko scaffold splitting on cleaned Tox21 dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to cleaned dataset CSV (default: data/processed/tox21_cleaned.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Destination for split JSON (default: data/processed/scaffold_split_indices.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Random seed for group tie-breaking (default: 42)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Target training fraction (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Target validation fraction (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Target test fraction (default: 0.1)",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default="canonical_smiles",
        help="Name of SMILES column (default: canonical_smiles)",
    )
    parser.add_argument(
        "--empty-scaffold-policy",
        type=str,
        default="single_group",
        choices=["single_group"],
        help="Policy for grouping acyclic molecules (default: single_group)",
    )
    return parser.parse_args()


def main() -> None:
    """Execute scaffold splitting CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    execute_scaffold_splitting(
        input_csv=args.input,
        output_json=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        empty_scaffold_policy=args.empty_scaffold_policy,
        smiles_column=args.smiles_column,
    )


if __name__ == "__main__":
    main()
