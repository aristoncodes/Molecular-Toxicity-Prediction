"""Morgan Chemical Fingerprint Generation Module.

Student 2 Module: Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks.

This module computes circular Morgan fingerprints (ECFP equivalent) using RDKit's modern
rdFingerprintGenerator API (`rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator`).

Theoretical Background:
-----------------------
Morgan fingerprints (Rogers and Hahn, 2010, Extended-Connectivity Fingerprints - ECFP)
capture circular topological atom environments up to a specified radius:
- Radius 2 corresponds to ECFP4 (diameter of 4 bonds).
- Each atom environment is hashed deterministically into a fixed-length bit vector (default 2048 bits).
- Binary fingerprints indicate presence/absence (0 or 1).
- Count fingerprints record the frequency of each atom environment occurrence.

RDKit API Documentation:
------------------------
- Modern API: `rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator(radius, fpSize)`
- Binary conversion: `generator.GetFingerprintAsNumPy(mol)` -> `np.ndarray` (dtype=uint8)
- Count conversion: `generator.GetCountFingerprintAsNumPy(mol)` -> `np.ndarray` (dtype=uint32)
- SMILES parsing: `rdkit.Chem.MolFromSmiles(smiles, sanitize=True)`

Guarantees:
-----------
- Strictly preserves input row sequence.
- Deterministic output for identical inputs and parameters.
- Rejects invalid structures with explicit `ValueError` identifying row index and invalid string.
- Never uses toxicity endpoint labels.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

logger = logging.getLogger(__name__)

DEFAULT_MORGAN_RADIUS: int = 2
DEFAULT_MORGAN_N_BITS: int = 2048


def get_morgan_generator(
    radius: int = DEFAULT_MORGAN_RADIUS,
    n_bits: int = DEFAULT_MORGAN_N_BITS,
) -> rdFingerprintGenerator.FingerprintGenerator:
    """Create and return an RDKit Morgan fingerprint generator.

    Uses RDKit's modern `rdFingerprintGenerator.GetMorganGenerator` API.

    Args:
        radius: Circular radius for atom environments (default 2 corresponds to ECFP4).
        n_bits: Number of bits in the hashed bit vector (default 2048).

    Returns:
        Configured FingerprintGenerator instance.

    Raises:
        ValueError: If radius < 1 or n_bits < 1.
    """
    if radius < 1:
        raise ValueError(f"Morgan radius must be >= 1, got {radius}")
    if n_bits < 1:
        raise ValueError(f"Fingerprint size (n_bits) must be >= 1, got {n_bits}")

    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def compute_morgan_fingerprint(
    structure: Union[str, Chem.Mol],
    radius: int = DEFAULT_MORGAN_RADIUS,
    n_bits: int = DEFAULT_MORGAN_N_BITS,
    count: bool = False,
    generator: rdFingerprintGenerator.FingerprintGenerator | None = None,
) -> np.ndarray:
    """Compute a single Morgan fingerprint for a molecule or SMILES string.

    Args:
        structure: Valid SMILES string or RDKit Chem.Mol object.
        radius: Morgan radius (default 2, ECFP4 equivalent).
        n_bits: Bit length (default 2048).
        count: If True, returns atom environment frequency count array (uint32);
               if False, returns binary bit vector (uint8).
        generator: Optional pre-configured generator to reuse.

    Returns:
        1D NumPy array of shape (n_bits,) with dtype uint8 (binary) or uint32 (count).

    Raises:
        ValueError: If structure is invalid or unparsable by RDKit.
    """
    if isinstance(structure, str):
        if not structure or not structure.strip():
            raise ValueError("Structure SMILES cannot be empty or blank")
        mol = Chem.MolFromSmiles(structure.strip(), sanitize=True)
        if mol is None:
            raise ValueError(f"RDKit failed to parse SMILES: '{structure}'")
    elif isinstance(structure, Chem.Mol):
        mol = structure
    else:
        raise TypeError(f"Expected str or Chem.Mol, got {type(structure).__name__}")

    gen = generator if generator is not None else get_morgan_generator(radius, n_bits)

    if count:
        return gen.GetCountFingerprintAsNumPy(mol)
    return gen.GetFingerprintAsNumPy(mol)


def compute_morgan_fingerprints(
    smiles_list: Sequence[str] | Iterable[str],
    radius: int = DEFAULT_MORGAN_RADIUS,
    n_bits: int = DEFAULT_MORGAN_N_BITS,
    count: bool = False,
) -> np.ndarray:
    """Compute Morgan fingerprints for a batch of SMILES strings.

    Preserves input sequence exactly. Does not drop or reorder any rows.
    Raises ValueError immediately if an invalid SMILES is encountered,
    reporting the 0-based index and the invalid string.

    Args:
        smiles_list: Sequence or iterable of SMILES strings.
        radius: Morgan circular radius (default 2).
        n_bits: Bit vector length (default 2048).
        count: If True, returns count fingerprints (uint32);
               if False, returns binary bit matrix (uint8).

    Returns:
        2D NumPy array of shape (N, n_bits) with dtype uint8 (binary) or uint32 (count).

    Raises:
        ValueError: If input is empty or if any SMILES fails RDKit parsing.
    """
    items = list(smiles_list)
    if not items:
        raise ValueError("Cannot compute fingerprints for an empty list of SMILES")

    gen = get_morgan_generator(radius=radius, n_bits=n_bits)
    dtype = np.uint32 if count else np.uint8
    matrix = np.empty((len(items), n_bits), dtype=dtype)

    for i, s in enumerate(items):
        if not isinstance(s, str) or not s.strip():
            raise ValueError(f"Invalid SMILES at row index {i}: '{s}'. Empty or non-string structure.")

        mol = Chem.MolFromSmiles(s.strip(), sanitize=True)
        if mol is None:
            raise ValueError(
                f"Invalid molecular structure at row index {i}: '{s}'. "
                f"RDKit Chem.MolFromSmiles parsing failed."
            )

        if count:
            matrix[i] = gen.GetCountFingerprintAsNumPy(mol)
        else:
            matrix[i] = gen.GetFingerprintAsNumPy(mol)

    return matrix


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for standalone fingerprint generation."""
    parser = argparse.ArgumentParser(
        description="Compute Morgan fingerprints for cleaned Tox21 molecular dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/tox21_cleaned.csv"),
        help="Path to cleaned dataset CSV (default: data/processed/tox21_cleaned.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/features/morgan_r2_2048.npy"),
        help="Output .npy file destination",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default="canonical_smiles",
        help="Name of SMILES column to use (default: canonical_smiles)",
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
        help="Fingerprint bit length (default: 2048)",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="If set, compute count fingerprints instead of binary",
    )
    return parser.parse_args()


def main() -> None:
    """Run CLI fingerprint extraction."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    import pandas as pd

    logger.info("Reading molecules from %s (column: %s)...", args.input, args.smiles_column)
    df = pd.read_csv(args.input)
    if args.smiles_column not in df.columns:
        raise KeyError(f"Column '{args.smiles_column}' not found in {args.input}. Columns: {list(df.columns)}")

    smiles = df[args.smiles_column].tolist()
    logger.info("Computing Morgan fingerprints (N=%d, radius=%d, n_bits=%d, count=%s)...",
                len(smiles), args.radius, args.n_bits, args.count)
    fp_matrix = compute_morgan_fingerprints(
        smiles,
        radius=args.radius,
        n_bits=args.n_bits,
        count=args.count,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, fp_matrix)
    logger.info("Successfully saved fingerprint matrix to %s (shape: %s, dtype: %s)",
                args.output, fp_matrix.shape, fp_matrix.dtype)


if __name__ == "__main__":
    main()
