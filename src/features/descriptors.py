"""Molecular Descriptors Computation Module.

Student 2 Module: Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks.

This module calculates nine standard physicochemical molecular descriptors using RDKit:
1. Molecular Weight (MolWt, g/mol)
2. Wildman-Crippen LogP (MolLogP, octanol-water partition coefficient)
3. Topological Polar Surface Area (TPSA, Å²)
4. Hydrogen-Bond Donors (NumHDonors, count)
5. Hydrogen-Bond Acceptors (NumHAcceptors, count)
6. Rotatable Bonds (NumRotatableBonds, count)
7. Ring Count (RingCount, count)
8. Heavy Atom Count (HeavyAtomCount, count)
9. Fraction of sp3 Carbons (FractionCSP3, ratio of sp3 carbons to total carbons)

RDKit Functions Used:
---------------------
- `rdkit.Chem.Descriptors.MolWt` (Average molecular weight)
- `rdkit.Chem.Crippen.MolLogP` (Wildman-Crippen octanol-water partition coefficient)
- `rdkit.Chem.Descriptors.TPSA` (Topological polar surface area)
- `rdkit.Chem.Lipinski.NumHDonors` (Number of H-bond donors)
- `rdkit.Chem.Lipinski.NumHAcceptors` (Number of H-bond acceptors)
- `rdkit.Chem.Lipinski.NumRotatableBonds` (Number of non-trivial rotatable bonds)
- `rdkit.Chem.Lipinski.RingCount` (Number of rings in the molecule)
- `rdkit.Chem.Lipinski.HeavyAtomCount` (Count of non-hydrogen atoms)
- `rdkit.Chem.Lipinski.FractionCSP3` (Fraction of carbons that are sp3 hybridized)

Guarantees:
-----------
- Matrix rows strictly match input molecule order.
- Feature column sequence is immutable and documented in `DESCRIPTOR_NAMES`.
- Strict validation rejects invalid structures and detects unexpected NaNs/Infs.
- Independent of label columns (no label leakage).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final, Iterable, Sequence, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

logger = logging.getLogger(__name__)

# Ordered tuple of standard descriptor names
DESCRIPTOR_NAMES: Final[tuple[str, ...]] = (
    "mol_weight",
    "logp",
    "tpsa",
    "num_h_donors",
    "num_h_acceptors",
    "num_rotatable_bonds",
    "ring_count",
    "heavy_atom_count",
    "fraction_csp3",
)

DESCRIPTOR_METADATA: Final[dict[str, dict[str, str]]] = {
    "mol_weight": {
        "rdkit_function": "Descriptors.MolWt",
        "unit": "g/mol",
        "description": "Average molecular weight computed from standard atomic weights",
    },
    "logp": {
        "rdkit_function": "Crippen.MolLogP",
        "unit": "dimensionless",
        "description": "Wildman-Crippen octanol-water partition coefficient",
    },
    "tpsa": {
        "rdkit_function": "Descriptors.TPSA",
        "unit": "Å²",
        "description": "Topological polar surface area based on polar fragments (N, O, etc.)",
    },
    "num_h_donors": {
        "rdkit_function": "Lipinski.NumHDonors",
        "unit": "count",
        "description": "Number of hydrogen bond donors (OH and NH groups)",
    },
    "num_h_acceptors": {
        "rdkit_function": "Lipinski.NumHAcceptors",
        "unit": "count",
        "description": "Number of hydrogen bond acceptors (O and N atoms)",
    },
    "num_rotatable_bonds": {
        "rdkit_function": "Lipinski.NumRotatableBonds",
        "unit": "count",
        "description": "Number of rotatable single bonds (excluding terminal bonds)",
    },
    "ring_count": {
        "rdkit_function": "Lipinski.RingCount",
        "unit": "count",
        "description": "Total number of rings in the smallest set of smallest rings (SSSR)",
    },
    "heavy_atom_count": {
        "rdkit_function": "Lipinski.HeavyAtomCount",
        "unit": "count",
        "description": "Number of non-hydrogen atoms in the molecule",
    },
    "fraction_csp3": {
        "rdkit_function": "Lipinski.FractionCSP3",
        "unit": "ratio (0.0 to 1.0)",
        "description": "Fraction of carbon atoms that are sp3 hybridized",
    },
}


def compute_molecular_descriptors(
    structure: Union[str, Chem.Mol],
) -> np.ndarray:
    """Compute the nine standard descriptors for a single molecule.

    Args:
        structure: Valid SMILES string or RDKit Chem.Mol object.

    Returns:
        1D NumPy array of shape (9,) with dtype float64 in `DESCRIPTOR_NAMES` order.

    Raises:
        ValueError: If SMILES cannot be parsed by RDKit or produces NaN/Inf.
        TypeError: If structure is neither str nor Chem.Mol.
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

    values = np.array(
        [
            float(Descriptors.MolWt(mol)),
            float(Crippen.MolLogP(mol)),
            float(Descriptors.TPSA(mol)),
            float(Lipinski.NumHDonors(mol)),
            float(Lipinski.NumHAcceptors(mol)),
            float(Lipinski.NumRotatableBonds(mol)),
            float(Lipinski.RingCount(mol)),
            float(Lipinski.HeavyAtomCount(mol)),
            float(Lipinski.FractionCSP3(mol)),
        ],
        dtype=np.float64,
    )

    if np.isnan(values).any() or np.isinf(values).any():
        nan_indices = [
            DESCRIPTOR_NAMES[idx]
            for idx in range(len(values))
            if np.isnan(values[idx]) or np.isinf(values[idx])
        ]
        raise ValueError(f"Descriptor calculation produced NaN or Inf for: {nan_indices}")

    return values


def compute_descriptors_matrix(
    smiles_list: Sequence[str] | Iterable[str],
) -> np.ndarray:
    """Compute the 9 descriptors for a batch of molecules.

    Preserves input sequence exactly. Does not drop or reorder rows.
    Raises ValueError immediately if an invalid SMILES or unexpected NaN/Inf occurs.

    Args:
        smiles_list: Sequence or iterable of SMILES strings.

    Returns:
        2D NumPy array of shape (N, 9) with dtype float64 in `DESCRIPTOR_NAMES` column order.

    Raises:
        ValueError: If input is empty or any SMILES cannot be parsed.
    """
    items = list(smiles_list)
    if not items:
        raise ValueError("Cannot compute descriptors for an empty list of SMILES")

    n_features = len(DESCRIPTOR_NAMES)
    matrix = np.empty((len(items), n_features), dtype=np.float64)

    for i, s in enumerate(items):
        if not isinstance(s, str) or not s.strip():
            raise ValueError(f"Invalid SMILES at row index {i}: '{s}'. Empty or non-string structure.")

        mol = Chem.MolFromSmiles(s.strip(), sanitize=True)
        if mol is None:
            raise ValueError(
                f"Invalid molecular structure at row index {i}: '{s}'. "
                f"RDKit Chem.MolFromSmiles parsing failed."
            )

        row_vals = compute_molecular_descriptors(mol)
        matrix[i] = row_vals

    return matrix


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for standalone descriptor computation."""
    parser = argparse.ArgumentParser(
        description="Compute 9 physicochemical molecular descriptors for cleaned Tox21 dataset."
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
        default=Path("data/processed/features/descriptors_9.npy"),
        help="Output .npy file destination",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default="canonical_smiles",
        help="Name of SMILES column to use (default: canonical_smiles)",
    )
    return parser.parse_args()


def main() -> None:
    """Run CLI descriptor extraction."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    import pandas as pd

    logger.info("Reading molecules from %s (column: %s)...", args.input, args.smiles_column)
    df = pd.read_csv(args.input)
    if args.smiles_column not in df.columns:
        raise KeyError(f"Column '{args.smiles_column}' not found in {args.input}. Columns: {list(df.columns)}")

    smiles = df[args.smiles_column].tolist()
    logger.info("Computing 9 descriptors for N=%d molecules...", len(smiles))
    desc_matrix = compute_descriptors_matrix(smiles)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, desc_matrix)
    logger.info("Successfully saved descriptor matrix to %s (shape: %s, dtype: %s)",
                args.output, desc_matrix.shape, desc_matrix.dtype)


if __name__ == "__main__":
    main()
