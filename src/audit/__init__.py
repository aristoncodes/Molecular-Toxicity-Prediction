"""Automated scientific audit and reproducibility suite.

Student 3 Module: Independent verification of dataset integrity, scaffold splitting,
feature leakage, evaluation metrics, and uncertainty estimation.
"""

from typing import Any

__all__ = [
    "audit_dataset_provenance_and_checksum",
    "audit_raw_vs_cleaned_accounting",
    "audit_canonical_smiles_integrity",
    "audit_label_encodings_and_missingness",
    "run_dataset_audit",
]


def __getattr__(name: str) -> Any:
    if name in (
        "audit_dataset_provenance_and_checksum",
        "audit_raw_vs_cleaned_accounting",
        "audit_canonical_smiles_integrity",
        "audit_label_encodings_and_missingness",
        "run_dataset_audit",
    ):
        from src.audit import dataset_audit

        return getattr(dataset_audit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
