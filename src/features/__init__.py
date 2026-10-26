"""Features module for molecular toxicity prediction.

Student 2 Module:
- Morgan chemical fingerprints (binary and count)
- 9 Physicochemical molecular descriptors
- Combined feature representations and manifests
"""

from typing import Any

__all__ = [
    "compute_morgan_fingerprint",
    "compute_morgan_fingerprints",
    "DEFAULT_MORGAN_RADIUS",
    "DEFAULT_MORGAN_N_BITS",
    "compute_molecular_descriptors",
    "compute_descriptors_matrix",
    "DESCRIPTOR_NAMES",
    "DESCRIPTOR_METADATA",
    "build_feature_representations",
    "generate_feature_names",
]


def __getattr__(name: str) -> Any:
    if name in (
        "compute_morgan_fingerprint",
        "compute_morgan_fingerprints",
        "DEFAULT_MORGAN_RADIUS",
        "DEFAULT_MORGAN_N_BITS",
    ):
        from src.features import fingerprints

        return getattr(fingerprints, name)
    if name in (
        "compute_molecular_descriptors",
        "compute_descriptors_matrix",
        "DESCRIPTOR_NAMES",
        "DESCRIPTOR_METADATA",
    ):
        from src.features import descriptors

        return getattr(descriptors, name)
    if name in (
        "build_feature_representations",
        "generate_feature_names",
    ):
        from src.features import build_features

        return getattr(build_features, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
