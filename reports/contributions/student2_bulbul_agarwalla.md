# Student 2 Contribution Report: Molecular Features and Split Integrity

**Name:** Bulbul Agarwalla  
**Role:** Feature Owner  
**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Scope:** Phase 1 — Chemical Feature Representations and Scaffold-Based Dataset Splitting  

---

## 1. Responsibilities

- Implement circular topological fingerprints (Morgan / ECFP4 equivalent) using modern RDKit generators.
- Implement the nine physicochemical molecular descriptors capturing global steric and pharmacokinetic properties.
- Construct the combined 2,057-dimensional feature matrix with verified row alignment and checksum manifests.
- Implement deterministic Bemis–Murcko scaffold-based dataset splitting (80% train, 10% val, 10% test).
- Mathematically verify that scaffolds and row indices do not leak across partitions, and formulate a clear policy for acyclic molecules.

---

## 2. Technical Deliverables

### Source Code (`src/features/` & `src/data/`)
- `src/features/fingerprints.py`: Generates Morgan circular fingerprints (radius 2, 2,048 bits) using RDKit's `rdFingerprintGenerator.GetMorganGenerator`. Supports binary (`uint8`) and count (`uint32`) representations.
- `src/features/descriptors.py`: Calculates 9 standard 2D molecular descriptors:
  1. `mol_weight` (Molecular Weight)
  2. `logp` (Wildman-Crippen LogP)
  3. `tpsa` (Topological Polar Surface Area)
  4. `num_h_donors` (Hydrogen Bond Donors)
  5. `num_h_acceptors` (Hydrogen Bond Acceptors)
  6. `num_rotatable_bonds` (Rotatable Bonds)
  7. `ring_count` (Ring Count)
  8. `heavy_atom_count` (Heavy Atom Count)
  9. `fraction_csp3` (Fraction $sp^3$ Carbons)
- `src/features/build_features.py`: Pipeline coordinator extracting all three representations, validating matrix dimensions, generating row alignment mapping (`feature_row_ids.json`), and writing SHA-256 provenance (`feature_manifest.json`).
- `src/data/split_data.py`: Bemis–Murcko scaffold grouping algorithm allocating molecules into train, validation, and test splits while preserving scaffold isolation.

### Generated Artifacts
- `data/processed/features/morgan_r2_2048.npy`: Binary Morgan fingerprint matrix of shape $(7823, 2048)$, dtype `uint8`.
- `data/processed/features/descriptors_9.npy`: Standardized continuous descriptor matrix of shape $(7823, 9)$, dtype `float64`.
- `data/processed/features/combined_features.npy`: Concatenated feature matrix of shape $(7823, 2057)$, dtype `float32`.
- `data/processed/features/feature_row_ids.json`: 1-to-1 index mapping aligning feature rows with cleaned dataset rows.
- `data/processed/features/feature_manifest.json`: Checksum manifest recording shapes, data types, and SHA-256 hashes.
- `data/processed/scaffold_split_indices.json`: Partition index mapping defining the disjoint train, validation, and test splits.
- `reports/features_and_splitting.md`: Technical documentation of feature engineering and scaffold partitioning.

---

## 3. Key Findings & Scientific Measurements

- **Feature Matrix Validity:** Confirmed 0 non-finite values (no NaNs, Infs). Active bit coverage reached 2,028 out of 2,048 bits across the dataset, with a mean bit density of 1.86% (38.1 bits set per molecule).
- **Physical Descriptor Bounds:** All 9 descriptors adhere to physical chemical ranges (e.g., MW in $[9.01, 1877.60]$ g/mol; TPSA in $[0.00, 777.64]$ $\text{Å}^2$).
- **Scaffold Splitting Proportions (Seed 42):**
  - Total valid molecules: 7,823 (2,325 unique scaffolds).
  - Train Partition: 6,258 molecules (79.995%) — 760 unique cyclic scaffolds + acyclic cluster.
  - Validation Partition: 782 molecules (9.996%) — 782 unique scaffolds.
  - Test Partition: 783 molecules (10.009%) — 783 unique scaffolds.
  - **Zero Scaffold Overlap:** $\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset$, $\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset$, $\mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$.
- **Acyclic Policy:** 1,775 molecules with empty scaffolds (`""`) are safely isolated strictly in the training partition under the `single_group` policy, preventing empty-scaffold cross-partition leakage.

---

## 4. Automated Unit & Integration Tests

Authored test modules in `tests/`:
- `tests/test_fingerprints.py`: Tests radius/bit dimensions, batch consistency, bit-vector non-negativity, and SMILES error handling.
- `tests/test_descriptors.py`: Tests descriptor mathematical validity, physical plausibility bounds, and zero NaN/Inf assertions.
- `tests/test_build_features.py`: Tests matrix concatenation, row ID alignment, and manifest checksum verification.
- `tests/test_split_integrity.py`: Validates partition disjointness, scaffold isolation, determinism under seed 42, and real-data accounting.

**Test Execution:** 23 unit and integration tests passing deterministically (`pytest tests/test_fingerprints.py tests/test_descriptors.py tests/test_split_integrity.py -v`).
