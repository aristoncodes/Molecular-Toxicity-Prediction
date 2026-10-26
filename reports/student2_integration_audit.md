# Student 2 Integration Audit & Verification Report

**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Author:** Student 2 (Features & Scaffold Splitting Specialist)  
**Date:** 2026-10-26  
**Status:** Verification Passed — Safe to Proceed  

---

## 1. Executive Summary

As Student 2, before implementing chemical fingerprint extraction, molecular descriptor computation, feature representation assembly, and scaffold-based train/validation/test splitting, I independently audited and verified Student 1's dataset acquisition, preprocessing pipeline, and exploratory data analysis.

All upstream checks, data contracts, and test suites have been verified with complete rigor. No blocking defects were found in Student 1's implementation. Student 2 can safely proceed with feature engineering and scaffold splitting against `data/processed/tox21_cleaned.csv`.

---

## 2. Files Inspected

The following files from Student 1's deliverable were inspected and analyzed:

- `README.md`: Verified documentation of reproduction pipeline, scientific policies, and validation record.
- `configs/data_config.yaml`: Examined source URL, reference SHA-256 (`45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360`), paths, endpoints, and canonicalization settings.
- `requirements.txt`: Examined pinned dependencies (Python 3.13/3.14 compatibility).
- `pytest.ini`: Inspected test discovery configuration and custom markers (`--run-real-data`).
- `src/__init__.py`, `src/data/__init__.py`: Package namespaces.
- `src/data/common.py`: Shared utilities for hashing, CSV reading/writing, config loading, and atomic file publication.
- `src/data/download_data.py`: Gzip streaming, atomic staging, CRC validation, and SHA-256 verification.
- `src/data/inspect_dataset.py`: Structure validation, label auditing, duplicate auditing, and quality report generation.
- `src/data/preprocess.py`: Canonicalization, duplicate policy enforcement, and cleaned dataset publication.
- `src/data/generate_eda_plots.py`: Generation of 10 publication-quality figures, JSON metrics, and markdown report.
- `data/processed/tox21_cleaned.csv`: Cleaned dataset artifact (7,823 records, 18 columns).
- `data/processed/dataset_manifest.json`: Publication manifest with checksums, record accounting, and configuration.
- `results/experiment_logs/data_quality_report.json`: Comprehensive data quality metrics and per-endpoint counts.
- `results/experiment_logs/exclusions.csv`: Audit log of 8 excluded unparsable structures.
- `results/experiment_logs/duplicates.csv`: Audit log of duplicate structures (0 duplicates found).
- `results/experiment_logs/structure_properties.csv`: Valid-row EDA physicochemical properties.
- `tests/conftest.py`, `tests/test_download_data.py`, `tests/test_inspect_dataset.py`, `tests/test_preprocess.py`, `tests/test_eda.py`: Complete unit test suite (30 test cases).

---

## 3. Environment & Dependencies

- **OS / Architecture:** macOS (Darwin arm64)
- **Python Version:** 3.13.6 (`.venv/bin/python`)
- **Package Status:** `python -m pip check` passed with 0 broken requirements.
- **Core Library Imports:** `rdkit` (2026.9.1), `pandas` (3.1.0), `numpy` (2.5.4), `matplotlib` (3.11.2), `PyYAML` (6.0.3), `pytest` (9.1.1) imported without errors.

---

## 4. Pipeline Execution & Test Results

The full upstream pipeline was executed sequentially from the repository root:

1. `python -m src.data.download_data`:
   - Downloaded `data/raw/tox21.csv.gz` (122,925 bytes).
   - Validated SHA-256: `45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360` (exact match with reference).
2. `python -m src.data.inspect_dataset`:
   - Parsed raw CSV: 7,831 records × 14 columns.
   - Identified 7,823 valid structures and 8 invalid exclusions.
   - Generated `results/experiment_logs/data_quality_report.json`.
3. `python -m src.data.preprocess`:
   - Published `data/processed/tox21_cleaned.csv` (7,823 records, SHA-256 `2a30e42b7addbebb02e8c440f88ac553eb7d20b775583d9bcff256c6c0d07270`).
   - Published `data/processed/dataset_manifest.json`.
4. `python -m src.data.generate_eda_plots`:
   - Generated 10 figures (both PNG and PDF formats) and 10 numerical JSON summaries in `results/figures/eda/`.
   - Generated `reports/dataset_eda_report.md`.
5. **Unit & Integration Test Suite (`pytest`):**
   - Offline tests: `pytest tests/ -v --tb=short` -> **29 passed, 1 skipped** (real-data test skipped without flag).
   - Full suite with real-data audit: `pytest tests/ -v --tb=short --run-real-data` -> **30 passed, 0 failed in 3.29s**.

---

## 5. Dataset Schema & Independent Structure Validation

### 5.1 Dataset Schema

- **Total Raw Records:** 7,831
- **Total Excluded Records:** 8 (All 8 due to unparsable aluminum valence violations, e.g. `NC(=O)NC1N=C(O[AlH3](O)O)NC1=O`, logged in `exclusions.csv`)
- **Total Cleaned Records:** 7,823 rows × 18 columns
- **Primary Molecular Structure Column:** `canonical_smiles` (RDKit sanitized canonical isomeric SMILES)
- **Original Structure Column:** `smiles`
- **Stable Row Identifier:** `source_row` (1-based integer index corresponding to the original raw CSV record number, ranging from 1 to 7,831; unique for all rows) and `mol_id` (Tox21 sample ID, e.g., `TOX3021`).
- **Audit Columns:** `duplicate_group` (empty string for unique structures), `duplicate_status` (`unique` for all 7,823 records).

### 5.2 Endpoint Columns & Label Encodings

The 12 expected Tox21 assay endpoints are present and encoded strictly as binary (`0.0`, `1.0`) or missing (`NaN`):

| Endpoint | Total Observed | Positive (1.0) | Negative (0.0) | Missing (NaN) | Pos. Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NR-AR** | 7,258 | 308 | 6,950 | 565 | 4.24% |
| **NR-AR-LBD** | 6,751 | 237 | 6,514 | 1,072 | 3.51% |
| **NR-AhR** | 6,542 | 768 | 5,774 | 1,281 | 11.74% |
| **NR-Aromatase** | 5,815 | 300 | 5,515 | 2,008 | 5.16% |
| **NR-ER** | 6,186 | 791 | 5,395 | 1,637 | 12.79% |
| **NR-ER-LBD** | 6,948 | 349 | 6,599 | 875 | 5.02% |
| **NR-PPAR-gamma** | 6,443 | 186 | 6,257 | 1,380 | 2.89% |
| **SR-ARE** | 5,825 | 942 | 4,883 | 1,998 | 16.17% |
| **SR-ATAD5** | 7,065 | 264 | 6,801 | 758 | 3.74% |
| **SR-HSE** | 6,460 | 372 | 6,088 | 1,363 | 5.76% |
| **SR-MMP** | 5,804 | 918 | 4,886 | 2,019 | 15.82% |
| **SR-p53** | 6,767 | 423 | 6,344 | 1,056 | 6.25% |

*(Note: Counts reflect valid cleaned rows. One positive in NR-AR was among the 8 unparsable exclusions, accounting for the difference between raw 309 and cleaned 308).*

### 5.3 Independent RDKit Chemical Structure Validation

An independent validation script using RDKit 2026.9.1 verified:
1. **Parsability:** 7,823 of 7,823 (100.0%) `canonical_smiles` entries parse successfully into valid `rdkit.Chem.Mol` objects with `sanitize=True`. Invalid count: **0**.
2. **Canonical Invariance:** Re-running `Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)` on every parsed molecule yielded exact string identity with `canonical_smiles`. Re-canonicalization mismatches: **0**.
3. **Uniqueness:** Number of distinct canonical SMILES in the cleaned dataset is **7,823 out of 7,823**. There are no duplicate structures or duplicate conflict groups in this release.
4. **Missing Label Preservation:** No missing assay values have been converted to 0.0 or imputed.

---

## 6. Shared Data Contract for Student 2

Based on this audit, the confirmed contract between Student 1 and Student 2 is:
- **Cleaned Dataset Path:** `data/processed/tox21_cleaned.csv`
- **Structure Input:** `canonical_smiles` (fallback: `smiles`)
- **Row Mapping:** `source_row` (stable integer identifier) must be preserved across feature matrices, labels, and split indices.
- **Matrix Dimension:** Exactly 7,823 rows in identical sequence to `tox21_cleaned.csv`.
- **Target Proportions for Scaffold Split:** ~80% Train, ~10% Validation, ~10% Test.
- **Labels:** Never included or mixed into feature matrices.

---

## 7. Blockers and Gate Decision

- **Integration Blockers:** None.
- **Environment Blockers:** None.
- **Data Quality Blockers:** None.
- **Gate Decision:** **PASSED — PROCEED TO PHASE C & D.**
