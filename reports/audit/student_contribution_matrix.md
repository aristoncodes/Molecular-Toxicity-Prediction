# Student Contribution Matrix

**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Audit Date:** 2026-10-26  
**Auditor:** Senior ML Engineer & Quality-Assurance Lead  
**Scope:** Verification of all commits, code artifacts, tests, and documentation from Students 1, 2, and 3.

---

## 1. Executive Summary of Student Contributions

| Student | Assigned Area | Active Commits | Files Added / Modified | Tests Added | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Student 1** (Amogha Raj Sandur) | Dataset Acquisition, Structure Curation, Preprocessing, EDA | 4 commits (`c0e142e`, `a58062c`, `49135ce`, `52b0961` [merge]) | 13 files (`src/data/common.py`, `download_data.py`, `inspect_dataset.py`, `preprocess.py`, `generate_eda_plots.py`, etc.) | 30 test cases (`tests/test_download_data.py`, `test_inspect_dataset.py`, `test_preprocess.py`, `test_eda.py`) | **VERIFIED** |
| **Student 2** (Bulbul Agarwalla) | Morgan Fingerprints, Physicochemical Descriptors, Combined Matrix, Scaffold Splitting | 7 commits (`4b28acd`, `872824b`, `e3658d5`, `ed44e28`, `fbadc4b`, `c17437b`, `8e45199`, plus PR merge `1f0e9ba`) | 11 files (`src/features/fingerprints.py`, `descriptors.py`, `build_features.py`, `src/data/split_data.py`, `src/features/__init__.py`, etc.) | 26 test cases (`tests/test_fingerprints.py`, `test_descriptors.py`, `test_build_features.py`, `test_split_integrity.py`) | **VERIFIED** |
| **Student 3** (Shourya Pratap) | Independent Audit, Preprocessing Leakage Prevention, Evaluation Metrics, Reproducibility Suite, IEEE Manuscript | 8 commits (`5a1ddb0`, `11d24eb`, `7407474`, `a24fe48`, `06c5653`, `927665e`, `05972a6`, `07752df`) | 17 files (`src/audit/dataset_audit.py`, `split_audit.py`, `run_audit.py`, `src/features/preprocessor.py`, `src/evaluation/metrics.py`, `aggregation.py`, `reports/ieee_phase1_manuscript.tex`, `reports/student3_phase1_audit.md`, etc.) | 29 test cases (`tests/test_audit_dataset.py`, `test_audit_split.py`, `test_preprocessor_leakage.py`, `test_metrics.py`, `test_aggregation.py`, `test_reproducibility_suite.py`) | **VERIFIED** |

---

## 2. Commit-by-Commit Contribution Attribution Matrix

| Student | Commit Hash | Author / Date | Files Changed | Contribution Summary | Verification Method & Evidence | Issues / Observations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Student 1** | `c0e142e` | Amogha Raj Sandur<br>2026-10-25 | 8 files (`src/data/download_data.py`, `inspect_dataset.py`, `preprocess.py`, `common.py`, `configs/data_config.yaml`, `.gitignore`) | Implemented verified Tox21 acquisition, gzip CRC32 checking, structure inspection, and preliminary preprocessing. | Unit test execution, download inspection against reference SHA-256 (`45d09792...`). | Initial commit required real data flag handling for offline testing. |
| **Student 1** | `a58062c` | Amogha Raj Sandur<br>2026-10-25 | 13 files (`src/data/generate_eda_plots.py`, `reports/dataset_eda_report.md`, `tests/conftest.py`, `tests/test_*.py`) | Added 10 EDA figures, JSON summaries, offline data quality fixtures, and unit test suite. | `pytest tests/` offline run: 29 passed. Inspected generated PNG/PDF figures. | None. Figures faithfully represent audited 7,823 structures. |
| **Student 1** | `49135ce` | Amogha Raj Sandur<br>2026-10-25 | 7 files (`README.md`, `src/data/common.py`, `tests/test_download_data.py`, `tests/test_eda.py`, etc.) | Validated real Tox21 pipeline against remote source; recorded measured EDA figures. | Ran real dataset audit: 30 passed (`--run-real-data`). | None. |
| **Student 1** | `52b0961` | Amogha Raj Sandur<br>2026-10-25 | Merge commit | Merged initial repo history with Student 1 data module. | Git log graph inspection. | Clean merge. |
| **Student 2** | `4b28acd` | Bulbul Agarwalla<br>2026-10-26 | 1 file (`reports/student2_integration_audit.md`) | Documented independent upstream audit of Student 1 data deliverables before feature work. | Verified against `reports/student2_integration_audit.md`. | Verified all checks passed. |
| **Student 2** | `872824b` | Bulbul Agarwalla<br>2026-10-26 | 3 files (`src/features/fingerprints.py`, `src/features/__init__.py`, `tests/test_fingerprints.py`) | Implemented circular Morgan fingerprints (radius 2, 2048 bits) with RDKit `rdFingerprintGenerator`. | `pytest tests/test_fingerprints.py`: 9 passed. Shape `(N, 2048)`, uint8. | Binary and count representations tested. |
| **Student 2** | `e3658d5` | Bulbul Agarwalla<br>2026-10-26 | 3 files (`src/features/descriptors.py`, `src/features/__init__.py`, `tests/test_descriptors.py`) | Implemented 9 physicochemical molecular descriptors with RDKit; verified against known molecules. | `pytest tests/test_descriptors.py`: 8 passed. Shape `(N, 9)`, float64. | No scaling in feature module (correctly deferred to training). |
| **Student 2** | `ed44e28` | Bulbul Agarwalla<br>2026-10-26 | 3 files (`src/features/build_features.py`, `src/features/__init__.py`, `tests/test_build_features.py`) | Assembled combined 2,057-dim feature matrix, row alignment mapping, and manifest. | `pytest tests/test_build_features.py`: 4 passed. Shape `(7823, 2057)`, float32. | Verified concatenation slice invariance. |
| **Student 2** | `fbadc4b` | Bulbul Agarwalla<br>2026-10-26 | 2 files (`src/data/split_data.py`, `tests/test_split_integrity.py`) | Implemented deterministic Bemis–Murcko scaffold split (80/10/10, seed 42) and integrity tests. | `pytest tests/test_split_integrity.py`: 5 passed. Confirmed 0 scaffold overlap. | Acyclic policy (`single_group`) places 1,775 molecules in train. |
| **Student 2** | `c17437b` | Bulbul Agarwalla<br>2026-10-26 | 2 files (`README.md`, `reports/student2_features_and_splitting.md`) | Documented feature extractions, proportions, scaffold accounting, and CLI reproduction. | Inspected report numbers against saved artifacts. | Accurately recorded split: 6258 / 782 / 783. |
| **Student 2** | `8e45199` | Bulbul Agarwalla<br>2026-10-26 | 1 file (`src/features/__init__.py`) | Optimized lazy loading of submodules via `__getattr__` for fast CLI invocation. | CLI execution verification without full module eager import. | Performance optimization. |
| **Student 2** | `1f0e9ba` | Bulbul Agarwalla<br>2026-10-26 | Merge commit | Merged PR #1 (`student-2-features-and-splitting`) into `main`. | Git graph verification. | Clean merge. |
| **Student 3** | `5a1ddb0` | Shourya Pratap<br>2026-10-26 | 3 files (`src/audit/__init__.py`, `src/audit/dataset_audit.py`, `tests/test_audit_dataset.py`) | Built independent dataset integrity auditor verifying raw CRC/SHA-256, exclusions, and canonical invariance. | `pytest tests/test_audit_dataset.py`: 5 passed. | Verified 8 exclusions are unparsable Al valence errors. |
| **Student 3** | `11d24eb` | Shourya Pratap<br>2026-10-26 | 3 files (`src/audit/split_audit.py`, `src/audit/__init__.py`, `tests/test_audit_split.py`) | Built independent scaffold isolation auditor; recomputed scaffolds from scratch with RDKit. | `pytest tests/test_audit_split.py`: 7 passed. | Confirmed 0 non-empty scaffold leakage between train/val/test. |
| **Student 3** | `7407474` | Shourya Pratap<br>2026-10-26 | 3 files (`src/features/preprocessor.py`, `src/features/__init__.py`, `tests/test_preprocessor_leakage.py`) | Fixed preprocessing leakage: implemented `PartitionPreprocessor` fitting scalers strictly on train indices. | `pytest tests/test_preprocessor_leakage.py`: 4 passed. | Prevents test distribution leakage while keeping Morgan bits binary. |
| **Student 3** | `a24fe48` | Shourya Pratap<br>2026-10-26 | 3 files (`src/evaluation/__init__.py`, `src/evaluation/metrics.py`, `tests/test_metrics.py`) | Implemented evaluation metrics: continuous ROC-AUC, PR-AUC (Average Precision), NaN masking, single-class handling. | `pytest tests/test_metrics.py`: 7 passed. | Catches undefined single-class endpoints gracefully. |
| **Student 3** | `06c5653` | Shourya Pratap<br>2026-10-26 | 3 files (`src/evaluation/aggregation.py`, `src/evaluation/__init__.py`, `tests/test_aggregation.py`) | Implemented macro aggregation disambiguating `±`: Mean, SD, SEM, and 95% Student-$t$ CI. | `pytest tests/test_aggregation.py`: 3 passed. | Resolves literature ambiguity between SEM and SD. |
| **Student 3** | `927665e` | Shourya Pratap<br>2026-10-26 | 2 files (`src/audit/run_audit.py`, `tests/test_reproducibility_suite.py`) | Built unified automated scientific audit CLI suite verifying all 7 criteria in one command. | `python -m src.audit.run_audit`: 7/7 checks passed. Output logged to JSON. | Verifies real vs unverified model claims. |
| **Student 3** | `05972a6` | Shourya Pratap<br>2026-10-26 | 2 files (`reports/ieee_phase1_manuscript.tex`, `reports/ieee_reconciliation.md`) | Authored formal IEEE conference manuscript; reconciled claims against real artifacts. | Verified LaTeX structure, citations, and claim-by-claim reconciliation matrix. | Reconciled premature model claims as Phase 2 scope. |
| **Student 3** | `07752df` | Shourya Pratap<br>2026-10-26 | 2 files (`README.md`, `reports/student3_phase1_audit.md`) | Documented audit findings, reproduction procedures, and final Phase 1 validation sign-off. | Documented all 75 passing test commands and deliverables. | Final deliverable for Student 3. |

---

## 3. Discrepancies and Clarifications

1. **Authorship and Commit Ownership:**
   - Student 1 commits are clearly authored by `Amogha Raj Sandur <amogharaj.sandur@yahoo.com>`.
   - Student 2 commits are authored by `Bulbul Agarwalla <agarwallabulbul@gmail.com>`.
   - Student 3 commits are authored by `Shourya Pratap <shouryapuja130406@gmail.com>`.
   - All commits follow clean logical scopes without cross-contamination.

2. **Phase 1 Boundary vs Phase 2 Baseline Models:**
   - An earlier conceptual outline mentioned baseline models (Logistic Regression, Random Forest, XGBoost) and benchmark ROC-AUC scores.
   - The audit confirmed that prior to Student 3, no model training scripts (`run_experiments.py`) or model weight files existed in the repository.
   - Student 3 correctly flagged this discrepancy, implemented the rigorous metrics and uncertainty modules needed for evaluation, and documented that baseline model training constitutes the beginning of **Phase 2**.
