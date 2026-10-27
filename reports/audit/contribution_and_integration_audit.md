# Student Contribution and Multi-Contributor Integration Audit

**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Scope:** Verifiable Git commit history, pull request merges, module authorship, and cross-contributor integration across Students 1, 2, 3, and 4.  
**Auditor:** Senior ML Engineer & Quality-Assurance Lead  
**Audit Date:** 2026-10-27  

---

## 1. Executive Contribution Accounting

All 4 students have executed their respective responsibilities through auditable Git commits, modular code design, thorough unit tests, and rigorous scientific documentation.

| Contributor | Official Role | Verified Git Commits | Core Modules Authored | Test Coverage | Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **Student 1** (Amogha Raj Sandur) | Data Acquisition, Sanitization & EDA | 4 commits (`c0e142e` .. `52b0961`) | `src/data/download_data.py`, `src/data/preprocess.py`, `src/data/inspect_dataset.py`, `src/data/generate_eda_plots.py` | 30 tests (`test_inspect_dataset.py`, `test_preprocess.py`, `test_eda.py`) | **VERIFIED** |
| **Student 2** (Bulbul Agarwalla) | Feature Extraction & Scaffold Splitting | 8 commits (`4b28acd` .. `1f0e9ba`) | `src/features/fingerprints.py`, `src/features/descriptors.py`, `src/features/build_features.py`, `src/data/split_data.py` | 23 tests (`test_fingerprints.py`, `test_split_integrity.py`) | **VERIFIED** |
| **Student 3** (Shourya Pratap) | Quality Assurance, Leakage Prevention & Audit | 9 commits (`5a1ddb0` .. `589ac3e`) | `src/features/preprocessor.py`, `src/evaluation/metrics.py`, `src/evaluation/aggregation.py`, `src/audit/` | 22 tests (`test_preprocessor_leakage.py`, `test_metrics.py`, `test_reproducibility_suite.py`) | **VERIFIED** |
| **Student 4** (Kanishka Dubey) | Model Evaluation & Benchmarking | 9 commits (`8a39c9e` .. `fe6413f`) | `src/evaluation/feature_validation.py`, `src/evaluation/endpoint_analysis.py`, `src/evaluation/model_evaluation.py`, `src/evaluation/error_analysis.py` | 42 tests (`test_feature_validation.py`, `test_endpoint_analysis.py`, `test_model_evaluation.py`, `test_error_analysis.py`, `test_final_evaluation_regression.py`) | **VERIFIED** |

---

## 2. Granular Student Contribution Audit

### 2.1 Student 1: Data Curation, Preprocessing, and EDA
- **Commit Range:** `c0e142e` to `52b0961` (Merged into `main`).
- **Primary Deliverables:**
  - Automated streaming download with SHA-256 validation (`45d09792...`) for `data/raw/tox21.csv.gz`.
  - Rigorous structure validation using RDKit sanitization: 7,831 raw records processed $\to$ 8 organometallic valence violations logged in `exclusions.csv` $\to$ 7,823 valid structures saved in `tox21_cleaned.csv`.
  - Multi-task duplicate and conflict detection ensuring 0 canonical duplicate groups and 0 contradictory labels.
  - Generation of 10 publication-quality distribution and prevalence figures in `results/figures/eda/` with paired numerical JSON summaries.
- **Verification:** All 30 upstream tests pass deterministically.

### 2.2 Student 2: Molecular Features and Scaffold Splitting
- **Commit Range:** `4b28acd` to `1f0e9ba` (8 atomic commits).
- **Primary Deliverables:**
  - Morgan circular topological fingerprints ($r=2$, 2048-bit, binary `uint8`) via modern RDKit `rdFingerprintGenerator`.
  - 9 physicochemical molecular descriptors (MW, LogP, TPSA, HBD, HBA, rotatable bonds, ring count, heavy atom count, fraction $sp^3$) in `descriptors_9.npy`.
  - Combined feature matrix concatenation (2,057 dimensions, `float32`) with row alignment mapping in `feature_row_ids.json` and checksums in `feature_manifest.json`.
  - Deterministic Bemis-Murcko scaffold splitting allocating molecules into Train (80%), Val (10%), and Test (10%) partitions with 0 scaffold overlap.
- **Verification:** All 23 feature and split integrity tests pass.

### 2.3 Student 3: QA Lead, Leakage Prevention, and Audit Suite
- **Commit Range:** `5a1ddb0` to `589ac3e` (9 commits).
- **Primary Deliverables:**
  - Independent verification of data provenance and scaffold isolation.
  - Discovered and resolved **ISSUE-01 (Preprocessing Leakage)**: built `LeakageSafeScaler` and `PartitionPreprocessor` to fit scaling strictly on training indices and keep Morgan bits unscaled.
  - Implemented continuous evaluation metrics (`compute_roc_auc`, `compute_pr_auc`, masking unlabelled $\text{NaN}$ values, and handling single-class edge cases).
  - Standardized uncertainty reporting via `aggregate_endpoint_metrics` with explicit $\text{Mean} \pm \text{SEM}$ ($K=12$) and $\text{SD}$.
  - Authored the unified automated audit runner `src/audit/run_audit.py` and IEEE reconciliation matrix.
- **Verification:** All 22 audit and leakage tests pass.

### 2.4 Student 4: Feature Validation, Modeling, and Benchmarking
- **Commit Range:** `8a39c9e` to `fe6413f` (9 commits; merged via PR #2).
- **Primary Deliverables:**
  - Comprehensive feature matrix validation suite verifying dimensions, value domains, and bit sparsity.
  - Multi-task label diagnostics quantifying active/inactive/missing counts across all 12 endpoints.
  - Multi-task baseline modeling pipeline training Logistic Regression, Random Forest, and XGBoost across Morgan, Descriptors, and Combined representations (9 benchmark configurations).
  - Out-of-sample test prediction generation and validation-only threshold tuning ($\theta^* \in [0.05, 0.95]$) for discrete classification metrics (F1, Balanced Accuracy).
  - Error analysis diagnosing endpoint difficulty and challenging chemotypes, producing 3 summary evaluation figures.
  - Technical research report `reports/phase1_model_evaluation.md`.
- **Verification:** All 42 validation and evaluation tests pass.

---

## 3. Integration Integrity and Architecture Cohesion

1. **Non-Conflicting Modular Structure:** Each student's modules strictly interface via well-defined data artifacts and standardized Python APIs without circular imports:
   $$\text{Student 1 (Data)} \longrightarrow \text{Student 2 (Features/Splits)} \longrightarrow \text{Student 3 (QA/Leakage/Metrics)} \longrightarrow \text{Student 4 (Modeling/Benchmarking)}$$
2. **Total Test Suite:** The full integration test suite comprises **117 automated tests**, all executing and passing in under 10 seconds.
3. **Repository State:** The default branch `main` is clean, synchronized with `origin/main`, and all pull requests have been properly merged without outstanding conflicts.
