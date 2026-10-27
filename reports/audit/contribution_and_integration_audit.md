# Student Contribution and Multi-Contributor Integration Audit

**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Scope:** Verifiable Git commit history, code authorship, automated test coverage, and cross-module pipeline integration across Students 1, 2, 3, and 4.  
**Auditor:** Senior ML Engineer & Quality-Assurance Lead  
**Audit Date:** 2026-10-27 (Final Audit)  

---

## 1. Executive Contribution Accounting

All 4 technical contributors have executed their assigned responsibilities through auditable Git commits, modular code design, thorough unit tests, and rigorous scientific documentation.

| Contributor | Official Role | Verified Git Commits | Core Modules Authored | Test Coverage | Integration Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **Student 1** (Amogha Raj Sindur) | Data Owner (Dataset, Preprocessing, EDA) | 4 commits (`c0e142e` .. `52b0961`) | `src/data/download_data.py`, `src/data/preprocess.py`, `src/data/inspect_dataset.py`, `src/data/generate_eda_plots.py` | 30 tests (`test_inspect_dataset.py`, `test_preprocess.py`, `test_eda.py`, etc.) | **VERIFIED & INTEGRATED** |
| **Student 2** (Bulbul Agarwalla) | Feature Owner (Features & Scaffold Splitting) | 8 commits (`4b28acd` .. `1f0e9ba`) | `src/features/fingerprints.py`, `src/features/descriptors.py`, `src/features/build_features.py`, `src/data/split_data.py` | 23 tests (`test_fingerprints.py`, `test_split_integrity.py`, etc.) | **VERIFIED & INTEGRATED** |
| **Student 3** (Shourya Pratap) | Model Owner (Classical ML Models & Experiments) | 11 commits (`5a1ddb0` .. `b4d5bb4`) | `src/models/ml/` (LR, RF, XGBoost, factory), `src/evaluation/run_experiments.py`, `src/features/preprocessor.py` | 27 tests (`test_classical_models.py`, `test_preprocessor_leakage.py`, `test_reproducibility_suite.py`, etc.) | **VERIFIED & INTEGRATED** |
| **Student 4** (Kanishka Dubey) | Validation Owner (Evaluation, Metrics, Audit) | 9 commits (`8a39c9e` .. `fe6413f`) | `src/evaluation/metrics.py`, `src/evaluation/aggregation.py`, `src/evaluation/endpoint_analysis.py`, `src/evaluation/error_analysis.py` | 42 tests (`test_feature_validation.py`, `test_endpoint_analysis.py`, `test_error_analysis.py`, `test_final_evaluation_regression.py`) | **VERIFIED & INTEGRATED** |
| **Student 5** (Aditya Yadav) | Documentation Owner (Literature Review, Paper, Presentation) | — | `reports/literature_review.md`, `reports/viva_preparation.md`, README maintenance | — | **PENDING USER AUTHORING OF IEEE REPORT & PPT** |

---

## 2. Granular Student Contribution Audit

### 2.1 Student 1: Data Owner (Amogha Raj Sindur)
- **Primary Deliverables:**
  - Automated streaming download with SHA-256 validation (`45d09792...`) for `data/raw/tox21.csv.gz`.
  - Structure validation using RDKit sanitization: 7,831 raw records processed $\to$ 8 organometallic valence violations logged in `exclusions.csv` $\to$ 7,823 valid structures saved in `tox21_cleaned.csv`.
  - Multi-task duplicate and conflict detection: 0 canonical duplicate groups and 0 contradictory labels.
  - Generation of 10 publication-quality distribution and prevalence figures in `results/figures/eda/` with paired numerical JSON summaries.
- **Verification:** All 30 upstream data tests pass deterministically.

### 2.2 Student 2: Feature Owner (Bulbul Agarwalla)
- **Primary Deliverables:**
  - Morgan circular topological fingerprints ($r=2$, 2048-bit, binary `uint8`) via modern RDKit `rdFingerprintGenerator`.
  - 9 physicochemical molecular descriptors (MW, LogP, TPSA, HBD, HBA, rotatable bonds, ring count, heavy atom count, fraction $sp^3$) in `descriptors_9.npy`.
  - Combined feature matrix concatenation (2,057 dimensions, `float32`) with row alignment mapping in `feature_row_ids.json` and checksums in `feature_manifest.json`.
  - Deterministic Bemis-Murcko scaffold splitting allocating molecules into Train (80%), Val (10%), and Test (10%) partitions with 0 scaffold overlap.
- **Verification:** All 23 feature and split integrity tests pass.

### 2.3 Student 3: Model Owner (Shourya Pratap)
- **Primary Deliverables:**
  - Model architectures in `src/models/ml/`:
    - `logistic_regression.py`: L2 regularized ($C=1.0$), `class_weight='balanced'`, `lbfgs` solver.
    - `random_forest.py`: 100 trees, max depth 12, min samples split 5, `class_weight='balanced'`, multi-threaded (`n_jobs=-1`).
    - `xgboost_classifier.py`: 100 estimators, max depth 5, learning rate 0.1, dynamic `scale_pos_weight` per endpoint.
    - `__init__.py`: Factory `build_classifier()`.
  - Preprocessing leakage prevention via `LeakageSafeScaler` and `PartitionPreprocessor` (`src/features/preprocessor.py`), fitting strictly on training indices.
  - Experiment execution pipeline in `src/evaluation/run_experiments.py`, training all 12 endpoints and serializing 108 models to `results/models/`.
- **Verification:** 27 tests pass (including 5 dedicated model tests in `test_classical_models.py`).

### 2.4 Student 4: Validation Owner (Kanishka Dubey)
- **Primary Deliverables:**
  - Metric computation suite in `src/evaluation/metrics.py` (ROC-AUC, PR-AUC, F1, Balanced Accuracy with unlabelled NaN masking and single-class edge case safety).
  - Validation-only threshold tuning: optimizes decision boundaries on validation split strictly to maximize F1, freezing thresholds for test evaluation.
  - Macro uncertainty aggregation in `src/evaluation/aggregation.py` with explicit `Mean ± SEM (K=12)` and SD.
  - Error analysis in `src/evaluation/error_analysis.py` diagnosing endpoint difficulty and challenging chemotypes.
  - 3 evaluation figures in `results/figures/`.
- **Verification:** All 42 validation and evaluation tests pass.

---

## 3. Integration Integrity and Architecture Cohesion

1. **Non-Conflicting Modular Structure:** Each student's modules strictly interface via well-defined data artifacts and standardized Python APIs without circular imports:
   $$\text{Student 1 (Data)} \longrightarrow \text{Student 2 (Features/Splits)} \longrightarrow \text{Student 3 (Models/Experiments)} \longrightarrow \text{Student 4 (Evaluation/Audit)}$$
2. **Total Test Suite:** The full integration test suite comprises **122 automated tests**, all executing and passing in under 10 seconds (`pytest tests/ -v --run-real-data`).
3. **Repository State:** The default branch `main` is clean, synchronized with `origin/main`, and all pull requests have been properly merged without outstanding conflicts.
