# Repository Folder Structure and Organization Audit

**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Auditor:** Senior ML Engineer & Quality-Assurance Lead  
**Audit Date:** 2026-10-27  

---

## 1. Verified Repository Directory Layout

```
aml-dl/                                         # Repository root
├── .gitignore                                  # Excludes .venv/, __pycache__/, raw data binaries
├── README.md                                   # Comprehensive multi-student project documentation
├── requirements.txt                            # Fully pinned Python environment (29 packages)
├── configs/
│   └── data_config.yaml                        # Centralized configuration (paths, SHA-256, seed, flags)
├── data/
│   ├── README.md                               # Data provenance, checksums, and reproduction guide
│   ├── raw/
│   │   ├── tox21.csv.gz                        # [gitignored] Raw compressed dataset (445,471 bytes)
│   │   └── download_provenance.json            # [gitignored] Source URL, timestamp, SHA-256
│   └── processed/
│       ├── tox21_cleaned.csv                   # [gitignored] 7,823 valid structures with canonical SMILES
│       ├── dataset_manifest.json               # [gitignored] Processing manifest and checksums
│       ├── scaffold_split_indices.json         # [gitignored] Train/val/test Bemis-Murcko split indices
│       └── features/
│           ├── morgan_r2_2048.npy              # [gitignored] Binary fingerprints (7823, 2048)
│           ├── descriptors_9.npy               # [gitignored] Physicochemical descriptors (7823, 9)
│           ├── combined_features.npy           # [gitignored] Combined matrix (7823, 2057)
│           ├── feature_row_ids.json            # [gitignored] Row alignment mapping
│           └── feature_manifest.json           # [gitignored] Feature generation manifest and hashes
├── results/
│   ├── experiment_logs/
│   │   ├── audit_results.json                  # [gitignored] Output of python -m src.audit.run_audit
│   │   ├── data_quality_report.json            # [gitignored] EDA statistics and schema report
│   │   ├── exclusions.csv                      # [gitignored] 8 excluded organometallic structures
│   │   ├── duplicates.csv                      # [gitignored] Duplicate group analysis
│   │   └── structure_properties.csv            # [gitignored] Per-molecule molecular property measurements
│   ├── figures/
│   │   ├── eda/                                # [gitignored] 10 × {.png, .pdf, .json} EDA figures
│   │   ├── endpoint_performance_roc_pr.png     # [gitignored] Student 4 per-endpoint performance
│   │   ├── model_feature_comparison_bars.png   # [gitignored] 9-model benchmark comparison bars
│   │   └── prevalence_vs_performance.png       # [gitignored] Prevalence vs. PR-AUC scatter
│   └── metrics/
│       ├── model_benchmark_summary.json        # [gitignored] Full per-endpoint benchmark results
│       ├── model_comparison_table.csv          # [gitignored] Condensed 9-model comparison table
│       ├── endpoint_label_diagnostics.json     # [gitignored] Per-endpoint label count diagnostics
│       ├── endpoint_error_analysis.json        # [gitignored] Error analysis by endpoint
│       └── predictions_*.csv                   # [gitignored] 9 test prediction CSVs (one per model×features)
├── src/
│   ├── __init__.py
│   ├── audit/
│   │   ├── __init__.py
│   │   ├── dataset_audit.py                    # Student 3: Independent dataset provenance verifier
│   │   ├── split_audit.py                      # Student 3: Independent scaffold isolation verifier
│   │   └── run_audit.py                        # Student 3: Unified CLI audit runner (7 checks)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_data.py                    # Student 1: Streaming download with SHA-256 gating
│   │   ├── generate_eda_plots.py               # Student 1: 10 EDA figures with numerical JSON
│   │   ├── inspect_dataset.py                  # Student 1: Schema/label/duplicate auditor
│   │   ├── preprocess.py                       # Student 1: RDKit sanitization and canonical SMILES
│   │   └── split_data.py                       # Student 2: Bemis-Murcko scaffold splitting
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── aggregation.py                      # Student 3: Macro averaging with SEM/SD uncertainty
│   │   ├── endpoint_analysis.py                # Student 4: Per-endpoint label diagnostics
│   │   ├── error_analysis.py                   # Student 4: Challenging molecule identification
│   │   ├── feature_validation.py               # Student 4: Feature matrix QA and validation
│   │   ├── metrics.py                          # Student 3: ROC-AUC, PR-AUC, threshold, NaN masking
│   │   └── model_evaluation.py                 # Student 4: Multi-task LR/RF/XGBoost benchmark
│   └── features/
│       ├── __init__.py
│       ├── build_features.py                   # Student 2: Combined feature matrix assembly
│       ├── descriptors.py                      # Student 2: 9 physicochemical descriptor calculator
│       ├── fingerprints.py                     # Student 2: Morgan/ECFP4 fingerprint generator
│       └── preprocessor.py                     # Student 3: Leakage-safe scaler and preprocessor
├── tests/
│   ├── conftest.py
│   ├── test_eda.py                             # Student 1 EDA plot generation tests
│   ├── test_endpoint_analysis.py               # Student 4 endpoint diagnostics tests
│   ├── test_error_analysis.py                  # Student 4 error analysis tests
│   ├── test_feature_validation.py              # Student 4 feature validation tests
│   ├── test_final_evaluation_regression.py     # Student 4 evaluation regression tests
│   ├── test_fingerprints.py                    # Student 2 fingerprint generation tests
│   ├── test_inspect_dataset.py                 # Student 1 dataset inspection tests
│   ├── test_metrics.py                         # Student 3 metric computation tests
│   ├── test_model_evaluation.py                # Student 4 model training and evaluation tests
│   ├── test_preprocess.py                      # Student 1 preprocessing tests
│   ├── test_preprocessor_leakage.py            # Student 3 leakage-safe scaler tests
│   ├── test_reproducibility_suite.py           # Student 3 reproducibility audit tests
│   └── test_split_integrity.py                 # Student 2 scaffold split integrity tests
└── reports/
    ├── dataset_eda_report.md                   # Student 1: Generated EDA measurement report
    ├── student2_features_and_splitting.md      # Student 2: Feature extraction technical report
    ├── student2_integration_audit.md           # Student 2: Upstream (Student 1) audit report
    ├── student3_phase1_audit.md               # Student 3: Independent Phase 1 verification report
    ├── phase1_model_evaluation.md              # Student 4: Benchmark evaluation technical report
    ├── ieee_phase1_manuscript.tex              # Student 3: IEEEtran LaTeX manuscript (Overleaf-ready)
    ├── ieee_reconciliation.md                  # Student 3: Claim reconciliation matrix
    ├── literature_review.md                    # QA: Comprehensive QSAR/fingerprints/GNN lit review
    ├── viva_preparation.md                     # QA: Technical viva Q&A defense guide
    └── audit/
        ├── issue_register.md                   # Living issue register (6 issues, all resolved or monitored)
        ├── phase1_requirements_audit.md        # CSAI rubric compliance audit (9/9 PASS)
        ├── scientific_validation_audit.md      # Dataset, features, splits, metrics deep-audit
        ├── contribution_and_integration_audit.md # Students 1-4 verified contribution summary
        ├── folder_structure_audit.md           # (This document)
        ├── final_submission_checklist.md       # Phase 1 submission checklist
        ├── final_readiness_report.md           # Executive readiness summary
        ├── phase1_scientific_audit.md          # Student 3 original scientific audit
        ├── student4_handoff.md                 # Student 3 handoff document for Student 4
        └── student_contribution_matrix.md      # Original contribution tracking matrix
```

---

## 2. Gitignore Conventions Verified

Audited `.gitignore` to confirm:
- Raw data binaries (`data/raw/`, `data/processed/`) are correctly excluded — no large files committed.
- Python build artifacts (`__pycache__/`, `*.pyc`, `.venv/`) excluded as per standard Python practice.
- Experiment logs and generated figures excluded; regenerated from source code.
- All source code, test suite, documentation, and configuration files are **tracked** in version control.

---

## 3. Justified Structural Changes Made in This Audit

| Change | Justification | Status |
| :--- | :--- | :---: |
| Created `data/README.md` | Documents data directory layout, checksums, and reproduction commands. Closes a documentation gap. | **COMPLETED** |
| Added `reports/literature_review.md` | Required academic documentation per CSAI rubric for Phase 1 submission. | **COMPLETED** |
| Added `reports/viva_preparation.md` | Required oral defense preparation guide per CSAI rubric criteria. | **COMPLETED** |
| Added `reports/audit/phase1_requirements_audit.md` | Formal rubric compliance audit mapping. | **COMPLETED** |
| Added `reports/audit/scientific_validation_audit.md` | Deep-dive scientific verification. | **COMPLETED** |
| Added `reports/audit/contribution_and_integration_audit.md` | Verified student contribution accounting. | **COMPLETED** |
| Added `reports/audit/folder_structure_audit.md` | (This document) | **COMPLETED** |
| Added `reports/audit/final_submission_checklist.md` | Phase 1 submission checklist. | **COMPLETED** |
| Added `reports/audit/final_readiness_report.md` | Executive readiness declaration. | **COMPLETED** |
| Updated `requirements.txt` | Added 5 missing ML dependencies: `scikit-learn`, `xgboost`, `scipy`, `seaborn`, `joblib`. | **COMPLETED** |
