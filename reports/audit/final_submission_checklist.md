# Phase 1 Final Submission Checklist

**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Course:** CSAI Student Project — Phase 1 Machine Learning  
**Audit Date:** 2026-10-27  
**Status:** ✅ READY FOR FINAL DOCUMENTATION AND SUBMISSION

---

## 1. Data & Preprocessing

| Requirement | Student | Status | Evidence |
| :--- | :---: | :---: | :--- |
| Tox21 dataset acquired with SHA-256 checksum verification | Student 1 | ✅ DONE | `data/raw/download_provenance.json` — SHA-256 `45d09792...` |
| Invalid structures identified and excluded with audit trail | Student 1 | ✅ DONE | `results/experiment_logs/exclusions.csv` — 8 organo-aluminum molecules |
| Duplicate and label-conflict audit with documented policy | Student 1 | ✅ DONE | `results/experiment_logs/duplicates.csv` — 0 conflicts found |
| Cleaned dataset with canonical SMILES saved | Student 1 | ✅ DONE | `data/processed/tox21_cleaned.csv` — 7,823 records |

---

## 2. Exploratory Data Analysis

| Requirement | Student | Status | Evidence |
| :--- | :---: | :---: | :--- |
| Missing label visualization per endpoint | Student 1 | ✅ DONE | `results/figures/eda/02_missing_labels.png` |
| Positive prevalence per endpoint (class imbalance analysis) | Student 1 | ✅ DONE | `results/figures/eda/04_positive_prevalence.png` |
| Molecular weight distribution | Student 1 | ✅ DONE | `results/figures/eda/06_molecular_weight.png` |
| Atom count and heavy atom distributions | Student 1 | ✅ DONE | `results/figures/eda/05_atom_count.png`, `07_heavy_atom_count.png` |
| Ring count and rotatable bond distributions | Student 1 | ✅ DONE | `results/figures/eda/08_ring_count.png`, `09_rotatable_bond_count.png` |
| Duplicate and conflict summary figure | Student 1 | ✅ DONE | `results/figures/eda/10_duplicates_conflicts.png` |
| All figures have numerical JSON backing statistics | Student 1 | ✅ DONE | 10 × `.json` files in `results/figures/eda/` |

---

## 3. Feature Extraction

| Requirement | Student | Status | Evidence |
| :--- | :---: | :---: | :--- |
| Morgan fingerprints (radius=2, 2048 bits) generated | Student 2 | ✅ DONE | `data/processed/features/morgan_r2_2048.npy` — shape (7823, 2048) |
| 9 physicochemical molecular descriptors computed | Student 2 | ✅ DONE | `data/processed/features/descriptors_9.npy` — shape (7823, 9) |
| Combined feature matrix (2057-dim) with row alignment | Student 2 | ✅ DONE | `data/processed/features/combined_features.npy` — shape (7823, 2057) |
| Feature manifest with SHA-256 checksums | Student 2 | ✅ DONE | `data/processed/features/feature_manifest.json` |

---

## 4. Scaffold Splitting and Leakage Prevention

| Requirement | Student | Status | Evidence |
| :--- | :---: | :---: | :--- |
| Bemis-Murcko scaffold split (80/10/10, seed=42) | Student 2 | ✅ DONE | `data/processed/scaffold_split_indices.json` |
| Zero scaffold overlap across train/val/test | Student 2 | ✅ DONE | Verified by `test_split_integrity.py` — exact 0 shared scaffolds |
| Acyclic molecules (1,775) confined to training set | Student 2 | ✅ DONE | `single_group` policy in `src/data/split_data.py` |
| Preprocessing scalers fit only on training partition | Student 3 | ✅ DONE | `src/features/preprocessor.py` — `LeakageSafeScaler` |
| Morgan bits excluded from scaling (binary preservation) | Student 3 | ✅ DONE | `PartitionPreprocessor` explicitly skips bit columns |

---

## 5. Baseline Machine Learning Models

| Requirement | Student | Status | Evidence |
| :--- | :---: | :---: | :--- |
| Logistic Regression trained across 3 feature sets | Student 4 | ✅ DONE | Predictions in `results/metrics/predictions_logistic_regression_*.csv` |
| Random Forest trained across 3 feature sets | Student 4 | ✅ DONE | Predictions in `results/metrics/predictions_random_forest_*.csv` |
| XGBoost trained across 3 feature sets | Student 4 | ✅ DONE | Predictions in `results/metrics/predictions_xgboost_*.csv` |
| Multi-task label missingness handled correctly | Student 4 | ✅ DONE | `src/evaluation/model_evaluation.py` — NaN masking per endpoint |
| Validation-only threshold tuning applied | Student 4 | ✅ DONE | Threshold grid on val, applied frozen to test |
| Full benchmark comparison table saved | Student 4 | ✅ DONE | `results/metrics/model_comparison_table.csv` |

---

## 6. Evaluation Metrics and Uncertainty

| Requirement | Student | Status | Evidence |
| :--- | :---: | :---: | :--- |
| Per-endpoint ROC-AUC with NaN masking | Student 3 | ✅ DONE | `src/evaluation/metrics.py` — `compute_roc_auc` |
| Per-endpoint PR-AUC (Average Precision) | Student 3 | ✅ DONE | `src/evaluation/metrics.py` — `compute_pr_auc` |
| Single-class endpoint safety (no division-by-zero) | Student 3 | ✅ DONE | Returns NaN with warning on single-class endpoints |
| Macro-averaging with explicit SEM and SD | Student 3 | ✅ DONE | `src/evaluation/aggregation.py` — `aggregate_endpoint_metrics` |
| Threshold-based Balanced Accuracy and F1-score | Student 4 | ✅ DONE | `src/evaluation/model_evaluation.py` |
| Error analysis and challenging molecule identification | Student 4 | ✅ DONE | `results/metrics/endpoint_error_analysis.json` |
| 3 evaluation figures produced | Student 4 | ✅ DONE | `results/figures/{endpoint_performance_roc_pr,model_feature_comparison_bars,prevalence_vs_performance}.png` |

---

## 7. Testing and Reproducibility

| Requirement | Student | Status | Evidence |
| :--- | :---: | :---: | :--- |
| Full test suite passing (117 tests) | All | ✅ DONE | `pytest tests/ -v --run-real-data` — 117 passed, 0 failed |
| Independent scientific audit suite | Student 3 | ✅ DONE | `python -m src.audit.run_audit` — all 7 checks pass |
| Single-command feature reproduction | Student 2 | ✅ DONE | `python -m src.features.build_features` |
| Pinned requirements.txt (29 packages) | QA | ✅ DONE | `requirements.txt` — includes `scikit-learn`, `xgboost`, `scipy`, `seaborn`, `joblib` |

---

## 8. Documentation and Academic Reports

| Requirement | Student | Status | Evidence |
| :--- | :---: | :---: | :--- |
| Data provenance and directory guide | QA | ✅ DONE | `data/README.md` |
| Comprehensive Literature Review (QSAR, fingerprints, GNNs) | QA | ✅ DONE | `reports/literature_review.md` — 15 key citations |
| Viva / Technical Defense Preparation Guide | QA | ✅ DONE | `reports/viva_preparation.md` — 13 answered exam questions |
| Phase 1 model evaluation technical report | Student 4 | ✅ DONE | `reports/phase1_model_evaluation.md` |
| Feature extraction and scaffold splitting report | Student 2 | ✅ DONE | `reports/student2_features_and_splitting.md` |
| Dataset EDA research report | Student 1 | ✅ DONE | `reports/dataset_eda_report.md` |
| Multi-contributor README | All | ✅ DONE | `README.md` — includes all 4 student sections |

---

## 9. Outstanding Items (User Action Required)

> [!IMPORTANT]
> The following are explicitly **reserved for manual completion by the team**. No agent-generated content will be produced for these items.

| Item | Owner | Notes |
| :--- | :---: | :--- |
| **IEEE Conference Manuscript (PDF)** | User / Team | Template `reports/ieee_phase1_manuscript.tex` is Overleaf-ready with all measured results. |
| **Oral Defense Presentation (PPTX / Slides)** | User / Team | All figures, benchmark numbers, and talking points are in `reports/viva_preparation.md`. |

---

## 10. Overall Phase 1 Readiness

**Status:** ✅ **PASS — ALL TECHNICAL REQUIREMENTS FULFILLED**  
All 9 auditable CSAI Phase 1 criteria are satisfied with verifiable evidence. Zero blocking defects remain.
