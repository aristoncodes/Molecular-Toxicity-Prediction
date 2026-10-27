# Project Requirements-to-Evidence Matrix: Project 8

**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Specification:** Authoritative Project 8 Checklist (Phase 1 Classical ML & Phase 2 Deep Learning)  
**Auditor:** Senior ML Researcher, Scientific Auditor & Quality-Assurance Lead  
**Audit Date:** 2026-10-27  

---

## 1. Compliance Status Summary

| Phase | Total Requirements | PASS | PARTIAL | FAIL / NOT IMPLEMENTED | BLOCKED |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 1: Classical Machine Learning** | 10 | **10** | 0 | 0 | 0 |
| **Phase 2: Graph Neural Networks** | 10 | 0 | 0 | **10** | 0 |
| **Final Scientific Comparison Objective** | 1 | 0 | 0 | **1** | 0 |

---

## 2. Phase 1 — Classical Machine Learning Baseline

| Req ID | Exact Requirement | Implementation Files & Functions | Artifact Paths | Verification Command & Result | Status | Severity if Failed | Corrective Action Taken |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **P1-01** | Load Tox21 dataset from DeepChem/MoleculeNet source | `src/data/download_data.py`: `download_tox21_dataset()` | `data/raw/tox21.csv.gz`, `data/raw/download_provenance.json` | Re-verified SHA-256 matches `45d09792...` (445,471 bytes). Checked by `src/audit/dataset_audit.py`. | **PASS** | Critical | None needed. Provenance verified. |
| **P1-02** | Validate and canonicalize SMILES strings | `src/data/preprocess.py`: `clean_dataset()`, `src/data/inspect_dataset.py` | `data/processed/tox21_cleaned.csv`, `results/experiment_logs/exclusions.csv` | 7,831 raw records parsed $\to$ exactly 8 organometallic exclusions documented $\to$ 7,823 valid structures with canonical isomeric SMILES (100% validity in RDKit). | **PASS** | Critical | None needed. Sanitization verified. |
| **P1-03** | Identify and handle duplicate compounds before splitting | `src/data/preprocess.py`: `audit_duplicates()` | `results/experiment_logs/duplicates.csv` | Confirmed 0 duplicate canonical SMILES groups and 0 conflicting labels in cleaned dataset. | **PASS** | High | Audited and verified. |
| **P1-04** | Preserve missing toxicity labels using explicit masks | `src/evaluation/metrics.py`: `mask_missing_labels()`, `src/evaluation/model_evaluation.py` | `data/processed/tox21_cleaned.csv` | Missing values stored as `NaN`, masked per-endpoint during training and evaluation. Never zero-imputed. | **PASS** | Critical | Verified by `test_metrics.py::test_missing_labels_not_treated_as_negative`. |
| **P1-05** | Generate Morgan fingerprints and basic molecular descriptors | `src/features/fingerprints.py`: `compute_morgan_fingerprints()`, `src/features/descriptors.py`: `compute_descriptors()` | `data/processed/features/morgan_r2_2048.npy` (7823, 2048), `descriptors_9.npy` (7823, 9), `combined_features.npy` (7823, 2057) | Verified 2048-bit radius=2 binary bits, 9 standard descriptors bounded and finite, combined matrix aligned. | **PASS** | High | Verified by `test_feature_validation.py` (all tests pass). |
| **P1-06** | Train Logistic Regression, Random Forest, and XGBoost models | `src/models/ml/` (`logistic_regression.py`, `random_forest.py`, `xgboost_classifier.py`), `src/evaluation/run_experiments.py` | `results/models/` (108 `.joblib` files), `results/metrics/predictions_*.csv` | All 9 model $\times$ feature configurations trained across 12 endpoints. Tested via `test_classical_models.py`. | **PASS** | High | Implemented modular model classes and runner; serialized model artifacts. |
| **P1-07** | Handle class imbalance appropriately | `src/models/ml/`: `class_weight='balanced'` in LR/RF; dynamic `scale_pos_weight` in XGBoost | `src/evaluation/model_evaluation.py`, `src/evaluation/run_experiments.py` | Dynamic pos/neg weight ratio calculated per-endpoint from training slice only. | **PASS** | High | Implemented and verified across extreme endpoints (down to 2.84% positive). |
| **P1-08** | Scaffold-based train/validation/test splits | `src/data/split_data.py`: `scaffold_split()` | `data/processed/scaffold_split_indices.json` | Bemis-Murcko splitting (seed 42, 80/10/10): Train 6,258 (79.995%), Val 782 (9.996%), Test 783 (10.009%). Exact zero scaffold overlap. Acyclic molecules isolated to train. | **PASS** | Critical | Verified by `test_split_integrity.py` and `src/audit/split_audit.py`. |
| **P1-09** | Evaluate endpoint-wise ROC-AUC and PR-AUC excluding missing labels | `src/evaluation/metrics.py`: `compute_roc_auc()`, `compute_pr_auc()` | `results/metrics/model_benchmark_summary.json` | Continuous prediction ranking; unlabelled samples masked per endpoint; single-class edge cases handled safely. | **PASS** | High | Verified by `test_metrics.py`. |
| **P1-10** | Establish reproducible baseline results for comparison | `src/evaluation/run_experiments.py`, `src/audit/run_audit.py` | `results/metrics/model_comparison_table.csv` | Full benchmark table produced. Champion: Random Forest + Combined (ROC-AUC $0.7488 \pm 0.0181$). 122 tests pass. | **PASS** | High | Pinned requirements; reproducible execution. |

---

## 3. Phase 2 — Deep Learning (Graph Neural Networks)

| Req ID | Exact Requirement | Implementation Files & Functions | Artifact Paths | Verification Command & Result | Status | Severity if Failed | Recommended Corrective Action |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **P2-01** | Represent each molecule as a graph | None | None | Grepped codebase: 0 PyTorch Geometric, DGL, or graph representation modules exist. | **NOT IMPLEMENTED** | High | Implement molecular featurizer converting RDKit Mol objects to graph node/edge tensors. |
| **P2-02** | Represent atoms as nodes and bonds as edges | None | None | No atom-level or bond-level feature extraction scripts exist. | **NOT IMPLEMENTED** | High | Build atomic feature vector (atomic number, hybridization, formal charge) and bond features (order, stereochemistry). |
| **P2-03** | Implement GCN or MPNN architecture | None | None | No PyTorch graph neural network layers or message-passing classes exist. | **NOT IMPLEMENTED** | High | Implement GCN / GAT / GIN model using PyTorch Geometric or custom message passing. |
| **P2-04** | Produce predictions for the 12 toxicity endpoints | None | None | No multi-task graph classifier head exists. | **NOT IMPLEMENTED** | High | Build 12-dimensional linear prediction head with sigmoid outputs. |
| **P2-05** | Masked loss so missing labels do not contribute to gradient | None | None | No masked binary cross-entropy loss module exists. | **NOT IMPLEMENTED** | High | Implement `MaskedBCEWithLogitsLoss(reduction='none') * mask`. |
| **P2-06** | Handle class imbalance in graph models | None | None | No pos_weight or focal loss implementation exists. | **NOT IMPLEMENTED** | Medium | Pass endpoint-specific `pos_weight` into masked loss. |
| **P2-07** | Train/validate using comparable scaffold split | None | None | Split indices exist (`scaffold_split_indices.json`), but no PyG DataLoader consumes them. | **NOT IMPLEMENTED** | High | Map existing scaffold split indices to graph dataset subsets. |
| **P2-08** | Validation-based model selection & early stopping | None | None | No training loop or checkpoint selection exists. | **NOT IMPLEMENTED** | High | Implement training loop with validation ROC-AUC early stopping. |
| **P2-09** | Evaluate using identical metrics and test protocol | None | None | Metric functions exist (`src/evaluation/metrics.py`), but no GNN predictions exist to evaluate. | **NOT IMPLEMENTED** | High | Connect GNN test output to `evaluate_multitask_predictions()`. |
| **P2-10** | Fair comparison against fingerprint baselines | None | None | Cannot compare without Phase 2 model execution. | **NOT IMPLEMENTED** | Critical | Evaluate GNN on the identical 783 test molecules and compare directly with RF ($0.7488$). |

---

## 4. Final Scientific Objective

| Req ID | Exact Requirement | Current Status | Scientific Assessment & Gap |
| :---: | :--- | :---: | :--- |
| **OBJ-01** | Endpoint-wise comparison of fingerprint-based AML and graph-based DL models, including scaffold generalization analysis | **INCOMPLETE (Phase 2 Pending)** | Phase 1 AML baselines are fully established and benchmarked. Phase 2 GNN models are not yet implemented. Therefore, the comparative empirical hypothesis test ("do learned molecular graphs outperform engineered fingerprints?") cannot yet be answered from repository evidence. |
