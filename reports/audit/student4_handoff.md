# Student 4 Handoff Guide: Phase 2 Preparation

**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Handoff Date:** 2026-10-26  
**Auditor / QA Lead:** Senior ML Engineer  
**Recipient:** Student 4 (Baseline Modeling, GNN Architecture, and Benchmark Evaluation)  

---

## 1. Verified Existing Pipeline & Capabilities

Student 4 inherits a thoroughly tested, leak-free, mathematically verified Phase 1 pipeline:
- **Cleaned Dataset:** `data/processed/tox21_cleaned.csv` (7,823 valid unique canonical structures, 12 endpoints).
- **Extracted Feature Matrices (`data/processed/features/`):**
  - `morgan_r2_2048.npy`: Binary Morgan circular fingerprints (2,048 bits, radius 2, uint8).
  - `descriptors_9.npy`: 9 physicochemical molecular descriptors (float64).
  - `combined_features.npy`: Combined 2,057-dim matrix (float32).
  - `feature_row_ids.json`: Explicit 1-to-1 row alignment (`source_row`, `mol_id`).
- **Scaffold Splitting:** `data/processed/scaffold_split_indices.json`
  - Train: 6,258 molecules (~80%)
  - Validation: 782 molecules (~10%)
  - Test: 783 molecules (~10%)
  - Confirmed 0 non-empty scaffold overlap between partitions.
- **Leakage-Safe Preprocessor:** `src/features/preprocessor.py` (`PartitionPreprocessor`, `LeakageSafeScaler`).
- **Evaluation & Uncertainty:** `src/evaluation/metrics.py` and `src/evaluation/aggregation.py` (continuous ROC-AUC, PR-AUC with Average Precision, and $\text{Mean} \pm \text{SEM}$ aggregation).
- **Test Suite:** 85 automated tests in `tests/` passing cleanly.

---

## 2. Files and Modules to Extend (Do Not Re-Implement)

To prevent code duplication, architectural conflicts, and regression bugs, Student 4 must extend the established modules:

1. **Feature Loading & Scaling:**
   - **Do NOT** recalculate Morgan fingerprints or re-standardize the entire dataset.
   - **DO** load `data/processed/features/combined_features.npy` (or `morgan_r2_2048.npy`) and load split indices from `data/processed/scaffold_split_indices.json`.
   - **DO** use `PartitionPreprocessor.fit(X, train_indices=split["indices"]["train"])` and transform `val` and `test` splits out-of-sample.
2. **Evaluation Metrics:**
   - **Do NOT** write separate ROC-AUC / PR-AUC calculation scripts.
   - **DO** import `from src.evaluation.metrics import evaluate_multitask_predictions, evaluate_endpoint` and `from src.evaluation.aggregation import aggregate_endpoint_metrics`.
   - This ensures benchmark metrics are strictly comparable across all models and follow the exact same NaN-masking and macro-averaging protocols.
3. **Audit Suite:**
   - **DO** run `python -m src.audit.run_audit` to ensure no upstream data artifacts are corrupted.

---

## 3. Preserved Artifacts (Never Overwrite)

The following files represent verified Phase 1 outputs and must **never** be deleted or overwritten:
- `data/raw/tox21.csv.gz` and `data/raw/download_provenance.json`
- `data/processed/tox21_cleaned.csv` and `data/processed/dataset_manifest.json`
- `data/processed/features/*` (Morgan, descriptors, combined features, row IDs, manifest)
- `data/processed/scaffold_split_indices.json`
- `results/experiment_logs/data_quality_report.json`, `exclusions.csv`, `duplicates.csv`
- Existing test suites in `tests/`

---

## 4. Priority Implementation Scope for Student 4 (Phase 2)

Student 4 should focus their substantive contributions on the following sequence:

### Step 1: Classical Baseline Model Implementation (`src/models/` or `src/evaluation/run_experiments.py`)
Implement multi-task or per-endpoint binary classifiers on the pre-computed features:
1. **Logistic Regression (L2 / ElasticNet regularized):**
   - Imbalance handling: `class_weight='balanced'`
   - Trained on Morgan fingerprints, 9 descriptors, and combined features.
2. **Random Forest Classifier:**
   - Tuned on validation split (`n_estimators`, `max_depth`, `min_samples_split`).
3. **XGBoost / LightGBM:**
   - Gradient boosted decision trees with `scale_pos_weight` tuned to handle Tox21 endpoint imbalance (prevalences 2.8% to 16.2%).

### Step 2: Validation-Based Model Selection & Threshold Tuning
- **Strict Rule:** Tune hyperparameters and decision thresholds *strictly* on the validation partition (`indices["val"]`).
- **Never** tune thresholds or select best models based on test-set metrics.
- Record validation curves and hyperparameter search grids in `results/experiment_logs/model_tuning.json`.

### Step 3: Out-of-Sample Test Evaluation & Artifact Logging
- Evaluate the best-validated models on the untouched test partition (`indices["test"]`).
- Save continuous probability predictions to `results/experiment_logs/predictions_<model_name>.csv`.
- Save aggregated macro metrics using `aggregate_endpoint_metrics(...)` to `results/experiment_logs/metrics_summary.json`.

### Step 4: Molecular Graph Neural Network (GNN) Exploration
- Represent molecules as molecular graphs ($G = (V, E)$) using atom features (atomic number, hybridization, formal charge, aromaticity) and bond features (bond type, stereochemistry).
- Implement Graph Convolutional Networks (GCN) or Graph Attention Networks (GAT) using PyTorch Geometric or DGL.
- Compare GNN test performance against the classical Morgan + Combined Random Forest / XGBoost baselines under identical scaffold splits.

---

## 5. Verification Checklist for Student 4

Before submitting any Phase 2 deliverable, verify:
- [ ] All 85 existing tests continue to pass (`pytest tests/ -v --run-real-data`).
- [ ] No descriptor scaling statistics leaked from validation or test partitions into training.
- [ ] Thresholds were chosen strictly using validation data.
- [ ] Test set was evaluated only once per final model configuration.
- [ ] Reported macro metrics explicitly specify $\text{Mean} \pm \text{SEM}$ ($K=12$) and $\text{SD}$.
- [ ] Prediction CSV files are saved with corresponding row identifiers for external auditability.
