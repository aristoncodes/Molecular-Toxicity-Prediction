# Student 3 Contribution Report: Classical ML Models and Experiments

**Name:** Shourya Pratap  
**Role:** Model Owner  
**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Scope:** Phase 1 — Baseline Model Implementations, Imbalance Handling, Preprocessing Leakage Prevention, and Experiment Execution  

---

## 1. Responsibilities

- Implement the classical machine learning baseline model architectures: Logistic Regression, Random Forest, and XGBoost.
- Train separate classifiers for all 12 Tox21 toxicity endpoints under multi-task missing-label conditions.
- Apply appropriate class-imbalance weighting strategies to prevent majority-class collapse.
- Conduct model and feature ablations across all 9 benchmark configurations (3 algorithms $\times$ 3 feature representations).
- Enforce strict test-set independence: prevent test labels from influencing feature scaling, model selection, or hyperparameter tuning.

---

## 2. Technical Deliverables

### Source Code (`src/models/ml/` & `src/features/` & `src/evaluation/`)
- `src/models/ml/logistic_regression.py`: L2-regularized Logistic Regression ($C=1.0$) with `class_weight='balanced'`, `lbfgs` solver, and `max_iter=1000`.
- `src/models/ml/random_forest.py`: Random Forest ensemble of 100 de-correlated decision trees with `max_depth=12`, `min_samples_split=5`, `class_weight='balanced'`, and multi-core parallelism (`n_jobs=-1`).
- `src/models/ml/xgboost_classifier.py`: XGBoost gradient boosted trees (100 estimators, `max_depth=5`, `learning_rate=0.1`) with dynamic per-endpoint `scale_pos_weight = N_neg / N_pos`.
- `src/models/ml/__init__.py`: Unified model factory `build_classifier()` supporting all three algorithms.
- `src/features/preprocessor.py`: Created `LeakageSafeScaler` and `PartitionPreprocessor` to eliminate data leakage by fitting standardization strictly on training indices and preserving binary Morgan bits without scaling.
- `src/evaluation/run_experiments.py`: Master experiment execution pipeline coordinating 12-endpoint model training, validation threshold optimization, test prediction generation, and model artifact serialization.

### Generated Artifacts
- `results/models/`: 108 serialized model artifacts ($9 \text{ configurations} \times 12 \text{ endpoints}$) saved as `.joblib` files under:
  - `results/models/logistic_regression_morgan/`, `..._descriptors/`, `..._combined/`
  - `results/models/random_forest_morgan/`, `..._descriptors/`, `..._combined/`
  - `results/models/xgboost_morgan/`, `..._descriptors/`, `..._combined/`
- `reports/classical_ml_models.md`: Technical documentation detailing baseline architectures, loss functions, and hyperparameters.

---

## 3. Key Findings & Scientific Measurements

- **Champion Baseline Model:** **Random Forest trained on Combined Features (2,057 dimensions)** achieved the highest test performance on unseen Bemis-Murcko scaffolds:
  - Test Macro ROC-AUC: **$0.7488 \pm 0.0181$ (SEM)** ($\text{SD} = 0.0628$)
  - Test Macro PR-AUC: **$0.3891 \pm 0.0320$ (SEM)**
  - Balanced Accuracy: **$0.6636$**, Macro F1: **$0.3781$**
- **Algorithm Comparison:**
  - Random Forest ($0.7488$ ROC-AUC) outperformed XGBoost ($0.7456$ ROC-AUC) and regularized Logistic Regression ($0.7138$ ROC-AUC).
  - Bagging ensembles proved more robust against the extreme class imbalance and sparsity of molecular fingerprints than second-order boosting.
- **Feature Ablation Impact:**
  - Combined (2057d) > Morgan (2048d) > Descriptors (9d).
  - Concatenating 9 continuous pharmacokinetic descriptors with 2,048 localized topological bits improved ROC-AUC by $+0.0280$ over Morgan bits alone.

---

## 4. Automated Unit & Integration Tests

Authored test modules in `tests/`:
- `tests/test_classical_models.py`: Validates model instantiation, `.fit()`, `.predict_proba()` output bounds, feature importances, factory dispatch, and joblib serialization.
- `tests/test_preprocessor_leakage.py`: Proves that `LeakageSafeScaler` standardizes test sets using train-derived $\mu$ and $\sigma$, confirms constant-feature safety, and verifies unscaled Morgan bit preservation.

**Test Execution:** 9 unit and regression tests passing deterministically (`pytest tests/test_classical_models.py tests/test_preprocessor_leakage.py -v`).
