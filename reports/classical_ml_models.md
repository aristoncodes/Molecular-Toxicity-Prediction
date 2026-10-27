# Technical Report: Classical Machine Learning Models & Baseline Benchmarks

**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Scope:** Architecture, Hyperparameters, Class Imbalance Handling, and Model Serialization  
**Status:** Complete, Verified & Reproducible  

---

## 1. Overview

This technical report documents the design, mathematical formulation, and implementation of the classical machine learning baseline models evaluated for multi-task molecular toxicity prediction across the 12 NIH Tox21 assay endpoints.

The classical baseline suite comprises three distinct algorithm families:
1. **Regularized Linear Models:** L2-regularized Logistic Regression
2. **Bagging Ensembles:** Random Forest (100 de-correlated decision trees)
3. **Gradient Boosting:** XGBoost (second-order gradient boosted decision trees)

All models are trained per endpoint with dedicated class-imbalance weighting, strict feature scaling isolation, and validation-only threshold tuning to establish benchmark performance before deploying Phase 2 Molecular Graph Neural Networks (GNNs).

---

## 2. Model Architectures & Design Choices

### 2.1 Logistic Regression (`src/models/ml/logistic_regression.py`)
- **Objective:** Provides a convex, linear decision boundary baseline to determine whether toxicological endpoints are linearly separable in chemical representation spaces.
- **Formulation:** Minimizes the negative log-likelihood with an L2 weight penalty:
  $$\min_{\mathbf{w}, b} \sum_{i=1}^{N_j} w_i \log\left(1 + e^{-y_i (\mathbf{w}^T \mathbf{x}_i + b)}\right) + \frac{1}{2C} \|\mathbf{w}\|_2^2$$
- **Hyperparameters:**
  - Regularization: L2 penalty ($C=1.0$) to stabilize high-dimensional Morgan bit spaces ($d=2048$).
  - Solver: `lbfgs` with `max_iter=1000`.
  - Class Weighting: `class_weight='balanced'` to scale loss inversely proportional to endpoint positive prevalence:
    $$w_c = \frac{N}{2 \cdot N_c}$$

### 2.2 Random Forest (`src/models/ml/random_forest.py`)
- **Objective:** An ensemble of de-correlated decision trees built via bootstrap aggregation (bagging) and random subspace feature selection.
- **Formulation:** Evaluates non-linear feature interactions without requiring feature scaling:
  $$\hat{y}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$
- **Hyperparameters:**
  - `n_estimators`: 100 decision trees.
  - `max_depth`: 12 (caps depth to prevent memorization of sparse scaffold clusters).
  - `min_samples_split`: 5 (prevents splits on isolated outlier molecules).
  - `class_weight`: `'balanced'` (re-weights classes at each bootstrap sample).
  - `n_jobs`: -1 (parallel tree construction across all CPU cores).

### 2.3 XGBoost (`src/models/ml/xgboost_classifier.py`)
- **Objective:** Additive tree boosting minimizing a second-order Taylor approximation of the loss function, focusing boosting iterations on difficult boundary samples.
- **Formulation:**
  $$\mathcal{L}^{(t)} = \sum_{i=1}^n \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
- **Hyperparameters:**
  - `n_estimators`: 100 boosting rounds.
  - `max_depth`: 5 (shallow trees to prevent overfitting in boosting).
  - `learning_rate`: 0.1 (shrinkage parameter).
  - `scale_pos_weight`: Dynamic per-endpoint weight ratio:
    $$\text{scale\_pos\_weight}_j = \frac{N_{j, \text{negative}}}{N_{j, \text{positive}}}$$
  - `eval_metric`: `'logloss'`.

---

## 3. Class Imbalance & Missing-Label Handling

Tox21 exhibits severe assay characteristics that require strict handling:
1. **Missing Label Masking:** Approximately 17% of endpoint measurements are unobserved. Samples with $\text{NaN}$ labels are masked during training and evaluation:
   $$\mathcal{L} = \sum_{j=1}^{12} \sum_{i \in \mathcal{S}_j} \ell(y_{ij}, \hat{y}_{ij}), \quad \mathcal{S}_j = \{i \mid y_{ij} \in \{0, 1\}\}$$
   Missing labels are **never imputed as negative (0.0)**.
2. **Extreme Class Imbalance:** Active prevalence ranges from $2.84\%$ (`SR-HSE`) to $15.82\%$ (`NR-AhR`). Standard accuracy is scientifically uninformative; models utilize balanced class weighting and dynamic loss re-weighting.

---

## 4. Benchmark Performance Summary (Test Set, Out-of-Sample)

| Model Architecture | Feature Representation | Dimensions | Macro ROC-AUC ($\text{Mean} \pm \text{SEM}$) | ROC-AUC SD | Macro PR-AUC ($\text{Mean} \pm \text{SEM}$) | Balanced Acc | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **Combined Features** | **2,057** | **$0.7488 \pm 0.0181$** | **0.0628** | **$0.3891 \pm 0.0320$** | **0.6636** | **0.3781** |
| **XGBoost** | Combined Features | 2,057 | $0.7456 \pm 0.0179$ | 0.0620 | $0.3495 \pm 0.0406$ | 0.6485 | 0.3591 |
| **Random Forest** | 9 Descriptors | 9 | $0.7215 \pm 0.0216$ | 0.0747 | $0.2901 \pm 0.0408$ | 0.6337 | 0.3036 |
| **Random Forest** | Morgan Fingerprints | 2,048 | $0.7208 \pm 0.0199$ | 0.0688 | $0.3458 \pm 0.0316$ | 0.6475 | 0.3712 |
| **Logistic Regression** | 9 Descriptors | 9 | $0.7203 \pm 0.0169$ | 0.0585 | $0.1968 \pm 0.0397$ | 0.6188 | 0.2472 |
| **Logistic Regression** | Combined Features | 2,057 | $0.7138 \pm 0.0158$ | 0.0546 | $0.2341 \pm 0.0338$ | 0.6404 | 0.3159 |
| **XGBoost** | Morgan Fingerprints | 2,048 | $0.7114 \pm 0.0181$ | 0.0627 | $0.3027 \pm 0.0308$ | 0.6340 | 0.3351 |
| **XGBoost** | 9 Descriptors | 9 | $0.7024 \pm 0.0256$ | 0.0887 | $0.2911 \pm 0.0454$ | 0.6394 | 0.3133 |
| **Logistic Regression** | Morgan Fingerprints | 2,048 | $0.6987 \pm 0.0149$ | 0.0515 | $0.2602 \pm 0.0341$ | 0.6216 | 0.3001 |

---

## 5. Artifact Serialization & Reproducibility

- **Execution Script:** `python -m src.evaluation.run_experiments`
- **Serialized Model Artifacts:** 108 trained `.joblib` model files stored in `results/models/`:
  - `results/models/<model>_<features>/<endpoint>.joblib`
- **Predictions & Metrics:** 9 test set prediction CSV files in `results/metrics/` and aggregated summary in `results/metrics/model_benchmark_summary.json`.
