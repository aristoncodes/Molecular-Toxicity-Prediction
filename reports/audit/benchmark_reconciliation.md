# Benchmark Reconciliation & Empirical Metric Verification

**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Scope:** Recomputation of test predictions, resolution of historical draft discrepancies, and determination of authoritative Phase 1 baselines.  
**Auditor:** Senior ML Engineer, Scientific Auditor & Software Quality Lead  
**Audit Date:** 2026-10-27  

---

## 1. Executive Summary

This document reconciles all benchmark metrics reported in previous project documents, outlines the exact mathematical calculation of each value, and provides verified empirical evidence from the 9 saved test prediction CSV files and model summary artifacts.

### Authoritative Benchmark Status: **FULLY RECONCILED & RESOLVED**
- All 9 model configurations (3 algorithms $\times$ 3 feature representations) have been evaluated out-of-sample on unseen Bemis-Murcko scaffolds (Test partition, $N=783$).
- Predictions are saved as per-molecule continuous probabilities across all 12 Tox21 toxicity endpoints in `results/metrics/predictions_<model>_<features>.csv`.
- Discrepancies between early project draft notes (hypothetical placeholder estimates $\sim 0.78$) and empirical measurements ($0.7488 \pm 0.0181$) have been fully reconciled.

---

## 2. Benchmark Reconciliation Table

The following table contrasts early project draft claims against the authoritative, recalculable empirical metrics:

| Model Architecture | Feature Representation | Early Project Notes Claim (Draft / Anecdotal) | Authoritative Empirical Test ROC-AUC ($\text{Mean} \pm \text{SEM}$) | Empirical ROC-AUC SD | Authoritative Empirical Test PR-AUC ($\text{Mean} \pm \text{SEM}$) | Empirical Balanced Accuracy | Empirical Macro F1 | Reconciled Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **Combined Features (2057d)** | $\sim 0.782$ (Hypothetical) | **$0.7488 \pm 0.0181$** | **0.0628** | **$0.3891 \pm 0.0320$** | **0.6636** | **0.3781** | **RECONCILED (Authoritative)** |
| **XGBoost** | Combined Features (2057d) | $\sim 0.765$ (Hypothetical) | **$0.7456 \pm 0.0179$** | **0.0620** | **$0.3495 \pm 0.0406$** | **0.6485** | **0.3591** | **RECONCILED (Authoritative)** |
| **Random Forest** | 9 Descriptors (9d) | $\sim 0.710$ (Hypothetical) | **$0.7215 \pm 0.0216$** | **0.0747** | **$0.2901 \pm 0.0408$** | **0.6337** | **0.3036** | **RECONCILED (Authoritative)** |
| **Random Forest** | Morgan Fingerprints (2048d) | $\sim 0.740$ (Hypothetical) | **$0.7208 \pm 0.0199$** | **0.0688** | **$0.3458 \pm 0.0316$** | **0.6475** | **0.3712** | **RECONCILED (Authoritative)** |
| **Logistic Regression** | 9 Descriptors (9d) | $\sim 0.690$ (Hypothetical) | **$0.7203 \pm 0.0169$** | **0.0585** | **$0.1968 \pm 0.0397$** | **0.6188** | **0.2472** | **RECONCILED (Authoritative)** |
| **Logistic Regression** | Combined Features (2057d) | $\sim 0.710$ (Hypothetical) | **$0.7138 \pm 0.0158$** | **0.0546** | **$0.2341 \pm 0.0338$** | **0.6404** | **0.3159** | **RECONCILED (Authoritative)** |
| **XGBoost** | Morgan Fingerprints (2048d) | $\sim 0.730$ (Hypothetical) | **$0.7114 \pm 0.0181$** | **0.0627** | **$0.3027 \pm 0.0308$** | **0.6340** | **0.3351** | **RECONCILED (Authoritative)** |
| **XGBoost** | 9 Descriptors (9d) | $\sim 0.680$ (Hypothetical) | **$0.7024 \pm 0.0256$** | **0.0887** | **$0.2911 \pm 0.0454$** | **0.6394** | **0.3133** | **RECONCILED (Authoritative)** |
| **Logistic Regression** | Morgan Fingerprints (2048d) | $\sim 0.670$ (Hypothetical) | **$0.6987 \pm 0.0149$** | **0.0515** | **$0.2602 \pm 0.0341$** | **0.6216** | **0.3001** | **RECONCILED (Authoritative)** |

---

## 3. Mathematical Metric Verification & Uncertainty Formulations

### 3.1 Endpoint-Wise Evaluation Formulations
For each of the $K=12$ toxicity endpoints $j \in \{1, \dots, 12\}$:
1. **Unlabelled Masking:** Any sample $i$ with $y_{ij} = \text{NaN}$ is masked out:
   $$\mathcal{S}_j = \{i \in \text{Test} \mid y_{ij} \in \{0, 1\}\}$$
2. **ROC-AUC:** Computed via trapezoidal numerical integration on rank-ordered continuous prediction scores:
   $$\text{ROC-AUC}_j = \int_0^1 \text{TPR}_j(\text{FPR}_j^{-1}(t)) \, dt$$
3. **PR-AUC (Average Precision):** Formulated strictly as the weighted mean of precisions achieved at each recall threshold:
   $$\text{PR-AUC}_j = \sum_{k} (R_{j, k} - R_{j, k-1}) P_{j, k}$$
   where $P_k$ and $R_k$ are precision and recall at the $k$-th probability threshold.
4. **Validation-Only Decision Thresholds:**
   Decision thresholds $\theta_j^* \in [0.05, 0.95]$ were determined solely on the validation partition to maximize F1-score:
   $$\theta_j^* = \arg\max_{\theta} \text{F1}(y_{\text{val}, j}, \mathbb{I}(p_{\text{val}, j} \ge \theta))$$
   These optimal thresholds were frozen and applied to test predictions $\hat{y}_{\text{test}, j} = \mathbb{I}(p_{\text{test}, j} \ge \theta_j^*)$, strictly preventing test label leakage.

### 3.2 Macro Aggregation & Uncertainty Mathematics
Macro averaging is performed across the $K=12$ valid endpoints:
$$\bar{\mu} = \frac{1}{K} \sum_{j=1}^K m_j$$
The reported uncertainty $\pm \text{SEM}$ is computed as the **Standard Error of the Mean across the 12 endpoints**:
$$\text{SD} = \sqrt{\frac{1}{K - 1} \sum_{j=1}^K (m_j - \bar{\mu})^2}, \quad \text{SEM} = \frac{\text{SD}}{\sqrt{K}} = \frac{\text{SD}}{\sqrt{12}}$$
- **Scientific Clarification:** This $\text{SEM}$ explicitly quantifies variance across biological endpoints/pathways, **not** repeated stochastic runs or confidence intervals across seeds.

---

## 4. Source Artifact Provenance

All empirical figures in the reconciliation table are directly verifiable from committed repository files:
1. `results/metrics/model_comparison_table.csv` — Full comparison matrix.
2. `results/metrics/model_benchmark_summary.json` — Exhaustive per-endpoint diagnostics and thresholds.
3. `results/metrics/predictions_random_forest_combined.csv` — Exact out-of-sample predictions for the champion baseline.
4. `tests/test_final_evaluation_regression.py` — Automated pytest regression test recomputing metrics directly from prediction CSVs.

**Verdict:** The empirical benchmarks are genuine, reproducible, leak-free, and authoritative for all Phase 1 documentation and reporting.
