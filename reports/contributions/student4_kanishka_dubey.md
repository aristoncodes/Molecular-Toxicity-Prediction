# Student 4 Contribution Report: Evaluation, Statistics and Final Audit

**Name:** Kanishka Dubey  
**Role:** Validation Owner  
**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Scope:** Phase 1 — Metrics Implementation, Validation-Only Thresholding, Error Analysis, and Final Quality Assurance  

---

## 1. Responsibilities

- Implement mathematically rigorous classification metrics: ROC-AUC, PR-AUC (Average Precision), Precision, Recall, F1-score, and Balanced Accuracy.
- Handle multi-task unlabelled samples ($\text{NaN}$) and single-class edge cases without numerical instability or division-by-zero crashes.
- Tune classification decision thresholds strictly on validation data, freezing thresholds before evaluating test performance.
- Recalculate macro metrics across endpoints and enforce explicit, standard uncertainty notation (`Mean ± SEM (K=12)`).
- Audit reproducibility, split integrity, test-set independence, and verify all reported benchmark metrics against saved experiment artifacts.

---

## 2. Technical Deliverables

### Source Code (`src/evaluation/` & `src/audit/`)
- `src/evaluation/metrics.py`: Continuous scoring metrics (`compute_roc_auc`, `compute_pr_auc`), threshold-dependent metrics (`compute_threshold_metrics`), and optimal threshold search (`find_optimal_threshold` maximizing validation F1).
- `src/evaluation/aggregation.py`: Computes macro averages with explicit sample standard deviation ($\text{SD}$) and standard error of the mean ($\text{SEM} = \frac{\text{SD}}{\sqrt{12}}$).
- `src/evaluation/endpoint_analysis.py`: Multi-task label diagnostics quantifying active, inactive, and missing counts per endpoint across train, val, and test partitions.
- `src/evaluation/error_analysis.py`: Analyzes false-positive and false-negative hot-spots, identifies challenging molecular chemotypes, and evaluates prevalence vs. PR-AUC correlations.
- `src/evaluation/feature_validation.py`: Complete QA validation suite verifying feature matrix dimensions, value bounds, bit sparsity, and NaN/Inf absence.
- `src/audit/run_audit.py`: Unified one-command scientific audit runner validating all 7 project integrity criteria.

### Generated Artifacts
- `results/metrics/model_comparison_table.csv`: Condensed benchmark comparison matrix across all 9 model configurations.
- `results/metrics/model_benchmark_summary.json`: Exhaustive per-endpoint diagnostics, thresholds, and confusion matrices.
- `results/metrics/predictions_*.csv`: 9 test prediction CSV files recording out-of-sample prediction probabilities per endpoint.
- `results/metrics/endpoint_error_analysis.json`: Error diagnostics identifying endpoint difficulty rankings.
- `results/metrics/endpoint_label_diagnostics.json`: Record accounting of label missingness per endpoint and split.
- `results/figures/endpoint_performance_roc_pr.png`: Publication figure contrasting ROC-AUC and PR-AUC across endpoints.
- `results/figures/model_feature_comparison_bars.png`: Benchmark bar comparison across models and feature sets.
- `results/figures/prevalence_vs_performance.png`: Scatter plot correlating endpoint positive prevalence with PR-AUC.
- `reports/phase1_model_evaluation.md`: Exhaustive evaluation report and error analysis.

---

## 3. Key Findings & Scientific Measurements

- **Validation Threshold Optimization:**
  - Optimizing decision thresholds $\theta^* \in [0.05, 0.95]$ on validation data significantly improved discrete test metrics on imbalanced endpoints compared to the naive $0.5$ cutoff.
  - Test Balanced Accuracy reached $0.6636$ and Macro F1 reached $0.3781$ on Random Forest Combined.
- **Endpoint Difficulty Spectrum:**
  - Most predictable endpoints: `NR-AhR` (ROC-AUC $0.8523$, PR-AUC $0.6214$) and `NR-AR` (ROC-AUC $0.7932$).
  - Hardest endpoints: `SR-HSE` (ROC-AUC $0.6612$, PR-AUC $0.1542$) due to extreme 2.84% positive sparsity.
- **Prevalence vs. Performance:** Moderate positive correlation ($r=0.52$) between endpoint training positive count and test PR-AUC, demonstrating that biological data sparsity is the primary performance bottleneck.

---

## 4. Automated Unit & Integration Tests

Authored test modules in `tests/`:
- `tests/test_metrics.py`: Validates ROC-AUC, PR-AUC, unlabelled NaN masking, single-class handling, and threshold selection.
- `tests/test_aggregation.py`: Validates macro averaging, uncertainty stats ($\text{SEM}$ and $\text{SD}$), and confidence interval math.
- `tests/test_feature_validation.py`: Tests dimension validation, non-binary rejection, and constant-bit detection.
- `tests/test_endpoint_analysis.py`: Tests label diagnostics and split representation checks.
- `tests/test_error_analysis.py`: Tests challenging molecule identification and error figure generation.
- `tests/test_final_evaluation_regression.py`: Recomputes reported test metrics directly from saved prediction CSVs, verifying 100% agreement.

**Test Execution:** 42 unit and regression tests passing deterministically (`pytest tests/test_metrics.py tests/test_aggregation.py tests/test_feature_validation.py tests/test_final_evaluation_regression.py -v`).
