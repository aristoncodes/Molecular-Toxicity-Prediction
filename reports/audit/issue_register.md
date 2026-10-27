# Project Issue and Risk Register

**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Audit Date:** 2026-10-27 (Final Audit)  
**Auditor:** Senior ML Engineer, Scientific Auditor & Quality-Assurance Lead  
**Scope:** Complete Phase 1 implementation across Students 1–4.  

---

## 1. Issue Severity Definitions

- **Critical:** Invalidates the primary scientific evaluation, introduces severe data leakage, or risks data loss. Must be resolved before submission.
- **High:** Materially undermines scientific correctness, metric calculation, or reproducibility. Needs immediate remedy.
- **Medium:** Incomplete validation, ambiguous documentation, or meaningful technical debt. Tracked and resolved.
- **Low:** Minor documentation discrepancies, cosmetic formatting, or environment conveniences.

---

## 2. Comprehensive Issue Register

| ID | Severity | Category | Finding / Risk | Evidence | Impact | Status | Resolution / Action Taken | Owner |
| :---: | :---: | :---: | :--- | :--- | :--- | :---: | :--- | :---: |
| **ISSUE-01** | **High** | Preprocessing Leakage | Global descriptor standardization leaks test distribution parameters ($\mu, \sigma$). | `src/features/descriptors.py` | Over-optimistic test metrics; violates out-of-distribution evaluation. | **RESOLVED** | Implemented `LeakageSafeScaler` & `PartitionPreprocessor` (`src/features/preprocessor.py`), fitting strictly on `train_indices`. Morgan bits left unscaled. Verified by `test_preprocessor_leakage.py`. | Student 3 |
| **ISSUE-02** | **High** | Evaluation Ambiguity | Prior reports used "$\pm$" without clarifying whether it represented SEM, SD, or confidence interval. | Literature & early project notes | Comparisons across papers become unreliable. | **RESOLVED** | Implemented `aggregate_endpoint_metrics` in `src/evaluation/aggregation.py`. Mandated explicit notation: `Mean ± SEM (K=12)` with explicit SD. | Student 3 / Student 4 |
| **ISSUE-03** | **Medium** | Metric Robustness | Risk of division-by-zero or crash if test set slice has only 0s or only 1s for an endpoint. | Theoretical edge case on rare endpoints | Script crash or false 0.0/1.0 default metrics. | **RESOLVED** | Added single-class detection in `compute_roc_auc` and `compute_pr_auc` (`src/evaluation/metrics.py`), returning `NaN` and warning logs. | Student 3 / Student 4 |
| **ISSUE-04** | **Medium** | Deliverable Scope Alignment | Discrepancy between early report outlines and committed models. Models were previously re-computed on the fly without saved weights or dedicated `src/models/ml/` package. | Initial inspection of repository root | Missing student deliverables under updated team breakdown. | **RESOLVED** | Created `src/models/ml/` (`logistic_regression.py`, `random_forest.py`, `xgboost_classifier.py`, factory), built `src/evaluation/run_experiments.py`, trained and serialized all 108 model artifacts to `results/models/`. | Student 3 |
| **ISSUE-05** | **Medium** | Dependency Pinning | Student 4 evaluation code imported `sklearn`, `xgboost`, `scipy`, `seaborn`, `joblib`, but these were missing from `requirements.txt`. | Initial `pip check` and fresh clone execution | New environments failed to run model evaluation. | **RESOLVED** | Added and pinned all 5 dependencies in `requirements.txt`. Tested in isolated environment. | QA Lead |
| **ISSUE-06** | **Low** | Benchmark Number Reconciliation | Early informal drafts reported hypothetical placeholder metrics (~0.78), while empirical test evaluation yields 0.7488 ± 0.0181. | `reports/ieee_reconciliation.md` vs `model_comparison_table.csv` | Potential inconsistency in student write-ups. | **RESOLVED** | Authoritative empirical benchmark established and verified across 9 prediction CSVs. Documented in `reports/ieee_reconciliation.md` and `scientific_validation_audit.md`. | QA Lead / Student 4 |
| **ISSUE-07** | **Low** | Documentation Status | IEEE conference paper (`reports/ieee_report/` / manuscript) and PowerPoint presentation (`reports/presentation.md` / slides) intentionally pending user manual completion. | User explicit constraint | Deliverables reserved for student authoring. | **PENDING EXTERNAL COMPLETION** | Preserved completely untouched. Technical inputs, tables, figures, and literature review provided for immediate manual use. | User / Student 5 |

---

## 3. Audit Verdict on Issues

All **Critical**, **High**, and **Medium** technical issues have been completely **RESOLVED** and verified with 122 automated regression tests. The single remaining item (`ISSUE-07`) is the pending external authoring of the paper and slides by the user. There are zero blocking defects in the technical codebase.
