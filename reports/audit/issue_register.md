# Project Issue and Risk Register

**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Audit Date:** 2026-10-26  
**Auditor:** Senior ML Engineer & Quality-Assurance Lead  
**Scope:** Active, resolved, and monitored issues across Phase 1 deliverables and transition to Phase 2.

---

## 1. Issue Severity Definitions

- **Critical:** Invalidates the primary scientific evaluation, introduces severe data leakage, or risks data loss. Must be resolved before further progress.
- **High:** Materially undermines scientific correctness, metric calculation, or reproducibility. Needs immediate remedy.
- **Medium:** Incomplete validation, ambiguous documentation, or meaningful technical debt. Tracked and resolved during standard development.
- **Low:** Minor documentation discrepancies, cosmetic formatting, or code organization enhancements.

---

## 2. Issue Register

| ID | Severity | Category | Finding / Risk | Evidence | Impact | Status | Recommended Fix / Resolution | Owner |
| :---: | :---: | :---: | :--- | :--- | :--- | :---: | :--- | :---: |
| **ISSUE-01** | **High** | Preprocessing Leakage | Continuous molecular descriptors have wide dynamic ranges (e.g. MW in $[9, 1877]$). Standardizing features across the entire dataset before splitting leaks test distribution parameters ($\mu, \sigma$) into training. | `src/features/descriptors.py`, `build_features.py` | Over-optimistic test metrics; violates scaffold out-of-distribution evaluation premise. | **RESOLVED** | Implemented `PartitionPreprocessor` and `LeakageSafeScaler` (`src/features/preprocessor.py`) which fit scalers strictly on training indices and keep Morgan bits unscaled. Verified by `test_preprocessor_leakage.py`. | Student 3 |
| **ISSUE-02** | **High** | Evaluation Ambiguity | Prior project reports used "$\pm$" without defining whether it represented sample Standard Deviation ($\text{SD}$), Standard Error of the Mean ($\text{SEM}$), or Confidence Intervals. | Literature reviews and prior informal reports | Results cannot be compared reliably against external literature benchmarks. | **RESOLVED** | Implemented `format_metric_with_uncertainty` and `aggregate_endpoint_metrics` in `src/evaluation/aggregation.py`. Mandated explicit labels (e.g. `0.7821 ± 0.0235 (SEM, K=12)`). | Student 3 |
| **ISSUE-03** | **Medium** | Metric Evaluation | Potential for division-by-zero or crashes if a small test set partition or cross-validation fold contains only negative or only positive instances for an endpoint. | Theoretical edge case in extreme imbalanced toxicity endpoints | Evaluation scripts would crash or return false 0.0/1.0 defaults during CV. | **RESOLVED** | Added explicit single-class detection in `compute_roc_auc` and `compute_pr_auc` (`src/evaluation/metrics.py`), returning `NaN` and flagging `single_class_undefined` with warning logs. | Student 3 |
| **ISSUE-04** | **Medium** | Deliverable Scope | Prior completion report outlines mentioned trained baseline models (Logistic Regression, Random Forest, XGBoost) and benchmark ROC-AUC scores, but no model code or weights existed in the repository. | Absence of `models/` directory, model training scripts, and prediction CSVs | Mismatch between informal narrative and actual code repository. | **RESOLVED** | Documented honestly in `reports/ieee_reconciliation.md` and `reports/ieee_phase1_manuscript.tex`. Reconciled that baseline model training is the primary initial deliverable for Phase 2 / Student 4. | Student 3 / QA Lead |
| **ISSUE-05** | **Low** | Environment Setup | Virtual environment `.venv` is gitignored (as per best practice) but requires explicit setup commands on new developer clones. | Fresh clone lacked `.venv/` until created from requirements | New developers must run setup commands in `README.md`. | **MONITORED** | Clearly documented in `README.md`: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. All requirements pinned. | Team |
| **ISSUE-06** | **Low** | LaTeX Tooling | Local macOS environment does not have `pdflatex` or `xelatex` installed in CLI path. | `which pdflatex` exited with code 1 | Manuscript cannot be locally compiled to PDF without installing TeX Live / MacTeX. | **MONITORED** | `reports/ieee_phase1_manuscript.tex` authored using standard IEEEtran format, fully compatible with Overleaf and standard TeX distributions. | Student 3 / Team |

---

## 3. Summary Assessment

All **Critical** and **High** severity issues have been fully resolved and verified with automated regression tests. The remaining issues are either documented or monitored for Phase 2. There are zero blocking defects preventing the commencement of Phase 2 work.
