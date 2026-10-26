# IEEE Report and Experiment Results Reconciliation

**Author:** Student 3 (Independent Phase 1 Auditor & Contributor)  
**Date:** 2026-10-26  
**Status:** Complete & Reconciled  

---

## 1. Executive Summary

This document reconciles all scientific, numerical, and structural claims in the project's documentation, reports, and IEEE report manuscript with the code, artifacts, and test suites in the repository.

As required by the project audit principles:
> **Accuracy and scientific integrity take priority over achieving a green checklist. If something cannot be verified, label it *not verified*. Do not invent results, tests, commits, or evidence.**

---

## 2. Granular Claim-by-Claim Reconciliation Matrix

| Category | Claim / Topic | Status in Codebase | Audit Finding & Action Taken |
| :--- | :--- | :---: | :--- |
| **Dataset Source** | DeepChem Tox21 URL and reference SHA-256 (`45d09792...`) | **VERIFIED** | Checksum matches exactly (`122,925` bytes). |
| **Raw Records** | 7,831 rows × 14 columns | **VERIFIED** | Counted directly from decompressed CSV. |
| **Exclusions** | 8 structures excluded due to unparsable valences | **VERIFIED** | All 8 logged in `exclusions.csv`; independently verified that all 8 fail RDKit parsing due to aluminum valences. |
| **Cleaned Records** | 7,823 valid unique canonical structures | **VERIFIED** | 100% parse in RDKit with 0 errors; canonical invariance rate is 100.0%; duplicate groups = 0. |
| **Endpoints** | 12 Tox21 assays with binary {0, 1} and NaN | **VERIFIED** | Confirmed strictly binary or null; no NaNs imputed or converted to 0. |
| **Morgan Fingerprints** | 2,048 bits, radius 2, uint8, shape `(7823, 2048)` | **VERIFIED** | Stored in `data/processed/features/morgan_r2_2048.npy`; matches manifest. |
| **Descriptors** | 9 physicochemical descriptors, float64, shape `(7823, 9)` | **VERIFIED** | Stored in `data/processed/features/descriptors_9.npy`; matches manifest. |
| **Combined Matrix** | Concatenation of bits + descriptors, shape `(7823, 2057)` | **VERIFIED** | Exact horizontal concatenation verified with zero numerical drift. |
| **Scaffold Splitting** | 6,258 train / 782 val / 783 test (Seed 42) | **VERIFIED** | Target 80/10/10 achieved with 79.995% / 9.996% / 10.009%. |
| **Scaffold Isolation** | Zero scaffold overlap across train, val, and test | **VERIFIED** | Recomputed all Bemis–Murcko scaffolds with RDKit; confirmed 0 overlap across partitions. |
| **Acyclic Molecules** | 1,775 acyclic molecules placed in train | **VERIFIED** | Handled via `single_group` policy; all 1,775 placed strictly in train partition. |
| **Model Baselines** | Logistic Regression, Random Forest, XGBoost trained models | **NOT VERIFIED (Phase 2)** | No trained model binaries (`.joblib`, `.pt`) existed in repo. Claim clarified as Phase 2 scope. |
| **Evaluation Metrics** | Continuous ROC-AUC and PR-AUC with NaN masking | **IMPLEMENTED & AUDITED** | Implemented by Student 3 in `src/evaluation/metrics.py` with 100% passing tests. |
| **Uncertainty Reporting**| Macro Mean ± Uncertainty | **RECONCILED & STANDARDIZED**| Enforced explicit notation: `Mean ± SEM (K=12)` vs `Mean ± SD (K=12)` to eliminate ambiguity. |

---

## 3. Discrepancies Resolved

1. **Clarification on Baseline Modeling Deliverables:**
   Prior hypothetical references to baseline classifiers (Logistic Regression, Random Forest, XGBoost) and test-set ROC-AUC numbers were investigated. The repository contained no implementation code, model weights, or saved prediction logs for these models prior to Student 3's contribution. Rather than claiming an unverified reproduction, the manuscript and documentation now explicitly classify model benchmarking under **Phase 2**.

2. **Elimination of "$\pm$" Ambiguity:**
   In prior reports, "$\pm$" was used without distinguishing sample standard deviation ($\text{SD}$) from standard error of the mean ($\text{SEM}$). In `src/evaluation/aggregation.py`, functions now explicitly require labeling the uncertainty metric.

3. **Preprocessing Leakage Prevention:**
   Addressed the risk of fitting descriptor scaling across the entire dataset. Implemented `PartitionPreprocessor` to guarantee that descriptor standardization is strictly fitted on training partition indices.
