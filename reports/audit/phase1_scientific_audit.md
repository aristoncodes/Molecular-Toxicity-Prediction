# Phase 1 Scientific and Technical Audit Report

**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Audit Date:** 2026-10-26  
**Auditor:** Senior ML Engineer & Quality-Assurance Lead  
**Scope:** Rigorous, evidence-based verification of Phase 1 implementation (Dataset, Features, Splitting, Leakage, Models, Metrics, Reproducibility, Reporting).

---

## 1. Executive Summary

A comprehensive, evidence-based scientific audit of the entire Phase 1 implementation was conducted. The audit verified all underlying code, data files, manifests, split indices, feature arrays, test cases, and documentation across the work of Student 1, Student 2, and Student 3.

**Key Findings:**
1. **Dataset Integrity (Student 1):** Verified from raw download to cleaned CSV. Gzip SHA-256 (`45d09792...`) matches reference. 7,831 raw records yield exactly 8 unparsable exclusions (aluminum valence violations) and 7,823 valid structures. Canonical isomeric SMILES invariance is 100.0% with 0 duplicate structures.
2. **Feature Engineering (Student 2):** 2,048-bit Morgan circular fingerprints ($r=2$, uint8), 9 physicochemical descriptors (float64), and 2,057-dimensional combined representations (float32) are properly constructed, aligned with row IDs, and hashed in manifests.
3. **Scaffold Splitting (Student 2):** Deterministic Bemis--Murcko scaffold splitting (seed 42) into 6,258 train (79.995%), 782 val (9.996%), and 783 test (10.009%) instances. Independent re-extraction confirmed **0 scaffold overlap** across partitions. Acyclic molecules (1,775 structures) are unified in train (`single_group` policy).
4. **Data Leakage & Preprocessing (Student 3):** Preprocessing distribution leakage hazard identified and resolved via `PartitionPreprocessor` (`src/features/preprocessor.py`), ensuring normalization statistics are fitted strictly on training partition indices.
5. **Evaluation Metrics & Uncertainty (Student 3):** Continuous ROC-AUC and PR-AUC (Average Precision) implemented in `src/evaluation/metrics.py` with missing label masking and single-class endpoint protection. Macro metric aggregation (`src/evaluation/aggregation.py`) strictly disambiguates $\text{Mean} \pm \text{SEM}$ ($K=12$) from sample $\text{SD}$ and 95% Student-$t$ confidence intervals.
6. **Reproducibility Suite (Student 3):** Unified CLI `python -m src.audit.run_audit` executes all 7 scientific audit criteria, outputting machine-readable JSON logs. 85 automated test cases in `tests/` pass with zero failures.
7. **Model Baselines Status:** No baseline models (Logistic Regression, Random Forest, XGBoost) were executed or persisted in Phase 1. Prior hypothetical mentions in project outlines were reconciled as the primary starting deliverable for Phase 2.

---

## 2. Granular Audit by Scientific Domain

### A. Dataset Verification
- **Provenance:** Sourced from DeepChem S3 (`tox21.csv.gz`). File size: 122,925 bytes. SHA-256: `45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360`. Provenance metadata logged in `data/raw/download_provenance.json`.
- **Raw Row Count:** 7,831 rows across 14 columns (12 endpoints, `mol_id`, `smiles`).
- **Cleaned Row Count:** 7,823 rows across 18 columns (original 14 + `source_row`, `canonical_smiles`, `duplicate_group`, `duplicate_status`).
- **Exclusion Accounting:** Exactly 8 excluded compounds logged in `results/experiment_logs/exclusions.csv`. Every excluded molecule was independently re-parsed in RDKit: all 8 failed sanitization due to explicit aluminum valence violations (e.g., `NC(=O)NC1N=C(O[AlH3](O)O)NC1=O` has valence 6 on Al). Zero valid molecules were erroneously excluded.
- **Canonicalization:** 100% of the 7,823 `canonical_smiles` entries parse into valid `Chem.Mol` objects without errors and re-canonicalize to identical strings (`Chem.MolToSmiles(..., canonical=True, isomericSmiles=True)`).
- **Duplicate Audit:** 0 duplicate canonical SMILES groups detected in this Tox21 release.
- **Label Integrity:** For all 12 endpoints, non-missing values are strictly in $\{0.0, 1.0\}$. Missing values ($\text{NaN}$) are strictly preserved and never imputed or converted to 0.

### B. Feature Engineering
- **Morgan Fingerprints:** Computed using modern RDKit `rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)`. Stored in `data/processed/features/morgan_r2_2048.npy` (shape $7823 \times 2048$, dtype `uint8`, SHA-256 `f804b3d7...`).
- **Physicochemical Descriptors:** 9 standard RDKit descriptors (`mol_weight`, `logp`, `tpsa`, `num_h_donors`, `num_h_acceptors`, `num_rotatable_bonds`, `ring_count`, `heavy_atom_count`, `fraction_csp3`). Stored in `data/processed/features/descriptors_9.npy` (shape $7823 \times 9$, dtype `float64`, SHA-256 `9bbcc071...`). Validated against exact literature values on reference molecules (Ethanol, Aspirin, Caffeine).
- **Combined Matrix:** Horizontal concatenation of Morgan bits and descriptors. Stored in `data/processed/features/combined_features.npy` (shape $7823 \times 2057$, dtype `float32`, SHA-256 `fe8af91f...`). Exact numerical preservation verified:
  $$\mathbf{X}[:, :2048] \equiv \mathbf{X}_{\text{morgan}}, \quad \mathbf{X}[:, 2048:] \approx \mathbf{X}_{\text{descriptors}}$$
- **Alignment:** 1-to-1 alignment between feature rows and cleaned CSV verified via `data/processed/features/feature_row_ids.json`.

### C. Train, Validation, and Test Splits
- **Splitting Strategy:** Deterministic Bemis--Murcko scaffold clustering (`src/data/split_data.py`, seed 42, target 80% train / 10% val / 10% test).
- **Partition Accounting:**
  - Train: 6,258 molecules (79.995%) — 760 scaffolds
  - Validation: 782 molecules (9.996%) — 782 scaffolds
  - Test: 783 molecules (10.009%) — 783 scaffolds
  - Total: 7,823 molecules (100.000%) — 2,325 scaffolds
- **Mathematical Integrity:**
  - $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$
  - $\text{Train} \cup \text{Val} \cup \text{Test} = \{0, 1, \dots, 7822\}$
  - Index bounds: $0 \le i < 7823$ for all indices.
- **Scaffold Isolation:** Bemis--Murcko scaffolds recomputed from scratch with RDKit on all 7,823 molecules confirmed **zero scaffold overlap** between partitions for all non-empty scaffolds:
  $$\text{Scaffolds}_{\text{Train}} \cap \text{Scaffolds}_{\text{Val}} = \emptyset, \quad \text{Scaffolds}_{\text{Train}} \cap \text{Scaffolds}_{\text{Test}} = \emptyset, \quad \text{Scaffolds}_{\text{Val}} \cap \text{Scaffolds}_{\text{Test}} = \emptyset$$
- **Acyclic Policy:** 1,775 acyclic molecules (empty scaffold `""`) are unified under the `single_group` policy and placed strictly in the training partition. This prevents arbitrary splitting of aliphatic chains across validation/test sets, ensuring out-of-sample evaluation assesses true cyclic structural generalization.

### D. Data Leakage and Preprocessing
- **Hazard Identified:** In molecular toxicity modeling, standardizing continuous descriptors (e.g., zero mean, unit variance) across the whole dataset before splitting leaks distribution parameters ($\mu_{\text{test}}, \sigma_{\text{test}}$) into the training set.
- **Hazard Resolved:** Implemented `PartitionPreprocessor` and `LeakageSafeScaler` (`src/features/preprocessor.py`). Descriptors are standardized using parameters fitted *strictly* on training partition indices. Binary Morgan fingerprint bits are left untouched, preserving binary sparsity and bit semantics.
- **Label Independence:** Feature extraction and splitting code never read or access toxicity endpoint columns.

### E. Model Implementation & Baseline Status
- **Audit Finding:** The repository currently contains no baseline model training scripts (e.g. `src/evaluation/run_experiments.py` was not implemented) and no trained model binaries (`models/*.joblib`) or prediction logs (`results/*.csv`).
- **Reconciliation:** Any prior claims of trained Random Forest, Logistic Regression, or XGBoost benchmark numbers were hypothetical/prospective rather than supported by real artifacts. Student 3 correctly flagged this, implemented the foundational evaluation and metric aggregation modules, and established baseline modeling as the starting scope for Phase 2.

### F. Metrics and Uncertainty Methodology
- **ROC-AUC (`compute_roc_auc`):** Uses continuous prediction scores / probabilities and Wilcoxon-Mann-Whitney rank-sum formulation. Automatically masks unobserved labels ($\text{NaN}$).
- **PR-AUC (`compute_pr_auc`):** Implements Average Precision ($AP = \sum (R_k - R_{k-1}) P_k$) as standard in machine learning, plus trapezoidal option, documenting why AP is mathematically sound for imbalanced data.
- **Single-Class Protection:** When an evaluation slice contains only 0s or only 1s, the metric returns $\text{NaN}$ and flags status `'single_class_undefined'` to prevent division-by-zero distortion.
- **Macro Aggregation (`aggregate_endpoint_metrics`):** Macro-averages across valid observed endpoints. Computes:
  - Macro Mean: $\frac{1}{K} \sum \text{Metric}_k$
  - Sample Standard Deviation ($\text{SD}$) across endpoints
  - Standard Error of the Mean ($\text{SEM} = \frac{\text{SD}}{\sqrt{K}}$)
  - 95% Student-$t$ Confidence Interval: $\text{Mean} \pm t_{0.975, df} \cdot \text{SEM}$
  - Mandates explicit labeling (e.g., `0.7821 ± 0.0235 (SEM, K=12)`) to eliminate "$\pm$" ambiguity.

### G. Test Suite & Reproducibility
- **Total Test Cases:** 85 tests across 12 test modules in `tests/`.
- **Execution Command:** `pytest tests/ -v --run-real-data` -> **85 passed in 8.01s** (0 failed, 0 errors).
- **Offline / Synthetic Support:** 84 tests pass without requiring local data artifacts; 1 real-data test safely skips when raw data is missing and passes when present.
- **Automated Audit CLI:** `python -m src.audit.run_audit` executes all 7 criteria and outputs `results/experiment_logs/audit_results.json`.

### H. Documentation & IEEE Manuscript
- `README.md`: Comprehensively updated with architecture, scientific policies, and CLI reproduction commands for Students 1, 2, and 3.
- `reports/ieee_phase1_manuscript.tex`: Complete, formal IEEE conference paper manuscript documenting audited Phase 1 findings, exact numbers, methodology, and citations (Tox21 Challenge, Bemis & Murcko, Rogers & Hahn).
- `reports/ieee_reconciliation.md`: Matrix reconciling claims against saved code and artifacts.

---

## 3. Conclusion

The Phase 1 foundation is mathematically sound, reproducible, and completely free of data leakage. The repository is in an exemplary state for the handover to Student 4.
