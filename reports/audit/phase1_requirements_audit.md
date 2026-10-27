# Phase 1 Requirements and Rubric Compliance Audit

**Project:** Molecular Toxicity Prediction via Fingerprints and Molecular Graph Networks  
**Course:** CSAI Student Project — Phase 1 Machine Learning Deliverables  
**Audit Standard:** Official CSAI Evaluation Rubric  
**Auditor:** Senior ML Engineer & Quality-Assurance Lead  
**Audit Date:** 2026-10-27  

---

## 1. Executive Compliance Summary

Every technical and scientific requirement mandated for Phase 1 has been audited against code, execution traces, regression tests, and saved evaluation artifacts. 

| Rubric Dimension | Requirement Description | Compliance Status | Verified Evidence & Source Code |
| :--- | :--- | :---: | :--- |
| **1. Dataset Selection & Acquisition** | Standard benchmark dataset with verifiable provenance, automated download, and checksum validation. | **PASS** | `src/data/download_data.py`, `data/raw/download_provenance.json`, SHA-256: `45d09792...` matching Tox21 gzip. |
| **2. Preprocessing & Quality Assurance** | Chemical structure sanitization, handling invalid SMILES, duplicate/conflict audit, and missing label integrity. | **PASS** | `src/data/preprocess.py`, `src/data/inspect_dataset.py`, 7,823 valid molecules, 8 organometallic exclusions documented, 0 label conflicts. |
| **3. Exploratory Data Analysis (EDA)** | Quantitative distributions, class balance, multi-task correlation, chemical property distributions, and exportable figures. | **PASS** | `src/data/generate_eda_plots.py`, 10 publication-quality figures with numeric JSON summaries in `results/figures/eda/`. |
| **4. Feature Extraction & Engineering** | Multi-modal molecular representations: circular fingerprints, physicochemical descriptors, and combined matrix. | **PASS** | `src/features/fingerprints.py` (2048-bit Morgan), `src/features/descriptors.py` (9 2D descriptors), `src/features/build_features.py` (2057-dim combined). |
| **5. Data Splitting & Leakage Prevention** | Out-of-distribution evaluation via Bemis-Murcko scaffold splitting with zero leakage and acyclic molecule isolation. | **PASS** | `src/data/split_data.py`, `src/features/preprocessor.py`, 80/10/10 split, exact 0 scaffold overlap, training-only scaler fitting. |
| **6. Baseline Machine Learning Models** | Multi-task training of at least 3 distinct classical ML architectures across feature configurations. | **PASS** | `src/evaluation/model_evaluation.py`, Logistic Regression, Random Forest, XGBoost trained across Morgan, Descriptors, Combined (9 setups). |
| **7. Metric Evaluation & Uncertainty** | Imbalance-aware evaluation (ROC-AUC, PR-AUC, F1, Balanced Acc), validation-only threshold tuning, and macro $\pm \text{SEM}$ uncertainty. | **PASS** | `src/evaluation/metrics.py`, `src/evaluation/aggregation.py`, validation threshold selection, test out-of-sample evaluation, 13 prediction/metrics files in `results/metrics/`. |
| **8. Testing & Reproducibility** | Comprehensive automated test suite validating data, features, splits, metrics, and models. Single-command reproducibility. | **PASS** | 117 passing automated unit/regression tests (`pytest tests/`), single-command audit runner (`python -m src.audit.run_audit`). |
| **9. Academic Documentation & Viva Preparation** | Complete technical reports, theoretical literature review, oral defense guide, and clear architecture documentation. | **PASS** | `reports/literature_review.md`, `reports/viva_preparation.md`, `reports/phase1_model_evaluation.md`, `README.md`. |
| **10. IEEE Manuscript & Presentation** | Formal IEEE manuscript and oral defense presentation slides. | **USER-RESERVED** | Explicitly reserved for manual student authoring per user instructions. Codebase and metrics ready for direct inclusion. |

---

## 2. Granular Audit by CSAI Rubric Criteria

### Criterion 1: Dataset Acquisition and Provenance
- **Audited Items:** Raw dataset URL, automated streaming retrieval, SHA-256 checksum matching, and raw metadata storage.
- **Evidence:** `data/raw/download_provenance.json` confirms source `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz`, file size 445,471 bytes, SHA-256 `45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360`.
- **Verdict:** **PASS**.

### Criterion 2: Preprocessing and Structure Sanitization
- **Audited Items:** Validation of SMILES strings, RDKit sanitization, identification of unparsable structures, canonicalization, and duplicate group resolution.
- **Evidence:** 7,831 raw records processed. Exactly 8 organo-aluminum structures excluded with explicit reasons in `results/experiment_logs/exclusions.csv`. 7,823 structures converted to canonical isomeric SMILES with 100% validity in RDKit. 0 duplicate groups, 0 conflicting labels.
- **Verdict:** **PASS**.

### Criterion 3: Exploratory Data Analysis (EDA)
- **Audited Items:** Multi-task label missingness and active prevalence, distribution of molecular properties (MW, HAC, ring count, rotatable bonds, atom counts).
- **Evidence:** 10 discrete figures generated in both PNG (300 DPI) and PDF formats with numerical JSON statistics in `results/figures/eda/`. Verified that positive prevalence ranges from 2.84% (`SR-HSE`) to 15.82% (`NR-AhR`).
- **Verdict:** **PASS**.

### Criterion 4: Feature Representation
- **Audited Items:** Extraction of Morgan fingerprints ($r=2$, 2048 bits), calculation of 9 physicochemical descriptors, assembly of combined feature matrix, and row ID alignment.
- **Evidence:** Generated artifacts in `data/processed/features/`: `morgan_r2_2048.npy` (7823, 2048), `descriptors_9.npy` (7823, 9), and `combined_features.npy` (7823, 2057). Verified determinism, non-negativity of bit vectors, and bounded physical descriptors.
- **Verdict:** **PASS**.

### Criterion 5: Scaffold Splitting and Leakage Prevention
- **Audited Items:** Bemis-Murcko scaffold identification, split partition proportions (80/10/10), scaffold isolation guarantee, handling of acyclic molecules, and preprocessing scaler isolation.
- **Evidence:** Evaluated on seed 42: Train 6,258 (79.995%), Val 782 (9.996%), Test 783 (10.009%). Exact 0 overlapping scaffolds across all three partitions. 1,775 acyclic molecules safely confined to training set. `LeakageSafeScaler` fits only on train indices.
- **Verdict:** **PASS**.

### Criterion 6: Multi-Task Modeling and Baseline Benchmarks
- **Audited Items:** Implementation and training of Logistic Regression, Random Forest, and XGBoost across 12 endpoints and 3 feature spaces (9 total benchmarks).
- **Evidence:** Execution of `src/evaluation/model_evaluation.py` generated 9 test prediction CSV files, summary JSON, and comparison table. Best model identified: Random Forest on Combined Features (Test Macro ROC-AUC $0.7488 \pm 0.0181$).
- **Verdict:** **PASS**.

### Criterion 7: Metric Rigor and Imbalance Handling
- **Audited Items:** ROC-AUC, PR-AUC, Balanced Accuracy, F1-score; unlabelled sample masking; validation-only threshold tuning; explicit uncertainty reporting ($\text{Mean} \pm \text{SEM}$, $\text{SD}$).
- **Evidence:** Implemented in `src/evaluation/metrics.py` and `src/evaluation/aggregation.py`. Thresholds selected exclusively on validation set to maximize F1-score and applied frozen to test set. All metrics report explicit $\pm \text{SEM}$ with $K=12$.
- **Verdict:** **PASS**.

### Criterion 8: Reproducibility and Quality Assurance
- **Audited Items:** Automated test coverage, single-command reproducibility, pinned requirements, and issue register tracking.
- **Evidence:** 117 automated pytest test cases passing (`pytest tests/ -v --run-real-data`). Unified audit script `python -m src.audit.run_audit` executes in 4.5 seconds and passes all 7 criteria. Fully pinned `requirements.txt`.
- **Verdict:** **PASS**.
