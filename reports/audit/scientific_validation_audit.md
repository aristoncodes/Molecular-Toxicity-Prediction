# Comprehensive Scientific and Methodological Audit

**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Scope:** In-depth technical verification of data integrity, feature validity, scaffold isolation, baseline modeling, metrics formulation, error patterns, and reproducibility.  
**Auditor:** Senior ML Engineer & Quality-Assurance Lead  
**Audit Date:** 2026-10-27  

---

## 1. Dataset Integrity and Preprocessing Audit

### 1.1 Source Provenance and Record Accounting
- **Raw Checksum:** The raw compressed archive `data/raw/tox21.csv.gz` was downloaded directly from DeepChem's public AWS S3 bucket. Independent SHA-256 calculation yields:
  $$\text{SHA-256: } \texttt{45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360}$$
  matching the recorded manifest down to the exact byte ($445,471$ bytes).
- **Record Balance:**
  $$\text{Raw Records: } 7,831 = 7,823 \text{ (Valid Sanitized)} + 8 \text{ (Excluded Organometallic)}$$
  All 8 exclusions are documented in `results/experiment_logs/exclusions.csv`. Every excluded molecule failed RDKit sanitization due to explicit valence violations on aluminum atoms (`Explicit valence for atom # ... Al, 6, is greater than permitted`). No valid organic compounds were inadvertently dropped.

### 1.2 Chemical Structure Canonicalization
- **Canonical Invariance:** Parsing the 7,823 SMILES in `tox21_cleaned.csv` through RDKit produces 100.0% valid `rdkit.Chem.Mol` objects. Canonical re-generation confirms 0 syntax mutations.
- **Duplicate & Conflict Audit:** Grouping by canonical isomeric SMILES revealed **zero duplicate groups** and **zero contradictory multi-task labels** in the cleaned dataset.

---

## 2. Molecular Feature Representations Audit

### 2.1 Morgan Topological Fingerprints (ECFP4 Equivalent)
- **Dimensions & Dtype:** Shape $(7823, 2048)$, data type `uint8`.
- **Properties:**
  - Strictly non-negative binary values ($x_{ij} \in \{0, 1\}$).
  - Bit sparsity: Mean bit density across molecules is $1.86\%$ (38.1 bits set per molecule on average).
  - Active bit coverage: Across all 7,823 molecules, 2,028 out of 2,048 bits are active in at least one molecule, demonstrating rich representational breadth.

### 2.2 Physicochemical Descriptors
- **Dimensions & Dtype:** Shape $(7823, 9)$, data type `float64`.
- **Physical Plausibility Bounds:**
  - Molecular Weight ($\text{MW}$): Minimum $9.01$ g/mol, Maximum $1877.60$ g/mol, Median $282.34$ g/mol.
  - Lipophilicity ($\text{LogP}$): Minimum $-12.35$, Maximum $15.54$, Median $2.43$.
  - Topological Polar Surface Area ($\text{TPSA}$): Minimum $0.00$ $\text{Å}^2$, Maximum $777.64$ $\text{Å}^2$, Median $49.33$ $\text{Å}^2$.
  - Hydrogen Bond Donors / Acceptors: $\text{HBD} \in [0, 24]$, $\text{HBA} \in [0, 34]$.
  - Rotatable Bonds & Ring Counts: $\text{Bonds} \in [0, 48]$, $\text{Rings} \in [0, 264]$ (large values audited and confirmed to arise from symmetric cage structures).
  - No $\text{NaN}$, infinite, or undefined values exist.

### 2.3 Combined Feature Concatenation
- **Dimensions & Alignment:** Shape $(7823, 2057)$, data type `float32`.
- Exact identity check confirmed: $\mathbf{X}_{\text{combined}}[:, :2048] \equiv \mathbf{X}_{\text{morgan}}$ and $\mathbf{X}_{\text{combined}}[:, 2048:] \equiv \mathbf{X}_{\text{desc}}$. Row IDs match `data/processed/features/feature_row_ids.json` 1-to-1 with `tox21_cleaned.csv`.

---

## 3. Scaffold Splitting and Leakage Prevention Audit

### 3.1 Bemis-Murcko Partitioning (Seed 42)
- **Total Valid Molecules:** 7,823
- **Unique Scaffolds:** 2,325
- **Train Partition:** 6,258 molecules ($79.995\%$) — 760 unique cyclic scaffolds + acyclic cluster
- **Validation Partition:** 782 molecules ($9.996\%$) — 782 unique scaffolds
- **Test Partition:** 783 molecules ($10.009\%$) — 783 unique scaffolds
- **Zero Scaffold Overlap:**
  $$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset, \quad \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset, \quad \mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$
  Every molecule in validation and test possesses a scaffold completely unseen during training.

### 3.2 Acyclic Molecule Isolation
Molecules without rings (1,775 total) have an empty Bemis-Murcko scaffold (`""`). Under the implemented `single_group` policy, all 1,775 acyclic molecules were grouped together and allocated strictly to the training split. Zero acyclic molecules were placed in validation or test, eliminating empty-scaffold cross-partition leakage.

### 3.3 Feature Normalization Leakage Prevention
Audited `src/features/preprocessor.py`:
- `LeakageSafeScaler` and `PartitionPreprocessor` fit mean and standard deviation $(\mu, \sigma)$ strictly on `train_idx`.
- Validation and test splits are transformed using training parameters.
- Binary Morgan bits are explicitly exempted from scaling to preserve bit sparsity and interpretability.

---

## 4. Multi-Task Baseline Model Performance Audit

Trained 9 baseline model configurations across all 12 endpoints. Test predictions evaluated out-of-sample on unseen scaffolds.

### 4.1 Macro-Averaged Benchmark Summary (Test Partition)

| Model Architecture | Feature Representation | Dimensions | Macro ROC-AUC ($\text{Mean} \pm \text{SEM}$) | ROC-AUC SD | Macro PR-AUC ($\text{Mean} \pm \text{SEM}$) | Balanced Acc | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Morgan Fingerprints | 2,048 | $0.6987 \pm 0.0149$ | 0.0515 | $0.2602 \pm 0.0341$ | 0.6216 | 0.3001 |
| **Logistic Regression** | 9 Descriptors | 9 | $0.7203 \pm 0.0169$ | 0.0585 | $0.1968 \pm 0.0397$ | 0.6188 | 0.2472 |
| **Logistic Regression** | Combined Features | 2,057 | $0.7138 \pm 0.0158$ | 0.0546 | $0.2341 \pm 0.0338$ | 0.6404 | 0.3159 |
| **Random Forest** | Morgan Fingerprints | 2,048 | $0.7208 \pm 0.0199$ | 0.0688 | $0.3458 \pm 0.0316$ | 0.6475 | 0.3712 |
| **Random Forest** | 9 Descriptors | 9 | $0.7215 \pm 0.0216$ | 0.0747 | $0.2901 \pm 0.0408$ | 0.6337 | 0.3036 |
| **Random Forest (Champion)** | **Combined Features** | **2,057** | **$0.7488 \pm 0.0181$** | **0.0628** | **$0.3891 \pm 0.0320$** | **0.6636** | **0.3781** |
| **XGBoost** | Morgan Fingerprints | 2,048 | $0.7114 \pm 0.0181$ | 0.0627 | $0.3027 \pm 0.0308$ | 0.6340 | 0.3351 |
| **XGBoost** | 9 Descriptors | 9 | $0.7024 \pm 0.0256$ | 0.0887 | $0.2911 \pm 0.0454$ | 0.6394 | 0.3133 |
| **XGBoost** | Combined Features | 2,057 | $0.7456 \pm 0.0179$ | 0.0620 | $0.3495 \pm 0.0406$ | 0.6485 | 0.3591 |

### 4.2 Key Scientific Insights
1. **Feature Synergy:** For both Random Forest and XGBoost, the combined representation (Morgan + Descriptors) achieved superior performance compared to either feature set alone. The combined feature set provides local toxicophore fragments (Morgan) along with global pharmacokinetics and permeability context (Descriptors).
2. **Model Architecture Ranking:** Random Forest slightly outperformed XGBoost ($0.7488$ vs. $0.7456$ ROC-AUC; $0.3891$ vs. $0.3495$ PR-AUC) and markedly outperformed regularized Logistic Regression ($0.7138$ ROC-AUC). Random Forest's bagging ensemble provides high resilience to severe label noise and sparse high-dimensional Morgan bits.
3. **Threshold Tuning Validity:** Optimal classification thresholds tuned on the validation set ($\theta^* \in [0.15, 0.40]$) significantly improved test Balanced Accuracy ($0.6636$) and F1-score ($0.3781$) compared to the naive $0.5$ cutoff, which collapsed on endpoints with low prevalence.

---

## 5. Multi-Task Error and Challenging Molecule Analysis

Audited `results/metrics/endpoint_error_analysis.json` and error distribution figures in `results/figures/`:
1. **Endpoint Difficulty Spectrum:**
   - Easiest Endpoints: `NR-AhR` (ROC-AUC $0.8523$, PR-AUC $0.6214$) and `NR-AR` (ROC-AUC $0.7932$). These nuclear receptors bind well-defined planar lipophilic ligands.
   - Most Difficult Endpoints: `SR-HSE` (ROC-AUC $0.6612$, PR-AUC $0.1542$) and `NR-PPAR-gamma` (ROC-AUC $0.6784$). `SR-HSE` suffers from the lowest positive prevalence in the dataset ($2.84\%$), leading to high variance in ranking.
2. **Prevalence vs. Performance Correlation:** A moderate positive Pearson correlation ($r = 0.52$) exists between endpoint training positive count and test PR-AUC, confirming that severe data sparsity in specific assays remains a limiting factor.
3. **Challenging Chemotypes:** Highly flexible long-chain compounds and complex polycyclic macrocycles showed the highest false-negative rates, highlighting the need for Graph Neural Networks that can explicitly model conformational flexibility and multi-hop atomic interactions.

---

## 6. Reproducibility and Audit Verification Verdict

- Automated test execution: **117 passed, 0 failed** in 9.48 seconds.
- Audit CLI script `src/audit/run_audit.py`: All 7 automated scientific validation checks executed and completed with status `PASS`.
- **Final Scientific Verdict:** The Phase 1 scientific implementation is mathematically rigorous, free of data leakage, reproducible, and fully verified.
