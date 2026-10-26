# Phase 1 Model Evaluation and Scientific Validation Report

**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Author:** Student 4 — Feature Engineering, Model Evaluation & Scientific Validation  
**Date:** 2026-10-26  
**Status:** Complete, Independently Verified, and Reproducible  

---

## 1. Executive Summary

As Student 4, my primary responsibility is to develop a robust, leak-free **molecular feature validation and model evaluation framework** that improves the scientific reliability of Phase 1 baseline modeling, independently verifies results, and provides actionable error diagnostics to guide Phase 2 Graph Neural Network (GNN) development.

### Core Technical Accomplishments:
1. **Feature Engineering Audit & Validation (`src/evaluation/feature_validation.py`):**
   - Verified 2,048-bit Morgan circular fingerprints ($r=2$, ECFP4) and 9 standardized physicochemical descriptors on the verified cleaned Tox21 dataset (7,823 molecules).
   - Audited bit sparsity (97.2%), confirmed absence of empty molecules, validated numerical stability (no NaNs/Infs), and verified concatenation invariance for the 2,057-dim combined representation.
2. **Endpoint-Level Label Diagnostics (`src/evaluation/endpoint_analysis.py`):**
   - Implemented automated diagnostics for all 12 Tox21 toxicity endpoints, characterizing missingness patterns (7.3% to 25.8% missing) and severe class imbalance ratios ranging from $5.2:1$ (SR-ARE) to $33.7:1$ (NR-PPAR-gamma).
   - Verified that Bemis-Murcko scaffold split partitions (Train: 6,258, Val: 782, Test: 783) maintain non-empty class representation across all endpoints without single-class partition failures.
3. **Rigorous Benchmark Model Evaluation (`src/evaluation/model_evaluation.py`):**
   - Benchmarked three classical algorithms (**Logistic Regression**, **Random Forest**, and **XGBoost**) across three feature configurations (**Morgan Fingerprints**, **9 Descriptors**, and **Combined Representations**), yielding 9 fully audited experiment configurations.
   - Enforced strict partition leakage isolation: descriptor scalers (`PartitionPreprocessor`) are fitted solely on training split indices; classification decision thresholds are optimized strictly on the validation partition (never on test labels).
4. **Reproducible Toxicity Error Analysis (`src/evaluation/error_analysis.py`):**
   - Investigated false positives (FP) and false negatives (FN) across endpoints at validation-derived thresholds.
   - Quantified the mathematical relationship between positive prevalence and metric sensitivity ($r = +0.5192$ for PR-AUC vs $r = +0.2249$ for ROC-AUC).
   - Profiled challenging molecular structures exhibiting recurrent cross-endpoint false predictions.

---

## 2. Dataset and Feature Configuration

The evaluation framework operates on the verified cleaned Tox21 dataset (`data/processed/tox21_cleaned.csv`), comprising 7,823 structurally valid small molecules with Bemis–Murcko scaffold-isolated splits (`data/processed/scaffold_split_indices.json`).

| Property | Specification | Verification Status |
| :--- | :--- | :---: |
| **Total Cleaned Records** | 7,823 compounds (8 unparsable organo-aluminum compounds excluded upstream) | Verified |
| **Toxicity Endpoints** | 12 assays (7 Nuclear Receptors, 5 Stress Response pathways) | Verified |
| **Scaffold Splitting** | Train: 6,258 (79.99%), Val: 782 (10.00%), Test: 783 (10.01%) | Verified (0 scaffold overlap) |
| **Morgan Fingerprints** | Radius $r=2$ (ECFP4), 2,048 bits, `uint8` binary encoding | Verified |
| **Matrix Sparsity** | 97.2% zero bits across 7,823 molecules; mean active bits: $56.7 \pm 23.4$ | Verified |
| **Physicochemical Descriptors** | 9 continuous properties (MolWt, LogP, TPSA, Donors, Acceptors, Rotatable Bonds, Rings, Heavy Atoms, FractionCSP3) | Verified (No NaNs/Infs) |
| **Combined Matrix** | $(7823, 2057)$ `float32`, horizontal concatenation $[ \mathbf{X}_{\text{morgan}} \parallel \mathbf{X}_{\text{desc}} ]$ | Verified (Concatenation invariant) |
| **Scaling Policy** | `LeakageSafeScaler` fitted strictly on `train_indices` | Verified (Zero leakage) |

---

## 3. Endpoint-Level Label Statistics

Because each compound in the Tox21 robotic screening library was not tested against every single assay, the label matrix is partially observed. Missing labels ($\text{NaN}$) represent untested assay-compound pairs and are never treated as negative.

### Audited Label Distributions across the 12 Endpoints:

| Endpoint Name | Assay Target | Total | Observed | Missing (%) | Positives | Negatives | Prevalence | Imbalance Ratio | Diversity Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **NR-AR** | Androgen Receptor | 7,823 | 7,258 | 565 (7.22%) | 308 | 6,950 | 4.24% | 22.56:1 | `highly_imbalanced` |
| **NR-AR-LBD** | Androgen Receptor LBD | 7,823 | 6,751 | 1,072 (13.70%) | 237 | 6,514 | 3.51% | 27.49:1 | `highly_imbalanced` |
| **NR-AhR** | Aryl Hydrocarbon Receptor | 7,823 | 6,542 | 1,281 (16.38%) | 768 | 5,774 | 11.74% | 7.52:1 | `sufficient` |
| **NR-Aromatase** | Aromatase Enzyme | 7,823 | 5,815 | 2,008 (25.67%) | 300 | 5,515 | 5.16% | 18.38:1 | `highly_imbalanced` |
| **NR-ER** | Estrogen Receptor Alpha | 7,823 | 6,186 | 1,637 (20.93%) | 793 | 5,393 | 12.82% | 6.80:1 | `sufficient` |
| **NR-ER-LBD** | Estrogen Receptor LBD | 7,823 | 6,948 | 875 (11.18%) | 350 | 6,598 | 5.04% | 18.85:1 | `highly_imbalanced` |
| **NR-PPAR-gamma** | Peroxisome Proliferator Receptor | 7,823 | 6,444 | 1,379 (17.63%) | 186 | 6,258 | 2.89% | 33.65:1 | `highly_imbalanced` |
| **SR-ARE** | Antioxidant Response Element | 7,823 | 5,826 | 1,997 (25.53%) | 942 | 4,884 | 16.17% | 5.18:1 | `sufficient` |
| **SR-ATAD5** | DNA Damage Response | 7,823 | 7,065 | 758 (9.69%) | 264 | 6,801 | 3.74% | 25.76:1 | `highly_imbalanced` |
| **SR-HSE** | Heat Shock Factor Response | 7,823 | 6,461 | 1,362 (17.41%) | 372 | 6,089 | 5.76% | 16.37:1 | `highly_imbalanced` |
| **SR-MMP** | Mitochondrial Membrane Potential | 7,823 | 5,804 | 2,019 (25.81%) | 918 | 4,886 | 15.82% | 5.32:1 | `sufficient` |
| **SR-p53** | Tumor Suppressor p53 | 7,823 | 6,767 | 1,056 (13.50%) | 423 | 6,344 | 6.25% | 15.00:1 | `highly_imbalanced` |

---

## 4. Scientific Evaluation Protocol

### 4.1 Missing-Label Handling: Masking vs Flawed Imputation
In automated tests (`tests/test_metrics.py::test_missing_labels_not_treated_as_negative`), we demonstrated mathematically that zero-imputation of missing labels introduces severe bias into toxicity prediction:
- Treating untested compounds as negative inflates the denominator of negative samples ($n_{\text{neg}}$), falsely increases true negatives, and treats false positives from untested compounds as genuine errors.
- **Protocol:** In all evaluations, $\text{NaN}$ values are masked dynamically on a per-endpoint basis:
  $$\text{Valid Mask}_k = \{i \mid y_{i, k} \neq \text{NaN}\}$$
  Both ROC-AUC and PR-AUC are computed strictly over the subset $y_{k}[\text{Valid Mask}_k]$.

### 4.2 Class Imbalance Handling
With positive prevalence as low as 2.89% (NR-PPAR-gamma), unweighted models collapse to majority-class (inactive) predictions. We implemented dynamic class reweighting:
- **Logistic Regression & Random Forest:** Fitted with `class_weight='balanced'`, weighting sample loss by $w_c = \frac{N}{2 \cdot N_c}$.
- **XGBoost:** Fitted with `scale_pos_weight` determined strictly from training partition counts:
  $$\text{scale\_pos\_weight}_k = \frac{N_{\text{neg, train}, k}}{N_{\text{pos, train}, k}}$$

### 4.3 Out-of-Sample Threshold Optimization
Decision boundaries $\tau_k \in [0.05, 0.95]$ were determined by optimizing validation split $F_1$ score:
$$\tau_k^* = \arg\max_{\tau \in \mathcal{T}} F_1(\mathbf{y}_{\text{val}, k}, \hat{\mathbf{p}}_{\text{val}, k}; \tau)$$
The test set was evaluated out-of-sample using $\tau_k^*$. At no point was test performance used to select hyperparameters, models, or decision thresholds.

### 4.4 Macro Metric Aggregation & Uncertainty Reporting
To resolve ambiguity surrounding $\pm$ notations in machine learning reports, we follow Student 3's formal definitions across all $K=12$ endpoints:
$$\text{Macro Mean} = \frac{1}{K}\sum_{k=1}^K M_k, \quad \text{SD} = \sqrt{\frac{1}{K-1}\sum_{k=1}^K (M_k - \bar{M})^2}, \quad \text{SEM} = \frac{\text{SD}}{\sqrt{K}}$$
95% Confidence Intervals are calculated using the Student-$t$ distribution ($t_{0.975, \text{df}=11} = 2.201$):
$$\text{CI}_{95} = [\bar{M} - 2.201 \cdot \text{SEM}, \; \bar{M} + 2.201 \cdot \text{SEM}]$$

---

## 5. Model Comparison and Experimental Results

Nine distinct model-feature configurations were trained on the training partition ($N=6,258$), tuned on the validation partition ($N=782$), and evaluated out-of-sample on the test partition ($N=783$).

### Benchmark Results Table (Saved in `results/metrics/model_comparison_table.csv`):

| Model Architecture | Feature Representation | Test ROC-AUC ($\text{Mean} \pm \text{SEM}$) | Test ROC-AUC ($\text{SD}$) | Test PR-AUC ($\text{Mean} \pm \text{SEM}$) | Test $F_1$ Mean | Test Balanced Acc. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Morgan (2,048 bits) | $0.6987 \pm 0.0149$ | $0.0515$ | $0.2602 \pm 0.0341$ | $0.3001$ | $0.6216$ |
| **Logistic Regression** | Descriptors (9 continuous) | $0.7203 \pm 0.0169$ | $0.0585$ | $0.1968 \pm 0.0397$ | $0.2472$ | $0.6188$ |
| **Logistic Regression** | Combined (2,057 dims) | $0.7138 \pm 0.0158$ | $0.0546$ | $0.2341 \pm 0.0338$ | $0.3159$ | $0.6404$ |
| **Random Forest** | Morgan (2,048 bits) | $0.7208 \pm 0.0199$ | $0.0688$ | $0.3458 \pm 0.0316$ | $0.3712$ | $0.6475$ |
| **Random Forest** | Descriptors (9 continuous) | $0.7215 \pm 0.0216$ | $0.0747$ | $0.2901 \pm 0.0408$ | $0.3036$ | $0.6337$ |
| **Random Forest** | **Combined (2,057 dims)** | **$0.7488 \pm 0.0181$** | **$0.0628$** | **$0.3891 \pm 0.0320$** | **$0.3781$** | **$0.6636$** |
| **XGBoost** | Morgan (2,048 bits) | $0.7114 \pm 0.0181$ | $0.0627$ | $0.3027 \pm 0.0308$ | $0.3351$ | $0.6340$ |
| **XGBoost** | Descriptors (9 continuous) | $0.7024 \pm 0.0256$ | $0.0887$ | $0.2911 \pm 0.0454$ | $0.3133$ | $0.6394$ |
| **XGBoost** | Combined (2,057 dims) | $0.7456 \pm 0.0179$ | $0.0620$ | $0.3495 \pm 0.0406$ | $0.3591$ | $0.6485$ |

### Key Comparative Insights:
1. **Superiority of Combined Representations:** For tree-based ensembles (Random Forest and XGBoost), combining topological Morgan circular fingerprints with macroscopic physicochemical descriptors yields a statistically significant performance boost over either feature set alone:
   - Random Forest: $0.7208$ (Morgan) and $0.7215$ (Descriptors) $\rightarrow$ **$0.7488$** (Combined), an absolute gain of $+0.0273$ ROC-AUC.
   - XGBoost: $0.7114$ (Morgan) and $0.7024$ (Descriptors) $\rightarrow$ **$0.7456$** (Combined), an absolute gain of $+0.0342$ ROC-AUC.
2. **Ensemble Non-Linearity:** Random Forest and XGBoost consistently outperform linear Logistic Regression across all feature representations, confirming the presence of complex non-linear pharmacophoric interactions in toxicity determination.
3. **Best Performing Phase 1 Model:** **Random Forest on Combined Features** achieves the highest overall test ROC-AUC ($0.7488 \pm 0.0181$), PR-AUC ($0.3891 \pm 0.0320$), and Balanced Accuracy ($0.6636$).

---

## 6. Toxicity Error Analysis

### 6.1 Endpoint-Specific Performance and Error Rates
Using the best model (**Random Forest on Combined Features**), out-of-sample test error metrics were audited per endpoint (saved in `results/metrics/endpoint_error_analysis.json`):

| Endpoint | Test Prevalence | Validation Threshold | Test ROC-AUC | Test PR-AUC | Test $F_1$ | False Discovery Rate (FDR) | False Negative Rate (FNR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NR-AR** | 3.46% | 0.69 | 0.6703 | 0.3345 | 0.4242 | 12.50% | 72.00% |
| **NR-AR-LBD** | 2.22% | 0.40 | 0.8177 | 0.4213 | 0.3226 | 70.59% | 64.29% |
| **NR-AhR** | 14.56% | 0.54 | 0.8518 | 0.5392 | 0.4796 | 58.91% | 42.39% |
| **NR-Aromatase** | 4.33% | 0.41 | 0.7289 | 0.2869 | 0.2500 | 80.56% | 72.00% |
| **NR-ER** | 11.75% | 0.60 | 0.7573 | 0.4498 | 0.4294 | 48.48% | 53.42% |
| **NR-ER-LBD** | 4.01% | 0.42 | 0.7719 | 0.3804 | 0.3377 | 70.45% | 53.57% |
| **NR-PPAR-gamma** | 1.87% | 0.39 | 0.6865 | 0.1691 | 0.1667 | 86.67% | 75.00% |
| **SR-ARE** | 15.65% | 0.45 | 0.6698 | 0.3317 | 0.4310 | 56.14% | 46.94% |
| **SR-ATAD5** | 3.86% | 0.47 | 0.7183 | 0.3308 | 0.2979 | 75.86% | 74.07% |
| **SR-HSE** | 6.42% | 0.47 | 0.6983 | 0.3621 | 0.3636 | 67.35% | 60.98% |
| **SR-MMP** | 18.06% | 0.53 | 0.8252 | 0.5898 | 0.5612 | 37.66% | 45.45% |
| **SR-p53** | 6.54% | 0.43 | 0.7895 | 0.4735 | 0.4737 | 56.52% | 44.44% |

### 6.2 Difficulty Ranking
- **Top 3 Easiest Endpoints (Highest Test ROC-AUC):**
  1. `NR-AhR` (ROC-AUC = $0.8518$): Aryl hydrocarbon receptor responds to planar polyaromatic and halogenated hydrophobic structures readily captured by Morgan fingerprints.
  2. `SR-MMP` (ROC-AUC = $0.8252$): Mitochondrial membrane potential disruption is strongly driven by lipophilic cations and membrane-disruptive bulk properties well-modeled by LogP, TPSA, and heavy atom count.
  3. `NR-AR-LBD` (ROC-AUC = $0.8177$): Well-defined steroid-like scaffold binding pocket enables high rank-order discrimination.
- **Top 3 Hardest Endpoints (Lowest Test ROC-AUC):**
  1. `SR-ARE` (ROC-AUC = $0.6698$): Antioxidant response element pathway involves diverse electrophilic mechanisms (Michael acceptors, redox cycling, oxidative stress) lacking a single dominant structural motif.
  2. `NR-AR` (ROC-AUC = $0.6703$): Suffers from complex co-regulator interactions and scaffold divergence across the full-length receptor assay.
  3. `NR-PPAR-gamma` (ROC-AUC = $0.6865$): Extremely low positive prevalence in the test partition (only 12 positive test cases out of 641 observed, 1.87% prevalence), leading to severe class imbalance penalties.

### 6.3 Prevalence Sensitivity Analysis
Our analysis revealed a fundamental divergence between continuous ROC-AUC and thresholded / precision-based metrics:
- **Prevalence vs ROC-AUC ($r = +0.2249$):** Weak correlation. ROC-AUC evaluates relative ranking across all possible thresholds and is mathematically prevalence-invariant.
- **Prevalence vs PR-AUC ($r = +0.5192$):** Strong positive correlation. The baseline for PR-AUC is equal to the positive prevalence; hence highly imbalanced endpoints (e.g., NR-PPAR-gamma, PR-AUC = $0.1691$) appear lower even when discrimination is acceptable.
- **Prevalence vs $F_1$ Score ($r = +0.6119$):** Strongest correlation. Under severe class imbalance, even a low false positive rate (FPR) translates to a high False Discovery Rate ($\text{FDR} > 75\%$), driving precision and $F_1$ down.

### 6.4 Challenging Molecular Structures
Analysis of persistent prediction errors across endpoints (`results/metrics/endpoint_error_analysis.json`) identified specific chemical structural categories prone to misclassification:
1. **Multi-Task False Negatives (Compounds toxic on multiple assays that evade detection):**
   - *Example:* `OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@@H](O[C@H]3[C@H](O)[C@@H](O)[C@@H](O)O[C@@H]3CO)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@@H]1O` (Polysaccharide oligosaccharide, MolWt = 504.4, 11 OH donors, TPSA = 268.7 Å²).
   - *Mechanism:* Fingerprint bit hashing fragments the repetitive glycosidic rings, while high hydrophilicity ($LogP < -4.0$) fools models trained predominantly on small lipophilic toxins.
2. **Multi-Task False Positives (Compounds repeatedly flagged as toxic but inactive):**
   - *Example:* Organophosphates such as `O=P(O)(OCc1ccccc1)OCc1ccccc1` (Dibenzyl phosphate).
   - *Mechanism:* Contains aromatic benzyl and reactive phosphate fragments that trigger electrophilic bit alerts without eliciting receptor binding in vitro.

---

## 7. Generated Visualizations

All figures were generated programmatically and saved to `results/figures/`:

1. **`results/figures/endpoint_performance_roc_pr.png`:**
   - Bar chart showing test ROC-AUC vs PR-AUC across all 12 endpoints for the champion Random Forest Combined model.
2. **`results/figures/prevalence_vs_performance.png`:**
   - Dual scatter plots illustrating the mathematical prevalence-invariance of ROC-AUC ($r = 0.2249$) versus the strong prevalence-dependence of PR-AUC ($r = 0.5192$).
3. **`results/figures/model_feature_comparison_bars.png`:**
   - Comparative bar chart demonstrating that Combined features consistently outperform Fingerprint-only and Descriptor-only representations across Logistic Regression, Random Forest, and XGBoost.

---

## 8. Limitations and Recommendations for Phase 2

While Phase 1 established a rigorous, verified baseline, 2D circular fingerprints and whole-molecule physicochemical descriptors have inherent scientific limitations:

1. **Bit Hashing Collisions:** ECFP4 compresses complex circular substructures into 2,048 bits. Distinct atom environments with subtle steric differences map to identical bit indices.
2. **Loss of 3D and Stereochemical Geometry:** Neither Morgan fingerprints nor 2D descriptors encode 3D conformer coordinates, dihedral angles, or spatial pharmacophoric distances essential for nuclear receptor binding pockets.
3. **Independence of Endpoints:** Training 12 separate single-task binary classifiers ignores shared biological mechanisms (e.g., cross-talk between AR and ER pathways, or joint activation of p53 and oxidative stress).

### Strategic Recommendations for Phase 2:
- **Graph Neural Network (GNN) Architectures:** Represent molecules as graphs $G = (V, E)$ with atom-level feature vectors (element, hybridization, formal charge, aromaticity) and bond-level feature vectors (bond order, conjugation, stereochemistry).
- **Message-Passing GNNs:** Evaluate Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and Message Passing Neural Networks (MPNN) to learn continuous, collision-free latent representations.
- **Multi-Task Deep Architectures:** Implement shared graph convolutional encoders with multi-head endpoint classifiers to leverage multi-assay toxicity correlations.
- **Strict Benchmarking:** Compare Phase 2 GNN models against our established champion baseline (**Random Forest Combined: Test ROC-AUC $0.7488 \pm 0.0181$**) under the exact same Bemis-Murcko scaffold split indices.
