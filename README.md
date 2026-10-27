# Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks

A B.Tech team project implementing a reproducible, scientifically rigorous machine learning pipeline for predicting molecular toxicity across 12 bioassay endpoints from the NIH Tox21 dataset.

---

## Project Overview

This project builds and evaluates classical machine learning baselines for multi-task molecular toxicity prediction, using handcrafted chemical representations (Morgan fingerprints and physicochemical descriptors). Phase 2 will extend the pipeline with molecular graph neural networks (GCN, GAT, GIN).

**Dataset:** Tox21 — 7,823 valid molecules, 12 toxicity endpoints (NR and SR pathways)  
**Task:** Multi-task binary classification (toxic / non-toxic) with extensive missing labels  
**Evaluation:** Scaffold-based out-of-distribution splits, ROC-AUC, PR-AUC, and macro uncertainty

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the complete environment (29 packages). Verified on Python 3.12.4, macOS arm64.

---

## Reproduce the Full Pipeline

Run commands in order from the repository root inside the virtual environment:

```bash
# 1. Download and verify Tox21 dataset (SHA-256 gated)
python -m src.data.download_data

# 2. Inspect raw schema, label encodings, and duplicates
python -m src.data.inspect_dataset

# 3. Sanitize SMILES, canonicalize molecules, produce cleaned CSV
python -m src.data.preprocess

# 4. Generate 10 EDA distribution and prevalence figures
python -m src.data.generate_eda_plots

# 5. Extract Morgan fingerprints, molecular descriptors, and combined matrix
python -m src.features.build_features

# 6. Create Bemis-Murcko scaffold-based train/val/test split (seed 42)
python -m src.data.split_data --seed 42

# 7. Run multi-task baseline benchmarks (LR, RF, XGBoost × 3 feature sets)
python -m src.evaluation.model_evaluation

# 8. Run the independent scientific audit suite
python -m src.audit.run_audit

# 9. Run the full automated test suite
pytest tests/ -v --run-real-data
```

Every module accepts `--config PATH`. Configuration lives in `configs/data_config.yaml`. All random seeds are recorded; no uncontrolled stochasticity is introduced.

---

## Repository Structure

```
aml-dl/
├── configs/                        # Centralized YAML configuration
├── data/
│   ├── README.md                   # Data provenance, checksums, and reproduction guide
│   ├── raw/                        # Gitignored: compressed Tox21 archive + provenance
│   └── processed/
│       └── features/               # Gitignored: numpy feature matrices and split indices
├── src/
│   ├── data/                       # Download, inspection, preprocessing, EDA, splitting
│   ├── features/                   # Morgan fingerprints, descriptors, combined matrix, scaler
│   ├── evaluation/                 # Metrics, aggregation, model training, error analysis
│   └── audit/                      # Independent dataset, split, and reproducibility auditors
├── tests/                          # 117 automated unit and regression tests
├── results/
│   ├── experiment_logs/            # Gitignored: audit JSON, data quality report, exclusions
│   ├── figures/                    # Gitignored: EDA plots and evaluation figures
│   └── metrics/                    # Gitignored: prediction CSVs and benchmark summary
└── reports/
    ├── literature_review.md
    ├── viva_preparation.md
    └── audit/                      # Full Phase 1 audit deliverables
```

---

## Key Scientific Policies

- **Label encoding:** `1.0` = toxic, `0.0` = non-toxic, empty = untested. Missing labels are **never imputed as negative**.
- **Scaffold splitting:** Bemis-Murcko scaffold assignment ensures zero scaffold overlap across train, validation, and test partitions. Acyclic molecules are confined to training.
- **Leakage prevention:** Feature scaling (mean/std) is fitted exclusively on the training partition and applied frozen to validation and test. Binary Morgan bits are not scaled.
- **Threshold tuning:** Decision thresholds are selected on the validation set to maximize F1-score and applied frozen to the test set. Test labels play no role in model selection.
- **Uncertainty:** All macro metrics are reported as `Mean ± SEM (K=12)` with explicit standard deviation. Endpoints that are single-class in a partition return `NaN` rather than false defaults.

---

## Dataset

| Property | Value |
| :--- | :--- |
| Source | [DeepChem / NIH Tox21](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz) |
| SHA-256 | `45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360` |
| Raw records | 7,831 rows × 14 columns |
| Valid molecules | 7,823 (8 organometallic structures excluded) |
| Endpoints | 12 binary toxicity assays (NR-AR, NR-AhR, SR-ARE, SR-MMP, …) |
| Positive prevalence | 2.84% (SR-HSE) to 15.82% (NR-AhR) |

---

## Features

| Representation | Shape | Dtype | Description |
| :--- | :---: | :---: | :--- |
| Morgan Fingerprints | (7823, 2048) | uint8 | Circular topological bits, radius=2 (ECFP4 equivalent) |
| Molecular Descriptors | (7823, 9) | float64 | MW, LogP, TPSA, HBD, HBA, rotatable bonds, ring count, HAC, Fsp³ |
| Combined Matrix | (7823, 2057) | float32 | Concatenated Morgan + descriptor representation |

---

## Scaffold Split (Seed 42)

| Partition | Molecules | Proportion | Unique Scaffolds |
| :--- | :---: | :---: | :---: |
| Train | 6,258 | 79.995% | 760 cyclic + acyclic group |
| Validation | 782 | 9.996% | 782 |
| Test | 783 | 10.009% | 783 |
| **Scaffold overlap** | — | — | **0** |

---

## Phase 1 Benchmark Results (Test Partition, Out-of-Sample)

All models evaluated on unseen scaffolds. Thresholds selected on validation only.

| Model | Feature Set | Macro ROC-AUC (Mean ± SEM) | ROC-AUC SD | Macro PR-AUC (Mean ± SEM) | Balanced Acc | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **Combined** | **0.7488 ± 0.0181** | 0.0628 | **0.3891 ± 0.0320** | **0.6636** | **0.3781** |
| XGBoost | Combined | 0.7456 ± 0.0179 | 0.0620 | 0.3495 ± 0.0406 | 0.6485 | 0.3591 |
| Random Forest | Morgan | 0.7208 ± 0.0199 | 0.0688 | 0.3458 ± 0.0316 | 0.6475 | 0.3712 |
| Random Forest | Descriptors | 0.7215 ± 0.0216 | 0.0747 | 0.2901 ± 0.0408 | 0.6337 | 0.3036 |
| Logistic Regression | Descriptors | 0.7203 ± 0.0169 | 0.0585 | 0.1968 ± 0.0397 | 0.6188 | 0.2472 |
| Logistic Regression | Combined | 0.7138 ± 0.0158 | 0.0546 | 0.2341 ± 0.0338 | 0.6404 | 0.3159 |
| XGBoost | Morgan | 0.7114 ± 0.0181 | 0.0627 | 0.3027 ± 0.0308 | 0.6340 | 0.3351 |
| XGBoost | Descriptors | 0.7024 ± 0.0256 | 0.0887 | 0.2911 ± 0.0454 | 0.6394 | 0.3133 |
| Logistic Regression | Morgan | 0.6987 ± 0.0149 | 0.0515 | 0.2602 ± 0.0341 | 0.6216 | 0.3001 |

Full per-endpoint results: `results/metrics/model_benchmark_summary.json`  
Prediction CSVs: `results/metrics/predictions_<model>_<features>.csv`

---

## Audit and Verification

```bash
# Run all 7 independent scientific checks
python -m src.audit.run_audit

# Run full test suite (117 tests, including real-data validation)
pytest tests/ -v --run-real-data
```

Phase 1 audit deliverables are in `reports/audit/`:

| Document | Description |
| :--- | :--- |
| `phase1_requirements_audit.md` | CSAI rubric compliance — 9/9 criteria PASS |
| `scientific_validation_audit.md` | Dataset, features, splits, metrics, and models deep-audit |
| `contribution_and_integration_audit.md` | Verified contribution accounting per student |
| `folder_structure_audit.md` | Annotated directory layout and change justifications |
| `issue_register.md` | Issue tracking — all Critical/High issues resolved |
| `final_submission_checklist.md` | Granular Phase 1 submission checklist |
| `final_readiness_report.md` | Executive readiness declaration |

---

## Reports

| Document | Description |
| :--- | :--- |
| `reports/literature_review.md` | Academic literature review: QSAR, fingerprints, scaffold splitting, GNNs (15 citations) |
| `reports/viva_preparation.md` | Technical exam preparation: 13 answered questions with mathematical derivations |
| `reports/phase1_model_evaluation.md` | Detailed Phase 1 model evaluation technical report |
| `reports/dataset_eda_report.md` | Dataset quality and EDA measurement report |

---

## Team Roles and Responsibilities

### Student 1 — Dataset, Preprocessing and EDA
**Data owner**

**Responsibilities**
- Download and document the Tox21 dataset.
- Validate SMILES, canonicalize molecules and handle duplicates.
- Audit missing toxicity labels and class imbalance.
- Generate dataset statistics and EDA plots.
- Document invalid structures and exclusions.

**Deliverables**
- `src/data/`
- `results/experiment_logs/data_quality_report.json`
- `results/figures/eda/`
- Dataset and EDA sections of the IEEE report.

**Viva preparation:** Tox21 endpoints, missing labels, SMILES, canonicalization and class imbalance.

---

### Student 2 — Molecular Features and Split Integrity
**Feature owner**

**Responsibilities**
- Implement Morgan fingerprints (radius 2, 2,048 bits).
- Implement the nine molecular descriptors used in the project.
- Generate combined fingerprint-plus-descriptor features.
- Create scaffold-based train/validation/test splits.
- Verify that scaffolds and row indices do not overlap across partitions.

**Deliverables**
- `src/features/fingerprints.py`
- `src/features/descriptors.py`
- `src/data/split_data.py`
- `data/processed/scaffold_split_indices.json`

**Viva preparation:** Molecular fingerprints, descriptors, Bemis–Murcko scaffolds and data leakage.

---

### Student 3 — Classical ML Models and Experiments
**Model owner**

**Responsibilities**
- Implement Logistic Regression, Random Forest and XGBoost.
- Train separate classifiers for the 12 toxicity endpoints.
- Apply appropriate class-imbalance handling.
- Conduct model and feature ablations.
- Keep test-set labels out of model selection and hyperparameter tuning.

**Deliverables**
- `src/models/ml/`
- `src/evaluation/run_experiments.py` (model execution component)
- Saved model configurations and experiment outputs.

**Viva preparation:** Model assumptions, class weighting, overfitting and why the models are suitable baselines.

---

### Student 4 — Evaluation, Statistics and Final Audit
**Validation owner**

**Responsibilities**
- Implement ROC-AUC, PR-AUC/average precision, precision, recall and F1.
- Handle missing labels and endpoints without both classes.
- Tune decision thresholds using validation data only.
- Recalculate macro metrics and clarify the meaning of reported uncertainty.
- Audit reproducibility, split integrity, test-set independence and leakage.
- Verify reported results against saved experiment artifacts.

**Deliverables**
- `src/evaluation/metrics.py`
- `src/evaluation/error_analysis.py`
- `tests/` audit tests
- `results/metrics/` and final audit checklist.

**Viva preparation:** ROC-AUC vs PR-AUC, macro averaging, uncertainty, scaffold generalization and evaluation leakage.

---

### Student 5 — Literature Review, IEEE Report and Presentation
**Documentation owner**

**Responsibilities**
- Review relevant molecular toxicity prediction research.
- Maintain accurate citations and the BibTeX bibliography.
- Assemble the IEEE paper using results verified by Students 1–4.
- Create presentation slides, comparison tables and result figures.
- Maintain README, setup instructions and contribution records.
- Coordinate the presentation and shared viva preparation.

**Deliverables**
- `reports/literature_review.md`
- `reports/ieee_report/`
- `reports/presentation.md` or final slide deck
- `README.md` and `reports/viva_preparation.md`

**Viva preparation:** Research gap, related work, methodology, limitations and interpretation of results.
