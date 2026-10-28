# Research-to-Implementation Traceability Matrix (Phase 1)

**Author:** Student 5 (Aditya Yadav)  
**Date:** 2026-10-28, commit `8a80819`  
**References:** keys from [`ieee_report/references.bib`](ieee_report/references.bib)

Each row links a methodological rationale to its source, the code that implements it, and the evidence that it works as stated. "Verified" means this was checked on 2026-10-28 by reading the code **and** running a test or regenerating the artifact. See [`experiment_reproducibility.md`](experiment_reproducibility.md).

Status key: ✅ verified · 🟡 partially verified · ❌ unresolved

| # | Rationale / research question | Reference | Implementation | Test / artifact evidence | Status | Limitation to disclose |
| :-: | :--- | :--- | :--- | :--- | :-: | :--- |
| 1 | Use Tox21 (12 NR/SR assays) as a public toxicity benchmark | `huang2016tox21`, `wu2018moleculenet` | `src/data/download_data.py`, `configs/data_config.yaml` | SHA-256 matched on fresh download; `tests/test_download_data.py` | ✅ | DeepChem CSV has 7,831 rows; MoleculeNet text reports 8,014 compounds for the original release |
| 2 | Sanitise structures; exclude unparsable molecules rather than repair them | `rdkit` | `src/data/preprocess.py`, `src/data/common.py` | 8 exclusions (Al valence) regenerated; audit check 2; `tests/test_preprocess.py` | ✅ | Organometallic compounds excluded |
| 3 | Canonical isomeric SMILES and duplicate audit | `rdkit` | `src/data/common.py` | 0 duplicate groups (`duplicates.csv` empty); audit check 3 | ✅ | — |
| 4 | Morgan/ECFP4: **radius 2, 2,048 bits** | `rogers2010ecfp`, `morgan1965generation` | `src/features/fingerprints.py` (`GetMorganGenerator(radius=2, fpSize=2048)`) | `feature_manifest.json`: shape (7823, 2048), uint8, radius 2; `tests/test_fingerprints.py` | ✅ | Hash folding causes collisions; chirality not included (RDKit default) |
| 5 | **Nine** physicochemical descriptors as documented | `wildman1999logp`, `ertl2000tpsa`, `lipinski1997rule`, `lovering2009flatland`, `todeschini2009descriptors` | `src/features/descriptors.py` (`DESCRIPTOR_NAMES`, 9 entries) | Manifest lists the same 9 names, shape (7823, 9); `tests/test_descriptors.py` | ✅ | Global scalars only |
| 6 | Combined representation | — (project design) | `src/features/build_features.py` | (7823, 2057) float32; `tests/test_build_features.py` | ✅ | — |
| 7 | Scaffold split for out-of-distribution evaluation; **no scaffold overlap** | `bemis1996frameworks`, `wu2018moleculenet` | `src/data/split_data.py` | Regenerated 6,258/782/783; audit check 5 (isolation, disjointness); `tests/test_split_integrity.py`, `tests/test_audit_split.py` | ✅ | Val/test hold only singleton scaffolds; acyclic molecules all in train; one seed; not comparable to MoleculeNet's random-split Tox21 |
| 8 | **Missing labels excluded** per endpoint, never imputed | `huang2016tox21` (sparse assay matrix) | `model_evaluation.py::train_and_evaluate_endpoint` (train mask); `metrics.py` (`valid_mask`) | `tests/test_metrics.py::test_missing_labels_not_treated_as_negative`; prediction CSVs are NaN exactly where labels are missing | ✅ | — |
| 9 | Handle class imbalance | `saito2015precision` (evaluation side) | `class_weight='balanced'` (LR, RF); `scale_pos_weight` from training counts (XGB) | Code read; `tests/test_classical_models.py` | ✅ | No resampling or tuning of the weights |
| 10 | **Preprocessing fitted on training data only** | `kapoor2023leakage` | `src/features/preprocessor.py` (`LeakageSafeScaler`, `PartitionPreprocessor`) | `tests/test_preprocessor_leakage.py`; audit check 6 | ✅ | — |
| 11 | **Thresholds chosen on validation only** | `kapoor2023leakage` | `metrics.py::find_optimal_threshold`, called on validation in `model_evaluation.py` | `tests/test_final_evaluation_regression.py::test_threshold_selection_test_label_independence` | ✅ | An F1-optimal threshold picked from a validation set with few positives can be noisy |
| 12 | Baselines LR / RF / XGBoost | `breiman2001random`, `chen2016xgboost` | `model_evaluation.py::create_classifier`; mirrored in `src/models/ml/` | `tests/test_model_evaluation.py`, `tests/test_classical_models.py` | ✅ | Hyperparameters fixed, not tuned |
| 13 | ROC-AUC as primary metric | `huang2016tox21`, `wu2018moleculenet` | `metrics.py::compute_roc_auc` | Matches `sklearn.metrics.roc_auc_score` exactly on all 108 endpoint × config pairs | ✅ | Prevalence-insensitive; can look good under heavy imbalance |
| 14 | PR-AUC = **Average Precision** under imbalance | `saito2015precision`, `davis2006relationship` | `metrics.py::compute_pr_auc(method="average_precision")` | Matches `average_precision_score` exactly | ✅ | Baseline equals prevalence, so compare against it |
| 15 | Macro aggregation with an explicitly defined uncertainty | — (statistics) | `src/evaluation/aggregation.py` | `tests/test_aggregation.py`; recomputed SEM/SD match the table | ✅ | SEM is **across endpoints, one run**, not repeated-seed uncertainty |
| 16 | **Documented metrics match saved outputs** | — | `results/metrics/*` | Recomputed from predictions: exact. Retrained: RF/LR differ (RF-combined 0.7625 vs 0.7488) | 🟡 | Package version behind committed results unrecorded. See `experiment_tracking_audit.md` §3 |
| 17 | Audit suite covers model results | — | `src/audit/run_audit.py` | Check 7 is hard-coded and says models were not run | ❌ | Audit "passed" does not cover the benchmark |
| 18 | Phase 2 GNN motivation | `kipf2017gcn`, `velickovic2018gat`, `xu2019gin`, `yang2019analyzing` | Not implemented (out of scope) | — | n/a | Motivation only |

## Scientific methods unchanged
No method was altered to align with the literature. Discrepancies are reported in [`experiment_tracking_audit.md`](experiment_tracking_audit.md) §4.
