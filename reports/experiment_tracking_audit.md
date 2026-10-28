# Experiment Tracking and Benchmark Consistency Audit (Phase 1)

**Author:** Student 5 (Aditya Yadav) — research and reproducibility integration  
**Date:** 2026-10-28  
**Repository state audited:** `main` at commit `8a80819`  
**Environment used:** macOS (arm64), Python 3.12.0, packages from `requirements.txt` in a fresh `.venv`

This audit checks whether every reported Phase 1 number can be traced to a saved artifact and whether it can be regenerated. It changes **no** results, data, splits, features or evaluation code. Commands and outputs are recorded in [`experiment_reproducibility.md`](experiment_reproducibility.md).

---

## 1. Summary

| Check | Result |
| :--- | :--- |
| Committed metrics recompute from committed prediction CSVs | ✅ **Exact match** for all 9 configurations, using `sklearn.metrics` independently of the project's own metric code |
| Data pipeline (download → split) regenerates documented counts | ✅ 7,831 raw → 7,823 valid + 8 excluded; split 6,258 / 782 / 783; 0 scaffold overlap |
| Audit suite `python -m src.audit.run_audit` | ✅ Exit 0, `overall_status: passed`. ⚠️ Check 7 is stale (§4.3) |
| Test suite (122 claimed) | ✅ **122 passed** with `--run-real-data`; 121 passed + 1 skipped without it. ⚠️ Needs OpenMP on macOS (§4.4) |
| Retraining reproduces committed benchmark | ⚠️ **Partially.** XGBoost (3/3) and LR-descriptors are byte-identical. **RF (3/3) and LR-Morgan/combined differ** (§3) |
| Documentation agrees with artifacts | ⚠️ Several discrepancies (§4) |

**Verdict.** The committed headline numbers are internally consistent and traceable to saved predictions. They are **not exactly regenerable** in the currently pinned environment: the scikit-learn version used to produce them was never recorded. The best configuration (RF + combined) and the overall ranking of feature sets do not change.

---

## 2. Traceability of Reported Metrics

| Reported metric | Source artifact | Producing code | Verification |
| :--- | :--- | :--- | :--- |
| Macro ROC-AUC / AP ± SEM, SD, F1, BalAcc (9 configs) | `results/metrics/model_comparison_table.csv` | `src/evaluation/model_evaluation.py::run_full_model_benchmarks` | Recomputed from predictions: exact |
| Per-endpoint val/test metrics, thresholds, confusion matrices | `results/metrics/model_benchmark_summary.json` | same | Per-endpoint ROC/AP: max abs. difference 2.2e-16 |
| Test probabilities | `results/metrics/predictions_<model>_<features>.csv` (783 rows; `scaffold_test_row_idx` column) | same | Row index equals `split["test"]`. Probabilities are present exactly where a label exists and NaN where the label is missing |
| Endpoint label statistics | `results/metrics/endpoint_label_diagnostics.json` | `src/evaluation/endpoint_analysis.py` | Overall missing 17.06%; prevalence 2.89–16.17% |
| RF-combined error analysis and figures | `results/metrics/endpoint_error_analysis.json`, `results/figures/*.png` | `src/evaluation/error_analysis.py` | Per-endpoint values equal the summary JSON; mean ROC-AUC = 0.7488 |

**Naming.** "PR-AUC" throughout the repository is **Average Precision** (`compute_pr_auc(method="average_precision")`), confirmed by matching `sklearn.metrics.average_precision_score` exactly.

**Uncertainty.** "± SEM" is $\text{SD}/\sqrt{12}$ over the 12 endpoint scores of **one** run (seed 42); see `src/evaluation/aggregation.py`. It measures how much performance varies across endpoints. It is not a repeated-seed or resampling interval, and overlapping or non-overlapping SEMs cannot be read as a significance test between models.

**Train / validation / test separation.** Models are fitted on train rows with observed labels. Thresholds are chosen by maximising F1 on validation (`find_optimal_threshold`, grid 0.05–0.95). Test labels are used only for the final metrics. This is confirmed by reading `train_and_evaluate_endpoint` and by the passing test `tests/test_final_evaluation_regression.py::test_threshold_selection_test_label_independence`. The validation metrics stored in the summary JSON are labelled `validation` and are not reported as test results anywhere.

---

## 3. Re-run Discrepancy (Retraining in the Pinned Environment)

Command: `python -m src.evaluation.model_evaluation --output <scratch dir>`. The output was written outside the repository; committed files were not overwritten. Features and splits were regenerated from the raw download beforehand.

| Configuration | Committed ROC-AUC (± SEM) | Re-run ROC-AUC (± SEM) | Committed AP | Re-run AP | Prediction CSV |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LR · Morgan | 0.6987 ± 0.0149 | 0.6989 ± 0.0148 | 0.2602 | 0.2605 | differs |
| LR · Descriptors | 0.7203 ± 0.0169 | 0.7203 ± 0.0169 | 0.1968 | 0.1968 | identical |
| LR · Combined | 0.7138 ± 0.0158 | 0.7139 ± 0.0158 | 0.2341 | 0.2341 | differs |
| RF · Morgan | 0.7208 ± 0.0199 | **0.7289 ± 0.0188** | 0.3458 | 0.3440 | differs |
| RF · Descriptors | 0.7215 ± 0.0216 | **0.7223 ± 0.0207** | 0.2901 | 0.2989 | differs |
| **RF · Combined** | **0.7488 ± 0.0181** | **0.7625 ± 0.0196** | **0.3891** | **0.4046** | differs |
| XGB · Morgan | 0.7114 ± 0.0181 | 0.7114 ± 0.0181 | 0.3027 | 0.3027 | identical |
| XGB · Descriptors | 0.7024 ± 0.0256 | 0.7024 ± 0.0256 | 0.2911 | 0.2911 | identical |
| XGB · Combined | 0.7456 ± 0.0179 | 0.7456 ± 0.0179 | 0.3495 | 0.3495 | identical |

**Diagnosis (evidence-based):**
- **Inputs are identical.** XGBoost predictions are byte-identical, which would not happen if the features or split indices differed.
- **The evaluation code is unchanged.** `git diff 5671f09 HEAD` on `model_evaluation.py`, `metrics.py`, `aggregation.py` and `preprocessor.py` is empty.
- **The re-run is deterministic.** Two consecutive in-process RF-combined runs both gave 0.762477.
- The results were committed in `5671f09` (2026-10-26). `scikit-learn` and `xgboost` were first pinned the next day in `9bf28fe`. **The scikit-learn version that produced the committed RF/LR outputs was never recorded.** The likeliest explanation is a different scikit-learn version, since RF bootstrap sampling and the lbfgs solver's numerics can change between releases. This cannot be confirmed without the original environment.

**Recommendation (for Students 3 and 4 to decide; not applied here).** Either regenerate `results/metrics/` in the pinned environment and update every document that quotes RF/LR numbers, or keep the committed artifacts and disclose that they were produced under an unrecorded scikit-learn version. In both cases, record the package versions inside `model_benchmark_summary.json`.

---

## 4. Documentation vs. Artifact Discrepancies

Each item lists the old value, the verified value and the source. Items in files owned by Student 5 were corrected in this change; the others are reported only.

| # | Location | Old value | Verified value / source | Action |
| :--- | :--- | :--- | :--- | :--- |
| D1 | `README.md`, `reports/literature_review.md`, `reports/viva_preparation.md` (Q10), `reports/contributions/student5_aditya_yadav.md` | Prevalence "2.84% (SR-HSE) to 15.82% (NR-AhR)" | **2.89% (NR-PPAR-gamma) to 16.17% (SR-ARE)**; SR-HSE 5.76%, NR-AhR 11.74% (`endpoint_label_diagnostics.json`) | Corrected (Student 5 files) |
| D2 | `reports/contributions/student5_aditya_yadav.md` | SR-HSE "ROC-AUC ~0.66", "PR-AUC 0.1542" | SR-HSE test ROC-AUC **0.7738**, AP **0.3670** (RF-combined, `model_benchmark_summary.json`) | Corrected |
| D3 | same | Random splitting inflates ROC-AUC by "+0.10 to +0.15" | No random-split experiment exists in the repository | Removed |
| D4 | `reports/phase1_model_evaluation.md` §6.1 (Student 4) | Per-endpoint table: 9 of 12 rows do not match artifacts (e.g. NR-Aromatase threshold 0.41 / ROC 0.7289; SR-MMP 0.8252) | `endpoint_error_analysis.json` / summary JSON: NR-Aromatase 0.55 / 0.7584; SR-MMP 0.8458. Only NR-AR, NR-AR-LBD and NR-AhR rows match | **Reported only** |
| D5 | `reports/phase1_model_evaluation.md` §6.2 (Student 4) | NR-PPAR-gamma "only 12 positive test cases out of 641 observed, 1.87%" | Test confusion matrix: TP 9 + FN 19 = **28 positives of 579 observed (4.84%)** | **Reported only** |
| D6 | `reports/phase1_model_evaluation.md` §5 (Student 4) | Combined features give a "statistically significant performance boost" | No significance test was run; SEMs are across endpoints from one seed | **Reported only** |
| D7 | `src/audit/run_audit.py` check 7 → `results/experiment_logs/audit_results.json` | `"model_training_status": "not_executed_in_phase1"`, `"model_predictions_status": "not_available"` | Hard-coded and stale: 9 prediction CSVs are committed. The audit's `passed` verdict therefore does **not** cover model results | **Reported only** (code owner: Student 4) |
| D8 | `src/audit/run_audit.py:131`, `final_submission_checklist.md`, `folder_structure_audit.md` | `reports/student2_features_and_splitting.md`, `reports/ieee_reconciliation.md` | Neither file exists: `student2_features_and_splitting.md` was deleted in `9c485be` (content now in `reports/features_and_splitting.md`), and `ieee_reconciliation.md` was deleted in `f03e8ff` | **Reported only** |
| D9 | `README.md` | 8 links of the form `file:///Users/shouryapratap/...` | Broken for every other user and on GitHub | Corrected to relative links |
| D10 | `README.md` tree | "117 automated tests" | 122 collected and passed | Corrected |
| D11 | `README.md` tree | `results/figures/` and `results/metrics/` described as "Gitignored" | Both are **tracked** (`git ls-files results`); only `results/figures/eda/`, `results/experiment_logs/` and `results/models/` are ignored | Corrected |
| D12 | `README.md` | Reproduction steps omit the commands that produce `endpoint_label_diagnostics.json`, `endpoint_error_analysis.json` and the 3 evaluation figures | `src.evaluation.endpoint_analysis`, `src.evaluation.error_analysis` | Documented in `experiment_reproducibility.md`; README links to it |
| D14 | `README.md` | "Every module accepts `--config PATH`" | Only steps 1–4 do (shared parser in `src/data/common.py`). `build_features`, `split_data`, `model_evaluation` and `run_audit` have no `--config` flag | Corrected |
| D13 | `reports/ieee_phase1_manuscript.tex` | States that no baseline models exist; endpoint table uses raw (pre-cleaning) counts | Contradicted by committed artifacts | **Not edited** (manuscript is out of Student 5's scope for this task). Flagged for the authors |

### 4.3 Note on the audit suite
Checks 1–6 run real computations and passed on the regenerated data. Check 7 only reports hard-coded claims (D7), so the audit verdict should not be cited as evidence about the model benchmark.

### 4.4 Environment issue (macOS)
The pinned `xgboost==3.4.1` wheel needs the OpenMP runtime `libomp.dylib`. On a Mac without it, importing `xgboost` fails and pytest stops at collection with 3 errors. The fix is `brew install libomp`. For this audit, `DYLD_LIBRARY_PATH` was pointed at scikit-learn's bundled copy (`.venv/lib/python3.12/site-packages/sklearn/.dylibs`) instead; nothing was installed system-wide. Use `python -m pytest`, not the `pytest` launcher, so the variable reaches the interpreter.

---

## 5. Experiment Metadata Checklist

| Item | Recorded? | Where |
| :--- | :---: | :--- |
| Dataset URL and SHA-256 | ✅ | `configs/data_config.yaml`; `data/raw/download_provenance.json` (generated) |
| Preprocessing configuration (canonical, isomeric SMILES) | ✅ | `configs/data_config.yaml` |
| Split seed, ratios, acyclic policy | ✅ | `data/processed/scaffold_split_indices.json` (generated) |
| Feature parameters, shapes, SHA-256 | ✅ | `data/processed/features/feature_manifest.json` (generated) |
| Model hyperparameters | ⚠️ In code only | `create_classifier` in `model_evaluation.py`. Not written into the results JSON |
| Model random seed | ✅ | `random_state` in each summary JSON entry |
| Package versions used for committed results | ❌ | Not recorded. `requirements.txt` was pinned after the results were committed |
| Per-endpoint missing counts and single-class handling | ✅ | `endpoint_label_diagnostics.json`. Single-class endpoints return NaN and are excluded from the macro average (`aggregation.py`). None occurred in this split |
| Link from predictions to rows | ✅ | `scaffold_test_row_idx` column → row in `tox21_cleaned.csv` |
| Serialized models | ❌ (by design) | `results/models/` is gitignored; produced only by `src.evaluation.run_experiments` |
| Two runners | ⚠️ | `model_evaluation.py` (Student 4) and `run_experiments.py` (Student 3) use the same hyperparameters and write to the **same** `results/metrics/` filenames. The committed artifacts came from `model_evaluation.py` (commit `5671f09`) |

---

## 6. Limitations to Disclose in Any Write-up

1. Single seed (42), single scaffold split: there is no run-to-run uncertainty.
2. No hyperparameter tuning was performed; the hyperparameters are fixed values set in the code.
3. Validation and test contain only singleton scaffolds (largest-first greedy split), and all acyclic molecules are in train.
4. Results use a scaffold split and are not directly comparable to MoleculeNet's Tox21 benchmark, which uses a random split.
5. The committed RF/LR numbers are not bit-reproducible in the pinned environment (§3).
