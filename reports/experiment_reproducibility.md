# Phase 1 Experiment Reproducibility Guide

**Author:** Student 5 (Aditya Yadav)  
**Verified on:** 2026-10-28, macOS arm64, Python 3.12.0, commit `8a80819`  
**Companion audit:** [`experiment_tracking_audit.md`](experiment_tracking_audit.md)

This guide lists the exact commands that reproduce Phase 1 from a clean clone, what each one produces, and what happened when each was run. Every result below comes from an actual run. Nothing here is a predicted or illustrative value.

---

## 1. Environment

```bash
python3.12 -m venv .venv          # README states 3.12.4; 3.12.0 also worked
source .venv/bin/activate
pip install -r requirements.txt   # 29 pinned packages
```

**macOS only: install OpenMP for XGBoost.** `xgboost==3.4.1` needs `libomp.dylib`. Without it, `import xgboost` fails and three test modules error at collection.

```bash
brew install libomp
```

If you cannot install it, scikit-learn ships its own copy, which you can use for the current shell session:

```bash
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs"
# then use `python -m pytest`, not `pytest`, so the variable is inherited
```

The system default Python 3.14 was **not** tested; some pinned wheels may not exist for it.

---

## 2. Pipeline Commands and Verified Outcomes

Run from the repository root with the virtual environment active.

| # | Command | Writes | Verified outcome (2026-10-28) |
| :-: | :--- | :--- | :--- |
| 1 | `python -m src.data.download_data` | `data/raw/tox21.csv.gz`, provenance JSON | 122,925 bytes; SHA-256 `45d09792…0c0d360` matches config |
| 2 | `python -m src.data.inspect_dataset` | `results/experiment_logs/data_quality_report.json` | Status `audited`; 7,831 raw rows |
| 3 | `python -m src.data.preprocess` | `data/processed/tox21_cleaned.csv`, manifest, exclusions | 7,823 valid, 8 excluded (Al valence errors), 0 duplicate groups |
| 4 | `python -m src.data.generate_eda_plots` | `results/figures/eda/`, **`reports/dataset_eda_report.md` (tracked)** | 10 figures created. ⚠️ This step overwrites a tracked report; see note below |
| 5 | `python -m src.features.build_features` | `data/processed/features/*.npy`, `feature_manifest.json` | Morgan (7823, 2048) uint8, r=2; descriptors (7823, 9); combined (7823, 2057) |
| 6 | `python -m src.data.split_data --seed 42` | `data/processed/scaffold_split_indices.json` | Train 6,258 / 760 scaffolds; Val 782 / 782; Test 783 / 783 |
| 7 | `python -m src.evaluation.model_evaluation` | **`results/metrics/` (tracked)** | Ran with `--output <scratch dir>` to avoid overwriting committed files. Outcome in §4 |
| 7b | `python -m src.evaluation.endpoint_analysis` | `results/metrics/endpoint_label_diagnostics.json` | Not re-run (tracked artifact checked by recomputation instead) |
| 7c | `python -m src.evaluation.error_analysis` | `results/metrics/endpoint_error_analysis.json`, `results/figures/*.png` | Not re-run (artifact values checked against summary JSON) |
| 8 | `python -m src.audit.run_audit` | `results/experiment_logs/audit_results.json` | Exit 0, `overall_status: passed` (see audit caveat D7) |
| 9 | `python -m pytest tests/ -v --run-real-data` | — | **122 passed**, 8 warnings (sklearn `penalty` FutureWarning), 13.4 s |
| 9' | `python -m pytest tests/` (no real data flag) | — | 121 passed, 1 skipped |

Steps 7b and 7c are not in the README's list. They produce the endpoint diagnostics, error analysis and the three committed evaluation figures.

**Avoiding changes to tracked files.** Steps 4 and 7 overwrite committed files. To reproduce without touching them:

```bash
sed "s#^report_path: .*#report_path: /tmp/repro/dataset_eda_report.md#" \
    configs/data_config.yaml > /tmp/repro_config.yaml
python -m src.data.generate_eda_plots --config /tmp/repro_config.yaml
python -m src.evaluation.model_evaluation --output /tmp/rerun_metrics
```

After the full run above, `git status` showed no modified tracked files.

---

## 3. Fixed Configuration

| Item | Value | Source |
| :--- | :--- | :--- |
| Dataset | DeepChem `tox21.csv.gz`, SHA-256 `45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360` | `configs/data_config.yaml` |
| Canonicalisation | RDKit canonical isomeric SMILES | `configs/data_config.yaml` |
| Morgan | radius 2, 2,048 bits, binary, `GetMorganGenerator` default invariants | `src/features/fingerprints.py` |
| Descriptors | MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds, RingCount, HeavyAtomCount, FractionCSP3 | `src/features/descriptors.py` |
| Scaling | Descriptors standardised with train-only mean/std; Morgan bits unscaled | `src/features/preprocessor.py` |
| Split | Bemis–Murcko, 80/10/10, largest-group-first, seed 42 for tie-breaking, acyclic → one group in train | `src/data/split_data.py` |
| LR | L2, C=1.0, lbfgs, max_iter=1000, `class_weight='balanced'` | `model_evaluation.py::create_classifier` |
| RF | 100 trees, max_depth=12, min_samples_split=5, `class_weight='balanced'` | same |
| XGBoost | 100 rounds, max_depth=5, lr=0.1, `scale_pos_weight` = train neg/pos per endpoint | same |
| Model seed | `random_state=42` | same |
| Threshold | Maximise validation F1 over 91 values in [0.05, 0.95]; applied unchanged to test | `src/evaluation/metrics.py` |
| Aggregation | Mean, SD (ddof=1), SEM = SD/√12, Student-t 95% CI over endpoints | `src/evaluation/aggregation.py` |

---

## 4. Benchmark Reproduction Outcome

**(a) From committed predictions.** The script recomputes ROC-AUC and Average Precision with `sklearn.metrics`, from `predictions_*.csv`, regenerated labels and split indices. All 9 configurations match `model_comparison_table.csv` to 4 decimals; the per-endpoint maximum difference from the summary JSON is 2.2e-16. **The committed numbers are traceable.**

**(b) By retraining.** XGBoost (all 3 feature sets) and LR-descriptors reproduce byte-for-byte. RF (all 3) and LR-Morgan/combined do not; RF-combined gives **0.7625 ± 0.0196** against the committed **0.7488 ± 0.0181**. Re-runs in the pinned environment are deterministic. The cause is an unrecorded package version at the time the results were committed. Full table and diagnosis: [`experiment_tracking_audit.md` §3](experiment_tracking_audit.md#3-re-run-discrepancy-retraining-in-the-pinned-environment).

---

## 5. Artifact Map

| Path | Tracked in Git? | Produced by |
| :--- | :---: | :--- |
| `data/raw/`, `data/processed/` | No (gitignored) | Steps 1–6 |
| `results/experiment_logs/` | No | Steps 2–3, 8 |
| `results/figures/eda/` | No | Step 4 |
| `results/figures/*.png` (3 evaluation figures) | **Yes** | Step 7c |
| `results/metrics/*.json`, `*.csv` | **Yes** | Steps 7, 7b, 7c |
| `results/models/` | No | `python -m src.evaluation.run_experiments` (Student 3 runner; saves `.joblib` models) |

---

## 6. Known Limitations

- Single seed and a single split, so there is no estimate of run-to-run variance.
- The committed RF/LR metrics are not bit-reproducible in the pinned environment (§4b).
- Package versions are not written into the results JSON.
- `src/audit/run_audit.py` check 7 contains stale hard-coded statements about model results.
