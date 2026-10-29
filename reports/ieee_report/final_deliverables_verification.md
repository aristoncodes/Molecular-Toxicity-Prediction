# Final Deliverables Verification Record

**Date:** 2026-10-28 · **Repository base:** `student-5-research-reproducibility` (on top of `8a80819`)

This note records how the final paper and presentation were checked. It supports the deliverables; it does not replace them.

## 1. Deliverables

| Deliverable | Path | State |
| :--- | :--- | :--- |
| IEEE manuscript (LaTeX) | `reports/ieee_report/phase1_final_manuscript.tex` | Complete source; **not compiled** (see §5) |
| Bibliography | `reports/ieee_report/references.bib` | 29 verified entries |
| Figures | `reports/ieee_report/figures/*.pdf, *.png` | 4 figures generated from artifacts |
| Presentation | `reports/presentation/phase1_final_presentation.pptx` | 18 editable slides, native charts and tables, speaker notes |
| Generators | `reports/ieee_report/make_figures.py`, `reports/presentation/build_presentation.py` | Re-runnable |

## 2. Source of every number

Both the paper and the slides read from `figures/figure_values.json`, which `make_figures.py` builds from:

- `results/metrics/model_benchmark_summary.json`: macro and per-endpoint metrics, thresholds, confusion matrices
- `results/metrics/endpoint_label_diagnostics.json`: label counts, missingness, prevalence
- `data/processed/scaffold_split_indices.json`: split sizes (regenerated and checked against the documented values)

**Checks run:**
- All 9 configurations recompute exactly from `predictions_*.csv` with `sklearn.metrics` (see `reports/experiment_tracking_audit.md`).
- A script compared every number in manuscript Tables I, III and IV with the JSON: **0 mismatches** after correction.
  - This pass caught and fixed 8 hand-typed AP standard deviations (off in the 4th decimal) and the SR-HSE label row (6,088 / 17.42%).
- The slide chart data was extracted with `slide_text.py` and matches the JSON.
- The re-run values (RF-combined 0.7625, and others) come from the retrain recorded in `reports/experiment_tracking_audit.md` §3.

**Uncertainty wording:** "±" is the SEM across 12 endpoints from one seed-42 run (`src/evaluation/aggregation.py`). Every place it appears says so. No significance claims are made.

## 3. Literature

| Check | Result |
| :--- | :--- |
| Entries | 29, all cited in the manuscript (scripted check: 0 undefined, 0 unused, 0 duplicate keys) |
| Metadata | DOIs resolved via Crossref; arXiv IDs via the arXiv API; scikit-learn via the JMLR page |
| Claim-level reading (abstract or full text) | Huang 2016 (challenge design, AUC ranges, 641-compound test set); Mayr 2016 (DeepTox result, RF/SVM comparison); Wu 2018 (Tox21 random split, scaffold-split wording); Svetnik 2003; Svetnik 2005; Sheridan 2013; Jiang 2021; Kapoor & Narayanan 2023; Yang 2019 (abstract only) |
| Metadata only | Foundational method papers cited only for their core contribution (Breiman, Chen & Guestrin, Rogers & Hahn, Morgan, Bemis & Murcko, Wildman & Crippen, Ertl, Lipinski, Lovering, Hansch & Fujita, Todeschini & Consonni, Davis & Goadrich, Saito & Rehmsmeier, Hartung, NRC, Kipf, Veličković, Xu, RDKit) |
| Published numbers | Quoted as context only; the text states that protocols differ (challenge test set, MoleculeNet random split) |

## 4. Methodology consistency (paper ↔ code)

| Claim | Code location |
| :--- | :--- |
| Morgan r=2, 2048 bits, binary | `src/features/fingerprints.py` |
| Nine descriptors | `src/features/descriptors.py::DESCRIPTOR_NAMES` |
| Train-only scaling, ε=1e-8 | `src/features/preprocessor.py::LeakageSafeScaler` |
| Scaffold: `MurckoScaffoldSmiles`, no chirality, largest-first, seed 42 | `src/data/split_data.py` |
| Hyperparameters (Table II) | `src/evaluation/model_evaluation.py::create_classifier` |
| Threshold grid 0.05–0.95 (91 values), `>=` | `src/evaluation/metrics.py` |
| AP = `average_precision` | `src/evaluation/metrics.py::compute_pr_auc` |
| SEM = SD(ddof=1)/√12 | `src/evaluation/aggregation.py` |

## 5. Checks that could not be completed

| Check | Status | Reason / mitigation |
| :--- | :--- | :--- |
| LaTeX compilation and PDF inspection | **Not done** | No TeX installed. Downloading the Tectonic binary was blocked by the session's safety policy. Mitigations: scripted checks for citations, labels/refs, brace and environment balance, figure files present, and ASCII-only source. **Compile on Overleaf and check the layout before submission.** |
| Rendering slides in PowerPoint/LibreOffice | **Not done** | LibreOffice is not installed, and PowerPoint refused AppleScript automation (error -9074). Mitigations: the skill's `validate.py` passed; a custom renderer using the real Calibri/Cambria font files checked every text box for overflow and off-slide placement (0 issues after fixes); chart data was verified by extraction. **Open the deck in PowerPoint and check the native charts.** |

## 6. Placeholders left for the authors

Affiliation, e-mail, supervisor and acknowledgement. The first author's surname also needs confirming ("Sindur" in the README, "Sandur" in the git log).
