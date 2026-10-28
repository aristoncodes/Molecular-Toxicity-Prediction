# Student 5 Contribution Report: Research Foundation, Literature Verification and Reproducibility Integration

**Name:** Aditya Yadav  
**Role:** Documentation Owner, Phase 1 research and reproducibility integration  
**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Scope:** Phase 1 (classical ML on Tox21) only. No Phase 2 / GNN implementation. The IEEE manuscript and presentation are authored separately by the team and were not edited as part of this work.

---

## 1. Responsibilities

- Review and verify the molecular toxicity prediction literature, and maintain an accurate bibliography.
- Link each methodological rationale to the code and tests that implement it.
- Document and independently check how Phase 1 experiments are reproduced and tracked.
- Keep the README and shared documentation consistent with the saved empirical results.
- Support viva preparation.

---

## 2. Work Completed (2026-10-28)

| File | Change | Why |
| :--- | :--- | :--- |
| `reports/literature_review.md` | Revised | Citation and factual errors corrected (wrong Huang et al. title; Fsp³ misattributed to a solubility paper; 3 in-text citations missing from the list; wrong prevalence range; unsupported numbers; MoleculeNet over-claim). Every statement is now marked as either a literature finding or a **[Project]** measurement. A verification log was added |
| `reports/ieee_report/references.bib` | **Created** | No BibTeX file existed. 24 entries, metadata checked against Crossref/arXiv. The manuscript itself was not touched |
| `reports/research_implementation_traceability.md` | **Created** | 18-row matrix: rationale → reference → code → test/artifact → status → limitation |
| `reports/experiment_reproducibility.md` | **Created** | Exact commands, configuration, artifact map and verified outcomes from a clean environment |
| `reports/experiment_tracking_audit.md` | **Created** | Benchmark traceability, re-run comparison and 14 documentation discrepancies (D1–D14) |
| `README.md` | Edited | Fixed 8 broken `file:///Users/shouryapratap/...` links, prevalence range, test count (117 → 122) and the tracked/ignored labels for `results/`. Added the macOS `libomp` requirement, the missing pipeline steps 7b, an explanation of what SEM means, and links to the new documents |
| `reports/viva_preparation.md` | Edited | Q10 prevalence example corrected; Q12 feature-ranking statement corrected; Q12 notes on SEM and reproducibility added |
| `reports/contributions/student5_aditya_yadav.md` | Rewritten (this file) | The previous version quoted SR-HSE metrics (ROC ~0.66, PR-AUC 0.1542) and a "+0.10 to +0.15" random-split inflation figure, neither of which is supported by any repository artifact |

This file serves as the requested `student5_contribution.md`. It was kept at the existing `reports/contributions/` path so it sits with the Student 1–4 reports instead of duplicating them.

---

## 3. Evidence

All of the following were executed, not assumed. Commands are in `reports/experiment_reproducibility.md`.

- **Fresh environment:** Python 3.12.0 venv with the pinned `requirements.txt`.
- **Pipeline steps 1–6:** checksum matched; 7,823 valid molecules / 8 excluded; split 6,258 / 782 / 783; 0 scaffold overlap.
- **`python -m src.audit.run_audit`:** exit 0, `passed`. Check 7 was found to be stale.
- **`python -m pytest tests/ --run-real-data`:** **122 passed**. This needed OpenMP for XGBoost on macOS.
- **Independent metric recomputation** from the 9 committed prediction CSVs with `sklearn.metrics`: exact match to the committed tables.
- **Retraining all 9 configurations** into a scratch directory: XGBoost and LR-descriptors byte-identical; RF and LR-Morgan/combined differ (RF-combined 0.7625 vs committed 0.7488). Traced to an unrecorded scikit-learn version.
- **24 references** resolved via Crossref/arXiv. MoleculeNet split and metric claims were checked against the full text.
- `git status` was confirmed clean of unintended changes after every run. Steps that overwrite tracked files were redirected to a scratch directory.

---

## 4. Key Findings

1. **The headline numbers are traceable but not exactly regenerable.** RF-combined 0.7488 ± 0.0181 recomputes exactly from the saved predictions. Retraining with pinned packages gives 0.7625 ± 0.0196. The ranking is unchanged.
2. **"± SEM" is across endpoints, from one run.** It is not repeated-seed uncertainty and cannot support claims of statistical significance.
3. **MoleculeNet benchmarks Tox21 with a random split**, so this project's scaffold-split results should not be compared with MoleculeNet's numbers directly.
4. **Validation and test sets contain only singleton scaffolds**, a consequence of the largest-first split. This should be disclosed.
5. **Reports owned by other students contain stale values.** `phase1_model_evaluation.md` §6.1 has 9 of 12 endpoint rows wrong, and the audit's check 7 is hard-coded. These were reported, not edited.

---

## 5. Limitations of This Work

- RF/LR results were not regenerated in the repository. Whether to replace the committed artifacts is a team decision for Students 3 and 4.
- Steps 7b and 7c (`endpoint_analysis`, `error_analysis`) were not re-run; their tracked outputs were checked against the summary JSON instead.
- Citations were verified for metadata; claim-level checks against source text were done only for MoleculeNet, Kapoor & Narayanan, and Yang et al.
- The XGBoost tests were run using scikit-learn's bundled `libomp` through `DYLD_LIBRARY_PATH`, not a system install.

---

## 6. How This Complements Students 1–4

Students 1–4 built the data pipeline, features, models and evaluation. Student 5 adds no new pipeline code. The contribution is independent verification: tying the scientific rationale to verified sources, checking the code against that rationale, regenerating the pipeline from scratch, and making sure every reported number can be traced to an artifact. This makes Phase 1 easier to reproduce, explain and defend.
