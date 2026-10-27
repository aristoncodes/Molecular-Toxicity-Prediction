# Student 5 Contribution Report: Literature Review, IEEE Report and Presentation

**Name:** Aditya Yadav  
**Role:** Documentation Owner  
**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Scope:** Phase 1 — Academic Literature Review, Citation Management, IEEE Conference Paper, and Oral Presentation  

---

## 1. Responsibilities

- Review relevant molecular toxicity prediction literature across QSAR, chemical fingerprints, scaffold splitting, and graph neural networks.
- Maintain accurate bibliographic citations and BibTeX records.
- Assemble the formal IEEE conference paper using the empirical benchmark results verified by Students 1–4.
- Create oral defense presentation slides, comparative summary tables, and presentation figures.
- Maintain the project README, setup instructions, and team contribution records.
- Coordinate team presentation readiness and technical viva defense preparation.

---

## 2. Technical Deliverables

### Technical Documentation & Academic Reports (`reports/`)
- `reports/literature_review.md`: Exhaustive 7-section literature review analyzing:
  - High-throughput toxicology challenges (Tox21 data sparsity, class imbalance).
  - 1D/2D physicochemical descriptors (Hansch, Lipinski, Ertl).
  - Circular topological fingerprints (ECFP / Morgan fingerprints, Rogers & Hahn 2010).
  - Classical QSAR machine learning algorithms (LR, RF, XGBoost).
  - Scaffold-based out-of-distribution evaluation (Bemis & Murcko 1996; Wu et al. 2018).
  - Imbalance-aware evaluation metrics (ROC-AUC vs. PR-AUC).
  - Research gap justifying Phase 2 Graph Neural Networks (GCN, GAT, GIN).
  - 15 peer-reviewed citations with formal bibliographic entries.
- `reports/viva_preparation.md`: Comprehensive 13-question technical examination guide covering:
  - Tox21 biological assay mechanisms (NR and SR pathways).
  - RDKit sanitization and organometallic exclusion rationale.
  - Mathematical derivation of Morgan fingerprint hashing iterations.
  - Scaffold splitting vs. random splitting leakage proofs.
  - Preprocessing leakage prevention via training-only standardization.
  - Unlabelled sample masking and threshold tuning protocols.
  - Phase 1 champion benchmark results and Phase 2 GNN motivation.
- `README.md`: Centralized project README detailing setup instructions, full reproduction commands, repository architecture, scientific policies, verified benchmark tables, and team roles.

### Final Submission Deliverables (User Authoring Responsibilities)
- `reports/ieee_phase1_manuscript.tex`: IEEEtran LaTeX conference manuscript template prepared for final paper authoring and compilation.
- Presentation Slides / Deck: Coordinated oral defense presentation based on `reports/viva_preparation.md` and benchmark comparison figures in `results/figures/`.

---

## 3. Key Findings & Literature Insights

- **The Scaffold Generalization Problem:** Random dataset splitting in chemical libraries allows peripheral analogs to cross partitions, inflating test ROC-AUC scores by $+0.10$ to $+0.15$. Bemis–Murcko scaffold partitioning provides the only scientifically honest out-of-distribution test for novel chemical discovery.
- **Metric Rigor in Toxicology:** In assays with $<5\%$ positive prevalence (e.g., `SR-HSE` at 2.84%), ROC-AUC remains artificially high ($\sim 0.66$) because large true negative counts buffer false positive rates. PR-AUC ($0.1542$) provides the true measure of actionable precision for experimental cherry-picking.
- **Phase 2 Research Gap:** Handcrafted Morgan fingerprints use fixed 2048-bit modulo hashing that suffers from bit collisions, lacks stereochemical awareness, and cannot adapt to target proteins. Graph Neural Networks operating directly on molecular graphs $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ are required to learn adaptive, end-to-end representations.
