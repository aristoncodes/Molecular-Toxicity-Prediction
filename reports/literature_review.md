# Literature Review: Machine Learning and Molecular Representations for Toxicity Prediction

**Project:** Molecular Toxicity Prediction via Fingerprints and Molecular Graph Networks  
**Scope:** Phase 1 (classical machine learning on Tox21). Graph neural networks are discussed only as literature motivating Phase 2; nothing in Phase 1 implements them.  
**Bibliography:** [`reports/ieee_report/references.bib`](ieee_report/references.bib). Every entry was checked against Crossref or arXiv on 2026-10-28 (see [Section 8](#8-citation-verification-log)).

> **Reading convention.** Statements attributed to a citation are established findings from that source. Statements marked **[Project]** are this repository's own measurements and are traceable to files under `results/`. The two are kept separate on purpose.

---

## 1. Introduction and Context

Predicting chemical toxicity has historically relied on low-throughput *in vivo* animal testing, which is costly, raises ethical concerns, and extrapolates imperfectly to humans (Hartung, 2009). The U.S. National Research Council's report *Toxicity Testing in the 21st Century* (NRC, 2007) argued for a shift toward high-throughput *in vitro* assays of toxicity pathways. The Tox21 programme and its 2014 Data Challenge follow that direction: thousands of environmental chemicals and drugs were screened against nuclear receptor (NR) and stress response (SR) pathway assays, and the resulting data were released as a modelling challenge (Huang et al., 2016). The challenge-winning DeepTox system is described by Mayr et al. (2016).

Quantitative Structure–Activity Relationship (QSAR) modelling maps a molecular representation $\mathcal{X}$ to a biological outcome $\mathcal{Y}$. Tox21 makes this harder in three ways:

1. **Missing labels.** Not every compound was tested in every assay, so the label matrix is sparse. **[Project]** In the 7,823 cleaned molecules, 17.06% of compound–endpoint labels are missing (`results/metrics/endpoint_label_diagnostics.json`).
2. **Class imbalance.** Active compounds are rare. **[Project]** Positive prevalence ranges from **2.89% (NR-PPAR-gamma) to 16.17% (SR-ARE)** across the 12 endpoints (same file). Plain accuracy is therefore uninformative.
3. **Distribution shift in chemical space.** Molecules cluster into structural series, so how the data are split strongly affects how optimistic the reported performance is (Wu et al., 2018; Section 4).

---

## 2. Chemical Representations

Statistical learning needs a fixed-length vector for each molecular graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$.

### 2.1 Physicochemical Descriptors
Early QSAR related biological activity to physicochemical parameters such as hydrophobicity and electronic effects (Hansch & Fujita, 1964). Todeschini & Consonni (2009) is a standard reference work cataloguing molecular descriptors. Phase 1 uses nine RDKit descriptors:

| Descriptor | Meaning | Source of method |
| :--- | :--- | :--- |
| Molecular weight, heavy atom count | Size / steric bulk | Standard (RDKit) |
| $\log P$ (`MolLogP`) | Lipophilicity, atom-contribution estimate | Wildman & Crippen (1999) |
| TPSA | Polar surface area from fragment contributions | Ertl et al. (2000) |
| H-bond donors / acceptors | Polar interaction capacity; used in the "rule of 5" | Lipinski et al. (1997) |
| Rotatable bonds, ring count | Flexibility and rigidity | Standard (RDKit) |
| Fraction $sp^3$ carbons ($\text{Fsp}^3$) | Saturation / three-dimensionality | Lovering et al. (2009) |

*Limitation.* Nine global scalars cannot describe where functional groups sit in the molecule or how they are connected.

### 2.2 Circular Fingerprints (ECFP / Morgan)
Rogers & Hahn (2010) introduced Extended-Connectivity Fingerprints (ECFP). They derive from the Morgan (1965) algorithm for producing unique identifiers for chemical structures.

- *Mechanism.* Each atom gets an integer identifier from its local invariants. At each iteration up to radius $r$, the identifier is updated by hashing it together with its neighbours' identifiers and the connecting bond types. The set of identifiers is then folded into a fixed-length bit vector of length $d$:
  $$\text{bit\_index} = \text{hash}(\text{substructure\_id}) \bmod d$$
  Radius 2 corresponds to ECFP4 (a diameter of 4 bonds).
- *Strengths.* It encodes the presence or absence of local substructures, such as functional groups and ring systems.
- *Limitations.* Folding causes **bit collisions**, where different substructures map to the same bit. The representation is fixed rather than learned, and in RDKit's default setting it ignores stereochemistry.

**[Project]** `src/features/fingerprints.py` uses `rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)` with RDKit default atom invariants and binary output. These are ECFP4-like connectivity fingerprints, not feature-based (FCFP) ones.

---

## 3. Classical Machine-Learning Baselines

These three learners are the standard non-deep baselines in molecular benchmarks such as MoleculeNet (Wu et al., 2018):

1. **L2-regularised Logistic Regression.** A linear model of the log-odds, $\sigma(\mathbf{w}^T\mathbf{x}+b)$. It is a convex, stable baseline but cannot represent feature interactions unless they are built explicitly.
2. **Random Forest** (Breiman, 2001). An ensemble of decision trees, each grown on a bootstrap sample with random feature subsets at each split. It captures non-linear interactions and is unaffected by monotonic rescaling of features.
3. **Gradient-boosted trees, XGBoost** (Chen & Guestrin, 2016). An additive tree ensemble fitted with a second-order approximation of the loss, plus regularisation on tree size and leaf weights:
   $$\mathcal{L}^{(t)} = \sum_{i=1}^n \left[ g_i f_t(\mathbf{x}_i) + \tfrac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \gamma T + \tfrac{1}{2} \lambda \sum_{j=1}^T w_j^2$$

The literature does not establish which of RF or boosting is better for toxicity data in general; the answer depends on the dataset, the split and the tuning. **[Project]** On this scaffold split with fixed default-style hyperparameters (no tuning), RF on combined features ranked first (Section 6).

---

## 4. Data Splitting and Leakage

### 4.1 Scaffold Splitting vs. Random Splitting
Bemis & Murcko (1996) defined a molecular framework (scaffold) as a molecule's ring systems plus the linkers joining them, with side chains removed.

- MoleculeNet describes scaffold splitting as separating samples "based on their two-dimensional structural frameworks". It notes that because this separates structurally different molecules, it "offers a greater challenge for learning algorithms than the random split" (Wu et al., 2018, Sec. "Data splitting").
- **Important caveat.** MoleculeNet's own recommended split for **Tox21 is random**, not scaffold (Wu et al., 2018, Table 1). It recommends scaffold splits for other datasets such as HIV, BACE and BBBP. The scaffold-split numbers in this project are therefore **not directly comparable** with MoleculeNet's published Tox21 results.
- Yang et al. (2019) benchmarked fingerprint/descriptor models against learned graph representations across many public and industrial datasets. The question they set out to test, which approach generalises better to new chemical space, is the same question that motivates this project's Phase 2.

The intuition for preferring scaffold splits is this: close analogues of a test molecule in the training set make the task easier than predicting genuinely new chemotypes. This is a widely stated rationale rather than a fixed numerical effect. **[Project]** No random-split experiment was run in Phase 1, so this repository makes no claim about how large the gap between random and scaffold splits is on Tox21.

**[Project] Split behaviour to disclose.** `src/data/split_data.py` assigns scaffold groups largest-first to train and then fills validation and test. All 1,775 acyclic molecules form one group placed in train. As a result, validation (782 molecules / 782 scaffolds) and test (783 / 783) contain **only singleton scaffolds**. The test set measures generalisation to rare, one-off chemotypes, and its composition differs systematically from the training set.

### 4.2 Preprocessing Leakage
Kapoor & Narayanan (2023) document data leakage as a widespread source of over-optimistic, irreproducible results in ML-based science, and propose a taxonomy of leakage types. One such type is fitting preprocessing on data that includes the test set. To avoid it, transformation statistics must be estimated on the training partition only:
$$\mathbf{z}_{\text{test}} = \frac{\mathbf{x}_{\text{test}} - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}} + \epsilon}$$
**[Project]** This is implemented in `src/features/preprocessor.py`, where `LeakageSafeScaler` and `PartitionPreprocessor` scale the descriptors only and leave the binary Morgan bits unscaled. Decision thresholds are likewise chosen on the validation split only (`src/evaluation/metrics.py::find_optimal_threshold`).

---

## 5. Metrics Under Imbalance and Missing Labels

1. **ROC-AUC** is the probability that a randomly chosen positive is ranked above a randomly chosen negative. Its value does not change with class prevalence. It was the official Tox21 Challenge metric (Huang et al., 2016) and is MoleculeNet's recommended Tox21 metric (Wu et al., 2018).
2. **PR-AUC / Average Precision (AP).** Davis & Goadrich (2006) showed how ROC and precision–recall curves relate. Saito & Rehmsmeier (2015) showed that precision–recall plots are more informative than ROC plots on imbalanced data. A random classifier's PR baseline equals the positive prevalence, so PR-AUC values are only meaningful relative to prevalence. **[Project]** "PR-AUC" in this repository means **Average Precision** (`sklearn.metrics.average_precision_score`), $\text{AP} = \sum_k (R_k - R_{k-1}) P_k$, not the trapezoidal area.
3. **Missing labels.** An untested compound–assay pair is not evidence of inactivity. **[Project]** Missing labels are masked per endpoint in both training and evaluation and are never imputed as 0.
4. **Aggregating over endpoints.** **[Project]** Macro metrics are the unweighted mean over $K=12$ endpoints. The reported "± SEM" is $\text{SD}/\sqrt{12}$ of the **12 endpoint scores from a single training run with seed 42**. It describes how much performance varies across endpoints. It is **not** an estimate of run-to-run, seed-to-seed or resampling uncertainty, and it should not be read as a significance test between models.

---

## 6. Phase 1 Results in Context, and the Gap Motivating Phase 2

**[Project]** The committed benchmark (`results/metrics/model_comparison_table.csv`) reports **Random Forest on combined features** as the best of the 9 configurations: test macro ROC-AUC $0.7488 \pm 0.0181$ (SEM over endpoints) and macro AP $0.3891 \pm 0.0320$. These numbers recompute exactly from the saved prediction files. Retraining in the currently pinned environment gives RF + combined 0.7625 instead; the ranking is unchanged. See [`experiment_tracking_audit.md`](experiment_tracking_audit.md) for details.

The representation limitations below are the literature's motivation for graph neural networks. They are hypotheses for Phase 2 to test, not results of this project:

| Dimension | Fingerprints / descriptors (Phase 1) | Message-passing GNNs (Phase 2, literature) |
| :--- | :--- | :--- |
| Representation | Fixed hashed bits plus global scalars | Learned atom/molecule embeddings |
| Collisions | Folding can merge distinct substructures | No folding step |
| Bond information | Encoded only through the hashing procedure | Explicit edge features possible |
| Task adaptation | Same features for all 12 endpoints | Trained end-to-end per task or multi-task |

Candidate architectures from the literature: GCN (Kipf & Welling, 2017), GAT (Veličković et al., 2018) and GIN (Xu et al., 2019). The GIN authors relate the expressive power of message passing to the 1-Weisfeiler–Lehman test. Whether any of these beats the Phase 1 baseline **on this scaffold split** is an open empirical question.

---

## 7. References

BibTeX keys refer to [`reports/ieee_report/references.bib`](ieee_report/references.bib).

1. Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks. *J. Med. Chem.*, 39(15), 2887–2893. doi:10.1021/jm9602928 — `bemis1996frameworks`
2. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. doi:10.1023/A:1010933404324 — `breiman2001random`
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proc. 22nd ACM SIGKDD*, 785–794. doi:10.1145/2939672.2939785 — `chen2016xgboost`
4. Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. *Proc. ICML '06*, 233–240. doi:10.1145/1143844.1143874 — `davis2006relationship`
5. Ertl, P., Rohde, B., & Selzer, P. (2000). Fast calculation of molecular polar surface area as a sum of fragment-based contributions and its application to the prediction of drug transport properties. *J. Med. Chem.*, 43(20), 3714–3717. doi:10.1021/jm000942e — `ertl2000tpsa`
6. Hansch, C., & Fujita, T. (1964). ρ-σ-π Analysis. A method for the correlation of biological activity and chemical structure. *J. Am. Chem. Soc.*, 86(8), 1616–1626. doi:10.1021/ja01062a035 — `hansch1964rho`
7. Hartung, T. (2009). Toxicology for the twenty-first century. *Nature*, 460(7252), 208–212. doi:10.1038/460208a — `hartung2009toxicology`
8. Huang, R., Xia, M., Nguyen, D.-T., et al. (2016). Tox21Challenge to build predictive models of nuclear receptor and stress response pathways as mediated by exposure to environmental chemicals and drugs. *Front. Environ. Sci.*, 3, 85. doi:10.3389/fenvs.2015.00085 — `huang2016tox21`
9. Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804. doi:10.1016/j.patter.2023.100804 — `kapoor2023leakage`
10. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*. arXiv:1609.02907 — `kipf2017gcn`
11. Lipinski, C. A., Lombardo, F., Dominy, B. W., & Feeney, P. J. (1997). Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings. *Adv. Drug Deliv. Rev.*, 23(1–3), 3–25. doi:10.1016/S0169-409X(96)00423-1 — `lipinski1997rule`
12. Lovering, F., Bikker, J., & Humblet, C. (2009). Escape from flatland: Increasing saturation as an approach to improving clinical success. *J. Med. Chem.*, 52(21), 6752–6756. doi:10.1021/jm901241e — `lovering2009flatland`
13. Mayr, A., Klambauer, G., Unterthiner, T., & Hochreiter, S. (2016). DeepTox: Toxicity prediction using deep learning. *Front. Environ. Sci.*, 3, 80. doi:10.3389/fenvs.2015.00080 — `mayr2016deeptox`
14. Morgan, H. L. (1965). The generation of a unique machine description for chemical structures—a technique developed at Chemical Abstracts Service. *J. Chem. Doc.*, 5(2), 107–113. doi:10.1021/c160017a018 — `morgan1965generation`
15. National Research Council (2007). *Toxicity Testing in the 21st Century: A Vision and a Strategy*. National Academies Press. doi:10.17226/11970 — `nrc2007toxicity`
16. Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. *J. Chem. Inf. Model.*, 50(5), 742–754. doi:10.1021/ci100050t — `rogers2010ecfp`
17. Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432. doi:10.1371/journal.pone.0118432 — `saito2015precision`
18. Todeschini, R., & Consonni, V. (2009). *Molecular Descriptors for Chemoinformatics*. Wiley-VCH. doi:10.1002/9783527628766 — `todeschini2009descriptors`
19. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. *ICLR*. arXiv:1710.10903 — `velickovic2018gat`
20. Wildman, S. A., & Crippen, G. M. (1999). Prediction of physicochemical parameters by atomic contributions. *J. Chem. Inf. Comput. Sci.*, 39(5), 868–873. doi:10.1021/ci990307l — `wildman1999logp`
21. Wu, Z., Ramsundar, B., Feinberg, E. N., et al. (2018). MoleculeNet: a benchmark for molecular machine learning. *Chem. Sci.*, 9(2), 513–530. doi:10.1039/C7SC02664A — `wu2018moleculenet`
22. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? *ICLR*. arXiv:1810.00826 — `xu2019gin`
23. Yang, K., Swanson, K., Jin, W., et al. (2019). Analyzing learned molecular representations for property prediction. *J. Chem. Inf. Model.*, 59(8), 3370–3388. doi:10.1021/acs.jcim.9b00237 — `yang2019analyzing`
24. Landrum, G., et al. RDKit: Open-source cheminformatics software. https://www.rdkit.org/ (version 2026.9.1 pinned) — `rdkit`

---

## 8. Citation Verification Log

**Method.** DOIs were resolved with the Crossref REST API (`api.crossref.org/works/<doi>`), and authors, title, venue, year, volume, issue and pages were compared with each entry. arXiv IDs were resolved with the arXiv API. The two MoleculeNet statements in Section 4.1 were checked against the paper's full text (arXiv:1703.00564). Date: 2026-10-28.

| Status | References |
| :--- | :--- |
| **Metadata verified** (Crossref / arXiv) | All of refs 1–23 |
| **Claim-level check against source text** | Wu et al. 2018 (Tox21 = random split in Table 1; scaffold-split description); Kapoor & Narayanan 2023 (abstract: leakage widespread, taxonomy); Yang et al. 2019 (abstract: fingerprints vs. learned representations on new chemical space) |
| **Software reference** (no DOI) | RDKit (ref 24): official website; version from `requirements.txt` |
| **Not verified at claim level** | All other references were verified for metadata only. They are cited for their well-known core contribution (e.g. Breiman for RF, Rogers & Hahn for ECFP), not for specific numbers. |

**Corrections made to the previous version of this review**

| # | Previous text | Problem | Correction |
| :--- | :--- | :--- | :--- |
| 1 | Prevalence "from 2.84% (SR-HSE) to 15.82% (NR-AhR)" | Contradicts `endpoint_label_diagnostics.json`: SR-HSE is 5.76%, NR-AhR is 11.74% | 2.89% (NR-PPAR-gamma) to 16.17% (SR-ARE) |
| 2 | Huang et al. (2016) titled "Modelling the Tox21 data challenge: A multi-task learning approach" | Wrong title | Crossref title, DOI added |
| 3 | Fsp³ attributed to "Yan & Gasteiger, 2003" | That paper (doi:10.1021/ci025590u) is about aqueous solubility prediction, not Fsp³; it was also missing from the reference list | Replaced with Lovering et al. (2009) |
| 4 | Hansch & Fujita (1964), Todeschini & Consonni (2009), NRC (2007) cited in text | Missing from reference list | Added with verified metadata |
| 5 | "Wu et al. (2018) demonstrated that scaffold splitting provides the gold-standard realistic benchmark" | Overstated; MoleculeNet lists Tox21 under **random** split | Rephrased using MoleculeNet's own wording, plus a comparability caveat |
| 6 | Random splitting gives "artificially inflated test metrics (ROC-AUC > 0.85)" | No source given for this number, and the project ran no random-split experiment | Removed the number |
| 7 | "Phase 1 establishes strong classical baselines using Morgan fingerprints and gradient boosting (achieving Test Macro ROC-AUC ≈ 0.7488)" | 0.7488 is **Random Forest on combined** features, not gradient boosting on Morgan | Corrected; reproducibility caveat added |
| 8 | "XGBoost frequently outperforms RF on tabular benchmarks"; RF "robust ... ($d \gg N$)" | Not supported by the cited sources | Softened to statements the sources support |
| 9 | Uncertainty described only as "SEM" | Did not say that the SEM is across endpoints from a single run | Clarified in Section 5 |
| 10 | — | Leakage section had no supporting citation | Added Kapoor & Narayanan (2023) |
