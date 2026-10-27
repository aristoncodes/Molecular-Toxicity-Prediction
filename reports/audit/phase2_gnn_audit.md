# Phase 2 Graph Neural Network Audit & Implementation Gap Analysis

**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Scope:** Investigation of Phase 2 Deep Learning and Graph Representation Learning Components  
**Auditor:** Senior ML Researcher, GNN Specialist & Scientific Auditor  
**Audit Date:** 2026-10-27  

---

## 1. Executive Status: Phase 2 Is NOT Yet Implemented

An exhaustive inspection of the codebase, dependency specifications, directory trees, and Git history reveals:
- **PyTorch / PyTorch Geometric Status:** Neither `torch`, `torch-geometric`, `dgl`, nor any deep learning framework is declared in `requirements.txt` or imported in `src/`.
- **Graph Representation Status:** Zero files exist for featurizing molecules as graphs (nodes $\mathcal{V}$, edges $\mathcal{E}$, node feature matrices $\mathbf{X}_v$, edge feature matrices $\mathbf{E}_{uv}$).
- **Graph Model Architecture Status:** Zero GNN modules, message-passing layers, graph pooling operations, or masked deep learning loss functions exist in the repository.
- **Verdict:** **Phase 2 (Deep Learning / Graph Neural Networks) has NOT been implemented.**

> [!IMPORTANT]
> A working Phase 1 classical ML baseline alone **does not satisfy the overarching project objective** of investigating whether learned molecular graph representations outperform engineered chemical fingerprints. Phase 2 must be implemented before the comparative scientific hypothesis can be resolved.

---

## 2. Granular Component Audit (Phase 2 Requirements)

| Phase 2 Requirement | Current Repository Evidence | Implementation Status | Technical Gap |
| :--- | :--- | :---: | :--- |
| **1. Molecular Graph Construction** | None. Molecules exist only as CSV SMILES strings or fixed numpy feature arrays. | **MISSING** | Need RDKit $\to$ Graph featurizer (`MolGraphFeaturizer`) producing node/edge feature tensors. |
| **2. Node & Edge Features** | None. | **MISSING** | Need atomic property encoding (element, hybridization, degree, formal charge, aromaticity) and bond encoding (order, conjugation, stereochemistry). |
| **3. GCN / MPNN Architecture** | None. | **MISSING** | Need PyTorch / PyG model class with message passing, graph pooling (mean/sum/attention), and multi-task classifier head. |
| **4. 12-Endpoint Predictions** | None. | **MISSING** | Need output layer with dimension $12$, yielding raw logits $\mathbf{z} \in \mathbb{R}^{B \times 12}$ for sigmoid probability transformation. |
| **5. Masked Multi-Task Loss** | None. | **MISSING** | Need custom loss module: $\mathcal{L} = \frac{\sum_{i,j} M_{ij} \cdot \text{BCE}(z_{ij}, y_{ij})}{\sum_{i,j} M_{ij} + \epsilon}$, where $M_{ij} \in \{0, 1\}$ indicates observed assay measurements. |
| **6. Imbalance Weighting** | None. | **MISSING** | Need positive class loss weighting via `pos_weight = (num_neg / num_pos)` per endpoint. |
| **7. Scaffold Split Alignment** | `scaffold_split_indices.json` exists. | **UNCONNECTED** | Need PyG `Dataset` / `DataLoader` consuming the exact train/val/test indices from Phase 1. |
| **8. Validation Early Stopping** | None. | **MISSING** | Need training loop with early stopping triggered by validation Macro ROC-AUC. |
| **9. Identical Evaluation Protocol** | `src/evaluation/metrics.py` exists. | **UNCONNECTED** | Need to pass GNN test predictions into the existing Phase 1 evaluation suite. |
| **10. Comparative Baseline Analysis**| Fingerprint results exist ($0.7488$). | **BLOCKED BY GNN** | Direct statistical hypothesis test blocked until GNN benchmark is executed. |

---

## 3. Recommended Phase 2 Implementation Blueprint

To complete Phase 2 and achieve the final project objective, the following modular components should be constructed:

### 3.1 Dependencies Required
Add to `requirements.txt`:
```txt
torch>=2.2.0
torch-geometric>=2.5.0
```

### 3.2 Directory Layout
```
src/
└── models/
    └── gnn/
        ├── __init__.py
        ├── featurizer.py           # RDKit Mol -> PyG Data(x, edge_index, edge_attr, y)
        ├── dataset.py              # PyG InMemoryDataset wrapping Tox21 graphs
        ├── architectures.py        # GCN, GAT, GIN implementations
        ├── loss.py                 # MaskedBCEWithLogitsLoss with pos_weight
        └── train.py                # Training loop with validation early stopping and checkpointing
```

### 3.3 Atomic & Bond Feature Definitions
- **Atom Features (Node Vectors $\mathbf{x}_v \in \mathbb{R}^{39}$):**
  - Atomic number (one-hot: C, N, O, S, F, Cl, Br, I, P, other)
  - Degree / heavy neighbor count (one-hot: 0 to 5)
  - Formal charge (scalar integer)
  - Hybridization (one-hot: SP, SP2, SP3, SP3D, SP3D2)
  - Aromaticity (binary)
  - Number of hydrogen atoms (one-hot: 0 to 4)
- **Bond Features (Edge Vectors $\mathbf{e}_{uv} \in \mathbb{R}^{10}$):**
  - Bond type (one-hot: single, double, triple, aromatic)
  - Conjugation (binary)
  - In-ring status (binary)
  - Stereochemistry (one-hot: None, Any, Z, E)

### 3.4 Architecture Blueprint: Graph Isomorphism Network (GIN) / GCN
1. **Message Passing Layers:** 3 to 5 graph convolution layers with BatchNorm and ReLU / GELU activations:
   $$\mathbf{h}_v^{(k)} = \text{MLP}^{(k)}\left( (1 + \epsilon^{(k)}) \mathbf{h}_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(k-1)} \right)$$
2. **Graph Readout (Global Pooling):**
   $$\mathbf{h}_{\mathcal{G}} = \text{Concat}\left[ \text{GlobalMeanPool}(\{\mathbf{h}_v\}), \text{GlobalMaxPool}(\{\mathbf{h}_v\}) \right] \in \mathbb{R}^{2 \cdot d}$$
3. **Multi-Task Classification Head:**
   $$\hat{\mathbf{y}}_{\mathcal{G}} = \sigma\left( \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{h}_{\mathcal{G}} + \mathbf{b}_1) + \mathbf{b}_2 \right) \in [0, 1]^{12}$$

### 3.5 Exact Comparability Mandate
To ensure scientific fairness with Phase 1:
- Must use the identical 7,823 molecules in `data/processed/tox21_cleaned.csv`.
- Must load the identical train (6,258), validation (782), and test (783) scaffold partitions from `data/processed/scaffold_split_indices.json`.
- Must evaluate using the exact same metrics (`compute_roc_auc`, `compute_pr_auc`, `find_optimal_threshold`) from `src/evaluation/metrics.py`.
- Must test hypothesis: $\text{Macro ROC-AUC}_{\text{GNN}} > 0.7488 \pm 0.0181$.
