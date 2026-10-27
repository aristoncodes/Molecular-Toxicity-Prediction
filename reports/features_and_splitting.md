# Technical Report: Molecular Features and Scaffold Splitting

**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Scope:** Circular Topological Fingerprints, Physicochemical Descriptors, and Bemis–Murcko Scaffold Splitting  
**Status:** Complete & Verified  

---

## 1. Overview & Architecture

This module provides the core cheminformatics feature extraction and partition architecture:
1. **Morgan chemical fingerprints** (binary and count representations, radius 2, 2,048 bits).
2. **Physicochemical molecular descriptors** (nine standardized RDKit descriptors).
3. **Combined feature representations** with row-level alignment and audit manifests.
4. **Reproducible Bemis–Murcko scaffold-based train/val/test splitting** (80/10/10 partition).
5. **Independent unit and integration tests** ensuring mathematical validity and zero data leakage.

All feature extractions and partition assignments were performed on the verified cleaned dataset `data/processed/tox21_cleaned.csv` (7,823 valid molecular structures, SHA-256 `2a30e42b7addbebb02e8c440f88ac553eb7d20b775583d9bcff256c6c0d07270`).

---

## 2. Morgan Chemical Fingerprints

### 2.1 Theoretical Rationale & Parameter Selection
Morgan fingerprints (Rogers and Hahn, 2010; circular Extended-Connectivity Fingerprints, ECFP) represent molecular chemical topology by capturing circular atom environments up to a designated bond radius:
- **Radius ($r=2$):** Corresponds to **ECFP4** (effective circular diameter of 4 bonds). Radius 2 strikes an established empirical optimum in medicinal chemistry and QSAR toxicity modeling: radius 1 captures only nearest-neighbor atom pairs, while radius $\ge 3$ creates highly fragmented, sparse subgraphs with increased hashing collision risks.
- **Bit Length ($n=2048$):** The hashed bit-vector length is set to 2,048 bits. This balances bit density and collision minimization across diverse small-molecule drug libraries like Tox21.
- **Binary Bit Representation:** Each bit signifies presence ($1$) or absence ($0$) of an atom environment. An optional frequency-count representation is also implemented in `src/features/fingerprints.py`.

### 2.2 RDKit API & Implementation
- **Implementation:** `src/features/fingerprints.py`
- **Modern RDKit Generator:** `rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)`
- **Matrix Function:** `compute_morgan_fingerprints(smiles_list, radius=2, n_bits=2048)`
- **Artifact:** `data/processed/features/morgan_r2_2048.npy`
- **Output Dimensions:** $(7823, 2048)$
- **Data Type:** `numpy.uint8`
- **Artifact SHA-256:** `f804b3d7ed36846b48192003280720dc68edef1c55e92bfaf5483fb26d4516a5`

Strict structure validation is enforced: any invalid or unparsable SMILES raises an explicit `ValueError` identifying the exact row index and offending SMILES string. No rows are silently dropped.

---

## 3. Physicochemical Molecular Descriptors

### 3.1 The Nine Descriptors
To complement topological circular substructures with macroscopic whole-molecule physical and electronic properties, nine well-defined molecular descriptors were implemented:

| Index | Feature Name | RDKit Function | Units | Physicochemical Significance |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `mol_weight` | `Descriptors.MolWt` | $\text{g/mol}$ | Molecular mass; governs absorption, permeability, and blood-brain barrier transport. |
| **1** | `logp` | `Crippen.MolLogP` | Dimensionless | Wildman-Crippen lipophilicity / octanol-water partition coefficient; controls membrane permeability and non-specific binding. |
| **2** | `tpsa` | `Descriptors.TPSA` | $\text{Å}^2$ | Topological polar surface area (sum of polar N and O fragments); correlates with bioavailability and passive transport. |
| **3** | `num_h_donors` | `Lipinski.NumHDonors` | Count | Number of hydrogen-bond donors ($\text{OH}$ and $\text{NH}$ groups); critical for Lipinski's Rule of 5. |
| **4** | `num_h_acceptors` | `Lipinski.NumHAcceptors` | Count | Number of hydrogen-bond acceptors ($\text{O}$ and $\text{N}$ atoms); influences target binding enthalpy. |
| **5** | `num_rotatable_bonds` | `Lipinski.NumRotatableBonds` | Count | Torsional degrees of freedom; reflects conformational flexibility and entropic penalty upon receptor binding. |
| **6** | `ring_count` | `Lipinski.RingCount` | Count | Number of rings (SSSR); modulates structural rigidity and aromatic stacking. |
| **7** | `heavy_atom_count` | `Lipinski.HeavyAtomCount` | Count | Number of non-hydrogen atoms; measures molecular skeleton size. |
| **8** | `fraction_csp3` | `Lipinski.FractionCSP3` | Ratio ($[0, 1]$) | Ratio of $sp^3$-hybridized carbons to total carbons; measures 3D structural complexity and saturation. |

### 3.2 Implementation & Numerical Integrity
- **Implementation:** `src/features/descriptors.py`
- **Feature Constant:** `DESCRIPTOR_NAMES` (ordered tuple of 9 strings)
- **Artifact:** `data/processed/features/descriptors_9.npy`
- **Output Dimensions:** $(7823, 9)$
- **Data Type:** `numpy.float64`
- **Artifact SHA-256:** `9bbcc071320a40d45f3f00c0f214f7cf110f82aadca7c5707783ff93b8cafe99`
- **Sanity Checks:** An independent test (`tests/test_descriptors.py`) validates the 9 values against exact chemical literature values on reference small molecules (Ethanol $\text{C}_2\text{H}_6\text{O}$, Aspirin $\text{C}_9\text{H}_8\text{O}_4$, Benzene, Caffeine, Alanine). Unexpected $\text{NaN}$ and infinite values are proactively detected and rejected.

No feature scaling, normalization, or imputation is applied in this module; those operations must be fitted strictly on the training partition by Student 3 to avoid target and variance leakage.

---

## 4. Combined Feature Representation

### 4.1 Representation Assembly & Memory Footprint
- **Implementation:** `src/features/build_features.py`
- **Combined Array:** Horizontal concatenation of Morgan bit vector ($2048$) and 9 descriptors ($9$).
- **Total Dimensions:** $2048 + 9 = 2057$ columns.
- **Data Type:** `numpy.float32`. This format maintains descriptor floating-point resolution without precision loss, while keeping the matrix memory consumption at ~64.3 MB (compared to ~128.6 MB for float64).
- **Artifact:** `data/processed/features/combined_features.npy`
- **Dimensions:** $(7823, 2057)$
- **Artifact SHA-256:** `fe8af91fd79234a0a5355ebb1ceb2beb07e33c066d99b178c83f58881849360b`

### 4.2 Concatenation Invariance
Unit tests verify with numerical assertions that:
$$\mathbf{X}_{\text{combined}}[:, 0:2048] \equiv \mathbf{X}_{\text{morgan}}$$
$$\mathbf{X}_{\text{combined}}[:, 2048:2057] \approx \mathbf{X}_{\text{descriptors}}$$

### 4.3 Row Mapping & Traceability
Row mappings are persisted in `data/processed/features/feature_row_ids.json` (SHA-256 `4977bf70eeae089754f937cb897a329f28780a1d19d9ca3eecab62269ccf9d5d`), preserving the exact 1-to-1 alignment with the cleaned dataset's `source_row` and `mol_id` fields.

---

## 5. Bemis–Murcko Scaffold-Based Splitting

### 5.1 Why Scaffold Splitting is Essential
In chemical toxicity datasets, naive random splitting yields optimistic performance estimates because congeneric series (molecules differing only by a minor alkyl substituent on an identical chemical core) end up scattered between train and test sets. Under random split, models can "memorize" scaffolds rather than learning generalized structure-activity rules.

Scaffold splitting groups molecules by their Bemis–Murcko scaffold (the two-dimensional ring systems and linkers connecting rings, stripping side chains). Assigning entire scaffold clusters to distinct partitions forces models to predict toxicity on unseen, structurally novel chemical scaffolds, simulating realistic drug discovery screening scenarios.

### 5.2 Acyclic Molecules and Empty Scaffolds
Acyclic compounds (aliphatic chains lacking rings, e.g., ethanol, butanol, EDTA) produce an empty string `""` from RDKit's `MurckoScaffoldSmiles`.
- In Tox21, **1,775 of 7,823 molecules (22.69%)** are acyclic.
- **Policy (`single_group`):** Acyclic molecules are treated as a single unified scaffold group `""`.
- **Impact on Proportions:** By sorting scaffold clusters by size descending, the acyclic group ($1,775$ molecules) and the benzene scaffold (`c1ccccc1`, $1,474$ molecules) are both placed into the training set (target capacity $\approx 80\% = 6,258$). The remaining $2,323$ distinct cyclic scaffolds then populate validation ($\approx 10\% = 782$) and test ($\approx 10\% = 783$).
- This policy ensures that validation and test sets evaluate models exclusively on structurally distinct, non-trivial cyclic scaffolds, without splitting the acyclic cohort into arbitrary fragments.

### 5.3 Measured Split Proportions and Scaffold Accounting
From `data/processed/scaffold_split_indices.json`:

| Partition | Molecule Count | Achieved Proportion | Target Proportion | Unique Scaffold Count | Nonempty Scaffolds |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | **6,258** | **79.995%** | 80.00% | 760 (incl. `""`) | 759 |
| **Validation** | **782** | **9.996%** | 10.00% | 782 | 782 |
| **Test** | **783** | **10.009%** | 10.00% | 783 | 783 |
| **Total** | **7,823** | **100.00%** | 100.00% | **2,325** | **2,324** |

### 5.4 Leakage Prevention Verification
Independent verification in `tests/test_split_integrity.py` confirms:
1. **Index Disjointness:** $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
2. **Partition Completeness:** $\text{Train} \cup \text{Val} \cup \text{Test} = \{0, 1, \dots, 7822\}$.
3. **Scaffold Isolation:** Recomputing Bemis–Murcko scaffolds from scratch with RDKit on all 7,823 molecules reveals **0 scaffold overlap** across partitions:
   - $\text{Scaffolds}_{\text{Train}} \cap \text{Scaffolds}_{\text{Val}} = \emptyset$
   - $\text{Scaffolds}_{\text{Train}} \cap \text{Scaffolds}_{\text{Test}} = \emptyset$
   - $\text{Scaffolds}_{\text{Val}} \cap \text{Scaffolds}_{\text{Test}} = \emptyset$
4. **Zero Label Contamination:** Endpoint toxicity labels were neither read nor used in the splitting algorithm.

---

## 6. How to Reproduce Features and Splits

Execute the following commands in the virtual environment from the repository root:

```bash
# 1. Generate Morgan fingerprints (standalone CLI)
python -m src.features.fingerprints --input data/processed/tox21_cleaned.csv --output data/processed/features/morgan_r2_2048.npy --radius 2 --n-bits 2048

# 2. Generate molecular descriptors (standalone CLI)
python -m src.features.descriptors --input data/processed/tox21_cleaned.csv --output data/processed/features/descriptors_9.npy

# 3. Assemble all 3 feature matrices, row IDs, and manifest
python -m src.features.build_features --input data/processed/tox21_cleaned.csv --output-dir data/processed/features

# 4. Generate deterministic scaffold split
python -m src.data.split_data --input data/processed/tox21_cleaned.csv --output data/processed/scaffold_split_indices.json --seed 42

# 5. Run complete test suite (unit + split integrity)
pytest tests/ -v --tb=short
```

---

## 7. Known Limitations & Recommendations for Student 3 & 4

1. **Out-of-Distribution Shift:** Because scaffold splitting rigorously isolates ring scaffolds, test set performance is expected to be substantially lower than under random split (often 0.05–0.15 lower ROC-AUC). This is chemically realistic and reflects generalizability rather than model failure.
2. **Descriptor Scaling:** The 9 physicochemical descriptors have differing dynamic ranges (e.g., molecular weight $\sim 10^2$, fraction $sp^3$ in $[0, 1]$). **Student 3 must fit scalers (e.g., StandardScaler) strictly on the training partition** and transform validation/test partitions without refitting.
3. **Acyclic Generalization:** Under the `single_group` policy, acyclic molecules reside in the training partition. If the team subsequently requires evaluating acyclic toxicity separately, the modular architecture permits specifying an alternative policy or evaluating an acyclic subset.
