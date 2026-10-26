# Molecular-Toxicity-Prediction

Student 1 — Dataset validation, preprocessing, and EDA

Auditable Tox21 acquisition, structure validation, duplicate/label audits, and EDA for the B.Tech project **Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks**.

This module does not implement fingerprints, model descriptors, splitting, classifiers, GNNs, or model evaluation. The five requested molecular-property summaries are calculated only for EDA.

## Setup

Run from the repository root. Python is available on the development machine as `python3`; activation below supplies `python` and `pytest`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins the complete installed environment. See the validation record below for the environment actually exercised; other Python versions and platforms are unverified.

## Reproduce

Run commands in order, stopping after any fatal error:

```bash
python -m src.data.download_data
python -m src.data.inspect_dataset
python -m src.data.preprocess
python -m src.data.generate_eda_plots
pytest tests/ -v
```

Every module accepts `--config PATH`; relative artifact paths in YAML resolve against the repository root. Configuration is in `configs/data_config.yaml`. No random transformations are used; the seed is recorded for reproducibility.

The first download needs network access. Later runs validate and reuse the raw artifact without fetching it. Unit tests use synthetic fixtures and mock HTTP responses, and never need the internet. To also audit a locally downloaded real artifact:

```bash
python -m pytest tests/ -v --run-real-data
```

The real-data test skips explicitly when the artifact is unavailable. It validates provenance, schema, label encodings, and record accounting; it does not assume that the dataset has no duplicate conflicts.

## Scientific policies

- Numeric `0` is negative, numeric `1` is positive, and an empty field is unknown. Unknown labels are never converted to negative. Unexpected encodings block preprocessing and are recorded by row/endpoint.
- Prevalence is `positive / (positive + negative)`. All-unlabelled endpoints have undefined prevalence, serialized as JSON `null`.
- RDKit sanitizes structures and generates canonical isomeric SMILES. Original SMILES and metadata remain available. No salt stripping, tautomer normalization, charge neutralization, or stereochemistry editing is performed.
- Invalid structures are excluded with row identifiers and reasons; a parse failure is not evidence of an organometallic compound.
- Canonical duplicates are audited across every endpoint. All valid source records are retained on success. Identical labels, different metadata/representations, compatible missingness, and observed 0/1 conflicts are distinguished. No labels are merged or selected.
- **Any observed-label conflict blocks publication of a new cleaned CSV.** Review `duplicates.csv` and agree a scientific policy with the team before changing this behavior. The cleaned dataset is not guaranteed to contain one row per unique canonical structure.
- A blocked preprocessing run writes a blocked manifest. Older cleaned files may remain for auditability, but are stale; EDA verifies the manifest, current configuration, and artifact checksums before consuming anything.
- Student 2 must keep all canonical duplicates together before scaffold splitting to avoid leakage. This module makes no split assignments.

## Outputs

| Location | Contents |
| --- | --- |
| `data/raw/tox21.csv.gz` | Verified upstream compressed artifact |
| `data/raw/download_provenance.json` | Source URL, UTC time, compressed size, SHA-256 |
| `results/experiment_logs/data_quality_report.json` | Schema, dtypes, labels, prevalence, properties, duplicates, counts |
| `results/experiment_logs/exclusions.csv` | Invalid source rows, original SMILES, RDKit/exclusion reasons |
| `results/experiment_logs/duplicates.csv` | Duplicate groups, affected source rows, conflict endpoints |
| `results/experiment_logs/structure_properties.csv` | Valid-row EDA measurements keyed by source row |
| `data/processed/tox21_cleaned.csv` | Valid source rows with canonical SMILES and audit columns; success only |
| `data/processed/dataset_manifest.json` | Publication status, policies, counts, checksums, versions, configuration |
| `results/figures/eda/` | Ten PNG/PDF figures and one numerical JSON per figure; success only |
| `reports/dataset_eda_report.md` | Generated measured report, including blocked/pending sections |

Audit source-row numbers are one-based CSV record numbers after the header. Blank endpoint fields in CSV remain missing. New cleaned columns are `source_row`, `canonical_smiles`, `duplicate_group`, and `duplicate_status`; inputs using those names are rejected rather than overwritten. All other original columns are preserved.

Raw and generated data artifacts are ignored by Git; regenerate them with the commands above. The generated research report is versioned. There are no secrets or environment files to commit.

## Download integrity and failures

Source: [DeepChem-hosted Tox21 gzip](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz).
Reference SHA-256: `45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360`.

Downloads stream to temporary files with timeouts and bounded HTTP retries. The entire gzip is read to validate its trailer/CRC before atomic publication. Failed transfers never become accepted raw artifacts. Existing artifacts are validated and never silently replaced.

A different checksum is quarantined under a `.candidate.<sha256>` filename with provenance and CSV header/row evidence. It is not accepted as the raw dataset. The checksum gate can only be changed through an explicit, documented configuration review. There is no automatic bypass. If an existing file has no provenance, its historical download timestamp is recorded as unknown, not invented.

## Figure interpretation

Raw endpoint counts and prevalence use all raw source records. Molecular distributions use valid source records and retain duplicate weighting, with sample sizes in titles and JSON. Atom counts refer to represented atoms after RDKit parsing, not implicit hydrogen totals. Molecular weight is in g/mol. Heavy atoms, rings, and rotatable bonds use the RDKit methods documented in the generated report. Duplicate and conflict counts overlap; conflicts are a subset, not an additional disjoint category.

## Validation record

Validated on Python 3.14.2, Darwin arm64. Tested primary versions: pandas 3.1.0, NumPy 2.5.4, RDKit 2026.9.1, Matplotlib 3.11.2, PyYAML 6.0.3, requests 2.34.2, pytest 9.1.1. Other environments remain unverified.

Measured source: **7,831 rows, 14 columns**, **7,823 valid structures**, **8 exclusions**. No duplicate raw SMILES, canonical duplicate groups, conflicting duplicate labels, or invalid endpoint encodings were found in this artifact under the recorded policy and RDKit version. The compressed checksum exactly matched the reference. The cleaned CSV contains 7,823 source records, and all ten PNG/PDF figures and underlying JSON statistics were generated.

Executed validation:

- All four module commands above completed successfully on the real dataset.
- `python -m pytest tests/ -v`: 29 passed, 1 explicitly skipped real-data test.
- `python -m pytest tests/ -v --run-real-data`: **30 passed**, including the real-data audit.
- `python -m compileall -q src tests`: successful.
- `python -m pip check`: no broken requirements.
- All ten generated figures were visually inspected; counts and histogram totals were reconciled with the audited populations.

The ring-count distribution contains large algorithm-defined values for symmetric cyclic structures (maximum 264 in this RDKit version). These values are retained. The ring plot uses an explicitly labelled symlog x-axis above 10; they must not be interpreted as independent cycle counts.

---

# Student 2 — Molecular Features and Scaffold Splitting

Implementation of circular chemical fingerprints, physicochemical molecular descriptors, combined feature representations, and Bemis–Murcko scaffold-based dataset splitting for **Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks**.

## Student 2 Architecture & Responsibilities

1. **Morgan Fingerprints (`src/features/fingerprints.py`)**: Circular topological fingerprints computed using modern RDKit `rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)`. Produces binary bit-vectors (dtype `uint8`) and optional count fingerprints (dtype `uint32`).
2. **Molecular Descriptors (`src/features/descriptors.py`)**: Calculates 9 standard physicochemical descriptors (`mol_weight`, `logp`, `tpsa`, `num_h_donors`, `num_h_acceptors`, `num_rotatable_bonds`, `ring_count`, `heavy_atom_count`, `fraction_csp3`).
3. **Combined Representation (`src/features/build_features.py`)**: Assembles Morgan fingerprints, 9 descriptors, and concatenated representations (2,057 dimensions, float32) with row alignment (`feature_row_ids.json`) and audit manifest (`feature_manifest.json`).
4. **Scaffold Splitting (`src/data/split_data.py`)**: Deterministic Bemis–Murcko scaffold grouping (`target: 80% train, 10% val, 10% test`, seed `42`). Guarantees zero scaffold leakage across partitions.
5. **Split Integrity Tests (`tests/test_split_integrity.py`)**: Independent test suite validating partition disjointness, completeness, row alignment, and zero scaffold overlap.

## Reproduce Feature Extraction and Splitting

Run in virtual environment from repository root:

```bash
# 1. Standalone Morgan fingerprint generation
python -m src.features.fingerprints --input data/processed/tox21_cleaned.csv --output data/processed/features/morgan_r2_2048.npy --radius 2 --n-bits 2048

# 2. Standalone molecular descriptor calculation
python -m src.features.descriptors --input data/processed/tox21_cleaned.csv --output data/processed/features/descriptors_9.npy

# 3. Assemble all 3 feature matrices, row IDs, and manifest
python -m src.features.build_features --input data/processed/tox21_cleaned.csv --output-dir data/processed/features

# 4. Deterministic scaffold splitting
python -m src.data.split_data --input data/processed/tox21_cleaned.csv --output data/processed/scaffold_split_indices.json --seed 42

# 5. Execute full test suite
pytest tests/ -v --tb=short
```

## Student 2 Outputs & Manifest

| Location | Shape / Contents | Dtype | Description |
| :--- | :---: | :---: | :--- |
| `data/processed/features/morgan_r2_2048.npy` | $(7823, 2048)$ | `uint8` | Binary Morgan fingerprints ($r=2$, ECFP4 equivalent) |
| `data/processed/features/descriptors_9.npy` | $(7823, 9)$ | `float64` | 9 physicochemical molecular descriptors |
| `data/processed/features/combined_features.npy` | $(7823, 2057)$ | `float32` | Concatenation of Morgan bits and 9 descriptors |
| `data/processed/features/feature_row_ids.json` | 7,823 records | JSON | Alignment mapping of `source_row` and `mol_id` |
| `data/processed/features/feature_manifest.json` | Metadata & hashes | JSON | Checksums, dimensions, and parameter provenance |
| `data/processed/scaffold_split_indices.json` | Train/Val/Test indices | JSON | Disjoint scaffold split indices and row IDs |
| `reports/student2_integration_audit.md` | Verification report | Markdown | Upstream audit and verification of Student 1 |
| `reports/student2_features_and_splitting.md` | Research report | Markdown | Technical report on features and scaffold splitting |

## Measured Scaffold Split Proportions (Seed 42)

- **Total Valid Molecules:** 7,823
- **Total Unique Scaffolds:** 2,325 (Acyclic empty scaffold `""`: 1,775 molecules)
- **Train Partition:** 6,258 molecules (**79.995%**) — 760 scaffolds
- **Validation Partition:** 782 molecules (**9.996%**) — 782 scaffolds
- **Test Partition:** 783 molecules (**10.009%**) — 783 scaffolds
- **Scaffold Leakage:** **0** overlapping scaffolds across train, validation, and test partitions.

---

# Student 3 — Independent Phase 1 Auditor and Contributor

Independent verification, audit suite, leakage prevention, and evaluation methodology for **Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks**.

## Student 3 Architecture & Deliverables

1. **Independent Dataset Audit (`src/audit/dataset_audit.py`)**: Re-verifies raw gzip SHA-256 (`45d09792...`), record accounting ($7823 + 8 = 7831$), canonical SMILES invariance (100.0% valid in RDKit), 0 duplicate conflicts, and label distributions.
2. **Split & Scaffold Isolation Audit (`src/audit/split_audit.py`)**: Mathematically validates split disjointness, complete coverage ($0 \dots 7822$), zero scaffold overlap via independent RDKit Bemis--Murcko extraction, and the `single_group` acyclic policy (1,775 acyclic molecules placed strictly in train).
3. **Leakage-Safe Preprocessor (`src/features/preprocessor.py`)**: Resolves preprocessing leakage by fitting feature scaling (`LeakageSafeScaler`, `PartitionPreprocessor`) strictly on training partition indices, preserving binary Morgan bits without scaling.
4. **Evaluation Metrics & Uncertainty Suite (`src/evaluation/`)**:
   - `src/evaluation/metrics.py`: Continuous ROC-AUC and PR-AUC (Average Precision), per-endpoint missing label masking, and single-class endpoint handling.
   - `src/evaluation/aggregation.py`: Macro averaging with explicit uncertainty notation: $\text{Mean} \pm \text{SEM}$ ($K=12$), $\text{SD}$, and 95% Student-$t$ confidence intervals.
5. **Unified Automated Audit CLI (`src/audit/run_audit.py`)**: One-command reproducibility check executing all 7 criteria and writing `results/experiment_logs/audit_results.json`.
6. **IEEE Manuscript & Reconciliation (`reports/`)**: Formal IEEE conference manuscript (`reports/ieee_phase1_manuscript.tex`) and claim reconciliation matrix (`reports/ieee_reconciliation.md`).

## Reproduce Student 3 Audit and Verification

Run from repository root in virtual environment:

```bash
# 1. Execute unified Phase 1 scientific audit suite
python -m src.audit.run_audit

# 2. Run complete test suite (all 75 tests)
pytest tests/ -v --tb=short --run-real-data
```

