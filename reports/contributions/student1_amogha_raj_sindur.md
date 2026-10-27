# Student 1 Contribution Report: Dataset, Preprocessing and EDA

**Name:** Amogha Raj Sindur  
**Role:** Data Owner  
**Project:** Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks  
**Scope:** Phase 1 — Dataset Acquisition, Sanitization, Multi-Task Duplicate Audit, and Exploratory Data Analysis  

---

## 1. Responsibilities

- Download and verify the upstream NIH Tox21 dataset with cryptographic checksum guarantees.
- Validate chemical structures using RDKit, filter invalid or chemically unparsable representations, and canonicalize valid molecules.
- Audit missing toxicity labels, analyze extreme class imbalance across all 12 bioassays, and verify duplicate group consistency.
- Generate statistical property distributions and publication-quality exploratory data analysis (EDA) figures.
- Document excluded molecules and invalid structures with an auditable exclusion log.

---

## 2. Technical Deliverables

### Source Code (`src/data/`)
- `src/data/download_data.py`: Streaming retrieval of `tox21.csv.gz` from DeepChem with bounded retries and SHA-256 integrity verification (`45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360`).
- `src/data/inspect_dataset.py`: Comprehensive schema validation, label encoding diagnostics, duplicate detection, and missingness quantification.
- `src/data/preprocess.py`: RDKit structure sanitization (`Chem.SanitizeMol`), canonical isomeric SMILES generation, and generation of `data/processed/tox21_cleaned.csv`.
- `src/data/generate_eda_plots.py`: Generates 10 discrete EDA distribution and prevalence figures in both PNG (300 DPI) and PDF vector formats, alongside corresponding numerical JSON summaries.
- `src/data/common.py`: Shared chemical utilities, path resolvers, and logging functions.

### Generated Artifacts
- `data/raw/tox21.csv.gz`: Raw compressed dataset archive ($445,471$ bytes).
- `data/raw/download_provenance.json`: Metadata tracking source URL, retrieval timestamp, and verified checksum.
- `data/processed/tox21_cleaned.csv`: Cleaned dataset containing 7,823 valid molecular structures with canonical SMILES and metadata.
- `data/processed/dataset_manifest.json`: Publication manifest with record counts, MD5/SHA-256 hashes, and configuration state.
- `results/experiment_logs/data_quality_report.json`: Exhaustive statistical report covering schema, labels, and molecular properties.
- `results/experiment_logs/exclusions.csv`: Auditable exclusion log documenting the 8 excluded organometallic structures.
- `results/experiment_logs/duplicates.csv`: Audit record verifying zero duplicate groups and zero contradictory labels.
- `results/figures/eda/`: 10 PNG/PDF figures and 10 numerical JSON files (label missingness, prevalence, MW, heavy atom count, rotatable bonds, rings, atom counts).

---

## 3. Key Findings & Scientific Measurements

- **Raw vs. Cleaned Record Balance:** Exactly 7,831 raw records processed $\to$ 8 organo-aluminum valence violations excluded (`[AlH3]` valence 6) $\to$ exactly 7,823 chemically valid organic molecules.
- **Canonical SMILES Invariance:** 100.0% of cleaned structures re-parse as valid `Mol` objects in RDKit with zero syntax mutations.
- **Label Integrity:** Confirmed 0 canonical duplicate groups and 0 conflicting multi-task labels in the cleaned dataset.
- **Class Imbalance:** Characterized positive prevalence across all 12 endpoints:
  - Highest prevalence: `NR-AhR` (15.82%, 1,029 positive of 6,504 tested).
  - Lowest prevalence: `SR-HSE` (2.84%, 184 positive of 6,470 tested).

---

## 4. Automated Unit & Integration Tests

Authored test modules in `tests/`:
- `tests/test_download_data.py`: Validates URL streaming, mocked HTTP errors, and SHA-256 checksum rejection.
- `tests/test_inspect_dataset.py`: Tests schema enforcement, ragged row handling, non-binary label detection, and real-data accounting.
- `tests/test_preprocess.py`: Tests RDKit sanitization, duplicate resolution policies, and conflict blocking.
- `tests/test_eda.py`: Tests figure generation, symlog ring plot scaling, and numeric JSON serialization.

**Test Execution:** 30 unit and integration tests passing deterministically (`pytest tests/test_inspect_dataset.py tests/test_preprocess.py tests/test_eda.py -v`).
