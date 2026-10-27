# Tox21 Data Directory and Provenance Guide

This directory manages both raw and processed artifacts for the **Molecular Toxicity Prediction via Fingerprints and Molecular Graph Networks** project.

In accordance with scientific reproducibility standards, large raw binaries and processed numpy feature matrices are managed deterministically via automated pipeline scripts and checksum verification.

---

## 1. Directory Structure

```
data/
├── README.md                          # Data documentation, checksums, and provenance guide
├── raw/
│   ├── download_provenance.json       # Source URL, download timestamp, size, and SHA-256
│   └── tox21.csv.gz                   # Raw compressed dataset from DeepChem / NIH Tox21
└── processed/
    ├── tox21_cleaned.csv              # Sanitized valid molecules (7,823 rows) with canonical SMILES
    ├── dataset_manifest.json          # Processing manifest, checksums, and exclusion accounting
    ├── scaffold_split_indices.json    # Bemis-Murcko scaffold train/val/test split indices
    └── features/
        ├── morgan_r2_2048.npy         # Binary Morgan fingerprints (7823, 2048), radius=2, uint8
        ├── descriptors_9.npy          # 9 Physicochemical descriptors (7823, 9), float64
        ├── combined_features.npy      # Concatenated feature matrix (7823, 2057), float32
        ├── feature_row_ids.json       # Alignment mapping to source CSV rows
        └── feature_manifest.json      # Feature generation parameters, shapes, and SHA-256 hashes
```

---

## 2. Dataset Provenance & Integrity

- **Upstream Source:** DeepChem S3 Mirror (NIH Tox21 Challenge Data)
- **Direct Download URL:** `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz`
- **Expected SHA-256 Checksum:** `45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360`
- **Raw Dimensions:** 7,831 rows × 14 columns (`formula`, 12 assay endpoints, `smiles`)
- **Cleaned Dimensions:** 7,823 rows (exactly 8 structures excluded due to organometallic valence violations `[AlH3]`)
- **Endpoint Labels:** Binary toxicity indicators across 12 targets:
  - Nuclear Receptor (NR) Signaling: `NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`, `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`
  - Stress Response (SR) Pathways: `SR-ARE`, `SR-ATAD5`, `SR-HSE`, `SR-MMP`, `SR-p53`
  - Values: `1.0` (active/toxic), `0.0` (inactive/non-toxic), or empty/NaN (untested/missing).

---

## 3. Reproduction Commands

To acquire data and reproduce all preprocessing, features, and scaffold splits:

```bash
# 1. Download and verify raw Tox21 gzip archive
python -m src.data.download_data

# 2. Inspect raw data schema and label distributions
python -m src.data.inspect_dataset

# 3. Sanitize structures and generate cleaned CSV
python -m src.data.preprocess

# 4. Generate 10 EDA distribution plots
python -m src.data.generate_eda_plots

# 5. Extract Morgan fingerprints, descriptors, and combined matrices
python -m src.features.build_features

# 6. Generate Bemis-Murcko scaffold split indices
python -m src.data.split_data --seed 42
```

---

## 4. Scaffold Split Summary (Seed 42)

- **Total Valid Molecules:** 7,823
- **Unique Scaffolds:** 2,325
- **Train Partition:** 6,258 molecules (79.995%) — 760 cyclic scaffolds + acyclic group
- **Validation Partition:** 782 molecules (9.996%) — 782 unique scaffolds
- **Test Partition:** 783 molecules (10.009%) — 783 unique scaffolds
- **Scaffold Leakage:** Exact **0** overlapping scaffolds between any pair of partitions.
