# Molecular Toxicity Prediction — Student 1

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

Implementation validation is in progress. Final test results and measured real-data status will replace this paragraph after execution; no test outcome is claimed yet.
