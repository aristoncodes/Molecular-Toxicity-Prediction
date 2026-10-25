# Tox21 dataset and EDA report

## Source and provenance

Source: [https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz). This is the downloaded DeepChem-hosted artifact; correspondence to any other Tox21 release is not assumed.

Measured compressed-file SHA-256: `45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360`.
Reference SHA-256: `45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360`.
Compressed size: 122925 bytes.
Download timestamp (UTC): 2026-10-25T12:54:45.620291+00:00.
Raw input: `/Users/amogharajsandur/Developer/Molecular-Toxicity-Prediction/data/raw/tox21.csv.gz`.

## Schema and endpoints

Measured dimensions: 7831 rows × 14 columns.
SMILES column: `smiles`.
Columns: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53, mol_id, smiles.
Additional columns are preserved: ['mol_id'].
Only numeric 0 and 1 are accepted endpoint labels. Empty fields remain unknown; tokens such as NA are not silently treated as missing.
Source row identifiers are one-based CSV record numbers after the header, not physical line numbers.

## Structure-validation methodology

RDKit MolFromSmiles uses sanitization; MolToSmiles uses canonical=True and isomericSmiles=True.
Original SMILES and all source metadata are retained. No salt removal, tautomer normalization, charge normalization, or stereochemistry editing is applied.
Canonicalization is deterministic within the recorded RDKit version, not a guarantee across future releases.
Blank, non-string, unparsable and zero-atom structures are excluded with row-level reasons in exclusions.csv.
A parsing failure does not establish that a compound is organometallic.
Measured structure counts: `{"valid_records": 7823, "invalid_records": 8, "unique_canonical_structures": 7823}`.

## Missing labels and class imbalance

Prevalence = positive / (positive + negative), excluding missing and invalid labels. An entirely unobserved endpoint has undefined prevalence.

| Endpoint | Positive | Negative | Missing | Invalid | Observed-label prevalence |
| --- | ---: | ---: | ---: | ---: | ---: |
| NR-AR | 309 | 6956 | 566 | 0 | 4.2533% |
| NR-AR-LBD | 237 | 6521 | 1073 | 0 | 3.5070% |
| NR-AhR | 768 | 5781 | 1282 | 0 | 11.7270% |
| NR-Aromatase | 300 | 5521 | 2010 | 0 | 5.1538% |
| NR-ER | 793 | 5400 | 1638 | 0 | 12.8048% |
| NR-ER-LBD | 350 | 6605 | 876 | 0 | 5.0324% |
| NR-PPAR-gamma | 186 | 6264 | 1381 | 0 | 2.8837% |
| SR-ARE | 942 | 4890 | 1999 | 0 | 16.1523% |
| SR-ATAD5 | 264 | 6808 | 759 | 0 | 3.7330% |
| SR-HSE | 372 | 6095 | 1364 | 0 | 5.7523% |
| SR-MMP | 918 | 4892 | 2021 | 0 | 15.8003% |
| SR-p53 | 423 | 6351 | 1057 | 0 | 6.2445% |

Counts above use all raw source records; downstream cleaned-data counts may differ after structural exclusions.

## Duplicate and conflicting-label findings

Measured duplicate audit: `{"raw_groups": 0, "raw_records": 0, "canonical_groups": 0, "canonical_records": 0, "canonical_excess_records": 0, "conflicting_groups": 0, "conflicting_records": 0, "record_status_counts": {}}`.
Groups use canonical isomeric SMILES. All valid source records are retained when publication succeeds; the output is not one row per unique canonical structure.
The audit distinguishes identical records, identical label vectors with differing source representations/metadata, compatible missingness, and observed 0/1 conflicts.
Missing versus observed is not a contradiction, and is never imputed or merged. Any observed 0/1 conflict blocks cleaned-data publication for review.
See duplicates.csv for affected source rows, group IDs and conflicting endpoints. Duplicate counts overlap with conflict counts and must not be added together.

## Molecular-property distributions

Properties are calculated on valid raw source records, including repeated structures, without hydrogen addition.
Atom count = RDKit GetNumAtoms (represented atoms, not total implicit hydrogens); heavy atoms = GetNumHeavyAtoms.
Molecular weight = Descriptors.MolWt in g/mol using RDKit's average atomic weights.
Ring count = Lipinski.RingCount using RDKit ring information; rotatable bonds = Lipinski.NumRotatableBonds with its installed-version defaults.
Ring count is an algorithm-defined measure, not necessarily the number of independent cycles. Symmetric cyclic structures can yield large counts; no such outliers are silently removed.
Count histograms use unit-width, integer-centered bins. When the maximum ring count exceeds 20, its x-axis uses symlog above 10 to show the main distribution and full tail.
These properties are EDA summaries only, not a descriptor feature pipeline.

| Property | Sample size | Mean | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| atom_count | 7823 | 18.568 | 16.000 | 1.000 | 132.000 |
| molecular_weight | 7823 | 276.144 | 240.302 | 9.012 | 1877.664 |
| heavy_atom_count | 7823 | 18.567 | 16.000 | 1.000 | 132.000 |
| ring_count | 7823 | 1.824 | 1.000 | 0.000 | 264.000 |
| rotatable_bond_count | 7823 | 4.303 | 3.000 | 0.000 | 47.000 |

Figures: generated in results/figures/eda, with PNG/PDF and per-figure numerical JSON.

## Exclusion accounting

Measured accounting: `{"input": 7831, "invalid_exclusions": 8, "valid_candidates": 7823, "reconciled": true}`.
Preprocessing publication status: **success**.
Published output records: 7823.

Blocked runs publish no new cleaned CSV; any older CSV is stale and cannot be consumed through the EDA command.

## Limitations and data-quality risks

Missing labels remain unknown; prevalence is descriptive of observed labels, not population toxicity risk.
Canonical SMILES do not resolve tautomeric equivalence or parent/salt equivalence. Retained duplicates weight record-level distributions more heavily.
The upstream artifact's assay metadata, measurement uncertainty, and original experimental provenance are not independently validated here.
Do not describe the dataset as conflict-free or ready for modeling unless the corresponding checks succeed.
Test and environment evidence belongs in README; dataset measurements above come from the saved quality report.

## Implications for scaffold-based splitting

Student 2 must keep all records of each canonical structure together before assigning scaffold groups or splits.
Otherwise repeated structures can leak across partitions. Conflicting labels need a reviewed policy before splitting.
No train/validation/test split, fingerprints, model features, models, or evaluation are implemented here.

## Reproduction

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.data.download_data
python -m src.data.inspect_dataset
python -m src.data.preprocess
python -m src.data.generate_eda_plots
pytest tests/ -v
python -m pytest tests/ -v --run-real-data
```

Stop downstream commands after a fatal error. All module commands accept `--config PATH`.
