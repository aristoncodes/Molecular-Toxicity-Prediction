"""Shared configuration and atomic artifact I/O."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PATH_KEYS = ("raw_path", "provenance_path", "processed_path", "manifest_path",
             "quality_report_path", "exclusion_path", "duplicate_path", "properties_path",
             "figures_path", "report_path")


def load_config(path: str | Path | None = None) -> dict:
    """Resolve all artifact paths relative to the repository, never the caller's cwd."""
    with open(path or ROOT / "configs/data_config.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if len(config["endpoints"]) != 12 or len(set(config["endpoints"])) != 12:
        raise ValueError("Configuration must specify 12 distinct endpoints")
    if config["canonicalization"] != {"canonical": True, "isomericSmiles": True}:
        raise ValueError("Only canonical isomeric SMILES policy is supported")
    for key in PATH_KEYS:
        config[key] = str((ROOT / config[key]).resolve())
    return config


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def versions() -> dict:
    return {"python": sys.version.split()[0], **{
        name: importlib.metadata.version(name)
        for name in ("pandas", "numpy", "rdkit", "matplotlib", "PyYAML", "requests", "pytest")}}


def json_safe(value: Any) -> Any:
    """Normalize undefined numbers to null and reject unsupported JSON values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is pd.NA or value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write(path: str | Path, content: str | bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content.encode("utf-8") if isinstance(content, str) else content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: str | Path, value: Any) -> None:
    atomic_write(path, json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def read_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    atomic_write(path, frame.to_csv(index=False, lineterminator="\n", na_rep=""))


def cli(action: Callable[[dict], Any]) -> None:
    parser = argparse.ArgumentParser(description=action.__doc__)
    parser.add_argument("--config", type=Path, help="YAML configuration (default: configs/data_config.yaml)")
    args = parser.parse_args()
    try:
        action(load_config(args.config))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


def write_report(config: dict, report: dict, manifest: dict | None = None, eda_complete: bool = False) -> None:
    """Build the research narrative only from measured audit artifacts."""
    source = report.get("source", {})
    labels = report.get("labels", {})
    label_table = "\n".join(
        f"| {name} | {s['positive']} | {s['negative']} | {s['missing']} | {s['invalid']} | "
        + (f"{s['positive_prevalence']:.4%}" if s['positive_prevalence'] is not None else "undefined") + " |"
        for name, s in labels.items())
    stats = report.get("structures", {})
    duplicates = report.get("duplicates", {})
    summaries = report.get("property_summary", {})
    property_table = "\n".join(
        f"| {name} | {s['count']:.0f} | {s['mean']:.3f} | {s['50%']:.3f} | {s['min']:.3f} | {s['max']:.3f} |"
        for name, s in summaries.items() if s.get("count", 0) > 0)
    publication = manifest.get("status", "pending") if manifest else "pending"
    text = f"""# Tox21 dataset and EDA report

## Source and provenance

Source: [{config['dataset_url']}]({config['dataset_url']}). This is the downloaded DeepChem-hosted artifact; correspondence to any other Tox21 release is not assumed.

Measured compressed-file SHA-256: `{source.get('sha256', 'pending')}`.
Reference SHA-256: `{config['reference_sha256']}`.
Compressed size: {source.get('size_bytes', 'pending')} bytes.
Download timestamp (UTC): {source.get('downloaded_at_utc', 'pending')}.
Raw input: `{config['raw_path']}`.

## Schema and endpoints

Measured dimensions: {report.get('rows', 'pending')} rows × {report.get('columns_count', 'pending')} columns.
SMILES column: `{report.get('smiles_column', 'pending')}`.
Columns: {', '.join(report.get('columns', [])) or 'pending'}.
Additional columns are preserved: {report.get('extra_columns', 'pending')}.
Only numeric 0 and 1 are accepted endpoint labels. Empty fields remain unknown; tokens such as NA are not silently treated as missing.
Source row identifiers are one-based CSV record numbers after the header, not physical line numbers.

## Structure-validation methodology

RDKit MolFromSmiles uses sanitization; MolToSmiles uses canonical=True and isomericSmiles=True.
Original SMILES and all source metadata are retained. No salt removal, tautomer normalization, charge normalization, or stereochemistry editing is applied.
Canonicalization is deterministic within the recorded RDKit version, not a guarantee across future releases.
Blank, non-string, unparsable and zero-atom structures are excluded with row-level reasons in exclusions.csv.
A parsing failure does not establish that a compound is organometallic.
Measured structure counts: `{json.dumps(json_safe(stats), allow_nan=False)}`.

## Missing labels and class imbalance

Prevalence = positive / (positive + negative), excluding missing and invalid labels. An entirely unobserved endpoint has undefined prevalence.

| Endpoint | Positive | Negative | Missing | Invalid | Observed-label prevalence |
| --- | ---: | ---: | ---: | ---: | ---: |
{label_table or '| Pending | | | | | |'}

Counts above use all raw source records; downstream cleaned-data counts may differ after structural exclusions.

## Duplicate and conflicting-label findings

Measured duplicate audit: `{json.dumps(json_safe(duplicates), allow_nan=False)}`.
Groups use canonical isomeric SMILES. All valid source records are retained when publication succeeds; the output is not one row per unique canonical structure.
The audit distinguishes identical records, identical label vectors with differing source representations/metadata, compatible missingness, and observed 0/1 conflicts.
Missing versus observed is not a contradiction, and is never imputed or merged. Any observed 0/1 conflict blocks cleaned-data publication for review.
See duplicates.csv for affected source rows, group IDs and conflicting endpoints. Duplicate counts overlap with conflict counts and must not be added together.

## Molecular-property distributions

Properties are calculated on valid raw source records, including repeated structures, without hydrogen addition.
Atom count = RDKit GetNumAtoms (represented atoms, not total implicit hydrogens); heavy atoms = GetNumHeavyAtoms.
Molecular weight = Descriptors.MolWt in g/mol using RDKit's average atomic weights.
Ring count = Lipinski.RingCount; rotatable bonds = Lipinski.NumRotatableBonds with its installed-version defaults.
These properties are EDA summaries only, not a descriptor feature pipeline.

| Property | Sample size | Mean | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
{property_table or '| Pending | | | | | |'}

Figures: {'generated in results/figures/eda, with PNG/PDF and per-figure numerical JSON' if eda_complete else 'pending; a successful preprocessing manifest is required for the EDA command'}.

## Exclusion accounting

Measured accounting: `{json.dumps(json_safe(report.get('accounting', {})), allow_nan=False)}`.
Preprocessing publication status: **{publication}**.
Published output records: {manifest.get('output_records', 'pending') if manifest else 'pending'}.
{manifest.get('error', '') if manifest else 'Run preprocessing to establish publication status.'}
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
"""
    atomic_write(config["report_path"], text)
