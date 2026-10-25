"""Inspect labels and structures without resolving duplicates or modifying chemistry."""
from __future__ import annotations

import csv
import gzip
import io
import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Lipinski

from .common import cli, sha256, write_csv, write_json, write_report
from .download_data import verify_raw

AUDIT_COLUMNS = {"source_row", "canonical_smiles", "duplicate_group", "duplicate_status"}
PROPERTY_COLUMNS = ["source_row", "atom_count", "molecular_weight", "heavy_atom_count", "ring_count", "rotatable_bond_count"]
DUPLICATE_COLUMNS = ["source_row", "canonical_smiles", "duplicate_group", "duplicate_status", "conflicting_endpoints"]


def validate_schema(columns: list[str], config: dict) -> str:
    """Require distinct headers, all endpoints, and an identifiable SMILES column."""
    if len(columns) != len(set(columns)) or any(not c.strip() for c in columns):
        raise ValueError("CSV headers must be nonempty and unique")
    collision = AUDIT_COLUMNS.intersection(columns)
    if collision:
        raise ValueError(f"Input uses reserved audit columns: {sorted(collision)}")
    missing = sorted(set(config["endpoints"]) - set(columns))
    if missing:
        raise ValueError(f"Missing endpoints: {missing}")
    configured = config.get("smiles_column")
    if configured in columns:
        return configured
    matches = [c for c in columns if c.casefold() == "smiles"]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify an unambiguous SMILES column: {matches}")
    return matches[0]


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Preserve literal text and distinguish blank fields from tokens such as NA."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            columns = next(reader)
        except StopIteration as error:
            raise ValueError("Dataset is empty") from error
        rows = []
        for row_number, row in enumerate(reader, start=1):
            if len(row) != len(columns):
                raise ValueError(f"Source row {row_number}: expected {len(columns)} fields; found {len(row)}")
            rows.append(row)
    return pd.DataFrame(rows, columns=columns).replace("", pd.NA)


def normalize_labels(frame: pd.DataFrame, endpoints: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Only observed numeric zero/one are valid; missing remains missing."""
    normalized = frame.copy()
    errors = []
    for endpoint in endpoints:
        values = pd.to_numeric(frame[endpoint], errors="coerce")
        invalid = frame[endpoint].notna() & ~values.isin([0, 1])
        for index in frame.index[invalid]:
            errors.append({"source_row": int(index) + 1, "endpoint": endpoint,
                           "value": str(frame.at[index, endpoint])})
        normalized[endpoint] = values.where(values.isin([0, 1])).astype("Float64")
    return normalized, errors


def label_statistics(frame: pd.DataFrame, endpoints: list[str], errors: list[dict]) -> dict:
    result = {}
    for endpoint in endpoints:
        values = frame[endpoint]
        positive, negative = int(values.eq(1).sum()), int(values.eq(0).sum())
        invalid = sum(e["endpoint"] == endpoint for e in errors)
        observed = positive + negative
        result[endpoint] = {"positive": positive, "negative": negative,
                            "missing": int(values.isna().sum()) - invalid, "invalid": invalid,
                            "observed": observed, "positive_prevalence": positive / observed if observed else None}
    return result


def parse_structure(value: object) -> tuple[object, str | None, str | None]:
    """Sanitize and canonicalize without salt removal, tautomerization, or stereo edits."""
    if value is None or value is pd.NA or (not isinstance(value, str) and pd.isna(value)):
        return None, None, "missing_smiles"
    if not isinstance(value, str):
        return None, None, "non_string_smiles"
    if not value.strip():
        return None, None, "empty_smiles"
    rdBase.LogToPythonLogger()
    logger = logging.getLogger("rdkit")
    capture = io.StringIO()
    handler = logging.StreamHandler(capture)
    previous_handlers, previous_propagate = logger.handlers[:], logger.propagate
    logger.handlers, logger.propagate = [handler], False
    try:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            return None, None, "unparsable_smiles: " + (capture.getvalue().strip() or "RDKit returned no molecule")
        if molecule.GetNumAtoms() == 0:
            return None, None, "zero_atom_structure"
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        return molecule, canonical, None
    except (ValueError, RuntimeError) as error:
        return None, None, f"rdkit_error: {error}"
    finally:
        logger.handlers, logger.propagate = previous_handlers, previous_propagate
        handler.close()


def structure_properties(molecule: object) -> dict:
    """EDA-only RDKit properties; these are not exported as ML feature vectors."""
    return {"atom_count": molecule.GetNumAtoms(), "molecular_weight": Descriptors.MolWt(molecule),
            "heavy_atom_count": molecule.GetNumHeavyAtoms(), "ring_count": Lipinski.RingCount(molecule),
            "rotatable_bond_count": Lipinski.NumRotatableBonds(molecule)}


def duplicate_audit(valid: pd.DataFrame, original_columns: list[str], endpoints: list[str]) -> pd.DataFrame:
    records = []
    groups = valid.groupby("canonical_smiles", sort=True)
    group_number = 0
    for canonical, group in groups:
        if len(group) < 2:
            continue
        group_number += 1
        conflicts = [endpoint for endpoint in endpoints if group[endpoint].dropna().nunique() > 1]
        same_labels = len(group[endpoints].drop_duplicates()) == 1
        same_metadata = len(group[original_columns].drop_duplicates()) == 1
        status = ("conflicting_labels" if conflicts else "identical_record" if same_labels and same_metadata
                  else "identical_labels" if same_labels else "compatible_missingness")
        for row in group["source_row"]:
            records.append({"source_row": int(row), "canonical_smiles": canonical,
                            "duplicate_group": f"D{group_number:05d}", "duplicate_status": status,
                            "conflicting_endpoints": ";".join(conflicts)})
    return pd.DataFrame(records, columns=DUPLICATE_COLUMNS)


def analyze(frame: pd.DataFrame, config: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return quality report, valid records, exclusions, duplicate audit, properties."""
    frame = frame.reset_index(drop=True)
    smiles = validate_schema(list(frame.columns), config)
    original_columns = list(frame.columns)
    normalized, errors = normalize_labels(frame, config["endpoints"])
    valid_records, exclusions, properties = [], [], []
    for index, row in normalized.iterrows():
        source_row = int(index) + 1
        molecule, canonical, reason = parse_structure(row[smiles])
        if reason:
            exclusions.append({"source_row": source_row, "original_smiles": row[smiles], "reason": reason})
            continue
        valid_records.append({**row.to_dict(), "source_row": source_row, "canonical_smiles": canonical})
        properties.append({"source_row": source_row, **structure_properties(molecule)})
    valid = pd.DataFrame(valid_records, columns=original_columns + ["source_row", "canonical_smiles"])
    excluded = pd.DataFrame(exclusions, columns=["source_row", "original_smiles", "reason"])
    props = pd.DataFrame(properties, columns=PROPERTY_COLUMNS)
    duplicates = duplicate_audit(valid, original_columns, config["endpoints"])
    conflict_rows = duplicates.loc[duplicates.duplicate_status.eq("conflicting_labels")]
    raw_counts = frame[smiles].dropna().value_counts()
    report = {
        "status": "invalid_labels" if errors else "conflicts_require_review" if len(conflict_rows) else "audited",
        "rows": len(frame), "columns_count": len(frame.columns), "columns": original_columns,
        "inferred_dtypes": {c: str(pd.to_numeric(frame[c], errors="raise").dtype)
                            if c in config["endpoints"] and not any(e["endpoint"] == c for e in errors)
                            else str(frame[c].infer_objects().dtype) for c in frame},
        "smiles_column": smiles, "expected_endpoints_present": True,
        "extra_columns": [c for c in original_columns if c not in [smiles, *config["endpoints"]]],
        "missing_per_column": frame.isna().sum().to_dict(),
        "unique_raw_smiles": int(frame[smiles].nunique()),
        "labels": label_statistics(normalized, config["endpoints"], errors), "invalid_labels": errors,
        "structures": {"valid_records": len(valid), "invalid_records": len(excluded),
                       "unique_canonical_structures": int(valid.canonical_smiles.nunique())},
        "duplicates": {"raw_groups": int(raw_counts.gt(1).sum()),
                       "raw_records": int(raw_counts[raw_counts.gt(1)].sum()),
                       "canonical_groups": int(duplicates.duplicate_group.nunique()),
                       "canonical_records": len(duplicates),
                       "canonical_excess_records": len(duplicates) - int(duplicates.duplicate_group.nunique()),
                       "conflicting_groups": int(conflict_rows.duplicate_group.nunique()),
                       "conflicting_records": len(conflict_rows),
                       "record_status_counts": duplicates.duplicate_status.value_counts().to_dict()},
        "property_summary": {column: props[column].describe().to_dict() for column in PROPERTY_COLUMNS[1:]},
        "accounting": {"input": len(frame), "invalid_exclusions": len(excluded),
                       "valid_candidates": len(valid), "reconciled": len(frame) == len(excluded) + len(valid)},
    }
    return report, valid, excluded, duplicates, props


def inspect(config: dict) -> tuple:
    """Audit the verified raw dataset and save machine-readable and row-level evidence."""
    provenance = verify_raw(config)
    try:
        result = analyze(read_dataset(config["raw_path"]), config)
    except (ValueError, csv.Error) as error:
        write_json(config["quality_report_path"], {"status": "schema_error", "error": str(error),
                                                 "input_sha256": sha256(config["raw_path"])})
        raise ValueError(str(error)) from error
    report, valid, exclusions, duplicates, properties = result
    report.update({"input_sha256": provenance["sha256"], "source": provenance,
                   "canonicalization": config["canonicalization"]})
    write_csv(config["exclusion_path"], exclusions)
    write_csv(config["duplicate_path"], duplicates)
    write_csv(config["properties_path"], properties)
    write_json(config["quality_report_path"], report)
    write_report(config, report)
    print(f"Raw: {report['rows']} rows × {report['columns_count']} columns; SMILES: {report['smiles_column']}")
    print(f"Structures: {len(valid)} valid, {len(exclusions)} excluded; duplicates: {report['duplicates']}")
    for endpoint, counts in report["labels"].items():
        print(f"{endpoint}: {counts}")
    print(f"Audit status: {report['status']}; report: {config['quality_report_path']}")
    if report["invalid_labels"]:
        raise ValueError("Unexpected endpoint encodings; see invalid_labels in quality report")
    return result


if __name__ == "__main__":
    cli(inspect)
