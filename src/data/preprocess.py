"""Publish valid records only when the label and duplicate audits permit it."""
from __future__ import annotations

from .common import cli, sha256, utc_now, versions, write_csv, write_json, write_report
from .inspect_dataset import inspect


def preprocess(config: dict) -> dict:
    """Retain every valid record; conflicting observed labels block publication."""
    manifest = {"status": "blocked", "processed_at_utc": utc_now(),
                "input_path": config["raw_path"], "dataset_source": config["dataset_url"],
                "canonicalization_method": "RDKit MolFromSmiles (sanitize=True), MolToSmiles(canonical=True, isomericSmiles=True)",
                "duplicate_policy": "retain all valid records, no merging or collapsing",
                "conflict_policy": "block publication if a canonical group has observed 0 and 1 for any endpoint",
                "one_row_per_unique_canonical_structure": False,
                "configuration": config, "software_versions": versions()}
    # Invalidate any prior success before attempting a new run; old CSVs are not current outputs.
    write_json(config["manifest_path"], manifest)
    try:
        report, valid, exclusions, duplicates, _ = inspect(config)
        manifest.update({"input_sha256": sha256(config["raw_path"]), "provenance": report["source"],
                         "input_records": report["rows"], "invalid_smiles_exclusions": len(exclusions),
                         "valid_candidate_records": len(valid), "duplicates": report["duplicates"],
                         "quality_report_sha256": sha256(config["quality_report_path"])})
        if report["duplicates"]["conflicting_records"]:
            raise ValueError("Conflicting labels require review; cleaned dataset was not published. See duplicates.csv")
        if valid.empty:
            raise ValueError("No valid molecular structures; cleaned dataset was not published")
        audit = duplicates[["source_row", "duplicate_group", "duplicate_status"]]
        cleaned = valid.merge(audit, on="source_row", how="left", validate="one_to_one", sort=False)
        cleaned["duplicate_group"] = cleaned["duplicate_group"].fillna("")
        cleaned["duplicate_status"] = cleaned["duplicate_status"].fillna("unique")
        if len(cleaned) + len(exclusions) != report["rows"]:
            raise ValueError("Exclusion accounting does not reconcile")
        write_csv(config["processed_path"], cleaned)
        manifest.update({"status": "success", "output_path": config["processed_path"],
                         "output_records": len(cleaned), "output_sha256": sha256(config["processed_path"]),
                         "properties_sha256": sha256(config["properties_path"]),
                         "exclusions_sha256": sha256(config["exclusion_path"]),
                         "duplicates_sha256": sha256(config["duplicate_path"])})
        write_json(config["manifest_path"], manifest)
        write_report(config, report, manifest)
        print(f"Published {len(cleaned)} valid source records; {len(exclusions)} exclusions; duplicates retained")
        return manifest
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        manifest.update({"status": "blocked", "output_records": None, "output_sha256": None, "error": str(error)})
        write_json(config["manifest_path"], manifest)
        if "report" in locals():
            write_report(config, report, manifest)
        raise


if __name__ == "__main__":
    cli(preprocess)
