from pathlib import Path

import pandas as pd
import pytest

from src.data.common import read_json
from src.data.inspect_dataset import analyze
from src.data.preprocess import preprocess


def test_duplicate_categories(config, rows):
    records = [rows("CCO"), rows("CCO"), rows("CCN"), rows("NCC"),
               rows("CCC", label=None), rows("CCC", label=1), rows("CC", label=0), rows("CC", label=1)]
    report, _, _, duplicates, _ = analyze(pd.DataFrame(records), config)
    assert set(duplicates.duplicate_status) == {"identical_record", "identical_labels", "compatible_missingness", "conflicting_labels"}
    assert report["duplicates"]["canonical_groups"] == 4
    assert report["duplicates"]["conflicting_records"] == 2
    assert report["duplicates"]["conflicting_groups"] == 1
    assert set(duplicates.loc[duplicates.duplicate_status.eq("conflicting_labels"), "conflicting_endpoints"]) == {";".join(config["endpoints"])}


def test_idempotency_preservation_and_counts(config, save_raw, rows):
    save_raw([rows("CCO", label=None), rows("OCC", label=1), rows("invalid!")])
    first = preprocess(config)
    output = Path(config["processed_path"]).read_bytes()
    second = preprocess(config)
    assert output == Path(config["processed_path"]).read_bytes()
    assert first["output_sha256"] == second["output_sha256"]
    cleaned = pd.read_csv(config["processed_path"])
    assert len(cleaned) == 2 and cleaned["NR-AR"].isna().sum() == 1
    assert cleaned.smiles.tolist() == ["CCO", "OCC"]
    assert cleaned.source_row.tolist() == [1, 2]
    assert first["input_records"] == first["output_records"] + first["invalid_smiles_exclusions"]
    assert first["one_row_per_unique_canonical_structure"] is False


def test_conflicts_block_publication(config, save_raw, rows):
    save_raw([rows("CCO", 0), rows("OCC", 1)])
    with pytest.raises(ValueError, match="Conflicting"):
        preprocess(config)
    assert not Path(config["processed_path"]).exists()
    manifest = read_json(config["manifest_path"])
    assert manifest["status"] == "blocked" and manifest["output_records"] is None
    assert len(pd.read_csv(config["duplicate_path"])) == 2
    assert "conflicts_require_review" in Path(config["quality_report_path"]).read_text()
    assert "blocked" in Path(config["report_path"]).read_text()


def test_old_success_invalidated(config, save_raw, rows):
    save_raw([rows()])
    preprocess(config)
    old_csv = Path(config["processed_path"]).read_bytes()
    save_raw([rows(label=0), rows(label=1)])
    with pytest.raises(ValueError, match="Conflicting"):
        preprocess(config)
    assert read_json(config["manifest_path"])["status"] == "blocked"
    assert Path(config["processed_path"]).read_bytes() == old_csv


def test_no_valid_structures_blocks(config, save_raw, rows):
    save_raw([rows("invalid!")])
    with pytest.raises(ValueError, match="No valid"):
        preprocess(config)
    assert not Path(config["processed_path"]).exists()
