import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.common import load_config, write_json
from src.data.inspect_dataset import analyze, inspect, parse_structure, read_dataset, validate_schema


def test_schema(config, rows):
    columns = list(rows())
    assert validate_schema(columns, config) == "smiles"
    with pytest.raises(ValueError, match="Missing endpoints"):
        validate_schema([c for c in columns if c != "NR-AR"], config)
    with pytest.raises(ValueError, match="unique"):
        validate_schema(columns + ["smiles"], config)
    assert validate_schema(["SMILES" if c == "smiles" else c for c in columns], config) == "SMILES"


def test_missing_labels_and_prevalence(config, rows):
    frame = pd.DataFrame([rows(label=1), rows(label=0), rows(label=None)])
    frame["NR-ER"] = None
    report, valid, *_ = analyze(frame, config)
    assert report["labels"]["NR-AR"] == {"positive": 1, "negative": 1, "missing": 1, "invalid": 0,
                                          "observed": 2, "positive_prevalence": 0.5}
    assert report["labels"]["NR-ER"]["positive_prevalence"] is None
    assert pd.isna(valid.iloc[2]["NR-AR"])


@pytest.mark.parametrize("value,reason", [(None, "missing"), (123, "non_string"), (" ", "empty"),
                                          ("invalid!", "unparsable"), ("[CH5]", "unparsable"), ("", "empty")])
def test_invalid_structures(value, reason):
    molecule, canonical, error = parse_structure(value)
    assert molecule is None and canonical is None and reason in error


def test_structures_preserve_policy():
    assert parse_structure("CCO")[1] == parse_structure("OCC")[1]
    assert parse_structure("[Na+].CC(=O)[O-]")[1].count(".") == 1
    assert parse_structure("F[C@H](Cl)Br")[1] != parse_structure("F[C@@H](Cl)Br")[1]
    assert parse_structure("CCO")[1] == parse_structure("CCO")[1]


def test_invalid_labels_reported_and_rejected(save_raw, config, rows):
    records = [rows(**{"NR-AR": "NA"}), rows(**{"NR-AR": "2"})]
    report, *_ = analyze(pd.DataFrame(records), config)
    assert report["labels"]["NR-AR"]["invalid"] == 2
    assert report["labels"]["NR-AR"]["missing"] == 0
    save_raw(records)
    with pytest.raises(ValueError, match="encodings"):
        inspect(config)
    assert len(json.loads(Path(config["quality_report_path"]).read_text())["invalid_labels"]) == 2


def test_csv_tokens_and_ragged_rows(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("smiles,extra\nNA,\nCCO,NA\n")
    frame = read_dataset(path)
    assert frame.iloc[0].smiles == "NA"
    assert pd.isna(frame.iloc[0].extra)
    assert frame.iloc[1].extra == "NA"
    path.write_text("a,b\n1,2,3\n")
    with pytest.raises(ValueError, match="fields"):
        read_dataset(path)


def test_strict_json(tmp_path):
    path = tmp_path / "report.json"
    write_json(path, {"nan": float("nan"), "inf": float("inf"), "missing": pd.NA})
    assert json.loads(path.read_text(), parse_constant=lambda token: pytest.fail(token)) == dict.fromkeys(["nan", "inf", "missing"])


def test_structure_and_exclusion_counts(save_raw, config, rows):
    save_raw([rows("CCO", extra="retained"), rows("invalid!"), rows(None)])
    report, valid, excluded, _, properties = inspect(config)
    assert report["rows"] == 3 and len(valid) == 1 and len(excluded) == 2
    assert report["accounting"]["reconciled"]
    assert valid.iloc[0]["extra"] == "retained"
    assert properties.iloc[0].atom_count == 3
    assert 46 < properties.iloc[0].molecular_weight < 47


def test_real_dataset_audit(request):
    if not request.config.getoption("--run-real-data"):
        pytest.skip("Use --run-real-data to audit the existing source artifact")
    config = load_config()
    if not Path(config["raw_path"]).exists():
        pytest.skip("Real raw artifact is unavailable; no network request made")
    from src.data.download_data import verify_raw
    verify_raw(config)
    report, valid, excluded, *_ = analyze(read_dataset(config["raw_path"]), config)
    assert report["rows"] == len(valid) + len(excluded)
    assert not report["invalid_labels"]
    assert set(config["endpoints"]).issubset(report["columns"])
