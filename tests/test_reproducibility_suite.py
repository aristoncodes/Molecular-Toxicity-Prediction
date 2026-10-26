"""Unit tests for the automated scientific audit and reproducibility suite."""

from pathlib import Path
import json
import pytest

from src.audit.run_audit import (
    DEFAULT_OUTPUT_AUDIT_LOG,
    audit_experiment_reproducibility,
    audit_feature_artifacts,
    run_full_suite,
)


def test_audit_feature_artifacts(tmp_path: Path):
    # Test behavior when artifacts are missing
    res = audit_feature_artifacts(features_dir=tmp_path / "empty_dir")
    assert res["status"] == "missing_artifacts"


def test_audit_experiment_reproducibility():
    res = audit_experiment_reproducibility()
    assert res["phase1_data_and_features_verified"] is True
    assert "Phase 1" in res["audit_finding"]
    assert res["model_training_status"] == "not_executed_in_phase1"


@pytest.mark.skipif(
    not Path("data/raw/tox21.csv.gz").exists() or not Path("data/processed/features/combined_features.npy").exists(),
    reason="Full real data and feature artifacts not available",
)
def test_full_audit_suite_execution(tmp_path: Path):
    out_json = tmp_path / "test_audit_out.json"
    report = run_full_suite(output_path=out_json)

    assert report["overall_status"] == "passed"
    assert report["summary"]["checks_passed"] == 7
    assert report["summary"]["checks_failed"] == 0
    assert out_json.exists()

    with open(out_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["overall_status"] == "passed"
