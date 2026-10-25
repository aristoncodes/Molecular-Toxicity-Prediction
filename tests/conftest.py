"""Offline, synthetic Tox21 fixtures; no real dataset assumptions."""
from pathlib import Path
import gzip
import hashlib

import pandas as pd
import pytest

from src.data.common import PATH_KEYS, load_config, utc_now, write_json


def pytest_addoption(parser):
    parser.addoption("--run-real-data", action="store_true", help="Audit an existing real raw artifact (no download)")


@pytest.fixture
def config(tmp_path):
    result = load_config()
    for key in PATH_KEYS:
        result[key] = str(tmp_path / Path(result[key]).name)
    result["plot"] = {"dpi": 72, "figsize": [9, 6]}
    return result


@pytest.fixture
def rows(config):
    def create(smiles="CCO", label=0, **metadata):
        return {**dict.fromkeys(config["endpoints"], label), "mol_id": "example", "smiles": smiles, **metadata}
    return create


@pytest.fixture
def save_raw(config):
    def save(records):
        payload = gzip.compress(pd.DataFrame(records).to_csv(index=False).encode(), mtime=0)
        Path(config["raw_path"]).write_bytes(payload)
        config["reference_sha256"] = hashlib.sha256(payload).hexdigest()
        write_json(config["provenance_path"], {"url": config["dataset_url"], "downloaded_at_utc": utc_now(),
                                              "size_bytes": len(payload), "sha256": config["reference_sha256"]})
        return config
    return save
