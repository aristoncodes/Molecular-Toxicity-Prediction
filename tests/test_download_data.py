import gzip
from pathlib import Path

import pytest
import requests

from src.data.common import read_json
from src.data.download_data import download, validate_gzip, verify_raw


class FakeResponse:
    def __init__(self, payload, fail=False):
        self.payload, self.fail = payload, fail

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.payload[:10]
        if self.fail:
            raise requests.ConnectionError("interrupted")
        yield self.payload[10:]


def test_gzip_integrity(tmp_path):
    path = tmp_path / "test.gz"
    payload = gzip.compress(b"example" * 100)
    path.write_bytes(payload)
    validate_gzip(path)
    for invalid in [b"", payload[:2], payload[:-5], b"not gzip", payload[:-8] + b"\x00" * 8]:
        path.write_bytes(invalid)
        with pytest.raises(ValueError, match="Invalid gzip"):
            validate_gzip(path)


def test_existing_reused_and_tampering_blocked(save_raw, rows, monkeypatch):
    config = save_raw([rows()])
    monkeypatch.setattr(requests.Session, "get", lambda *a, **k: pytest.fail("Unexpected network"))
    first = download(config)
    assert download(config) == first
    Path(config["raw_path"]).write_bytes(gzip.compress(b"changed"))
    with pytest.raises(ValueError, match="Checksum"):
        download(config)


def test_download_success(save_raw, rows, monkeypatch):
    config = save_raw([rows()])
    payload = Path(config["raw_path"]).read_bytes()
    Path(config["raw_path"]).unlink()
    Path(config["provenance_path"]).unlink()
    monkeypatch.setattr(requests.Session, "get", lambda *a, **k: FakeResponse(payload))
    result = download(config)
    assert result == verify_raw(config)
    assert result["downloaded_at_utc"].endswith("+00:00")
    assert not list(Path(config["raw_path"]).parent.glob("*.part"))


def test_interrupted_download_never_published(config, monkeypatch):
    monkeypatch.setattr(requests.Session, "get", lambda *a, **k: FakeResponse(b"partial", fail=True))
    with pytest.raises(RuntimeError, match="interrupted"):
        download(config)
    assert not Path(config["raw_path"]).exists()
    assert not list(Path(config["raw_path"]).parent.glob("*.part"))
    assert not Path(config["provenance_path"]).exists()


def test_mismatched_candidate_quarantined(config, rows, monkeypatch):
    import pandas as pd
    payload = gzip.compress(pd.DataFrame([rows()]).to_csv(index=False).encode())
    config["reference_sha256"] = "0" * 64
    monkeypatch.setattr(requests.Session, "get", lambda *a, **k: FakeResponse(payload))
    with pytest.raises(ValueError, match="review required"):
        download(config)
    assert not Path(config["raw_path"]).exists()
    evidence = list(Path(config["raw_path"]).parent.glob("*.candidate.*.json"))
    assert len(evidence) == 1
    assert read_json(evidence[0])["rows"] == 1


def test_invalid_download_not_published(config, monkeypatch):
    monkeypatch.setattr(requests.Session, "get", lambda *a, **k: FakeResponse(b"not gzip"))
    with pytest.raises(ValueError, match="Invalid gzip"):
        download(config)
    assert not Path(config["raw_path"]).exists()


def test_provenance_tampering(save_raw, rows):
    config = save_raw([rows()])
    from src.data.common import write_json
    provenance = read_json(config["provenance_path"])
    provenance["url"] = "https://different.example/data"
    write_json(config["provenance_path"], provenance)
    with pytest.raises(ValueError, match="provenance"):
        verify_raw(config)
