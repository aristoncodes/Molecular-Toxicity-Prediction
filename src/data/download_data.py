"""Safely acquire a gzip artifact with checksum and provenance gates."""
from __future__ import annotations

import csv
import gzip
import os
from pathlib import Path
import tempfile
import zlib

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .common import cli, read_json, sha256, utc_now, write_json


def validate_gzip(path: str | Path) -> None:
    """Read through the trailer so truncated archives and CRC failures are rejected."""
    try:
        with open(path, "rb") as compressed:
            if compressed.read(2) != b"\x1f\x8b":
                raise ValueError(f"Invalid gzip archive {path}: missing gzip header")
        with gzip.open(path, "rb") as stream:
            while stream.read(1024 * 1024):
                pass
    except (OSError, EOFError, zlib.error) as error:
        raise ValueError(f"Invalid gzip archive {path}: {error}") from error


def check_reference(path: str | Path, config: dict) -> str:
    digest = sha256(path)
    if digest != config["reference_sha256"]:
        raise ValueError(f"Checksum review required: measured {digest}; reference {config['reference_sha256']}")
    return digest


def verify_raw(config: dict) -> dict:
    """Require an intact, version-approved raw file and matching provenance."""
    validate_gzip(config["raw_path"])
    digest = check_reference(config["raw_path"], config)
    provenance = read_json(config["provenance_path"])
    if provenance["sha256"] != digest or provenance["url"] != config["dataset_url"]:
        raise ValueError("Raw artifact differs from recorded provenance")
    if provenance["size_bytes"] != Path(config["raw_path"]).stat().st_size:
        raise ValueError("Raw file size differs from recorded provenance")
    return provenance


def artifact_evidence(path: Path) -> dict:
    """Inspect an unapproved candidate without accepting it for analysis."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            columns = next(reader)
            count = sum(1 for _ in reader)
        return {"columns": columns, "rows": count}
    except (UnicodeError, csv.Error, StopIteration) as error:
        return {"schema_error": str(error)}


def download(config: dict) -> dict:
    """Download Tox21 or validate and reuse an existing artifact."""
    target = Path(config["raw_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        validate_gzip(target)
        check_reference(target, config)
        if not Path(config["provenance_path"]).exists():
            # Existing files have no trustworthy historical download timestamp.
            write_json(config["provenance_path"], {
                "url": config["dataset_url"], "downloaded_at_utc": None,
                "validated_at_utc": utc_now(), "origin": "existing file; historical origin unverified",
                "size_bytes": target.stat().st_size, "sha256": sha256(target)})
        result = verify_raw(config)
        print(f"Validated existing artifact: {target} ({result['sha256']})")
        return result
    descriptor, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".part", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream, requests.Session() as session:
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
            session.mount("https://", HTTPAdapter(max_retries=retries))
            with session.get(config["dataset_url"], stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        validate_gzip(temporary)
        provenance = {"url": config["dataset_url"], "downloaded_at_utc": utc_now(),
                      "size_bytes": temporary.stat().st_size, "sha256": sha256(temporary)}
        if provenance["sha256"] != config["reference_sha256"]:
            candidate = target.with_name(target.name + ".candidate." + provenance["sha256"])
            os.replace(temporary, candidate)
            write_json(str(candidate) + ".json", {**provenance, **artifact_evidence(candidate),
                                                 "status": "checksum_review_required"})
            raise ValueError(f"Checksum review required; candidate retained at {candidate}; measured {provenance['sha256']}")
        os.replace(temporary, target)
        write_json(config["provenance_path"], provenance)
        print(f"Downloaded {target}: {provenance['size_bytes']} bytes; SHA-256 {provenance['sha256']}")
        return provenance
    except requests.RequestException as error:
        raise RuntimeError(f"Dataset download failed; no partial artifact accepted: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    cli(download)
