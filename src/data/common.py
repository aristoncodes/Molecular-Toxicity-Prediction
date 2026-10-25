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
