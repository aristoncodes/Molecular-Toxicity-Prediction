"""Generate measured EDA figures, underlying statistics, and the research report."""
from __future__ import annotations

import os
from pathlib import Path

from .common import ROOT, cli, read_json, sha256, write_json, write_report

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .download_data import verify_raw


def verify_outputs(config: dict) -> tuple[dict, dict]:
    """Reject stale, partial, blocked, or modified preprocessing artifacts."""
    verify_raw(config)
    manifest = read_json(config["manifest_path"])
    if manifest.get("status") != "success":
        raise ValueError("EDA requires successful preprocessing; manifest is blocked or pending")
    if manifest["configuration"] != config:
        raise ValueError("Configuration differs from successful preprocessing; rerun preprocessing")
    checks = {"input_sha256": "raw_path", "output_sha256": "processed_path",
              "quality_report_sha256": "quality_report_path", "properties_sha256": "properties_path",
              "duplicates_sha256": "duplicate_path", "exclusions_sha256": "exclusion_path"}
    for key, path_key in checks.items():
        if sha256(config[path_key]) != manifest[key]:
            raise ValueError(f"Artifact checksum mismatch: {path_key}; rerun preprocessing")
    return manifest, read_json(config["quality_report_path"])


def generate_eda(config: dict) -> None:
    """Create ten PNG/PDF figures and numerical JSON files from verified outputs."""
    manifest, report = verify_outputs(config)
    folder = Path(config["figures_path"])
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": config["plot"]["dpi"], "pdf.fonttype": 42})

    def save(name: str, figure: object, statistics: dict) -> None:
        figure.tight_layout()
        for extension in ("png", "pdf"):
            figure.savefig(folder / f"{name}.{extension}", bbox_inches="tight")
        plt.close(figure)
        write_json(folder / f"{name}.json", {"input_sha256": manifest["input_sha256"],
                                           "output_sha256": manifest["output_sha256"], **statistics})

    def endpoint_bars(name: str, key: str, title: str, xlabel: str) -> None:
        figure, axis = plt.subplots(figsize=config["plot"]["figsize"])
        endpoints = config["endpoints"]
        values = [report["labels"][e][key] for e in endpoints]
        axis.barh(endpoints, [float("nan") if v is None else v for v in values], color="#35658d")
        axis.invert_yaxis()
        axis.set(title=title, xlabel=xlabel, ylabel="Tox21 endpoint")
        save(name, figure, {"population": "all raw source records", "sample_size": report["rows"],
                            "values": dict(zip(endpoints, values)), "labels": report["labels"]})

    figure, axes = plt.subplots(1, 2, figsize=config["plot"]["figsize"])
    dimensions = {"Rows": report["rows"], "Columns": report["columns_count"]}
    validity = {"Valid": report["structures"]["valid_records"], "Excluded": report["structures"]["invalid_records"]}
    for axis, values, title, ylabel in [(axes[0], dimensions, "Raw dataset dimensions", "Count (rows or columns)"),
                                         (axes[1], validity, "Structure validation", "Source records")]:
        bars = axis.bar(list(values), list(values.values()), color=["#35658d", "#b75d47"])
        axis.bar_label(bars, padding=3)
        axis.set(title=title, ylabel=ylabel)
        axis.margins(y=0.15)
    save("01_dimensions_validity", figure, {"dimensions": dimensions, "validity": validity})
    endpoint_bars("02_missing_labels", "missing", f"Missing endpoint labels (raw n={report['rows']})", "Unlabelled source records")

    figure, axis = plt.subplots(figsize=config["plot"]["figsize"])
    endpoints = config["endpoints"]
    positions = np.arange(len(endpoints))
    negative = [report["labels"][e]["negative"] for e in endpoints]
    positive = [report["labels"][e]["positive"] for e in endpoints]
    axis.barh(positions - 0.2, negative, height=0.4, label="Negative (0)", color="#35658d")
    axis.barh(positions + 0.2, positive, height=0.4, label="Positive (1)", color="#b75d47")
    axis.set_yticks(positions, endpoints)
    axis.invert_yaxis()
    axis.set(title=f"Observed labels by endpoint (raw n={report['rows']})", xlabel="Source records", ylabel="Tox21 endpoint")
    axis.legend()
    save("03_label_counts", figure, {"population": "all raw source records", "labels": report["labels"]})
    endpoint_bars("04_positive_prevalence", "positive_prevalence", "Positive prevalence among observed labels", "Positive / (positive + negative), fraction")

    properties = pd.read_csv(config["properties_path"])
    if len(properties) != manifest["output_records"]:
        raise ValueError("Property sample size differs from cleaned output record count")
    definitions = [("atom_count", "Molecular atom count", "Represented atoms (count)"),
                   ("molecular_weight", "Molecular weight", "Molecular weight (g/mol)"),
                   ("heavy_atom_count", "Molecular heavy-atom count", "Heavy atoms (count)"),
                   ("ring_count", "Molecular ring count", "Rings (count)"),
                   ("rotatable_bond_count", "Molecular rotatable-bond count", "Rotatable bonds (count)")]
    for number, (column, title, xlabel) in enumerate(definitions, start=5):
        values = properties[column].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite molecular properties: {column}")
        bins = np.histogram_bin_edges(values, bins="auto")
        counts, edges = np.histogram(values, bins=bins)
        figure, axis = plt.subplots(figsize=config["plot"]["figsize"])
        axis.stairs(counts, edges, fill=True, color="#35658d")
        axis.set(title=f"{title} (valid n={len(values)}, duplicates retained)", xlabel=xlabel, ylabel="Source records")
        save(f"{number:02d}_{column}", figure, {"sample_size": len(values), "population": "valid source records; duplicates retained",
                                              "source_rows": properties.source_row.tolist(), "values": values.tolist(),
                                              "bin_edges": edges.tolist(), "bin_counts": counts.tolist(),
                                              "summary": report["property_summary"][column]})

    duplicates = report["duplicates"]
    figure, axes = plt.subplots(1, 2, figsize=config["plot"]["figsize"])
    for axis, suffix, ylabel in [(axes[0], "groups", "Canonical groups"), (axes[1], "records", "Source records")]:
        bars = axis.bar(["Duplicates", "Conflicts"], [duplicates[f"canonical_{suffix}"], duplicates[f"conflicting_{suffix}"]],
                        color=["#35658d", "#b75d47"])
        axis.bar_label(bars, padding=3)
        axis.set(ylabel=ylabel, title=f"Duplicate audit: {suffix}")
        axis.margins(y=0.15)
    figure.suptitle("Conflicts are a subset of canonical duplicates")
    save("10_duplicates_conflicts", figure, duplicates)
    write_report(config, report, manifest, eda_complete=True)
    print(f"Created 10 figures (PNG + PDF), numerical JSON files, and {config['report_path']}")


if __name__ == "__main__":
    cli(generate_eda)
