import json
from pathlib import Path

import pytest

from src.data.generate_eda_plots import generate_eda, verify_outputs
from src.data.preprocess import preprocess


def test_all_figures_and_statistics(config, save_raw, rows):
    save_raw([rows("CCO", 1), rows("CC", 0), rows("C1CCCCC1", None), rows("invalid!")])
    preprocess(config)
    generate_eda(config)
    folder = Path(config["figures_path"])
    for extension in ("png", "pdf", "json"):
        assert len(list(folder.glob(f"*.{extension}"))) == 10
        assert all(p.stat().st_size > 0 for p in folder.glob(f"*.{extension}"))
    stats = json.loads((folder / "05_atom_count.json").read_text())
    assert stats["sample_size"] == 3
    assert sum(stats["bin_counts"]) == 3
    assert stats["values"] == [3, 2, 6]
    assert "generated" in Path(config["report_path"]).read_text()


def test_eda_rejects_modified_artifacts(config, save_raw, rows):
    save_raw([rows()])
    preprocess(config)
    with open(config["processed_path"], "a") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="checksum"):
        generate_eda(config)


def test_eda_rejects_blocked_manifest(config, save_raw, rows):
    save_raw([rows(label=0), rows(label=1)])
    with pytest.raises(ValueError):
        preprocess(config)
    with pytest.raises(ValueError, match="successful preprocessing"):
        generate_eda(config)
    assert not Path(config["figures_path"]).exists()


def test_configuration_change_requires_reprocessing(config, save_raw, rows):
    save_raw([rows()])
    preprocess(config)
    config["seed"] += 1
    with pytest.raises(ValueError, match="Configuration"):
        verify_outputs(config)
