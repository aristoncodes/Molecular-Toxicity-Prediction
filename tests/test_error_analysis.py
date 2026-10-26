"""Unit tests for endpoint-level error analysis and figure generation.

Student 4 Module: Validates confusion matrices, FDR/FNR rates,
prevalence correlations, and figure generation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.error_analysis import (
    TOX21_ENDPOINTS,
    analyze_endpoint_errors,
    generate_error_analysis_figures,
    identify_challenging_molecules,
)


@pytest.fixture
def mock_benchmark_data() -> dict:
    """Mock benchmark JSON with known endpoints and confusion matrices."""
    endpoints_mock = {}
    rng = np.random.RandomState(42)

    for i, ep in enumerate(TOX21_ENDPOINTS):
        endpoints_mock[ep] = {
            "val_threshold": 0.45,
            "test": {
                "roc_auc": 0.65 + 0.02 * i,
                "pr_auc": 0.20 + 0.03 * i,
                "f1": 0.35 + 0.01 * i,
                "precision": 0.40,
                "recall": 0.60,
                "balanced_accuracy": 0.68,
                "confusion_matrix": {
                    "tp": 20 + i,
                    "fp": 30 - i,
                    "tn": 600,
                    "fn": 15,
                },
            },
        }

    return {
        "random_forest_combined": {
            "macro_metrics": {
                "roc_auc": {"mean": 0.7488, "sem": 0.0181},
                "pr_auc": {"mean": 0.3891, "sem": 0.0320},
            },
            "endpoints": endpoints_mock,
        },
        "logistic_regression_combined": {
            "macro_metrics": {
                "roc_auc": {"mean": 0.7138, "sem": 0.0158},
            },
        },
        "xgboost_combined": {
            "macro_metrics": {
                "roc_auc": {"mean": 0.7456, "sem": 0.0179},
            },
        },
        "random_forest_morgan": {
            "macro_metrics": {"roc_auc": {"mean": 0.7208, "sem": 0.0199}},
        },
        "logistic_regression_morgan": {
            "macro_metrics": {"roc_auc": {"mean": 0.6987, "sem": 0.0149}},
        },
        "xgboost_morgan": {
            "macro_metrics": {"roc_auc": {"mean": 0.7114, "sem": 0.0181}},
        },
        "random_forest_descriptors": {
            "macro_metrics": {"roc_auc": {"mean": 0.7215, "sem": 0.0216}},
        },
        "logistic_regression_descriptors": {
            "macro_metrics": {"roc_auc": {"mean": 0.7203, "sem": 0.0169}},
        },
        "xgboost_descriptors": {
            "macro_metrics": {"roc_auc": {"mean": 0.7024, "sem": 0.0256}},
        },
    }


def test_analyze_endpoint_errors_structure(mock_benchmark_data):
    """Verify endpoint error analysis produces expected keys and valid metrics."""
    res = analyze_endpoint_errors(mock_benchmark_data, model_key="random_forest_combined")

    assert res["model_analyzed"] == "random_forest_combined"
    assert "endpoint_error_details" in res
    assert len(res["endpoint_error_details"]) == 12
    assert "difficulty_ranking" in res
    assert len(res["difficulty_ranking"]["hardest_by_roc_auc"]) == 3
    assert len(res["difficulty_ranking"]["easiest_by_roc_auc"]) == 3
    assert "prevalence_correlations" in res

    # Check error rates for an endpoint
    ar_stats = res["endpoint_error_details"]["NR-AR"]
    assert "false_discovery_rate" in ar_stats
    assert 0.0 <= ar_stats["false_discovery_rate"] <= 1.0
    assert "false_negative_rate" in ar_stats
    assert 0.0 <= ar_stats["false_negative_rate"] <= 1.0


def test_identify_challenging_molecules():
    """Verify identification of multi-task false negative compounds."""
    cleaned_df = pd.DataFrame({
        "source_row": [1, 2, 3],
        "mol_id": ["m1", "m2", "m3"],
        "canonical_smiles": ["CC", "CCO", "c1ccccc1"],
        "NR-AR": [1.0, 1.0, 0.0],
        "NR-AhR": [1.0, 0.0, 0.0],
    })
    # Molecule 0 is positive for both endpoints, but predicted probability is 0.1 (false negative on both)
    predictions_df = pd.DataFrame({
        "NR-AR": [0.1, 0.9, 0.1],
        "NR-AhR": [0.1, 0.1, 0.1],
    })

    benchmark_data = {
        "rf_test": {
            "endpoints": {
                "NR-AR": {"val_threshold": 0.5},
                "NR-AhR": {"val_threshold": 0.5},
            }
        }
    }

    res = identify_challenging_molecules(
        cleaned_df,
        predictions_df,
        test_indices=[0, 1, 2],
        benchmark_data=benchmark_data,
        model_key="rf_test",
    )

    assert res["total_test_molecules"] == 3
    fns = res["promiscuous_false_negatives"]
    assert len(fns) >= 1
    assert fns[0]["mol_id"] == "m1"
    assert fns[0]["fn_count"] == 2


def test_generate_error_analysis_figures(tmp_path, mock_benchmark_data):
    """Verify figures are generated and saved without errors."""
    res = analyze_endpoint_errors(mock_benchmark_data, model_key="random_forest_combined")
    figs = generate_error_analysis_figures(res, mock_benchmark_data, output_dir=str(tmp_path))

    assert len(figs) == 3
    for f in figs:
        assert Path(f).exists()
        assert Path(f).stat().st_size > 1000
