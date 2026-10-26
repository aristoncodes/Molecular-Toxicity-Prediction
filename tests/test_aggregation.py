"""Unit tests for metric aggregation and uncertainty reporting."""

import numpy as np
import pytest

from src.evaluation.aggregation import (
    aggregate_endpoint_metrics,
    compute_uncertainty_stats,
    format_metric_with_uncertainty,
)


def test_compute_uncertainty_stats():
    # 3 values: 0.70, 0.80, 0.90 -> mean = 0.80
    # sample var = ((0.01 + 0 + 0.01) / 2) = 0.01 -> sd = 0.10
    # sem = 0.10 / sqrt(3) ~ 0.057735
    vals = [0.70, 0.80, 0.90]
    stats = compute_uncertainty_stats(vals)

    assert stats["k"] == 3
    assert np.isclose(stats["mean"], 0.80)
    assert np.isclose(stats["sd"], 0.10)
    assert np.isclose(stats["sem"], 0.10 / np.sqrt(3))
    assert stats["ci_95_lower"] < 0.80 < stats["ci_95_upper"]


def test_format_metric_with_uncertainty():
    stats = {
        "k": 12,
        "mean": 0.78214,
        "sd": 0.08123,
        "sem": 0.02351,
        "ci_95_lower": 0.7305,
        "ci_95_upper": 0.8337,
    }

    sem_str = format_metric_with_uncertainty(stats, "sem", precision=4)
    assert "0.7821 ± 0.0235 (SEM, K=12)" in sem_str

    sd_str = format_metric_with_uncertainty(stats, "sd", precision=4)
    assert "0.7821 ± 0.0812 (SD, K=12)" in sd_str

    ci_str = format_metric_with_uncertainty(stats, "ci95", precision=4)
    assert "95% CI" in ci_str


def test_aggregate_endpoint_metrics_with_exclusions():
    endpoint_data = {
        "EP1": {"roc_auc": 0.80, "pr_auc": 0.50, "status": "valid"},
        "EP2": {"roc_auc": 0.90, "pr_auc": 0.60, "status": "valid"},
        "EP3": {"roc_auc": np.nan, "pr_auc": np.nan, "status": "single_class_undefined"},
    }

    res = aggregate_endpoint_metrics(endpoint_data)
    assert res["status"] == "passed"
    assert res["total_endpoints_evaluated"] == 3
    assert res["valid_endpoints_count"] == 2
    assert res["excluded_endpoints"] == ["EP3"]

    assert np.isclose(res["macro_roc_auc"]["mean"], 0.85)
    assert np.isclose(res["macro_pr_auc"]["mean"], 0.55)
    assert "SEM" in res["macro_roc_auc"]["formatted_sem"]
