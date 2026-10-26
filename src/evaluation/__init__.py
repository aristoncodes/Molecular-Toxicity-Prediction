"""Evaluation module for molecular toxicity prediction.

Student 3 Module: Independent implementation and audit of evaluation metrics,
uncertainty quantification, and cross-endpoint aggregation.
"""

from typing import Any

__all__ = [
    "compute_roc_auc",
    "compute_pr_auc",
    "evaluate_endpoint",
    "evaluate_multitask_predictions",
    "aggregate_endpoint_metrics",
    "compute_uncertainty_stats",
    "format_metric_with_uncertainty",
]


def __getattr__(name: str) -> Any:
    if name in (
        "compute_roc_auc",
        "compute_pr_auc",
        "evaluate_endpoint",
        "evaluate_multitask_predictions",
    ):
        from src.evaluation import metrics

        return getattr(metrics, name)
    if name in (
        "aggregate_endpoint_metrics",
        "compute_uncertainty_stats",
        "format_metric_with_uncertainty",
    ):
        from src.evaluation import aggregation

        return getattr(aggregation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
