"""Macro Metric Aggregation and Uncertainty Reporting Module.

Student 3 Module: Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks.

Disambiguates Metric Aggregation and Uncertainty Notation (±):
--------------------------------------------------------------
In the literature and machine learning reports, "Mean ± Uncertainty" is frequently ambiguous.
This module strictly distinguishes and computes:
1. Macro Mean: Unweighted average across valid (non-single-class) endpoints:
       Mean = (1 / K) * sum_{k=1}^K Metric_k
2. Standard Deviation across endpoints (SD):
       SD = sqrt( (1 / (K - 1)) * sum_{k=1}^K (Metric_k - Mean)^2 )
   Measures cross-endpoint variance / heterogeneity of task difficulty.
3. Standard Error of the Mean (SEM):
       SEM = SD / sqrt(K)
   Measures precision of the macro-mean estimate.
4. 95% Confidence Interval (CI_95):
       CI_95 = [Mean - t_{0.975, df} * SEM, Mean + t_{0.975, df} * SEM]
5. Single-class and unobserved endpoint handling: Endpoints with undefined AUCs (NaN)
   are explicitly filtered out with warnings, and K_valid is tracked.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Two-tailed Student-t critical values at alpha=0.05 for df = 1..30
STUDENT_T_CRITICAL_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,  # 12 endpoints - 1 = 11 df
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}


def compute_uncertainty_stats(
    values: Sequence[float],
) -> dict[str, float | int]:
    """Compute mean, SD, SEM, and 95% confidence interval for a sequence of metric values.

    Args:
        values: Sequence of numeric metric values (NaNs filtered out).

    Returns:
        Dictionary with count, mean, sd, sem, ci_95_lower, ci_95_upper, and ci_95_half_width.
    """
    clean_vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    k = len(clean_vals)

    if k == 0:
        return {
            "k": 0,
            "mean": float(np.nan),
            "sd": float(np.nan),
            "sem": float(np.nan),
            "ci_95_lower": float(np.nan),
            "ci_95_upper": float(np.nan),
            "ci_95_half_width": float(np.nan),
        }

    mean_val = float(np.mean(clean_vals))

    if k == 1:
        return {
            "k": 1,
            "mean": mean_val,
            "sd": 0.0,
            "sem": 0.0,
            "ci_95_lower": mean_val,
            "ci_95_upper": mean_val,
            "ci_95_half_width": 0.0,
        }

    # Sample standard deviation (ddof=1)
    sd_val = float(np.std(clean_vals, ddof=1))
    sem_val = float(sd_val / math.sqrt(k))

    df = k - 1
    t_crit = STUDENT_T_CRITICAL_95.get(df, 1.960 if df > 30 else 2.0)
    half_width = float(t_crit * sem_val)

    return {
        "k": k,
        "mean": mean_val,
        "sd": sd_val,
        "sem": sem_val,
        "ci_95_lower": float(mean_val - half_width),
        "ci_95_upper": float(mean_val + half_width),
        "ci_95_half_width": half_width,
    }


def format_metric_with_uncertainty(
    stats: dict[str, Any],
    uncertainty_type: str = "sem",
    precision: int = 4,
) -> str:
    """Format metric string with explicit uncertainty label to prevent ambiguity.

    Args:
        stats: Output dictionary from compute_uncertainty_stats.
        uncertainty_type: 'sem', 'sd', or 'ci95'.
        precision: Decimal places for reporting.

    Returns:
        String such as '0.7821 ± 0.0245 (SEM)' or '0.7821 ± 0.0812 (SD)'.
    """
    mean = stats["mean"]
    if np.isnan(mean):
        return "N/A"

    if uncertainty_type == "sem":
        return f"{mean:.{precision}f} ± {stats['sem']:.{precision}f} (SEM, K={stats['k']})"
    elif uncertainty_type == "sd":
        return f"{mean:.{precision}f} ± {stats['sd']:.{precision}f} (SD, K={stats['k']})"
    elif uncertainty_type == "ci95":
        return (
            f"{mean:.{precision}f} (95% CI [{stats['ci_95_lower']:.{precision}f}, "
            f"{stats['ci_95_upper']:.{precision}f}], K={stats['k']})"
        )
    else:
        raise ValueError(f"Unknown uncertainty type: '{uncertainty_type}'")


def aggregate_endpoint_metrics(
    endpoint_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-endpoint results into macro-averaged metrics with full uncertainty.

    Args:
        endpoint_results: Dict mapping endpoint name to dict containing 'roc_auc' and 'pr_auc'.

    Returns:
        Comprehensive dictionary with per-endpoint breakdown, macro averages, SD, SEM, and 95% CIs.
    """
    roc_aucs: dict[str, float] = {}
    pr_aucs: dict[str, float] = {}
    excluded_endpoints: list[str] = []

    for name, res in endpoint_results.items():
        roc = res.get("roc_auc", np.nan)
        pr = res.get("pr_auc", np.nan)

        if np.isnan(roc) or np.isnan(pr) or res.get("status") == "single_class_undefined":
            excluded_endpoints.append(name)
            logger.warning(
                "Endpoint '%s' excluded from macro average due to undefined metric (status=%s)",
                name,
                res.get("status"),
            )
        else:
            roc_aucs[name] = float(roc)
            pr_aucs[name] = float(pr)

    roc_stats = compute_uncertainty_stats(list(roc_aucs.values()))
    pr_stats = compute_uncertainty_stats(list(pr_aucs.values()))

    return {
        "status": "passed",
        "total_endpoints_evaluated": len(endpoint_results),
        "valid_endpoints_count": len(roc_aucs),
        "excluded_endpoints": excluded_endpoints,
        "per_endpoint": {
            "roc_auc": roc_aucs,
            "pr_auc": pr_aucs,
        },
        "macro_roc_auc": {
            **roc_stats,
            "formatted_sem": format_metric_with_uncertainty(roc_stats, "sem"),
            "formatted_sd": format_metric_with_uncertainty(roc_stats, "sd"),
            "formatted_ci95": format_metric_with_uncertainty(roc_stats, "ci95"),
        },
        "macro_pr_auc": {
            **pr_stats,
            "formatted_sem": format_metric_with_uncertainty(pr_stats, "sem"),
            "formatted_sd": format_metric_with_uncertainty(pr_stats, "sd"),
            "formatted_ci95": format_metric_with_uncertainty(pr_stats, "ci95"),
        },
    }
