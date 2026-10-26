"""Evaluation Metrics for Molecular Toxicity Prediction.

Student 3 Module: Independent implementation of ROC-AUC, PR-AUC, and missing-label handling.

Key Scientific Guarantees:
--------------------------
1. Continuous Probability Scores: ROC-AUC and PR-AUC require continuous prediction scores
   (calibrated probabilities or decision values in [0, 1]), not discrete {0, 1} predictions.
2. Missing-Label Masking: In Tox21, assay values contain NaNs (untested compounds).
   Evaluations must mask NaNs per endpoint rather than imputing or treating as negative.
3. Single-Class Endpoint Handling: If an evaluation partition (or cross-validation fold) contains
   only negative or only positive instances for an endpoint, ROC-AUC and PR-AUC are mathematically
   undefined (division by 0). This module detects single-class slices, returns np.nan, and marks
   status='single_class_undefined' so the endpoint is gracefully excluded from macro averaging.
4. PR-AUC Definition: Clarifies the difference between Average Precision (standard in ML,
   weighted rectangular sum of precision values) and trapezoidal PR-curve area.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_roc_auc(
    y_true: Sequence[float | int | None] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
) -> float:
    """Compute Receiver Operating Characteristic Area Under Curve (ROC-AUC).

    Uses the Mann-Whitney U / Wilcoxon rank-sum statistic with continuous scores:
        AUC = (sum(ranks_pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    Args:
        y_true: Binary ground-truth labels (0, 1, or NaN).
        y_score: Continuous predicted scores / probabilities in [0, 1].

    Returns:
        ROC-AUC float in [0.0, 1.0], or np.nan if single class or no valid labels.
    """
    yt = np.asarray(y_true, dtype=np.float64).flatten()
    ys = np.asarray(y_score, dtype=np.float64).flatten()

    if len(yt) != len(ys):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)}, len(y_score)={len(ys)}")

    # Mask out missing labels (NaNs)
    valid_mask = ~np.isnan(yt)
    yt_valid = yt[valid_mask]
    ys_valid = ys[valid_mask]

    if len(yt_valid) == 0:
        return float(np.nan)

    # Verify binary ground truth
    unique_labels = set(np.unique(yt_valid))
    if not unique_labels.issubset({0.0, 1.0, 0, 1}):
        raise ValueError(f"Ground truth labels must be binary {0, 1}, got: {unique_labels}")

    n_pos = int(np.sum(yt_valid == 1.0))
    n_neg = int(np.sum(yt_valid == 0.0))

    # Single-class edge case: ROC-AUC is undefined when all instances belong to one class
    if n_pos == 0 or n_neg == 0:
        logger.warning(
            "ROC-AUC undefined: single-class partition detected (pos=%d, neg=%d)", n_pos, n_neg
        )
        return float(np.nan)

    # Mann-Whitney U rank-sum calculation with average tie ranking
    # Sort order: ascending by y_score
    order = np.argsort(ys_valid)
    ranked_ys = ys_valid[order]
    ranked_yt = yt_valid[order]

    # Compute ranks handling ties (1-based ranks)
    ranks = np.empty(len(ys_valid), dtype=np.float64)
    i = 0
    while i < len(ranked_ys):
        j = i
        while j < len(ranked_ys) and ranked_ys[j] == ranked_ys[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[i:j] = avg_rank
        i = j

    sum_ranks_pos = np.sum(ranks[ranked_yt == 1.0])
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1.0)) / 2.0
    auc = u_stat / (n_pos * n_neg)

    return float(np.clip(auc, 0.0, 1.0))


def compute_pr_auc(
    y_true: Sequence[float | int | None] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    method: str = "average_precision",
) -> float:
    """Compute Precision-Recall Area Under Curve (PR-AUC).

    Args:
        y_true: Binary ground-truth labels (0, 1, or NaN).
        y_score: Continuous predicted scores / probabilities in [0, 1].
        method: 'average_precision' (default; standard integral of precision w.r.t recall)
                or 'trapezoid' (piecewise linear trapezoidal integration).

    Returns:
        PR-AUC float in [0.0, 1.0], or np.nan if single class or no valid labels.
    """
    yt = np.asarray(y_true, dtype=np.float64).flatten()
    ys = np.asarray(y_score, dtype=np.float64).flatten()

    if len(yt) != len(ys):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)}, len(y_score)={len(ys)}")

    valid_mask = ~np.isnan(yt)
    yt_valid = yt[valid_mask]
    ys_valid = ys[valid_mask]

    if len(yt_valid) == 0:
        return float(np.nan)

    n_pos = int(np.sum(yt_valid == 1.0))
    n_neg = int(np.sum(yt_valid == 0.0))

    if n_pos == 0 or n_neg == 0:
        logger.warning(
            "PR-AUC undefined: single-class partition detected (pos=%d, neg=%d)", n_pos, n_neg
        )
        return float(np.nan)

    # Sort descending by predicted score
    desc_order = np.argsort(ys_valid)[::-1]
    y_sorted = yt_valid[desc_order]
    scores_sorted = ys_valid[desc_order]

    # Identify distinct score thresholds
    distinct_mask = np.diff(scores_sorted) != 0
    distinct_indices = np.where(distinct_mask)[0]
    threshold_idxs = np.concatenate([distinct_indices, [len(y_sorted) - 1]])

    true_positives = np.cumsum(y_sorted == 1.0)[threshold_idxs]
    false_positives = np.cumsum(y_sorted == 0.0)[threshold_idxs]

    recall = true_positives / n_pos
    precision = true_positives / (true_positives + false_positives)

    if method == "average_precision":
        # Average Precision: sum of P_k * (R_k - R_{k-1})
        recall_diffs = np.diff(np.concatenate([[0.0], recall]))
        ap = np.sum(precision * recall_diffs)
        return float(np.clip(ap, 0.0, 1.0))
    elif method == "trapezoid":
        # Piecewise trapezoidal rule on PR curve
        # Prepend operating point (R=0, P=P_first)
        r_curve = np.concatenate([[0.0], recall])
        p_curve = np.concatenate([[precision[0]], precision])
        auc = np.sum(0.5 * (p_curve[:-1] + p_curve[1:]) * np.diff(r_curve))
        return float(np.clip(auc, 0.0, 1.0))
    else:
        raise ValueError(f"Unknown PR-AUC method: '{method}' (choose 'average_precision' or 'trapezoid')")


def evaluate_endpoint(
    y_true: Sequence[float | int | None] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    endpoint_name: str = "",
) -> dict[str, Any]:
    """Evaluate predictions for a single toxicity endpoint."""
    yt = np.asarray(y_true, dtype=np.float64).flatten()
    ys = np.asarray(y_score, dtype=np.float64).flatten()

    total_samples = len(yt)
    valid_mask = ~np.isnan(yt)
    observed = int(np.sum(valid_mask))
    missing = total_samples - observed

    if observed == 0:
        return {
            "endpoint": endpoint_name,
            "status": "no_observed_samples",
            "total_samples": total_samples,
            "observed": 0,
            "missing": missing,
            "positive_count": 0,
            "negative_count": 0,
            "prevalence": float(np.nan),
            "roc_auc": float(np.nan),
            "pr_auc": float(np.nan),
        }

    n_pos = int(np.sum(yt[valid_mask] == 1.0))
    n_neg = int(np.sum(yt[valid_mask] == 0.0))
    prevalence = n_pos / observed if observed > 0 else 0.0

    if n_pos == 0 or n_neg == 0:
        return {
            "endpoint": endpoint_name,
            "status": "single_class_undefined",
            "total_samples": total_samples,
            "observed": observed,
            "missing": missing,
            "positive_count": n_pos,
            "negative_count": n_neg,
            "prevalence": prevalence,
            "roc_auc": float(np.nan),
            "pr_auc": float(np.nan),
        }

    roc_auc = compute_roc_auc(yt, ys)
    pr_auc = compute_pr_auc(yt, ys, method="average_precision")

    return {
        "endpoint": endpoint_name,
        "status": "valid",
        "total_samples": total_samples,
        "observed": observed,
        "missing": missing,
        "positive_count": n_pos,
        "negative_count": n_neg,
        "prevalence": prevalence,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def evaluate_multitask_predictions(
    y_true_matrix: np.ndarray | pd.DataFrame,
    y_score_matrix: np.ndarray,
    endpoint_names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Evaluate predictions across multiple endpoints."""
    yt_mat = np.asarray(y_true_matrix)
    ys_mat = np.asarray(y_score_matrix)

    n_samples, n_tasks = yt_mat.shape
    if ys_mat.shape != (n_samples, n_tasks):
        raise ValueError(
            f"Shape mismatch: y_true shape {yt_mat.shape} != y_score shape {ys_mat.shape}"
        )
    if len(endpoint_names) != n_tasks:
        raise ValueError(
            f"Endpoint names count ({len(endpoint_names)}) != task count ({n_tasks})"
        )

    results = {}
    for task_idx, name in enumerate(endpoint_names):
        res = evaluate_endpoint(
            yt_mat[:, task_idx],
            ys_mat[:, task_idx],
            endpoint_name=name,
        )
        results[name] = res

    return results
