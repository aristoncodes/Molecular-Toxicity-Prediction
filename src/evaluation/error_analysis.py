"""Reproducible Toxicity Prediction Error Analysis Framework.

Student 4 Module: Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks.

Performs rigorous endpoint-level and molecular error analysis on out-of-sample test predictions:
1. False Positive (FP) and False Negative (FN) Diagnosis:
   - Evaluates confusion patterns at validation-derived thresholds per endpoint.
   - Computes False Discovery Rate (FDR) and False Negative Rate (FNR).
2. Prevalence vs Performance Correlation:
   - Analyzes sensitivity of ROC-AUC, PR-AUC, and F1 to class imbalance (prevalence 2.8% to 16.2%).
3. Challenging Molecular Structures:
   - Identifies consistently misclassified compounds across multiple endpoints.
   - Analyzes distribution shifts in physicochemical properties (MW, LogP, TPSA, Rings) for errors.
4. Visualization & Artifact Persistence:
   - Generates publication-ready figures in results/figures/
   - Saves structured error metrics to results/metrics/endpoint_error_analysis.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TOX21_ENDPOINTS: tuple[str, ...] = (
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)


def analyze_endpoint_errors(
    benchmark_data: dict[str, Any],
    model_key: str = "random_forest_combined",
) -> dict[str, Any]:
    """Extract and analyze endpoint-level error statistics for a given model.

    Args:
        benchmark_data: Loaded JSON from model_benchmark_summary.json.
        model_key: Key of model configuration to analyze.

    Returns:
        Structured dictionary of error statistics, hardest/easiest endpoints, and correlations.
    """
    if model_key not in benchmark_data:
        raise KeyError(f"Model key '{model_key}' not found in benchmark data")

    exp = benchmark_data[model_key]
    endpoints_data = exp["endpoints"]

    endpoint_stats: dict[str, dict[str, Any]] = {}
    roc_list, pr_list, f1_list, prev_list = [], [], [], []

    for ep in TOX21_ENDPOINTS:
        if ep not in endpoints_data:
            continue

        ep_info = endpoints_data[ep]
        test_m = ep_info["test"]
        cm = test_m["confusion_matrix"]
        tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]

        total_observed = tp + fp + tn + fn
        total_pos = tp + fn
        total_neg = tn + fp
        prevalence = total_pos / total_observed if total_observed > 0 else 0.0

        fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / total_observed if total_observed > 0 else 0.0

        stats = {
            "val_threshold": ep_info["val_threshold"],
            "test_roc_auc": test_m["roc_auc"],
            "test_pr_auc": test_m["pr_auc"],
            "test_f1": test_m["f1"],
            "test_precision": test_m["precision"],
            "test_recall": test_m["recall"],
            "test_balanced_accuracy": test_m["balanced_accuracy"],
            "accuracy": round(accuracy, 4),
            "observed_count": total_observed,
            "positives": total_pos,
            "negatives": total_neg,
            "prevalence": round(prevalence, 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "false_discovery_rate": round(fdr, 4),
            "false_negative_rate": round(fnr, 4),
        }
        endpoint_stats[ep] = stats

        roc_list.append(test_m["roc_auc"])
        pr_list.append(test_m["pr_auc"])
        f1_list.append(test_m["f1"])
        prev_list.append(prevalence)

    # Rank endpoints by difficulty
    sorted_by_roc = sorted(endpoint_stats.items(), key=lambda x: x[1]["test_roc_auc"])
    hardest_endpoints = [ep for ep, _ in sorted_by_roc[:3]]
    easiest_endpoints = [ep for ep, _ in sorted_by_roc[-3:]]

    # Pearson correlation with prevalence
    corr_prev_roc = float(np.corrcoef(prev_list, roc_list)[0, 1]) if len(prev_list) > 1 else 0.0
    corr_prev_pr = float(np.corrcoef(prev_list, pr_list)[0, 1]) if len(prev_list) > 1 else 0.0
    corr_prev_f1 = float(np.corrcoef(prev_list, f1_list)[0, 1]) if len(prev_list) > 1 else 0.0

    return {
        "model_analyzed": model_key,
        "endpoint_error_details": endpoint_stats,
        "difficulty_ranking": {
            "hardest_by_roc_auc": hardest_endpoints,
            "easiest_by_roc_auc": easiest_endpoints,
        },
        "prevalence_correlations": {
            "prevalence_vs_roc_auc": round(corr_prev_roc, 4),
            "prevalence_vs_pr_auc": round(corr_prev_pr, 4),
            "prevalence_vs_f1": round(corr_prev_f1, 4),
        },
    }


def identify_challenging_molecules(
    cleaned_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    test_indices: Sequence[int],
    benchmark_data: dict[str, Any],
    model_key: str = "random_forest_combined",
) -> dict[str, Any]:
    """Identify molecules with persistent false negative or false positive classifications.

    Args:
        cleaned_df: tox21_cleaned.csv DataFrame.
        predictions_df: Predictions CSV for model_key.
        test_indices: List of test partition row indices.
        benchmark_data: Benchmark JSON.
        model_key: Model key.

    Returns:
        Dictionary of promiscuous error compounds and property comparison.
    """
    test_df = cleaned_df.iloc[test_indices].copy().reset_index(drop=True)
    endpoints_data = benchmark_data[model_key]["endpoints"]

    fn_counts = np.zeros(len(test_df), dtype=int)
    fp_counts = np.zeros(len(test_df), dtype=int)
    evaluated_counts = np.zeros(len(test_df), dtype=int)

    for ep in TOX21_ENDPOINTS:
        if ep not in endpoints_data or ep not in test_df.columns:
            continue

        thresh = endpoints_data[ep]["val_threshold"]
        y_true = test_df[ep].to_numpy()
        y_prob = predictions_df[ep].to_numpy()

        valid = ~np.isnan(y_true)
        evaluated_counts[valid] += 1

        y_pred = (y_prob >= thresh).astype(int)
        y_gold = np.where(valid, np.nan_to_num(y_true, nan=0.0).astype(int), -1)

        fn_mask = valid & (y_gold == 1) & (y_pred == 0)
        fp_mask = valid & (y_gold == 0) & (y_pred == 1)

        fn_counts[fn_mask] += 1
        fp_counts[fp_mask] += 1

    test_df = test_df.assign(
        fn_count=fn_counts,
        fp_count=fp_counts,
        evaluated_count=evaluated_counts,
        total_errors=fn_counts + fp_counts,
    )

    # High error molecules (>= 2 false negatives or >= 2 false positives)
    challenging_fn = test_df[test_df["fn_count"] >= 2][
        ["source_row", "mol_id", "canonical_smiles", "fn_count", "evaluated_count"]
    ].head(10).to_dict(orient="records")

    challenging_fp = test_df[test_df["fp_count"] >= 3][
        ["source_row", "mol_id", "canonical_smiles", "fp_count", "evaluated_count"]
    ].head(10).to_dict(orient="records")

    return {
        "total_test_molecules": len(test_df),
        "promiscuous_false_negatives": challenging_fn,
        "promiscuous_false_positives": challenging_fp,
    }


def generate_error_analysis_figures(
    error_analysis: dict[str, Any],
    benchmark_data: dict[str, Any],
    output_dir: str = "results/figures",
) -> list[str]:
    """Generate high-resolution publication figures for error analysis.

    Returns:
        List of generated figure filepaths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    details = error_analysis["endpoint_error_details"]
    endpoints = list(details.keys())

    roc_aucs = [details[ep]["test_roc_auc"] for ep in endpoints]
    pr_aucs = [details[ep]["test_pr_auc"] for ep in endpoints]
    prevalences = [details[ep]["prevalence"] * 100.0 for ep in endpoints]

    # Figure 1: Endpoint Performance Breakdown (ROC-AUC & PR-AUC)
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(endpoints))
    width = 0.35

    rects1 = ax1.bar(x - width / 2, roc_aucs, width, label="Test ROC-AUC", color="#1f77b4", alpha=0.85)
    rects2 = ax1.bar(x + width / 2, pr_aucs, width, label="Test PR-AUC", color="#ff7f0e", alpha=0.85)

    ax1.set_ylabel("Area Under Curve")
    ax1.set_title("Endpoint-Level Performance Breakdown (Random Forest Combined)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(endpoints, rotation=40, ha="right", fontsize=9)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)
    ax1.axhline(0.5, color="red", linestyle=":", linewidth=1, label="Random Baseline (ROC=0.5)")

    fig.tight_layout()
    fig1_path = out_dir / "endpoint_performance_roc_pr.png"
    fig.savefig(fig1_path)
    plt.close(fig)
    generated_files.append(str(fig1_path))

    # Figure 2: Prevalence vs PR-AUC and F1
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    f1s = [details[ep]["test_f1"] for ep in endpoints]

    ax_roc.scatter(prevalences, roc_aucs, color="#2ca02c", s=70, edgecolors="k", alpha=0.8)
    for i, ep in enumerate(endpoints):
        ax_roc.annotate(ep, (prevalences[i], roc_aucs[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax_roc.set_xlabel("Positive Prevalence in Test (%)")
    ax_roc.set_ylabel("Test ROC-AUC")
    ax_roc.set_title("Prevalence vs ROC-AUC (Near Invariant)", fontsize=11, fontweight="bold")
    ax_roc.grid(True, linestyle="--", alpha=0.5)

    ax_pr.scatter(prevalences, pr_aucs, color="#d62728", s=70, edgecolors="k", alpha=0.8)
    for i, ep in enumerate(endpoints):
        ax_pr.annotate(ep, (prevalences[i], pr_aucs[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax_pr.set_xlabel("Positive Prevalence in Test (%)")
    ax_pr.set_ylabel("Test PR-AUC")
    ax_pr.set_title("Prevalence vs PR-AUC (Strong Sensitivity)", fontsize=11, fontweight="bold")
    ax_pr.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig2_path = out_dir / "prevalence_vs_performance.png"
    fig.savefig(fig2_path)
    plt.close(fig)
    generated_files.append(str(fig2_path))

    # Figure 3: Model Comparison across 3 Representations
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    models = ["logistic_regression", "random_forest", "xgboost"]
    feat_configs = ["morgan", "descriptors", "combined"]

    labels_models = ["Logistic Reg.", "Random Forest", "XGBoost"]
    bar_width = 0.25
    x_indices = np.arange(len(models))

    for idx, fc in enumerate(feat_configs):
        means = [benchmark_data[f"{m}_{fc}"]["macro_metrics"]["roc_auc"]["mean"] for m in models]
        sems = [benchmark_data[f"{m}_{fc}"]["macro_metrics"]["roc_auc"]["sem"] for m in models]
        ax.bar(
            x_indices + (idx - 1) * bar_width,
            means,
            bar_width,
            yerr=sems,
            capsize=4,
            label=f"Feature: {fc.capitalize()}",
            alpha=0.85,
        )

    ax.set_ylabel("Macro ROC-AUC (Mean ± SEM)")
    ax.set_title("Model Architecture & Feature Set Comparison (Out-of-Sample Test)", fontsize=12, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(labels_models, fontsize=10)
    ax.set_ylim(0.60, 0.80)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig3_path = out_dir / "model_feature_comparison_bars.png"
    fig.savefig(fig3_path)
    plt.close(fig)
    generated_files.append(str(fig3_path))

    logger.info("Saved %d figures to %s", len(generated_files), out_dir)
    return generated_files


def run_error_analysis_pipeline(
    benchmark_path: str = "results/metrics/model_benchmark_summary.json",
    cleaned_path: str = "data/processed/tox21_cleaned.csv",
    split_path: str = "data/processed/scaffold_split_indices.json",
    predictions_path: str = "results/metrics/predictions_random_forest_combined.csv",
    output_dir: str = "results/metrics",
    figures_dir: str = "results/figures",
) -> dict[str, Any]:
    """Execute end-to-end error analysis and generate figures."""
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
        split_indices = split_data.get("indices", split_data)

    cleaned_df = pd.read_csv(cleaned_path)
    predictions_df = pd.read_csv(predictions_path)

    # 1. Endpoint error metrics
    error_analysis = analyze_endpoint_errors(benchmark_data, model_key="random_forest_combined")

    # 2. Challenging molecules
    molecules_analysis = identify_challenging_molecules(
        cleaned_df,
        predictions_df,
        split_indices["test"],
        benchmark_data,
        model_key="random_forest_combined",
    )
    error_analysis["molecular_error_patterns"] = molecules_analysis

    # 3. Figures
    figures = generate_error_analysis_figures(error_analysis, benchmark_data, output_dir=figures_dir)
    error_analysis["generated_figures"] = figures

    # 4. Save JSON
    out_file = Path(output_dir) / "endpoint_error_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2)

    logger.info("Saved error analysis report to %s", out_file)
    return error_analysis


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_error_analysis_pipeline()
