"""Model Execution and Experiment Runner Component.

Student 3 Module: Classical ML Models and Experiments (Model Owner).

Coordinates training of baseline classifiers (Logistic Regression, Random Forest, XGBoost)
across the 12 Tox21 endpoints and saves fitted model artifacts to results/models/.
Keeps test-set labels strictly out of model selection and hyperparameter tuning.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from src.models.ml import build_classifier
from src.evaluation.aggregation import (
    aggregate_endpoint_metrics,
    compute_uncertainty_stats,
)
from src.evaluation.metrics import (
    compute_pr_auc,
    compute_roc_auc,
    compute_threshold_metrics,
    find_optimal_threshold,
)
from src.features.preprocessor import (
    LeakageSafeScaler,
    PartitionPreprocessor,
)

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


def train_and_save_endpoint_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    endpoint_name: str,
    model_type: str = "random_forest",
    random_state: int = 42,
    model_save_path: Path | None = None,
) -> dict[str, Any]:
    """Train classifier on training split, tune decision threshold on validation split,
    evaluate on test partition, and optionally serialize the fitted model to disk.
    """
    train_mask = ~np.isnan(y_train)
    X_tr_valid = X_train[train_mask]
    y_tr_valid = y_train[train_mask].astype(int)

    n_pos_tr = int(np.sum(y_tr_valid == 1))
    n_neg_tr = int(np.sum(y_tr_valid == 0))

    if n_pos_tr == 0 or n_neg_tr == 0:
        logger.warning("Endpoint %s has single-class training slice", endpoint_name)
        return {"endpoint": endpoint_name, "status": "single_class_train"}

    pos_neg_ratio = n_neg_tr / n_pos_tr if n_pos_tr > 0 else 1.0

    # Instantiate via Student 3 model package
    clf = build_classifier(model_type, pos_neg_ratio=pos_neg_ratio, random_state=random_state)
    clf.fit(X_tr_valid, y_tr_valid)

    # Serialize fitted model artifact
    if model_save_path is not None:
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, model_save_path)

    # Predict on Validation partition
    val_mask = ~np.isnan(y_val)
    val_probs = np.full(len(y_val), np.nan, dtype=np.float64)
    if np.sum(val_mask) > 0:
        val_probs[val_mask] = clf.predict_proba(X_val[val_mask])[:, 1]

    # Tune decision threshold on validation partition only
    val_roc_auc = compute_roc_auc(y_val, val_probs)
    val_pr_auc = compute_pr_auc(y_val, val_probs)
    val_threshold = find_optimal_threshold(y_val, val_probs, metric="f1")
    val_thresh_metrics = compute_threshold_metrics(y_val, val_probs, threshold=val_threshold)

    # Predict on unseen Test partition
    test_mask = ~np.isnan(y_test)
    test_probs = np.full(len(y_test), np.nan, dtype=np.float64)
    if np.sum(test_mask) > 0:
        test_probs[test_mask] = clf.predict_proba(X_test[test_mask])[:, 1]

    # Evaluate on Test partition using the validation-derived threshold
    test_roc_auc = compute_roc_auc(y_test, test_probs)
    test_pr_auc = compute_pr_auc(y_test, test_probs)
    test_thresh_metrics = compute_threshold_metrics(y_test, test_probs, threshold=val_threshold)

    return {
        "endpoint": endpoint_name,
        "model_type": model_type,
        "train_samples": int(np.sum(train_mask)),
        "train_positives": n_pos_tr,
        "train_negatives": n_neg_tr,
        "val_threshold": val_threshold,
        "validation": {
            "roc_auc": val_roc_auc,
            "pr_auc": val_pr_auc,
            "f1": val_thresh_metrics["f1"],
            "precision": val_thresh_metrics["precision"],
            "recall": val_thresh_metrics["recall"],
            "balanced_accuracy": val_thresh_metrics["balanced_accuracy"],
        },
        "test": {
            "roc_auc": test_roc_auc,
            "pr_auc": test_pr_auc,
            "f1": test_thresh_metrics["f1"],
            "precision": test_thresh_metrics["precision"],
            "recall": test_thresh_metrics["recall"],
            "balanced_accuracy": test_thresh_metrics["balanced_accuracy"],
            "confusion_matrix": {
                "tp": test_thresh_metrics["tp"],
                "fp": test_thresh_metrics["fp"],
                "tn": test_thresh_metrics["tn"],
                "fn": test_thresh_metrics["fn"],
            },
        },
        "val_probabilities": val_probs,
        "test_probabilities": test_probs,
    }


def run_experiments(
    features_path: str = "data/processed/features/combined_features.npy",
    labels_path: str = "data/processed/tox21_cleaned.csv",
    split_path: str = "data/processed/scaffold_split_indices.json",
    metrics_output_dir: str = "results/metrics",
    models_output_dir: str = "results/models",
    random_state: int = 42,
    save_models: bool = True,
) -> dict[str, Any]:
    """Execute complete multi-task benchmark suite, saving models and metrics."""
    logger.info("Loading feature matrix from %s...", features_path)
    X_combined = np.load(features_path)

    logger.info("Loading labels from %s...", labels_path)
    df_labels = pd.read_csv(labels_path)
    Y_labels = df_labels[list(TOX21_ENDPOINTS)].to_numpy(dtype=np.float64)

    logger.info("Loading scaffold splits from %s...", split_path)
    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
        split_indices = split_data.get("indices", split_data)

    models = ["logistic_regression", "random_forest", "xgboost"]
    feature_sets = ["morgan", "descriptors", "combined"]

    all_experiments: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    metrics_out = Path(metrics_output_dir)
    metrics_out.mkdir(parents=True, exist_ok=True)
    models_out = Path(models_output_dir)
    models_out.mkdir(parents=True, exist_ok=True)

    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    for model in models:
        for feat in feature_sets:
            exp_key = f"{model}_{feat}"
            logger.info("Running experiment: %s...", exp_key)

            # Feature subsetting & leak-free preprocessing
            if feat == "morgan":
                X_feat = X_combined[:, :2048]
                X_train, X_val, X_test = X_feat[train_idx], X_feat[val_idx], X_feat[test_idx]
            elif feat == "descriptors":
                X_feat = X_combined[:, 2048:]
                scaler = LeakageSafeScaler(with_mean=True, with_std=True)
                scaler.fit(X_feat[train_idx])
                X_train = scaler.transform(X_feat[train_idx])
                X_val = scaler.transform(X_feat[val_idx])
                X_test = scaler.transform(X_feat[test_idx])
            elif feat == "combined":
                preprocessor = PartitionPreprocessor(n_morgan_bits=2048, n_descriptors=9)
                preprocessor.fit(X_combined, train_indices=train_idx)
                X_train = preprocessor.transform(X_combined[train_idx])
                X_val = preprocessor.transform(X_combined[val_idx])
                X_test = preprocessor.transform(X_combined[test_idx])
            else:
                raise ValueError(f"Unknown feature set: {feat}")

            per_endpoint_results: dict[str, Any] = {}
            test_roc_aucs: list[float] = []
            test_pr_aucs: list[float] = []
            test_f1s: list[float] = []
            test_baccs: list[float] = []

            test_pred_matrix = np.full((len(test_idx), len(TOX21_ENDPOINTS)), np.nan, dtype=np.float32)

            for i, ep in enumerate(TOX21_ENDPOINTS):
                y_tr = Y_labels[train_idx, i]
                y_v = Y_labels[val_idx, i]
                y_te = Y_labels[test_idx, i]

                model_file = models_out / exp_key / f"{ep}.joblib" if save_models else None

                ep_eval = train_and_save_endpoint_model(
                    X_train, y_tr,
                    X_val, y_v,
                    X_test, y_te,
                    endpoint_name=ep,
                    model_type=model,
                    random_state=random_state,
                    model_save_path=model_file,
                )

                per_endpoint_results[ep] = {k: v for k, v in ep_eval.items() if not k.endswith("probabilities")}
                test_pred_matrix[:, i] = ep_eval["test_probabilities"]

                test_roc_aucs.append(ep_eval["test"]["roc_auc"])
                test_pr_aucs.append(ep_eval["test"]["pr_auc"])
                test_f1s.append(ep_eval["test"]["f1"])
                test_baccs.append(ep_eval["test"]["balanced_accuracy"])

            # Save test prediction scores
            pred_df = pd.DataFrame(test_pred_matrix, columns=list(TOX21_ENDPOINTS))
            pred_df["scaffold_test_row_idx"] = test_idx
            pred_file = metrics_out / f"predictions_{exp_key}.csv"
            pred_df.to_csv(pred_file, index=False)

            # Macro uncertainty aggregation
            ep_summary = {
                ep: {
                    "roc_auc": per_endpoint_results[ep]["test"]["roc_auc"],
                    "pr_auc": per_endpoint_results[ep]["test"]["pr_auc"],
                    "status": "valid" if not np.isnan(per_endpoint_results[ep]["test"]["roc_auc"]) else "single_class_undefined",
                }
                for ep in TOX21_ENDPOINTS
            }
            macro_agg = aggregate_endpoint_metrics(ep_summary)
            macro_f1 = compute_uncertainty_stats(test_f1s)
            macro_bacc = compute_uncertainty_stats(test_baccs)

            roc_m = macro_agg["macro_roc_auc"]
            pr_m = macro_agg["macro_pr_auc"]

            summary_rows.append({
                "Model": model,
                "Feature Set": feat,
                "ROC-AUC (Mean ± SEM)": f"{roc_m['mean']:.4f} ± {roc_m['sem']:.4f}",
                "ROC-AUC Mean": round(roc_m["mean"], 4),
                "ROC-AUC SD": round(roc_m["sd"], 4),
                "ROC-AUC SEM": round(roc_m["sem"], 4),
                "PR-AUC (Mean ± SEM)": f"{pr_m['mean']:.4f} ± {pr_m['sem']:.4f}",
                "PR-AUC Mean": round(pr_m["mean"], 4),
                "PR-AUC SEM": round(pr_m["sem"], 4),
                "F1 Mean": round(macro_f1["mean"], 4),
                "Balanced Acc Mean": round(macro_bacc["mean"], 4),
            })

            res_serializable = {
                "model_type": model,
                "feature_config": feat,
                "random_state": random_state,
                "macro_metrics": {
                    "roc_auc": macro_agg["macro_roc_auc"],
                    "pr_auc": macro_agg["macro_pr_auc"],
                    "f1": macro_f1,
                    "balanced_accuracy": macro_bacc,
                },
                "endpoints": per_endpoint_results,
            }
            all_experiments[exp_key] = res_serializable

    # Save summary JSON and table CSV
    summary_json_file = metrics_out / "model_benchmark_summary.json"
    with open(summary_json_file, "w", encoding="utf-8") as f:
        json.dump(all_experiments, f, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    table_csv_file = metrics_out / "model_comparison_table.csv"
    summary_df.to_csv(table_csv_file, index=False)

    logger.info("All experiments completed successfully. Models saved to %s", models_out)
    return {"experiments": all_experiments, "summary_table": summary_rows}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Tox21 Classical ML Experiments Runner")
    parser.add_argument("--features", default="data/processed/features/combined_features.npy", help="Features path")
    parser.add_argument("--labels", default="data/processed/tox21_cleaned.csv", help="Cleaned labels path")
    parser.add_argument("--split", default="data/processed/scaffold_split_indices.json", help="Split indices path")
    parser.add_argument("--metrics-output", default="results/metrics", help="Metrics output directory")
    parser.add_argument("--models-output", default="results/models", help="Models output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-save-models", action="store_true", help="Do not serialize model weights to disk")
    args = parser.parse_args()

    run_experiments(
        features_path=args.features,
        labels_path=args.labels,
        split_path=args.split,
        metrics_output_dir=args.metrics_output,
        models_output_dir=args.models_output,
        random_state=args.seed,
        save_models=not args.no_save_models,
    )
