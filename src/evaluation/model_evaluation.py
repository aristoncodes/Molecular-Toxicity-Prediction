"""Model Training, Validation-Tuning, and Benchmark Evaluation Framework.

Student 4 Module: Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks.

Implements rigorous, leak-free model evaluation across the 12 Tox21 endpoints:
1. Baseline Algorithms:
   - Logistic Regression (L2 regularized, class_weight='balanced')
   - Random Forest (100 estimators, class_weight='balanced')
   - XGBoost (scale_pos_weight tuned per-endpoint from training imbalance)
2. Feature Configurations:
   - Morgan Fingerprints (2048-dim binary bit vectors)
   - Physicochemical Descriptors (9 standardized continuous features)
   - Combined Representation (2057-dim concatenation with isolated descriptor scaling)
3. Scientific Protocol:
   - Partition isolation: strictly respects Bemis-Murcko scaffold split indices.
   - Preprocessing isolation: PartitionPreprocessor fits scalers strictly on training partition.
   - Threshold tuning: Decision boundaries optimized strictly on VALIDATION split (never test).
   - Missing labels: Evaluated strictly per-endpoint with NaN masking (never zero-imputed).
   - Aggregation: Macro Mean ± SEM, SD, and 95% Student-t confidence intervals across 12 endpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

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


def create_classifier(
    model_type: str,
    pos_neg_ratio: float = 1.0,
    random_state: int = 42,
) -> Any:
    """Instantiate a baseline binary classifier with class-imbalance weighting."""
    if model_type == "logistic_regression":
        return LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "xgboost":
        # scale_pos_weight handles imbalance: count(negative) / count(positive)
        scale_pos = max(1.0, float(pos_neg_ratio))
        return XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}' (choose logistic_regression, random_forest, or xgboost)")


def train_and_evaluate_endpoint(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    endpoint_name: str,
    model_type: str = "random_forest",
    random_state: int = 42,
) -> dict[str, Any]:
    """Train on training partition, tune threshold on validation, and evaluate on test partition.

    Args:
        X_train, y_train: Training features and labels (may contain NaNs in y).
        X_val, y_val: Validation features and labels.
        X_test, y_test: Test features and labels.
        endpoint_name: Name of toxicity endpoint.
        model_type: 'logistic_regression', 'random_forest', or 'xgboost'.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with validation threshold, validation metrics, and test metrics.
    """
    # 1. Filter out missing labels from training set
    train_mask = ~np.isnan(y_train)
    X_tr_valid = X_train[train_mask]
    y_tr_valid = y_train[train_mask].astype(int)

    n_pos_tr = int(np.sum(y_tr_valid == 1))
    n_neg_tr = int(np.sum(y_tr_valid == 0))

    if n_pos_tr == 0 or n_neg_tr == 0:
        logger.warning("Endpoint %s has single-class training slice", endpoint_name)
        return {"endpoint": endpoint_name, "status": "single_class_train"}

    pos_neg_ratio = n_neg_tr / n_pos_tr if n_pos_tr > 0 else 1.0

    # 2. Fit model strictly on training data
    clf = create_classifier(model_type, pos_neg_ratio=pos_neg_ratio, random_state=random_state)
    clf.fit(X_tr_valid, y_tr_valid)

    # 3. Predict on Validation partition
    val_mask = ~np.isnan(y_val)
    val_probs = np.full(len(y_val), np.nan, dtype=np.float64)
    if np.sum(val_mask) > 0:
        val_probs[val_mask] = clf.predict_proba(X_val[val_mask])[:, 1]

    # 4. Strictly Validation-Derived Threshold Tuning
    val_roc_auc = compute_roc_auc(y_val, val_probs)
    val_pr_auc = compute_pr_auc(y_val, val_probs)
    val_threshold = find_optimal_threshold(y_val, val_probs, metric="f1")
    val_thresh_metrics = compute_threshold_metrics(y_val, val_probs, threshold=val_threshold)

    # 5. Predict on Test partition
    test_mask = ~np.isnan(y_test)
    test_probs = np.full(len(y_test), np.nan, dtype=np.float64)
    if np.sum(test_mask) > 0:
        test_probs[test_mask] = clf.predict_proba(X_test[test_mask])[:, 1]

    # 6. Evaluate on Test partition using the validation-derived threshold
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


def evaluate_benchmark_experiment(
    X_raw: np.ndarray,
    Y_labels: np.ndarray,
    split_indices: Mapping[str, Sequence[int]],
    feature_config: str = "combined",
    model_type: str = "random_forest",
    endpoint_names: Sequence[str] = TOX21_ENDPOINTS,
    random_state: int = 42,
) -> dict[str, Any]:
    """Execute a complete 12-endpoint benchmark experiment under strict partition leakage isolation.

    Args:
        X_raw: Feature matrix (N, D).
        Y_labels: Label matrix (N, 12).
        split_indices: Dict with 'train', 'val', 'test' index lists.
        feature_config: 'morgan', 'descriptors', or 'combined'.
        model_type: 'logistic_regression', 'random_forest', or 'xgboost'.
        endpoint_names: Names of endpoints.
        random_state: Seed.

    Returns:
        Dictionary with per-endpoint results and aggregated macro uncertainty statistics.
    """
    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    # Preprocessing with strict out-of-sample scaling
    if feature_config == "morgan":
        X_feat = X_raw[:, :2048]
        X_train, X_val, X_test = X_feat[train_idx], X_feat[val_idx], X_feat[test_idx]
    elif feature_config == "descriptors":
        X_feat = X_raw[:, 2048:] if X_raw.shape[1] == 2057 else X_raw
        scaler = LeakageSafeScaler(with_mean=True, with_std=True)
        scaler.fit(X_feat[train_idx])
        X_train = scaler.transform(X_feat[train_idx])
        X_val = scaler.transform(X_feat[val_idx])
        X_test = scaler.transform(X_feat[test_idx])
    elif feature_config == "combined":
        preprocessor = PartitionPreprocessor(n_morgan_bits=2048, n_descriptors=9)
        preprocessor.fit(X_raw, train_indices=train_idx)
        X_train = preprocessor.transform(X_raw[train_idx])
        X_val = preprocessor.transform(X_raw[val_idx])
        X_test = preprocessor.transform(X_raw[test_idx])
    else:
        raise ValueError(f"Unknown feature_config: '{feature_config}'")

    per_endpoint_results: dict[str, Any] = {}
    test_roc_aucs: list[float] = []
    test_pr_aucs: list[float] = []
    test_f1s: list[float] = []
    test_baccs: list[float] = []

    test_pred_matrix = np.full((len(test_idx), len(endpoint_names)), np.nan, dtype=np.float32)

    for i, ep in enumerate(endpoint_names):
        y_tr = Y_labels[train_idx, i]
        y_v = Y_labels[val_idx, i]
        y_te = Y_labels[test_idx, i]

        ep_eval = train_and_evaluate_endpoint(
            X_train, y_tr,
            X_val, y_v,
            X_test, y_te,
            endpoint_name=ep,
            model_type=model_type,
            random_state=random_state,
        )
        per_endpoint_results[ep] = {k: v for k, v in ep_eval.items() if not k.endswith("probabilities")}
        test_pred_matrix[:, i] = ep_eval["test_probabilities"]

        test_roc_aucs.append(ep_eval["test"]["roc_auc"])
        test_pr_aucs.append(ep_eval["test"]["pr_auc"])
        test_f1s.append(ep_eval["test"]["f1"])
        test_baccs.append(ep_eval["test"]["balanced_accuracy"])

    # Macro aggregation using standard aggregation module
    ep_summary_dict = {
        ep: {
            "roc_auc": per_endpoint_results[ep]["test"]["roc_auc"],
            "pr_auc": per_endpoint_results[ep]["test"]["pr_auc"],
            "status": "valid" if not np.isnan(per_endpoint_results[ep]["test"]["roc_auc"]) else "single_class_undefined",
        }
        for ep in endpoint_names
    }
    macro_agg = aggregate_endpoint_metrics(ep_summary_dict)
    macro_f1 = compute_uncertainty_stats(test_f1s)
    macro_bacc = compute_uncertainty_stats(test_baccs)

    return {
        "model_type": model_type,
        "feature_config": feature_config,
        "random_state": random_state,
        "macro_metrics": {
            "roc_auc": macro_agg["macro_roc_auc"],
            "pr_auc": macro_agg["macro_pr_auc"],
            "f1": macro_f1,
            "balanced_accuracy": macro_bacc,
        },
        "endpoints": per_endpoint_results,
        "test_predictions": test_pred_matrix,
    }


def run_full_model_benchmarks(
    features_path: str = "data/processed/features/combined_features.npy",
    labels_path: str = "data/processed/tox21_cleaned.csv",
    split_path: str = "data/processed/scaffold_split_indices.json",
    output_dir: str = "results/metrics",
    random_state: int = 42,
) -> dict[str, Any]:
    """Run comparative benchmark across all 3 models and 3 feature sets (9 experiment configurations).

    Saves results to results/metrics/model_benchmark_summary.json and comparative CSV table.
    """
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

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for model in models:
        for feat in feature_sets:
            exp_key = f"{model}_{feat}"
            logger.info("Evaluating configuration: %s...", exp_key)

            res = evaluate_benchmark_experiment(
                X_combined,
                Y_labels,
                split_indices,
                feature_config=feat,
                model_type=model,
                random_state=random_state,
            )

            # Save predictions CSV for auditability
            pred_df = pd.DataFrame(res["test_predictions"], columns=list(TOX21_ENDPOINTS))
            pred_df["scaffold_test_row_idx"] = split_indices["test"]
            pred_file = out_path / f"predictions_{exp_key}.csv"
            pred_df.to_csv(pred_file, index=False)

            # Record summary metrics
            roc_m = res["macro_metrics"]["roc_auc"]
            pr_m = res["macro_metrics"]["pr_auc"]
            f1_m = res["macro_metrics"]["f1"]
            bacc_m = res["macro_metrics"]["balanced_accuracy"]

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
                "F1 Mean": round(f1_m["mean"], 4),
                "Balanced Acc Mean": round(bacc_m["mean"], 4),
            })

            # Strip array for JSON serialization
            res_serializable = {k: v for k, v in res.items() if k != "test_predictions"}
            all_experiments[exp_key] = res_serializable

    # Save summary JSON
    summary_json_file = out_path / "model_benchmark_summary.json"
    with open(summary_json_file, "w", encoding="utf-8") as f:
        json.dump(all_experiments, f, indent=2)

    # Save comparative table CSV
    summary_df = pd.DataFrame(summary_rows)
    table_csv_file = out_path / "model_comparison_table.csv"
    summary_df.to_csv(table_csv_file, index=False)

    logger.info("Saved benchmark summary to %s and comparison table to %s", summary_json_file, table_csv_file)
    return {"experiments": all_experiments, "summary_table": summary_rows}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Tox21 Benchmark Model Evaluation")
    parser.add_argument("--features", default="data/processed/features/combined_features.npy", help="Features path")
    parser.add_argument("--labels", default="data/processed/tox21_cleaned.csv", help="Cleaned labels path")
    parser.add_argument("--split", default="data/processed/scaffold_split_indices.json", help="Split indices path")
    parser.add_argument("--output", default="results/metrics", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_full_model_benchmarks(
        features_path=args.features,
        labels_path=args.labels,
        split_path=args.split,
        output_dir=args.output,
        random_state=args.seed,
    )
