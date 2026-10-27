"""Classical ML baseline classifiers for Tox21 multi-task toxicity prediction.

Exports:
    LogisticRegressionClassifier
    RandomForestClassifier
    XGBoostClassifier
    build_classifier
"""

from src.models.ml.logistic_regression import LogisticRegressionClassifier
from src.models.ml.random_forest import RandomForestClassifier
from src.models.ml.xgboost_classifier import XGBoostClassifier


def build_classifier(model_type: str, pos_neg_ratio: float = 1.0, random_state: int = 42):
    """Factory function: instantiate a baseline classifier by name.

    Args:
        model_type: One of 'logistic_regression', 'random_forest', 'xgboost'.
        pos_neg_ratio: Ratio of negative to positive samples for XGBoost
            scale_pos_weight. Unused by LR and RF (use class_weight='balanced').
        random_state: Integer seed for reproducibility.

    Returns:
        Fitted-ready scikit-learn compatible classifier instance.

    Raises:
        ValueError: If model_type is not recognised.
    """
    if model_type == "logistic_regression":
        return LogisticRegressionClassifier(random_state=random_state)
    elif model_type == "random_forest":
        return RandomForestClassifier(random_state=random_state)
    elif model_type == "xgboost":
        return XGBoostClassifier(pos_neg_ratio=pos_neg_ratio, random_state=random_state)
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            "Choose one of: 'logistic_regression', 'random_forest', 'xgboost'."
        )


__all__ = [
    "LogisticRegressionClassifier",
    "RandomForestClassifier",
    "XGBoostClassifier",
    "build_classifier",
]
