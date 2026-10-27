"""Unit and regression tests for Student 3 Classical ML Models.

Tests:
- LogisticRegressionClassifier instantiation, fit, predict_proba, coef shapes
- RandomForestClassifier instantiation, fit, predict_proba, feature importances
- XGBoostClassifier instantiation, fit, predict_proba, scale_pos_weight
- Factory build_classifier functionality
- Model serialization and deserialization via joblib
"""

import tempfile
from pathlib import Path
import joblib
import numpy as np
import pytest

from src.models.ml import (
    LogisticRegressionClassifier,
    RandomForestClassifier,
    XGBoostClassifier,
    build_classifier,
)


@pytest.fixture
def synthetic_binary_data():
    """Create reproducible synthetic binary dataset with 10% positive imbalance."""
    rng = np.random.RandomState(42)
    n_samples = 150
    n_features = 32
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = np.zeros(n_samples, dtype=int)
    y[rng.choice(n_samples, size=15, replace=False)] = 1
    return X, y


def test_logistic_regression_fit_predict(synthetic_binary_data):
    X, y = synthetic_binary_data
    clf = LogisticRegressionClassifier(C=1.0, random_state=42)
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (len(X), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert set(preds).issubset({0, 1})
    assert clf.coef_.shape == (1, X.shape[1])
    assert clf.intercept_.shape == (1,)


def test_random_forest_fit_predict(synthetic_binary_data):
    X, y = synthetic_binary_data
    clf = RandomForestClassifier(n_estimators=15, max_depth=6, random_state=42)
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (len(X), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert set(preds).issubset({0, 1})
    assert len(clf.feature_importances_) == X.shape[1]
    assert np.isclose(np.sum(clf.feature_importances_), 1.0, atol=1e-4)


def test_xgboost_fit_predict(synthetic_binary_data):
    X, y = synthetic_binary_data
    clf = XGBoostClassifier(pos_neg_ratio=9.0, n_estimators=15, max_depth=3, random_state=42)
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (len(X), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert set(preds).issubset({0, 1})


def test_build_classifier_factory():
    lr = build_classifier("logistic_regression")
    assert isinstance(lr, LogisticRegressionClassifier)

    rf = build_classifier("random_forest")
    assert isinstance(rf, RandomForestClassifier)

    xgb = build_classifier("xgboost", pos_neg_ratio=5.0)
    assert isinstance(xgb, XGBoostClassifier)

    with pytest.raises(ValueError, match="Unknown model_type"):
        build_classifier("support_vector_machine")


def test_model_joblib_serialization(synthetic_binary_data):
    X, y = synthetic_binary_data
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X, y)
    orig_probs = rf.predict_proba(X)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_rf.joblib"
        joblib.dump(rf, model_path)
        assert model_path.exists()

        loaded_rf = joblib.load(model_path)
        loaded_probs = loaded_rf.predict_proba(X)
        np.testing.assert_allclose(orig_probs, loaded_probs, atol=1e-6)
