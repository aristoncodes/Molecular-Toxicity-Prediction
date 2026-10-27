"""Random Forest baseline classifier for Tox21 multi-task toxicity prediction.

Student 3 Module — Classical ML Models and Experiments.

Design choices
--------------
* n_estimators=100: Standard ensemble size providing a good bias-variance trade-off
  without excessive compute. Each tree sees a bootstrap sample and a random feature
  subset, producing decorrelated predictors.
* max_depth=12: Caps individual tree depth to prevent individual trees from
  perfectly memorising training scaffold clusters. Deep trees in high-dimensional
  fingerprint spaces can overfit rare chemical sub-structures.
* min_samples_split=5: Requires at least 5 training samples to split a node,
  preventing highly specific splits on single molecules.
* class_weight='balanced': Assigns sample weights inversely proportional to class
  frequencies at each bootstrap: weight_i = n_samples / (n_classes * bin_count[y_i]).
  This corrects for 2.84%–15.82% positive prevalence across endpoints.
* n_jobs=-1: Uses all available CPU cores for parallel tree construction.

Why Random Forest is an effective QSAR baseline
-----------------------------------------------
* Non-linearity: Captures non-additive interactions between Morgan bits
  (e.g., a toxicophore requiring both a nitro group AND an aromatic amine).
* Feature importance: Gini impurity reduction across trees yields interpretable
  per-bit importance scores; the top Morgan bits correlate with known toxicophores.
* Robustness: Bagging reduces variance from scaffold clustering. RF is empirically
  one of the most reliable QSAR models (Svetnik et al., 2003; Mayr et al., 2016).
* No scaling: Binary Morgan bits require no normalisation; RF is invariant to
  monotonic feature transformations.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier as _SKLearnRF
from sklearn.base import BaseEstimator, ClassifierMixin


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    """Ensemble Random Forest classifier with balanced class weighting.

    Wraps sklearn.ensemble.RandomForestClassifier with project-standard
    hyperparameters validated on the Tox21 scaffold split benchmark.

    Parameters
    ----------
    n_estimators : int
        Number of decision trees in the ensemble.
    max_depth : int or None
        Maximum depth of each decision tree. None = unlimited.
    min_samples_split : int
        Minimum samples required to split an internal node.
    random_state : int
        Seed for reproducibility of bootstrap sampling and feature selection.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 12,
        min_samples_split: int = 5,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self._clf = _SKLearnRF(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Fit the Random Forest ensemble.

        Args:
            X: Feature matrix of shape (n_samples, n_features). Accepts binary
               Morgan bits, scaled descriptors, or combined matrices.
            y: Binary label vector of shape (n_samples,). Must be 0 or 1.

        Returns:
            self
        """
        self._clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return averaged class probability estimates from all trees.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2) with columns [P(class=0), P(class=1)].
        """
        return self._clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return majority-vote hard class predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Integer label array of shape (n_samples,).
        """
        return self._clf.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Mean Gini impurity decrease per feature across all trees.

        Shape: (n_features,). Useful for identifying predictive Morgan bits
        or physicochemical descriptor importance.
        """
        return self._clf.feature_importances_

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "RandomForestClassifier":
        for k, v in params.items():
            setattr(self, k, v)
        self._clf.set_params(**{k: v for k, v in params.items() if hasattr(self._clf, k)})
        return self
