"""Logistic Regression baseline classifier for Tox21 multi-task toxicity prediction.

Student 3 Module — Classical ML Models and Experiments.

Design choices
--------------
* L2 regularisation (C=1.0) prevents unbounded weight growth in high-dimensional
  Morgan fingerprint spaces (d=2048), where many bits are zero for any molecule.
* class_weight='balanced' automatically reweights training samples proportionally
  to inverse class frequency, addressing positive-class prevalence as low as 2.84%.
* solver='lbfgs' is the recommended multi-class / binary solver for dense or
  moderately sparse feature matrices. saga is available as an alternative for very
  large datasets where stochastic gradient steps are preferred.
* max_iter=1000 is sufficient given normalised descriptors and binary Morgan bits.

Assumptions and limitations
-----------------------------
* Linearity: LR assumes a log-linear decision boundary in feature space. It cannot
  model non-linear interactions between Morgan sub-structural bits or descriptor pairs.
* Independence: Bits are treated as orthogonal features; sub-structural hierarchy
  (e.g., a ring containing a carbonyl) is invisible to LR.
* Serves as the minimal linear baseline: any non-linear model that fails to beat
  this baseline on scaffold-split test data is not adding value.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):
    """L2-regularised Logistic Regression with balanced class weighting.

    Wraps sklearn.linear_model.LogisticRegression with project-standard
    hyperparameters and a consistent scikit-learn compatible interface.

    Parameters
    ----------
    C : float
        Inverse regularisation strength. Smaller values apply stronger L2 penalty.
    max_iter : int
        Maximum number of solver iterations.
    random_state : int
        Seed for reproducibility of solver initialisation.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self._clf = LogisticRegression(
            penalty="l2",
            C=self.C,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        """Fit the logistic regression model.

        Args:
            X: Feature matrix of shape (n_samples, n_features). Must contain
               only labelled samples (NaN rows must be filtered upstream).
            y: Binary label vector of shape (n_samples,). Values must be 0 or 1.

        Returns:
            self
        """
        self._clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2) with columns [P(class=0), P(class=1)].
        """
        return self._clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions using the default 0.5 threshold.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Integer label array of shape (n_samples,).
        """
        return self._clf.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        """Coefficient matrix of shape (1, n_features) for binary classification."""
        return self._clf.coef_

    @property
    def intercept_(self) -> np.ndarray:
        """Intercept (bias) term of shape (1,)."""
        return self._clf.intercept_

    def get_params(self, deep: bool = True) -> dict:
        return {"C": self.C, "max_iter": self.max_iter, "random_state": self.random_state}

    def set_params(self, **params) -> "LogisticRegressionClassifier":
        for k, v in params.items():
            setattr(self, k, v)
        self._clf.set_params(**{k: v for k, v in params.items() if hasattr(self._clf, k)})
        return self
