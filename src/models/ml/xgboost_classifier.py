"""XGBoost gradient boosting classifier for Tox21 multi-task toxicity prediction.

Student 3 Module — Classical ML Models and Experiments.

Design choices
--------------
* n_estimators=100: Boosting rounds producing an additive tree ensemble.
  Unlike RF (parallel bagging), XGBoost builds each tree to correct the residual
  errors of the current ensemble, focusing on hard-to-classify boundary samples.
* max_depth=5: Shallower than RF (12) because boosting builds many successive
  weak learners; deep trees in boosting cause rapid overfitting. max_depth=5
  is the commonly used default in QSAR literature (Chen & Guestrin, 2016).
* learning_rate=0.1: Shrinks each tree's contribution by 0.1 (step-size
  regularisation). Smaller values require more trees but generalise better.
* scale_pos_weight: Controls the balance of positive and negative weights.
  XGBoost lacks native class_weight='balanced'; instead the recommended approach
  is: scale_pos_weight = count(negative samples) / count(positive samples).
  This is computed per-endpoint from the training partition labels and passed
  at instantiation.
* eval_metric='logloss': Binary cross-entropy loss used during boosting.

Gradient boosting formulation
------------------------------
At round t, XGBoost minimises the regularised objective with a second-order
Taylor approximation:
    L^(t) = sum_i [ g_i * f_t(x_i) + 0.5 * h_i * f_t^2(x_i) ] + gamma*T + 0.5*lambda*||w||^2

where g_i = d_y L(y_i, y_hat_i), h_i = d_yy L(y_i, y_hat_i) are first- and
second-order gradients, T is the number of leaves, and w are leaf weights.

Compared to Random Forest
--------------------------
XGBoost is adaptive: it focuses boosting rounds on samples the ensemble currently
predicts incorrectly (boundary molecules). RF is non-adaptive and purely variance-
reducing via decorrelated bagging. XGBoost typically performs comparably to or
slightly below RF on scaffold-split Tox21 data where the primary challenge is
generalisation to novel scaffolds rather than boundary memorisation.
"""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier as _XGB
from sklearn.base import BaseEstimator, ClassifierMixin


class XGBoostClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost gradient boosted trees classifier with per-endpoint imbalance handling.

    Parameters
    ----------
    pos_neg_ratio : float
        Ratio of negative to positive training samples for scale_pos_weight.
        Should be computed from the training partition labels per endpoint.
        Default 1.0 (balanced) is safe but suboptimal; pass the real ratio.
    n_estimators : int
        Number of boosting rounds (trees).
    max_depth : int
        Maximum tree depth per boosting round.
    learning_rate : float
        Shrinkage factor applied to each tree's contribution.
    random_state : int
        Seed for deterministic column and row subsampling.
    """

    def __init__(
        self,
        pos_neg_ratio: float = 1.0,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        self.pos_neg_ratio = pos_neg_ratio
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._build_clf()

    def _build_clf(self) -> None:
        scale_pos = max(1.0, float(self.pos_neg_ratio))
        self._clf = _XGB(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifier":
        """Fit the XGBoost ensemble via sequential gradient boosting.

        Args:
            X: Feature matrix of shape (n_samples, n_features). XGBoost handles
               both binary (Morgan) and continuous (descriptor) features natively.
            y: Binary label vector of shape (n_samples,). Must be 0 or 1.

        Returns:
            self
        """
        self._clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return sigmoid-transformed ensemble probability estimates.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2) with columns [P(class=0), P(class=1)].
        """
        return self._clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions at the default 0.5 decision threshold.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Integer label array of shape (n_samples,).
        """
        return self._clf.predict(X)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "pos_neg_ratio": self.pos_neg_ratio,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "XGBoostClassifier":
        for k, v in params.items():
            setattr(self, k, v)
        self._build_clf()
        return self
