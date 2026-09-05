"""Frozen classifier families for timing-feature robustness analysis."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASSIFIER_NAMES = ("random_forest", "logistic_regression", "rbf_svm", "hist_gradient_boosting")


def build_classifier(name: str, random_state: int):
    """Construct one preregistered classifier without outcome-driven tuning."""
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        )
    if name == "logistic_regression":
        return Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                tol=1e-4,
                random_state=random_state,
            )),
        ])
    if name == "rbf_svm":
        return Pipeline([
            ("scale", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=False,
                random_state=random_state,
            )),
        ])
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"Unknown classifier family: {name}")


def positive_class_scores(model, values: np.ndarray) -> np.ndarray:
    """Return consistently oriented continuous scores for label 1."""
    if hasattr(model, "decision_function"):
        scores = model.decision_function(values)
    else:
        positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
        scores = model.predict_proba(values)[:, positive_index]
    result = np.asarray(scores, dtype=np.float64)
    if result.shape != (len(values),) or not np.all(np.isfinite(result)):
        raise ValueError("Classifier produced invalid positive-class scores.")
    return result
