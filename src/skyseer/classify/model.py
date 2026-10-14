"""Learned classifier: gradient-boosted trees with calibrated probabilities.

The hand-written rules (``rules.py``) stay as the transparent baseline; this model learns
from labelled tracks how to combine the same measurements, and the rule scores
themselves are among its inputs, so it can learn *corrections* to the rules rather than
starting from nothing.

Choices, and why:

- **Gradient-boosted decision trees** (scikit-learn's HistGradientBoostingClassifier):
  the standard strong model for tabular measurements, fast enough for a Raspberry Pi,
  and needing far less data than a neural network.
- **Scale-free inputs only**: speeds in image diagonals per second, brightness as signal
  to noise, rates in Hz. Anything in raw pixels or 0-255 levels would tie the model to
  one camera, resolution and gain.
- **Calibrated probabilities** (isotonic calibration): "0.8 Satellite" should be right
  about 80% of the time. The training report measures it (Brier score, expected
  calibration error).
- **Grouped validation**: tracks from one video are never split between training and
  testing, because they share a sky, noise and camera, which would flatter the score.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RulesConfig
from ..models import ObjectClass
from .rules import Classification, RuleClassifier

log = logging.getLogger(__name__)

CLASSES: tuple[str, ...] = ("Satellite", "Meteor", "Plane", "Junk")

# Measurements that mean the same at any resolution, frame rate and camera gain.
TRACK_FEATURES: tuple[str, ...] = (
    "duration",
    "detection_rate",
    "speed_frac_per_s",
    "speed_cv",
    "max_speed_ratio",
    "path_length_frac",
    "linearity",
    "direction_change_rate",
    "accel_rel",
    "size_consistency",
    "avg_aspect_ratio",
    "aspect_ratio_std",
    "brightness_consistency",
    "brightness_variation_coeff",
    "blinking_score",
    "snr_median",
    "snr_max",
    "snr_cv",
    "flash_rate_hz",
    "flash_regularity",
    "flash_score",
    "cloudiness",
    "path_deviation",
    "path_deviation_max",
)
RULE_FEATURES: tuple[str, ...] = tuple(f"rule_{c.lower()}" for c in CLASSES)
MODEL_FEATURES: tuple[str, ...] = TRACK_FEATURES + RULE_FEATURES


def rule_shares(features: dict[str, float], rules: RuleClassifier) -> dict[str, float]:
    """Each class's share of the rule scores (0..1): the rules' opinion, as features."""
    scores = rules.scores(features)
    total = sum(scores.values()) or 1.0
    return {f"rule_{str(k).lower()}": v / total for k, v in scores.items()}


def feature_vector(features: dict[str, float], rules: RuleClassifier) -> np.ndarray:
    row = {**features, **rule_shares(features, rules)}
    return np.array([float(row.get(name, np.nan)) for name in MODEL_FEATURES])


@dataclass
class TrainingReport:
    """Held-out results for the model and, on the same tracks, the rules."""

    n_train: int
    n_test: int
    classes: list[str]
    model: dict[str, Any]
    rules: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "classes": self.classes,
            "model": self.model,
            "rules": self.rules,
        }


def _scores(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    labels = list(CLASSES)
    present = [c for c in labels if c in set(y_true) | set(y_pred)]  # absent classes: no F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro = f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
    out: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(macro), 4),
        "per_class": {
            c: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            }
            for c, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if proba is not None:
        onehot = np.array([[t == c for c in labels] for t in y_true], dtype=float)
        out["brier"] = round(
            float(np.mean([brier_score_loss(onehot[:, k], proba[:, k]) for k in range(4)])), 4
        )
        out["ece"] = round(expected_calibration_error(y_true, proba), 4)
    return out


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    """How far stated confidence is from actual accuracy, averaged over confidence bins."""
    confidence = proba.max(axis=1)
    correct = np.array(CLASSES)[proba.argmax(axis=1)] == y_true
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (confidence > lo) & (confidence <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - confidence[sel].mean())
    return float(ece)


def _new_model(groups: np.ndarray, y: np.ndarray) -> Any:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    trees = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=0,
    )
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups))
    return CalibratedClassifierCV(trees, method="isotonic", cv=folds)


class LearnedClassifier:
    """Drop-in replacement for :class:`RuleClassifier` backed by a trained model."""

    def __init__(self, estimator: Any, rules: RuleClassifier, info: dict[str, Any]):
        self.estimator = estimator
        self.rules = rules
        self.info = info
        self._index = {c: i for i, c in enumerate(estimator.classes_)}

    @classmethod
    def train(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        rules_config: RulesConfig,
        info: dict[str, Any] | None = None,
    ) -> LearnedClassifier:
        estimator = _new_model(groups, y)
        estimator.fit(X, y)
        return cls(estimator, RuleClassifier(rules_config), info or {})

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        """(N, 4) calibrated probabilities in ``CLASSES`` order."""
        raw = self.estimator.predict_proba(X)
        out = np.zeros((len(X), len(CLASSES)))
        for k, c in enumerate(CLASSES):
            if c in self._index:
                out[:, k] = raw[:, self._index[c]]
        return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)

    def classify(self, features: dict[str, float]) -> Classification:
        proba = self.probabilities(feature_vector(features, self.rules)[None, :])[0]
        k = int(proba.argmax())
        scores = {c: float(p) for c, p in zip(CLASSES, proba, strict=True)}
        return Classification(ObjectClass(CLASSES[k]), float(proba[k]), scores)

    def evaluate(self, X: np.ndarray, y: np.ndarray, rule_labels: np.ndarray) -> dict[str, Any]:
        proba = self.probabilities(X)
        predicted = np.array(CLASSES)[proba.argmax(axis=1)]
        return {"model": _scores(y, predicted, proba), "rules": _scores(y, rule_labels, None)}

    # ------------------------------------------------------------------- storage
    def save(self, path: str | Path) -> None:
        """Pickle the model (our own file: never load pickles from untrusted sources)."""
        import joblib
        import sklearn

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, path)
        meta = {**self.info, "sklearn": sklearn.__version__, "features": list(MODEL_FEATURES)}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, rules_config: RulesConfig) -> LearnedClassifier:
        import joblib
        import sklearn

        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if meta.get("features") != list(MODEL_FEATURES):
            raise ValueError(f"{path} was trained on different features; retrain it")
        if meta.get("sklearn") != sklearn.__version__:
            raise ValueError(
                f"{path} was saved by scikit-learn {meta.get('sklearn')}, "
                f"this is {sklearn.__version__}; retrain it"
            )
        return cls(joblib.load(path), RuleClassifier(rules_config), meta)


def dataset_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
