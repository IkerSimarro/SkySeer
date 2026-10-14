"""Track classifiers: the hand-written rules (baseline) and the learned model."""

from __future__ import annotations

import logging
from typing import Protocol

from ..config import Config
from .rules import Classification, RuleClassifier

log = logging.getLogger(__name__)

__all__ = ["Classification", "Classifier", "RuleClassifier", "make_classifier"]


class Classifier(Protocol):
    def classify(self, features: dict[str, float]) -> Classification: ...


def make_classifier(config: Config) -> Classifier:
    """The classifier the config asks for. A missing or out-of-date model falls back to
    the rules with a warning, so analysis never stops for want of a model file."""
    if config.classifier.method == "model":
        from .model import LearnedClassifier

        try:
            return LearnedClassifier.load(config.classifier.model_path, config.rules)
        except (OSError, ValueError) as exc:
            log.warning("Learned classifier unavailable (%s); using the rules", exc)
    return RuleClassifier(config.rules)
