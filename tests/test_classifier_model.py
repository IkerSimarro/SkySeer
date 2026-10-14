"""The learned classifier: training, calibration measure, storage, fallback, scoring."""

import numpy as np
import pytest

from skyseer.classify import RuleClassifier, make_classifier
from skyseer.classify.dataset import object_scores, synthetic_rows
from skyseer.classify.model import (
    CLASSES,
    MODEL_FEATURES,
    LearnedClassifier,
    expected_calibration_error,
)
from skyseer.config import load_config
from skyseer.evaluate import evaluate
from skyseer.pipeline import run
from skyseer.sources import FileSource


@pytest.fixture(scope="module")
def toy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows whose classes differ in a few features, spread over 10 'videos'."""
    rng = np.random.default_rng(0)
    centre = {"Satellite": 0.0, "Meteor": 3.0, "Plane": 6.0, "Junk": 9.0}
    X, y, groups = [], [], []
    for video in range(10):
        for label in CLASSES:
            for _ in range(15):
                row = rng.normal(0, 1, len(MODEL_FEATURES))
                row[:3] += centre[label]
                X.append(row)
                y.append(label)
                groups.append(f"v{video}")
    return np.array(X), np.array(y), np.array(groups)


@pytest.fixture(scope="module")
def trained(toy) -> LearnedClassifier:
    X, y, groups = toy
    return LearnedClassifier.train(X, y, groups, load_config().rules, {"note": "toy"})


def test_model_learns_and_gives_probabilities(trained, toy):
    X, y, _ = toy
    proba = trained.probabilities(X)
    assert proba.shape == (len(X), 4)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (np.array(CLASSES)[proba.argmax(axis=1)] == y).mean() > 0.95


def test_expected_calibration_error():
    rng = np.random.default_rng(1)
    confidence = rng.uniform(0.3, 1.0, 20000)
    labels = np.array(CLASSES)
    proba = np.zeros((20000, 4))
    proba[:, 0] = confidence
    proba[:, 1:] = ((1 - confidence) / 3)[:, None]
    # Right exactly as often as claimed: calibrated.
    truth = np.where(rng.random(20000) < confidence, labels[0], labels[1])
    assert expected_calibration_error(truth, proba) < 0.02
    # Claims 30-100% but is never right: badly calibrated.
    assert expected_calibration_error(np.full(20000, labels[1]), proba) > 0.6


def test_save_and_load_round_trip(trained, toy, tmp_path):
    X, _, _ = toy
    path = tmp_path / "model.joblib"
    trained.save(path)
    loaded = LearnedClassifier.load(path, load_config().rules)
    assert np.allclose(loaded.probabilities(X[:20]), trained.probabilities(X[:20]))
    assert loaded.info["note"] == "toy"


def test_a_model_for_other_features_is_refused(trained, tmp_path):
    import json

    path = tmp_path / "model.joblib"
    trained.save(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    meta["features"] = meta["features"][:-1]
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="different features"):
        LearnedClassifier.load(path, load_config().rules)


def test_missing_model_falls_back_to_the_rules(tmp_path):
    cfg = load_config(classifier={"method": "model", "model_path": str(tmp_path / "none")})
    assert isinstance(make_classifier(cfg), RuleClassifier)


def test_pipeline_uses_the_model_when_configured(trained, synth_clip, tmp_path):
    path = tmp_path / "model.joblib"
    trained.save(path)
    cfg = load_config(classifier={"method": "model", "model_path": str(path)})
    events = list(run(FileSource(synth_clip[0]), cfg))
    classified = [e for e in events if e.label_source != "starfield"]
    assert classified and all(e.label_source == "model" for e in classified)
    assert all(set(e.scores) == set(CLASSES) for e in classified)


def test_object_scores_from_rows_match_the_benchmark(synth_clip):
    """Scoring a classifier from stored rows gives the same numbers as the benchmark."""
    path, truth = synth_clip
    cfg = load_config()
    expected = evaluate(list(run(FileSource(path), cfg)), truth)
    rows = synthetic_rows(path, path.with_suffix(".json"), cfg, "clip")
    scores = object_scores(rows, np.array([r["rule_label"] for r in rows]))
    assert scores["objects"] == expected.n_truth
    assert scores["recall"] == round(expected.recall, 4)
    assert scores["accuracy"] == round(expected.accuracy, 4)
    assert scores["false_positives"] == expected.false_positives
