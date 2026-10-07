import pydantic
import pytest

from skyseer.config import load_config


def test_defaults():
    cfg = load_config()
    assert cfg.detector.sensitivity == 5
    assert cfg.detector.min_area == 2  # tiny on purpose: faint satellites can be 1-2 px
    assert load_config(detector={"min_area_px": 7}).detector.min_area == 7


def test_yaml_and_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("detector:\n  sensitivity: 8\ntracker:\n  max_distance: 40\n")
    cfg = load_config(path, pipeline={"frame_skip": 2})
    assert cfg.detector.sensitivity == 8
    assert cfg.detector.mog2_history == 500  # untouched defaults survive
    assert cfg.tracker.max_distance == 40
    assert cfg.pipeline.frame_skip == 2


def test_invalid_values_rejected(tmp_path):
    with pytest.raises(pydantic.ValidationError):
        load_config(detector={"sensitivity": 42})
    with pytest.raises(pydantic.ValidationError):
        load_config(detector={"typo_key": 1})


def test_min_detections_never_below_two():
    """Regression: the old app could compute a minimum of 0 detections."""
    with pytest.raises(pydantic.ValidationError):
        load_config(filter={"min_detections": 0})
