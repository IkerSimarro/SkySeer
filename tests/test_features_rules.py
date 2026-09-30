import pytest

from skyseer.classify import RuleClassifier
from skyseer.config import RulesConfig, StarFilterConfig
from skyseer.features import FEATURE_NAMES, extract_features
from skyseer.models import ObjectClass
from skyseer.starfield import StarFieldFilter

from .conftest import linear_points, make_track


def test_straight_constant_track_features():
    f = extract_features(make_track(linear_points(0, 31, 10, 10, 3, 4)))
    assert set(f) == set(FEATURE_NAMES)
    assert f["avg_speed"] == pytest.approx(5.0)
    assert f["speed_consistency"] == pytest.approx(1.0)
    assert f["linearity"] == pytest.approx(1.0)
    assert f["direction_changes"] == 0
    assert f["duration"] == pytest.approx(1.0)


def test_speed_is_per_source_frame_regardless_of_skip():
    skipped = extract_features(make_track(linear_points(0, 30, 0, 0, 2, 0, step=3)))
    assert skipped["avg_speed"] == pytest.approx(2.0)


def test_zigzag_counts_direction_changes():
    pts = [(i, 10.0 * i, 0.0 if i % 2 else 10.0) for i in range(10)]
    f = extract_features(make_track(pts))
    assert f["direction_changes"] >= 7
    assert f["linearity"] < 0.9


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        (linear_points(0, 15, 10, 10, 25, 0), ObjectClass.METEOR),  # 0.5 s, 25 px/frame
        (linear_points(0, 300, 10, 10, 1.5, 0.5), ObjectClass.SATELLITE),  # 10 s, steady
        ([(i, 100.0, 100.0 + (i % 2) * 0.2) for i in range(60)], ObjectClass.JUNK),  # stationary
    ],
)
def test_rule_classifier(points, expected):
    result = RuleClassifier(RulesConfig()).classify(extract_features(make_track(points)))
    assert result.label is expected
    assert 0 < result.confidence <= 1


def test_starfield_flags_comoving_group_only():
    stars = [
        make_track(linear_points(0, 90, 100 + 40 * k, 200, 0.3, 0.1), track_id=k) for k in range(4)
    ]
    loner = make_track(linear_points(0, 90, 300, 300, -1.5, 0.8), track_id=99)
    filt = StarFieldFilter(StarFilterConfig())
    assert filt.star_ids(stars + [loner]) == {0, 1, 2, 3}
    assert filt.is_star(stars[0], stars + [loner])
    assert not filt.is_star(loner, stars + [loner])
