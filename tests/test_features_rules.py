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


def _noise_chain() -> list[tuple[int, float, float]]:
    """What the tracker builds from noise on a compressed video: 60 specks scattered over
    30 s that drift on average, each landing ~4 px off the line (the old rules called
    tracks like this satellites)."""
    import numpy as np

    rng = np.random.default_rng(3)
    frames = np.sort(rng.choice(range(900), 60, replace=False))
    return [
        (int(f), 100 + 0.4 * f + rng.normal(0, 4), 200 + 0.15 * f + rng.normal(0, 4))
        for f in frames
    ]


def _two_specks() -> list[tuple[int, float, float]]:
    """Like a fake found in real video: ~6 s at one spot, a slow 180 px jump across 81
    frames, ~4 s at another (twinkling stars or noise joined up)."""
    import numpy as np

    rng = np.random.default_rng(2)
    a = sorted(rng.choice(range(190), 33, replace=False))
    b = sorted(rng.choice(range(271, 400), 34, replace=False))
    return [(int(f), 590 + rng.normal(0, 0.4), 921 + rng.normal(0, 0.4)) for f in a] + [
        (int(f), 772 + rng.normal(0, 0.4), 935 + rng.normal(0, 0.4)) for f in b
    ]


def test_smooth_motion_and_sudden_jumps_are_measured():
    steady = extract_features(make_track(linear_points(0, 60, 10, 10, 1.5, 0.5)))
    assert steady["motion_residual"] < 0.01 and steady["sudden_jump"] == 0
    assert extract_features(make_track(_noise_chain()))["motion_residual"] > 4
    assert extract_features(make_track(_two_specks()))["sudden_jump"] > 0.7
    # A faint satellite that fades for 2 s and reappears further on moves at its pace.
    gap = [p for p in linear_points(0, 300, 10, 10, 1.5, 0.5) if not 100 <= p[0] < 160]
    assert extract_features(make_track(gap))["sudden_jump"] == 0


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        (_noise_chain(), ObjectClass.JUNK),
        (_two_specks(), ObjectClass.JUNK),
        # A faint satellite seen in only 1 frame in 8, but 60 times over a long pass.
        (linear_points(0, 480, 10, 10, 1.5, 0.5, step=8), ObjectClass.SATELLITE),
    ],
)
def test_noise_linked_into_tracks_is_not_reported(points, expected):
    """Both noise tracks were Satellites before the smooth-motion and jump tests."""
    f = extract_features(make_track(points))
    assert RuleClassifier(RulesConfig()).classify(f).label is expected
    off = RulesConfig(smooth_px=1e9, plane_smooth_px=1e9, jump_share=1.0, evidence_count=1e-9)
    assert RuleClassifier(off).classify(f).label is ObjectClass.SATELLITE
