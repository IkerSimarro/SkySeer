from pathlib import Path

from skyseer.config import Config, load_config

DEFAULT_YAML = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_default_yaml_matches_builtin_defaults():
    """configs/default.yaml is documentation; it must not drift from the code."""
    assert load_config(DEFAULT_YAML) == Config()


def test_raspberry_pi_profile_is_valid():
    """configs/pi.yaml (what the Pi service runs) loads, and only changes what it means to."""
    from skyseer.config import load_config

    pi = load_config("configs/pi.yaml")
    assert pi.station.camera == "picamera" and pi.gps.source == "none"  # budget kit: no GPS
    assert pi.aircraft.source == "opensky" and pi.clips.buffer_mb <= 512  # fits a 2 GB Pi
    assert (pi.camera.width, pi.camera.height, pi.camera.fps) == (2028, 1520, 15.0)
    assert pi.rules == load_config().rules  # same classification as everywhere else
