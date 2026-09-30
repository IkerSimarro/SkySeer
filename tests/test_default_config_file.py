from pathlib import Path

from skyseer.config import Config, load_config

DEFAULT_YAML = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_default_yaml_matches_builtin_defaults():
    """configs/default.yaml is documentation; it must not drift from the code."""
    assert load_config(DEFAULT_YAML) == Config()
