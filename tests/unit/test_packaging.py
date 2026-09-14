"""Validates pyproject.toml metadata reflects the corrected Phase 0 architecture decisions:
the nflreadpy (not nfl_data_py) loader choice and the conservative Python version policy.
See docs/ARCHITECTURE.md#python-version-policy and docs/DATA_SOURCES.md.
"""

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_pyproject() -> dict:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def test_pyproject_is_valid_toml_with_expected_project_name():
    config = _load_pyproject()
    assert config["project"]["name"] == "nfl-predict"


def test_python_version_policy_is_conservative_and_excludes_314():
    config = _load_pyproject()
    requires_python = config["project"]["requires-python"]
    assert requires_python == ">=3.11,<3.14"


def test_data_extra_uses_nflreadpy_not_nfl_data_py():
    config = _load_pyproject()
    data_deps = config["project"]["optional-dependencies"]["data"]
    joined = " ".join(data_deps).lower()

    assert any(dep.lower().startswith("nflreadpy") for dep in data_deps)
    assert "nfl_data_py" not in joined
    assert "nfl-data-py" not in joined


def test_data_extra_declares_polars_explicitly():
    config = _load_pyproject()
    data_deps = config["project"]["optional-dependencies"]["data"]
    assert any(dep.lower().startswith("polars") for dep in data_deps)
