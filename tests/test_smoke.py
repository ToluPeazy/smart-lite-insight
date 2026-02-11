"""Smoke tests — verify the dev environment is working."""


def test_imports():
    """Core dependencies are importable."""
    import fastapi
    import pandas
    import plotly
    import sklearn
    import streamlit

    assert pandas.__version__ >= "2.2"
    assert sklearn.__version__ >= "1.4"


def test_project_structure(project_root):
    """Key project directories exist."""
    expected_dirs = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src",
        "seed",
        "dashboard",
        "models",
        "tests",
        "docs/schemas",
    ]
    for d in expected_dirs:
        assert (project_root / d).is_dir(), f"Missing directory: {d}"


def test_schema_exists(project_root):
    """Telemetry schema v1 is present."""
    schema = project_root / "docs" / "schemas" / "telemetry_v1.json"
    assert schema.is_file(), "Telemetry schema v1 not found"
