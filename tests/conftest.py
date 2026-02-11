"""Shared test fixtures for Smart-Lite Insight."""

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_db():
    """Create a temporary SQLite database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def sample_raw_row():
    """A single raw row from the UCI dataset (semicolon-separated)."""
    return "16/12/2006;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000"


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent
