"""Shared test fixtures for Simulating Anything."""

import pytest


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    out = tmp_path / "output"
    out.mkdir()
    return out
