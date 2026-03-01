"""Tests for CLI entry point.

Verifies that all CLI commands work correctly.
"""
from __future__ import annotations

import subprocess
import sys

import pytest


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run the CLI with the given arguments."""
    return subprocess.run(
        [sys.executable, "-m", "simulating_anything", *args],
        capture_output=True, text=True, timeout=120,
    )


class TestCLI:
    """Test CLI commands."""

    def test_version(self):
        result = _run_cli("version")
        assert result.returncode == 0
        assert "simulating-anything" in result.stdout
        assert "0.2.0" in result.stdout

    def test_help(self):
        result = _run_cli("help")
        assert result.returncode == 0
        assert "demo" in result.stdout
        assert "dashboard" in result.stdout
        assert "ablation" in result.stdout
        assert "sensitivity" in result.stdout
        assert "aggregate" in result.stdout

    def test_no_args(self):
        result = _run_cli()
        assert result.returncode == 0
        assert "Usage:" in result.stdout

    def test_unknown_command(self):
        result = _run_cli("nonexistent")
        assert result.returncode == 1
        assert "Unknown command" in result.stdout

    def test_version_flag(self):
        result = _run_cli("--version")
        assert result.returncode == 0
        assert "0.2.0" in result.stdout

    def test_help_flag(self):
        result = _run_cli("-h")
        assert result.returncode == 0
        assert "Usage:" in result.stdout
