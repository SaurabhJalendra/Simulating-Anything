"""Symbolic regression via PySR for discovering governing equations."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from simulating_anything.types.discovery import Discovery, DiscoveryType, Evidence

logger = logging.getLogger(__name__)

try:
    from pysr import PySRRegressor

    _HAS_PYSR = True
except ImportError:
    _HAS_PYSR = False


def run_symbolic_regression(
    X: np.ndarray,
    y: np.ndarray,
    variable_names: list[str] | None = None,
    n_iterations: int = 40,
    binary_operators: list[str] | None = None,
    unary_operators: list[str] | None = None,
    max_complexity: int = 20,
) -> list[Discovery]:
    """Run PySR symbolic regression to find equations relating X to y.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        variable_names: Names for each feature column.
        n_iterations: Number of PySR evolution iterations.
        binary_operators: Allowed binary ops (default: +, -, *, /).
        unary_operators: Allowed unary ops (default: sin, cos, exp, log, sqrt).
        max_complexity: Maximum expression complexity.

    Returns:
        List of Discovery objects ordered by fit quality.
    """
    if not _HAS_PYSR:
        logger.warning("PySR not installed — returning empty results")
        return []

    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/"]
    if unary_operators is None:
        unary_operators = ["sin", "cos", "exp", "log", "sqrt", "square"]

    model = PySRRegressor(
        niterations=n_iterations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        maxsize=max_complexity,
        variable_names=variable_names,
        verbosity=0,
    )

    model.fit(X, y)

    discoveries = []
    equations = model.equations_
    if equations is not None:
        for i, row in equations.iterrows():
            r2 = float(row.get("score", 0.0))
            expr = str(row.get("equation", ""))
            complexity = int(row.get("complexity", 0))

            discovery = Discovery(
                id=f"sr_{i}",
                type=DiscoveryType.GOVERNING_EQUATION,
                confidence=min(r2, 1.0),
                expression=expr,
                description=f"Symbolic regression (complexity={complexity})",
                evidence=Evidence(
                    fit_r_squared=r2,
                    n_supporting=len(y),
                ),
            )
            discoveries.append(discovery)

    # Sort by confidence descending
    discoveries.sort(key=lambda d: d.confidence, reverse=True)
    return discoveries
