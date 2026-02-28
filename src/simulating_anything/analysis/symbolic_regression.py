"""Symbolic regression via PySR for discovering governing equations."""

from __future__ import annotations

import logging

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
    populations: int = 15,
    population_size: int = 33,
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
        populations: Number of independent populations for search.
        population_size: Size of each population.

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
        populations=populations,
        population_size=population_size,
        verbosity=0,
        progress=False,
    )

    model.fit(X, y, variable_names=variable_names)

    # Compute R2 from the best model
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    best_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    discoveries = []
    equations = model.equations_
    if equations is not None:
        for i, row in equations.iterrows():
            expr = str(row.get("equation", ""))
            complexity = int(row.get("complexity", 0))
            loss = float(row.get("loss", 1.0))

            # Compute per-equation R2: R2 = 1 - loss/variance(y)
            var_y = np.var(y) if np.var(y) > 0 else 1.0
            eq_r2 = max(0.0, 1.0 - loss / var_y)

            discovery = Discovery(
                id=f"sr_{i}",
                type=DiscoveryType.GOVERNING_EQUATION,
                confidence=min(eq_r2, 1.0),
                expression=expr,
                description=f"Symbolic regression (complexity={complexity}, loss={loss:.6g})",
                evidence=Evidence(
                    fit_r_squared=eq_r2,
                    n_supporting=len(y),
                ),
            )
            discoveries.append(discovery)

    # Sort by confidence descending
    discoveries.sort(key=lambda d: d.confidence, reverse=True)
    return discoveries
