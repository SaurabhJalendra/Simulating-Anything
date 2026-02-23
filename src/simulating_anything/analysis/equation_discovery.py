"""SINDy-based equation discovery for dynamical systems."""

from __future__ import annotations

import logging

import numpy as np

from simulating_anything.types.discovery import Discovery, DiscoveryType, Evidence

logger = logging.getLogger(__name__)

try:
    import pysindy as ps

    _HAS_SINDY = True
except ImportError:
    _HAS_SINDY = False


def run_sindy(
    states: np.ndarray,
    dt: float,
    feature_names: list[str] | None = None,
    threshold: float = 0.1,
    max_iter: int = 20,
    poly_degree: int = 3,
) -> list[Discovery]:
    """Run SINDy to discover governing ODEs from time-series data.

    Args:
        states: State array (n_timesteps, n_variables).
        dt: Timestep between observations.
        feature_names: Names for state variables.
        threshold: Sparsity threshold for STLSQ optimizer.
        max_iter: Max optimizer iterations.
        poly_degree: Maximum polynomial degree in library.

    Returns:
        List of Discovery objects for each discovered equation.
    """
    if not _HAS_SINDY:
        logger.warning("PySINDy not installed — returning empty results")
        return []

    if states.ndim == 1:
        states = states.reshape(-1, 1)

    n_vars = states.shape[1]
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_vars)]

    optimizer = ps.STLSQ(threshold=threshold, max_iter=max_iter)
    library = ps.PolynomialLibrary(degree=poly_degree)

    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=library,
        feature_names=feature_names,
    )
    model.fit(states, t=dt)

    # Score the model
    score = model.score(states, t=dt)

    # Extract equations
    equations = model.equations()
    discoveries = []
    for i, eq in enumerate(equations):
        var_name = feature_names[i] if i < len(feature_names) else f"x{i}"
        discoveries.append(
            Discovery(
                id=f"sindy_{i}",
                type=DiscoveryType.GOVERNING_EQUATION,
                confidence=max(0.0, min(float(score), 1.0)),
                expression=f"d({var_name})/dt = {eq}",
                description=f"SINDy-discovered ODE for {var_name}",
                evidence=Evidence(
                    fit_r_squared=float(score),
                    n_supporting=states.shape[0],
                ),
            )
        )

    return discoveries
