"""Magnetic pendulum rediscovery.

Targets:
- Fractal basin boundaries (boundary fraction as complexity measure)
- 3-fold rotational symmetry of basin structure
- Sensitivity: nearby initial conditions reach different magnets
- Damping sweep: more damping = smoother boundaries
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.magnetic_pendulum import MagneticPendulumSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(**overrides: float) -> SimulationConfig:
    """Create a default magnetic pendulum config with optional overrides."""
    params = {
        "gamma": 0.1,
        "omega0_sq": 0.5,
        "alpha": 1.0,
        "R": 1.0,
        "d": 0.3,
        "n_magnets": 3.0,
        "x_0": 0.5,
        "y_0": 0.5,
        "vx_0": 0.0,
        "vy_0": 0.0,
    }
    params.update(overrides)
    return SimulationConfig(
        domain=Domain.MAGNETIC_PENDULUM,
        dt=0.01,
        n_steps=20000,
        parameters=params,
    )


def generate_basin_map(grid_size: int = 50) -> dict:
    """Generate basin-of-attraction map and measure boundary complexity."""
    config = _make_config()
    sim = MagneticPendulumSimulation(config)
    sim.reset()

    logger.info(f"  Computing basin map ({grid_size}x{grid_size})...")
    basin = sim.basin_map(grid_size=grid_size, t_max=200.0)

    # Count unique attractors
    unique_attractors = np.unique(basin)
    attractor_counts = {int(a): int(np.sum(basin == a)) for a in unique_attractors}

    # Compute boundary fraction
    boundary_count = 0
    interior_count = 0
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            interior_count += 1
            val = basin[i, j]
            neighbors = [
                basin[i - 1, j], basin[i + 1, j],
                basin[i, j - 1], basin[i, j + 1],
            ]
            if any(n != val for n in neighbors):
                boundary_count += 1

    boundary_fraction = boundary_count / interior_count if interior_count > 0 else 0.0

    return {
        "grid_size": grid_size,
        "n_attractors_found": len(unique_attractors),
        "attractor_counts": attractor_counts,
        "boundary_fraction": boundary_fraction,
        "basin": basin,
    }


def test_symmetry(basin: np.ndarray) -> dict:
    """Test approximate 3-fold rotational symmetry of basin map.

    Rotate the grid by 120 degrees and compare. Since the magnets have
    3-fold symmetry, the basin map should too (modulo attractor relabeling).
    """
    n = basin.shape[0]
    center = (n - 1) / 2.0

    # For each cell, find where it maps under 120-degree rotation
    # and check if it maps to the "correct" attractor (shifted by 1)
    match_count = 0
    total = 0

    cos120 = np.cos(2 * np.pi / 3)
    sin120 = np.sin(2 * np.pi / 3)

    for i in range(n):
        for j in range(n):
            # Map grid coords to centered coords
            x = j - center
            y = i - center

            # Rotate by 120 degrees
            xr = cos120 * x - sin120 * y
            yr = sin120 * x + cos120 * y

            # Map back to grid
            jr = int(round(xr + center))
            ir = int(round(yr + center))

            if 0 <= ir < n and 0 <= jr < n:
                total += 1
                # Under 120-degree rotation, attractor k should map to (k+1) mod 3
                original = basin[i, j]
                rotated = basin[ir, jr]
                if (original + 1) % 3 == rotated:
                    match_count += 1

    symmetry_score = match_count / total if total > 0 else 0.0
    return {
        "symmetry_score": symmetry_score,
        "total_compared": total,
        "matched": match_count,
    }


def test_sensitivity(n_pairs: int = 50, eps: float = 1e-4) -> dict:
    """Test sensitivity: nearby ICs that reach different magnets."""
    rng = np.random.default_rng(42)
    different_count = 0

    for _ in range(n_pairs):
        x0 = rng.uniform(-1.0, 1.0)
        y0 = rng.uniform(-1.0, 1.0)

        # First IC
        config1 = _make_config(x_0=x0, y_0=y0)
        sim1 = MagneticPendulumSimulation(config1)
        sim1.reset()
        m1 = sim1.find_attractor()

        # Slightly perturbed IC
        config2 = _make_config(x_0=x0 + eps, y_0=y0)
        sim2 = MagneticPendulumSimulation(config2)
        sim2.reset()
        m2 = sim2.find_attractor()

        if m1 != m2:
            different_count += 1

    return {
        "n_pairs": n_pairs,
        "eps": eps,
        "different_count": different_count,
        "sensitivity_fraction": different_count / n_pairs,
    }


def sweep_damping(
    gamma_values: np.ndarray | None = None,
    grid_size: int = 30,
) -> dict:
    """Sweep damping gamma and measure how boundary complexity changes.

    Higher damping should produce smoother (less fractal) basin boundaries.
    """
    if gamma_values is None:
        gamma_values = np.array([0.05, 0.1, 0.2, 0.5, 1.0])

    boundary_fractions = []
    for gamma in gamma_values:
        config = _make_config(gamma=gamma)
        sim = MagneticPendulumSimulation(config)
        sim.reset()
        basin = sim.basin_map(grid_size=grid_size, t_max=200.0)

        # Compute boundary fraction
        boundary_count = 0
        interior_count = 0
        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                interior_count += 1
                val = basin[i, j]
                neighbors = [
                    basin[i - 1, j], basin[i + 1, j],
                    basin[i, j - 1], basin[i, j + 1],
                ]
                if any(n != val for n in neighbors):
                    boundary_count += 1

        bf = boundary_count / interior_count if interior_count > 0 else 0.0
        boundary_fractions.append(bf)
        logger.info(f"  gamma={gamma:.3f}: boundary_fraction={bf:.4f}")

    return {
        "gamma_values": gamma_values.tolist(),
        "boundary_fractions": boundary_fractions,
    }


def run_magnetic_pendulum_rediscovery(
    output_dir: str | Path = "output/rediscovery/magnetic_pendulum",
    n_iterations: int = 40,
) -> dict:
    """Run magnetic pendulum rediscovery analysis.

    Args:
        output_dir: Directory for output files.
        n_iterations: PySR iterations (unused -- no symbolic regression target).

    Returns:
        Results dict with basin map, symmetry, sensitivity, and damping sweep.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "magnetic_pendulum",
        "targets": {
            "fractal_basins": "Fractal basin boundaries with 3-fold symmetry",
            "sensitivity": "Nearby ICs reach different magnets",
            "damping_smoothing": "More damping => smoother boundaries",
        },
    }

    # Part 1: Basin map
    logger.info("Part 1: Basin-of-attraction map...")
    basin_data = generate_basin_map(grid_size=50)
    basin = basin_data.pop("basin")
    results["basin_map"] = basin_data
    logger.info(
        f"  Found {basin_data['n_attractors_found']} attractors, "
        f"boundary fraction = {basin_data['boundary_fraction']:.4f}"
    )

    # Part 2: 3-fold symmetry test
    logger.info("Part 2: Symmetry test...")
    sym = test_symmetry(basin)
    results["symmetry"] = sym
    logger.info(f"  Symmetry score: {sym['symmetry_score']:.4f}")

    # Part 3: Sensitivity test
    logger.info("Part 3: Sensitivity test...")
    sens = test_sensitivity(n_pairs=100, eps=1e-4)
    results["sensitivity"] = sens
    logger.info(
        f"  {sens['different_count']}/{sens['n_pairs']} pairs diverged "
        f"(eps={sens['eps']})"
    )

    # Part 4: Damping sweep
    logger.info("Part 4: Damping sweep...")
    damping = sweep_damping(grid_size=30)
    results["damping_sweep"] = damping

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save basin map as npz
    np.savez(output_path / "basin_map.npz", basin=basin)

    return results
