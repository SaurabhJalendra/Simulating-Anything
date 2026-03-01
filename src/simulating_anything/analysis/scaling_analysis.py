"""Scaling analysis: how pipeline performance varies with domain complexity.

Measures:
- Simulation runtime vs state dimension
- Rediscovery quality vs data quantity
- Cross-domain analogy density vs number of domains
"""
from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass

import numpy as np

from simulating_anything.analysis.domain_statistics import DOMAIN_REGISTRY
from simulating_anything.types.simulation import SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class ScalingResult:
    """Result of a scaling experiment."""

    name: str
    independent_var: str
    independent_values: list[float]
    dependent_var: str
    dependent_values: list[float]
    fit_type: str  # "linear", "log-linear", "power-law"
    fit_params: dict[str, float]


def measure_simulation_scaling(
    n_steps_list: list[int] | None = None,
) -> ScalingResult:
    """Measure how simulation runtime scales with trajectory length.

    Uses the harmonic oscillator as a representative ODE domain.
    """
    if n_steps_list is None:
        n_steps_list = [100, 500, 1000, 2500, 5000, 10000]

    spec = DOMAIN_REGISTRY["harmonic_oscillator"]
    mod = importlib.import_module(spec["module"])
    cls = getattr(mod, spec["cls"])

    times: list[float] = []
    for n in n_steps_list:
        config = SimulationConfig(
            domain=spec["domain"], dt=spec["dt"], n_steps=n,
            parameters=spec["params"],
        )
        sim = cls(config)
        sim.reset()

        # Warm up
        sim.run(n_steps=10)
        sim.reset()

        # Measure
        t0 = time.perf_counter()
        sim.run(n_steps=n)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        times.append(elapsed)

    # Linear fit: time = a * n_steps + b
    coeffs = np.polyfit(n_steps_list, times, 1)

    return ScalingResult(
        name="simulation_runtime_vs_steps",
        independent_var="n_steps",
        independent_values=[float(n) for n in n_steps_list],
        dependent_var="runtime_ms",
        dependent_values=times,
        fit_type="linear",
        fit_params={"slope_ms_per_step": float(coeffs[0]),
                    "intercept_ms": float(coeffs[1])},
    )


def measure_dimension_scaling() -> ScalingResult:
    """Measure simulation runtime across domains with different state dimensions."""
    dims: list[float] = []
    times: list[float] = []

    for name, spec in sorted(DOMAIN_REGISTRY.items()):
        mod = importlib.import_module(spec["module"])
        cls = getattr(mod, spec["cls"])
        config = SimulationConfig(
            domain=spec["domain"], dt=spec["dt"], n_steps=spec["n_steps"],
            parameters=spec["params"],
        )
        sim = cls(config)
        sim.reset()

        t0 = time.perf_counter()
        traj = sim.run(n_steps=spec["n_steps"])
        elapsed = (time.perf_counter() - t0) * 1000

        obs = traj.states
        obs_dim = int(np.prod(obs.shape[1:])) if obs.ndim > 1 else 1
        dims.append(float(obs_dim))
        times.append(elapsed)

        logger.info(f"  {name:25s}: dim={obs_dim:6d}, time={elapsed:.1f}ms")

    # Log-linear fit
    log_dims = np.log10(np.array(dims) + 1)
    coeffs = np.polyfit(log_dims, times, 1)

    return ScalingResult(
        name="runtime_vs_dimension",
        independent_var="obs_dim",
        independent_values=dims,
        dependent_var="runtime_ms",
        dependent_values=times,
        fit_type="log-linear",
        fit_params={"slope": float(coeffs[0]), "intercept": float(coeffs[1])},
    )


def measure_data_quantity_scaling() -> ScalingResult:
    """Measure how projectile R² degrades with fewer data points."""

    # Generate dataset with 5% observation noise
    rng = np.random.default_rng(42)
    n_total = 500
    g = 9.81
    v0s_all = rng.uniform(5, 50, n_total)
    thetas_all = rng.uniform(np.radians(5), np.radians(85), n_total)
    ranges_true = v0s_all ** 2 * np.sin(2 * thetas_all) / g
    noise = rng.normal(0, 0.05 * np.std(ranges_true), n_total)
    ranges_noisy = ranges_true + noise

    sample_sizes = [5, 10, 25, 50, 100, 200, 500]
    r2_values: list[float] = []

    for n in sample_sizes:
        # Subsample
        idx = rng.choice(n_total, size=min(n, n_total), replace=False)
        v0s = v0s_all[idx]
        thetas = thetas_all[idx]
        ranges = ranges_noisy[idx]

        # Fit R = c * v0^2 * sin(2*theta)
        X = (v0s ** 2) * np.sin(2 * thetas)
        if np.std(X) < 1e-10:
            r2_values.append(0.0)
            continue

        c = np.sum(X * ranges) / np.sum(X ** 2)
        y_pred = c * X
        ss_res = np.sum((ranges - y_pred) ** 2)
        ss_tot = np.sum((ranges - np.mean(ranges)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        r2_values.append(r2)

    return ScalingResult(
        name="r2_vs_data_quantity",
        independent_var="n_samples",
        independent_values=[float(n) for n in sample_sizes],
        dependent_var="r_squared",
        dependent_values=r2_values,
        fit_type="linear",
        fit_params={},
    )


def run_scaling_analysis() -> list[ScalingResult]:
    """Run all scaling analyses."""
    results = []

    logger.info("1/3: Simulation runtime vs trajectory length")
    results.append(measure_simulation_scaling())

    logger.info("2/3: Runtime vs state dimension")
    results.append(measure_dimension_scaling())

    logger.info("3/3: R² vs data quantity")
    results.append(measure_data_quantity_scaling())

    return results


def format_scaling_results(results: list[ScalingResult]) -> str:
    """Format scaling results as text."""
    lines = []
    for r in results:
        lines.append(f"\n{r.name}")
        lines.append("=" * len(r.name))
        lines.append(f"  Independent: {r.independent_var}")
        lines.append(f"  Dependent:   {r.dependent_var}")
        lines.append(f"  Fit type:    {r.fit_type}")
        for k, v in r.fit_params.items():
            lines.append(f"  {k}: {v:.4f}")
        lines.append(f"  Data points: {len(r.independent_values)}")
        for x, y in zip(r.independent_values, r.dependent_values):
            lines.append(f"    {r.independent_var}={x:.0f} -> {r.dependent_var}={y:.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = run_scaling_analysis()
    print(format_scaling_results(results))
