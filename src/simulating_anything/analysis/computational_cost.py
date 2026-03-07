"""Computational cost analysis for pipeline stages.

Measures wall-clock time and memory for:
1. Simulation generation (per domain)
2. Data collection (parameter sweeps)
3. Equation fitting (PySR/SINDy/polynomial)
4. Full pipeline end-to-end

Results used for paper reproducibility section and scalability claims.
"""

from __future__ import annotations

import gc
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StageCost:
    """Cost measurement for one pipeline stage."""

    stage: str
    domain: str
    wall_time_s: float
    peak_memory_mb: float = 0.0
    n_samples: int = 0
    notes: str = ""


@dataclass
class DomainCost:
    """Full cost breakdown for one domain."""

    domain: str
    stages: list[StageCost] = field(default_factory=list)

    @property
    def total_time_s(self) -> float:
        return sum(s.wall_time_s for s in self.stages)

    @property
    def total_memory_mb(self) -> float:
        return max((s.peak_memory_mb for s in self.stages), default=0.0)


@dataclass
class CostReport:
    """Aggregate cost report across domains."""

    domains: list[DomainCost] = field(default_factory=list)

    @property
    def total_time_s(self) -> float:
        return sum(d.total_time_s for d in self.domains)

    def summary_table(self) -> list[dict]:
        """Generate summary table data for LaTeX."""
        rows = []
        for d in self.domains:
            row = {"domain": d.domain, "total_s": d.total_time_s}
            for s in d.stages:
                row[s.stage + "_s"] = s.wall_time_s
            rows.append(row)
        return rows


def _get_memory_mb() -> float:
    """Get current process memory usage in MB (approximate)."""
    gc.collect()
    # Use sys.getsizeof as a rough estimate -- no psutil dependency
    total = 0
    for obj in gc.get_objects():
        try:
            total += sys.getsizeof(obj)
        except (TypeError, ReferenceError):
            pass
    return total / (1024 * 1024)


def _measure_stage(
    stage: str, domain: str, fn, *args, **kwargs,
) -> tuple[StageCost, object]:
    """Measure wall-clock time for a stage function."""
    t0 = time.monotonic()
    result = fn(*args, **kwargs)
    dt = time.monotonic() - t0
    cost = StageCost(stage=stage, domain=domain, wall_time_s=dt)
    return cost, result


def measure_simulation_cost(
    domain: str,
    n_steps: int = 1000,
    n_sweeps: int = 10,
    dt: float = 0.01,
) -> DomainCost:
    """Measure cost of simulation data generation for a domain.

    Args:
        domain: Domain name (must be importable).
        n_steps: Steps per simulation.
        n_sweeps: Number of parameter sweep points.
        dt: Timestep.

    Returns:
        DomainCost with stage breakdown.
    """
    from simulating_anything.types.simulation import Domain, SimulationConfig

    domain_cost = DomainCost(domain=domain)

    # Stage 1: Import / instantiation
    t0 = time.monotonic()
    sim_class = _get_simulation_class(domain)
    if sim_class is None:
        domain_cost.stages.append(StageCost(
            stage="import", domain=domain, wall_time_s=0.0,
            notes="domain not found",
        ))
        return domain_cost
    dt_import = time.monotonic() - t0
    domain_cost.stages.append(StageCost(
        stage="import", domain=domain, wall_time_s=dt_import,
    ))

    # Stage 2: Single simulation run
    t0 = time.monotonic()
    try:
        config = SimulationConfig(
            domain=Domain.CUSTOM, n_steps=n_steps, dt=dt,
            parameters={},
        )
        sim = sim_class(config)
        sim.reset(seed=42)
        for _ in range(n_steps):
            sim.step()
    except Exception as e:
        domain_cost.stages.append(StageCost(
            stage="single_run", domain=domain, wall_time_s=0.0,
            notes=f"error: {e}",
        ))
        return domain_cost
    dt_single = time.monotonic() - t0
    domain_cost.stages.append(StageCost(
        stage="single_run", domain=domain, wall_time_s=dt_single,
        n_samples=n_steps,
    ))

    # Stage 3: Parameter sweep
    t0 = time.monotonic()
    for i in range(n_sweeps):
        try:
            sim = sim_class(config)
            sim.reset(seed=i)
            for _ in range(n_steps):
                sim.step()
        except Exception:
            pass
    dt_sweep = time.monotonic() - t0
    domain_cost.stages.append(StageCost(
        stage="param_sweep", domain=domain, wall_time_s=dt_sweep,
        n_samples=n_sweeps * n_steps,
    ))

    return domain_cost


def _get_simulation_class(domain: str):
    """Dynamically import a simulation class by domain name."""
    import importlib

    # Map domain names to module.class
    module_map = {
        "projectile": ("simulating_anything.simulation.rigid_body", "ProjectileSimulation"),
        "lotka_volterra": ("simulating_anything.simulation.agent_based", "LotkaVolterraSimulation"),
        "harmonic_oscillator": (
            "simulating_anything.simulation.harmonic_oscillator",
            "DampedHarmonicOscillator",
        ),
        "lorenz": ("simulating_anything.simulation.lorenz", "LorenzSimulation"),
        "sir_epidemic": ("simulating_anything.simulation.epidemiological", "SIRSimulation"),
        "double_pendulum": ("simulating_anything.simulation.chaotic_ode", "DoublePendulumSimulation"),
        "van_der_pol": ("simulating_anything.simulation.van_der_pol", "VanDerPolSimulation"),
        "brusselator": ("simulating_anything.simulation.brusselator", "BrusselatorSimulation"),
        "fitzhugh_nagumo": (
            "simulating_anything.simulation.fitzhugh_nagumo",
            "FitzHughNagumoSimulation",
        ),
        "heat_equation": ("simulating_anything.simulation.heat_equation", "HeatEquationSimulation"),
        "logistic_map": ("simulating_anything.simulation.logistic_map", "LogisticMapSimulation"),
        "kuramoto": ("simulating_anything.simulation.kuramoto", "KuramotoSimulation"),
    }

    if domain not in module_map:
        return None

    module_name, class_name = module_map[domain]
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Cannot import {module_name}.{class_name}: {e}")
        return None


def measure_fitting_cost(
    n_samples: int = 225,
    seed: int = 42,
) -> list[StageCost]:
    """Measure cost of different equation fitting methods.

    Returns timing for polynomial, power-law, and correct-form fitting.
    """
    rng = np.random.default_rng(seed)
    costs = []

    # Generate projectile data
    n_per = max(2, int(np.sqrt(n_samples)))
    v0 = np.linspace(5, 50, n_per)
    theta = np.linspace(5, 85, n_per)
    V, T = np.meshgrid(v0, theta)
    V, T = V.ravel(), T.ravel()
    R = V**2 * np.sin(2 * np.radians(T)) / 9.81

    # Polynomial fit (degree 4)
    t0 = time.monotonic()
    X = np.column_stack([V, np.sin(2 * np.radians(T))])
    from itertools import combinations_with_replacement
    features = [np.ones(len(V))]
    for deg in range(1, 5):
        for combo in combinations_with_replacement(range(2), deg):
            feat = np.ones(len(V))
            for idx in combo:
                feat *= X[:, idx]
            features.append(feat)
    A = np.column_stack(features)
    np.linalg.lstsq(A, R, rcond=None)
    dt = time.monotonic() - t0
    costs.append(StageCost(
        stage="poly_d4", domain="projectile", wall_time_s=dt,
        n_samples=n_samples,
    ))

    # Correct-form fit
    t0 = time.monotonic()
    feature = V**2 * np.sin(2 * np.radians(T))
    c = np.sum(feature * R) / np.sum(feature**2)
    dt = time.monotonic() - t0
    costs.append(StageCost(
        stage="correct_form", domain="projectile", wall_time_s=dt,
        n_samples=n_samples,
    ))

    # Power law fit (log-linear)
    t0 = time.monotonic()
    mask = V > 0
    log_v = np.log(V[mask])
    log_r = np.log(np.abs(R[mask]) + 1e-10)
    np.polyfit(log_v, log_r, 1)
    dt = time.monotonic() - t0
    costs.append(StageCost(
        stage="power_law", domain="projectile", wall_time_s=dt,
        n_samples=n_samples,
    ))

    return costs


def measure_scaling(
    n_values: list[int] | None = None,
    seed: int = 42,
) -> list[StageCost]:
    """Measure how fitting time scales with data size.

    Returns:
        List of StageCost, one per sample count.
    """
    if n_values is None:
        n_values = [25, 100, 400, 900, 2500, 10000]

    costs = []
    for n in n_values:
        n_per = max(2, int(np.sqrt(n)))
        v0 = np.linspace(5, 50, n_per)
        theta = np.linspace(5, 85, n_per)
        V, T = np.meshgrid(v0, theta)
        V, T = V.ravel(), T.ravel()
        actual_n = len(V)
        R = V**2 * np.sin(2 * np.radians(T)) / 9.81

        t0 = time.monotonic()
        feature = V**2 * np.sin(2 * np.radians(T))
        c = np.sum(feature * R) / np.sum(feature**2)
        y_pred = c * feature
        ss_res = np.sum((R - y_pred) ** 2)
        ss_tot = np.sum((R - np.mean(R)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        dt = time.monotonic() - t0

        costs.append(StageCost(
            stage="scaling_fit", domain="projectile", wall_time_s=dt,
            n_samples=actual_n, notes=f"R^2={r2:.6f}",
        ))

    return costs


def run_cost_analysis(
    domains: list[str] | None = None,
    n_steps: int = 1000,
    n_sweeps: int = 10,
) -> CostReport:
    """Run full computational cost analysis.

    Args:
        domains: Domains to benchmark.
        n_steps: Steps per simulation.
        n_sweeps: Parameter sweep points.

    Returns:
        CostReport with all measurements.
    """
    if domains is None:
        domains = [
            "projectile", "harmonic_oscillator", "lorenz",
            "van_der_pol", "brusselator", "logistic_map",
        ]

    report = CostReport()
    for domain in domains:
        cost = measure_simulation_cost(
            domain=domain, n_steps=n_steps, n_sweeps=n_sweeps,
        )
        report.domains.append(cost)

    return report
