"""Domain statistics: quick summary metrics for all simulation domains.

Computes runtime performance, trajectory statistics, and domain properties
for all 14 domains. Useful for benchmarking and paper reporting.
"""
from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Domain registry: module, class, domain enum, default params
DOMAIN_REGISTRY: dict[str, dict[str, Any]] = {
    "projectile": {
        "module": "simulating_anything.simulation.rigid_body",
        "cls": "ProjectileSimulation",
        "domain": Domain.RIGID_BODY,
        "params": {
            "initial_speed": 30.0, "launch_angle": 45.0,
            "gravity": 9.81, "drag_coefficient": 0.1, "mass": 1.0,
        },
        "dt": 0.01, "n_steps": 500, "math_class": "Algebraic",
    },
    "lotka_volterra": {
        "module": "simulating_anything.simulation.agent_based",
        "cls": "LotkaVolterraSimulation",
        "domain": Domain.AGENT_BASED,
        "params": {
            "alpha": 1.1, "beta": 0.4, "gamma": 0.4, "delta": 0.1,
            "prey_0": 40.0, "predator_0": 9.0,
        },
        "dt": 0.01, "n_steps": 500, "math_class": "Nonlinear ODE",
    },
    "sir_epidemic": {
        "module": "simulating_anything.simulation.epidemiological",
        "cls": "SIRSimulation",
        "domain": Domain.EPIDEMIOLOGICAL,
        "params": {"beta": 0.3, "gamma": 0.1, "S_0": 0.99, "I_0": 0.01},
        "dt": 0.1, "n_steps": 500, "math_class": "Nonlinear ODE",
    },
    "double_pendulum": {
        "module": "simulating_anything.simulation.chaotic_ode",
        "cls": "DoublePendulumSimulation",
        "domain": Domain.CHAOTIC_ODE,
        "params": {
            "m1": 1.0, "m2": 1.0, "L1": 1.0, "L2": 1.0, "g": 9.81,
            "theta1_0": 1.0, "theta2_0": 1.5,
            "omega1_0": 0.0, "omega2_0": 0.0,
        },
        "dt": 0.001, "n_steps": 500, "math_class": "Chaotic ODE",
    },
    "harmonic_oscillator": {
        "module": "simulating_anything.simulation.harmonic_oscillator",
        "cls": "DampedHarmonicOscillator",
        "domain": Domain.HARMONIC_OSCILLATOR,
        "params": {"k": 4.0, "m": 1.0, "c": 0.4, "x_0": 2.0, "v_0": 0.0},
        "dt": 0.01, "n_steps": 500, "math_class": "Linear ODE",
    },
    "lorenz": {
        "module": "simulating_anything.simulation.lorenz",
        "cls": "LorenzSimulation",
        "domain": Domain.LORENZ_ATTRACTOR,
        "params": {"sigma": 10.0, "rho": 28.0, "beta": 2.667},
        "dt": 0.01, "n_steps": 500, "math_class": "Chaotic ODE",
    },
    "navier_stokes": {
        "module": "simulating_anything.simulation.navier_stokes",
        "cls": "NavierStokes2DSimulation",
        "domain": Domain.NAVIER_STOKES_2D,
        "params": {"nu": 0.01, "N": 32},
        "dt": 0.01, "n_steps": 50, "math_class": "PDE",
    },
    "van_der_pol": {
        "module": "simulating_anything.simulation.van_der_pol",
        "cls": "VanDerPolSimulation",
        "domain": Domain.VAN_DER_POL,
        "params": {"mu": 1.0, "x_0": 2.0, "v_0": 0.0},
        "dt": 0.01, "n_steps": 500, "math_class": "Nonlinear ODE",
    },
    "kuramoto": {
        "module": "simulating_anything.simulation.kuramoto",
        "cls": "KuramotoSimulation",
        "domain": Domain.KURAMOTO,
        "params": {"N": 20, "K": 2.0, "omega_std": 1.0},
        "dt": 0.01, "n_steps": 500, "math_class": "Collective ODE",
    },
    "brusselator": {
        "module": "simulating_anything.simulation.brusselator",
        "cls": "BrusselatorSimulation",
        "domain": Domain.BRUSSELATOR,
        "params": {"a": 1.0, "b": 3.0, "u_0": 1.0, "v_0": 1.0},
        "dt": 0.01, "n_steps": 500, "math_class": "Nonlinear ODE",
    },
    "fitzhugh_nagumo": {
        "module": "simulating_anything.simulation.fitzhugh_nagumo",
        "cls": "FitzHughNagumoSimulation",
        "domain": Domain.FITZHUGH_NAGUMO,
        "params": {
            "a": 0.7, "b": 0.8, "eps": 0.08,
            "I_ext": 0.5, "v_0": -1.0, "w_0": -0.5,
        },
        "dt": 0.1, "n_steps": 500, "math_class": "Nonlinear ODE",
    },
    "heat_equation": {
        "module": "simulating_anything.simulation.heat_equation",
        "cls": "HeatEquation1DSimulation",
        "domain": Domain.HEAT_EQUATION_1D,
        "params": {"D": 0.1, "N": 64},
        "dt": 0.01, "n_steps": 200, "math_class": "Linear PDE",
    },
    "logistic_map": {
        "module": "simulating_anything.simulation.logistic_map",
        "cls": "LogisticMapSimulation",
        "domain": Domain.LOGISTIC_MAP,
        "params": {"r": 3.9, "x_0": 0.5},
        "dt": 1.0, "n_steps": 500, "math_class": "Discrete Chaos",
    },
}


@dataclass
class DomainStats:
    """Statistics for a single domain."""

    name: str
    math_class: str
    obs_dim: int
    n_steps: int
    run_time_ms: float
    state_mean: float
    state_std: float
    state_min: float
    state_max: float
    is_finite: bool
    is_deterministic: bool


def compute_domain_stats(domain_name: str) -> DomainStats:
    """Compute statistics for a single domain."""
    spec = DOMAIN_REGISTRY[domain_name]
    mod = importlib.import_module(spec["module"])
    cls = getattr(mod, spec["cls"])

    config = SimulationConfig(
        domain=spec["domain"], dt=spec["dt"], n_steps=spec["n_steps"],
        parameters=spec["params"],
    )
    sim = cls(config)

    # Benchmark
    t0 = time.perf_counter()
    traj = sim.run(n_steps=spec["n_steps"])
    run_time = (time.perf_counter() - t0) * 1000  # ms

    states = traj.states
    obs_dim = states.shape[1] if states.ndim > 1 else 1

    # Determinism check
    sim2 = cls(config)
    traj2 = sim2.run(n_steps=spec["n_steps"])
    is_det = np.allclose(states, traj2.states, atol=1e-10)

    return DomainStats(
        name=domain_name,
        math_class=spec["math_class"],
        obs_dim=obs_dim,
        n_steps=spec["n_steps"],
        run_time_ms=run_time,
        state_mean=float(np.mean(states)),
        state_std=float(np.std(states)),
        state_min=float(np.min(states)),
        state_max=float(np.max(states)),
        is_finite=bool(np.all(np.isfinite(states))),
        is_deterministic=is_det,
    )


def compute_all_stats(
    skip_kuramoto: bool = True,
) -> list[DomainStats]:
    """Compute statistics for all domains."""
    results = []
    for name in DOMAIN_REGISTRY:
        if skip_kuramoto and name == "kuramoto":
            # Kuramoto uses random frequencies, not deterministic
            continue
        try:
            stats = compute_domain_stats(name)
            results.append(stats)
            logger.info(
                f"  {name:25s}: dim={stats.obs_dim:4d}, "
                f"time={stats.run_time_ms:8.1f}ms, "
                f"range=[{stats.state_min:.3f}, {stats.state_max:.3f}]"
            )
        except Exception as e:
            logger.warning(f"  {name}: FAILED - {e}")
    return results


def print_stats_table(stats: list[DomainStats]) -> str:
    """Format stats as a text table."""
    lines = []
    lines.append(f"{'Domain':25s} {'Class':15s} {'Dim':>4s} {'Steps':>5s} "
                 f"{'Time(ms)':>8s} {'Range':>20s} {'Det':>4s}")
    lines.append("-" * 90)
    total_time = 0.0
    for s in stats:
        total_time += s.run_time_ms
        det = "Y" if s.is_deterministic else "N"
        lines.append(
            f"{s.name:25s} {s.math_class:15s} {s.obs_dim:4d} {s.n_steps:5d} "
            f"{s.run_time_ms:8.1f} [{s.state_min:8.3f}, {s.state_max:8.3f}] "
            f"{det:>4s}"
        )
    lines.append("-" * 90)
    lines.append(f"{'Total':25s} {'':15s} {'':4s} {'':5s} {total_time:8.1f}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Computing domain statistics...")
    stats = compute_all_stats(skip_kuramoto=False)
    print()
    print(print_stats_table(stats))
