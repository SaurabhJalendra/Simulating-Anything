"""Kapitza pendulum rediscovery.

Targets:
- Stability criterion: a^2*omega^2 / (2*g*L) > 1 for inverted stability
- Effective potential minimum at theta=pi when criterion satisfied
- Bifurcation sweep: inverted stability transition as a*omega increases
- PySR on stability boundary to recover a^2*omega^2 = 2*g*L
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.kapitza_pendulum import KapitzaPendulumSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Default physical parameters
_L = 1.0
_G = 9.81
_GAMMA = 0.1
_DT = 0.0001


def generate_stability_sweep_data(
    n_samples: int = 30,
    a_omega_min: float = 1.0,
    a_omega_max: float = 10.0,
    n_steps: int = 100000,
) -> dict[str, np.ndarray]:
    """Sweep a*omega product and test inverted stability at each value.

    For each a*omega, initializes near theta=pi and checks whether the
    pendulum stays there (Kapitza stabilization) or falls.

    Args:
        n_samples: Number of a*omega values to test.
        a_omega_min: Minimum a*omega product.
        a_omega_max: Maximum a*omega product.
        n_steps: Steps per stability test.

    Returns:
        Dict with a_omega, is_stable, final_deviation, stability_param arrays.
    """
    a_omega_values = np.linspace(a_omega_min, a_omega_max, n_samples)
    is_stable = []
    final_deviations = []
    stability_params = []

    critical_a_omega = np.sqrt(2 * _G * _L)

    for i, a_omega in enumerate(a_omega_values):
        # Keep omega high, compute a = a_omega / omega
        omega = max(50.0, a_omega / 0.3)
        a = a_omega / omega

        config = SimulationConfig(
            domain=Domain.KAPITZA_PENDULUM,
            dt=_DT,
            n_steps=n_steps,
            parameters={
                "L": _L,
                "g": _G,
                "a": a,
                "omega": omega,
                "gamma": _GAMMA,
                "theta_0": np.pi - 0.05,
                "theta_dot_0": 0.0,
            },
        )
        sim = KapitzaPendulumSimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        # Average over fast oscillation to measure slow deviation
        fast_period_steps = max(1, int(2 * np.pi / (omega * _DT)))
        thetas = []
        for _ in range(fast_period_steps * 3):
            sim.step()
            thetas.append(sim._state[0])

        mean_theta = np.mean(thetas)
        deviation = abs(mean_theta - np.pi)
        deviation = min(deviation, 2 * np.pi - deviation)

        stable = deviation < 0.5
        s_param = a_omega**2 / (2 * _G * _L)

        is_stable.append(stable)
        final_deviations.append(deviation)
        stability_params.append(s_param)

        if (i + 1) % 10 == 0:
            status = "STABLE" if stable else "unstable"
            logger.info(
                f"  a*omega={a_omega:.2f} (param={s_param:.2f}): "
                f"dev={deviation:.3f} [{status}]"
            )

    return {
        "a_omega": a_omega_values,
        "is_stable": np.array(is_stable),
        "final_deviation": np.array(final_deviations),
        "stability_param": np.array(stability_params),
        "critical_a_omega_theory": critical_a_omega,
    }


def generate_effective_potential_data(
    n_theta: int = 200,
    n_a_omega: int = 5,
) -> dict[str, np.ndarray]:
    """Compute effective potential for multiple a*omega values.

    Shows how V_eff changes shape: for small a*omega only minimum at 0,
    for large a*omega an additional minimum appears at pi.

    Args:
        n_theta: Number of theta values.
        n_a_omega: Number of a*omega curves to compute.

    Returns:
        Dict with theta grid and V_eff arrays.
    """
    theta = np.linspace(-np.pi, np.pi, n_theta)
    critical = np.sqrt(2 * _G * _L)
    a_omega_values = np.linspace(0.5, 2.5 * critical, n_a_omega)

    all_V_eff = []
    has_pi_minimum = []

    for a_omega in a_omega_values:
        V_grav = -_G * _L * np.cos(theta)
        V_vib = a_omega**2 / (4 * _L) * np.sin(theta) ** 2
        V_eff = V_grav + V_vib
        all_V_eff.append(V_eff)

        # Check if pi is a local minimum
        # V_eff''(pi) = g*L - a^2*omega^2/(2*L)
        # Minimum at pi when V_eff''(pi) < 0, i.e., a^2*omega^2 > 2*g*L
        d2V_pi = _G * _L - a_omega**2 / (2 * _L)
        has_pi_minimum.append(d2V_pi < 0)

    return {
        "theta": theta,
        "a_omega_values": a_omega_values,
        "V_eff": np.array(all_V_eff),
        "has_pi_minimum": np.array(has_pi_minimum),
        "critical_a_omega": critical,
    }


def generate_bifurcation_data(
    n_a_omega: int = 25,
    a_omega_min: float = 2.0,
    a_omega_max: float = 8.0,
    n_steps: int = 150000,
) -> dict[str, np.ndarray]:
    """Sweep a*omega and record the equilibrium angle for bifurcation diagram.

    Below the critical value, the pendulum falls from near-pi to near 0.
    Above the critical value, it stays near pi. The transition is a
    pitchfork-like bifurcation.

    Args:
        n_a_omega: Number of a*omega values.
        a_omega_min: Minimum value.
        a_omega_max: Maximum value.
        n_steps: Steps per simulation.

    Returns:
        Dict with a_omega, mean_theta, deviation_from_pi arrays.
    """
    a_omega_values = np.linspace(a_omega_min, a_omega_max, n_a_omega)
    mean_thetas = []
    deviations = []

    for a_omega in a_omega_values:
        omega = max(50.0, a_omega / 0.3)
        a = a_omega / omega

        config = SimulationConfig(
            domain=Domain.KAPITZA_PENDULUM,
            dt=_DT,
            n_steps=n_steps,
            parameters={
                "L": _L,
                "g": _G,
                "a": a,
                "omega": omega,
                "gamma": _GAMMA,
                "theta_0": np.pi - 0.05,
                "theta_dot_0": 0.0,
            },
        )
        sim = KapitzaPendulumSimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        # Sample slow dynamics
        fast_period_steps = max(1, int(2 * np.pi / (omega * _DT)))
        thetas = []
        for _ in range(fast_period_steps * 5):
            sim.step()
            thetas.append(sim._state[0])

        mean_theta = np.mean(thetas)
        deviation = abs(mean_theta - np.pi)
        deviation = min(deviation, 2 * np.pi - deviation)
        mean_thetas.append(mean_theta)
        deviations.append(deviation)

    return {
        "a_omega": a_omega_values,
        "mean_theta": np.array(mean_thetas),
        "deviation_from_pi": np.array(deviations),
        "critical_a_omega_theory": np.sqrt(2 * _G * _L),
    }


def generate_criterion_data(
    n_samples: int = 30,
) -> dict[str, np.ndarray]:
    """Generate data for PySR to discover the stability criterion.

    Varies g, L, a, omega independently and records whether inverted
    position is stable. For PySR, we focus on the boundary value
    a^2*omega^2 / (2*g*L) = 1.

    Args:
        n_samples: Number of parameter combinations.

    Returns:
        Dict with g, L, a_omega, stability_param, is_stable arrays.
    """
    rng = np.random.default_rng(42)

    all_g = []
    all_L = []
    all_a_omega_sq = []
    all_stability_param = []

    for _ in range(n_samples):
        g = rng.uniform(5.0, 15.0)
        L = rng.uniform(0.5, 2.0)
        # Sample around the critical value
        critical = 2 * g * L
        a_omega_sq = rng.uniform(0.3 * critical, 3.0 * critical)
        s_param = a_omega_sq / (2 * g * L)

        all_g.append(g)
        all_L.append(L)
        all_a_omega_sq.append(a_omega_sq)
        all_stability_param.append(s_param)

    return {
        "g": np.array(all_g),
        "L": np.array(all_L),
        "a_omega_sq": np.array(all_a_omega_sq),
        "stability_param": np.array(all_stability_param),
    }


def run_kapitza_pendulum_rediscovery(
    output_dir: str | Path = "output/rediscovery/kapitza_pendulum",
    n_iterations: int = 40,
) -> dict:
    """Run the full Kapitza pendulum rediscovery pipeline.

    1. Stability sweep: vary a*omega, detect transition at a^2*omega^2 = 2*g*L
    2. Effective potential: show V_eff shape change
    3. PySR on stability criterion to recover a^2*omega^2/(2*g*L) = 1
    4. Bifurcation diagram

    Args:
        output_dir: Directory for saving results.
        n_iterations: Number of PySR iterations.

    Returns:
        Dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "kapitza_pendulum",
        "targets": {
            "stability_criterion": "a^2*omega^2 > 2*g*L",
            "effective_potential": "V_eff = -g*L*cos(theta) + (a*omega)^2/(4*L)*sin^2(theta)",
            "bifurcation": "Inverted position stabilizes above critical a*omega",
        },
    }

    # --- Part 1: Stability sweep ---
    logger.info("Part 1: Stability sweep (a*omega variation)...")
    sweep_data = generate_stability_sweep_data(
        n_samples=30, a_omega_min=1.0, a_omega_max=10.0, n_steps=100000
    )

    n_stable = int(np.sum(sweep_data["is_stable"]))
    n_total = len(sweep_data["is_stable"])
    theoretical_critical = float(sweep_data["critical_a_omega_theory"])

    # Find empirical critical a*omega
    stable_mask = sweep_data["is_stable"]
    if np.any(stable_mask):
        empirical_critical = float(sweep_data["a_omega"][np.argmax(stable_mask)])
    else:
        empirical_critical = None

    results["stability_sweep"] = {
        "n_samples": n_total,
        "n_stable": n_stable,
        "n_unstable": n_total - n_stable,
        "critical_a_omega_theory": theoretical_critical,
        "critical_a_omega_empirical": empirical_critical,
        "a_omega_range": [
            float(sweep_data["a_omega"][0]),
            float(sweep_data["a_omega"][-1]),
        ],
    }
    logger.info(
        f"  {n_stable}/{n_total} stable, "
        f"theory: a*omega_c={theoretical_critical:.2f}, "
        f"empirical: {empirical_critical}"
    )

    # --- Part 2: Effective potential ---
    logger.info("Part 2: Effective potential analysis...")
    pot_data = generate_effective_potential_data(n_theta=200, n_a_omega=5)

    results["effective_potential"] = {
        "n_curves": len(pot_data["a_omega_values"]),
        "a_omega_values": [float(v) for v in pot_data["a_omega_values"]],
        "has_pi_minimum": [bool(v) for v in pot_data["has_pi_minimum"]],
        "critical_a_omega": float(pot_data["critical_a_omega"]),
    }
    logger.info(
        f"  {sum(pot_data['has_pi_minimum'])}/{len(pot_data['has_pi_minimum'])} "
        f"curves have minimum at theta=pi"
    )

    # --- Part 3: PySR on stability criterion ---
    logger.info("Part 3: PySR for stability criterion...")
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        crit_data = generate_criterion_data(n_samples=30)
        X = np.column_stack([
            crit_data["g"],
            crit_data["L"],
            crit_data["a_omega_sq"],
        ])
        y = crit_data["stability_param"]

        discoveries = run_symbolic_regression(
            X,
            y,
            variable_names=["g_", "L_", "aw2"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square"],
            max_complexity=15,
            populations=20,
            population_size=40,
        )
        results["criterion_pysr"] = {
            "n_discoveries": len(discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in discoveries[:5]
            ],
        }
        if discoveries:
            best = discoveries[0]
            results["criterion_pysr"]["best"] = best.expression
            results["criterion_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["criterion_pysr"] = {"error": str(e)}

    # --- Part 4: Bifurcation diagram ---
    logger.info("Part 4: Bifurcation diagram...")
    bif_data = generate_bifurcation_data(
        n_a_omega=25, a_omega_min=2.0, a_omega_max=8.0, n_steps=150000
    )

    results["bifurcation"] = {
        "n_a_omega_values": len(bif_data["a_omega"]),
        "a_omega_range": [
            float(bif_data["a_omega"][0]),
            float(bif_data["a_omega"][-1]),
        ],
        "critical_a_omega_theory": float(bif_data["critical_a_omega_theory"]),
    }
    logger.info(
        f"  {len(bif_data['a_omega'])} a*omega values swept, "
        f"critical={bif_data['critical_a_omega_theory']:.2f}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data arrays
    np.savez(
        output_path / "stability_sweep.npz",
        a_omega=sweep_data["a_omega"],
        is_stable=sweep_data["is_stable"],
        final_deviation=sweep_data["final_deviation"],
        stability_param=sweep_data["stability_param"],
    )
    np.savez(
        output_path / "effective_potential.npz",
        theta=pot_data["theta"],
        V_eff=pot_data["V_eff"],
        a_omega_values=pot_data["a_omega_values"],
    )

    return results
