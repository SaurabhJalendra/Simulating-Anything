"""Two-Patch Predator-Prey rediscovery.

Targets:
- Migration-synchronization relationship: high migration -> synchronized patches
- Low migration -> asynchronous dynamics with rescue effect
- ODE recovery via SINDy for the 4-variable coupled system
- Coexistence equilibrium verification
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.two_patch import TwoPatchSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    r: float = 1.0,
    K: float = 10.0,
    a: float = 0.5,
    e: float = 0.4,
    d: float = 0.2,
    m_N: float = 0.1,
    m_P: float = 0.05,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Args:
        r: Prey growth rate.
        K: Carrying capacity.
        a: Attack rate.
        e: Conversion efficiency.
        d: Predator death rate.
        m_N: Prey migration rate.
        m_P: Predator migration rate.
        n_steps: Number of integration steps.
        dt: Timestep.

    Returns:
        Dict with 'time', 'states', and individual variable arrays.
    """
    config = SimulationConfig(
        domain=Domain.TWO_PATCH,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r": r, "K": K, "a": a, "e": e, "d": d,
            "m_N": m_N, "m_P": m_P,
            "N1_0": 5.0, "P1_0": 2.0,
            "N2_0": 3.0, "P2_0": 1.0,
        },
    )
    sim = TwoPatchSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    times = np.arange(n_steps + 1) * dt

    return {
        "time": times,
        "states": states_arr,
        "N1": states_arr[:, 0],
        "P1": states_arr[:, 1],
        "N2": states_arr[:, 2],
        "P2": states_arr[:, 3],
        "r": r, "K": K, "a": a, "e": e, "d": d,
        "m_N": m_N, "m_P": m_P,
        "dt": dt,
    }


def generate_migration_sweep_data(
    n_m: int = 25,
    m_min: float = 0.0,
    m_max: float = 1.0,
    n_steps: int = 8000,
    n_transient: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep prey migration rate and measure steady-state patch synchrony.

    Args:
        n_m: Number of migration values to test.
        m_min: Minimum migration rate.
        m_max: Maximum migration rate.
        n_steps: Measurement steps after transient.
        n_transient: Transient steps to discard.
        dt: Timestep.

    Returns:
        Dict with 'm_N', 'mean_diff', and 'mean_total_prey' arrays.
    """
    m_values = np.linspace(m_min, m_max, n_m)
    mean_diffs = np.empty(n_m)
    mean_totals = np.empty(n_m)

    for i, m_val in enumerate(m_values):
        config = SimulationConfig(
            domain=Domain.TWO_PATCH,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": 1.0, "K": 10.0, "a": 0.5, "e": 0.4, "d": 0.2,
                "m_N": float(m_val), "m_P": 0.05,
                "N1_0": 5.0, "P1_0": 2.0,
                "N2_0": 3.0, "P2_0": 1.0,
            },
        )
        sim = TwoPatchSimulation(config)
        sim.reset()

        for _ in range(n_transient):
            sim.step()

        diff_sum = 0.0
        total_sum = 0.0
        for _ in range(n_steps):
            sim.step()
            diff_sum += sim.patch_difference()
            total_sum += sim.total_prey()

        mean_diffs[i] = diff_sum / n_steps
        mean_totals[i] = total_sum / n_steps

        if (i + 1) % 5 == 0:
            logger.info(
                f"  m_N={m_val:.3f}: mean_diff={mean_diffs[i]:.4f}, "
                f"total_prey={mean_totals[i]:.4f}"
            )

    return {
        "m_N": m_values,
        "mean_diff": mean_diffs,
        "mean_total_prey": mean_totals,
    }


def generate_sync_comparison_data(
    n_steps: int = 15000,
    dt: float = 0.01,
) -> dict[str, dict]:
    """Compare low vs high migration outcomes.

    Args:
        n_steps: Total integration steps.
        dt: Timestep.

    Returns:
        Dict with 'low_migration' and 'high_migration' sub-dicts.
    """
    results = {}

    for label, m_N, m_P in [
        ("low_migration", 0.001, 0.001),
        ("high_migration", 1.0, 0.5),
    ]:
        config = SimulationConfig(
            domain=Domain.TWO_PATCH,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": 1.0, "K": 10.0, "a": 0.5, "e": 0.4, "d": 0.2,
                "m_N": m_N, "m_P": m_P,
                "N1_0": 8.0, "P1_0": 1.0,
                "N2_0": 2.0, "P2_0": 3.0,
            },
        )
        sim = TwoPatchSimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        # Measure over final portion
        n_measure = 3000
        diff_sum = 0.0
        sync_count = 0
        for _ in range(n_measure):
            sim.step()
            diff_sum += sim.patch_difference()
            if sim.is_synchronized(threshold=0.5):
                sync_count += 1

        results[label] = {
            "m_N": m_N,
            "m_P": m_P,
            "mean_patch_diff": diff_sum / n_measure,
            "sync_fraction": sync_count / n_measure,
            "final_state": sim.observe().tolist(),
            "total_prey": sim.total_prey(),
            "total_predator": sim.total_predator(),
        }

    return results


def run_two_patch_rediscovery(
    output_dir: str | Path = "output/rediscovery/two_patch",
    n_iterations: int = 40,
) -> dict:
    """Run the full two-patch predator-prey rediscovery pipeline.

    1. Generate trajectory for SINDy ODE recovery
    2. Sweep migration rate to show sync transition
    3. Compare low vs high migration dynamics
    4. Verify coexistence equilibrium

    Args:
        output_dir: Directory for saving results and data files.
        n_iterations: Number of PySR iterations.

    Returns:
        Dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "two_patch",
        "targets": {
            "ode": (
                "dN1/dt = r*N1*(1-N1/K) - a*N1*P1 + m_N*(N2-N1), "
                "dP1/dt = e*a*N1*P1 - d*P1 + m_P*(P2-P1), "
                "dN2/dt = r*N2*(1-N2/K) - a*N2*P2 + m_N*(N1-N2), "
                "dP2/dt = e*a*N2*P2 - d*P2 + m_P*(P1-P2)"
            ),
            "sync_transition": (
                "High migration -> synchronized patches, "
                "low migration -> asynchronous"
            ),
            "equilibrium": "N* = d/(e*a), P* = (r/a)*(1 - N*/K)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery for two-patch predator-prey...")
    ode_data = generate_ode_data(n_steps=10000, dt=0.01)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["N1", "P1", "N2", "P2"],
            threshold=0.05,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {
                    "expression": d_.expression,
                    "r_squared": d_.evidence.fit_r_squared,
                }
                for d_ in sindy_discoveries[:8]
            ],
        }
        if sindy_discoveries:
            best = sindy_discoveries[0]
            results["sindy_ode"]["best"] = best.expression
            results["sindy_ode"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  SINDy best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as exc:
        logger.warning(f"SINDy failed: {exc}")
        results["sindy_ode"] = {"error": str(exc)}

    # --- Part 2: Migration sweep ---
    logger.info("Part 2: Sweeping prey migration rate m_N...")
    sweep_data = generate_migration_sweep_data(
        n_m=25, m_min=0.0, m_max=1.0,
        n_steps=8000, n_transient=5000, dt=0.01,
    )

    results["migration_sweep"] = {
        "n_values": len(sweep_data["m_N"]),
        "m_range": [float(sweep_data["m_N"][0]), float(sweep_data["m_N"][-1])],
        "min_diff": float(np.min(sweep_data["mean_diff"])),
        "max_diff": float(np.max(sweep_data["mean_diff"])),
        "diff_at_m0": float(sweep_data["mean_diff"][0]),
        "diff_at_mmax": float(sweep_data["mean_diff"][-1]),
    }

    # Check that patch difference decreases with migration
    low_m_diff = np.mean(sweep_data["mean_diff"][:5])
    high_m_diff = np.mean(sweep_data["mean_diff"][-5:])
    results["migration_sweep"]["low_m_mean_diff"] = float(low_m_diff)
    results["migration_sweep"]["high_m_mean_diff"] = float(high_m_diff)
    results["migration_sweep"]["diff_decreases"] = bool(
        high_m_diff < low_m_diff
    )

    # --- Part 3: Low vs high migration comparison ---
    logger.info("Part 3: Comparing low vs high migration dynamics...")
    sync_data = generate_sync_comparison_data(n_steps=15000, dt=0.01)

    results["sync_comparison"] = {}
    for label, data in sync_data.items():
        results["sync_comparison"][label] = {
            k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                v if isinstance(v, (list, str)) else float(v))
            for k, v in data.items()
        }

    logger.info(
        f"  Low migration: mean_diff={sync_data['low_migration']['mean_patch_diff']:.4f}, "
        f"sync_frac={sync_data['low_migration']['sync_fraction']:.4f}"
    )
    logger.info(
        f"  High migration: mean_diff={sync_data['high_migration']['mean_patch_diff']:.4f}, "
        f"sync_frac={sync_data['high_migration']['sync_fraction']:.4f}"
    )

    # --- Part 4: Coexistence equilibrium verification ---
    logger.info("Part 4: Verifying coexistence equilibrium...")
    config_eq = SimulationConfig(
        domain=Domain.TWO_PATCH,
        dt=0.01,
        n_steps=1000,
        parameters={
            "r": 1.0, "K": 10.0, "a": 0.5, "e": 0.4, "d": 0.2,
            "m_N": 0.0, "m_P": 0.0,
            "N1_0": 5.0, "P1_0": 2.0,
            "N2_0": 5.0, "P2_0": 2.0,
        },
    )
    sim_eq = TwoPatchSimulation(config_eq)
    sim_eq.reset()
    N_star, P_star = sim_eq.coexistence_equilibrium()

    results["equilibrium"] = {
        "N_star": N_star,
        "P_star": P_star,
        "formula_N": "d / (e*a)",
        "formula_P": "(r/a) * (1 - N*/K)",
    }
    logger.info(f"  Equilibrium: N*={N_star:.4f}, P*={P_star:.4f}")

    # --- Save results ---
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "migration_sweep.npz",
        m_N=sweep_data["m_N"],
        mean_diff=sweep_data["mean_diff"],
        mean_total_prey=sweep_data["mean_total_prey"],
    )

    return results
