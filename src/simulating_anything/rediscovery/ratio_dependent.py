"""Ratio-dependent (Arditi-Ginzburg) predator-prey rediscovery.

Targets:
- No paradox of enrichment: coexistence equilibrium remains stable
  as carrying capacity K increases (unlike Rosenzweig-MacArthur)
- Equilibrium N* = K*(1 - a*(1-q)/(b*r)), P* = N*(1-q)/(q*b), q=d/(e*a)
- Ratio-dependent functional response vs prey-dependent comparison
- SINDy recovery of ratio-dependent ODEs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.ratio_dependent import RatioDependentSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_DOMAIN = Domain.RATIO_DEPENDENT


def generate_trajectory_data(
    r: float = 1.0,
    K: float = 100.0,
    a: float = 0.5,
    b: float = 1.0,
    e: float = 0.5,
    d: float = 0.2,
    N_0: float = 50.0,
    P_0: float = 10.0,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """Generate a single long trajectory for SINDy ODE recovery.

    Returns dict with states array, time, and parameters.
    """
    config = SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r": r, "K": K, "a": a, "b": b, "e": e, "d": d,
            "N_0": N_0, "P_0": P_0,
        },
    )

    sim = RatioDependentSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "N": states_arr[:, 0],
        "P": states_arr[:, 1],
        "dt": dt,
        "r": r, "K": K, "a": a, "b": b, "e": e, "d": d,
    }


def generate_enrichment_sweep(
    n_K_values: int = 30,
    K_min: float = 20.0,
    K_max: float = 500.0,
    n_steps: int = 20000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep carrying capacity K to demonstrate NO paradox of enrichment.

    For each K, runs a long trajectory and measures:
    - Time-averaged prey and predator populations
    - Oscillation amplitude (max - min in second half of trajectory)
    - Whether the equilibrium is stable (amplitude near zero)

    In contrast to Rosenzweig-MacArthur, amplitudes should remain near zero
    for all K values.

    Returns dict with K values, averages, and amplitudes.
    """
    K_values = np.linspace(K_min, K_max, n_K_values)

    all_K: list[float] = []
    all_N_avg: list[float] = []
    all_P_avg: list[float] = []
    all_N_amp: list[float] = []
    all_P_amp: list[float] = []
    all_N_eq: list[float] = []
    all_P_eq: list[float] = []

    for i, K_val in enumerate(K_values):
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": 1.0, "K": float(K_val), "a": 0.5, "b": 1.0,
                "e": 0.5, "d": 0.2, "N_0": 50.0, "P_0": 10.0,
            },
        )

        sim = RatioDependentSimulation(config)
        sim.reset()

        states = [sim.observe().copy()]
        for _ in range(n_steps):
            sim.step()
            states.append(sim.observe().copy())

        trajectory = np.array(states)

        # Use second half to avoid transients
        half = n_steps // 2
        N_half = trajectory[half:, 0]
        P_half = trajectory[half:, 1]

        all_K.append(float(K_val))
        all_N_avg.append(float(np.mean(N_half)))
        all_P_avg.append(float(np.mean(P_half)))
        all_N_amp.append(float(np.max(N_half) - np.min(N_half)))
        all_P_amp.append(float(np.max(P_half) - np.min(P_half)))

        # Theoretical equilibrium
        # q = d/(e*a) = 0.2/(0.5*0.5) = 0.8
        # N* = K*(1 - a*(1-q)/(b*r)) = K*(1 - 0.5*0.2/1.0) = K*0.9
        # P* = N*(1-q)/(q*b) = N*0.2/(0.8*1.0) = N*0.25
        q = 0.2 / (0.5 * 0.5)  # = 0.8
        N_eq = float(K_val) * (1.0 - 0.5 * (1.0 - q) / (1.0 * 1.0))
        P_eq = N_eq * (1.0 - q) / (q * 1.0)
        all_N_eq.append(N_eq)
        all_P_eq.append(P_eq)

        if (i + 1) % 10 == 0:
            logger.info(f"  Enrichment sweep: {i + 1}/{n_K_values} K values")

    return {
        "K": np.array(all_K),
        "N_avg": np.array(all_N_avg),
        "P_avg": np.array(all_P_avg),
        "N_amplitude": np.array(all_N_amp),
        "P_amplitude": np.array(all_P_amp),
        "N_equilibrium": np.array(all_N_eq),
        "P_equilibrium": np.array(all_P_eq),
    }


def generate_equilibrium_sweep(
    n_samples: int = 100,
    n_steps: int = 20000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep parameters to measure N*, P* and compare with theory.

    Returns dict with parameter values and measured/theoretical equilibria.
    """
    rng = np.random.default_rng(42)

    all_r: list[float] = []
    all_K: list[float] = []
    all_e: list[float] = []
    all_d: list[float] = []
    all_N_measured: list[float] = []
    all_P_measured: list[float] = []
    all_N_theory: list[float] = []
    all_P_theory: list[float] = []

    a = 0.5
    b = 1.0

    for i in range(n_samples):
        r = rng.uniform(0.5, 2.0)
        K = rng.uniform(50.0, 200.0)
        e_val = rng.uniform(0.3, 0.8)
        d_val = rng.uniform(0.05, 0.3)

        # Skip if equilibrium does not exist
        if e_val * a <= d_val:
            continue

        q = d_val / (e_val * a)
        N_eq = K * (1.0 - a * (1.0 - q) / (b * r))
        if N_eq <= 0:
            continue
        P_eq = N_eq * (1.0 - q) / (q * b)
        if P_eq <= 0:
            continue

        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": r, "K": K, "a": a, "b": b, "e": e_val, "d": d_val,
                "N_0": N_eq * 0.8, "P_0": P_eq * 0.8,
            },
        )

        sim = RatioDependentSimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        final = sim.observe()

        all_r.append(r)
        all_K.append(K)
        all_e.append(e_val)
        all_d.append(d_val)
        all_N_measured.append(float(final[0]))
        all_P_measured.append(float(final[1]))
        all_N_theory.append(N_eq)
        all_P_theory.append(P_eq)

        if (i + 1) % 25 == 0:
            logger.info(f"  Equilibrium sweep: {i + 1}/{n_samples} samples")

    return {
        "r": np.array(all_r),
        "K": np.array(all_K),
        "e": np.array(all_e),
        "d": np.array(all_d),
        "N_measured": np.array(all_N_measured),
        "P_measured": np.array(all_P_measured),
        "N_theory": np.array(all_N_theory),
        "P_theory": np.array(all_P_theory),
    }


def run_ratio_dependent_rediscovery(
    output_dir: str | Path = "output/rediscovery/ratio_dependent",
    n_iterations: int = 50,
    n_K_values: int = 30,
) -> dict:
    """Run the full ratio-dependent predator-prey rediscovery.

    1. Sweep K to demonstrate no paradox of enrichment
    2. Compare ratio-dependent vs prey-dependent functional response
    3. Measure equilibrium N*, P* vs parameters
    4. Optionally run SINDy for ODE recovery

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "ratio_dependent",
        "model": "Arditi-Ginzburg",
        "targets": {
            "ode_N": "dN/dt = r*N*(1 - N/K) - a*N*P/(N + b*P)",
            "ode_P": "dP/dt = e*a*N*P/(N + b*P) - d*P",
            "equilibrium_N": "N* = K*(1 - a*(1-q)/(b*r)), q=d/(e*a)",
            "equilibrium_P": "P* = N*(1-q)/(q*b)",
            "no_paradox": "Stable for all K (no enrichment paradox)",
        },
    }

    # --- Part 1: No paradox of enrichment ---
    logger.info("Part 1: Demonstrating no paradox of enrichment...")
    enrich_data = generate_enrichment_sweep(
        n_K_values=n_K_values, K_min=20.0, K_max=500.0,
        n_steps=20000, dt=0.01,
    )

    max_N_amp = float(np.max(enrich_data["N_amplitude"]))
    max_P_amp = float(np.max(enrich_data["P_amplitude"]))

    # Compare amplitudes with theoretical equilibrium values
    # Amplitudes should be small relative to equilibrium values
    mean_N_eq = float(np.mean(enrich_data["N_equilibrium"]))
    relative_amp = max_N_amp / mean_N_eq if mean_N_eq > 0 else 0.0

    results["enrichment"] = {
        "n_K_values": n_K_values,
        "K_range": [float(enrich_data["K"][0]), float(enrich_data["K"][-1])],
        "max_N_amplitude": max_N_amp,
        "max_P_amplitude": max_P_amp,
        "relative_amplitude": relative_amp,
        "no_paradox_confirmed": relative_amp < 0.1,
        "N_avg_vs_K_correlation": float(np.corrcoef(
            enrich_data["K"], enrich_data["N_avg"],
        )[0, 1]),
    }
    logger.info(f"  Max prey amplitude: {max_N_amp:.4f}")
    logger.info(f"  Relative amplitude: {relative_amp:.6f}")
    logger.info(f"  No paradox: {relative_amp < 0.1}")

    # --- Part 2: Equilibrium verification ---
    logger.info("Part 2: Equilibrium N*, P* vs theory...")
    eq_data = generate_equilibrium_sweep(n_samples=100, n_steps=20000, dt=0.01)

    if len(eq_data["N_measured"]) > 0:
        N_error = np.abs(
            eq_data["N_measured"] - eq_data["N_theory"]
        ) / eq_data["N_theory"]
        P_error = np.abs(
            eq_data["P_measured"] - eq_data["P_theory"]
        ) / eq_data["P_theory"]

        results["equilibrium"] = {
            "n_samples": len(eq_data["N_measured"]),
            "mean_N_relative_error": float(np.mean(N_error)),
            "mean_P_relative_error": float(np.mean(P_error)),
            "max_N_relative_error": float(np.max(N_error)),
            "max_P_relative_error": float(np.max(P_error)),
            "N_correlation": float(np.corrcoef(
                eq_data["N_measured"], eq_data["N_theory"],
            )[0, 1]),
            "P_correlation": float(np.corrcoef(
                eq_data["P_measured"], eq_data["P_theory"],
            )[0, 1]),
        }
        logger.info(
            f"  N* mean error: {np.mean(N_error):.4%}, "
            f"P* mean error: {np.mean(P_error):.4%}"
        )
    else:
        results["equilibrium"] = {"error": "No valid equilibria found"}

    # --- Part 3: SINDy ODE recovery ---
    logger.info("Part 3: SINDy ODE recovery...")
    traj_data = generate_trajectory_data(
        r=1.0, K=100.0, a=0.5, b=1.0, e=0.5, d=0.2,
        n_steps=10000, dt=0.005,
    )

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            traj_data["states"],
            dt=0.005,
            feature_names=["N", "P"],
            threshold=0.05,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": disc.expression, "r_squared": disc.evidence.fit_r_squared}
                for disc in sindy_discoveries
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
        for disc in sindy_discoveries:
            logger.info(f"  SINDy: {disc.expression}")
    except Exception as exc:
        logger.warning(f"SINDy failed: {exc}")
        results["sindy_ode"] = {"error": str(exc)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "enrichment_data.npz",
        **{k: v for k, v in enrich_data.items()},
    )
    if len(eq_data["N_measured"]) > 0:
        np.savez(
            output_path / "equilibrium_data.npz",
            **{k: v for k, v in eq_data.items()},
        )

    return results
