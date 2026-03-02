"""Beddington-DeAngelis predator-prey rediscovery.

Targets:
- ODE recovery via SINDy:
    dN/dt = r*N*(1 - N/K) - a*N*P / (1 + b*N + c*P)
    dP/dt = e*a*N*P / (1 + b*N + c*P) - d*P
- Stabilizing effect of predator interference c
- Comparison with Holling II (c=0) showing paradox of enrichment suppression
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.beddington_deangelis import (
    BeddingtonDeAngelisSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_DOMAIN = Domain.BEDDINGTON_DEANGELIS


def generate_sweep_data(
    n_samples: int = 200,
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate BD trajectories with varied parameters for PySR analysis.

    Sweeps across r, K, a, b, c, e, d and records equilibrium properties:
    - Time-averaged prey and predator populations
    - Oscillation amplitude (to detect limit cycles vs stable spirals)
    - Functional response at equilibrium

    Returns dict with parameter arrays and measured quantities.
    """
    rng = np.random.default_rng(42)

    all_r = []
    all_K = []
    all_a = []
    all_b = []
    all_c = []
    all_e = []
    all_d = []
    all_N_avg = []
    all_P_avg = []
    all_N_amp = []
    all_P_amp = []
    all_fr_avg = []

    for i in range(n_samples):
        r = rng.uniform(0.5, 2.0)
        K = rng.uniform(5.0, 20.0)
        a = rng.uniform(0.3, 2.0)
        b = rng.uniform(0.1, 1.0)
        c = rng.uniform(0.0, 2.0)
        e = rng.uniform(0.2, 0.8)
        d = rng.uniform(0.1, 0.6)

        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": r, "K": K, "a": a, "b": b, "c": c,
                "e": e, "d": d, "N_0": K / 2, "P_0": 1.0,
            },
        )

        sim = BeddingtonDeAngelisSimulation(config)
        sim.reset()

        states = []
        for _ in range(n_steps):
            sim.step()
            states.append(sim.observe().copy())

        trajectory = np.array(states)

        # Use second half to avoid transients
        half = n_steps // 2
        N_half = trajectory[half:, 0]
        P_half = trajectory[half:, 1]

        # Compute average functional response in second half
        fr_values = []
        for j in range(half, len(trajectory)):
            Nj, Pj = trajectory[j]
            if 1.0 + b * Nj + c * Pj > 0:
                fr_values.append(a * Nj / (1.0 + b * Nj + c * Pj))
        fr_avg = float(np.mean(fr_values)) if fr_values else 0.0

        all_r.append(r)
        all_K.append(K)
        all_a.append(a)
        all_b.append(b)
        all_c.append(c)
        all_e.append(e)
        all_d.append(d)
        all_N_avg.append(float(np.mean(N_half)))
        all_P_avg.append(float(np.mean(P_half)))
        all_N_amp.append(float(np.max(N_half) - np.min(N_half)))
        all_P_amp.append(float(np.max(P_half) - np.min(P_half)))
        all_fr_avg.append(fr_avg)

        if (i + 1) % 50 == 0:
            logger.info(f"  Sweep: {i + 1}/{n_samples} samples")

    return {
        "r": np.array(all_r),
        "K": np.array(all_K),
        "a": np.array(all_a),
        "b": np.array(all_b),
        "c": np.array(all_c),
        "e": np.array(all_e),
        "d": np.array(all_d),
        "N_avg": np.array(all_N_avg),
        "P_avg": np.array(all_P_avg),
        "N_amplitude": np.array(all_N_amp),
        "P_amplitude": np.array(all_P_amp),
        "fr_avg": np.array(all_fr_avg),
    }


def generate_ode_data(
    r: float = 1.0,
    K: float = 10.0,
    a: float = 1.0,
    b: float = 0.5,
    c: float = 0.5,
    e: float = 0.5,
    d: float = 0.3,
    N_0: float = 5.0,
    P_0: float = 2.0,
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
            "r": r, "K": K, "a": a, "b": b, "c": c,
            "e": e, "d": d, "N_0": N_0, "P_0": P_0,
        },
    )

    sim = BeddingtonDeAngelisSimulation(config)
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
        "r": r, "K": K, "a": a, "b": b, "c": c, "e": e, "d": d,
    }


def generate_interference_sweep(
    n_c_values: int = 30,
    c_min: float = 0.0,
    c_max: float = 3.0,
    n_steps: int = 20000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep predator interference parameter c to detect stabilization.

    For each c, runs a long trajectory and measures oscillation amplitude.
    When c=0 (Holling II), expect limit cycles; for large c, stable equilibrium.

    Returns dict with c values and measured amplitudes.
    """
    c_values = np.linspace(c_min, c_max, n_c_values)

    all_c = []
    all_N_avg = []
    all_P_avg = []
    all_N_amp = []
    all_P_amp = []

    for i, c_val in enumerate(c_values):
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": 1.0, "K": 10.0, "a": 1.0, "b": 0.5,
                "c": float(c_val), "e": 0.5, "d": 0.3,
                "N_0": 5.0, "P_0": 2.0,
            },
        )

        sim = BeddingtonDeAngelisSimulation(config)
        sim.reset()

        states = []
        for _ in range(n_steps):
            sim.step()
            states.append(sim.observe().copy())

        trajectory = np.array(states)

        # Use second half
        half = n_steps // 2
        N_half = trajectory[half:, 0]
        P_half = trajectory[half:, 1]

        all_c.append(float(c_val))
        all_N_avg.append(float(np.mean(N_half)))
        all_P_avg.append(float(np.mean(P_half)))
        all_N_amp.append(float(np.max(N_half) - np.min(N_half)))
        all_P_amp.append(float(np.max(P_half) - np.min(P_half)))

        if (i + 1) % 10 == 0:
            logger.info(f"  Interference sweep: {i + 1}/{n_c_values}")

    return {
        "c": np.array(all_c),
        "N_avg": np.array(all_N_avg),
        "P_avg": np.array(all_P_avg),
        "N_amplitude": np.array(all_N_amp),
        "P_amplitude": np.array(all_P_amp),
    }


def run_beddington_deangelis_rediscovery(
    output_dir: str | Path = "output/rediscovery/beddington_deangelis",
    n_iterations: int = 40,
    n_samples: int = 200,
) -> dict:
    """Run the full Beddington-DeAngelis rediscovery.

    1. Generate a trajectory for SINDy ODE recovery
    2. Run SINDy to recover the coupled ODEs
    3. Sweep c to detect stabilization by predator interference
    4. Run PySR on sweep data to find functional relationships

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "beddington_deangelis",
        "targets": {
            "ode_N": "dN/dt = r*N*(1-N/K) - a*N*P/(1+b*N+c*P)",
            "ode_P": "dP/dt = e*a*N*P/(1+b*N+c*P) - d*P",
            "interference_stabilization": "c>0 stabilizes coexistence",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery for Beddington-DeAngelis model...")
    data = generate_ode_data(n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
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

    # --- Part 2: Interference stabilization sweep ---
    logger.info("Part 2: Sweeping predator interference c...")
    interf_data = generate_interference_sweep(
        n_c_values=30, c_min=0.0, c_max=3.0,
        n_steps=20000, dt=0.01,
    )

    # Detect stabilization: find c where amplitude drops below threshold
    amp_threshold = 0.1
    c_stabilize = None
    for j, amp in enumerate(interf_data["N_amplitude"]):
        if amp < amp_threshold and interf_data["c"][j] > 0.01:
            c_stabilize = float(interf_data["c"][j])
            break

    results["interference_sweep"] = {
        "n_c_values": 30,
        "c_range": [float(interf_data["c"][0]), float(interf_data["c"][-1])],
        "c_stabilize": c_stabilize,
        "max_N_amplitude": float(np.max(interf_data["N_amplitude"])),
        "holling2_amplitude": float(interf_data["N_amplitude"][0]),
    }

    logger.info(f"  Holling II (c=0) amplitude = {interf_data['N_amplitude'][0]:.4f}")
    logger.info(f"  Stabilization at c ~ {c_stabilize}")

    # --- Part 3: PySR for parameter relationships ---
    logger.info("Part 3: Running PySR on parameter sweep...")
    sweep_data = generate_sweep_data(
        n_samples=n_samples, n_steps=10000, dt=0.01,
    )

    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Predict prey amplitude as function of key parameters
        # Filter to cases with coexistence (positive averages)
        mask = (sweep_data["N_avg"] > 0.1) & (sweep_data["P_avg"] > 0.1)
        X_params = np.column_stack([
            sweep_data["c"][mask],
            sweep_data["K"][mask],
            sweep_data["a"][mask],
        ])

        pysr_discoveries = run_symbolic_regression(
            X_params,
            sweep_data["N_amplitude"][mask],
            variable_names=["c_", "K_", "a_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square", "exp"],
            max_complexity=15,
            populations=20,
            population_size=40,
        )

        results["pysr_amplitude"] = {
            "n_coexistence": int(mask.sum()),
            "n_discoveries": len(pysr_discoveries),
            "discoveries": [
                {"expression": disc.expression, "r_squared": disc.evidence.fit_r_squared}
                for disc in pysr_discoveries[:5]
            ],
        }
        if pysr_discoveries:
            best = pysr_discoveries[0]
            results["pysr_amplitude"]["best"] = best.expression
            results["pysr_amplitude"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  PySR amplitude: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as exc:
        logger.warning(f"PySR failed: {exc}")
        results["pysr_amplitude"] = {"error": str(exc)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "ode_data.npz",
        states=data["states"],
    )
    np.savez(
        output_path / "interference_data.npz",
        **{k: v for k, v in interf_data.items()},
    )
    np.savez(
        output_path / "sweep_data.npz",
        **{k: v for k, v in sweep_data.items()},
    )

    return results
