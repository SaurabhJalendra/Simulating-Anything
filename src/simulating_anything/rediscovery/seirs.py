"""SEIRS epidemic model rediscovery.

Targets:
- R0 = beta / gamma (basic reproduction number)
- Endemic equilibrium I* when R0 > 1 (sustained by waning immunity)
- Damped oscillations in I(t) toward endemic state
- Population conservation S + E + I + R = N
- SINDy recovery of SEIRS ODEs
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.seirs import SEIRSSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_seirs_sweep_data(
    n_samples: int = 200,
    n_steps: int = 5000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate SEIRS trajectories with varied parameters to study R0 and endemic state.

    For each parameter set, run to completion and record:
    - Peak infectious fraction
    - Endemic (final) infectious fraction
    - Final epidemic size (fraction that left S at peak)
    - Oscillation period (if detectable)
    - Conservation error
    """
    rng = np.random.default_rng(42)

    all_beta = []
    all_sigma = []
    all_gamma = []
    all_xi = []
    all_R0 = []
    all_peak_I = []
    all_endemic_I = []
    all_conservation_error = []
    all_oscillation_detected = []

    N0 = 1000.0

    for i in range(n_samples):
        beta = rng.uniform(0.1, 0.8)
        sigma = rng.uniform(0.05, 0.5)
        gamma = rng.uniform(0.02, 0.3)
        xi = rng.uniform(0.005, 0.2)
        S_0 = N0 * rng.uniform(0.95, 1.0)
        E_0 = 0.0
        I_0 = N0 - S_0
        R_0_init = 0.0

        config = SimulationConfig(
            domain=Domain.SEIRS,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "beta": beta,
                "sigma": sigma,
                "gamma": gamma,
                "xi": xi,
                "N0": N0,
                "S_0": S_0,
                "E_0": E_0,
                "I_0": I_0,
                "R_0_init": R_0_init,
            },
        )
        sim = SEIRSSimulation(config)
        sim.reset()

        peak_I = 0.0
        max_conservation_err = 0.0
        I_history = []
        for step in range(n_steps):
            state = sim.step()
            if state[2] > peak_I:
                peak_I = state[2]
            cons_err = abs(state.sum() - N0)
            if cons_err > max_conservation_err:
                max_conservation_err = cons_err
            # Record I values in second half for oscillation detection
            if step > n_steps // 2:
                I_history.append(state[2])

        final_I = sim.observe()[2]

        # Detect oscillations: check for sign changes in dI
        osc_detected = False
        if len(I_history) > 10:
            dI = np.diff(I_history)
            sign_changes = np.sum(np.diff(np.sign(dI)) != 0)
            osc_detected = sign_changes >= 4

        all_beta.append(beta)
        all_sigma.append(sigma)
        all_gamma.append(gamma)
        all_xi.append(xi)
        all_R0.append(beta / gamma)
        all_peak_I.append(peak_I / N0)
        all_endemic_I.append(final_I / N0)
        all_conservation_error.append(max_conservation_err)
        all_oscillation_detected.append(float(osc_detected))

        if (i + 1) % 50 == 0:
            logger.info(f"  Generated {i + 1}/{n_samples} SEIRS trajectories")

    return {
        "beta": np.array(all_beta),
        "sigma": np.array(all_sigma),
        "gamma": np.array(all_gamma),
        "xi": np.array(all_xi),
        "R0": np.array(all_R0),
        "peak_I": np.array(all_peak_I),
        "endemic_I": np.array(all_endemic_I),
        "conservation_error": np.array(all_conservation_error),
        "oscillation_detected": np.array(all_oscillation_detected),
    }


def generate_seirs_ode_data(
    n_steps: int = 3000,
    dt: float = 0.1,
) -> dict[str, np.ndarray | float]:
    """Generate a single SEIRS trajectory for SINDy ODE recovery."""
    N0 = 1000.0
    config = SimulationConfig(
        domain=Domain.SEIRS,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "beta": 0.5,
            "sigma": 0.2,
            "gamma": 0.1,
            "xi": 0.05,
            "N0": N0,
            "S_0": 990.0,
            "E_0": 5.0,
            "I_0": 5.0,
            "R_0_init": 0.0,
        },
    )
    sim = SEIRSSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "beta": 0.5,
        "sigma": 0.2,
        "gamma": 0.1,
        "xi": 0.05,
        "N0": N0,
    }


def generate_seirs_conservation_data(
    n_steps: int = 2000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate conservation check data across multiple parameter sets."""
    rng = np.random.default_rng(123)
    N0 = 1000.0
    max_errors = []
    params_list = []

    for _ in range(20):
        beta = rng.uniform(0.1, 0.8)
        gamma = rng.uniform(0.05, 0.3)
        sigma = rng.uniform(0.05, 0.5)
        xi = rng.uniform(0.01, 0.2)

        config = SimulationConfig(
            domain=Domain.SEIRS,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "beta": beta, "sigma": sigma, "gamma": gamma,
                "xi": xi, "N0": N0,
                "S_0": 990.0, "E_0": 5.0, "I_0": 5.0, "R_0_init": 0.0,
            },
        )
        sim = SEIRSSimulation(config)
        sim.reset()
        max_err = 0.0
        for _ in range(n_steps):
            state = sim.step()
            err = abs(state.sum() - N0)
            if err > max_err:
                max_err = err
        max_errors.append(max_err)
        params_list.append([beta, sigma, gamma, xi])

    return {
        "max_conservation_errors": np.array(max_errors),
        "parameters": np.array(params_list),
    }


def generate_seirs_oscillation_data(
    n_steps: int = 10000,
    dt: float = 0.1,
) -> dict[str, np.ndarray | float]:
    """Generate a long trajectory to observe damped oscillations toward endemic state.

    Uses parameters that produce clear oscillatory dynamics.
    """
    N0 = 1000.0
    config = SimulationConfig(
        domain=Domain.SEIRS,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "beta": 0.5,
            "sigma": 0.2,
            "gamma": 0.1,
            "xi": 0.05,
            "N0": N0,
            "S_0": 990.0,
            "E_0": 5.0,
            "I_0": 5.0,
            "R_0_init": 0.0,
        },
    )
    sim = SEIRSSimulation(config)
    sim.reset()

    I_values = [sim.observe()[2]]
    times = [0.0]
    for step in range(n_steps):
        state = sim.step()
        I_values.append(state[2])
        times.append((step + 1) * dt)

    # Detect oscillation peaks
    I_arr = np.array(I_values)
    peaks = []
    for j in range(1, len(I_arr) - 1):
        if I_arr[j] > I_arr[j - 1] and I_arr[j] > I_arr[j + 1]:
            peaks.append(j)

    # Compute periods between consecutive peaks
    periods = []
    for j in range(1, len(peaks)):
        periods.append((peaks[j] - peaks[j - 1]) * dt)

    return {
        "I_values": I_arr,
        "times": np.array(times),
        "peak_indices": np.array(peaks) if peaks else np.array([], dtype=int),
        "periods": np.array(periods) if periods else np.array([]),
        "endemic_I_theory": float(
            sim.endemic_equilibrium()[2] if sim.endemic_equilibrium() is not None else 0.0
        ),
    }


def run_seirs_rediscovery(
    output_dir: str | Path = "output/rediscovery/seirs",
    n_iterations: int = 50,
    n_samples: int = 200,
) -> dict:
    """Run the full SEIRS epidemic rediscovery.

    1. Sweep beta/gamma/xi parameter space
    2. Measure endemic equilibrium I* vs R0
    3. Measure oscillation period (damped oscillations to endemic state)
    4. Verify S+E+I+R=N conservation
    5. Run PySR to find R0 = beta/gamma
    6. Run SINDy to recover SEIRS ODEs
    """
    from simulating_anything.analysis.symbolic_regression import (
        run_symbolic_regression,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "seirs",
        "targets": {
            "R0": "beta / gamma",
            "ode_S": "dS/dt = -beta * S * I / N + xi * R",
            "ode_E": "dE/dt = beta * S * I / N - sigma * E",
            "ode_I": "dI/dt = sigma * E - gamma * I",
            "ode_R": "dR/dt = gamma * I - xi * R",
        },
    }

    # --- Part 1: R0 rediscovery via PySR ---
    logger.info("Part 1: Generating SEIRS parameter sweep data...")
    data = generate_seirs_sweep_data(n_samples=n_samples, n_steps=5000, dt=0.1)

    # Filter to epidemics that actually occurred (R0 > 1)
    mask = data["R0"] > 1.0
    X_filtered = np.column_stack([data["beta"][mask], data["gamma"][mask]])

    logger.info(f"  {mask.sum()}/{len(mask)} epidemics with R0 > 1")
    logger.info("  Running PySR for R0 = f(beta, gamma)...")

    r0_discoveries = run_symbolic_regression(
        X_filtered,
        data["R0"][mask],
        variable_names=["b_", "g_"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        max_complexity=10,
        populations=20,
        population_size=40,
    )

    results["R0_pysr"] = {
        "n_epidemics": int(mask.sum()),
        "n_discoveries": len(r0_discoveries),
        "discoveries": [
            {
                "expression": d.expression,
                "r_squared": d.evidence.fit_r_squared,
            }
            for d in r0_discoveries[:5]
        ],
    }
    if r0_discoveries:
        best = r0_discoveries[0]
        results["R0_pysr"]["best"] = best.expression
        results["R0_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(
            f"  Best R0: {best.expression} "
            f"(R2={best.evidence.fit_r_squared:.6f})"
        )

    # --- Part 2: Endemic equilibrium I* vs R0 ---
    logger.info("Part 2: Endemic equilibrium analysis...")
    endemic_mask = mask & (data["endemic_I"] > 1e-4)
    results["endemic_equilibrium"] = {
        "n_endemic": int(endemic_mask.sum()),
        "mean_endemic_I": float(np.mean(data["endemic_I"][endemic_mask]))
        if endemic_mask.sum() > 0 else 0.0,
        "correlation_R0_endemic_I": float(np.corrcoef(
            data["R0"][endemic_mask], data["endemic_I"][endemic_mask]
        )[0, 1]) if endemic_mask.sum() > 2 else 0.0,
    }

    if endemic_mask.sum() > 10:
        logger.info("  Running PySR for endemic I* = f(beta, gamma, xi)...")
        X_endemic = np.column_stack([
            data["beta"][endemic_mask],
            data["gamma"][endemic_mask],
            data["xi"][endemic_mask],
        ])
        endemic_discoveries = run_symbolic_regression(
            X_endemic,
            data["endemic_I"][endemic_mask],
            variable_names=["b_", "g_", "x_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=[],
            max_complexity=15,
            populations=20,
            population_size=40,
        )
        results["endemic_I_pysr"] = {
            "n_discoveries": len(endemic_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in endemic_discoveries[:5]
            ],
        }
        if endemic_discoveries:
            best = endemic_discoveries[0]
            results["endemic_I_pysr"]["best"] = best.expression
            results["endemic_I_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best endemic I*: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )

    # --- Part 3: Oscillation period ---
    logger.info("Part 3: Oscillation analysis...")
    osc_data = generate_seirs_oscillation_data(n_steps=10000, dt=0.1)
    results["oscillation"] = {
        "n_peaks": len(osc_data["peak_indices"]),
        "periods": osc_data["periods"].tolist(),
        "endemic_I_theory": osc_data["endemic_I_theory"],
    }
    if len(osc_data["periods"]) > 0:
        results["oscillation"]["mean_period"] = float(np.mean(osc_data["periods"]))
        logger.info(
            f"  Detected {len(osc_data['peak_indices'])} oscillation peaks, "
            f"mean period = {np.mean(osc_data['periods']):.1f}"
        )

    # --- Part 4: Conservation verification ---
    logger.info("Part 4: Conservation verification...")
    cons_data = generate_seirs_conservation_data(n_steps=2000, dt=0.1)
    results["conservation"] = {
        "max_error": float(np.max(cons_data["max_conservation_errors"])),
        "mean_error": float(np.mean(cons_data["max_conservation_errors"])),
        "all_below_threshold": bool(
            np.all(cons_data["max_conservation_errors"] < 1e-4)
        ),
    }
    logger.info(
        f"  Max conservation error: {np.max(cons_data['max_conservation_errors']):.2e}"
    )

    # --- Part 5: SINDy ODE recovery ---
    logger.info("Part 5: SINDy ODE recovery...")
    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        ode_data = generate_seirs_ode_data(n_steps=3000, dt=0.1)
        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["S", "E", "I", "R"],
            threshold=0.01,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in sindy_discoveries
            ],
            "true_beta": ode_data["beta"],
            "true_sigma": ode_data["sigma"],
            "true_gamma": ode_data["gamma"],
            "true_xi": ode_data["xi"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "sweep_data.npz",
        **{k: v for k, v in data.items()},
    )

    return results
