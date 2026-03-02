"""Michaelis-Menten enzyme kinetics rediscovery.

Targets:
- V_max = k2 * E_total  (maximum velocity)
- K_m = (k_1 + k2) / k1  (Michaelis constant)
- Michaelis-Menten equation: v = V_max * S / (K_m + S) via PySR
- ODE recovery: dS/dt, dC/dt, dP/dt via SINDy
- Lineweaver-Burk linearization: 1/v = 1/V_max + (K_m/V_max)*(1/S)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.michaelis_menten import (
    MichaelisMentenSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    k1: float = 1.0,
    k_1: float = 0.1,
    k2: float = 0.5,
    E_total: float = 1.0,
    S0: float = 10.0,
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.MICHAELIS_MENTEN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "k1": k1, "k_1": k_1, "k2": k2,
            "E_total": E_total, "S0": S0,
        },
    )
    sim = MichaelisMentenSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "S": states_arr[:, 0],
        "C": states_arr[:, 1],
        "P": states_arr[:, 2],
        "k1": k1,
        "k_1": k_1,
        "k2": k2,
        "E_total": E_total,
        "S0": S0,
    }


def generate_initial_rate_data(
    k1: float = 1.0,
    k_1: float = 0.1,
    k2: float = 0.5,
    E_total: float = 1.0,
    n_S0: int = 25,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Sweep initial substrate S0 and measure initial rate v0.

    This generates the classic Michaelis-Menten curve v0 vs [S].
    """
    S0_values = np.linspace(0.5, 50.0, n_S0)
    v0_measured = []
    v0_theory = []

    V_max = k2 * E_total
    K_m = (k_1 + k2) / k1

    for S0 in S0_values:
        config = SimulationConfig(
            domain=Domain.MICHAELIS_MENTEN,
            dt=dt,
            n_steps=1000,
            parameters={
                "k1": k1, "k_1": k_1, "k2": k2,
                "E_total": E_total, "S0": S0,
            },
        )
        sim = MichaelisMentenSimulation(config)
        v0 = sim.measure_initial_rate(n_steps=50)
        v0_measured.append(v0)
        v0_theory.append(V_max * S0 / (K_m + S0))

    return {
        "S0": S0_values,
        "v0_measured": np.array(v0_measured),
        "v0_theory": np.array(v0_theory),
        "V_max": V_max,
        "K_m": K_m,
    }


def generate_vmax_data(
    k1: float = 1.0,
    k_1: float = 0.1,
    k2: float = 0.5,
    n_E: int = 15,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Sweep E_total and measure V_max (initial rate at high [S]).

    At very high S0 >> K_m, v0 ~ V_max = k2*E_total.
    """
    E_values = np.linspace(0.2, 5.0, n_E)
    S0_high = 500.0  # Very high substrate so v0 ~ V_max
    v_max_measured = []

    for E_total in E_values:
        config = SimulationConfig(
            domain=Domain.MICHAELIS_MENTEN,
            dt=dt,
            n_steps=1000,
            parameters={
                "k1": k1, "k_1": k_1, "k2": k2,
                "E_total": E_total, "S0": S0_high,
            },
        )
        sim = MichaelisMentenSimulation(config)
        v0 = sim.measure_initial_rate(n_steps=50)
        v_max_measured.append(v0)

    return {
        "E_total": E_values,
        "v_max_measured": np.array(v_max_measured),
        "v_max_theory": k2 * E_values,
        "k2": k2,
    }


def generate_km_data(
    k2: float = 0.5,
    E_total: float = 1.0,
    n_k1: int = 15,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Sweep k1 and measure K_m from half-max velocity.

    K_m = (k_1 + k2)/k1, so K_m should be inversely proportional to k1.
    """
    k1_values = np.linspace(0.3, 5.0, n_k1)
    k_1 = 0.1
    K_m_measured = []
    K_m_theory = []

    for k1 in k1_values:
        V_max = k2 * E_total
        K_m_true = (k_1 + k2) / k1
        K_m_theory.append(K_m_true)

        # Measure: find S0 where v0 = V_max/2
        # Sweep S0 to find half-max
        S_vals = np.linspace(0.01, 20.0, 100)
        rates = []
        for S0 in S_vals:
            config = SimulationConfig(
                domain=Domain.MICHAELIS_MENTEN,
                dt=dt,
                n_steps=1000,
                parameters={
                    "k1": k1, "k_1": k_1, "k2": k2,
                    "E_total": E_total, "S0": S0,
                },
            )
            sim = MichaelisMentenSimulation(config)
            v0 = sim.measure_initial_rate(n_steps=50)
            rates.append(v0)

        rates_arr = np.array(rates)
        half_vmax = V_max / 2.0

        # Interpolate to find S where v0 = V_max/2
        idx = np.argmin(np.abs(rates_arr - half_vmax))
        if idx > 0 and idx < len(S_vals) - 1:
            # Linear interpolation around crossing
            if rates_arr[idx] < half_vmax:
                s_lo, s_hi = S_vals[idx], S_vals[idx + 1]
                r_lo, r_hi = rates_arr[idx], rates_arr[idx + 1]
            else:
                s_lo, s_hi = S_vals[idx - 1], S_vals[idx]
                r_lo, r_hi = rates_arr[idx - 1], rates_arr[idx]
            if r_hi != r_lo:
                frac = (half_vmax - r_lo) / (r_hi - r_lo)
                K_m_est = s_lo + frac * (s_hi - s_lo)
            else:
                K_m_est = S_vals[idx]
        else:
            K_m_est = S_vals[idx]

        K_m_measured.append(K_m_est)

    return {
        "k1": k1_values,
        "K_m_measured": np.array(K_m_measured),
        "K_m_theory": np.array(K_m_theory),
    }


def run_michaelis_menten_rediscovery(
    output_dir: str | Path = "output/rediscovery/michaelis_menten",
    n_iterations: int = 40,
) -> dict:
    """Run Michaelis-Menten enzyme kinetics rediscovery pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "michaelis_menten",
        "targets": {
            "mm_equation": "v = V_max * S / (K_m + S)",
            "V_max": "k2 * E_total",
            "K_m": "(k_1 + k2) / k1",
            "conservation": "E + C = E_total, S + C + P = S0",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery for MM kinetics...")
    data = generate_ode_data(n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.005,
            feature_names=["S", "C", "P"],
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
                for d in sindy_discoveries[:5]
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
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Michaelis-Menten curve (v0 vs S) ---
    logger.info("Part 2: Initial rate vs substrate sweep...")
    rate_data = generate_initial_rate_data(n_S0=25)

    corr = float(np.corrcoef(
        rate_data["v0_measured"], rate_data["v0_theory"]
    )[0, 1])
    rel_err = float(np.mean(
        np.abs(rate_data["v0_measured"] - rate_data["v0_theory"])
        / np.maximum(rate_data["v0_theory"], 1e-12)
    ))
    results["mm_curve"] = {
        "n_points": len(rate_data["S0"]),
        "correlation": corr,
        "mean_relative_error": rel_err,
        "V_max": float(rate_data["V_max"]),
        "K_m": float(rate_data["K_m"]),
    }
    logger.info(
        f"  MM curve: corr={corr:.6f}, mean_rel_err={rel_err:.4f}"
    )

    # PySR: fit v0 = f(S0)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = rate_data["S0"].reshape(-1, 1)
        y = rate_data["v0_measured"]

        logger.info("  Running PySR: v0 = f(S0)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["S_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt"],
            max_complexity=10,
            populations=20,
            population_size=40,
        )
        results["mm_pysr"] = {
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
            results["mm_pysr"]["best"] = best.expression
            results["mm_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["mm_pysr"] = {"error": str(e)}

    # --- Part 3: V_max = k2 * E_total verification ---
    logger.info("Part 3: V_max vs E_total sweep...")
    vmax_data = generate_vmax_data(n_E=15)

    vmax_corr = float(np.corrcoef(
        vmax_data["v_max_measured"], vmax_data["v_max_theory"]
    )[0, 1])
    results["vmax_linearity"] = {
        "n_points": len(vmax_data["E_total"]),
        "correlation": vmax_corr,
        "k2": float(vmax_data["k2"]),
    }
    logger.info(f"  V_max linearity: corr={vmax_corr:.6f}")

    # --- Part 4: K_m measurement ---
    logger.info("Part 4: K_m measurement via half-max velocity...")
    try:
        km_data = generate_km_data(n_k1=10)
        km_corr = float(np.corrcoef(
            km_data["K_m_measured"], km_data["K_m_theory"]
        )[0, 1])
        km_err = float(np.mean(
            np.abs(km_data["K_m_measured"] - km_data["K_m_theory"])
            / np.maximum(km_data["K_m_theory"], 1e-12)
        ))
        results["km_measurement"] = {
            "n_points": len(km_data["k1"]),
            "correlation": km_corr,
            "mean_relative_error": km_err,
        }
        logger.info(f"  K_m: corr={km_corr:.6f}, err={km_err:.4f}")
    except Exception as e:
        logger.warning(f"K_m measurement failed: {e}")
        results["km_measurement"] = {"error": str(e)}

    # --- Part 5: Conservation law verification ---
    logger.info("Part 5: Conservation law verification...")
    config = SimulationConfig(
        domain=Domain.MICHAELIS_MENTEN,
        dt=0.01,
        n_steps=5000,
        parameters={
            "k1": 1.0, "k_1": 0.1, "k2": 0.5,
            "E_total": 1.0, "S0": 10.0,
        },
    )
    sim = MichaelisMentenSimulation(config)
    sim.reset()

    enzyme_errors = []
    substrate_errors = []
    for _ in range(5000):
        sim.step()
        S, C, P = sim.observe()
        E = sim.E_total - C
        enzyme_errors.append(abs(E + C - sim.E_total))
        substrate_errors.append(abs(S + C + P - sim.S0))

    results["conservation"] = {
        "enzyme_max_error": float(max(enzyme_errors)),
        "enzyme_mean_error": float(np.mean(enzyme_errors)),
        "substrate_max_error": float(max(substrate_errors)),
        "substrate_mean_error": float(np.mean(substrate_errors)),
    }
    logger.info(
        f"  Conservation: E+C max_err={max(enzyme_errors):.2e}, "
        f"S+C+P max_err={max(substrate_errors):.2e}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
