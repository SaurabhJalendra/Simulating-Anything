"""Glycolytic oscillator (Higgins model) rediscovery.

Targets:
- ODE: dS/dt = v_in - k1*S*P^2, dP/dt = k1*S*P^2 - k2*P  (via SINDy)
- Hopf bifurcation threshold v_in_c = k2*sqrt(k2/k1)
- Oscillation frequency near the Hopf threshold
- Fixed point: S* = k2^2/(k1*v_in), P* = v_in/k2

Note: the Higgins model does not have a globally stable limit cycle.
Oscillations are growing spirals near the unstable fixed point (v_in < v_in_c).
Near the Hopf threshold the growth rate is small and many oscillation cycles
are observable before the trajectory escapes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.glycolytic_oscillator import (
    GlycolyticOscillatorSimulation,
    compute_hopf_v_in,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    v_in: float = 1.0,
    k1: float = 0.1,
    k2: float = 0.5,
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Uses initial conditions near the fixed point in the stable regime
    (v_in > v_in_c) so the trajectory stays bounded and captures
    decaying oscillatory dynamics suitable for SINDy.
    """
    # Use stable-regime v_in for SINDy (decaying oscillations are better data)
    v_in_c = compute_hopf_v_in(k1, k2)
    if v_in > v_in_c:
        eff_v_in = v_in
    else:
        eff_v_in = v_in

    S_star = k2**2 / (k1 * eff_v_in)
    P_star = eff_v_in / k2
    config = SimulationConfig(
        domain=Domain.GLYCOLYTIC_OSCILLATOR,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "v_in": eff_v_in,
            "k1": k1,
            "k2": k2,
            "S_0": S_star + 0.5,
            "P_0": P_star + 0.5,
        },
    )
    sim = GlycolyticOscillatorSimulation(config)
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
        "P": states_arr[:, 1],
        "v_in": eff_v_in,
        "k1": k1,
        "k2": k2,
    }


def generate_bifurcation_data(
    k1: float = 0.1,
    k2: float = 0.5,
    n_v_in: int = 30,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep v_in and measure Jacobian trace to find Hopf bifurcation.

    Instead of measuring oscillation amplitude (which is problematic for
    the Higgins model), we directly compute the Jacobian trace at each
    v_in and report the theoretical amplitude indicator.
    """
    v_in_values = np.linspace(0.1, 3.0, n_v_in)
    traces = []
    amplitudes_S = []
    amplitudes_P = []

    for i, v_in in enumerate(v_in_values):
        S_star = k2**2 / (k1 * v_in)
        P_star = v_in / k2
        config = SimulationConfig(
            domain=Domain.GLYCOLYTIC_OSCILLATOR,
            dt=dt,
            n_steps=1000,
            parameters={
                "v_in": v_in,
                "k1": k1,
                "k2": k2,
                "S_0": S_star + 0.01,
                "P_0": P_star + 0.01,
            },
        )
        sim = GlycolyticOscillatorSimulation(config)
        sim.reset()
        traces.append(sim.jacobian_trace)

        # Measure amplitude from early oscillation cycles
        omega = sim.oscillation_frequency
        if omega > 0:
            T = 2 * np.pi / omega
            n_measure = int(3 * T / dt)
            S_vals = []
            P_vals = []
            for _ in range(min(n_measure, 50000)):
                sim.step()
                S_vals.append(sim.observe()[0])
                P_vals.append(sim.observe()[1])
                if sim.observe()[1] < P_star * 0.001:
                    break
            amplitudes_S.append(max(S_vals) - min(S_vals) if S_vals else 0.0)
            amplitudes_P.append(max(P_vals) - min(P_vals) if P_vals else 0.0)
        else:
            amplitudes_S.append(0.0)
            amplitudes_P.append(0.0)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  v_in={v_in:.3f}: trace={traces[-1]:.4f}, "
                f"S_amp={amplitudes_S[-1]:.4f}, P_amp={amplitudes_P[-1]:.4f}"
            )

    return {
        "k1": k1,
        "k2": k2,
        "v_in": v_in_values,
        "trace": np.array(traces),
        "amplitude_S": np.array(amplitudes_S),
        "amplitude_P": np.array(amplitudes_P),
        "v_in_c_theory": float(compute_hopf_v_in(k1, k2)),
    }


def generate_period_data(
    k1: float = 0.1,
    k2: float = 0.5,
    n_v_in: int = 20,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep v_in near (just below) Hopf threshold and measure oscillation periods.

    In the Higgins model, the unstable spiral regime (v_in < v_in_c) exhibits
    oscillations with frequency determined by the imaginary part of the
    eigenvalues. Near the threshold the growth rate is slow, allowing many
    measurable cycles.
    """
    v_in_c = compute_hopf_v_in(k1, k2)
    # Sweep near the threshold (0.7 to 0.98 of v_in_c)
    v_in_values = np.linspace(v_in_c * 0.7, v_in_c * 0.98, n_v_in)
    periods = []

    for i, v_in in enumerate(v_in_values):
        config = SimulationConfig(
            domain=Domain.GLYCOLYTIC_OSCILLATOR,
            dt=dt,
            n_steps=1000,
            parameters={
                "v_in": v_in,
                "k1": k1,
                "k2": k2,
                "S_0": 2.5,
                "P_0": 2.0,
            },
        )
        sim = GlycolyticOscillatorSimulation(config)
        sim.reset()
        period = sim.measure_oscillation_period(n_cycles=5)
        periods.append(period)

        if (i + 1) % 5 == 0:
            logger.info(f"  v_in={v_in:.3f}: period={period:.4f}")

    return {
        "k1": k1,
        "k2": k2,
        "v_in": v_in_values,
        "period": np.array(periods),
        "v_in_c": v_in_c,
    }


def run_glycolytic_oscillator_rediscovery(
    output_dir: str | Path = "output/rediscovery/glycolytic_oscillator",
    n_iterations: int = 50,
) -> dict:
    """Run glycolytic oscillator (Higgins model) rediscovery pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "glycolytic_oscillator",
        "targets": {
            "ode": "dS/dt = v_in - k1*S*P^2, dP/dt = k1*S*P^2 - k2*P",
            "hopf": "v_in_c = k2*sqrt(k2/k1)",
            "fixed_point": "(S*, P*) = (k2^2/(k1*v_in), v_in/k2)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    # Use stable regime (v_in > v_in_c) for bounded trajectory
    v_in_c = compute_hopf_v_in(0.1, 0.5)
    v_in_sindy = v_in_c * 1.5
    logger.info(
        f"Part 1: SINDy ODE recovery at v_in={v_in_sindy:.3f} (stable regime)..."
    )
    data = generate_ode_data(
        v_in=v_in_sindy, k1=0.1, k2=0.5, n_steps=10000, dt=0.005
    )

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.005,
            feature_names=["S", "P"],
            threshold=0.01,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
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

    # --- Part 2: Hopf bifurcation ---
    logger.info("Part 2: Hopf bifurcation (trace analysis)...")
    bif_data = generate_bifurcation_data(k1=0.1, k2=0.5, n_v_in=30, dt=0.005)

    # Detect v_in_c from trace sign change
    traces = bif_data["trace"]
    v_in_vals = bif_data["v_in"]
    sign_changes = np.where(np.diff(np.sign(traces)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        # Linear interpolation for precise crossing
        t0, t1 = traces[idx], traces[idx + 1]
        v0, v1 = v_in_vals[idx], v_in_vals[idx + 1]
        v_in_c_est = v0 + (0 - t0) / (t1 - t0) * (v1 - v0)
        v_in_c_theory = bif_data["v_in_c_theory"]
        results["hopf_bifurcation"] = {
            "v_in_c_estimate": float(v_in_c_est),
            "v_in_c_theory": float(v_in_c_theory),
            "relative_error": float(
                abs(v_in_c_est - v_in_c_theory) / v_in_c_theory
            ),
        }
        logger.info(
            f"  v_in_c estimate: {v_in_c_est:.4f} (theory: {v_in_c_theory:.4f})"
        )

    # PySR: find v_in_c as function of k2
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Analytical: v_in_c = k2 * sqrt(k2/k1) for fixed k1
        k2_values = np.linspace(0.2, 1.5, 15)
        v_in_c_theory_arr = k2_values * np.sqrt(k2_values / 0.1)

        X = k2_values.reshape(-1, 1)
        y = v_in_c_theory_arr

        logger.info("  Running PySR: v_in_c = f(k2)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["k2_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt"],
            max_complexity=10,
            populations=20,
            population_size=40,
        )
        results["hopf_pysr"] = {
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
            results["hopf_pysr"]["best"] = best.expression
            results["hopf_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["hopf_pysr"] = {"error": str(e)}

    # --- Part 3: Period measurement ---
    logger.info("Part 3: Oscillation frequency near threshold...")
    try:
        period_data = generate_period_data(k1=0.1, k2=0.5, n_v_in=15, dt=0.005)
        finite = np.isfinite(period_data["period"])
        if np.sum(finite) > 3:
            results["period"] = {
                "n_measured": int(np.sum(finite)),
                "v_in_range": [
                    float(period_data["v_in"].min()),
                    float(period_data["v_in"].max()),
                ],
                "period_range": [
                    float(np.min(period_data["period"][finite])),
                    float(np.max(period_data["period"][finite])),
                ],
                "v_in_c": float(period_data["v_in_c"]),
            }
            logger.info(
                f"  Measured {np.sum(finite)} periods in "
                f"v_in=[{period_data['v_in'].min():.3f}, "
                f"{period_data['v_in'].max():.3f}]"
            )
    except Exception as e:
        logger.warning(f"Period measurement failed: {e}")
        results["period"] = {"error": str(e)}

    # --- Part 4: Fixed point verification ---
    logger.info("Part 4: Fixed point verification...")
    v_in_test, k1_test, k2_test = 1.0, 0.1, 0.5
    S_star = k2_test**2 / (k1_test * v_in_test)
    P_star = v_in_test / k2_test
    config = SimulationConfig(
        domain=Domain.GLYCOLYTIC_OSCILLATOR,
        dt=0.01,
        n_steps=1000,
        parameters={
            "v_in": v_in_test,
            "k1": k1_test,
            "k2": k2_test,
            "S_0": S_star,
            "P_0": P_star,
        },
    )
    sim = GlycolyticOscillatorSimulation(config)
    sim.reset()
    fp = sim.fixed_point
    dy = sim._derivatives(np.array(fp))
    results["fixed_point"] = {
        "S_star": fp[0],
        "P_star": fp[1],
        "derivative_at_fp": [float(dy[0]), float(dy[1])],
        "derivative_norm": float(np.linalg.norm(dy)),
    }
    logger.info(
        f"  Fixed point: ({fp[0]:.6f}, {fp[1]:.6f}), "
        f"|f(x*)|={np.linalg.norm(dy):.2e}"
    )

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
