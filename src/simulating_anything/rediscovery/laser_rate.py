"""Semiconductor laser rate equations rediscovery.

Targets:
- ODE: dN/dt = P - gamma_N*N - g*(N-N_tr)*S,
       dS/dt = Gamma*g*(N-N_tr)*S - gamma_S*S + Gamma*beta*gamma_N*N  (via SINDy)
- Threshold pump: P_th = gamma_N*N_tr + gamma_S/(Gamma*g)
- Steady-state photon density: S_ss = Gamma*(P - P_th)/gamma_S
- Relaxation oscillation frequency: omega_r ~ sqrt(gamma_S * g * S_ss)
- L-I curve (light output vs pump current)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.laser_rate import LaserRateSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_LASER_DOMAIN = Domain.LASER_RATE

# Default semiconductor laser parameters
_DEFAULT_PARAMS = {
    "P": 2.0,
    "gamma_N": 1.0,
    "gamma_S": 100.0,
    "g": 1000.0,
    "N_tr": 0.5,
    "Gamma": 0.3,
    "beta": 1e-4,
    "N_0": 0.5,
    "S_0": 0.01,
}


def _make_sim(dt: float = 0.001, n_steps: int = 1000, **overrides) -> LaserRateSimulation:
    """Create a LaserRateSimulation with default parameters plus overrides."""
    params = dict(_DEFAULT_PARAMS)
    params.update(overrides)
    config = SimulationConfig(
        domain=_LASER_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )
    return LaserRateSimulation(config)


def generate_ode_data(
    n_steps: int = 10000,
    dt: float = 0.001,
    P: float = 2.0,
) -> dict[str, np.ndarray]:
    """Generate a single laser trajectory for SINDy ODE recovery.

    Uses above-threshold pump rate to capture full carrier-photon dynamics.
    """
    sim = _make_sim(dt=dt, n_steps=n_steps, P=P)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    states_arr = np.array(states)
    return {
        "states": states_arr,
        "dt": dt,
        "time": np.arange(n_steps + 1) * dt,
        "N": states_arr[:, 0],
        "S": states_arr[:, 1],
        "P": P,
    }


def generate_threshold_sweep(
    n_P: int = 30,
    dt: float = 0.001,
    n_settle: int = 50000,
) -> dict[str, np.ndarray]:
    """Sweep pump rate P and measure steady-state photon density.

    The threshold P_th = gamma_N*N_tr + gamma_S/(Gamma*g) should emerge
    as a sharp kink where S_ss transitions from near-zero to linear growth.
    """
    # Compute theoretical threshold for default parameters
    P_th_theory = (
        _DEFAULT_PARAMS["gamma_N"] * _DEFAULT_PARAMS["N_tr"]
        + _DEFAULT_PARAMS["gamma_S"]
        / (_DEFAULT_PARAMS["Gamma"] * _DEFAULT_PARAMS["g"])
    )

    P_values = np.linspace(0.1, P_th_theory * 2.5, n_P)
    S_steady = []
    N_steady = []

    for i, P in enumerate(P_values):
        sim = _make_sim(dt=dt, n_steps=100, P=P, N_0=0.5, S_0=0.001)
        sim.reset()

        for _ in range(n_settle):
            sim.step()

        N_val, S_val = sim.observe()
        N_steady.append(N_val)
        S_steady.append(S_val)

        if (i + 1) % 10 == 0:
            logger.info(f"  P={P:.3f}: N_ss={N_val:.6f}, S_ss={S_val:.6f}")

    return {
        "P": P_values,
        "N_steady": np.array(N_steady),
        "S_steady": np.array(S_steady),
        "P_th_theory": P_th_theory,
    }


def generate_li_curve(
    n_P: int = 40,
    dt: float = 0.001,
    n_settle: int = 50000,
) -> dict[str, np.ndarray]:
    """Generate the L-I curve (light output vs injection current).

    Above threshold, S_ss should scale linearly with P - P_th.
    """
    P_th_theory = (
        _DEFAULT_PARAMS["gamma_N"] * _DEFAULT_PARAMS["N_tr"]
        + _DEFAULT_PARAMS["gamma_S"]
        / (_DEFAULT_PARAMS["Gamma"] * _DEFAULT_PARAMS["g"])
    )

    P_values = np.linspace(P_th_theory * 0.5, P_th_theory * 3.0, n_P)
    light_output = []

    for P in P_values:
        sim = _make_sim(dt=dt, n_steps=100, P=P, N_0=0.5, S_0=0.001)
        sim.reset()

        for _ in range(n_settle):
            sim.step()

        _, S_val = sim.observe()
        light_output.append(S_val)

    return {
        "P": P_values,
        "S_steady": np.array(light_output),
        "P_th_theory": P_th_theory,
    }


def generate_relaxation_frequency_data(
    n_P: int = 20,
    dt: float = 0.0001,
    n_transient: int = 5000,
    n_measure: int = 30000,
) -> dict[str, np.ndarray]:
    """Measure relaxation oscillation frequency vs pump rate.

    Above threshold, the laser exhibits damped relaxation oscillations
    whose frequency scales as omega_r ~ sqrt(gamma_S * g * S_ss).
    """
    P_th_theory = (
        _DEFAULT_PARAMS["gamma_N"] * _DEFAULT_PARAMS["N_tr"]
        + _DEFAULT_PARAMS["gamma_S"]
        / (_DEFAULT_PARAMS["Gamma"] * _DEFAULT_PARAMS["g"])
    )

    # Only above threshold
    P_values = np.linspace(P_th_theory * 1.2, P_th_theory * 3.0, n_P)
    frequencies = []
    freq_theory = []

    for i, P in enumerate(P_values):
        sim = _make_sim(dt=dt, n_steps=100, P=P, N_0=0.5, S_0=0.001)
        sim.reset()

        # Run past transient
        for _ in range(n_transient):
            sim.step()

        # Collect photon density time series
        s_vals = []
        for _ in range(n_measure):
            sim.step()
            s_vals.append(sim.observe()[1])

        s_vals = np.array(s_vals)
        # Estimate frequency from FFT
        s_detrend = s_vals - np.mean(s_vals)
        if np.std(s_detrend) > 1e-12:
            fft_mag = np.abs(np.fft.rfft(s_detrend))
            freqs = np.fft.rfftfreq(len(s_detrend), d=dt)
            fft_mag[0] = 0  # Skip DC
            peak_idx = np.argmax(fft_mag)
            freq = freqs[peak_idx]
            frequencies.append(2 * np.pi * freq)
        else:
            frequencies.append(0.0)

        # Theoretical frequency
        S_ss_theory = max(
            0.0,
            _DEFAULT_PARAMS["Gamma"] * (P - P_th_theory) / _DEFAULT_PARAMS["gamma_S"],
        )
        omega_theory = np.sqrt(
            _DEFAULT_PARAMS["gamma_S"] * _DEFAULT_PARAMS["g"] * S_ss_theory
        )
        freq_theory.append(omega_theory)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  P={P:.3f}: omega_r_meas={frequencies[-1]:.2f}, "
                f"omega_r_theory={freq_theory[-1]:.2f}"
            )

    return {
        "P": P_values,
        "omega_r_measured": np.array(frequencies),
        "omega_r_theory": np.array(freq_theory),
        "P_th_theory": P_th_theory,
    }


def generate_parameter_sweep(
    n_samples: int = 20,
    dt: float = 0.001,
    n_settle: int = 50000,
) -> dict[str, np.ndarray]:
    """Sweep multiple parameters to study steady-state dependencies."""
    rng = np.random.default_rng(42)

    all_P = []
    all_gamma_S = []
    all_Gamma = []
    all_S_ss = []
    all_P_th = []

    for _ in range(n_samples):
        P = rng.uniform(1.0, 5.0)
        gamma_S = rng.uniform(50.0, 200.0)
        Gamma_val = rng.uniform(0.1, 0.5)

        sim = _make_sim(
            dt=dt, n_steps=100,
            P=P, gamma_S=gamma_S, Gamma=Gamma_val,
            N_0=0.5, S_0=0.001,
        )
        sim.reset()

        for _ in range(n_settle):
            sim.step()

        _, S_val = sim.observe()
        all_P.append(P)
        all_gamma_S.append(gamma_S)
        all_Gamma.append(Gamma_val)
        all_S_ss.append(S_val)
        all_P_th.append(sim.threshold_pump)

    return {
        "P": np.array(all_P),
        "gamma_S": np.array(all_gamma_S),
        "Gamma": np.array(all_Gamma),
        "S_ss": np.array(all_S_ss),
        "P_th": np.array(all_P_th),
    }


def run_laser_rate_rediscovery(
    output_dir: str | Path = "output/rediscovery/laser_rate",
    n_iterations: int = 40,
    **kwargs,
) -> dict:
    """Run the full semiconductor laser rate equations rediscovery.

    1. Generate trajectory for SINDy ODE recovery
    2. Threshold detection via P sweep
    3. L-I curve analysis
    4. Relaxation oscillation frequency measurement
    5. Steady-state analysis

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    P_th_theory = (
        _DEFAULT_PARAMS["gamma_N"] * _DEFAULT_PARAMS["N_tr"]
        + _DEFAULT_PARAMS["gamma_S"]
        / (_DEFAULT_PARAMS["Gamma"] * _DEFAULT_PARAMS["g"])
    )

    results: dict = {
        "domain": "laser_rate",
        "targets": {
            "ode_N": "dN/dt = P - gamma_N*N - g*(N-N_tr)*S",
            "ode_S": "dS/dt = Gamma*g*(N-N_tr)*S - gamma_S*S + Gamma*beta*gamma_N*N",
            "threshold": f"P_th = gamma_N*N_tr + gamma_S/(Gamma*g) = {P_th_theory:.4f}",
            "steady_state": "S_ss = Gamma*(P - P_th)/gamma_S",
            "relaxation": "omega_r = sqrt(gamma_S * g * S_ss)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating laser trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=10000, dt=0.001, P=2.0)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["N_c", "S_p"],
            threshold=0.05,
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
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Threshold detection from P sweep ---
    logger.info("Part 2: Threshold detection via P sweep...")
    th_data = generate_threshold_sweep(n_P=30, dt=0.001, n_settle=30000)

    # Detect threshold: first P where S_ss > small value
    S_threshold = 1e-4
    above = th_data["S_steady"] > S_threshold
    if np.any(above):
        idx = np.argmax(above)
        P_th_est = float(th_data["P"][max(0, idx - 1)])
        results["threshold"] = {
            "P_th_estimate": P_th_est,
            "P_th_theory": float(th_data["P_th_theory"]),
            "relative_error": float(
                abs(P_th_est - P_th_theory) / P_th_theory
            ),
        }
        logger.info(
            f"  P_th estimate: {P_th_est:.4f} "
            f"(theory: {P_th_theory:.4f})"
        )
    else:
        results["threshold"] = {"error": "No lasing detected in sweep range"}

    # --- Part 3: L-I curve ---
    logger.info("Part 3: L-I curve analysis...")
    li_data = generate_li_curve(n_P=30, dt=0.001, n_settle=30000)

    # Above threshold, S should be linear in P
    P_vals = li_data["P"]
    S_vals = li_data["S_steady"]
    above_th = P_vals > P_th_theory * 1.2
    if np.sum(above_th) > 3:
        # Linear fit S = slope * (P - P_th)
        P_above = P_vals[above_th] - P_th_theory
        S_above = S_vals[above_th]
        A = np.column_stack([P_above, np.ones(len(P_above))])
        coeffs = np.linalg.lstsq(A, S_above, rcond=None)[0]
        slope = coeffs[0]
        # Theory: slope = Gamma / gamma_S
        slope_theory = _DEFAULT_PARAMS["Gamma"] / _DEFAULT_PARAMS["gamma_S"]
        results["li_curve"] = {
            "n_points": int(np.sum(above_th)),
            "slope_measured": float(slope),
            "slope_theory": float(slope_theory),
            "relative_error": float(
                abs(slope - slope_theory) / slope_theory
            ),
        }
        logger.info(
            f"  L-I slope: {slope:.6f} "
            f"(theory: {slope_theory:.6f})"
        )

    # --- Part 4: Relaxation oscillation frequency ---
    logger.info("Part 4: Relaxation oscillation frequency...")
    try:
        relax_data = generate_relaxation_frequency_data(
            n_P=15, dt=0.0001, n_transient=3000, n_measure=20000,
        )
        omega_meas = relax_data["omega_r_measured"]
        omega_theory = relax_data["omega_r_theory"]
        valid = omega_meas > 0
        if np.sum(valid) > 3:
            corr = float(np.corrcoef(omega_meas[valid], omega_theory[valid])[0, 1])
            results["relaxation_oscillation"] = {
                "n_points": int(np.sum(valid)),
                "omega_range": [
                    float(np.min(omega_meas[valid])),
                    float(np.max(omega_meas[valid])),
                ],
                "correlation_with_theory": corr,
            }
            logger.info(
                f"  {int(np.sum(valid))} frequency measurements, "
                f"correlation={corr:.4f}"
            )
    except Exception as e:
        logger.warning(f"Relaxation frequency analysis failed: {e}")
        results["relaxation_oscillation"] = {"error": str(e)}

    # --- Part 5: Steady-state verification ---
    logger.info("Part 5: Steady-state verification...")
    sim_ss = _make_sim(dt=0.001, n_steps=100, P=2.0)
    sim_ss.reset()
    N_ss_theory, S_ss_theory = sim_ss.steady_state
    results["steady_state"] = {
        "N_ss_theory": float(N_ss_theory),
        "S_ss_theory": float(S_ss_theory),
        "P_th": float(sim_ss.threshold_pump),
        "omega_r": float(sim_ss.relaxation_frequency),
    }

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data arrays
    np.savez(
        output_path / "ode_data.npz",
        states=ode_data["states"],
    )

    return results
