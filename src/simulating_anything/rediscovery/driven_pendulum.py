"""Damped driven pendulum rediscovery.

Targets:
- Bifurcation diagram: period-doubling route to chaos as A increases
- Resonance curve: steady-state amplitude vs driving frequency omega_d
- Lyapunov exponent vs driving amplitude A (chaos onset detection)
- PySR on resonance curve to find amplitude-frequency relationship
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.driven_pendulum import DrivenPendulum
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Default physical parameters for the driven pendulum
_GAMMA = 0.5
_OMEGA0 = 1.5
_OMEGA_D = 2.0 / 3.0


def generate_bifurcation_data(
    n_A: int = 40,
    A_min: float = 0.5,
    A_max: float = 2.0,
    n_transient_periods: int = 300,
    n_sample_periods: int = 100,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep driving amplitude A and record Poincare section theta values.

    After discarding transient, samples theta at each drive period.
    Period-1 orbit gives 1 unique theta, period-2 gives 2, chaos gives many.

    Args:
        n_A: Number of A values to sweep.
        A_min: Minimum driving amplitude.
        A_max: Maximum driving amplitude.
        n_transient_periods: Number of drive periods to discard as transient.
        n_sample_periods: Number of drive periods to sample.
        dt: Integration timestep.

    Returns:
        Dict with A_values, poincare_theta (list of arrays), poincare_omega.
    """
    A_values = np.linspace(A_min, A_max, n_A)
    T_d = 2 * np.pi / _OMEGA_D
    steps_per_period = int(round(T_d / dt))

    all_thetas = []
    all_omegas = []

    for i, A in enumerate(A_values):
        config = SimulationConfig(
            domain=Domain.DRIVEN_PENDULUM,
            dt=dt,
            n_steps=1000,
            parameters={
                "gamma": _GAMMA,
                "omega0": _OMEGA0,
                "A_drive": A,
                "omega_d": _OMEGA_D,
                "theta_0": 0.1,
                "omega_init": 0.0,
            },
        )
        sim = DrivenPendulum(config)
        sim.reset()

        # Discard transient
        for _ in range(n_transient_periods * steps_per_period):
            sim.step()

        # Sample at drive period intervals
        thetas = []
        omegas = []
        for _ in range(n_sample_periods):
            for _ in range(steps_per_period):
                sim.step()
            theta = sim._state[0]
            # Normalize to [-pi, pi]
            theta = (theta + np.pi) % (2 * np.pi) - np.pi
            thetas.append(theta)
            omegas.append(sim._state[1])

        all_thetas.append(np.array(thetas))
        all_omegas.append(np.array(omegas))

        if (i + 1) % 10 == 0:
            n_unique = len(set(np.round(thetas, 2)))
            logger.info(
                f"  A={A:.3f}: {n_unique} unique Poincare points "
                f"(out of {n_sample_periods})"
            )

    return {
        "A_values": A_values,
        "poincare_theta": all_thetas,
        "poincare_omega": all_omegas,
    }


def generate_resonance_data(
    n_omega: int = 30,
    omega_min: float = 0.1,
    omega_max: float = 3.0,
    A_drive: float = 0.3,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep driving frequency and measure steady-state amplitude.

    Uses small driving amplitude to stay in the linear/weakly nonlinear regime
    where a clear resonance peak is visible.

    Args:
        n_omega: Number of frequency values.
        omega_min: Minimum driving frequency.
        omega_max: Maximum driving frequency.
        A_drive: Driving amplitude (keep small for clean resonance).
        dt: Integration timestep.

    Returns:
        Dict with omega_d values and measured amplitudes.
    """
    omega_values = np.linspace(omega_min, omega_max, n_omega)
    amplitudes = []

    for i, omega_d in enumerate(omega_values):
        config = SimulationConfig(
            domain=Domain.DRIVEN_PENDULUM,
            dt=dt,
            n_steps=1000,
            parameters={
                "gamma": _GAMMA,
                "omega0": _OMEGA0,
                "A_drive": A_drive,
                "omega_d": omega_d,
                "theta_0": 0.0,
                "omega_init": 0.0,
            },
        )
        sim = DrivenPendulum(config)
        sim.reset()
        amp = sim.measure_steady_amplitude(n_periods=20)
        amplitudes.append(amp)

        if (i + 1) % 10 == 0:
            logger.info(f"  omega_d={omega_d:.3f}: amplitude={amp:.4f}")

    return {
        "omega_d": omega_values,
        "amplitude": np.array(amplitudes),
        "A_drive": A_drive,
        "gamma": _GAMMA,
        "omega0": _OMEGA0,
    }


def generate_lyapunov_data(
    n_A: int = 30,
    A_min: float = 0.5,
    A_max: float = 2.0,
    n_steps: int = 20000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep driving amplitude and compute Lyapunov exponent at each value.

    Identifies the transition from periodic to chaotic dynamics.

    Args:
        n_A: Number of amplitude values.
        A_min: Minimum driving amplitude.
        A_max: Maximum driving amplitude.
        n_steps: Steps for Lyapunov computation at each A.
        dt: Integration timestep.

    Returns:
        Dict with A values and Lyapunov exponents.
    """
    A_values = np.linspace(A_min, A_max, n_A)
    lyapunov_exps = []

    for i, A in enumerate(A_values):
        config = SimulationConfig(
            domain=Domain.DRIVEN_PENDULUM,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "gamma": _GAMMA,
                "omega0": _OMEGA0,
                "A_drive": A,
                "omega_d": _OMEGA_D,
                "theta_0": 0.1,
                "omega_init": 0.0,
            },
        )
        sim = DrivenPendulum(config)
        sim.reset()

        # Skip transient
        T_d = 2 * np.pi / _OMEGA_D
        transient_steps = int(100 * T_d / dt)
        for _ in range(transient_steps):
            sim.step()

        lam = sim.compute_lyapunov(n_steps=n_steps)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            regime = "chaotic" if lam > 0.01 else "periodic"
            logger.info(f"  A={A:.3f}: Lyapunov={lam:.4f} ({regime})")

    return {
        "A_values": A_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def run_driven_pendulum_rediscovery(
    output_dir: str | Path = "output/rediscovery/driven_pendulum",
    n_iterations: int = 40,
) -> dict:
    """Run the full driven pendulum rediscovery pipeline.

    1. Generate bifurcation diagram (A sweep, Poincare section)
    2. Generate resonance curve (omega_d sweep, amplitude)
    3. Generate Lyapunov exponent vs A (chaos onset)
    4. Run PySR on resonance data

    Args:
        output_dir: Directory for saving results.
        n_iterations: Number of PySR iterations.

    Returns:
        Dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "driven_pendulum",
        "targets": {
            "ode": "theta'' + gamma*theta' + omega0^2*sin(theta) = A*cos(omega_d*t)",
            "period_doubling": "Period-doubling at A ~ 1.2",
            "chaos_onset": "Chaos at A ~ 1.5",
            "resonance": "Peak amplitude near omega_d ~ omega0",
        },
    }

    # --- Part 1: Bifurcation diagram ---
    logger.info("Part 1: Bifurcation diagram (A sweep)...")
    bif_data = generate_bifurcation_data(n_A=40, dt=0.005)

    # Analyze bifurcation: count distinct Poincare points at each A
    n_poincare_points = []
    for thetas in bif_data["poincare_theta"]:
        # Cluster nearby points to count distinct orbits
        rounded = np.round(thetas, 1)
        n_unique = len(set(rounded))
        n_poincare_points.append(n_unique)

    results["bifurcation"] = {
        "n_A_values": len(bif_data["A_values"]),
        "A_range": [
            float(bif_data["A_values"][0]),
            float(bif_data["A_values"][-1]),
        ],
        "n_poincare_points": n_poincare_points,
    }
    logger.info(
        f"  A range: [{bif_data['A_values'][0]:.2f}, "
        f"{bif_data['A_values'][-1]:.2f}]"
    )

    # --- Part 2: Resonance curve ---
    logger.info("Part 2: Resonance curve (omega_d sweep)...")
    res_data = generate_resonance_data(n_omega=30, A_drive=0.3, dt=0.005)

    # Find resonance peak
    peak_idx = np.argmax(res_data["amplitude"])
    peak_omega = float(res_data["omega_d"][peak_idx])
    peak_amp = float(res_data["amplitude"][peak_idx])

    results["resonance"] = {
        "n_omega_values": len(res_data["omega_d"]),
        "omega_range": [
            float(res_data["omega_d"][0]),
            float(res_data["omega_d"][-1]),
        ],
        "peak_omega_d": peak_omega,
        "peak_amplitude": peak_amp,
        "omega0": _OMEGA0,
    }
    logger.info(
        f"  Resonance peak at omega_d={peak_omega:.3f} "
        f"(omega0={_OMEGA0}), amplitude={peak_amp:.4f}"
    )

    # PySR: find amplitude(omega_d)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = res_data["omega_d"].reshape(-1, 1)
        y = res_data["amplitude"]

        logger.info("  Running PySR: amplitude = f(omega_d)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["w_d"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "sin", "cos"],
            max_complexity=15,
            populations=20,
            population_size=40,
        )
        results["resonance_pysr"] = {
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
            results["resonance_pysr"]["best"] = best.expression
            results["resonance_pysr"]["best_r2"] = (
                best.evidence.fit_r_squared
            )
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["resonance_pysr"] = {"error": str(e)}

    # --- Part 3: Lyapunov exponent vs A ---
    logger.info("Part 3: Lyapunov exponent vs A (chaos onset)...")
    lyap_data = generate_lyapunov_data(n_A=30, n_steps=20000, dt=0.005)

    # Find chaos onset: first A where Lyapunov > 0
    lam = lyap_data["lyapunov_exponent"]
    A_vals = lyap_data["A_values"]
    chaos_mask = lam > 0.01
    if np.any(chaos_mask):
        A_c_approx = float(A_vals[np.argmax(chaos_mask)])
    else:
        A_c_approx = None

    results["lyapunov_analysis"] = {
        "n_A_values": len(A_vals),
        "A_range": [float(A_vals[0]), float(A_vals[-1])],
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
        "A_c_approx": A_c_approx,
        "n_chaotic": int(np.sum(chaos_mask)),
        "n_periodic": int(np.sum(~chaos_mask)),
    }
    if A_c_approx is not None:
        logger.info(f"  Chaos onset at A ~ {A_c_approx:.3f}")
    logger.info(
        f"  Max Lyapunov: {np.max(lam):.4f}, "
        f"{int(np.sum(chaos_mask))} chaotic / "
        f"{int(np.sum(~chaos_mask))} periodic"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data arrays
    np.savez(
        output_path / "resonance_data.npz",
        omega_d=res_data["omega_d"],
        amplitude=res_data["amplitude"],
    )
    np.savez(
        output_path / "lyapunov_data.npz",
        A_values=lyap_data["A_values"],
        lyapunov_exponent=lyap_data["lyapunov_exponent"],
    )

    return results
