"""Wilberforce pendulum rediscovery.

Targets:
- Energy transfer period vs coupling strength eps
- Normal mode frequencies: omega_z = sqrt(k/m), omega_theta = sqrt(kappa/I)
- SINDy recovery of coupled ODEs
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.wilberforce import Wilberforce
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_energy_transfer_data(
    n_samples: int = 50,
    n_steps: int = 100000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Generate data studying energy transfer period vs coupling strength.

    With omega_z ~ omega_theta, the energy transfer period should depend
    on the coupling eps and the frequency detuning.

    For the resonant case (omega_z = omega_theta), the transfer period
    is determined by the coupling strength: T ~ 4*pi*sqrt(m*I) / eps.
    """
    rng = np.random.default_rng(42)

    all_eps = []
    all_m = []
    all_I_val = []
    all_k = []
    all_kappa = []
    all_T_transfer_measured = []
    all_T_transfer_theory = []

    for i in range(n_samples):
        # Choose parameters so omega_z ~ omega_theta (near-resonant)
        m = rng.uniform(0.3, 1.0)
        k = rng.uniform(2.0, 10.0)
        omega_z = np.sqrt(k / m)
        # Set kappa/I so omega_theta = omega_z (exact resonance)
        I_val = rng.uniform(5e-5, 5e-4)
        kappa = omega_z**2 * I_val
        eps = rng.uniform(1e-4, 5e-3)

        config = SimulationConfig(
            domain=Domain.WILBERFORCE,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "m": m, "k": k, "I": I_val, "kappa": kappa, "eps": eps,
                "z_0": 0.1, "z_dot_0": 0.0,
                "theta_0": 0.0, "theta_dot_0": 0.0,
            },
        )
        sim = Wilberforce(config)
        sim.reset()

        # Track translational energy over time
        trans_energies = [sim.translational_energy]
        for _ in range(n_steps):
            sim.step()
            trans_energies.append(sim.translational_energy)

        trans_energies = np.array(trans_energies)

        # Find minima of translational energy (transfer points)
        # Use a coarse window to find where energy has transferred to rotation
        window = max(50, int(0.2 / dt))
        minima_times = []
        for j in range(window, len(trans_energies) - window):
            local_min = np.min(trans_energies[j - window:j + window + 1])
            if (trans_energies[j] == local_min
                    and trans_energies[j] < 0.3 * trans_energies[0]):
                minima_times.append(j * dt)

        # Deduplicate close minima
        if len(minima_times) >= 2:
            deduped = [minima_times[0]]
            for t in minima_times[1:]:
                if t - deduped[-1] > 0.5 / omega_z:
                    deduped.append(t)
            minima_times = deduped

        if len(minima_times) >= 2:
            # Transfer period = time between successive minima
            periods = np.diff(minima_times)
            T_measured = float(np.median(periods))

            # Theory for exact resonance: T_transfer = 4*pi*sqrt(m*I) / eps
            T_theory = 4.0 * np.pi * np.sqrt(m * I_val) / eps

            all_eps.append(eps)
            all_m.append(m)
            all_I_val.append(I_val)
            all_k.append(k)
            all_kappa.append(kappa)
            all_T_transfer_measured.append(T_measured)
            all_T_transfer_theory.append(T_theory)

        if (i + 1) % 10 == 0:
            logger.info(f"  Energy transfer measurement {i + 1}/{n_samples}")

    return {
        "eps": np.array(all_eps),
        "m": np.array(all_m),
        "I": np.array(all_I_val),
        "k": np.array(all_k),
        "kappa": np.array(all_kappa),
        "T_transfer_measured": np.array(all_T_transfer_measured),
        "T_transfer_theory": np.array(all_T_transfer_theory),
    }


def generate_normal_mode_data(
    n_samples: int = 100,
    n_steps: int = 30000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Generate data verifying normal mode frequencies.

    Vary k, m, kappa, I independently (with eps=0 for pure modes),
    measure oscillation frequency from zero crossings.
    """
    rng = np.random.default_rng(123)

    all_k = []
    all_m = []
    all_kappa = []
    all_I_val = []
    all_omega_z_measured = []
    all_omega_z_theory = []
    all_omega_theta_measured = []
    all_omega_theta_theory = []

    for i in range(n_samples):
        m = rng.uniform(0.3, 2.0)
        k = rng.uniform(1.0, 20.0)
        I_val = rng.uniform(1e-5, 1e-3)
        kappa = rng.uniform(1e-4, 1e-2)

        omega_z_theory = np.sqrt(k / m)
        omega_theta_theory = np.sqrt(kappa / I_val)

        # --- Measure omega_z: excite z only, eps=0 ---
        config_z = SimulationConfig(
            domain=Domain.WILBERFORCE,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "m": m, "k": k, "I": I_val, "kappa": kappa, "eps": 0.0,
                "z_0": 0.1, "z_dot_0": 0.0,
                "theta_0": 0.0, "theta_dot_0": 0.0,
            },
        )
        sim_z = Wilberforce(config_z)
        sim_z.reset()

        positions_z = [sim_z.observe()[0]]
        for _ in range(n_steps):
            positions_z.append(sim_z.step()[0])
        positions_z = np.array(positions_z)

        omega_z_meas = _measure_frequency(positions_z, dt)

        # --- Measure omega_theta: excite theta only, eps=0 ---
        config_t = SimulationConfig(
            domain=Domain.WILBERFORCE,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "m": m, "k": k, "I": I_val, "kappa": kappa, "eps": 0.0,
                "z_0": 0.0, "z_dot_0": 0.0,
                "theta_0": 0.5, "theta_dot_0": 0.0,
            },
        )
        sim_t = Wilberforce(config_t)
        sim_t.reset()

        positions_t = [sim_t.observe()[2]]  # theta component
        for _ in range(n_steps):
            positions_t.append(sim_t.step()[2])
        positions_t = np.array(positions_t)

        omega_theta_meas = _measure_frequency(positions_t, dt)

        if omega_z_meas is not None and omega_theta_meas is not None:
            all_k.append(k)
            all_m.append(m)
            all_kappa.append(kappa)
            all_I_val.append(I_val)
            all_omega_z_measured.append(omega_z_meas)
            all_omega_z_theory.append(omega_z_theory)
            all_omega_theta_measured.append(omega_theta_meas)
            all_omega_theta_theory.append(omega_theta_theory)

        if (i + 1) % 25 == 0:
            logger.info(f"  Normal mode measurement {i + 1}/{n_samples}")

    return {
        "k": np.array(all_k),
        "m": np.array(all_m),
        "kappa": np.array(all_kappa),
        "I": np.array(all_I_val),
        "omega_z_measured": np.array(all_omega_z_measured),
        "omega_z_theory": np.array(all_omega_z_theory),
        "omega_theta_measured": np.array(all_omega_theta_measured),
        "omega_theta_theory": np.array(all_omega_theta_theory),
    }


def _measure_frequency(positions: np.ndarray, dt: float) -> float | None:
    """Measure oscillation frequency from positive-going zero crossings."""
    crossings = []
    for j in range(1, len(positions)):
        if positions[j - 1] < 0 and positions[j] >= 0:
            frac = -positions[j - 1] / (positions[j] - positions[j - 1])
            crossings.append((j - 1 + frac) * dt)

    if len(crossings) >= 3:
        periods = np.diff(crossings)
        T = float(np.median(periods))
        if T > 0:
            return 2.0 * np.pi / T
    return None


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Generate a Wilberforce trajectory for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.WILBERFORCE,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "m": 0.5, "k": 5.0, "I": 1e-4, "kappa": 1e-3, "eps": 1e-3,
            "z_0": 0.1, "z_dot_0": 0.0,
            "theta_0": 0.0, "theta_dot_0": 0.0,
        },
    )
    sim = Wilberforce(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "m": 0.5,
        "k": 5.0,
        "I": 1e-4,
        "kappa": 1e-3,
        "eps": 1e-3,
    }


def run_wilberforce_rediscovery(
    output_dir: str | Path = "output/rediscovery/wilberforce",
    n_iterations: int = 40,
) -> dict:
    """Run the full Wilberforce pendulum rediscovery.

    1. Normal mode frequencies: omega_z = sqrt(k/m), omega_theta = sqrt(kappa/I)
    2. Energy transfer period vs coupling
    3. SINDy ODE recovery
    """
    from simulating_anything.analysis.symbolic_regression import (
        run_symbolic_regression,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "wilberforce",
        "targets": {
            "omega_z": "omega_z = sqrt(k/m)",
            "omega_theta": "omega_theta = sqrt(kappa/I)",
            "energy_transfer": "T_transfer = 4*pi*sqrt(m*I)/eps (resonant)",
            "ode": "m*z'' = -k*z - eps/2*theta, I*theta'' = -kappa*theta - eps/2*z",
        },
    }

    # --- Part 1: Normal mode frequencies ---
    logger.info("Part 1: Normal mode frequency measurement...")
    mode_data = generate_normal_mode_data(n_samples=100, n_steps=30000, dt=0.001)

    if len(mode_data["k"]) > 0:
        z_error = (
            np.abs(mode_data["omega_z_measured"] - mode_data["omega_z_theory"])
            / mode_data["omega_z_theory"]
        )
        t_error = (
            np.abs(
                mode_data["omega_theta_measured"] - mode_data["omega_theta_theory"]
            )
            / mode_data["omega_theta_theory"]
        )
        results["normal_modes"] = {
            "n_samples": len(mode_data["k"]),
            "omega_z_mean_error": float(np.mean(z_error)),
            "omega_theta_mean_error": float(np.mean(t_error)),
        }
        logger.info(f"  omega_z error: {np.mean(z_error):.4%}")
        logger.info(f"  omega_theta error: {np.mean(t_error):.4%}")

        # PySR: omega_z = f(k, m)
        logger.info(
            f"  Running PySR for omega_z = f(k, m) "
            f"with {n_iterations} iterations..."
        )
        X_z = np.column_stack([mode_data["k"], mode_data["m"]])
        y_z = mode_data["omega_z_measured"]

        z_discoveries = run_symbolic_regression(
            X_z,
            y_z,
            variable_names=["k_", "m_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square"],
            max_complexity=15,
            populations=20,
            population_size=40,
        )

        results["omega_z_pysr"] = {
            "n_discoveries": len(z_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in z_discoveries[:5]
            ],
        }
        if z_discoveries:
            best = z_discoveries[0]
            results["omega_z_pysr"]["best"] = best.expression
            results["omega_z_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )

        # PySR: omega_theta = f(kappa, I)
        logger.info("  Running PySR for omega_theta = f(kappa, I)...")
        X_t = np.column_stack([mode_data["kappa"], mode_data["I"]])
        y_t = mode_data["omega_theta_measured"]

        t_discoveries = run_symbolic_regression(
            X_t,
            y_t,
            variable_names=["kap_", "I_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square"],
            max_complexity=15,
            populations=20,
            population_size=40,
        )

        results["omega_theta_pysr"] = {
            "n_discoveries": len(t_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in t_discoveries[:5]
            ],
        }
        if t_discoveries:
            best = t_discoveries[0]
            results["omega_theta_pysr"]["best"] = best.expression
            results["omega_theta_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    else:
        results["normal_modes"] = {"n_samples": 0, "error": "no data"}

    # --- Part 2: Energy transfer period ---
    logger.info("Part 2: Energy transfer period vs coupling...")
    transfer_data = generate_energy_transfer_data(
        n_samples=50, n_steps=100000, dt=0.001,
    )

    if len(transfer_data["eps"]) > 0:
        rel_error = (
            np.abs(
                transfer_data["T_transfer_measured"]
                - transfer_data["T_transfer_theory"]
            )
            / np.maximum(transfer_data["T_transfer_theory"], 1e-10)
        )
        results["energy_transfer"] = {
            "n_samples": len(transfer_data["eps"]),
            "mean_relative_error": float(np.mean(rel_error)),
        }
        logger.info(
            f"  Transfer period accuracy: mean error = {np.mean(rel_error):.4%}"
        )

        # PySR: T_transfer = f(eps, m, I)
        logger.info("  Running PySR for T_transfer = f(eps, m, I)...")
        X_tr = np.column_stack([
            transfer_data["eps"],
            transfer_data["m"],
            transfer_data["I"],
        ])
        y_tr = transfer_data["T_transfer_measured"]

        tr_discoveries = run_symbolic_regression(
            X_tr,
            y_tr,
            variable_names=["eps_", "m_", "I_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square"],
            max_complexity=20,
            populations=20,
            population_size=40,
        )

        results["transfer_pysr"] = {
            "n_discoveries": len(tr_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in tr_discoveries[:5]
            ],
        }
        if tr_discoveries:
            best = tr_discoveries[0]
            results["transfer_pysr"]["best"] = best.expression
            results["transfer_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    else:
        results["energy_transfer"] = {"n_samples": 0, "error": "no data"}

    # --- Part 3: SINDy ODE recovery ---
    logger.info("Part 3: SINDy ODE recovery...")
    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        ode_data = generate_ode_data(n_steps=5000, dt=0.001)
        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["z", "z_dot", "theta", "theta_dot"],
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
            "true_m": ode_data["m"],
            "true_k": ode_data["k"],
            "true_I": ode_data["I"],
            "true_kappa": ode_data["kappa"],
            "true_eps": ode_data["eps"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    if len(mode_data["k"]) > 0:
        np.savez(
            output_path / "normal_mode_data.npz",
            k=mode_data["k"],
            m=mode_data["m"],
            kappa=mode_data["kappa"],
            I=mode_data["I"],
            omega_z_measured=mode_data["omega_z_measured"],
            omega_z_theory=mode_data["omega_z_theory"],
            omega_theta_measured=mode_data["omega_theta_measured"],
            omega_theta_theory=mode_data["omega_theta_theory"],
        )

    if len(transfer_data["eps"]) > 0:
        np.savez(
            output_path / "transfer_data.npz",
            eps=transfer_data["eps"],
            m=transfer_data["m"],
            I=transfer_data["I"],
            T_transfer_measured=transfer_data["T_transfer_measured"],
            T_transfer_theory=transfer_data["T_transfer_theory"],
        )

    return results
