"""SIS Adaptive Behavior epidemic rediscovery.

Targets:
- Endemic equilibrium I* = (beta0 - gamma) / (beta0 + kappa*gamma)
- Epidemic threshold R0 = beta0 / gamma > 1
- Higher kappa -> lower endemic level
- kappa=0 recovers standard SIS: I* = 1 - gamma/beta0
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.sis_adaptive import SISAdaptiveSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_DOMAIN = Domain.SIS_ADAPTIVE  # Placeholder domain enum


def _make_config(
    beta0: float = 0.5,
    gamma: float = 0.1,
    kappa: float = 5.0,
    I_0: float = 0.01,
    dt: float = 0.1,
    n_steps: int = 5000,
) -> SimulationConfig:
    """Create a SimulationConfig for SIS adaptive behavior."""
    return SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "beta0": beta0,
            "gamma": gamma,
            "kappa": kappa,
            "I_0": I_0,
        },
    )


def generate_kappa_sweep_data(
    n_kappa: int = 20,
    beta0: float = 0.5,
    gamma: float = 0.1,
    n_steps: int = 50000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep kappa from 0 to 20 and measure endemic equilibrium I*.

    For each kappa, run to steady state and record the long-term average I.
    Compares against the theoretical formula:
        I* = (beta0 - gamma) / (beta0 + kappa * gamma)
    """
    kappa_values = np.linspace(0.0, 20.0, n_kappa)
    measured_I_star = []
    theory_I_star = []

    for i, kappa in enumerate(kappa_values):
        config = _make_config(
            beta0=beta0, gamma=gamma, kappa=kappa,
            I_0=0.1, dt=dt, n_steps=n_steps,
        )
        sim = SISAdaptiveSimulation(config)
        sim.reset()

        # Run transient
        n_transient = n_steps // 2
        for _ in range(n_transient):
            sim.step()

        # Average over remaining steps
        I_samples = []
        for _ in range(n_steps - n_transient):
            sim.step()
            I_samples.append(float(sim._state[0]))
        measured = float(np.mean(I_samples))
        measured_I_star.append(measured)

        # Theoretical
        if beta0 > gamma:
            theory = (beta0 - gamma) / (beta0 + kappa * gamma)
        else:
            theory = 0.0
        theory_I_star.append(theory)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  kappa={kappa:.2f}: I*_meas={measured:.6f}, "
                f"I*_theory={theory:.6f}"
            )

    return {
        "kappa": kappa_values,
        "measured_I_star": np.array(measured_I_star),
        "theory_I_star": np.array(theory_I_star),
        "beta0": beta0,
        "gamma": gamma,
    }


def generate_beta_sweep_data(
    n_beta: int = 20,
    gamma: float = 0.1,
    kappa: float = 5.0,
    n_steps: int = 50000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep beta0 from 0.05 to 1.0 and find epidemic threshold.

    The epidemic threshold is at beta0 = gamma (R0 = 1).
    Above threshold, I* = (beta0 - gamma) / (beta0 + kappa*gamma).
    """
    beta_values = np.linspace(0.05, 1.0, n_beta)
    measured_I_star = []
    theory_I_star = []
    R0_values = []

    for i, beta0 in enumerate(beta_values):
        config = _make_config(
            beta0=beta0, gamma=gamma, kappa=kappa,
            I_0=0.01, dt=dt, n_steps=n_steps,
        )
        sim = SISAdaptiveSimulation(config)
        sim.reset()

        # Run transient
        n_transient = n_steps // 2
        for _ in range(n_transient):
            sim.step()

        # Average over remaining steps
        I_samples = []
        for _ in range(n_steps - n_transient):
            sim.step()
            I_samples.append(float(sim._state[0]))
        measured = float(np.mean(I_samples))
        measured_I_star.append(measured)

        R0 = beta0 / gamma
        R0_values.append(R0)

        if R0 > 1:
            theory = (beta0 - gamma) / (beta0 + kappa * gamma)
        else:
            theory = 0.0
        theory_I_star.append(theory)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  beta0={beta0:.3f}, R0={R0:.2f}: "
                f"I*_meas={measured:.6f}, I*_theory={theory:.6f}"
            )

    return {
        "beta0": beta_values,
        "R0": np.array(R0_values),
        "measured_I_star": np.array(measured_I_star),
        "theory_I_star": np.array(theory_I_star),
        "gamma": gamma,
        "kappa": kappa,
    }


def run_sis_adaptive_rediscovery(
    output_dir: str | Path = "output/rediscovery/sis_adaptive",
    n_iterations: int = 40,
) -> dict:
    """Run the SIS adaptive behavior rediscovery pipeline.

    Discovers:
    1. Endemic equilibrium I* = (beta0 - gamma) / (beta0 + kappa*gamma) via kappa sweep
    2. Epidemic threshold R0 = beta0/gamma via beta sweep
    3. Formula verification: measured vs theoretical I*
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "sis_adaptive",
        "targets": {
            "endemic_eq": "I* = (beta0 - gamma) / (beta0 + kappa*gamma)",
            "threshold": "R0 = beta0/gamma > 1 for epidemic",
            "behavioral_beta": "beta(I) = beta0 / (1 + kappa*I)",
        },
    }

    # --- Part 1: Kappa sweep ---
    logger.info("Part 1: Kappa sweep for I* vs kappa...")
    kappa_data = generate_kappa_sweep_data(
        n_kappa=20, beta0=0.5, gamma=0.1,
        n_steps=30000, dt=0.1,
    )

    # Compare measured vs theory
    errors = np.abs(
        kappa_data["measured_I_star"] - kappa_data["theory_I_star"]
    )
    # Only compare where theory > 0
    valid = kappa_data["theory_I_star"] > 1e-6
    if np.any(valid):
        rel_errors = errors[valid] / kappa_data["theory_I_star"][valid]
        mean_rel_error = float(np.mean(rel_errors))
    else:
        mean_rel_error = 0.0

    results["kappa_sweep"] = {
        "n_points": len(kappa_data["kappa"]),
        "kappa_range": [
            float(kappa_data["kappa"].min()),
            float(kappa_data["kappa"].max()),
        ],
        "mean_absolute_error": float(np.mean(errors)),
        "mean_relative_error": mean_rel_error,
        "max_absolute_error": float(np.max(errors)),
        "I_star_range": [
            float(kappa_data["measured_I_star"].min()),
            float(kappa_data["measured_I_star"].max()),
        ],
    }
    logger.info(
        f"  Kappa sweep: mean rel error = {mean_rel_error:.6f}, "
        f"I* range [{kappa_data['measured_I_star'].min():.4f}, "
        f"{kappa_data['measured_I_star'].max():.4f}]"
    )

    # --- Part 2: Beta sweep for epidemic threshold ---
    logger.info("Part 2: Beta sweep for epidemic threshold...")
    beta_data = generate_beta_sweep_data(
        n_beta=20, gamma=0.1, kappa=5.0,
        n_steps=30000, dt=0.1,
    )

    # Find threshold: first beta where I* > small threshold
    threshold_I = 1e-4
    above_threshold = beta_data["measured_I_star"] > threshold_I
    if np.any(above_threshold):
        idx = np.argmax(above_threshold)
        beta_c_est = float(beta_data["beta0"][idx])
    else:
        beta_c_est = float(beta_data["beta0"][-1])

    beta_c_theory = beta_data["gamma"]  # R0 = beta0/gamma = 1

    results["beta_sweep"] = {
        "n_points": len(beta_data["beta0"]),
        "beta_range": [
            float(beta_data["beta0"].min()),
            float(beta_data["beta0"].max()),
        ],
        "beta_c_estimate": beta_c_est,
        "beta_c_theory": float(beta_c_theory),
        "R0_range": [
            float(beta_data["R0"].min()),
            float(beta_data["R0"].max()),
        ],
    }
    logger.info(
        f"  beta_c estimate: {beta_c_est:.4f} "
        f"(theory: {beta_c_theory:.4f})"
    )

    # --- Part 3: Formula verification ---
    logger.info("Part 3: Formula verification I* = (b-g)/(b+k*g)...")
    # Use kappa sweep data for correlation
    if np.any(valid):
        corr = float(np.corrcoef(
            kappa_data["measured_I_star"][valid],
            kappa_data["theory_I_star"][valid],
        )[0, 1])
    else:
        corr = 0.0

    # Also verify with beta sweep data
    valid_beta = beta_data["theory_I_star"] > 1e-6
    if np.any(valid_beta):
        beta_errors = np.abs(
            beta_data["measured_I_star"][valid_beta]
            - beta_data["theory_I_star"][valid_beta]
        )
        beta_rel_errors = (
            beta_errors / beta_data["theory_I_star"][valid_beta]
        )
        beta_corr = float(np.corrcoef(
            beta_data["measured_I_star"][valid_beta],
            beta_data["theory_I_star"][valid_beta],
        )[0, 1])
    else:
        beta_rel_errors = np.array([0.0])
        beta_corr = 0.0

    results["formula_verification"] = {
        "kappa_sweep_correlation": corr,
        "kappa_sweep_mean_rel_error": mean_rel_error,
        "beta_sweep_correlation": beta_corr,
        "beta_sweep_mean_rel_error": float(np.mean(beta_rel_errors)),
    }
    logger.info(
        f"  Kappa sweep correlation: {corr:.6f}"
    )
    logger.info(
        f"  Beta sweep correlation: {beta_corr:.6f}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save sweep data
    np.savez(
        output_path / "kappa_sweep.npz",
        **{
            k: v
            for k, v in kappa_data.items()
            if isinstance(v, np.ndarray)
        },
    )
    np.savez(
        output_path / "beta_sweep.npz",
        **{
            k: v
            for k, v in beta_data.items()
            if isinstance(v, np.ndarray)
        },
    )

    return results
