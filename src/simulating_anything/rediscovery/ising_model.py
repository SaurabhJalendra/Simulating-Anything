"""2D Ising model rediscovery.

Targets:
- Phase transition at T_c = 2*J / ln(1 + sqrt(2)) ~ 2.269
- Spontaneous magnetization M(T) via Onsager solution
- Susceptibility divergence at T_c
- Energy per spin vs temperature
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.ising_model import IsingModel2D
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_magnetization_vs_temperature(
    n_T: int = 30,
    T_range: tuple[float, float] = (1.0, 4.0),
    N: int = 16,
    J: float = 1.0,
    h: float = 0.0,
    n_equil: int = 1000,
    n_measure: int = 2000,
) -> dict[str, np.ndarray]:
    """Sweep temperature and measure equilibrated |<m>|.

    Returns dict with arrays: T, magnetization, theory_magnetization.
    Target: phase transition at T_c ~ 2.269 for J=1.
    """
    T_values = np.linspace(T_range[0], T_range[1], n_T)
    mag_values = []
    theory_mag = []

    for i, T in enumerate(T_values):
        config = SimulationConfig(
            domain=Domain.ISING_MODEL,
            dt=1.0,
            n_steps=n_equil + n_measure,
            parameters={"N": float(N), "J": J, "h": h, "T": T},
        )
        sim = IsingModel2D(config)
        result = sim.measure_equilibrium(
            n_equil_sweeps=n_equil,
            n_measure_sweeps=n_measure,
            seed=42 + i,
        )
        mag_values.append(result["magnetization"])
        theory_mag.append(IsingModel2D.onsager_magnetization(T, J))

        if (i + 1) % 10 == 0:
            logger.info(
                f"  T={T:.2f}: |m|={result['magnetization']:.4f}, "
                f"theory={theory_mag[-1]:.4f}"
            )

    return {
        "T": T_values,
        "magnetization": np.array(mag_values),
        "theory_magnetization": np.array(theory_mag),
        "N": N,
        "J": J,
    }


def generate_susceptibility_data(
    n_T: int = 30,
    T_range: tuple[float, float] = (1.0, 4.0),
    N: int = 16,
    J: float = 1.0,
    n_equil: int = 1000,
    n_measure: int = 2000,
) -> dict[str, np.ndarray]:
    """Sweep temperature and measure susceptibility chi.

    chi = N^2 * (<m^2> - <|m|>^2) / T. Peaks at T_c.
    """
    T_values = np.linspace(T_range[0], T_range[1], n_T)
    chi_values = []

    for i, T in enumerate(T_values):
        config = SimulationConfig(
            domain=Domain.ISING_MODEL,
            dt=1.0,
            n_steps=n_equil + n_measure,
            parameters={"N": float(N), "J": J, "h": 0.0, "T": T},
        )
        sim = IsingModel2D(config)
        result = sim.measure_equilibrium(
            n_equil_sweeps=n_equil,
            n_measure_sweeps=n_measure,
            seed=42 + i,
        )
        chi_values.append(result["susceptibility"])

        if (i + 1) % 10 == 0:
            logger.info(f"  T={T:.2f}: chi={result['susceptibility']:.4f}")

    return {
        "T": T_values,
        "susceptibility": np.array(chi_values),
        "N": N,
        "J": J,
    }


def generate_energy_vs_temperature(
    n_T: int = 30,
    T_range: tuple[float, float] = (1.0, 4.0),
    N: int = 16,
    J: float = 1.0,
    n_equil: int = 1000,
    n_measure: int = 2000,
) -> dict[str, np.ndarray]:
    """Sweep temperature and measure <E>/N^2 (energy per spin).

    Returns dict with arrays: T, energy_per_spin.
    """
    T_values = np.linspace(T_range[0], T_range[1], n_T)
    energy_values = []

    for i, T in enumerate(T_values):
        config = SimulationConfig(
            domain=Domain.ISING_MODEL,
            dt=1.0,
            n_steps=n_equil + n_measure,
            parameters={"N": float(N), "J": J, "h": 0.0, "T": T},
        )
        sim = IsingModel2D(config)
        result = sim.measure_equilibrium(
            n_equil_sweeps=n_equil,
            n_measure_sweeps=n_measure,
            seed=42 + i,
        )
        energy_values.append(result["energy_per_spin"])

        if (i + 1) % 10 == 0:
            logger.info(
                f"  T={T:.2f}: E/N^2={result['energy_per_spin']:.4f}"
            )

    return {
        "T": T_values,
        "energy_per_spin": np.array(energy_values),
        "N": N,
        "J": J,
    }


def run_ising_model_rediscovery(
    output_dir: str | Path = "output/rediscovery/ising_model",
    n_iterations: int = 40,
) -> dict:
    """Run the full 2D Ising model rediscovery pipeline.

    1. Sweep T, measure |<m>| to map the phase transition
    2. Sweep T, measure chi to locate T_c from susceptibility peak
    3. Sweep T, measure <E>/N^2
    4. Run PySR on magnetization curve to recover critical behavior

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    T_c_theory = IsingModel2D.critical_temperature(1.0)

    results: dict = {
        "domain": "ising_model",
        "targets": {
            "critical_temperature": f"T_c = {T_c_theory:.4f}",
            "onsager_magnetization": "M = (1 - sinh(2J/T)^{-4})^{1/8}",
            "susceptibility_peak": "chi diverges at T_c",
        },
    }

    # --- Part 1: Magnetization vs temperature ---
    logger.info("Part 1: Magnetization vs temperature sweep...")
    mag_data = generate_magnetization_vs_temperature(
        n_T=30, N=16, n_equil=500, n_measure=1000,
    )

    # Check ordered vs disordered phase
    T_low_mask = mag_data["T"] < T_c_theory - 0.5
    T_high_mask = mag_data["T"] > T_c_theory + 0.5
    if np.any(T_low_mask) and np.any(T_high_mask):
        m_low = np.mean(mag_data["magnetization"][T_low_mask])
        m_high = np.mean(mag_data["magnetization"][T_high_mask])
        results["magnetization_ordered"] = float(m_low)
        results["magnetization_disordered"] = float(m_high)
        logger.info(
            f"  Ordered phase <|m|>={m_low:.4f}, "
            f"Disordered <|m|>={m_high:.4f}"
        )

    # Correlation with Onsager theory (for T < T_c data)
    valid = mag_data["theory_magnetization"] > 0.01
    if np.sum(valid) > 3:
        corr = np.corrcoef(
            mag_data["magnetization"][valid],
            mag_data["theory_magnetization"][valid],
        )[0, 1]
        results["onsager_correlation"] = float(corr)
        logger.info(f"  Onsager correlation: {corr:.4f}")

    results["magnetization_sweep"] = {
        "n_T": len(mag_data["T"]),
        "T_range": [float(mag_data["T"].min()), float(mag_data["T"].max())],
        "N": mag_data["N"],
    }

    # --- Part 2: Susceptibility ---
    logger.info("Part 2: Susceptibility vs temperature...")
    chi_data = generate_susceptibility_data(
        n_T=30, N=16, n_equil=500, n_measure=1000,
    )

    # Locate T_c from susceptibility peak
    peak_idx = np.argmax(chi_data["susceptibility"])
    T_c_chi = float(chi_data["T"][peak_idx])
    results["T_c_from_chi"] = T_c_chi
    results["T_c_theory"] = float(T_c_theory)
    results["T_c_relative_error"] = float(
        abs(T_c_chi - T_c_theory) / T_c_theory
    )
    logger.info(
        f"  T_c from chi peak: {T_c_chi:.3f} "
        f"(theory: {T_c_theory:.3f}, "
        f"error: {abs(T_c_chi - T_c_theory) / T_c_theory:.1%})"
    )

    results["susceptibility_sweep"] = {
        "n_T": len(chi_data["T"]),
        "max_chi": float(np.max(chi_data["susceptibility"])),
        "T_at_max_chi": float(T_c_chi),
    }

    # --- Part 3: Energy ---
    logger.info("Part 3: Energy per spin vs temperature...")
    e_data = generate_energy_vs_temperature(
        n_T=30, N=16, n_equil=500, n_measure=1000,
    )
    results["energy_sweep"] = {
        "n_T": len(e_data["T"]),
        "E_min": float(np.min(e_data["energy_per_spin"])),
        "E_max": float(np.max(e_data["energy_per_spin"])),
    }

    # --- Part 4: PySR on magnetization ---
    logger.info("Part 4: Running PySR on magnetization curve...")
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use the subcritical region where M > 0
        subcritical = mag_data["magnetization"] > 0.05
        if np.sum(subcritical) > 5:
            X = mag_data["T"][subcritical].reshape(-1, 1)
            y = mag_data["magnetization"][subcritical]

            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["T_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square", "log", "exp"],
                max_complexity=15,
                populations=20,
                population_size=40,
            )
            results["pysr_magnetization"] = {
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
                results["pysr_magnetization"]["best"] = best.expression
                results["pysr_magnetization"]["best_r2"] = (
                    best.evidence.fit_r_squared
                )
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["pysr_magnetization"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "magnetization_data.npz",
        T=mag_data["T"],
        magnetization=mag_data["magnetization"],
        theory_magnetization=mag_data["theory_magnetization"],
    )
    np.savez(
        output_path / "susceptibility_data.npz",
        T=chi_data["T"],
        susceptibility=chi_data["susceptibility"],
    )
    np.savez(
        output_path / "energy_data.npz",
        T=e_data["T"],
        energy_per_spin=e_data["energy_per_spin"],
    )

    return results
