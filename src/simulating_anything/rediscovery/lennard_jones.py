"""Lennard-Jones molecular dynamics rediscovery.

Targets:
- Equation of state: P = rho * T (ideal gas limit at low density)
- Temperature from kinetic energy: T = 2*KE / N_dof
- Diffusion coefficient D(T) from mean square displacement
- Total energy conservation (NVE ensemble)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.lennard_jones import LennardJonesSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_eos_data(
    n_T_values: int = 10,
    n_rho_values: int = 5,
    N: int = 25,
    n_equil: int = 200,
    n_measure: int = 200,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep temperature and density, measure pressure.

    At low density, should recover ideal gas law P = rho * T.
    At higher density, LJ corrections appear.

    Args:
        n_T_values: Number of temperature values.
        n_rho_values: Number of density values (via box size L).
        N: Number of particles.
        n_equil: Equilibration steps.
        n_measure: Measurement steps.
        dt: Timestep.

    Returns:
        Dict with T, rho, P, KE arrays and particle count.
    """
    T_values = np.linspace(0.5, 3.0, n_T_values)
    # Low density regime: rho = N/L^2, large L = low density
    L_values = np.linspace(10.0, 20.0, n_rho_values)

    all_T = []
    all_rho = []
    all_P = []
    all_KE = []
    all_T_measured = []

    for T in T_values:
        for L in L_values:
            config = SimulationConfig(
                domain=Domain.LENNARD_JONES,
                dt=dt,
                n_steps=1,
                parameters={
                    "N": float(N),
                    "L": L,
                    "T": T,
                    "r_cut": 2.5,
                },
                seed=42,
            )
            sim = LennardJonesSimulation(config)
            sim.reset(seed=42)

            # Equilibrate
            for _ in range(n_equil):
                sim.step()

            # Measure
            pressures = []
            kes = []
            temps = []
            for _ in range(n_measure):
                sim.step()
                pressures.append(sim.pressure())
                kes.append(sim.kinetic_energy())
                temps.append(sim.temperature())

            rho = N / L ** 2
            all_T.append(T)
            all_rho.append(rho)
            all_P.append(float(np.mean(pressures)))
            all_KE.append(float(np.mean(kes)))
            all_T_measured.append(float(np.mean(temps)))

    return {
        "T_target": np.array(all_T),
        "rho": np.array(all_rho),
        "P": np.array(all_P),
        "KE": np.array(all_KE),
        "T_measured": np.array(all_T_measured),
        "N": N,
    }


def generate_diffusion_data(
    n_T_values: int = 8,
    N: int = 25,
    L: float = 12.0,
    n_equil: int = 200,
    n_track: int = 300,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep temperature and measure diffusion coefficient from MSD.

    D = lim_{t->inf} MSD(t) / (2*d*t), where d=2 in 2D.
    Fit the linear slope of MSD(t) at long times.

    Args:
        n_T_values: Number of temperature values.
        N: Number of particles.
        L: Box size.
        n_equil: Equilibration steps.
        n_track: Steps to track MSD.
        dt: Timestep.

    Returns:
        Dict with T values and diffusion coefficients.
    """
    T_values = np.linspace(0.5, 3.0, n_T_values)
    diffusion_coeffs = []

    for i, T in enumerate(T_values):
        config = SimulationConfig(
            domain=Domain.LENNARD_JONES,
            dt=dt,
            n_steps=1,
            parameters={
                "N": float(N),
                "L": L,
                "T": T,
                "r_cut": 2.5,
            },
            seed=42,
        )
        sim = LennardJonesSimulation(config)
        sim.reset(seed=42)

        # Equilibrate
        for _ in range(n_equil):
            sim.step()

        # Track unwrapped positions for MSD
        # Note: simulation uses PBC, so we track displacements
        ref_pos = sim._pos.copy()
        unwrapped = sim._pos.copy()
        msd_vals = [0.0]

        for step_j in range(n_track):
            old_pos = sim._pos.copy()
            sim.step()
            new_pos = sim._pos.copy()

            # Compute displacement handling PBC
            disp = new_pos - old_pos
            disp -= L * np.round(disp / L)
            unwrapped += disp

            diff = unwrapped - ref_pos
            msd_vals.append(float(np.mean(np.sum(diff ** 2, axis=1))))

        msd_arr = np.array(msd_vals)
        times = np.arange(len(msd_arr)) * dt

        # Fit D from linear regime (skip initial ballistic, use second half)
        half = len(times) // 2
        if half > 2:
            coeffs = np.polyfit(times[half:], msd_arr[half:], 1)
            # MSD = 2*d*D*t in d dimensions, d=2
            D = coeffs[0] / (2 * 2)
            diffusion_coeffs.append(max(D, 0.0))
        else:
            diffusion_coeffs.append(0.0)

        if (i + 1) % 4 == 0:
            logger.info(
                f"  Diffusion T={T:.2f}: D={diffusion_coeffs[-1]:.4f}"
            )

    return {
        "T_values": T_values,
        "D_values": np.array(diffusion_coeffs),
        "N": N,
        "L": L,
    }


def generate_energy_conservation_data(
    N: int = 25,
    L: float = 12.0,
    T: float = 1.0,
    n_equil: int = 200,
    n_track: int = 500,
    dt: float = 0.002,
) -> dict[str, np.ndarray | float]:
    """Track total energy to verify NVE conservation.

    Velocity Verlet is symplectic, so energy should be well-conserved.

    Args:
        N: Number of particles.
        L: Box size.
        T: Initial temperature.
        n_equil: Equilibration steps.
        n_track: Tracking steps.
        dt: Timestep (small for energy conservation).

    Returns:
        Dict with energy time series and drift statistics.
    """
    config = SimulationConfig(
        domain=Domain.LENNARD_JONES,
        dt=dt,
        n_steps=1,
        parameters={
            "N": float(N),
            "L": L,
            "T": T,
            "r_cut": 2.5,
        },
        seed=42,
    )
    sim = LennardJonesSimulation(config)
    sim.reset(seed=42)

    # Equilibrate
    for _ in range(n_equil):
        sim.step()

    E0 = sim.total_energy()
    energies = [E0]
    ke_vals = [sim.kinetic_energy()]
    pe_vals = [sim.potential_energy()]

    for _ in range(n_track):
        sim.step()
        energies.append(sim.total_energy())
        ke_vals.append(sim.kinetic_energy())
        pe_vals.append(sim.potential_energy())

    energies_arr = np.array(energies)
    drift = np.abs(energies_arr - E0) / (np.abs(E0) + 1e-15)

    return {
        "energies": energies_arr,
        "ke": np.array(ke_vals),
        "pe": np.array(pe_vals),
        "E0": float(E0),
        "max_drift": float(np.max(drift)),
        "mean_drift": float(np.mean(drift)),
        "final_drift": float(drift[-1]),
    }


def run_lennard_jones_rediscovery(
    output_dir: str | Path = "output/rediscovery/lennard_jones",
    n_iterations: int = 40,
) -> dict:
    """Run the full Lennard-Jones molecular dynamics rediscovery.

    1. Sweep T and density, measure P, check ideal gas law P = rho*T
    2. Run PySR on T_measured = f(KE) to recover T = 2*KE/N_dof
    3. Measure diffusion coefficient D(T)
    4. Verify energy conservation (NVE ensemble)

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "lennard_jones",
        "targets": {
            "ideal_gas_law": "P = rho * T (low density limit)",
            "temperature_ke": "T = 2*KE / N_dof",
            "diffusion": "D(T) increases with temperature",
            "energy_conservation": "Total energy conserved in NVE",
        },
    }

    # --- Part 1: Equation of state ---
    logger.info("Part 1: Generating equation of state data...")
    eos_data = generate_eos_data(
        n_T_values=8, n_rho_values=3, N=25, n_equil=200, n_measure=200, dt=0.005,
    )

    # Check ideal gas law: P/(rho*T) at lowest density
    rho_min = np.min(eos_data["rho"])
    low_density = eos_data["rho"] == rho_min
    if np.any(low_density):
        P_low = eos_data["P"][low_density]
        T_low = eos_data["T_target"][low_density]
        rho_low = eos_data["rho"][low_density]
        ratio = P_low / (rho_low * T_low)
        results["ideal_gas"] = {
            "mean_ratio": float(np.mean(ratio)),
            "std_ratio": float(np.std(ratio)),
            "rho": float(rho_min),
            "note": "P/(rho*T) ~ 1 for ideal gas at low density",
        }
        logger.info(
            f"  Ideal gas ratio P/(rho*T): {np.mean(ratio):.3f} +/- "
            f"{np.std(ratio):.3f} at rho={rho_min:.4f}"
        )

    # Temperature from KE: T_measured vs T_target correlation
    T_corr = float(np.corrcoef(eos_data["T_target"], eos_data["T_measured"])[0, 1])
    n_dof = 2 * (eos_data["N"] - 1)
    T_from_ke = 2 * eos_data["KE"] / n_dof
    ke_corr = float(np.corrcoef(eos_data["T_target"], T_from_ke)[0, 1])
    results["temperature_from_ke"] = {
        "T_measured_correlation": T_corr,
        "T_from_2KE_Ndof_correlation": ke_corr,
        "N_dof": n_dof,
    }
    logger.info(f"  T vs T_measured correlation: {T_corr:.4f}")
    logger.info(f"  T vs 2*KE/N_dof correlation: {ke_corr:.4f}")

    # PySR: T = f(KE, N) -- should find T ~ KE / N
    logger.info(
        f"  Running PySR for T = f(KE) with {n_iterations} iterations..."
    )
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = eos_data["KE"].reshape(-1, 1)
        y = eos_data["T_measured"]

        discoveries = run_symbolic_regression(
            X,
            y,
            variable_names=["KE_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square"],
            max_complexity=10,
            populations=20,
            population_size=40,
        )

        results["temperature_pysr"] = {
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
            results["temperature_pysr"]["best"] = best.expression
            results["temperature_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["temperature_pysr"] = {"error": str(e)}

    # --- Part 2: Diffusion coefficient vs temperature ---
    logger.info("Part 2: Measuring diffusion coefficient D(T)...")
    diff_data = generate_diffusion_data(
        n_T_values=8, N=25, L=12.0, n_equil=200, n_track=300, dt=0.005,
    )

    # D should increase with T
    D_T_corr = float(
        np.corrcoef(diff_data["T_values"], diff_data["D_values"])[0, 1]
    )
    results["diffusion"] = {
        "n_T_values": len(diff_data["T_values"]),
        "D_range": [
            float(np.min(diff_data["D_values"])),
            float(np.max(diff_data["D_values"])),
        ],
        "D_T_correlation": D_T_corr,
        "T_range": [
            float(np.min(diff_data["T_values"])),
            float(np.max(diff_data["T_values"])),
        ],
    }
    logger.info(
        f"  D range: [{np.min(diff_data['D_values']):.4f}, "
        f"{np.max(diff_data['D_values']):.4f}]"
    )
    logger.info(f"  D-T correlation: {D_T_corr:.4f}")

    # PySR: D = f(T) -- should find D ~ sqrt(T) or D ~ T
    logger.info(
        f"  Running PySR for D = f(T) with {n_iterations} iterations..."
    )
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X_diff = diff_data["T_values"].reshape(-1, 1)
        y_diff = diff_data["D_values"]

        # Only fit if we have reasonable data
        if np.all(np.isfinite(y_diff)) and np.any(y_diff > 0):
            discoveries_d = run_symbolic_regression(
                X_diff,
                y_diff,
                variable_names=["T_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square", "exp"],
                max_complexity=10,
                populations=20,
                population_size=40,
            )

            results["diffusion_pysr"] = {
                "n_discoveries": len(discoveries_d),
                "discoveries": [
                    {
                        "expression": d.expression,
                        "r_squared": d.evidence.fit_r_squared,
                    }
                    for d in discoveries_d[:5]
                ],
            }
            if discoveries_d:
                best_d = discoveries_d[0]
                results["diffusion_pysr"]["best"] = best_d.expression
                results["diffusion_pysr"]["best_r2"] = best_d.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best_d.expression} "
                    f"(R2={best_d.evidence.fit_r_squared:.6f})"
                )
        else:
            results["diffusion_pysr"] = {"error": "Invalid diffusion data"}

    except Exception as e:
        logger.warning(f"PySR for D(T) failed: {e}")
        results["diffusion_pysr"] = {"error": str(e)}

    # --- Part 3: Energy conservation ---
    logger.info("Part 3: Verifying energy conservation (NVE)...")
    energy_data = generate_energy_conservation_data(
        N=25, L=12.0, T=1.0, n_equil=200, n_track=500, dt=0.002,
    )
    results["energy_conservation"] = {
        "E0": energy_data["E0"],
        "max_drift": energy_data["max_drift"],
        "mean_drift": energy_data["mean_drift"],
        "final_drift": energy_data["final_drift"],
    }
    logger.info(
        f"  Energy drift: max={energy_data['max_drift']:.2e}, "
        f"mean={energy_data['mean_drift']:.2e}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "eos_data.npz",
        T_target=eos_data["T_target"],
        rho=eos_data["rho"],
        P=eos_data["P"],
        KE=eos_data["KE"],
        T_measured=eos_data["T_measured"],
    )
    np.savez(
        output_path / "diffusion_data.npz",
        T_values=diff_data["T_values"],
        D_values=diff_data["D_values"],
    )
    np.savez(
        output_path / "energy_data.npz",
        energies=energy_data["energies"],
        ke=energy_data["ke"],
        pe=energy_data["pe"],
    )

    return results
