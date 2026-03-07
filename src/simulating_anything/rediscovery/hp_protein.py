"""HP lattice protein folding rediscovery / analysis.

Targets:
- Ground state energy vs chain length (E scales with N)
- Folding temperature T_f from specific heat peak (C = var(E)/T^2)
- Radius of gyration collapse at T_f
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.hp_protein import HPProteinSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_energy_vs_temperature_data(
    chain_length: int = 15,
    h_fraction: float = 0.5,
    n_T_values: int = 15,
    T_range: tuple[float, float] = (0.1, 5.0),
    n_equil: int = 500,
    n_measure: int = 300,
) -> dict[str, np.ndarray]:
    """Sweep temperature and measure average energy, specific heat, and Rg.

    For each temperature, equilibrate the chain with MC, then collect
    energy statistics to compute specific heat C = var(E)/T^2.

    Args:
        chain_length: Number of residues.
        h_fraction: Fraction of hydrophobic residues.
        n_T_values: Number of temperature values.
        T_range: Min and max temperature.
        n_equil: Equilibration MC sweeps.
        n_measure: Measurement MC sweeps.

    Returns:
        Dict with T values, mean energy, specific heat, and Rg.
    """
    T_values = np.linspace(T_range[0], T_range[1], n_T_values)
    mean_energies = []
    specific_heats = []
    mean_rg = []

    for i, T in enumerate(T_values):
        config = SimulationConfig(
            domain=Domain.HP_PROTEIN,
            dt=1.0,
            n_steps=1,
            parameters={
                "N": float(chain_length),
                "h_fraction": h_fraction,
                "T": T,
                "epsilon": 1.0,
            },
            seed=42,
        )
        sim = HPProteinSimulation(config)
        sim.reset()

        # Equilibrate
        for _ in range(n_equil):
            sim.step()

        # Measure: state layout is [x1,y1,...,xN,yN, energy, n_contacts, Rg, T]
        energies = []
        rgs = []
        energy_idx = 2 * chain_length
        rg_idx = 2 * chain_length + 2
        for _ in range(n_measure):
            state = sim.step()
            energies.append(state[energy_idx])
            rgs.append(state[rg_idx])

        energies_arr = np.array(energies)
        mean_energies.append(float(np.mean(energies_arr)))
        specific_heats.append(sim.specific_heat(energies_arr))
        mean_rg.append(float(np.mean(rgs)))

        if (i + 1) % 5 == 0:
            logger.info(
                f"  T={T:.2f}: <E>={np.mean(energies_arr):.2f}, "
                f"C={sim.specific_heat(energies_arr):.3f}, Rg={np.mean(rgs):.2f}"
            )

    return {
        "T_values": T_values,
        "mean_energy": np.array(mean_energies),
        "specific_heat": np.array(specific_heats),
        "mean_rg": np.array(mean_rg),
        "chain_length": chain_length,
    }


def generate_energy_vs_length_data(
    lengths: list[int] | None = None,
    h_fraction: float = 0.5,
    T: float = 0.5,
    n_equil: int = 1000,
    n_measure: int = 500,
) -> dict[str, np.ndarray]:
    """Sweep chain length and measure ground state energy.

    At low temperature, the system finds near-optimal conformations.
    Energy should scale with chain length.

    Args:
        lengths: List of chain lengths to try.
        h_fraction: Fraction of hydrophobic residues.
        T: Low temperature for near-ground-state sampling.
        n_equil: Equilibration sweeps.
        n_measure: Measurement sweeps.

    Returns:
        Dict with lengths and best/mean energies.
    """
    if lengths is None:
        lengths = [8, 10, 12, 14, 16, 18, 20]

    best_energies = []
    mean_energies = []
    n_h_residues = []

    for N in lengths:
        n_h = int(N * h_fraction)
        n_h_residues.append(n_h)

        config = SimulationConfig(
            domain=Domain.HP_PROTEIN,
            dt=1.0,
            n_steps=1,
            parameters={
                "N": float(N),
                "h_fraction": h_fraction,
                "T": T,
                "epsilon": 1.0,
            },
            seed=42,
        )
        sim = HPProteinSimulation(config)
        sim.reset()

        # Equilibrate at low T
        for _ in range(n_equil):
            sim.step()

        # Measure
        energies = []
        energy_idx = 2 * N
        for _ in range(n_measure):
            state = sim.step()
            energies.append(state[energy_idx])

        energies_arr = np.array(energies)
        best_energies.append(float(np.min(energies_arr)))
        mean_energies.append(float(np.mean(energies_arr)))

        logger.info(
            f"  N={N}: best E={np.min(energies_arr):.1f}, "
            f"<E>={np.mean(energies_arr):.2f}, n_H={n_h}"
        )

    return {
        "lengths": np.array(lengths, dtype=np.float64),
        "best_energy": np.array(best_energies),
        "mean_energy": np.array(mean_energies),
        "n_h_residues": np.array(n_h_residues, dtype=np.float64),
        "h_fraction": h_fraction,
        "T": T,
    }


def generate_folding_curve_data(
    chain_length: int = 15,
    h_fraction: float = 0.5,
    n_T_values: int = 20,
    T_range: tuple[float, float] = (0.1, 3.0),
    n_equil: int = 500,
    n_measure: int = 300,
) -> dict[str, np.ndarray | float]:
    """Generate folding curve: Rg vs T to find folding transition.

    The radius of gyration collapses sharply at the folding temperature,
    which coincides with the specific heat peak.

    Args:
        chain_length: Number of residues.
        h_fraction: Fraction of hydrophobic residues.
        n_T_values: Number of temperature points.
        T_range: Temperature range.
        n_equil: Equilibration sweeps.
        n_measure: Measurement sweeps.

    Returns:
        Dict with T values, Rg, and estimated T_f.
    """
    data = generate_energy_vs_temperature_data(
        chain_length=chain_length,
        h_fraction=h_fraction,
        n_T_values=n_T_values,
        T_range=T_range,
        n_equil=n_equil,
        n_measure=n_measure,
    )

    # Find folding temperature as specific heat peak
    Cv = data["specific_heat"]
    T_vals = data["T_values"]
    peak_idx = np.argmax(Cv)
    T_fold = float(T_vals[peak_idx])

    return {
        "T_values": data["T_values"],
        "specific_heat": data["specific_heat"],
        "mean_rg": data["mean_rg"],
        "mean_energy": data["mean_energy"],
        "T_fold": T_fold,
        "Cv_max": float(Cv[peak_idx]),
    }


def run_hp_protein_analysis(
    output_dir: str | Path = "output/rediscovery/hp_protein",
    n_iterations: int = 40,
) -> dict:
    """Run the full HP protein folding analysis.

    1. Sweep temperature, measure specific heat to find T_f
    2. Sweep chain length, measure ground state energy scaling
    3. Run PySR on energy vs chain length

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "hp_protein",
        "targets": {
            "folding_temperature": "T_f from specific heat peak C = var(E)/T^2",
            "energy_vs_length": "Ground state energy scaling with N",
            "rg_collapse": "Radius of gyration collapse at T_f",
        },
    }

    # --- Part 1: Folding temperature from specific heat peak ---
    logger.info("Part 1: Sweeping temperature for specific heat peak...")
    fold_data = generate_folding_curve_data(
        chain_length=15,
        h_fraction=0.5,
        n_T_values=15,
        T_range=(0.1, 3.0),
        n_equil=500,
        n_measure=300,
    )
    results["folding"] = {
        "T_fold": fold_data["T_fold"],
        "Cv_max": fold_data["Cv_max"],
        "chain_length": 15,
        "h_fraction": 0.5,
        "rg_at_low_T": float(fold_data["mean_rg"][0]),
        "rg_at_high_T": float(fold_data["mean_rg"][-1]),
        "energy_at_low_T": float(fold_data["mean_energy"][0]),
        "energy_at_high_T": float(fold_data["mean_energy"][-1]),
    }
    logger.info(f"  Folding temperature: T_f = {fold_data['T_fold']:.2f}")
    logger.info(f"  Max specific heat: Cv = {fold_data['Cv_max']:.3f}")
    logger.info(
        f"  Rg collapse: {fold_data['mean_rg'][-1]:.2f} -> "
        f"{fold_data['mean_rg'][0]:.2f}"
    )

    # --- Part 2: Energy vs chain length ---
    logger.info("Part 2: Sweeping chain length for energy scaling...")
    length_data = generate_energy_vs_length_data(
        lengths=[8, 10, 12, 14, 16, 18, 20],
        h_fraction=0.5,
        T=0.5,
        n_equil=1000,
        n_measure=500,
    )
    results["energy_vs_length"] = {
        "lengths": length_data["lengths"].tolist(),
        "best_energy": length_data["best_energy"].tolist(),
        "mean_energy": length_data["mean_energy"].tolist(),
        "n_h_residues": length_data["n_h_residues"].tolist(),
    }

    # Linear fit: E ~ a*N + b
    N_arr = length_data["lengths"]
    E_arr = length_data["best_energy"]
    if len(N_arr) >= 3:
        coeffs = np.polyfit(N_arr, E_arr, 1)
        results["energy_vs_length"]["linear_fit_slope"] = float(coeffs[0])
        results["energy_vs_length"]["linear_fit_intercept"] = float(coeffs[1])
        logger.info(
            f"  Energy scaling: E ~ {coeffs[0]:.3f}*N + {coeffs[1]:.2f}"
        )

    # PySR: E = f(N) for ground state energy
    logger.info(
        f"  Running PySR for E = f(N) with {n_iterations} iterations..."
    )
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = length_data["lengths"].reshape(-1, 1)
        y = length_data["best_energy"]

        discoveries = run_symbolic_regression(
            X,
            y,
            variable_names=["N_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square"],
            max_complexity=10,
            populations=20,
            population_size=40,
        )

        results["energy_pysr"] = {
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
            results["energy_pysr"]["best"] = best.expression
            results["energy_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["energy_pysr"] = {"error": str(e)}

    # --- Part 3: Contact density analysis ---
    logger.info("Part 3: Analyzing contact density vs temperature...")
    T_low_e = fold_data["mean_energy"][0]
    T_high_e = fold_data["mean_energy"][-1]
    results["contact_analysis"] = {
        "energy_ratio_low_high": (
            float(T_low_e / T_high_e) if T_high_e != 0 else 0.0
        ),
        "n_T_values": len(fold_data["T_values"]),
    }

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "folding_data.npz",
        T_values=fold_data["T_values"],
        specific_heat=fold_data["specific_heat"],
        mean_rg=fold_data["mean_rg"],
        mean_energy=fold_data["mean_energy"],
    )
    np.savez(
        output_path / "length_data.npz",
        lengths=length_data["lengths"],
        best_energy=length_data["best_energy"],
        mean_energy=length_data["mean_energy"],
    )

    return results
