"""Boltzmann gas 2D rediscovery.

Targets:
- Ideal gas law: PV = NkT (k_B = 1)
- Maxwell-Boltzmann speed distribution: f(v) = (m*v/kT)*exp(-m*v^2/(2kT))
- Pressure proportional to temperature at fixed N, V
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.boltzmann_gas import BoltzmannGas2D
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_pressure_temperature_data(
    n_T: int = 20,
    T_range: tuple[float, float] = (0.5, 10.0),
    N: int = 100,
    L: float = 10.0,
    equil_steps: int = 2000,
    measure_steps: int = 3000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep temperature at fixed N and L, measure average pressure.

    Returns dict with arrays: T, pressure, theory_pressure.
    Theory: P = N*T / (L^2) with k_B = 1, V = L^2.
    """
    T_values = np.linspace(T_range[0], T_range[1], n_T)
    pressures = []

    for i, T in enumerate(T_values):
        config = SimulationConfig(
            domain=Domain.BOLTZMANN_GAS,
            dt=dt,
            n_steps=equil_steps + measure_steps,
            parameters={
                "N": float(N),
                "L": L,
                "T": T,
                "particle_radius": 0.05,
                "m": 1.0,
            },
        )
        sim = BoltzmannGas2D(config)
        sim.reset(seed=42 + i)

        # Equilibrate
        for _ in range(equil_steps):
            sim.step()

        # Reset pressure accumulator and measure
        sim.reset_pressure()
        for _ in range(measure_steps):
            sim.step()

        pressures.append(sim.pressure)

        if (i + 1) % 5 == 0:
            V = L ** 2
            theory_P = N * T / V
            logger.info(
                f"  T={T:.1f}: P={pressures[-1]:.4f}, "
                f"theory={theory_P:.4f}"
            )

    V = L ** 2
    return {
        "T": T_values,
        "pressure": np.array(pressures),
        "theory_pressure": N * T_values / V,
        "N": N,
        "L": L,
        "V": V,
    }


def generate_speed_distribution_data(
    T: float = 2.0,
    N: int = 200,
    L: float = 15.0,
    equil_steps: int = 3000,
    sample_steps: int = 5000,
    sample_interval: int = 50,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Collect speed samples after equilibration and compare to MB.

    Returns dict with speed_samples and MB theory parameters.
    """
    config = SimulationConfig(
        domain=Domain.BOLTZMANN_GAS,
        dt=dt,
        n_steps=equil_steps + sample_steps,
        parameters={
            "N": float(N),
            "L": L,
            "T": T,
            "particle_radius": 0.05,
            "m": 1.0,
        },
    )
    sim = BoltzmannGas2D(config)
    sim.reset(seed=123)

    # Equilibrate
    for _ in range(equil_steps):
        sim.step()

    # Collect speed samples at intervals
    all_speeds = []
    for step in range(sample_steps):
        sim.step()
        if step % sample_interval == 0:
            all_speeds.append(sim.speeds().copy())

    speed_samples = np.concatenate(all_speeds)

    return {
        "speed_samples": speed_samples,
        "T": T,
        "m": 1.0,
        "N": N,
    }


def generate_pv_nkt_data(
    n_points: int = 30,
    equil_steps: int = 2000,
    measure_steps: int = 3000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep N and T, measure P*V, verify PV = N*k*T.

    Returns dict with arrays: N_vals, T_vals, PV, NkT.
    """
    rng = np.random.default_rng(42)
    N_vals = rng.integers(30, 150, size=n_points)
    T_vals = rng.uniform(0.5, 8.0, size=n_points)
    L = 10.0
    V = L ** 2

    PV_measured = []
    NkT_theory = []

    for i in range(n_points):
        N_i = int(N_vals[i])
        T_i = float(T_vals[i])

        config = SimulationConfig(
            domain=Domain.BOLTZMANN_GAS,
            dt=dt,
            n_steps=equil_steps + measure_steps,
            parameters={
                "N": float(N_i),
                "L": L,
                "T": T_i,
                "particle_radius": 0.03,
                "m": 1.0,
            },
        )
        sim = BoltzmannGas2D(config)
        sim.reset(seed=100 + i)

        # Equilibrate
        for _ in range(equil_steps):
            sim.step()

        # Measure pressure
        sim.reset_pressure()
        for _ in range(measure_steps):
            sim.step()

        P = sim.pressure
        PV_measured.append(P * V)
        NkT_theory.append(N_i * T_i)  # k_B = 1

        if (i + 1) % 10 == 0:
            logger.info(
                f"  [{i+1}/{n_points}] N={N_i}, T={T_i:.1f}: "
                f"PV={PV_measured[-1]:.2f}, NkT={NkT_theory[-1]:.2f}"
            )

    return {
        "N": N_vals.astype(float),
        "T": T_vals,
        "PV": np.array(PV_measured),
        "NkT": np.array(NkT_theory),
        "L": L,
        "V": V,
    }


def run_boltzmann_gas_rediscovery(
    output_dir: str | Path = "output/rediscovery/boltzmann_gas",
    n_iterations: int = 40,
) -> dict:
    """Run the full Boltzmann gas rediscovery.

    1. Sweep T, measure P to verify P proportional to T
    2. Sweep N and T, measure PV to recover PV = NkT
    3. Run PySR on P vs N*T/V
    4. Collect speed distribution and compare to Maxwell-Boltzmann

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "boltzmann_gas",
        "targets": {
            "ideal_gas_law": "PV = NkT (k_B = 1)",
            "maxwell_boltzmann": "f(v) = (m*v/kT)*exp(-m*v^2/(2kT))",
            "pressure_vs_T": "P proportional to T at fixed N, V",
        },
    }

    # --- Part 1: P vs T sweep ---
    logger.info("Part 1: Pressure vs temperature sweep...")
    pt_data = generate_pressure_temperature_data(
        n_T=20, equil_steps=2000, measure_steps=3000
    )

    valid = np.isfinite(pt_data["pressure"]) & (pt_data["pressure"] > 0)
    if np.sum(valid) > 5:
        corr = np.corrcoef(
            pt_data["T"][valid], pt_data["pressure"][valid]
        )[0, 1]
        rel_err = np.abs(
            pt_data["pressure"][valid] - pt_data["theory_pressure"][valid]
        ) / pt_data["theory_pressure"][valid]
        results["pressure_vs_T"] = {
            "n_samples": int(np.sum(valid)),
            "correlation": float(corr),
            "mean_relative_error": float(np.mean(rel_err)),
        }
        logger.info(
            f"  P-T correlation: {corr:.4f}, "
            f"mean rel error: {np.mean(rel_err):.4f}"
        )

    # --- Part 2: PV = NkT verification ---
    logger.info("Part 2: PV = NkT verification...")
    pv_data = generate_pv_nkt_data(
        n_points=30, equil_steps=2000, measure_steps=3000
    )

    pv_valid = np.isfinite(pv_data["PV"]) & (pv_data["PV"] > 0)
    if np.sum(pv_valid) > 5:
        corr_pv = np.corrcoef(
            pv_data["PV"][pv_valid], pv_data["NkT"][pv_valid]
        )[0, 1]
        ratio = pv_data["PV"][pv_valid] / pv_data["NkT"][pv_valid]
        results["pv_nkt"] = {
            "n_samples": int(np.sum(pv_valid)),
            "correlation": float(corr_pv),
            "mean_ratio": float(np.mean(ratio)),
            "std_ratio": float(np.std(ratio)),
        }
        logger.info(
            f"  PV/NkT ratio: {np.mean(ratio):.4f} +/- {np.std(ratio):.4f}"
        )

    # --- Part 3: PySR on pressure data ---
    logger.info("Part 3: Running PySR on pressure = f(N*T/V)...")
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use the PV = NkT data: P = f(N, T) with fixed V
        V = pv_data["V"]
        X = np.column_stack([
            pv_data["N"][pv_valid],
            pv_data["T"][pv_valid],
        ])
        y = pv_data["PV"][pv_valid] / V  # pressure

        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["N_", "T_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square"],
            max_complexity=10,
            populations=15,
            population_size=30,
        )
        results["pysr_pressure"] = {
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
            results["pysr_pressure"]["best"] = best.expression
            results["pysr_pressure"]["best_r2"] = (
                best.evidence.fit_r_squared
            )
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["pysr_pressure"] = {"error": str(e)}

    # --- Part 4: Speed distribution ---
    logger.info("Part 4: Speed distribution analysis...")
    speed_data = generate_speed_distribution_data(
        T=2.0, N=200, equil_steps=3000, sample_steps=5000
    )

    speeds = speed_data["speed_samples"]
    T_measured = speed_data["T"]
    m = speed_data["m"]

    # Fit: for 2D MB, <v^2> = 2kT/m
    mean_v_sq = np.mean(speeds ** 2)
    T_from_speeds = m * mean_v_sq / 2.0
    mean_speed = np.mean(speeds)
    # Theoretical mean speed in 2D MB: <v> = sqrt(pi*kT/(2m))
    theory_mean_speed = np.sqrt(np.pi * T_measured / (2.0 * m))

    results["speed_distribution"] = {
        "n_samples": len(speeds),
        "mean_speed": float(mean_speed),
        "theory_mean_speed": float(theory_mean_speed),
        "mean_v_squared": float(mean_v_sq),
        "T_from_speeds": float(T_from_speeds),
        "T_target": float(T_measured),
        "T_relative_error": float(
            abs(T_from_speeds - T_measured) / T_measured
        ),
    }
    logger.info(
        f"  T from speeds: {T_from_speeds:.3f} "
        f"(target: {T_measured:.3f}, "
        f"error: {abs(T_from_speeds - T_measured) / T_measured:.2%})"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "pt_data.npz",
        T=pt_data["T"],
        pressure=pt_data["pressure"],
        theory_pressure=pt_data["theory_pressure"],
    )
    np.savez(
        output_path / "pv_data.npz",
        N=pv_data["N"],
        T=pv_data["T"],
        PV=pv_data["PV"],
        NkT=pv_data["NkT"],
    )
    np.savez(
        output_path / "speed_data.npz",
        speed_samples=speeds,
    )

    return results
