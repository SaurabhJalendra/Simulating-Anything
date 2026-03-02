"""Gierer-Meinhardt reaction-diffusion rediscovery.

Targets:
- Homogeneous steady state verification via parameter sweep
- Turing instability analysis: patterns form when D_h >> D_a
- Pattern wavelength scaling with diffusion ratio
- SINDy ODE recovery in the well-mixed limit (D_a=D_h=0)
- PySR wavelength = f(D_h) symbolic regression
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.gierer_meinhardt import GiererMeinhardtSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_DOMAIN = Domain.GIERER_MEINHARDT


def _make_config(
    dt: float = 0.1,
    n_steps: int = 500,
    N_grid: int = 128,
    L: float = 50.0,
    seed: int = 42,
    **params: float,
) -> SimulationConfig:
    """Build a SimulationConfig for the Gierer-Meinhardt domain."""
    defaults = {
        "rho_a": 0.1,
        "rho_h": 0.1,
        "mu_a": 0.02,
        "mu_h": 0.03,
        "D_a": 0.005,
        "D_h": 0.2,
        "rho_0": 0.001,
        "N_grid": float(N_grid),
        "L": L,
    }
    defaults.update(params)
    return SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters=defaults,
        seed=seed,
    )


def generate_steady_state_data(
    n_samples: int = 10,
) -> dict[str, np.ndarray]:
    """Verify homogeneous steady state by sweeping rho_a and mu_a.

    Runs the well-mixed system (no diffusion) to check convergence
    to the analytical steady state.
    """
    rho_a_values = np.linspace(0.05, 0.2, n_samples)
    mu_a_values = np.linspace(0.01, 0.05, n_samples)

    all_rho_a = []
    all_mu_a = []
    all_a0_theory = []
    all_h0_theory = []
    all_a0_sim = []
    all_h0_sim = []

    for rho_a in rho_a_values:
        for mu_a in mu_a_values:
            config = _make_config(
                dt=0.1, n_steps=100, N_grid=4, L=1.0,
                rho_a=rho_a, mu_a=mu_a,
                D_a=0.0, D_h=0.0,
            )
            sim = GiererMeinhardtSimulation(config)
            a0_theory, h0_theory = sim.homogeneous_steady_state()

            sim.reset()
            for _ in range(20000):
                sim.step()

            a0_sim = np.mean(sim.a_field)
            h0_sim = np.mean(sim.h_field)

            all_rho_a.append(rho_a)
            all_mu_a.append(mu_a)
            all_a0_theory.append(a0_theory)
            all_h0_theory.append(h0_theory)
            all_a0_sim.append(a0_sim)
            all_h0_sim.append(h0_sim)

    return {
        "rho_a": np.array(all_rho_a),
        "mu_a": np.array(all_mu_a),
        "a0_theory": np.array(all_a0_theory),
        "h0_theory": np.array(all_h0_theory),
        "a0_sim": np.array(all_a0_sim),
        "h0_sim": np.array(all_h0_sim),
    }


def generate_ode_data(
    rho_a: float = 0.1,
    rho_h: float = 0.1,
    mu_a: float = 0.02,
    mu_h: float = 0.03,
    rho_0: float = 0.001,
    n_steps: int = 10000,
    dt: float = 0.05,
) -> dict[str, np.ndarray]:
    """Generate well-mixed trajectory data for SINDy ODE recovery.

    Runs the GM model without diffusion (D_a=D_h=0) so the dynamics
    reduce to the ODE:
        da/dt = rho_a * a^2/h - mu_a*a + rho_0
        dh/dt = rho_h * a^2 - mu_h*h
    """
    config = _make_config(
        dt=dt, n_steps=n_steps, N_grid=4, L=1.0,
        rho_a=rho_a, rho_h=rho_h, mu_a=mu_a, mu_h=mu_h,
        rho_0=rho_0, D_a=0.0, D_h=0.0,
    )
    sim = GiererMeinhardtSimulation(config)
    sim.reset()

    # Start from a perturbed state for richer dynamics
    a0, h0 = sim.homogeneous_steady_state()
    sim._a[:] = a0 * 1.3
    sim._h[:] = h0 * 0.7
    sim._state = np.concatenate([sim._a, sim._h])

    states = []
    for _ in range(n_steps):
        sim.step()
        a_mean = np.mean(sim.a_field)
        h_mean = np.mean(sim.h_field)
        states.append([a_mean, h_mean])

    states_arr = np.array(states)
    return {
        "time": np.arange(1, n_steps + 1) * dt,
        "states": states_arr,
        "a": states_arr[:, 0],
        "h": states_arr[:, 1],
        "rho_a": rho_a,
        "rho_h": rho_h,
        "mu_a": mu_a,
        "mu_h": mu_h,
        "rho_0": rho_0,
    }


def generate_wavelength_data(
    n_Dh: int = 12,
    n_steps: int = 10000,
    N_grid: int = 128,
    L: float = 50.0,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep D_h and measure dominant pattern wavelength.

    Uses default reaction parameters and varies D_h to see how
    the Turing pattern wavelength scales with inhibitor diffusion.
    """
    D_h_values = np.linspace(0.1, 1.0, n_Dh)
    wavelengths = []
    energies = []

    for i, D_h in enumerate(D_h_values):
        config = _make_config(
            dt=dt, n_steps=n_steps, N_grid=N_grid, L=L,
            D_h=float(D_h), seed=42 + i,
        )
        sim = GiererMeinhardtSimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        wl = sim.dominant_wavelength()
        energy = sim.compute_pattern_energy()
        wavelengths.append(wl)
        energies.append(energy)

        if (i + 1) % 4 == 0:
            logger.info(
                f"  D_h={D_h:.3f}: wavelength={wl:.2f}, energy={energy:.6f}"
            )

    return {
        "D_h": D_h_values,
        "wavelength": np.array(wavelengths),
        "energy": np.array(energies),
    }


def generate_pattern_data(
    n_Dh: int = 8,
    n_Da: int = 6,
    n_steps: int = 5000,
    N_grid: int = 64,
    L: float = 50.0,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep D_a and D_h to detect Turing pattern formation.

    For each (D_a, D_h) pair, run the simulation and measure whether
    spatial patterns have formed (via spatial heterogeneity of a).
    """
    D_a_values = np.logspace(-3, -1, n_Da)  # 0.001 to 0.1
    D_h_values = np.logspace(-1.5, 0.5, n_Dh)  # ~0.03 to ~3.16

    all_Da = []
    all_Dh = []
    all_het_a = []
    all_patterned = []

    for i, D_a in enumerate(D_a_values):
        for j, D_h in enumerate(D_h_values):
            config = _make_config(
                dt=dt, n_steps=n_steps, N_grid=N_grid, L=L,
                D_a=float(D_a), D_h=float(D_h),
                seed=i * 100 + j,
            )
            sim = GiererMeinhardtSimulation(config)
            sim.reset()

            for _ in range(n_steps):
                sim.step()

            het_a = sim.spatial_heterogeneity_a

            all_Da.append(D_a)
            all_Dh.append(D_h)
            all_het_a.append(het_a)
            all_patterned.append(het_a > 0.05)

        if (i + 1) % 3 == 0:
            logger.info(f"  D_a sweep {i + 1}/{n_Da} complete")

    return {
        "D_a": np.array(all_Da),
        "D_h": np.array(all_Dh),
        "heterogeneity_a": np.array(all_het_a),
        "patterned": np.array(all_patterned),
    }


def run_gierer_meinhardt_rediscovery(
    output_dir: str | Path = "output/rediscovery/gierer_meinhardt",
    n_iterations: int = 40,
) -> dict:
    """Run Gierer-Meinhardt rediscovery pipeline.

    1. Turing instability analysis
    2. Verify homogeneous steady state via simulation
    3. SINDy ODE recovery (well-mixed limit)
    4. Wavelength vs D_h sweep with PySR

    Returns:
        Results dictionary with Turing analysis, steady state verification,
        SINDy ODE recovery, and PySR wavelength discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "gierer_meinhardt",
        "targets": {
            "steady_state": "a0 = (rho_a*mu_h/rho_h + rho_0)/mu_a, "
                            "h0 = rho_h*a0^2/mu_h",
            "turing": "Pattern formation for D_h >> D_a",
            "ode": "da/dt = rho_a*a^2/h - mu_a*a + rho_0, "
                   "dh/dt = rho_h*a^2 - mu_h*h",
        },
    }

    # --- Part 1: Turing analysis ---
    logger.info("Part 1: Turing instability analysis...")
    config = _make_config(dt=0.1, n_steps=100, N_grid=64)
    sim = GiererMeinhardtSimulation(config)
    turing = sim.turing_analysis()
    results["turing_analysis"] = {
        k: float(v) if isinstance(v, (int, float, np.floating)) else bool(v)
        for k, v in turing.items()
    }
    logger.info(f"  Turing unstable: {turing['turing_unstable']}")
    if "wavelength_most_unstable" in turing:
        logger.info(
            f"  Most unstable wavelength: "
            f"{turing['wavelength_most_unstable']:.2f}"
        )

    # --- Part 2: Steady state verification ---
    logger.info("Part 2: Steady state verification (small sweep)...")
    ss_data = generate_steady_state_data(n_samples=5)
    a_err = np.abs(ss_data["a0_sim"] - ss_data["a0_theory"])
    h_err = np.abs(ss_data["h0_sim"] - ss_data["h0_theory"])
    # Relative errors where theory is non-zero
    a_rel = a_err / np.maximum(np.abs(ss_data["a0_theory"]), 1e-15)
    h_rel = h_err / np.maximum(np.abs(ss_data["h0_theory"]), 1e-15)
    results["steady_state"] = {
        "n_samples": len(ss_data["rho_a"]),
        "a_mean_rel_error": float(np.mean(a_rel)),
        "h_mean_rel_error": float(np.mean(h_rel)),
        "a_max_rel_error": float(np.max(a_rel)),
        "h_max_rel_error": float(np.max(h_rel)),
    }
    logger.info(f"  a mean rel error: {np.mean(a_rel):.6f}")
    logger.info(f"  h mean rel error: {np.mean(h_rel):.6f}")

    # --- Part 3: SINDy ODE recovery ---
    logger.info("Part 3: SINDy ODE recovery (well-mixed limit)...")
    ode_data = generate_ode_data(n_steps=10000, dt=0.05)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=0.05,
            feature_names=["a", "h"],
            threshold=0.01,
            poly_degree=3,
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

    # --- Part 4: Wavelength vs D_h with PySR ---
    logger.info("Part 4: Wavelength vs D_h sweep...")
    wl_data = generate_wavelength_data(
        n_Dh=12, n_steps=5000, N_grid=128, L=50.0, dt=0.1,
    )

    valid = (wl_data["wavelength"] > 0) & np.isfinite(wl_data["wavelength"])
    valid &= wl_data["energy"] > 1e-6
    n_patterned = int(np.sum(valid))
    results["wavelength_sweep"] = {
        "n_total": len(wl_data["D_h"]),
        "n_patterned": n_patterned,
    }

    if n_patterned > 3:
        Dh_valid = wl_data["D_h"][valid]
        wl_valid = wl_data["wavelength"][valid]

        # Test wavelength ~ sqrt(D_h) correlation
        sqrt_Dh = np.sqrt(Dh_valid)
        corr = float(np.corrcoef(wl_valid, sqrt_Dh)[0, 1])
        results["wavelength_sweep"]["correlation_sqrt_Dh"] = corr
        logger.info(f"  Wavelength ~ sqrt(D_h) correlation: {corr:.4f}")

        # PySR: find wavelength = f(D_h)
        try:
            from simulating_anything.analysis.symbolic_regression import (
                run_symbolic_regression,
            )

            X = Dh_valid.reshape(-1, 1)
            y = wl_valid

            logger.info("  Running PySR: wavelength = f(D_h)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["Dh"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square"],
                max_complexity=10,
                populations=20,
                population_size=40,
            )
            results["wavelength_pysr"] = {
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
                results["wavelength_pysr"]["best"] = best.expression
                results["wavelength_pysr"]["best_r2"] = (
                    best.evidence.fit_r_squared
                )
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
        except Exception as e:
            logger.warning(f"PySR failed: {e}")
            results["wavelength_pysr"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
