"""Schnakenberg reaction-diffusion rediscovery.

Targets:
- Homogeneous steady state: u* = a + b, v* = b / (a + b)^2
- Turing instability onset and pattern wavelength scaling
- ODE recovery at homogeneous level via SINDy (well-mixed limit)
- Wavelength vs D_v relationship via PySR
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.schnakenberg import SchnakenbergSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use the SCHNAKENBERG domain enum value
_DOMAIN = Domain.SCHNAKENBERG


def generate_steady_state_data(
    n_samples: int = 20,
) -> dict[str, np.ndarray]:
    """Generate steady state data by sweeping a and b.

    Returns measured steady states vs theoretical predictions.
    """
    a_values = np.linspace(0.05, 0.3, n_samples)
    b_values = np.linspace(0.5, 1.5, n_samples)

    all_a = []
    all_b = []
    all_u_star_theory = []
    all_v_star_theory = []
    all_u_star_sim = []
    all_v_star_sim = []

    for a_val in a_values:
        for b_val in b_values:
            u_theory = a_val + b_val
            v_theory = b_val / (u_theory**2)

            # Run a well-mixed (no diffusion) simulation to verify
            config = SimulationConfig(
                domain=_DOMAIN,
                dt=0.01,
                n_steps=1000,
                parameters={
                    "a": a_val, "b": b_val,
                    "D_u": 0.0, "D_v": 0.0,
                    "N": 4.0, "L": 1.0,
                },
            )
            sim = SchnakenbergSimulation(config)
            sim.reset()

            # Evolve to steady state
            for _ in range(10000):
                sim.step()

            u_final = np.mean(sim.u_field)
            v_final = np.mean(sim.v_field)

            all_a.append(a_val)
            all_b.append(b_val)
            all_u_star_theory.append(u_theory)
            all_v_star_theory.append(v_theory)
            all_u_star_sim.append(u_final)
            all_v_star_sim.append(v_final)

    return {
        "a": np.array(all_a),
        "b": np.array(all_b),
        "u_star_theory": np.array(all_u_star_theory),
        "v_star_theory": np.array(all_v_star_theory),
        "u_star_sim": np.array(all_u_star_sim),
        "v_star_sim": np.array(all_v_star_sim),
    }


def generate_wavelength_data(
    n_Dv: int = 15,
    n_steps: int = 5000,
    N: int = 64,
    L: float = 50.0,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep D_v and measure dominant pattern wavelength.

    Uses default a=0.1, b=0.9, D_u=1.0 and varies D_v.
    """
    D_v_values = np.linspace(10.0, 80.0, n_Dv)
    wavelengths = []
    energies = []

    for i, D_v in enumerate(D_v_values):
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "a": 0.1, "b": 0.9,
                "D_u": 1.0, "D_v": float(D_v),
                "N": float(N), "L": L,
            },
            seed=42,
        )
        sim = SchnakenbergSimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        wl = sim.compute_pattern_wavelength()
        energy = sim.compute_pattern_energy()
        wavelengths.append(wl)
        energies.append(energy)

        if (i + 1) % 5 == 0:
            logger.info(f"  D_v={D_v:.1f}: wavelength={wl:.2f}, energy={energy:.6f}")

    return {
        "D_v": D_v_values,
        "wavelength": np.array(wavelengths),
        "energy": np.array(energies),
    }


def generate_ode_data(
    a: float = 0.1,
    b: float = 0.9,
    n_steps: int = 10000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Generate well-mixed trajectory data for SINDy ODE recovery.

    Runs the Schnakenberg model without diffusion (D_u=D_v=0) so the
    dynamics reduce to the ODE:
        du/dt = a - u + u^2*v
        dv/dt = b - u^2*v
    """
    config = SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a, "b": b,
            "D_u": 0.0, "D_v": 0.0,
            "N": 4.0, "L": 1.0,
        },
    )
    sim = SchnakenbergSimulation(config)
    sim.reset()

    # Start from a perturbed state for richer dynamics
    u_star, v_star = sim.homogeneous_steady_state()
    sim._u[:] = u_star + 0.3
    sim._v[:] = v_star - 0.2
    sim._state = np.concatenate([sim._u.ravel(), sim._v.ravel()])

    states = []
    for _ in range(n_steps):
        sim.step()
        u_mean = np.mean(sim.u_field)
        v_mean = np.mean(sim.v_field)
        states.append([u_mean, v_mean])

    states = np.array(states)
    return {
        "time": np.arange(1, n_steps + 1) * dt,
        "states": states,
        "u": states[:, 0],
        "v": states[:, 1],
        "a": a,
        "b": b,
    }


def run_schnakenberg_rediscovery(
    output_dir: str | Path = "output/rediscovery/schnakenberg",
    n_iterations: int = 40,
) -> dict:
    """Run Schnakenberg rediscovery pipeline.

    1. Verify homogeneous steady state
    2. Turing instability analysis
    3. Wavelength vs D_v sweep with PySR
    4. SINDy ODE recovery (well-mixed limit)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "schnakenberg",
        "targets": {
            "steady_state": "u* = a + b, v* = b / (a + b)^2",
            "turing": "Pattern formation for D_v/D_u >> 1",
            "ode": "du/dt = a - u + u^2*v, dv/dt = b - u^2*v",
        },
    }

    # --- Part 1: Turing analysis ---
    logger.info("Part 1: Turing instability analysis...")
    config = SimulationConfig(
        domain=_DOMAIN,
        dt=0.01,
        n_steps=100,
        parameters={"a": 0.1, "b": 0.9, "D_u": 1.0, "D_v": 40.0, "N": 64.0},
    )
    sim = SchnakenbergSimulation(config)
    turing = sim.turing_analysis()
    results["turing_analysis"] = {
        k: float(v) if isinstance(v, (int, float, np.floating)) else v
        for k, v in turing.items()
    }
    logger.info(f"  Turing unstable: {turing['turing_unstable']}")
    if "wavelength_most_unstable" in turing:
        logger.info(f"  Most unstable wavelength: {turing['wavelength_most_unstable']:.2f}")

    # --- Part 2: Steady state verification ---
    logger.info("Part 2: Steady state verification (small sweep)...")
    ss_data = generate_steady_state_data(n_samples=5)
    u_err = np.abs(ss_data["u_star_sim"] - ss_data["u_star_theory"])
    v_err = np.abs(ss_data["v_star_sim"] - ss_data["v_star_theory"])
    results["steady_state"] = {
        "n_samples": len(ss_data["a"]),
        "u_mean_error": float(np.mean(u_err)),
        "v_mean_error": float(np.mean(v_err)),
        "u_max_error": float(np.max(u_err)),
        "v_max_error": float(np.max(v_err)),
    }
    logger.info(f"  u mean error: {np.mean(u_err):.6f}")
    logger.info(f"  v mean error: {np.mean(v_err):.6f}")

    # --- Part 3: SINDy ODE recovery ---
    logger.info("Part 3: SINDy ODE recovery (well-mixed limit)...")
    ode_data = generate_ode_data(a=0.1, b=0.9, n_steps=10000, dt=0.001)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=0.001,
            feature_names=["u", "v"],
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

    # --- Part 4: Wavelength vs D_v with PySR ---
    logger.info("Part 4: Wavelength vs D_v sweep...")
    wl_data = generate_wavelength_data(n_Dv=12, n_steps=3000, N=64, dt=0.01)

    # Filter for points where patterns formed
    valid = (wl_data["wavelength"] > 0) & (wl_data["energy"] > 1e-6)
    n_patterned = int(np.sum(valid))
    results["wavelength_sweep"] = {
        "n_total": len(wl_data["D_v"]),
        "n_patterned": n_patterned,
    }

    if n_patterned > 3:
        Dv_valid = wl_data["D_v"][valid]
        wl_valid = wl_data["wavelength"][valid]

        # Test wavelength ~ sqrt(D_v) correlation
        sqrt_Dv = np.sqrt(Dv_valid)
        corr = float(np.corrcoef(wl_valid, sqrt_Dv)[0, 1])
        results["wavelength_sweep"]["correlation_sqrt_Dv"] = corr
        logger.info(f"  Wavelength ~ sqrt(D_v) correlation: {corr:.4f}")

        # PySR: find wavelength = f(D_v)
        try:
            from simulating_anything.analysis.symbolic_regression import (
                run_symbolic_regression,
            )

            X = Dv_valid.reshape(-1, 1)
            y = wl_valid

            logger.info("  Running PySR: wavelength = f(D_v)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["Dv"],
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
