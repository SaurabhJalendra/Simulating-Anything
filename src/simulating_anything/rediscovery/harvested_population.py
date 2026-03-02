"""Logistic growth with harvesting rediscovery.

Targets:
- Maximum sustainable yield: H_c = r*K/4 (saddle-node bifurcation)
- Equilibrium structure: two equilibria for H < H_c, none for H > H_c
- Extinction dynamics: population collapse when H > H_c
- PySR: recover H_c = f(r, K)
- SINDy: recover ODE dx/dt = r*x*(1-x/K) - H
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.harvested_population import (
    HarvestedPopulationSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    r: float = 1.0,
    K: float = 1.0,
    H: float = 0.0,
    x_0: float | None = None,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    """Create a SimulationConfig for the harvested population model."""
    params: dict[str, float] = {"r": r, "K": K, "H": H}
    if x_0 is not None:
        params["x_0"] = x_0
    return SimulationConfig(
        domain=Domain.HARVESTED_POPULATION,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


def generate_ode_data(
    r: float = 1.0,
    K: float = 1.0,
    H: float = 0.1,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery."""
    config = _make_config(r=r, K=K, H=H, x_0=K * 0.8, dt=dt, n_steps=n_steps)
    sim = HarvestedPopulationSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "x": states_arr[:, 0],
        "r": r,
        "K": K,
        "H": H,
    }


def generate_bifurcation_data(
    r: float = 1.0,
    K: float = 1.0,
    n_H: int = 30,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep H and measure final population to map the bifurcation diagram.

    For each H value, simulate from x_0 = K*0.9 (near stable equilibrium)
    and record the final population after transient.
    """
    H_max = r * K / 4.0 * 1.5  # Go past the MSY
    H_values = np.linspace(0.0, H_max, n_H)
    final_pop = []
    extinct = []

    for i, H in enumerate(H_values):
        config = _make_config(r=r, K=K, H=H, x_0=K * 0.9, dt=dt, n_steps=10000)
        sim = HarvestedPopulationSimulation(config)
        sim.reset()

        for _ in range(10000):
            sim.step()

        x_final = float(sim.observe()[0])
        final_pop.append(x_final)
        extinct.append(x_final <= 0.0)

        if (i + 1) % 10 == 0:
            logger.info(f"  H={H:.4f}: x_final={x_final:.4f}")

    return {
        "r": r,
        "K": K,
        "H": H_values,
        "final_pop": np.array(final_pop),
        "extinct": np.array(extinct),
        "H_c_theory": r * K / 4.0,
    }


def generate_msy_data(
    n_r: int = 15,
    n_K: int = 15,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep r and K, estimate H_c by binary search, for PySR recovery."""
    r_values = np.linspace(0.5, 3.0, n_r)
    K_values = np.linspace(0.5, 3.0, n_K)

    r_arr = []
    K_arr = []
    H_c_measured = []
    H_c_theory = []

    for r in r_values:
        for K in K_values:
            # Binary search for the critical H
            H_low = 0.0
            H_high = r * K  # Upper bound (well above MSY)
            for _ in range(30):  # 30 iterations of bisection
                H_mid = (H_low + H_high) / 2.0
                config = _make_config(
                    r=r, K=K, H=H_mid, x_0=K * 0.9, dt=dt, n_steps=5000,
                )
                sim = HarvestedPopulationSimulation(config)
                sim.reset()
                for _ in range(5000):
                    sim.step()
                if sim.observe()[0] > 1e-6:
                    H_low = H_mid
                else:
                    H_high = H_mid

            H_c_est = (H_low + H_high) / 2.0
            r_arr.append(r)
            K_arr.append(K)
            H_c_measured.append(H_c_est)
            H_c_theory.append(r * K / 4.0)

    return {
        "r": np.array(r_arr),
        "K": np.array(K_arr),
        "H_c_measured": np.array(H_c_measured),
        "H_c_theory": np.array(H_c_theory),
    }


def run_harvested_population_rediscovery(
    output_dir: str | Path = "output/rediscovery/harvested_population",
    n_iterations: int = 40,
) -> dict:
    """Run harvested population rediscovery pipeline.

    Discovers:
    - Maximum sustainable yield H_c = r*K/4
    - Equilibrium structure and saddle-node bifurcation
    - ODE recovery via SINDy
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "harvested_population",
        "targets": {
            "msy": "H_c = r*K/4",
            "ode": "dx/dt = r*x*(1-x/K) - H",
            "equilibria": "x* = K/2 +/- sqrt(K^2/4 - H*K/r)",
        },
    }

    # --- Part 1: Equilibrium verification ---
    logger.info("Part 1: Equilibrium verification...")
    r, K = 1.0, 2.0
    for H in [0.0, 0.1, 0.4]:
        config = _make_config(r=r, K=K, H=H, x_0=K * 0.9)
        sim = HarvestedPopulationSimulation(config)
        sim.reset()
        eqs = sim.find_equilibria()
        msy = sim.compute_msy()
        logger.info(f"  r={r}, K={K}, H={H}: MSY={msy:.4f}, eq={eqs}")

    # Test MSY formula for a specific case
    config_msy = _make_config(r=2.0, K=4.0)
    sim_msy = HarvestedPopulationSimulation(config_msy)
    msy_val = sim_msy.compute_msy()
    results["msy_verification"] = {
        "r": 2.0,
        "K": 4.0,
        "computed_msy": msy_val,
        "theory_msy": 2.0,  # r*K/4 = 2*4/4 = 2
        "match": abs(msy_val - 2.0) < 1e-10,
    }
    logger.info(f"  MSY test: computed={msy_val}, theory=2.0")

    # --- Part 2: Bifurcation diagram ---
    logger.info("Part 2: Bifurcation diagram (H sweep)...")
    bif_data = generate_bifurcation_data(r=1.0, K=2.0, n_H=30, dt=0.01)

    # Estimate H_c from the data: first H where population goes extinct
    extinct_mask = bif_data["extinct"]
    if np.any(extinct_mask):
        idx = np.argmax(extinct_mask)
        H_c_est = float(bif_data["H"][max(0, idx - 1)])
    else:
        H_c_est = float(bif_data["H"][-1])

    H_c_theory = bif_data["H_c_theory"]
    results["bifurcation"] = {
        "H_c_estimate": H_c_est,
        "H_c_theory": H_c_theory,
        "relative_error": float(abs(H_c_est - H_c_theory) / H_c_theory)
        if H_c_theory > 0 else 0.0,
        "n_extinct": int(np.sum(extinct_mask)),
        "n_surviving": int(np.sum(~extinct_mask)),
    }
    logger.info(
        f"  H_c estimate: {H_c_est:.4f} (theory: {H_c_theory:.4f})"
    )

    # --- Part 3: MSY data for PySR ---
    logger.info("Part 3: MSY sweep for PySR H_c = f(r, K)...")
    msy_data = generate_msy_data(n_r=10, n_K=10, dt=0.01)

    correlation = float(np.corrcoef(
        msy_data["H_c_measured"], msy_data["H_c_theory"],
    )[0, 1])
    mean_error = float(np.mean(np.abs(
        msy_data["H_c_measured"] - msy_data["H_c_theory"],
    )))
    results["msy_sweep"] = {
        "n_points": len(msy_data["r"]),
        "correlation": correlation,
        "mean_absolute_error": mean_error,
    }
    logger.info(
        f"  MSY sweep: {len(msy_data['r'])} points, "
        f"correlation={correlation:.4f}, MAE={mean_error:.4f}"
    )

    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = np.column_stack([msy_data["r"], msy_data["K"]])
        y = msy_data["H_c_measured"]

        logger.info("  Running PySR: H_c = f(r, K)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["r_", "K_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt"],
            max_complexity=8,
            populations=20,
            population_size=40,
        )
        results["msy_pysr"] = {
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
            results["msy_pysr"]["best"] = best.expression
            results["msy_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["msy_pysr"] = {"error": str(e)}

    # --- Part 4: SINDy ODE recovery ---
    logger.info("Part 4: SINDy ODE recovery...")
    ode_data = generate_ode_data(r=1.0, K=2.0, H=0.2, n_steps=5000, dt=0.01)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=0.01,
            feature_names=["x"],
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

    # --- Part 5: Extinction dynamics ---
    logger.info("Part 5: Extinction dynamics...")
    r_ext, K_ext = 1.0, 2.0
    H_c_ext = r_ext * K_ext / 4.0
    extinction_times = []
    H_above = np.linspace(H_c_ext * 1.05, H_c_ext * 2.0, 10)
    for H in H_above:
        config = _make_config(r=r_ext, K=K_ext, H=H, x_0=K_ext * 0.9, dt=0.01)
        sim = HarvestedPopulationSimulation(config)
        t_ext = sim.time_to_extinction(max_steps=50000)
        extinction_times.append(t_ext)

    results["extinction"] = {
        "H_values": [float(h) for h in H_above],
        "extinction_times": [float(t) for t in extinction_times],
        "all_extinct": all(t < float("inf") for t in extinction_times),
    }
    logger.info(
        f"  Extinction times for H > H_c: "
        f"min={min(extinction_times):.2f}, max={max(extinction_times):.2f}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
