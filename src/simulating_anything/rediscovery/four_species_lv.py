"""Four-species Lotka-Volterra food web rediscovery.

Targets:
- SINDy ODE recovery (4 coupled equations):
    dx1/dt = x1*(r1 - a11*x1 - a12*x2 - b1*y1)
    dx2/dt = x2*(r2 - a21*x1 - a22*x2 - b2*y2)
    dy1/dt = y1*(-d1 + c1*x1)
    dy2/dt = y2*(-d2 + c2*x2)
- Coexistence equilibrium: x1*=d1/c1, x2*=d2/c2
- Jacobian eigenvalue stability analysis
- Competition strength sweep (a12, a21 vary)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.four_species_lv import FourSpeciesLVSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use AGENT_BASED as stand-in until FOUR_SPECIES_LV is added to the Domain enum
_DOMAIN = Domain.AGENT_BASED


def _make_config(
    r1: float = 1.0,
    r2: float = 0.8,
    a11: float = 0.1,
    a12: float = 0.05,
    a21: float = 0.05,
    a22: float = 0.1,
    b1: float = 0.5,
    b2: float = 0.5,
    c1: float = 0.3,
    c2: float = 0.3,
    d1: float = 0.4,
    d2: float = 0.4,
    x1_0: float = 0.5,
    x2_0: float = 0.5,
    y1_0: float = 0.3,
    y2_0: float = 0.3,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    """Build a SimulationConfig for the four-species LV model."""
    return SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r1": r1, "r2": r2,
            "a11": a11, "a12": a12, "a21": a21, "a22": a22,
            "b1": b1, "b2": b2,
            "c1": c1, "c2": c2,
            "d1": d1, "d2": d2,
            "x1_0": x1_0, "x2_0": x2_0, "y1_0": y1_0, "y2_0": y2_0,
        },
    )


def generate_trajectory_data(
    n_steps: int = 10000,
    dt: float = 0.005,
    **kwargs: float,
) -> dict[str, np.ndarray | float]:
    """Generate a single long trajectory for SINDy ODE recovery.

    Returns dict with states array, time, and parameters.
    """
    config = _make_config(n_steps=n_steps, dt=dt, **kwargs)
    sim = FourSpeciesLVSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "x1": states_arr[:, 0],
        "x2": states_arr[:, 1],
        "y1": states_arr[:, 2],
        "y2": states_arr[:, 3],
        "dt": dt,
        "r1": sim.r1, "r2": sim.r2,
        "a11": sim.a11, "a12": sim.a12, "a21": sim.a21, "a22": sim.a22,
        "b1": sim.b1, "b2": sim.b2,
        "c1": sim.c1, "c2": sim.c2,
        "d1": sim.d1, "d2": sim.d2,
    }


def generate_competition_sweep_data(
    n_alpha: int = 25,
    n_steps: int = 20000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep cross-competition a12 and measure final populations.

    Varies a12 from 0 to 0.5 while keeping a21 fixed. Records
    whether all four species coexist and the time-averaged populations.

    Returns dict with alpha values, final populations, coexistence flags.
    """
    a12_values = np.linspace(0.0, 0.5, n_alpha)
    final_pops_list = []
    n_surviving_list = []
    coexisting_list = []

    for i, a12 in enumerate(a12_values):
        config = _make_config(a12=a12, dt=dt, n_steps=n_steps)
        sim = FourSpeciesLVSimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        final_pops_list.append(sim.observe().copy())
        n_surviving_list.append(sim.n_surviving())
        coexisting_list.append(sim.is_coexisting)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  a12={a12:.3f}: surviving={n_surviving_list[-1]}, "
                f"coexist={coexisting_list[-1]}"
            )

    return {
        "a12": a12_values,
        "final_populations": np.array(final_pops_list),
        "n_surviving": np.array(n_surviving_list),
        "coexisting": np.array(coexisting_list),
    }


def generate_stability_data(
    n_samples: int = 50,
) -> dict[str, np.ndarray]:
    """Sample random parameters and compute eigenvalue stability.

    Returns dict with max real eigenvalue, stability and feasibility flags,
    and a competition strength measure.
    """
    rng = np.random.default_rng(42)
    all_max_real_eig = []
    all_is_stable = []
    all_is_feasible = []
    all_competition_strength = []

    for _ in range(n_samples):
        r1 = rng.uniform(0.5, 1.5)
        r2 = rng.uniform(0.5, 1.5)
        a11 = rng.uniform(0.05, 0.3)
        a12 = rng.uniform(0.0, 0.3)
        a21 = rng.uniform(0.0, 0.3)
        a22 = rng.uniform(0.05, 0.3)
        b1 = rng.uniform(0.3, 0.8)
        b2 = rng.uniform(0.3, 0.8)
        c1 = rng.uniform(0.1, 0.5)
        c2 = rng.uniform(0.1, 0.5)
        d1 = rng.uniform(0.2, 0.6)
        d2 = rng.uniform(0.2, 0.6)

        config = _make_config(
            r1=r1, r2=r2, a11=a11, a12=a12, a21=a21, a22=a22,
            b1=b1, b2=b2, c1=c1, c2=c2, d1=d1, d2=d2,
            dt=0.01, n_steps=100,
        )
        sim = FourSpeciesLVSimulation(config)
        sim.reset()

        eq = sim.coexistence_equilibrium()
        feasible = bool(np.all(eq > 0) and np.all(np.isfinite(eq)))

        eigs = sim.stability_eigenvalues()
        max_real = float(np.max(np.real(eigs)))
        stable = feasible and (max_real < 0)

        all_max_real_eig.append(max_real)
        all_is_stable.append(stable)
        all_is_feasible.append(feasible)
        all_competition_strength.append(float(a12 + a21))

    return {
        "max_real_eigenvalue": np.array(all_max_real_eig),
        "is_stable": np.array(all_is_stable),
        "is_feasible": np.array(all_is_feasible),
        "competition_strength": np.array(all_competition_strength),
    }


def run_four_species_lv_rediscovery(
    output_dir: str | Path = "output/rediscovery/four_species_lv",
    n_iterations: int = 40,
    n_stability_samples: int = 50,
    **kwargs: float,
) -> dict:
    """Run the full four-species LV food web rediscovery pipeline.

    1. Generate trajectory for SINDy ODE recovery
    2. Run SINDy to recover 4 coupled ODEs
    3. Verify coexistence equilibrium
    4. Competition strength sweep
    5. Jacobian eigenvalue stability analysis

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "four_species_lv",
        "targets": {
            "ode_x1": "dx1/dt = x1*(r1 - a11*x1 - a12*x2 - b1*y1)",
            "ode_x2": "dx2/dt = x2*(r2 - a21*x1 - a22*x2 - b2*y2)",
            "ode_y1": "dy1/dt = y1*(-d1 + c1*x1)",
            "ode_y2": "dy2/dt = y2*(-d2 + c2*x2)",
            "equilibrium_x1": "x1* = d1/c1",
            "equilibrium_x2": "x2* = d2/c2",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery for four-species LV food web...")
    data = generate_trajectory_data(n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.005,
            feature_names=["x1", "x2", "y1", "y2"],
            threshold=0.05,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries
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
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Coexistence equilibrium verification ---
    logger.info("Part 2: Coexistence equilibrium verification...")
    config = _make_config(dt=0.005, n_steps=40000)
    sim = FourSpeciesLVSimulation(config)
    sim.reset()

    states_list = [sim.observe().copy()]
    for _ in range(40000):
        sim.step()
        states_list.append(sim.observe().copy())

    trajectory = np.array(states_list)
    # Skip initial transient (first 50%)
    skip = len(trajectory) // 2
    time_avg = np.mean(trajectory[skip:], axis=0)

    eq = sim.coexistence_equilibrium()
    eq_error = np.abs(time_avg - eq) / np.maximum(np.abs(eq), 1e-10)

    results["equilibrium"] = {
        "analytical": eq.tolist(),
        "time_averaged": time_avg.tolist(),
        "relative_error": eq_error.tolist(),
        "mean_relative_error": float(np.mean(eq_error)),
        "x1_star_theory": float(sim.d1 / sim.c1),
        "x2_star_theory": float(sim.d2 / sim.c2),
    }
    logger.info(f"  Analytical equilibrium: {eq}")
    logger.info(f"  Time-averaged: {time_avg}")
    logger.info(f"  Mean relative error: {np.mean(eq_error):.4%}")

    # --- Part 3: Competition strength sweep ---
    logger.info("Part 3: Competition strength sweep (a12: 0 -> 0.5)...")
    sweep_data = generate_competition_sweep_data(n_alpha=25, n_steps=20000, dt=0.01)

    n_coexist = int(np.sum(sweep_data["coexisting"]))
    results["competition_sweep"] = {
        "n_alpha_values": len(sweep_data["a12"]),
        "a12_range": [float(sweep_data["a12"][0]), float(sweep_data["a12"][-1])],
        "n_coexisting": n_coexist,
        "fraction_coexisting": float(n_coexist / len(sweep_data["a12"])),
        "min_surviving": int(np.min(sweep_data["n_surviving"])),
        "max_surviving": int(np.max(sweep_data["n_surviving"])),
    }
    logger.info(
        f"  Coexistence fraction: {n_coexist}/{len(sweep_data['a12'])}"
    )

    # --- Part 4: Stability eigenvalue analysis ---
    logger.info("Part 4: Jacobian eigenvalue stability analysis...")
    stab_data = generate_stability_data(n_samples=n_stability_samples)

    n_feasible = int(np.sum(stab_data["is_feasible"]))
    n_stable = int(np.sum(stab_data["is_stable"]))

    results["stability_analysis"] = {
        "n_samples": n_stability_samples,
        "n_feasible": n_feasible,
        "n_stable": n_stable,
        "fraction_feasible": float(n_feasible / n_stability_samples),
        "fraction_stable": float(n_stable / n_stability_samples),
        "mean_max_real_eigenvalue": float(np.mean(stab_data["max_real_eigenvalue"])),
    }
    logger.info(f"  Feasible equilibria: {n_feasible}/{n_stability_samples}")
    logger.info(f"  Stable coexistence: {n_stable}/{n_stability_samples}")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "trajectory_data.npz",
        states=data["states"],
    )
    np.savez(
        output_path / "competition_sweep.npz",
        **{k: v for k, v in sweep_data.items()},
    )

    return results
