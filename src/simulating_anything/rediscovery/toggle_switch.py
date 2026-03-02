"""Genetic toggle switch rediscovery.

Targets:
- ODE: du/dt = alpha1/(1+v^beta) - u, dv/dt = alpha2/(1+u^gamma) - v  (via SINDy)
- Bistability: two stable fixed points + one saddle
- Nullcline intersection verification
- Alpha sweep: pitchfork-like bifurcation
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.toggle_switch import ToggleSwitchSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    alpha1: float = 4.0,
    alpha2: float = 4.0,
    beta: float = 4.0,
    gamma: float = 4.0,
    n_steps: int = 10000,
    dt: float = 0.01,
    u_0: float = 2.0,
    v_0: float = 2.0,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.TOGGLE_SWITCH,  # placeholder domain enum
        dt=dt,
        n_steps=n_steps,
        parameters={
            "alpha1": alpha1,
            "alpha2": alpha2,
            "beta": beta,
            "gamma": gamma,
            "u_0": u_0,
            "v_0": v_0,
        },
    )
    sim = ToggleSwitchSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "u": states_arr[:, 0],
        "v": states_arr[:, 1],
        "alpha1": alpha1,
        "alpha2": alpha2,
        "beta": beta,
        "gamma": gamma,
    }


def generate_bistability_data(
    alpha1: float = 4.0,
    alpha2: float = 4.0,
    beta: float = 4.0,
    gamma: float = 4.0,
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict:
    """Demonstrate bistability: two ICs converge to different steady states."""
    results = {}

    # IC1: high u, low v
    config1 = SimulationConfig(
        domain=Domain.TOGGLE_SWITCH,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "alpha1": alpha1, "alpha2": alpha2,
            "beta": beta, "gamma": gamma,
            "u_0": alpha1, "v_0": 0.1,
        },
    )
    sim1 = ToggleSwitchSimulation(config1)
    sim1.reset()
    for _ in range(n_steps):
        sim1.step()
    state1 = sim1.observe()
    results["ic_high_u"] = {"u_0": alpha1, "v_0": 0.1}
    results["final_high_u"] = {"u": float(state1[0]), "v": float(state1[1])}

    # IC2: low u, high v
    config2 = SimulationConfig(
        domain=Domain.TOGGLE_SWITCH,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "alpha1": alpha1, "alpha2": alpha2,
            "beta": beta, "gamma": gamma,
            "u_0": 0.1, "v_0": alpha2,
        },
    )
    sim2 = ToggleSwitchSimulation(config2)
    sim2.reset()
    for _ in range(n_steps):
        sim2.step()
    state2 = sim2.observe()
    results["ic_high_v"] = {"u_0": 0.1, "v_0": alpha2}
    results["final_high_v"] = {"u": float(state2[0]), "v": float(state2[1])}

    # Check bistability: final states should differ
    diff = np.linalg.norm(state1 - state2)
    results["state_difference"] = float(diff)
    results["is_bistable"] = bool(diff > 0.5)

    return results


def generate_alpha_sweep_data(
    alpha_range: tuple[float, float] = (0.5, 8.0),
    n_alpha: int = 30,
    beta: float = 4.0,
    gamma: float = 4.0,
    dt: float = 0.01,
    n_steps: int = 5000,
) -> dict[str, np.ndarray]:
    """Sweep alpha1=alpha2 to detect bifurcation point."""
    config = SimulationConfig(
        domain=Domain.TOGGLE_SWITCH,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "alpha1": 4.0, "alpha2": 4.0,
            "beta": beta, "gamma": gamma,
            "u_0": 3.0, "v_0": 0.5,
        },
    )
    sim = ToggleSwitchSimulation(config)
    return sim.alpha_sweep(
        alpha_range=alpha_range,
        n_alpha=n_alpha,
        n_steps=n_steps,
    )


def run_toggle_switch_rediscovery(
    output_dir: str | Path = "output/rediscovery/toggle_switch",
    n_iterations: int = 40,
) -> dict:
    """Run genetic toggle switch rediscovery pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "toggle_switch",
        "targets": {
            "ode": "du/dt = alpha1/(1+v^beta) - u, dv/dt = alpha2/(1+u^gamma) - v",
            "bistability": "Two stable fixed points + one saddle",
            "bifurcation": "Pitchfork-like as alpha varies",
        },
    }

    # --- Part 1: Fixed point finding and classification ---
    logger.info("Part 1: Fixed point analysis...")
    config = SimulationConfig(
        domain=Domain.TOGGLE_SWITCH,
        dt=0.01,
        n_steps=1000,
        parameters={
            "alpha1": 4.0, "alpha2": 4.0,
            "beta": 4.0, "gamma": 4.0,
            "u_0": 3.0, "v_0": 0.5,
        },
    )
    sim = ToggleSwitchSimulation(config)
    sim.reset()

    fps = sim.find_fixed_points()
    fp_info = []
    for u_fp, v_fp in fps:
        stability = sim.classify_fixed_point(u_fp, v_fp)
        deriv = sim._derivatives(np.array([u_fp, v_fp]))
        fp_info.append({
            "u": float(u_fp),
            "v": float(v_fp),
            "stability": stability,
            "derivative_norm": float(np.linalg.norm(deriv)),
        })
        logger.info(f"  FP: ({u_fp:.4f}, {v_fp:.4f}) -> {stability}")

    n_stable = sum(1 for fp in fp_info if fp["stability"] == "stable")
    n_saddle = sum(1 for fp in fp_info if fp["stability"] == "saddle")

    results["fixed_points"] = {
        "n_found": len(fps),
        "n_stable": n_stable,
        "n_saddle": n_saddle,
        "points": fp_info,
    }

    # --- Part 2: Nullcline computation ---
    logger.info("Part 2: Nullcline computation...")
    nullclines = sim.compute_nullclines(u_range=(0.0, 6.0), n_points=200)
    results["nullclines"] = {
        "u_range": [0.0, 6.0],
        "n_points": 200,
        "u_null_u_range": [
            float(np.min(nullclines["u_null_u"])),
            float(np.max(nullclines["u_null_u"])),
        ],
        "v_null_v_range": [
            float(np.min(nullclines["v_null_v"])),
            float(np.max(nullclines["v_null_v"])),
        ],
    }
    logger.info(
        f"  u-nullcline: u in [{nullclines['u_null_u'].min():.3f}, "
        f"{nullclines['u_null_u'].max():.3f}]"
    )

    # --- Part 3: Bistability demonstration ---
    logger.info("Part 3: Bistability demonstration...")
    bistab = generate_bistability_data(
        alpha1=4.0, alpha2=4.0, beta=4.0, gamma=4.0,
    )
    results["bistability"] = bistab
    logger.info(
        f"  IC1 final: ({bistab['final_high_u']['u']:.4f}, "
        f"{bistab['final_high_u']['v']:.4f})"
    )
    logger.info(
        f"  IC2 final: ({bistab['final_high_v']['u']:.4f}, "
        f"{bistab['final_high_v']['v']:.4f})"
    )
    logger.info(
        f"  Bistable: {bistab['is_bistable']} "
        f"(diff={bistab['state_difference']:.4f})"
    )

    # --- Part 4: Alpha sweep for bifurcation ---
    logger.info("Part 4: Alpha sweep for bifurcation detection...")
    sweep_data = generate_alpha_sweep_data(
        alpha_range=(0.5, 8.0), n_alpha=30, n_steps=5000,
    )

    # Detect bifurcation: find where high-u and low-u trajectories diverge
    diff_u = np.abs(sweep_data["final_u_high"] - sweep_data["final_u_low"])
    threshold = 0.5
    bistable_mask = diff_u > threshold
    if np.any(bistable_mask):
        idx = np.argmax(bistable_mask)
        alpha_c_est = float(sweep_data["alpha"][max(0, idx - 1)])
    else:
        alpha_c_est = float("nan")

    results["alpha_sweep"] = {
        "alpha_range": [0.5, 8.0],
        "n_alpha": 30,
        "alpha_c_estimate": alpha_c_est,
        "n_bistable": int(np.sum(bistable_mask)),
        "max_diff": float(np.max(diff_u)),
    }
    logger.info(f"  Bifurcation alpha_c estimate: {alpha_c_est:.3f}")
    logger.info(f"  Bistable points: {np.sum(bistable_mask)}/{len(diff_u)}")

    # --- Part 5: SINDy ODE recovery ---
    logger.info("Part 5: SINDy ODE recovery...")
    data = generate_ode_data(
        alpha1=4.0, alpha2=4.0, beta=4.0, gamma=4.0,
        n_steps=10000, dt=0.005, u_0=2.0, v_0=2.0,
    )

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.005,
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

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
