"""Ueda oscillator rediscovery.

Targets:
- ODE recovery via SINDy: dy/dt = -delta*y - x^3 + B*cos(t)
- Chaos onset as B increases (period-doubling cascade)
- Lyapunov exponent vs forcing amplitude B
- Poincare section structure at various B values
- Strange attractor verification at delta=0.05, B=7.5
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.ueda import UedaSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Placeholder domain until UEDA is added to Domain enum
_DOMAIN = Domain.UEDA


def generate_ode_data(
    delta: float = 0.05,
    B: float = 0.0,
    n_steps: int = 10000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Uses B=0 (unforced) by default so SINDy can cleanly identify the
    cubic restoring force and damping coefficients.
    """
    config = SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "delta": delta,
            "B": B,
            "x_0": 2.5,
            "y_0": 0.0,
        },
    )
    sim = UedaSimulation(config)
    sim.reset()

    states = [sim._state.copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim._state.copy())

    states = np.array(states)
    times = np.arange(n_steps + 1) * dt

    return {
        "time": times,
        "states": states,
        "x": states[:, 0],
        "y": states[:, 1],
        "delta": delta,
        "B": B,
        "dt": dt,
    }


def generate_lyapunov_vs_B_data(
    n_B: int = 30,
    n_steps: int = 30000,
    dt: float = 0.005,
    delta: float = 0.05,
) -> dict[str, np.ndarray]:
    """Sweep forcing amplitude B to map the transition from periodic to chaotic.

    Focuses on B in [0.5, 12.0] to capture the period-doubling cascade
    and the onset of the Ueda strange attractor.
    """
    B_values = np.linspace(0.5, 12.0, n_B)
    lyapunov_exps = []

    for i, B_val in enumerate(B_values):
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "delta": delta,
                "B": B_val,
                "x_0": 2.5,
                "y_0": 0.0,
            },
        )
        sim = UedaSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        lam = sim.compute_lyapunov_exponent(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  B={B_val:.2f}: Lyapunov={lam:.4f}")

    return {
        "B": B_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def generate_poincare_data(
    B: float = 7.5,
    delta: float = 0.05,
    n_transient: int = 500,
    n_points: int = 1000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Generate Poincare section data by strobing at the forcing period.

    At the classic chaotic parameters (delta=0.05, B=7.5), the Poincare
    section reveals the fractal structure of the Ueda attractor.
    """
    config = SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=1000,
        parameters={
            "delta": delta,
            "B": B,
            "x_0": 2.5,
            "y_0": 0.0,
        },
    )
    sim = UedaSimulation(config)
    sim.reset()

    points = sim.compute_poincare_section(
        n_transient=n_transient,
        n_points=n_points,
    )

    return {
        "x": points[:, 0],
        "y": points[:, 1],
        "B": B,
        "delta": delta,
        "n_points": n_points,
    }


def generate_chaos_sweep(
    n_B: int = 30,
    dt: float = 0.005,
    delta: float = 0.05,
) -> dict[str, np.ndarray]:
    """Sweep B and measure Poincare section spread to detect chaos.

    For each B value, collect stroboscopic points and compute their
    standard deviation. Low spread indicates periodic motion; high
    spread indicates chaotic motion.
    """
    B_values = np.linspace(0.5, 12.0, n_B)
    T_force = 2.0 * np.pi
    n_transient = int(100 * T_force / dt)
    n_poincare = 200
    poincare_spread = []

    for i, B_val in enumerate(B_values):
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=1000,
            parameters={
                "delta": delta,
                "B": B_val,
                "x_0": 2.5,
                "y_0": 0.0,
            },
        )
        sim = UedaSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(n_transient):
            sim.step()

        # Collect Poincare section points
        steps_per_period = int(round(T_force / dt))
        x_poincare = []
        for _ in range(n_poincare):
            for _ in range(steps_per_period):
                sim.step()
            x_poincare.append(sim._state[0])

        spread = float(np.std(np.array(x_poincare)))
        poincare_spread.append(spread)

        if (i + 1) % 10 == 0:
            logger.info(f"  B={B_val:.2f}: Poincare spread={spread:.6f}")

    return {
        "B": B_values,
        "poincare_spread": np.array(poincare_spread),
        "delta": delta,
    }


def run_ueda_rediscovery(
    output_dir: str | Path = "output/rediscovery/ueda",
    n_iterations: int = 40,
) -> dict:
    """Run the full Ueda oscillator rediscovery.

    1. Generate trajectory for SINDy ODE recovery (unforced for clean coefficients)
    2. Sweep B to map chaos transition (Lyapunov exponent)
    3. Generate Poincare section at classic chaotic parameters
    4. Compute Poincare spread across B values
    5. Measure trajectory statistics on the strange attractor

    Args:
        output_dir: Directory to save results.
        n_iterations: Number of PySR iterations.

    Returns:
        Results dict with all discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "ueda",
        "targets": {
            "ode_x": "dx/dt = y",
            "ode_y": "dy/dt = -delta*y - x^3 + B*cos(t)",
            "chaos": "Strange attractor at delta=0.05, B=7.5",
            "period_doubling": "Period-doubling cascade as B increases",
        },
    }

    # --- Part 1: SINDy ODE recovery (unforced) ---
    logger.info("Part 1: SINDy ODE recovery (unforced, delta=0.05)...")
    ode_data = generate_ode_data(
        delta=0.05, B=0.0, n_steps=10000, dt=0.005
    )

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["x", "v"],
            threshold=0.05,
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

    # --- Part 2: Lyapunov vs B sweep ---
    logger.info("Part 2: Mapping chaos transition (B sweep)...")
    lyap_data = generate_lyapunov_vs_B_data(n_B=30, n_steps=30000, dt=0.005)

    # Find chaos onset: first B where Lyapunov > 0
    lam = lyap_data["lyapunov_exponent"]
    B_vals = lyap_data["B"]
    zero_crossings = []
    for j in range(len(lam) - 1):
        if lam[j] <= 0 and lam[j + 1] > 0:
            frac = -lam[j] / (lam[j + 1] - lam[j])
            B_cross = B_vals[j] + frac * (B_vals[j + 1] - B_vals[j])
            zero_crossings.append(float(B_cross))

    n_chaotic = int(np.sum(lam > 0.01))
    n_periodic = int(np.sum(lam < -0.01))

    results["chaos_transition"] = {
        "n_B_values": len(B_vals),
        "n_chaotic": n_chaotic,
        "n_periodic": n_periodic,
        "B_range": [float(B_vals[0]), float(B_vals[-1])],
        "zero_crossings": zero_crossings,
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
    }
    if zero_crossings:
        logger.info(f"  Lyapunov zero crossings at B = {zero_crossings}")

    # --- Part 3: Poincare section at chaotic parameters ---
    logger.info("Part 3: Poincare section at B=7.5...")
    poincare_data = generate_poincare_data(
        B=7.5, delta=0.05, n_transient=500, n_points=1000, dt=0.005
    )

    x_range = float(np.ptp(poincare_data["x"]))
    y_range = float(np.ptp(poincare_data["y"]))
    results["poincare_section"] = {
        "n_points": poincare_data["n_points"],
        "x_range": x_range,
        "y_range": y_range,
        "x_std": float(np.std(poincare_data["x"])),
        "y_std": float(np.std(poincare_data["y"])),
        "is_chaotic": x_range > 0.5,
    }
    logger.info(
        f"  Poincare x-range: {x_range:.3f}, y-range: {y_range:.3f}"
    )

    # --- Part 4: Chaos sweep (Poincare spread) ---
    logger.info("Part 4: Chaos sweep (Poincare spread vs B)...")
    chaos_data = generate_chaos_sweep(n_B=30, dt=0.005)

    chaos_threshold = 0.1
    above = chaos_data["poincare_spread"] > chaos_threshold
    if np.any(above):
        idx = np.argmax(above)
        B_c = float(chaos_data["B"][idx])
    else:
        B_c = float("inf")

    results["chaos_sweep"] = {
        "n_B": len(chaos_data["B"]),
        "B_range": [
            float(chaos_data["B"].min()),
            float(chaos_data["B"].max()),
        ],
        "max_spread": float(np.max(chaos_data["poincare_spread"])),
        "B_c_estimate": B_c,
    }
    logger.info(f"  Chaos onset estimate: B ~ {B_c:.2f}")

    # --- Part 5: Trajectory statistics on attractor ---
    logger.info("Part 5: Trajectory statistics on attractor...")
    config_stats = SimulationConfig(
        domain=_DOMAIN,
        dt=0.005,
        n_steps=20000,
        parameters={
            "delta": 0.05,
            "B": 7.5,
            "x_0": 2.5,
            "y_0": 0.0,
        },
    )
    sim_stats = UedaSimulation(config_stats)
    traj_stats = sim_stats.compute_trajectory_statistics(
        n_steps=15000, n_transient=5000
    )
    results["trajectory_statistics"] = traj_stats
    logger.info(
        f"  x range: [{traj_stats['x_min']:.2f}, {traj_stats['x_max']:.2f}]"
    )

    # --- Part 6: Lyapunov at classic parameters ---
    logger.info("Part 6: Lyapunov at classic chaotic parameters...")
    config_classic = SimulationConfig(
        domain=_DOMAIN,
        dt=0.005,
        n_steps=50000,
        parameters={
            "delta": 0.05,
            "B": 7.5,
            "x_0": 2.5,
            "y_0": 0.0,
        },
    )
    sim_classic = UedaSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.compute_lyapunov_exponent(
        n_steps=50000, dt=0.005
    )
    results["classic_parameters"] = {
        "delta": 0.05,
        "B": 7.5,
        "lyapunov_exponent": float(lam_classic),
        "is_chaotic": bool(lam_classic > 0.01),
    }
    logger.info(f"  Classic Lyapunov: {lam_classic:.4f}")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "lyapunov_vs_B.npz",
        B=lyap_data["B"],
        lyapunov_exponent=lyap_data["lyapunov_exponent"],
    )
    np.savez(
        output_path / "poincare_section.npz",
        x=poincare_data["x"],
        y=poincare_data["y"],
    )
    np.savez(
        output_path / "chaos_sweep.npz",
        B=chaos_data["B"],
        poincare_spread=chaos_data["poincare_spread"],
    )

    return results
