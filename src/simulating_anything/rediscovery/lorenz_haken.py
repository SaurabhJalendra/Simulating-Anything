"""Lorenz-Haken (Maxwell-Bloch) laser system rediscovery.

Targets:
- SINDy recovery of Maxwell-Bloch ODEs:
    x' = sigma*(y-x), y' = (r-z)*x - y, z' = x*y - b*z
- Lasing threshold detection at r = 1
- Second (Hopf) instability threshold: r_H = sigma*(sigma+b+3)/(sigma-b-1)
- Pump-parameter sweep mapping chaos transition
- Lyapunov exponent as a function of pump parameter r
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.lorenz_haken import LorenzHakenSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use LORENZ_ATTRACTOR as domain placeholder (structurally identical equations)
_LORENZ_HAKEN_DOMAIN = Domain.LORENZ_ATTRACTOR


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    sigma: float = 3.0,
    r: float = 25.0,
    b: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate a single Lorenz-Haken trajectory for SINDy ODE recovery.

    Uses classic laser chaotic parameters by default.
    """
    config = SimulationConfig(
        domain=_LORENZ_HAKEN_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "sigma": sigma,
            "r": r,
            "b": b,
            "x_0": 1.0,
            "y_0": 1.0,
            "z_0": 1.0,
        },
    )
    sim = LorenzHakenSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "sigma": sigma,
        "r": r,
        "b": b,
    }


def generate_pump_sweep_data(
    n_r: int = 50,
    n_steps: int = 20000,
    dt: float = 0.01,
    sigma: float = 3.0,
    b: float = 1.0,
) -> dict[str, np.ndarray]:
    """Sweep pump parameter r to map lasing onset and chaos transition.

    For each r value, compute the Lyapunov exponent, steady-state
    intensity, and classify the attractor type.
    """
    r_values = np.linspace(0.5, 35.0, n_r)
    lyapunov_exps = []
    attractor_types = []
    mean_intensities = []

    for i, r_val in enumerate(r_values):
        config = SimulationConfig(
            domain=_LORENZ_HAKEN_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={"sigma": sigma, "r": r_val, "b": b},
        )
        sim = LorenzHakenSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        # Estimate Lyapunov exponent
        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        # Measure mean intensity (x^2)
        intensities = []
        for _ in range(5000):
            state = sim.step()
            intensities.append(state[0] ** 2)
        mean_int = float(np.mean(intensities))
        mean_intensities.append(mean_int)

        # Classify attractor
        if r_val < 1.0:
            atype = "non_lasing"
        elif mean_int < 0.01:
            atype = "non_lasing"
        elif lam > 0.1:
            atype = "chaotic"
        elif lam < -0.1:
            atype = "steady_lasing"
        else:
            atype = "periodic_or_transient"
        attractor_types.append(atype)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  r={r_val:.1f}: Lyapunov={lam:.3f}, "
                f"intensity={mean_int:.2f}, type={atype}"
            )

    return {
        "r": r_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
        "mean_intensity": np.array(mean_intensities),
        "attractor_type": np.array(attractor_types),
    }


def generate_lyapunov_fine_sweep(
    n_r: int = 30,
    n_steps: int = 30000,
    dt: float = 0.005,
    sigma: float = 3.0,
    b: float = 1.0,
) -> dict[str, np.ndarray]:
    """Fine sweep of Lyapunov exponent near the second threshold.

    For sigma=3, b=1: r_H = 3*(3+1+3)/(3-1-1) = 21.
    Focuses on r in [15, 30] to resolve the chaos transition.
    """
    r_values = np.linspace(15.0, 30.0, n_r)
    lyapunov_exps = []

    for i, r_val in enumerate(r_values):
        config = SimulationConfig(
            domain=_LORENZ_HAKEN_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={"sigma": sigma, "r": r_val, "b": b},
        )
        sim = LorenzHakenSimulation(config)
        sim.reset()

        # Transient
        for _ in range(10000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  r={r_val:.2f}: Lyapunov={lam:.4f}")

    return {
        "r": r_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def run_lorenz_haken_rediscovery(
    output_dir: str | Path = "output/rediscovery/lorenz_haken",
    n_iterations: int = 40,
) -> dict:
    """Run the full Lorenz-Haken (Maxwell-Bloch) laser rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep pump parameter r to map lasing onset and chaos transition
    3. Fine Lyapunov sweep near second threshold
    4. Verify lasing threshold and fixed points

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "lorenz_haken",
        "targets": {
            "ode_x": "dx/dt = sigma*(y - x)",
            "ode_y": "dy/dt = (r - z)*x - y",
            "ode_z": "dz/dt = x*y - b*z",
            "lasing_threshold": "r = 1",
            "second_threshold": "r_H = sigma*(sigma+b+3)/(sigma-b-1) = 21 for sigma=3, b=1",
            "physics": "Maxwell-Bloch single-mode laser equations",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Lorenz-Haken trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=5000, dt=0.01)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["x", "y", "z"],
            threshold=0.1,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries
            ],
            "true_sigma": ode_data["sigma"],
            "true_r": ode_data["r"],
            "true_b": ode_data["b"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Pump parameter sweep ---
    logger.info("Part 2: Mapping lasing onset and chaos transition (r sweep)...")
    sweep_data = generate_pump_sweep_data(n_r=50, n_steps=20000, dt=0.01)

    n_chaotic = int(np.sum(sweep_data["attractor_type"] == "chaotic"))
    n_lasing = int(np.sum(sweep_data["attractor_type"] == "steady_lasing"))
    n_non_lasing = int(np.sum(sweep_data["attractor_type"] == "non_lasing"))
    logger.info(
        f"  Found {n_non_lasing} non-lasing, {n_lasing} steady-lasing, "
        f"{n_chaotic} chaotic regimes"
    )

    results["pump_sweep"] = {
        "n_r_values": len(sweep_data["r"]),
        "n_non_lasing": n_non_lasing,
        "n_steady_lasing": n_lasing,
        "n_chaotic": n_chaotic,
        "r_range": [float(sweep_data["r"][0]), float(sweep_data["r"][-1])],
    }

    # Detect lasing threshold (first r with non-trivial intensity)
    intensity_threshold = 0.01
    mask_lasing = sweep_data["mean_intensity"] > intensity_threshold
    if np.any(mask_lasing):
        r_lasing_approx = float(sweep_data["r"][np.argmax(mask_lasing)])
        results["pump_sweep"]["r_lasing_approx"] = r_lasing_approx
        logger.info(
            f"  Approximate lasing onset: r={r_lasing_approx:.2f} (theory: r=1)"
        )

    # Detect chaos onset (first positive Lyapunov)
    mask_positive = sweep_data["lyapunov_exponent"] > 0.1
    if np.any(mask_positive):
        r_chaos_approx = float(sweep_data["r"][np.argmax(mask_positive)])
        results["pump_sweep"]["r_chaos_approx"] = r_chaos_approx
        logger.info(f"  Approximate chaos onset: r={r_chaos_approx:.1f}")

    # --- Part 3: Fine Lyapunov sweep ---
    logger.info("Part 3: Fine Lyapunov sweep near second threshold...")
    fine_data = generate_lyapunov_fine_sweep(n_r=30, n_steps=30000, dt=0.005)

    lam = fine_data["lyapunov_exponent"]
    r_fine = fine_data["r"]
    zero_crossings = []
    for j in range(len(lam) - 1):
        if lam[j] <= 0 < lam[j + 1]:
            frac = -lam[j] / (lam[j + 1] - lam[j])
            r_cross = r_fine[j] + frac * (r_fine[j + 1] - r_fine[j])
            zero_crossings.append(float(r_cross))

    results["lyapunov_analysis"] = {
        "n_points": len(r_fine),
        "r_range": [float(r_fine[0]), float(r_fine[-1])],
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
        "zero_crossings": zero_crossings,
    }
    if zero_crossings:
        logger.info(f"  Lyapunov zero crossings at r = {zero_crossings}")

    # --- Part 4: Verify thresholds and fixed points ---
    logger.info("Part 4: Verifying thresholds and fixed points...")
    config_classic = SimulationConfig(
        domain=_LORENZ_HAKEN_DOMAIN,
        dt=0.005,
        n_steps=50000,
        parameters={"sigma": 3.0, "r": 25.0, "b": 1.0},
    )
    sim_classic = LorenzHakenSimulation(config_classic)
    sim_classic.reset()

    # Lasing threshold
    results["lasing_threshold"] = {
        "value": sim_classic.lasing_threshold,
        "theory": 1.0,
    }

    # Second threshold
    r_H = sim_classic.second_threshold
    r_H_theory = 3.0 * (3.0 + 1.0 + 3.0) / (3.0 - 1.0 - 1.0)
    results["second_threshold"] = {
        "value": float(r_H),
        "theory": float(r_H_theory),
        "relative_error": float(abs(r_H - r_H_theory) / r_H_theory),
    }
    logger.info(f"  Second threshold r_H = {r_H:.2f} (theory: {r_H_theory:.2f})")

    # Lyapunov at classic chaotic parameters
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.005)

    results["classic_parameters"] = {
        "sigma": 3.0,
        "r": 25.0,
        "b": 1.0,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic laser Lyapunov: {lam_classic:.4f}")

    # Fixed points
    fps = sim_classic.fixed_points
    results["fixed_points"] = {
        "n_fixed_points": len(fps),
        "points": [fp.tolist() for fp in fps],
    }

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "ode_data.npz",
        states=ode_data["states"],
    )
    np.savez(
        output_path / "pump_sweep.npz",
        **{k: v for k, v in sweep_data.items() if isinstance(v, np.ndarray)},
    )
    np.savez(
        output_path / "lyapunov_fine.npz",
        r=fine_data["r"],
        lyapunov_exponent=fine_data["lyapunov_exponent"],
    )

    return results
