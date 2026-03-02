"""Chen attractor rediscovery.

Targets:
- SINDy recovery of Chen ODEs: x'=a*(y-x), y'=(c-a)*x-x*z+c*y, z'=x*y-b*z
- Lyapunov exponent estimation (positive for chaotic regime)
- c-parameter sweep mapping chaos transition
- Fixed point analysis (origin + two symmetric)
- Comparison with Lorenz coefficients (dual classification)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.chen import ChenSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use CHAOTIC_ODE as the domain enum placeholder until Domain.CHEN is added
_CHEN_DOMAIN = Domain.CHAOTIC_ODE


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.002,
    a: float = 35.0,
    b: float = 3.0,
    c: float = 28.0,
) -> dict[str, np.ndarray]:
    """Generate a single Chen trajectory for SINDy ODE recovery.

    Uses standard chaotic parameters by default. Small dt is needed because
    the Chen system has faster dynamics than Lorenz (larger coefficients).
    """
    config = SimulationConfig(
        domain=_CHEN_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a,
            "b": b,
            "c": c,
            "x_0": 1.0,
            "y_0": 1.0,
            "z_0": 1.0,
        },
    )
    sim = ChenSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "a": a,
        "b": b,
        "c": c,
    }


def generate_chaos_transition_data(
    n_c: int = 30,
    n_steps: int = 20000,
    dt: float = 0.002,
) -> dict[str, np.ndarray]:
    """Sweep c to map the transition to chaos.

    For the Chen system with a=35, b=3, chaos emerges as c increases.
    The condition c > 2a - b = 67 is a rough boundary, but for the standard
    parameter set chaos exists around c=28.
    """
    c_values = np.linspace(15.0, 35.0, n_c)
    lyapunov_exps = []
    attractor_types = []
    max_amplitudes = []

    for i, c in enumerate(c_values):
        config = SimulationConfig(
            domain=_CHEN_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={"a": 35.0, "b": 3.0, "c": c},
        )
        sim = ChenSimulation(config)
        sim.reset()

        # Run to skip transient
        for _ in range(5000):
            sim.step()

        # Estimate Lyapunov exponent
        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        # Run more steps to measure amplitude
        x_vals = []
        for _ in range(5000):
            state = sim.step()
            x_vals.append(state[0])

        max_amp = np.max(np.abs(x_vals))
        max_amplitudes.append(max_amp)

        # Classify attractor
        if lam > 0.5:
            atype = "chaotic"
        elif lam < -0.1:
            atype = "fixed_point"
        else:
            atype = "periodic_or_transient"
        attractor_types.append(atype)

        if (i + 1) % 10 == 0:
            logger.info(f"  c={c:.1f}: Lyapunov={lam:.3f}, type={atype}")

    return {
        "c": c_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
        "max_amplitude": np.array(max_amplitudes),
        "attractor_type": np.array(attractor_types),
    }


def generate_lyapunov_vs_c_data(
    n_c: int = 30,
    n_steps: int = 30000,
    dt: float = 0.002,
) -> dict[str, np.ndarray]:
    """Fine sweep of Lyapunov exponent as a function of c.

    Focuses on c in [20, 35] to capture the chaotic regime transitions.
    """
    c_values = np.linspace(20.0, 35.0, n_c)
    lyapunov_exps = []

    for i, c in enumerate(c_values):
        config = SimulationConfig(
            domain=_CHEN_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={"a": 35.0, "b": 3.0, "c": c},
        )
        sim = ChenSimulation(config)
        sim.reset()

        # Transient
        for _ in range(10000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  c={c:.2f}: Lyapunov={lam:.4f}")

    return {
        "c": c_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def run_chen_rediscovery(
    output_dir: str | Path = "output/rediscovery/chen",
    n_iterations: int = 40,
) -> dict:
    """Run the full Chen attractor rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep c to map chaos transition (Lyapunov exponent)
    3. Fine Lyapunov sweep
    4. Fixed point analysis
    5. Lorenz comparison (dual classification)

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "chen",
        "targets": {
            "ode_x": "dx/dt = a*(y - x)",
            "ode_y": "dy/dt = (c - a)*x - x*z + c*y",
            "ode_z": "dz/dt = x*y - b*z",
            "chaos_regime": "a=35, b=3, c=28 (standard chaotic parameters)",
            "lorenz_dual": "a12*a21 < 0 (opposite sign to Lorenz)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Chen trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=10000, dt=0.002)

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
            "true_a": ode_data["a"],
            "true_b": ode_data["b"],
            "true_c": ode_data["c"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Chaos transition sweep ---
    logger.info("Part 2: Mapping chaos transition (c sweep)...")
    chaos_data = generate_chaos_transition_data(n_c=30, n_steps=20000, dt=0.002)

    n_chaotic = int(np.sum(chaos_data["attractor_type"] == "chaotic"))
    n_fixed = int(np.sum(chaos_data["attractor_type"] == "fixed_point"))
    logger.info(f"  Found {n_chaotic} chaotic, {n_fixed} fixed-point regimes")

    results["chaos_transition"] = {
        "n_c_values": len(chaos_data["c"]),
        "n_chaotic": n_chaotic,
        "n_fixed_point": n_fixed,
        "c_range": [float(chaos_data["c"][0]), float(chaos_data["c"][-1])],
    }

    # --- Part 3: Fine Lyapunov sweep ---
    logger.info("Part 3: Fine Lyapunov exponent sweep...")
    fine_data = generate_lyapunov_vs_c_data(n_c=30, n_steps=30000, dt=0.002)

    lam = fine_data["lyapunov_exponent"]
    c_fine = fine_data["c"]

    results["lyapunov_analysis"] = {
        "n_points": len(c_fine),
        "c_range": [float(c_fine[0]), float(c_fine[-1])],
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
    }

    # --- Part 4: Lyapunov at classic parameters ---
    logger.info("Part 4: Lyapunov exponent at classic chaotic parameters...")
    config_classic = SimulationConfig(
        domain=_CHEN_DOMAIN,
        dt=0.002,
        n_steps=50000,
        parameters={"a": 35.0, "b": 3.0, "c": 28.0},
    )
    sim_classic = ChenSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.002)

    results["classic_parameters"] = {
        "a": 35.0,
        "b": 3.0,
        "c": 28.0,
        "lyapunov_exponent": float(lam_classic),
        "lyapunov_known": 2.027,  # Literature value for standard Chen
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic Chen Lyapunov: {lam_classic:.4f} (known: ~2.027)")

    # --- Part 5: Fixed points ---
    sim_fp = ChenSimulation(config_classic)
    sim_fp.reset()
    fps = sim_fp.fixed_points
    results["fixed_points"] = {
        "n_fixed_points": len(fps),
        "points": [fp.tolist() for fp in fps],
    }
    logger.info(f"  Fixed points: {len(fps)} found")
    for i, fp in enumerate(fps):
        derivs = sim_fp._derivatives(fp)
        logger.info(
            f"    FP{i+1}: [{fp[0]:.4f}, {fp[1]:.4f}, {fp[2]:.4f}], "
            f"|deriv|={np.linalg.norm(derivs):.2e}"
        )

    # --- Part 6: Lorenz comparison ---
    lorenz_comparison = sim_fp.compare_with_lorenz()
    results["lorenz_comparison"] = lorenz_comparison
    logger.info(
        f"  Lorenz dual check: a12*a21 = {lorenz_comparison['a12_times_a21']:.1f} "
        f"(Chen type: {lorenz_comparison['is_chen_type']})"
    )

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
        output_path / "chaos_transition.npz",
        **{k: v for k, v in chaos_data.items() if isinstance(v, np.ndarray)},
    )
    np.savez(
        output_path / "lyapunov_fine.npz",
        c=fine_data["c"],
        lyapunov_exponent=fine_data["lyapunov_exponent"],
    )

    return results
