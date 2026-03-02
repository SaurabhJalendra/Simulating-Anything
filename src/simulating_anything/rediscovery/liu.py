"""Liu chaotic attractor rediscovery.

Targets:
- SINDy recovery of Liu ODEs: x'=-a*x-e*y^2, y'=b*y-k*x*z, z'=-c*z+m*x*y
- Lyapunov exponent estimation (positive for chaotic regime)
- a-parameter sweep mapping chaos transition
- Fixed point analysis (origin only for typical parameters)
- Dissipation rate verification: div = -(a - b + c)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.liu import LiuSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_LIU_DOMAIN = Domain.LIU


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.005,
    a: float = 1.0,
    b: float = 2.5,
    c: float = 5.0,
    e: float = 1.0,
    k: float = 4.0,
    m: float = 4.0,
) -> dict[str, np.ndarray]:
    """Generate a single Liu trajectory for SINDy ODE recovery.

    Uses standard chaotic parameters by default.
    """
    config = SimulationConfig(
        domain=_LIU_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a,
            "b": b,
            "c": c,
            "e": e,
            "k": k,
            "m": m,
            "x_0": 0.2,
            "y_0": 0.0,
            "z_0": 0.5,
        },
    )
    sim = LiuSimulation(config)
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
        "e": e,
        "k": k,
        "m": m,
    }


def generate_chaos_transition_data(
    n_a: int = 30,
    n_steps: int = 20000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep parameter a to map the transition to chaos.

    For the Liu system, varying a changes the balance between linear damping
    and nonlinear coupling, affecting the chaotic behavior.
    """
    a_values = np.linspace(0.1, 3.0, n_a)
    lyapunov_exps = []
    attractor_types = []
    max_amplitudes = []

    for i, a in enumerate(a_values):
        config = SimulationConfig(
            domain=_LIU_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "a": a, "b": 2.5, "c": 5.0,
                "e": 1.0, "k": 4.0, "m": 4.0,
            },
        )
        sim = LiuSimulation(config)
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
            logger.info(f"  a={a:.2f}: Lyapunov={lam:.3f}, type={atype}")

    return {
        "a": a_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
        "max_amplitude": np.array(max_amplitudes),
        "attractor_type": np.array(attractor_types),
    }


def generate_lyapunov_vs_a_data(
    n_a: int = 30,
    n_steps: int = 30000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Fine sweep of Lyapunov exponent as a function of a.

    Focuses on the region where chaos transitions occur.
    """
    a_values = np.linspace(0.5, 2.5, n_a)
    lyapunov_exps = []

    for i, a in enumerate(a_values):
        config = SimulationConfig(
            domain=_LIU_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "a": a, "b": 2.5, "c": 5.0,
                "e": 1.0, "k": 4.0, "m": 4.0,
            },
        )
        sim = LiuSimulation(config)
        sim.reset()

        # Transient
        for _ in range(10000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  a={a:.2f}: Lyapunov={lam:.4f}")

    return {
        "a": a_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def run_liu_rediscovery(
    output_dir: str | Path = "output/rediscovery/liu",
    n_iterations: int = 40,
) -> dict:
    """Run the full Liu attractor rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep a to map chaos transition (Lyapunov exponent)
    3. Fine Lyapunov sweep
    4. Fixed point analysis
    5. Dissipation rate verification

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "liu",
        "targets": {
            "ode_x": "dx/dt = -a*x - e*y^2",
            "ode_y": "dy/dt = b*y - k*x*z",
            "ode_z": "dz/dt = -c*z + m*x*y",
            "chaos_regime": "a=1, b=2.5, c=5, e=1, k=4, m=4",
            "divergence": "div = -a + b - c = -3.5 (dissipative)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Liu trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=10000, dt=0.005)

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
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in sindy_discoveries
            ],
            "true_a": ode_data["a"],
            "true_b": ode_data["b"],
            "true_c": ode_data["c"],
            "true_e": ode_data["e"],
            "true_k": ode_data["k"],
            "true_m": ode_data["m"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as exc:
        logger.warning(f"SINDy failed: {exc}")
        results["sindy_ode"] = {"error": str(exc)}

    # --- Part 2: Chaos transition sweep ---
    logger.info("Part 2: Mapping chaos transition (a sweep)...")
    chaos_data = generate_chaos_transition_data(n_a=30, n_steps=20000, dt=0.005)

    n_chaotic = int(np.sum(chaos_data["attractor_type"] == "chaotic"))
    n_fixed = int(np.sum(chaos_data["attractor_type"] == "fixed_point"))
    logger.info(f"  Found {n_chaotic} chaotic, {n_fixed} fixed-point regimes")

    results["chaos_transition"] = {
        "n_a_values": len(chaos_data["a"]),
        "n_chaotic": n_chaotic,
        "n_fixed_point": n_fixed,
        "a_range": [float(chaos_data["a"][0]), float(chaos_data["a"][-1])],
    }

    # --- Part 3: Fine Lyapunov sweep ---
    logger.info("Part 3: Fine Lyapunov exponent sweep...")
    fine_data = generate_lyapunov_vs_a_data(n_a=30, n_steps=30000, dt=0.005)

    lam = fine_data["lyapunov_exponent"]
    a_fine = fine_data["a"]

    results["lyapunov_analysis"] = {
        "n_points": len(a_fine),
        "a_range": [float(a_fine[0]), float(a_fine[-1])],
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
    }

    # --- Part 4: Lyapunov at classic parameters ---
    logger.info("Part 4: Lyapunov exponent at classic chaotic parameters...")
    config_classic = SimulationConfig(
        domain=_LIU_DOMAIN,
        dt=0.005,
        n_steps=50000,
        parameters={
            "a": 1.0, "b": 2.5, "c": 5.0,
            "e": 1.0, "k": 4.0, "m": 4.0,
        },
    )
    sim_classic = LiuSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.005)

    results["classic_parameters"] = {
        "a": 1.0,
        "b": 2.5,
        "c": 5.0,
        "e": 1.0,
        "k": 4.0,
        "m": 4.0,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic Liu Lyapunov: {lam_classic:.4f}")

    # --- Part 5: Fixed points ---
    sim_fp = LiuSimulation(config_classic)
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

    # --- Part 6: Dissipation rate ---
    div = sim_fp.compute_divergence()
    results["dissipation"] = {
        "divergence": float(div),
        "expected": float(-1.0 + 2.5 - 5.0),
        "is_dissipative": bool(div < 0),
    }
    logger.info(f"  Divergence: {div:.2f} (expected -3.5)")

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
        a=fine_data["a"],
        lyapunov_exponent=fine_data["lyapunov_exponent"],
    )

    return results
