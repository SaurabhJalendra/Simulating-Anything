"""Newton-Leipnik attractor rediscovery.

Targets:
- SINDy recovery of Newton-Leipnik ODEs:
    x' = -a*x + y + 10*y*z
    y' = -x - 0.4*y + 5*x*z
    z' = b*z - 5*x*y
- Lyapunov exponent estimation (positive for chaotic regime)
- a-parameter sweep mapping chaos transition
- Fixed point analysis
- Multistability verification (two coexisting attractors)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.newton_leipnik import NewtonLeipnikSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use CHAOTIC_ODE as the domain enum placeholder until Domain.NEWTON_LEIPNIK is added
_NL_DOMAIN = Domain.CHAOTIC_ODE


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.005,
    a: float = 0.4,
    b: float = 0.175,
) -> dict[str, np.ndarray]:
    """Generate a single Newton-Leipnik trajectory for SINDy ODE recovery.

    Uses standard chaotic parameters by default. Small dt is needed because
    the bilinear coupling terms (10*y*z, 5*x*z) can create fast dynamics.
    """
    config = SimulationConfig(
        domain=_NL_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a,
            "b": b,
            "x_0": 0.349,
            "y_0": 0.0,
            "z_0": -0.16,
        },
    )
    sim = NewtonLeipnikSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "a": a,
        "b": b,
    }


def generate_chaos_transition_data(
    n_a: int = 30,
    n_steps: int = 20000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep a to map the transition to chaos.

    For the Newton-Leipnik system with b=0.175, chaos depends on the
    balance between dissipation (-a term) and nonlinear coupling.
    """
    a_values = np.linspace(0.1, 1.0, n_a)
    lyapunov_exps = []
    attractor_types = []
    max_amplitudes = []

    for i, a in enumerate(a_values):
        config = SimulationConfig(
            domain=_NL_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={"a": a, "b": 0.175, "x_0": 0.349, "y_0": 0.0, "z_0": -0.16},
        )
        sim = NewtonLeipnikSimulation(config)
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
        if lam > 0.1:
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

    Focuses on a in [0.1, 1.0] to capture the chaotic regime transitions.
    """
    a_values = np.linspace(0.1, 1.0, n_a)
    lyapunov_exps = []

    for i, a in enumerate(a_values):
        config = SimulationConfig(
            domain=_NL_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={"a": a, "b": 0.175, "x_0": 0.349, "y_0": 0.0, "z_0": -0.16},
        )
        sim = NewtonLeipnikSimulation(config)
        sim.reset()

        # Transient
        for _ in range(10000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  a={a:.3f}: Lyapunov={lam:.4f}")

    return {
        "a": a_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def generate_multistability_data(
    dt: float = 0.005,
    n_steps: int = 10000,
    a: float = 0.4,
    b: float = 0.175,
) -> dict[str, np.ndarray]:
    """Demonstrate multistability by running from two different initial conditions.

    The Newton-Leipnik system has two coexisting attractors. Different initial
    conditions lead to qualitatively different trajectories.
    """
    # Attractor 1: default initial conditions
    config1 = SimulationConfig(
        domain=_NL_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={"a": a, "b": b, "x_0": 0.349, "y_0": 0.0, "z_0": -0.16},
    )
    sim1 = NewtonLeipnikSimulation(config1)
    sim1.reset()
    states1 = [sim1.observe().copy()]
    for _ in range(n_steps):
        states1.append(sim1.step().copy())

    # Attractor 2: different initial conditions
    config2 = SimulationConfig(
        domain=_NL_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={"a": a, "b": b, "x_0": 0.349, "y_0": 0.0, "z_0": 0.16},
    )
    sim2 = NewtonLeipnikSimulation(config2)
    sim2.reset()
    states2 = [sim2.observe().copy()]
    for _ in range(n_steps):
        states2.append(sim2.step().copy())

    arr1 = np.array(states1)
    arr2 = np.array(states2)

    # Compute z-range statistics to check for distinct attractors
    z_mean1 = float(np.mean(arr1[n_steps // 2:, 2]))
    z_mean2 = float(np.mean(arr2[n_steps // 2:, 2]))

    return {
        "states_attractor1": arr1,
        "states_attractor2": arr2,
        "z_mean_attractor1": z_mean1,
        "z_mean_attractor2": z_mean2,
    }


def run_newton_leipnik_rediscovery(
    output_dir: str | Path = "output/rediscovery/newton_leipnik",
    n_iterations: int = 40,
) -> dict:
    """Run the full Newton-Leipnik attractor rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep a to map chaos transition (Lyapunov exponent)
    3. Fine Lyapunov sweep
    4. Fixed point analysis
    5. Multistability demonstration

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "newton_leipnik",
        "targets": {
            "ode_x": "dx/dt = -a*x + y + 10*y*z",
            "ode_y": "dy/dt = -x - 0.4*y + 5*x*z",
            "ode_z": "dz/dt = b*z - 5*x*y",
            "chaos_regime": "a=0.4, b=0.175 (standard chaotic parameters)",
            "multistability": "Two coexisting strange attractors",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Newton-Leipnik trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["x", "y", "z"],
            threshold=0.05,
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
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

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
        domain=_NL_DOMAIN,
        dt=0.005,
        n_steps=50000,
        parameters={"a": 0.4, "b": 0.175, "x_0": 0.349, "y_0": 0.0, "z_0": -0.16},
    )
    sim_classic = NewtonLeipnikSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.005)

    results["classic_parameters"] = {
        "a": 0.4,
        "b": 0.175,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
        "divergence": float(sim_classic.divergence),
    }
    logger.info(f"  Classic Newton-Leipnik Lyapunov: {lam_classic:.4f}")

    # --- Part 5: Fixed points ---
    sim_fp = NewtonLeipnikSimulation(config_classic)
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

    # --- Part 6: Multistability ---
    logger.info("Part 6: Demonstrating multistability...")
    multi_data = generate_multistability_data(dt=0.005, n_steps=10000)
    results["multistability"] = {
        "z_mean_attractor1": multi_data["z_mean_attractor1"],
        "z_mean_attractor2": multi_data["z_mean_attractor2"],
        "attractors_differ": bool(
            abs(multi_data["z_mean_attractor1"] - multi_data["z_mean_attractor2"])
            > 0.01
        ),
    }
    logger.info(
        f"  Attractor 1 z_mean: {multi_data['z_mean_attractor1']:.4f}, "
        f"Attractor 2 z_mean: {multi_data['z_mean_attractor2']:.4f}"
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
        a=fine_data["a"],
        lyapunov_exponent=fine_data["lyapunov_exponent"],
    )

    return results
