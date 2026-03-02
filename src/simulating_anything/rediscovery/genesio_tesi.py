"""Genesio-Tesi system rediscovery.

Targets:
- SINDy recovery of Genesio-Tesi ODEs: x'=y, y'=z, z'=-c*x-b*y-a*z+x^2
- Jerk form: x''' + a*x'' + b*x' + c*x = x^2
- Hopf bifurcation / chaos transition as a varies
- Lyapunov exponent at classic parameters (a=0.44, b=1.1, c=1.0)
- Fixed point verification (origin and x=c)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.genesio_tesi import GenesioTesiSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_trajectory_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    a: float = 0.44,
    b: float = 1.1,
    c: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate a single Genesio-Tesi trajectory for SINDy ODE recovery.

    Uses standard chaotic parameters by default.
    """
    config = SimulationConfig(
        domain=Domain.GENESIO_TESI,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a,
            "b": b,
            "c": c,
            "x_0": 0.1,
            "y_0": 0.1,
            "z_0": 0.1,
        },
    )
    sim = GenesioTesiSimulation(config)
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


def generate_lyapunov_vs_a_data(
    n_a: int = 30,
    n_steps: int = 30000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep parameter a to map the chaos transition.

    For b=1.1, c=1.0:
    - a ~ 0.44: chaotic regime
    - a > ~0.5: periodic / stable
    - a < ~0.3: can diverge (unbounded)

    Focuses on a in [0.30, 0.60] to capture the transition.
    """
    a_values = np.linspace(0.30, 0.60, n_a)
    lyapunov_exps = []

    for i, a in enumerate(a_values):
        config = SimulationConfig(
            domain=Domain.GENESIO_TESI,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "a": a,
                "b": 1.1,
                "c": 1.0,
                "x_0": 0.1,
                "y_0": 0.1,
                "z_0": 0.1,
            },
        )
        sim = GenesioTesiSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  a={a:.3f}: Lyapunov={lam:.4f}")

    return {
        "a": a_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def generate_jerk_verification_data(
    n_steps: int = 2000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Verify the jerk form: x''' + a*x'' + b*x' + c*x = x^2.

    Collects the trajectory and computes all three derivatives of x
    to verify they satisfy the jerk equation.
    """
    a, b, c = 0.44, 1.1, 1.0
    config = SimulationConfig(
        domain=Domain.GENESIO_TESI,
        dt=dt,
        n_steps=n_steps,
        parameters={"a": a, "b": b, "c": c, "x_0": 0.1, "y_0": 0.1, "z_0": 0.1},
    )
    sim = GenesioTesiSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    jerks = []
    for _ in range(n_steps):
        state = sim.step().copy()
        states.append(state)
        jerks.append(sim.compute_jerk(state))

    states_arr = np.array(states)
    jerks_arr = np.array(jerks)

    # The jerk should equal dz/dt at each point
    # dz/dt = -c*x - b*y - a*z + x^2 = jerk
    x_vals = states_arr[1:, 0]
    jerk_from_eq = x_vals**2 - c * x_vals - b * states_arr[1:, 1] - a * states_arr[1:, 2]

    return {
        "states": states_arr,
        "jerk_computed": jerks_arr,
        "jerk_from_equation": jerk_from_eq,
        "max_jerk_error": float(np.max(np.abs(jerks_arr - jerk_from_eq))),
    }


def run_genesio_tesi_rediscovery(
    output_dir: str | Path = "output/rediscovery/genesio_tesi",
    n_iterations: int = 40,
) -> dict:
    """Run the full Genesio-Tesi system rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep a to map chaos transition (Lyapunov exponent)
    3. Verify jerk form
    4. Compute Lyapunov at standard parameters
    5. Verify fixed points

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "genesio_tesi",
        "targets": {
            "ode_x": "dx/dt = y",
            "ode_y": "dy/dt = z",
            "ode_z": "dz/dt = -c*x - b*y - a*z + x^2",
            "jerk_form": "x''' + a*x'' + b*x' + c*x = x^2",
            "chaos_regime": "a=0.44, b=1.1, c=1.0",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Genesio-Tesi trajectory for SINDy...")
    ode_data = generate_trajectory_data(n_steps=10000, dt=0.01)

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
            "true_c": ode_data["c"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Lyapunov vs a sweep ---
    logger.info("Part 2: Mapping chaos transition (a sweep)...")
    lyap_data = generate_lyapunov_vs_a_data(n_a=30, n_steps=30000, dt=0.01)

    # Find approximate critical a (Lyapunov zero crossing)
    lam = lyap_data["lyapunov_exponent"]
    a_vals = lyap_data["a"]
    zero_crossings = []
    for j in range(len(lam) - 1):
        if lam[j] > 0 and lam[j + 1] <= 0:
            frac = lam[j] / (lam[j] - lam[j + 1])
            a_cross = a_vals[j] + frac * (a_vals[j + 1] - a_vals[j])
            zero_crossings.append(float(a_cross))

    n_chaotic = int(np.sum(lam > 0.01))
    n_stable = int(np.sum(lam < -0.01))

    results["chaos_transition"] = {
        "n_a_values": len(a_vals),
        "n_chaotic": n_chaotic,
        "n_stable": n_stable,
        "a_range": [float(a_vals[0]), float(a_vals[-1])],
        "zero_crossings": zero_crossings,
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
    }
    if zero_crossings:
        logger.info(f"  Lyapunov zero crossings at a = {zero_crossings}")

    # --- Part 3: Jerk form verification ---
    logger.info("Part 3: Verifying jerk form...")
    jerk_data = generate_jerk_verification_data(n_steps=2000, dt=0.01)

    results["jerk_form"] = {
        "max_jerk_error": jerk_data["max_jerk_error"],
        "verified": jerk_data["max_jerk_error"] < 1e-10,
    }
    logger.info(f"  Jerk form error: {jerk_data['max_jerk_error']:.2e}")

    # --- Part 4: Lyapunov at classic parameters ---
    logger.info("Part 4: Lyapunov exponent at classic chaotic parameters...")
    config_classic = SimulationConfig(
        domain=Domain.GENESIO_TESI,
        dt=0.01,
        n_steps=50000,
        parameters={"a": 0.44, "b": 1.1, "c": 1.0},
    )
    sim_classic = GenesioTesiSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.01)

    results["classic_parameters"] = {
        "a": 0.44,
        "b": 1.1,
        "c": 1.0,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic Genesio-Tesi Lyapunov: {lam_classic:.4f}")

    # --- Part 5: Fixed points ---
    logger.info("Part 5: Verifying fixed points...")
    sim_fp = GenesioTesiSimulation(config_classic)
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
            f"    FP{i + 1}: [{fp[0]:.4f}, {fp[1]:.4f}, {fp[2]:.4f}], "
            f"|deriv|={np.linalg.norm(derivs):.2e}"
        )

    # --- Part 6: Eigenvalue analysis at origin ---
    logger.info("Part 6: Eigenvalue analysis...")
    eigs_origin = sim_fp.eigenvalues_at_origin()
    results["eigenvalues_origin"] = {
        "eigenvalues": [complex(e).real for e in eigs_origin],
        "eigenvalues_imag": [complex(e).imag for e in eigs_origin],
        "any_positive_real": bool(any(complex(e).real > 0 for e in eigs_origin)),
    }
    logger.info(f"  Eigenvalues at origin: {eigs_origin}")

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
        output_path / "lyapunov_vs_a.npz",
        a=lyap_data["a"],
        lyapunov_exponent=lyap_data["lyapunov_exponent"],
    )

    return results
