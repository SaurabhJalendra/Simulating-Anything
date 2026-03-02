"""Hadley circulation rediscovery.

Targets:
- SINDy recovery of Hadley ODEs
- Lyapunov exponent sweep over F (chaos transition)
- Divergence = -(a + 2) verification
- Fixed point analysis
- Hadley fixed point verification for G=0
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.hadley import HadleySimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_DOMAIN = Domain.HADLEY


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    a: float = 0.2,
    b: float = 4.0,
    F: float = 8.0,
    G: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate a single Hadley trajectory for SINDy ODE recovery.

    Uses standard parameters by default. The system exhibits chaotic
    dynamics at these values.
    """
    config = SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a, "b": b, "F": F, "G": G,
            "x_0": 0.0, "y_0": 1.0, "z_0": 0.0,
        },
    )
    sim = HadleySimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "a": a,
        "b": b,
        "F": F,
        "G": G,
    }


def generate_lyapunov_vs_F_data(
    n_F: int = 30,
    n_steps: int = 30000,
    dt: float = 0.01,
    a: float = 0.2,
    b: float = 4.0,
    G: float = 1.0,
) -> dict[str, np.ndarray]:
    """Sweep F to map the Lyapunov exponent transition.

    For the Hadley system with a=0.2, b=4, G=1, chaos emerges as F
    increases through a sequence of bifurcations.
    """
    F_values = np.linspace(1.0, 10.0, n_F)
    lyapunov_exps = []
    attractor_types = []

    for i, F in enumerate(F_values):
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "a": a, "b": b, "F": F, "G": G,
                "x_0": 0.0, "y_0": 1.0, "z_0": 0.0,
            },
        )
        sim = HadleySimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        lam = sim.lyapunov_exponent(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if lam < -0.01:
            atype = "fixed_point"
        elif lam < 0.01:
            atype = "periodic_or_quasiperiodic"
        else:
            atype = "chaotic"
        attractor_types.append(atype)

        if (i + 1) % 10 == 0:
            logger.info(f"  F={F:.2f}: Lyapunov={lam:.4f}, type={atype}")

    return {
        "F": F_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
        "attractor_type": np.array(attractor_types),
    }


def generate_hadley_fp_verification_data(
    n_F: int = 20,
    n_steps: int = 10000,
    dt: float = 0.01,
    a: float = 0.2,
    b: float = 4.0,
) -> dict[str, np.ndarray]:
    """Verify the Hadley fixed point x* = F for G=0 across F values.

    The Hadley fixed point (F, 0, 0) is linearly stable when the
    eigenvalues of the Jacobian at this point all have negative real part.
    We sweep F in the stable range.
    """
    F_values = np.linspace(0.1, 0.9, n_F)
    x_final = []
    y_final = []
    z_final = []

    for F in F_values:
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "a": a, "b": b, "F": F, "G": 0.0,
                "x_0": F + 0.1, "y_0": 0.1, "z_0": 0.1,
            },
        )
        sim = HadleySimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        state = sim.observe()
        x_final.append(state[0])
        y_final.append(state[1])
        z_final.append(state[2])

    return {
        "F": F_values,
        "x_final": np.array(x_final),
        "y_final": np.array(y_final),
        "z_final": np.array(z_final),
    }


def run_hadley_rediscovery(
    output_dir: str | Path = "output/rediscovery/hadley",
    n_iterations: int = 40,
) -> dict:
    """Run the full Hadley circulation rediscovery pipeline.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep F to map chaos transition (Lyapunov exponent)
    3. Verify Hadley fixed point x* = F for G=0
    4. Fixed point analysis
    5. Divergence verification

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "hadley",
        "targets": {
            "ode_x": "dx/dt = -y^2 - z^2 - a*x + a*F",
            "ode_y": "dy/dt = x*y - b*x*z - y + G",
            "ode_z": "dz/dt = b*x*y + x*z - z",
            "hadley_fp": "x* = F, y* = 0, z* = 0 (for G=0)",
            "divergence": "div = -(a + 2)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Hadley trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=5000, dt=0.01)

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
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in sindy_discoveries
            ],
            "true_a": ode_data["a"],
            "true_b": ode_data["b"],
            "true_F": ode_data["F"],
            "true_G": ode_data["G"],
        }
        if sindy_discoveries:
            best = sindy_discoveries[0]
            results["sindy_ode"]["best"] = best.expression
            results["sindy_ode"]["best_r2"] = best.evidence.fit_r_squared
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Lyapunov sweep over F ---
    logger.info("Part 2: Lyapunov exponent sweep over F...")
    lyap_data = generate_lyapunov_vs_F_data(n_F=30, n_steps=20000, dt=0.01)

    n_chaotic = int(np.sum(lyap_data["attractor_type"] == "chaotic"))
    n_fixed = int(np.sum(lyap_data["attractor_type"] == "fixed_point"))
    n_periodic = int(
        np.sum(lyap_data["attractor_type"] == "periodic_or_quasiperiodic")
    )
    logger.info(
        f"  Found {n_chaotic} chaotic, {n_fixed} fixed-point, "
        f"{n_periodic} periodic/QP regimes"
    )

    results["lyapunov_sweep"] = {
        "n_F_values": len(lyap_data["F"]),
        "n_chaotic": n_chaotic,
        "n_fixed_point": n_fixed,
        "n_periodic": n_periodic,
        "F_range": [float(lyap_data["F"][0]), float(lyap_data["F"][-1])],
        "max_lyapunov": float(np.max(lyap_data["lyapunov_exponent"])),
        "min_lyapunov": float(np.min(lyap_data["lyapunov_exponent"])),
    }

    # Find approximate critical F for chaos onset
    mask_positive = lyap_data["lyapunov_exponent"] > 0.01
    if np.any(mask_positive):
        F_c_approx = float(lyap_data["F"][np.argmax(mask_positive)])
        results["lyapunov_sweep"]["F_c_approx"] = F_c_approx
        logger.info(f"  Approximate chaos onset F_c: {F_c_approx:.2f}")

    # --- Part 3: Hadley fixed point verification ---
    logger.info("Part 3: Verifying Hadley fixed point x* = F for G=0...")
    fp_data = generate_hadley_fp_verification_data(
        n_F=20, n_steps=10000, dt=0.01
    )

    errors = np.abs(fp_data["x_final"] - fp_data["F"])
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))

    y_max = float(np.max(np.abs(fp_data["y_final"])))
    z_max = float(np.max(np.abs(fp_data["z_final"])))

    results["hadley_verification"] = {
        "n_F_values": len(fp_data["F"]),
        "F_range": [float(fp_data["F"][0]), float(fp_data["F"][-1])],
        "mean_error_x_vs_F": mean_error,
        "max_error_x_vs_F": max_error,
        "max_abs_y": y_max,
        "max_abs_z": z_max,
        "verified": mean_error < 0.1,
    }
    logger.info(
        f"  Hadley FP: mean |x*-F| = {mean_error:.6f}, "
        f"max |y| = {y_max:.6f}, max |z| = {z_max:.6f}"
    )

    # --- Part 4: Fixed points at standard parameters ---
    logger.info("Part 4: Fixed point analysis...")
    config_std = SimulationConfig(
        domain=_DOMAIN,
        dt=0.01,
        n_steps=10000,
        parameters={"a": 0.2, "b": 4.0, "F": 8.0, "G": 1.0},
    )
    sim_fp = HadleySimulation(config_std)
    sim_fp.reset()
    fps = sim_fp.fixed_points()
    results["fixed_points"] = {
        "n_fixed_points": len(fps),
        "points": [fp.tolist() for fp in fps],
    }
    logger.info(f"  Found {len(fps)} fixed points")
    for i, fp in enumerate(fps):
        derivs = sim_fp._derivatives(fp)
        logger.info(
            f"    FP{i+1}: [{fp[0]:.4f}, {fp[1]:.4f}, {fp[2]:.4f}], "
            f"|deriv|={np.linalg.norm(derivs):.2e}"
        )

    # --- Part 5: Divergence verification ---
    logger.info("Part 5: Divergence verification...")
    div_expected = -(0.2 + 2.0)  # = -2.2
    div_computed = sim_fp.compute_divergence()
    results["divergence"] = {
        "expected": div_expected,
        "computed": div_computed,
        "match": bool(np.isclose(div_computed, div_expected)),
    }
    logger.info(
        f"  Divergence: computed={div_computed:.4f}, "
        f"expected={div_expected:.4f}"
    )

    # --- Part 6: Lyapunov at standard parameters ---
    logger.info("Part 6: Lyapunov exponent at standard parameters...")
    config_classic = SimulationConfig(
        domain=_DOMAIN,
        dt=0.005,
        n_steps=50000,
        parameters={
            "a": 0.2, "b": 4.0, "F": 8.0, "G": 1.0,
            "x_0": 0.0, "y_0": 1.0, "z_0": 0.0,
        },
    )
    sim_classic = HadleySimulation(config_classic)
    sim_classic.reset()

    for _ in range(10000):
        sim_classic.step()

    lam_classic = sim_classic.lyapunov_exponent(n_steps=50000, dt=0.005)

    results["classic_parameters"] = {
        "a": 0.2,
        "b": 4.0,
        "F": 8.0,
        "G": 1.0,
        "lyapunov_exponent": float(lam_classic),
        "is_chaotic": bool(lam_classic > 0.01),
    }
    logger.info(f"  Standard Hadley Lyapunov: {lam_classic:.4f}")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data arrays
    np.savez(
        output_path / "ode_data.npz",
        states=ode_data["states"],
    )
    np.savez(
        output_path / "lyapunov_sweep.npz",
        F=lyap_data["F"],
        lyapunov_exponent=lyap_data["lyapunov_exponent"],
    )
    np.savez(
        output_path / "hadley_fp_data.npz",
        F=fp_data["F"],
        x_final=fp_data["x_final"],
        y_final=fp_data["y_final"],
        z_final=fp_data["z_final"],
    )

    return results
