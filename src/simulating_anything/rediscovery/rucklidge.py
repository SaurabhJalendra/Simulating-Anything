"""Rucklidge attractor rediscovery.

Targets:
- SINDy recovery of Rucklidge ODEs:
    x'=-kappa*x + lambda*y - y*z, y'=x, z'=-z + y^2
- Lyapunov exponent estimation (positive for chaotic regime)
- lambda-parameter sweep mapping chaos transition
- Fixed point analysis (origin + two symmetric at y=+-sqrt(lambda))
- Divergence verification: div = -(kappa + 1)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.rucklidge import RucklidgeSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_RUCKLIDGE_DOMAIN = Domain.RUCKLIDGE


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    kappa: float = 2.0,
    lambda_param: float = 6.7,
) -> dict[str, np.ndarray]:
    """Generate a single Rucklidge trajectory for SINDy ODE recovery.

    Uses standard chaotic parameters by default.
    """
    config = SimulationConfig(
        domain=_RUCKLIDGE_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "kappa": kappa,
            "lambda_param": lambda_param,
            "x_0": 1.0,
            "y_0": 0.0,
            "z_0": 4.5,
        },
    )
    sim = RucklidgeSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "kappa": kappa,
        "lambda_param": lambda_param,
    }


def generate_chaos_transition_data(
    n_lam: int = 30,
    n_steps: int = 20000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep lambda to map the transition to chaos.

    For the Rucklidge system with kappa=2.0, chaos emerges as lambda increases.
    """
    lam_values = np.linspace(2.0, 10.0, n_lam)
    lyapunov_exps = []
    attractor_types = []
    max_amplitudes = []

    for i, lam in enumerate(lam_values):
        config = SimulationConfig(
            domain=_RUCKLIDGE_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={"kappa": 2.0, "lambda_param": lam},
        )
        sim = RucklidgeSimulation(config)
        sim.reset()

        # Run to skip transient
        for _ in range(5000):
            sim.step()

        # Estimate Lyapunov exponent
        lyap = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lyap)

        # Run more steps to measure amplitude
        x_vals = []
        for _ in range(5000):
            state = sim.step()
            x_vals.append(state[0])

        max_amp = np.max(np.abs(x_vals))
        max_amplitudes.append(max_amp)

        # Classify attractor
        if lyap > 0.1:
            atype = "chaotic"
        elif lyap < -0.1:
            atype = "fixed_point"
        else:
            atype = "periodic_or_transient"
        attractor_types.append(atype)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  lambda={lam:.1f}: Lyapunov={lyap:.3f}, type={atype}"
            )

    return {
        "lambda_param": lam_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
        "max_amplitude": np.array(max_amplitudes),
        "attractor_type": np.array(attractor_types),
    }


def generate_lyapunov_vs_lambda_data(
    n_lam: int = 30,
    n_steps: int = 30000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Fine sweep of Lyapunov exponent as a function of lambda.

    Focuses on lambda in [4, 10] to capture chaotic regime transitions.
    """
    lam_values = np.linspace(4.0, 10.0, n_lam)
    lyapunov_exps = []

    for i, lam in enumerate(lam_values):
        config = SimulationConfig(
            domain=_RUCKLIDGE_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={"kappa": 2.0, "lambda_param": lam},
        )
        sim = RucklidgeSimulation(config)
        sim.reset()

        # Transient
        for _ in range(10000):
            sim.step()

        lyap = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lyap)

        if (i + 1) % 10 == 0:
            logger.info(f"  lambda={lam:.2f}: Lyapunov={lyap:.4f}")

    return {
        "lambda_param": lam_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def run_rucklidge_rediscovery(
    output_dir: str | Path = "output/rediscovery/rucklidge",
    n_iterations: int = 40,
) -> dict:
    """Run the full Rucklidge attractor rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep lambda to map chaos transition (Lyapunov exponent)
    3. Fine Lyapunov sweep
    4. Fixed point analysis
    5. Divergence verification

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "rucklidge",
        "targets": {
            "ode_x": "dx/dt = -kappa*x + lambda*y - y*z",
            "ode_y": "dy/dt = x",
            "ode_z": "dz/dt = -z + y^2",
            "chaos_regime": "kappa=2.0, lambda=6.7 (standard chaotic)",
            "divergence": "-(kappa + 1)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Rucklidge trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=10000, dt=0.01)

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
            "true_kappa": ode_data["kappa"],
            "true_lambda": ode_data["lambda_param"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Chaos transition sweep ---
    logger.info("Part 2: Mapping chaos transition (lambda sweep)...")
    chaos_data = generate_chaos_transition_data(
        n_lam=30, n_steps=20000, dt=0.01
    )

    n_chaotic = int(np.sum(chaos_data["attractor_type"] == "chaotic"))
    n_fixed = int(np.sum(chaos_data["attractor_type"] == "fixed_point"))
    logger.info(
        f"  Found {n_chaotic} chaotic, {n_fixed} fixed-point regimes"
    )

    results["chaos_transition"] = {
        "n_lambda_values": len(chaos_data["lambda_param"]),
        "n_chaotic": n_chaotic,
        "n_fixed_point": n_fixed,
        "lambda_range": [
            float(chaos_data["lambda_param"][0]),
            float(chaos_data["lambda_param"][-1]),
        ],
    }

    # --- Part 3: Fine Lyapunov sweep ---
    logger.info("Part 3: Fine Lyapunov exponent sweep...")
    fine_data = generate_lyapunov_vs_lambda_data(
        n_lam=30, n_steps=30000, dt=0.01
    )

    lam_lyap = fine_data["lyapunov_exponent"]
    lam_vals = fine_data["lambda_param"]

    results["lyapunov_analysis"] = {
        "n_points": len(lam_vals),
        "lambda_range": [float(lam_vals[0]), float(lam_vals[-1])],
        "max_lyapunov": float(np.max(lam_lyap)),
        "min_lyapunov": float(np.min(lam_lyap)),
    }

    # --- Part 4: Lyapunov at classic parameters ---
    logger.info(
        "Part 4: Lyapunov exponent at classic chaotic parameters..."
    )
    config_classic = SimulationConfig(
        domain=_RUCKLIDGE_DOMAIN,
        dt=0.01,
        n_steps=50000,
        parameters={"kappa": 2.0, "lambda_param": 6.7},
    )
    sim_classic = RucklidgeSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.01)

    results["classic_parameters"] = {
        "kappa": 2.0,
        "lambda_param": 6.7,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic Rucklidge Lyapunov: {lam_classic:.4f}")

    # --- Part 5: Fixed points ---
    sim_fp = RucklidgeSimulation(config_classic)
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

    # --- Part 6: Divergence verification ---
    div_val = sim_fp.compute_divergence()
    expected_div = -(2.0 + 1.0)
    results["divergence"] = {
        "computed": float(div_val),
        "expected": float(expected_div),
        "match": bool(np.isclose(div_val, expected_div)),
    }
    logger.info(
        f"  Divergence: {div_val:.4f} (expected {expected_div:.4f})"
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
        **{
            k: v
            for k, v in chaos_data.items()
            if isinstance(v, np.ndarray)
        },
    )
    np.savez(
        output_path / "lyapunov_fine.npz",
        lambda_param=fine_data["lambda_param"],
        lyapunov_exponent=fine_data["lyapunov_exponent"],
    )

    return results
