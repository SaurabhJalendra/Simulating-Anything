"""Vallis ENSO model rediscovery.

Targets:
- SINDy recovery of Vallis ODEs: x'=B*y-C*(x-p), y'=-y+x*z, z'=-z-x*y+1
- Lyapunov exponent estimation (positive for chaotic regime)
- B-parameter sweep mapping chaos transition
- Fixed point analysis
- Constant divergence = -(C + 2) verification
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.vallis import VallisSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_VALLIS_DOMAIN = Domain.VALLIS


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.005,
    B: float = 102.0,
    C: float = 3.0,
    p: float = 0.0,
) -> dict[str, np.ndarray]:
    """Generate a single Vallis trajectory for SINDy ODE recovery.

    Uses standard parameters by default. The system models coupled
    ocean-atmosphere dynamics for the El Nino-Southern Oscillation.
    """
    config = SimulationConfig(
        domain=_VALLIS_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "B": B,
            "C": C,
            "p": p,
            "x_0": 0.1,
            "y_0": 0.2,
            "z_0": 0.3,
        },
    )
    sim = VallisSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "B": B,
        "C": C,
        "p": p,
    }


def generate_lyapunov_vs_B_data(
    n_B: int = 30,
    n_steps: int = 30000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep B to map the transition to chaos.

    The Vallis system exhibits chaotic dynamics for sufficiently large
    coupling strength B. This sweep captures the Lyapunov exponent
    as a function of B.
    """
    B_values = np.linspace(10.0, 150.0, n_B)
    lyapunov_exps = []

    for i, B in enumerate(B_values):
        config = SimulationConfig(
            domain=_VALLIS_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={"B": B, "C": 3.0, "p": 0.0},
        )
        sim = VallisSimulation(config)
        sim.reset()

        # Transient
        for _ in range(10000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  B={B:.1f}: Lyapunov={lam:.4f}")

    return {
        "B": B_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def run_vallis_rediscovery(
    output_dir: str | Path = "output/rediscovery/vallis",
    n_iterations: int = 40,
) -> dict:
    """Run the full Vallis ENSO model rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. B-parameter sweep mapping chaos transition (Lyapunov exponent)
    3. Fixed point analysis
    4. Divergence verification
    5. Lyapunov at classic parameters

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "vallis",
        "targets": {
            "ode_x": "dx/dt = B*y - C*(x - p)",
            "ode_y": "dy/dt = -y + x*z",
            "ode_z": "dz/dt = -z - x*y + 1",
            "divergence": "-(C + 2) = -5 for default C=3",
            "physics": "ENSO coupled ocean-atmosphere dynamics",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Vallis trajectory for SINDy...")
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
            "true_B": ode_data["B"],
            "true_C": ode_data["C"],
            "true_p": ode_data["p"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Lyapunov sweep over B ---
    logger.info("Part 2: Lyapunov exponent sweep over B...")
    lyap_data = generate_lyapunov_vs_B_data(
        n_B=30, n_steps=30000, dt=0.005
    )

    lam = lyap_data["lyapunov_exponent"]
    B_vals = lyap_data["B"]

    n_chaotic = int(np.sum(lam > 0.01))
    n_stable = int(np.sum(lam < -0.01))
    logger.info(
        f"  Found {n_chaotic} chaotic, {n_stable} stable regimes"
    )

    results["lyapunov_sweep"] = {
        "n_B_values": len(B_vals),
        "B_range": [float(B_vals[0]), float(B_vals[-1])],
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
        "n_chaotic": n_chaotic,
        "n_stable": n_stable,
    }

    # --- Part 3: Lyapunov at classic parameters ---
    logger.info("Part 3: Lyapunov exponent at classic parameters...")
    config_classic = SimulationConfig(
        domain=_VALLIS_DOMAIN,
        dt=0.005,
        n_steps=50000,
        parameters={"B": 102.0, "C": 3.0, "p": 0.0},
    )
    sim_classic = VallisSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(
        n_steps=50000, dt=0.005
    )

    results["classic_parameters"] = {
        "B": 102.0,
        "C": 3.0,
        "p": 0.0,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic Vallis Lyapunov: {lam_classic:.4f}")

    # --- Part 4: Fixed points ---
    logger.info("Part 4: Fixed point analysis...")
    sim_fp = VallisSimulation(config_classic)
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

    # --- Part 5: Divergence verification ---
    logger.info("Part 5: Divergence verification...")
    expected_div = -(3.0 + 2.0)  # -(C + 2) for C=3
    computed_div = sim_fp.compute_divergence(
        np.array([1.0, 2.0, 3.0])
    )
    results["divergence"] = {
        "expected": float(expected_div),
        "computed": float(computed_div),
        "match": bool(abs(computed_div - expected_div) < 1e-12),
    }
    logger.info(
        f"  Divergence: expected={expected_div}, "
        f"computed={computed_div}"
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
        output_path / "lyapunov_sweep.npz",
        B=lyap_data["B"],
        lyapunov_exponent=lyap_data["lyapunov_exponent"],
    )

    return results
