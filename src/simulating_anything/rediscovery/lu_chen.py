"""Lu-Chen attractor rediscovery.

Targets:
- SINDy recovery of Lu-Chen ODEs: x'=a*(y-x), y'=x-x*z+c*y, z'=x*y-b*z
- Lyapunov exponent estimation (positive for chaotic regime)
- c-parameter sweep mapping Lu/Lu-Chen/Chen transitions
- Fixed point analysis (origin + two symmetric)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.lu_chen import LuChenSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use CHAOTIC_ODE as the domain enum placeholder until Domain.LU_CHEN is added
_LU_CHEN_DOMAIN = Domain.CHAOTIC_ODE


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.001,
    a: float = 36.0,
    b: float = 3.0,
    c: float = 20.0,
) -> dict[str, np.ndarray]:
    """Generate a single Lu-Chen trajectory for SINDy ODE recovery.

    Uses standard chaotic parameters by default. Small dt is needed because
    the Lu-Chen system has fast dynamics due to large coefficients (a=36).
    """
    config = SimulationConfig(
        domain=_LU_CHEN_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a,
            "b": b,
            "c": c,
            "x_0": 0.1,
            "y_0": 0.3,
            "z_0": -0.6,
        },
    )
    sim = LuChenSimulation(config)
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


def generate_c_sweep_data(
    n_c: int = 30,
    n_steps: int = 20000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Sweep c to map the Lu/Lu-Chen/Chen transition.

    For the Lu-Chen system with a=36, b=3:
    - c ~ 12: Lu-like attractor
    - c ~ 20: Lu-Chen chaotic attractor
    - c ~ 28: Chen-like attractor
    """
    c_values = np.linspace(5.0, 30.0, n_c)
    lyapunov_exps = []
    attractor_types = []
    max_amplitudes = []

    for i, c in enumerate(c_values):
        config = SimulationConfig(
            domain=_LU_CHEN_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={"a": 36.0, "b": 3.0, "c": c},
        )
        sim = LuChenSimulation(config)
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

        # Classify attractor regime
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
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Fine sweep of Lyapunov exponent as a function of c.

    Covers c in [5, 30] to capture all three regimes (Lu, Lu-Chen, Chen).
    """
    c_values = np.linspace(5.0, 30.0, n_c)
    lyapunov_exps = []

    for i, c in enumerate(c_values):
        config = SimulationConfig(
            domain=_LU_CHEN_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={"a": 36.0, "b": 3.0, "c": c},
        )
        sim = LuChenSimulation(config)
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


def run_lu_chen_rediscovery(
    output_dir: str | Path = "output/rediscovery/lu_chen",
    n_iterations: int = 40,
) -> dict:
    """Run the full Lu-Chen attractor rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep c to map Lu/Lu-Chen/Chen transitions
    3. Fine Lyapunov sweep
    4. Fixed point analysis
    5. Regime classification across c values

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "lu_chen",
        "targets": {
            "ode_x": "dx/dt = a*(y - x)",
            "ode_y": "dy/dt = x - x*z + c*y",
            "ode_z": "dz/dt = x*y - b*z",
            "chaos_regime": "a=36, b=3, c=20 (standard Lu-Chen chaotic)",
            "unification": "c=12 (Lu), c=20 (Lu-Chen), c=28 (Chen)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Lu-Chen trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=10000, dt=0.001)

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

    # --- Part 2: c-parameter sweep ---
    logger.info("Part 2: Mapping Lu/Lu-Chen/Chen transition (c sweep)...")
    c_sweep_data = generate_c_sweep_data(n_c=30, n_steps=20000, dt=0.001)

    n_chaotic = int(np.sum(c_sweep_data["attractor_type"] == "chaotic"))
    n_fixed = int(np.sum(c_sweep_data["attractor_type"] == "fixed_point"))
    logger.info(f"  Found {n_chaotic} chaotic, {n_fixed} fixed-point regimes")

    results["c_sweep"] = {
        "n_c_values": len(c_sweep_data["c"]),
        "n_chaotic": n_chaotic,
        "n_fixed_point": n_fixed,
        "c_range": [float(c_sweep_data["c"][0]), float(c_sweep_data["c"][-1])],
    }

    # --- Part 3: Fine Lyapunov sweep ---
    logger.info("Part 3: Fine Lyapunov exponent sweep...")
    fine_data = generate_lyapunov_vs_c_data(n_c=30, n_steps=30000, dt=0.001)

    lam = fine_data["lyapunov_exponent"]
    c_fine = fine_data["c"]

    results["lyapunov_analysis"] = {
        "n_points": len(c_fine),
        "c_range": [float(c_fine[0]), float(c_fine[-1])],
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
    }

    # --- Part 4: Lyapunov at classic parameters ---
    logger.info("Part 4: Lyapunov exponent at classic Lu-Chen parameters...")
    config_classic = SimulationConfig(
        domain=_LU_CHEN_DOMAIN,
        dt=0.001,
        n_steps=50000,
        parameters={"a": 36.0, "b": 3.0, "c": 20.0},
    )
    sim_classic = LuChenSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.001)

    results["classic_parameters"] = {
        "a": 36.0,
        "b": 3.0,
        "c": 20.0,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic Lu-Chen Lyapunov: {lam_classic:.4f}")

    # --- Part 5: Fixed points ---
    sim_fp = LuChenSimulation(config_classic)
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

    # --- Part 6: Regime classification ---
    regimes = {}
    for c_val in [12.0, 20.0, 28.0]:
        config_regime = SimulationConfig(
            domain=_LU_CHEN_DOMAIN,
            dt=0.001,
            n_steps=10000,
            parameters={"a": 36.0, "b": 3.0, "c": c_val},
        )
        sim_regime = LuChenSimulation(config_regime)
        sim_regime.reset()
        regime = sim_regime.classify_regime()
        regimes[f"c={c_val}"] = regime
        logger.info(f"  c={c_val}: regime={regime}")

    results["regime_classification"] = regimes

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
        output_path / "c_sweep.npz",
        **{k: v for k, v in c_sweep_data.items() if isinstance(v, np.ndarray)},
    )
    np.savez(
        output_path / "lyapunov_fine.npz",
        c=fine_data["c"],
        lyapunov_exponent=fine_data["lyapunov_exponent"],
    )

    return results
