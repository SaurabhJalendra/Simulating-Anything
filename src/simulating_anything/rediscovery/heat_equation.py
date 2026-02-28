"""Heat equation 1D rediscovery.

Targets:
- Fourier mode decay: a_k(t) = a_k(0) * exp(-D*k^2*t)
- Decay rate = D*k^2 (PySR: decay_rate = f(D))
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.heat_equation import HeatEquation1DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_decay_data(
    n_D: int = 25,
    n_steps: int = 200,
    dt: float = 0.01,
    N: int = 128,
) -> dict[str, np.ndarray]:
    """Generate mode decay rate vs diffusion coefficient data."""
    D_values = np.logspace(-2, 0, n_D)  # 0.01 to 1.0
    L = 2 * np.pi
    mode = 1  # First Fourier mode
    k_mode = 2 * np.pi * mode / L  # = 1.0

    all_D = []
    all_decay = []

    for i, D in enumerate(D_values):
        config = SimulationConfig(
            domain=Domain.HEAT_EQUATION_1D,
            dt=dt,
            n_steps=n_steps,
            parameters={"D": D, "N": float(N), "L": L},
        )
        sim = HeatEquation1DSimulation(config)
        sim.init_type = "sine"
        sim.reset()

        # Measure amplitude of mode 1 at t=0
        u_hat_0 = np.fft.fft(sim.observe())
        a0 = np.abs(u_hat_0[mode])

        # Run and measure at end
        for _ in range(n_steps):
            sim.step()
        u_hat_f = np.fft.fft(sim.observe())
        af = np.abs(u_hat_f[mode])

        t_total = n_steps * dt
        if af > 1e-15 and a0 > 1e-15:
            decay_rate = -np.log(af / a0) / t_total
        else:
            decay_rate = np.inf

        all_D.append(D)
        all_decay.append(decay_rate)

        if (i + 1) % 10 == 0:
            theory = D * k_mode**2
            logger.info(f"  D={D:.4f}: decay={decay_rate:.4f}, theory={theory:.4f}")

    return {
        "D": np.array(all_D),
        "decay_rate": np.array(all_decay),
        "k_mode": k_mode,
        "L": L,
    }


def run_heat_equation_rediscovery(
    output_dir: str | Path = "output/rediscovery/heat_equation",
    n_iterations: int = 40,
) -> dict:
    """Run heat equation rediscovery."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "heat_equation_1d",
        "targets": {
            "decay_rate": "lambda_k = D * k^2",
            "gaussian_spreading": "sigma(t) = sqrt(2*D*t)",
        },
    }

    logger.info("Generating decay rate data...")
    data = generate_decay_data(n_D=25, n_steps=200, dt=0.01)

    k = data["k_mode"]
    theory_rate = data["D"] * k**2
    valid = np.isfinite(data["decay_rate"]) & (data["decay_rate"] > 0)

    if np.sum(valid) > 5:
        rel_err = np.abs(data["decay_rate"][valid] - theory_rate[valid]) / theory_rate[valid]
        results["decay_rate_data"] = {
            "n_samples": int(np.sum(valid)),
            "mean_relative_error": float(np.mean(rel_err)),
            "correlation": float(
                np.corrcoef(data["decay_rate"][valid], theory_rate[valid])[0, 1]
            ),
        }
        logger.info(f"  Mean relative error: {np.mean(rel_err):.4%}")

    # PySR
    try:
        from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

        X = data["D"][valid].reshape(-1, 1)
        y = data["decay_rate"][valid]

        logger.info("Running PySR: decay_rate = f(D)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["D"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square"],
            max_complexity=8,
            populations=15,
            population_size=30,
        )
        results["decay_rate_pysr"] = {
            "n_discoveries": len(discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in discoveries[:5]
            ],
        }
        if discoveries:
            best = discoveries[0]
            results["decay_rate_pysr"]["best"] = best.expression
            results["decay_rate_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})")
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["decay_rate_pysr"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
