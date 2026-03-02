"""2D diffusion equation rediscovery.

Targets:
- Fourier mode decay: u_hat(kx,ky,t) = u_hat(kx,ky,0) * exp(-D*(kx^2+ky^2)*t)
- Decay rate = D * (kx^2 + ky^2) for mode (kx, ky)
- PySR: decay_rate = f(D, kx_sq_plus_ky_sq)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.diffusion_2d import Diffusion2DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_decay_data(
    n_D: int = 25,
    n_steps: int = 200,
    dt: float = 0.01,
    N_grid: int = 64,
) -> dict[str, np.ndarray]:
    """Generate mode decay rate vs diffusion coefficient data.

    Measures the decay rate of the (1, 1) Fourier mode for different D values.
    Theory: decay_rate = D * (kx^2 + ky^2) where kx = ky = 2*pi/L = 1.
    So decay_rate = D * 2 for mode (1, 1) on [0, 2*pi]^2.
    """
    D_values = np.logspace(-2, 0, n_D)  # 0.01 to 1.0
    L = 2 * np.pi
    kx_mode, ky_mode = 1, 1
    k_sq = (2 * np.pi * kx_mode / L) ** 2 + (2 * np.pi * ky_mode / L) ** 2

    all_D = []
    all_decay = []

    for i, D in enumerate(D_values):
        config = SimulationConfig(
            domain=Domain.DIFFUSION_2D,  # placeholder
            dt=dt,
            n_steps=n_steps,
            parameters={"D": D, "N_grid": float(N_grid), "L": L},
        )
        sim = Diffusion2DSimulation(config)
        sim.init_type = "single_mode"
        sim.reset()

        # Measure amplitude of mode (1, 1) at t=0
        a0 = sim.mode_amplitude(kx_mode, ky_mode)

        # Run and measure at end
        for _ in range(n_steps):
            sim.step()
        af = sim.mode_amplitude(kx_mode, ky_mode)

        t_total = n_steps * dt
        if af > 1e-15 and a0 > 1e-15:
            decay_rate = -np.log(af / a0) / t_total
        else:
            decay_rate = np.inf

        all_D.append(D)
        all_decay.append(decay_rate)

        if (i + 1) % 10 == 0:
            theory = D * k_sq
            logger.info(f"  D={D:.4f}: decay={decay_rate:.4f}, theory={theory:.4f}")

    return {
        "D": np.array(all_D),
        "decay_rate": np.array(all_decay),
        "k_sq": k_sq,
        "L": L,
    }


def generate_multimode_decay_data(
    D: float = 0.1,
    n_steps: int = 200,
    dt: float = 0.01,
    N_grid: int = 64,
) -> dict[str, np.ndarray]:
    """Generate decay rate data for multiple Fourier modes at fixed D.

    Measures the decay rate of modes (1,0), (0,1), (1,1), (2,0), (0,2), (2,1), (1,2),
    (2,2) to verify lambda = D * (kx^2 + ky^2).
    """
    L = 2 * np.pi
    modes = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2), (3, 0), (0, 3)]

    all_k_sq = []
    all_decay = []

    for mx, my in modes:
        config = SimulationConfig(
            domain=Domain.DIFFUSION_2D,  # placeholder
            dt=dt,
            n_steps=n_steps,
            parameters={"D": D, "N_grid": float(N_grid), "L": L},
        )
        sim = Diffusion2DSimulation(config)
        # Initialize with a specific mode by setting init_type then manually setting _u
        sim.reset()

        # Set initial condition to pure mode (mx, my)
        kx_val = 2 * np.pi * mx / L
        ky_val = 2 * np.pi * my / L
        sim._u = np.sin(kx_val * sim.X + ky_val * sim.Y)
        sim._state = sim._u.ravel()

        a0 = sim.mode_amplitude(mx, my)

        for _ in range(n_steps):
            sim.step()
        af = sim.mode_amplitude(mx, my)

        t_total = n_steps * dt
        if af > 1e-15 and a0 > 1e-15:
            decay_rate = -np.log(af / a0) / t_total
        else:
            decay_rate = np.inf

        k_sq = kx_val**2 + ky_val**2
        all_k_sq.append(k_sq)
        all_decay.append(decay_rate)

        logger.info(
            f"  mode ({mx},{my}): k_sq={k_sq:.2f}, "
            f"decay={decay_rate:.4f}, theory={D * k_sq:.4f}"
        )

    return {
        "k_sq": np.array(all_k_sq),
        "decay_rate": np.array(all_decay),
        "D": D,
        "modes": modes,
        "L": L,
    }


def run_diffusion_2d_rediscovery(
    output_dir: str | Path = "output/rediscovery/diffusion_2d",
    n_iterations: int = 40,
) -> dict:
    """Run 2D diffusion equation rediscovery.

    1. Sweep D to find decay rate of mode (1,1): lambda = D * 2
    2. Sweep modes at fixed D to find lambda = D * (kx^2 + ky^2)
    3. Run PySR on both datasets
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "diffusion_2d",
        "targets": {
            "decay_rate_D_sweep": "lambda = D * (kx^2 + ky^2) = 2*D for mode (1,1)",
            "decay_rate_mode_sweep": "lambda = D * (kx^2 + ky^2)",
            "energy_decay": "E(t) decays monotonically",
        },
    }

    # --- Part 1: D sweep for mode (1,1) ---
    logger.info("Part 1: Generating decay rate vs D data (mode 1,1)...")
    data = generate_decay_data(n_D=25, n_steps=200, dt=0.01, N_grid=64)

    k_sq = data["k_sq"]  # should be 2.0 for mode (1,1) on [0,2pi]^2
    theory_rate = data["D"] * k_sq
    valid = np.isfinite(data["decay_rate"]) & (data["decay_rate"] > 0)

    if np.sum(valid) > 5:
        rel_err = (
            np.abs(data["decay_rate"][valid] - theory_rate[valid]) / theory_rate[valid]
        )
        results["decay_rate_D_sweep"] = {
            "n_samples": int(np.sum(valid)),
            "mean_relative_error": float(np.mean(rel_err)),
            "correlation": float(
                np.corrcoef(data["decay_rate"][valid], theory_rate[valid])[0, 1]
            ),
            "k_sq": float(k_sq),
        }
        logger.info(f"  Mean relative error: {np.mean(rel_err):.4%}")

    # PySR: decay_rate = f(D)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = data["D"][valid].reshape(-1, 1)
        y = data["decay_rate"][valid]

        logger.info("Running PySR: decay_rate = f(D) for mode (1,1)...")
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
        results["D_sweep_pysr"] = {
            "n_discoveries": len(discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in discoveries[:5]
            ],
        }
        if discoveries:
            best = discoveries[0]
            results["D_sweep_pysr"]["best"] = best.expression
            results["D_sweep_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed for D sweep: {e}")
        results["D_sweep_pysr"] = {"error": str(e)}

    # --- Part 2: Multi-mode sweep at fixed D ---
    logger.info("Part 2: Generating multi-mode decay data...")
    mode_data = generate_multimode_decay_data(D=0.1, n_steps=200, dt=0.01, N_grid=64)

    valid_m = np.isfinite(mode_data["decay_rate"]) & (mode_data["decay_rate"] > 0)
    if np.sum(valid_m) > 3:
        theory_m = mode_data["D"] * mode_data["k_sq"]
        rel_err_m = (
            np.abs(mode_data["decay_rate"][valid_m] - theory_m[valid_m])
            / theory_m[valid_m]
        )
        results["multimode_data"] = {
            "n_modes": int(np.sum(valid_m)),
            "mean_relative_error": float(np.mean(rel_err_m)),
            "correlation": float(
                np.corrcoef(
                    mode_data["decay_rate"][valid_m], theory_m[valid_m]
                )[0, 1]
            ),
        }
        logger.info(f"  Multi-mode mean relative error: {np.mean(rel_err_m):.4%}")

    # PySR: decay_rate = f(k_sq) at fixed D
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X_m = mode_data["k_sq"][valid_m].reshape(-1, 1)
        y_m = mode_data["decay_rate"][valid_m]

        logger.info("Running PySR: decay_rate = f(k_sq)...")
        disc_m = run_symbolic_regression(
            X_m, y_m,
            variable_names=["k_sq"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=[],
            max_complexity=8,
            populations=15,
            population_size=30,
        )
        results["mode_sweep_pysr"] = {
            "n_discoveries": len(disc_m),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in disc_m[:5]
            ],
        }
        if disc_m:
            best = disc_m[0]
            results["mode_sweep_pysr"]["best"] = best.expression
            results["mode_sweep_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed for mode sweep: {e}")
        results["mode_sweep_pysr"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
