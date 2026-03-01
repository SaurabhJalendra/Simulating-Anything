"""FitzHugh-Nagumo spatial PDE rediscovery.

Targets:
- Traveling pulse wave speed c ~ f(D_v, eps)
- Pulse shape and width characterization
- For D_v=0: reduces to local FHN dynamics at each grid point
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.fhn_spatial import FHNSpatial
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    dt: float = 0.05,
    n_steps: int = 2000,
    N: int = 128,
    L: float = 50.0,
    **params: float,
) -> SimulationConfig:
    """Build a SimulationConfig for the FHN spatial domain."""
    defaults = {
        "a": 0.7,
        "b": 0.8,
        "eps": 0.08,
        "D_v": 1.0,
        "N": float(N),
        "L": L,
    }
    defaults.update(params)
    return SimulationConfig(
        domain=Domain.FHN_SPATIAL,
        dt=dt,
        n_steps=n_steps,
        parameters=defaults,
    )


def generate_wave_speed_data(
    n_D: int = 15,
    n_steps: int = 3000,
    dt: float = 0.05,
    N: int = 256,
    L: float = 100.0,
) -> dict[str, np.ndarray]:
    """Measure traveling pulse speed vs voltage diffusion coefficient D_v.

    Strategy: initialize with a localized pulse, let it propagate, and
    track the peak position at two well-separated times to compute speed.

    Returns:
        Dictionary with D_v values and measured wave speeds.
    """
    D_v_values = np.logspace(-1, 1, n_D)  # 0.1 to 10.0
    all_D = []
    all_speed = []

    for i, D_v in enumerate(D_v_values):
        config = _make_config(
            dt=dt, n_steps=n_steps, N=N, L=L, D_v=D_v,
        )
        try:
            sim = FHNSpatial(config)
        except ValueError:
            # CFL violation for this combination
            logger.warning(f"  D_v={D_v:.4f}: skipped (CFL violation)")
            continue

        sim.reset(seed=42)

        speed = sim.measure_wave_speed(n_steps=n_steps)

        all_D.append(D_v)
        all_speed.append(speed)

        if (i + 1) % 5 == 0:
            logger.info(f"  D_v={D_v:.4f}: wave_speed={speed:.4f}")

    return {
        "D_v": np.array(all_D),
        "wave_speed": np.array(all_speed),
    }


def generate_wave_speed_vs_eps_data(
    n_eps: int = 12,
    n_steps: int = 3000,
    dt: float = 0.05,
    N: int = 256,
    L: float = 100.0,
    D_v: float = 1.0,
) -> dict[str, np.ndarray]:
    """Measure traveling pulse speed vs timescale separation eps.

    Returns:
        Dictionary with eps values and measured wave speeds.
    """
    eps_values = np.logspace(-2, -0.3, n_eps)  # 0.01 to ~0.5
    all_eps = []
    all_speed = []

    for i, eps in enumerate(eps_values):
        config = _make_config(
            dt=dt, n_steps=n_steps, N=N, L=L, D_v=D_v, eps=eps,
        )
        sim = FHNSpatial(config)
        sim.reset(seed=42)

        speed = sim.measure_wave_speed(n_steps=n_steps)

        all_eps.append(eps)
        all_speed.append(speed)

        if (i + 1) % 4 == 0:
            logger.info(f"  eps={eps:.4f}: wave_speed={speed:.4f}")

    return {
        "eps": np.array(all_eps),
        "wave_speed": np.array(all_speed),
    }


def generate_pulse_data(
    D_v: float = 1.0,
    n_steps: int = 2000,
    dt: float = 0.05,
    N: int = 256,
    L: float = 100.0,
) -> dict[str, np.ndarray]:
    """Generate pulse shape data at steady propagation.

    Returns:
        Dictionary with spatial profile of v and w after the pulse
        has reached steady shape.
    """
    config = _make_config(dt=dt, n_steps=n_steps, N=N, L=L, D_v=D_v)
    sim = FHNSpatial(config)
    sim.reset(seed=42)

    # Let pulse develop and propagate to steady shape
    for _ in range(n_steps):
        sim.step()

    # Measure pulse width (FWHM of v above rest)
    v = sim.v_field
    v_rest = sim._find_rest_v()
    v_max = np.max(v)
    half_max = (v_rest + v_max) / 2.0
    above = v > half_max
    fwhm = float(np.sum(above)) * sim.dx

    return {
        "x": sim.x.copy(),
        "v": v,
        "w": sim.w_field,
        "v_rest": v_rest,
        "v_max": float(v_max),
        "pulse_fwhm": fwhm,
        "pulse_count": sim.pulse_count,
    }


def run_fhn_spatial_rediscovery(
    output_dir: str | Path = "output/rediscovery/fhn_spatial",
    n_iterations: int = 40,
) -> dict:
    """Run FHN spatial PDE rediscovery pipeline.

    1. Measure wave speed vs D_v
    2. Measure wave speed vs eps
    3. Characterize pulse shape
    4. Run PySR on wave_speed ~ f(D_v)

    Returns:
        Results dictionary with all measurements and PySR discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "fhn_spatial",
        "targets": {
            "wave_speed_vs_D": "c ~ sqrt(D_v) (diffusion scaling)",
            "wave_speed_vs_eps": "c decreases with eps (slower recovery = faster pulse)",
            "pulse_shape": "traveling pulse profile",
        },
    }

    # --- Part 1: Wave speed vs D_v ---
    logger.info("Part 1: Measuring wave speed vs D_v...")
    wave_data = generate_wave_speed_data(
        n_D=15, n_steps=3000, dt=0.05, N=256, L=100.0,
    )

    valid = wave_data["wave_speed"] > 0.01
    n_valid = int(np.sum(valid))
    results["wave_speed_vs_D"] = {
        "n_samples": n_valid,
        "D_v_range": [
            float(wave_data["D_v"].min()),
            float(wave_data["D_v"].max()),
        ],
        "speed_range": [
            float(wave_data["wave_speed"][valid].min()) if n_valid > 0 else 0.0,
            float(wave_data["wave_speed"][valid].max()) if n_valid > 0 else 0.0,
        ],
    }

    # PySR: wave_speed = f(D_v)
    if n_valid >= 5:
        try:
            from simulating_anything.analysis.symbolic_regression import (
                run_symbolic_regression,
            )

            X = wave_data["D_v"][valid].reshape(-1, 1)
            y = wave_data["wave_speed"][valid]

            logger.info("Running PySR: wave_speed = f(D_v)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["D_v"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square"],
                max_complexity=10,
                populations=15,
                population_size=30,
            )
            results["wave_speed_pysr"] = {
                "n_discoveries": len(discoveries),
                "discoveries": [
                    {
                        "expression": d.expression,
                        "r_squared": d.evidence.fit_r_squared,
                    }
                    for d in discoveries[:5]
                ],
            }
            if discoveries:
                best = discoveries[0]
                results["wave_speed_pysr"]["best"] = best.expression
                results["wave_speed_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
        except Exception as e:
            logger.warning(f"PySR failed: {e}")
            results["wave_speed_pysr"] = {"error": str(e)}
    else:
        logger.warning(f"Only {n_valid} valid wave speed measurements -- skipping PySR")
        results["wave_speed_pysr"] = {"error": "insufficient valid data"}

    # --- Part 2: Wave speed vs eps ---
    logger.info("Part 2: Measuring wave speed vs eps...")
    eps_data = generate_wave_speed_vs_eps_data(
        n_eps=12, n_steps=3000, dt=0.05, N=256, L=100.0,
    )

    valid_eps = eps_data["wave_speed"] > 0.01
    n_valid_eps = int(np.sum(valid_eps))
    results["wave_speed_vs_eps"] = {
        "n_samples": n_valid_eps,
        "eps_range": [
            float(eps_data["eps"].min()),
            float(eps_data["eps"].max()),
        ],
        "speed_range": [
            float(eps_data["wave_speed"][valid_eps].min()) if n_valid_eps > 0 else 0.0,
            float(eps_data["wave_speed"][valid_eps].max()) if n_valid_eps > 0 else 0.0,
        ],
    }

    # --- Part 3: Pulse shape ---
    logger.info("Part 3: Characterizing pulse shape...")
    pulse_data = generate_pulse_data(D_v=1.0, n_steps=2000, dt=0.05, N=256, L=100.0)
    results["pulse_shape"] = {
        "v_rest": pulse_data["v_rest"],
        "v_max": pulse_data["v_max"],
        "pulse_fwhm": pulse_data["pulse_fwhm"],
        "pulse_count": pulse_data["pulse_count"],
    }
    logger.info(f"  v_rest={pulse_data['v_rest']:.4f}, v_max={pulse_data['v_max']:.4f}")
    logger.info(f"  Pulse FWHM={pulse_data['pulse_fwhm']:.4f}")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
