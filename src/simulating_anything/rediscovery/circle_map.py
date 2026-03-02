"""Circle map rediscovery.

Targets:
- Devil's staircase: w(Omega) at K=1 is a complete devil's staircase
- Arnold tongue boundaries in (Omega, K) space
- Lyapunov exponent: zero for K<=1 (mode-locked), positive for K>1 (chaos)
- Mode-locking detection: identify p/q plateaus and tongue widths
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.circle_map import CircleMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    omega: float = 0.5,
    K: float = 1.0,
    theta_0: float = 0.1,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.CIRCLE_MAP,  # Placeholder until CIRCLE_MAP enum added
        dt=1.0,
        n_steps=1000,
        parameters={"omega": omega, "K": K, "theta_0": theta_0},
    )


def generate_devils_staircase_data(
    n_omega: int = 500,
    K: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate devil's staircase data: w(Omega) at fixed K."""
    sim = CircleMapSimulation(_make_config(K=K))
    return sim.devils_staircase(n_omega=n_omega, K=K)


def generate_lyapunov_sweep(
    n_omega: int = 100,
    K: float = 1.5,
) -> dict[str, np.ndarray]:
    """Generate Lyapunov exponent vs Omega at fixed K."""
    omega_values = np.linspace(0.0, 1.0, n_omega)
    lyapunovs = np.zeros(n_omega, dtype=np.float64)

    sim = CircleMapSimulation(_make_config(K=K))

    for i, omega in enumerate(omega_values):
        lyapunovs[i] = sim.compute_lyapunov(
            n_transient=500,
            n_compute=5000,
            omega=omega,
            K=K,
        )

        if (i + 1) % 25 == 0:
            logger.info(f"  Omega={omega:.4f}: Lyapunov={lyapunovs[i]:.6f}")

    return {
        "omega": omega_values,
        "lyapunov": lyapunovs,
    }


def generate_lyapunov_K_sweep(
    n_K: int = 50,
    omega: float = 0.5,
) -> dict[str, np.ndarray]:
    """Generate Lyapunov exponent vs K at fixed Omega."""
    K_values = np.linspace(0.0, 3.0, n_K)
    lyapunovs = np.zeros(n_K, dtype=np.float64)

    sim = CircleMapSimulation(_make_config(omega=omega))

    for i, K in enumerate(K_values):
        lyapunovs[i] = sim.compute_lyapunov(
            n_transient=500,
            n_compute=5000,
            omega=omega,
            K=K,
        )

    return {
        "K": K_values,
        "lyapunov": lyapunovs,
    }


def generate_arnold_tongue_data(
    n_omega: int = 100,
    n_K: int = 50,
) -> dict[str, np.ndarray]:
    """Generate Arnold tongue winding number grid."""
    sim = CircleMapSimulation(_make_config())
    return sim.arnold_tongue_data(
        omega_range=(0.0, 1.0),
        K_range=(0.0, 2.0),
        n_omega=n_omega,
        n_K=n_K,
    )


def run_circle_map_rediscovery(
    output_dir: str | Path = "output/rediscovery/circle_map",
    n_iterations: int = 40,
) -> dict:
    """Run circle map rediscovery analysis.

    Generates:
    1. Devil's staircase w(Omega) at K=1
    2. Arnold tongue winding number grid
    3. Lyapunov exponent sweeps
    4. Mode-locking plateau detection
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "circle_map",
        "targets": {
            "devils_staircase": "w(Omega) at K=1 is complete devil's staircase",
            "mode_locking": "Rational w = p/q plateaus (Arnold tongues)",
            "lyapunov_K0": "lambda = 0 for K=0 (pure rotation)",
            "lyapunov_K_gt_1": "lambda > 0 for K > 1 (chaos regions)",
        },
    }

    # Part 1: Devil's staircase at K=1
    logger.info("Part 1: Devil's staircase at K=1...")
    staircase = generate_devils_staircase_data(n_omega=500, K=1.0)

    # Check monotonicity (devil's staircase should be non-decreasing)
    diffs = np.diff(staircase["winding_number"])
    n_monotone = int(np.sum(diffs >= -1e-6))
    results["devils_staircase"] = {
        "n_omega": 500,
        "K": 1.0,
        "w_min": float(np.min(staircase["winding_number"])),
        "w_max": float(np.max(staircase["winding_number"])),
        "monotone_fraction": float(n_monotone / len(diffs)),
    }
    logger.info(
        f"  w range: [{results['devils_staircase']['w_min']:.4f}, "
        f"{results['devils_staircase']['w_max']:.4f}]"
    )

    np.savez(
        output_path / "staircase_data.npz",
        omega=staircase["omega"],
        winding_number=staircase["winding_number"],
    )

    # Part 2: Mode-locking detection
    logger.info("Part 2: Mode-locking detection...")
    sim = CircleMapSimulation(_make_config())
    tongues = sim.detect_mode_locking(
        staircase["omega"],
        staircase["winding_number"],
        tolerance=5e-4,
        max_q=8,
    )
    results["mode_locking"] = {
        "n_tongues": len(tongues),
        "tongues": tongues[:15],  # Top 15 tongues
    }
    if tongues:
        logger.info(f"  Detected {len(tongues)} mode-locked tongues")
        for t in tongues[:5]:
            logger.info(
                f"    w={t['p']}/{t['q']}={t['winding_number']:.4f}, "
                f"Omega_center={t['omega_center']:.4f}, "
                f"width={t['omega_width']:.4f}"
            )

    # Part 3: Lyapunov sweep at K=1.5 (above critical)
    logger.info("Part 3: Lyapunov exponent vs Omega at K=1.5...")
    lyap_data = generate_lyapunov_sweep(n_omega=80, K=1.5)

    chaotic = lyap_data["lyapunov"] > 0
    n_chaotic = int(np.sum(chaotic))
    results["lyapunov_omega_sweep"] = {
        "K": 1.5,
        "n_omega": 80,
        "n_chaotic": n_chaotic,
        "chaotic_fraction": float(n_chaotic / len(lyap_data["lyapunov"])),
        "max_lyapunov": float(np.max(lyap_data["lyapunov"])),
        "mean_lyapunov": float(np.mean(lyap_data["lyapunov"])),
    }
    logger.info(
        f"  Chaotic fraction at K=1.5: {results['lyapunov_omega_sweep']['chaotic_fraction']:.3f}"
    )

    np.savez(
        output_path / "lyapunov_omega_data.npz",
        omega=lyap_data["omega"],
        lyapunov=lyap_data["lyapunov"],
    )

    # Part 4: Lyapunov vs K sweep
    logger.info("Part 4: Lyapunov exponent vs K at Omega=0.5...")
    lyap_K_data = generate_lyapunov_K_sweep(n_K=40, omega=0.5)

    # Verify lambda ~ 0 for K=0
    K0_idx = np.argmin(np.abs(lyap_K_data["K"]))
    results["lyapunov_K_sweep"] = {
        "omega": 0.5,
        "n_K": 40,
        "lyapunov_at_K0": float(lyap_K_data["lyapunov"][K0_idx]),
        "max_lyapunov": float(np.max(lyap_K_data["lyapunov"])),
    }

    np.savez(
        output_path / "lyapunov_K_data.npz",
        K=lyap_K_data["K"],
        lyapunov=lyap_K_data["lyapunov"],
    )

    # Part 5: Arnold tongue grid (smaller resolution for speed)
    logger.info("Part 5: Arnold tongue winding number grid...")
    tongue_data = generate_arnold_tongue_data(n_omega=60, n_K=30)
    results["arnold_tongues"] = {
        "n_omega": 60,
        "n_K": 30,
        "omega_range": [0.0, 1.0],
        "K_range": [0.0, 2.0],
    }

    np.savez(
        output_path / "arnold_tongue_data.npz",
        omega_values=tongue_data["omega_values"],
        K_values=tongue_data["K_values"],
        winding_numbers=tongue_data["winding_numbers"],
    )

    # PySR: fit winding number vs Omega at K=0 (should recover w=Omega)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Generate clean data at K=0 for symbolic regression
        sim_k0 = CircleMapSimulation(_make_config(K=0.0))
        staircase_k0 = sim_k0.devils_staircase(n_omega=50, K=0.0)
        X = staircase_k0["omega"].reshape(-1, 1)
        y = staircase_k0["winding_number"]

        logger.info("  Running PySR: w = f(Omega) at K=0...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["Om"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos"],
            max_complexity=8,
            populations=15,
            population_size=30,
        )
        results["pysr_k0"] = {
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
            results["pysr_k0"]["best"] = best.expression
            results["pysr_k0"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["pysr_k0"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
