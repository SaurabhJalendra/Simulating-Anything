"""Duffing-Van der Pol hybrid oscillator rediscovery.

Targets:
- ODE: x'' + mu*(x^2-1)*x' + alpha*x + beta*x^3 = F*cos(omega*t)  (via SINDy)
- Unforced limit cycle amplitude ~2 (Van der Pol limit for small beta)
- Period and amplitude dependence on mu, beta
- Bifurcation diagram under forcing amplitude F sweep
- Lyapunov exponent vs F (chaos detection)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.duffing_van_der_pol import (
    DuffingVanDerPolSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

DOMAIN_ENUM = Domain.DUFFING_VAN_DER_POL


def _make_sim(
    mu: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.2,
    F: float = 0.3,
    omega: float = 1.0,
    x_0: float = 0.1,
    y_0: float = 0.0,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> DuffingVanDerPolSimulation:
    """Helper to create a configured simulation instance."""
    config = SimulationConfig(
        domain=DOMAIN_ENUM,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "mu": mu, "alpha": alpha, "beta": beta,
            "F": F, "omega": omega, "x_0": x_0, "y_0": y_0,
        },
    )
    return DuffingVanDerPolSimulation(config)


def generate_ode_data(
    mu: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.2,
    n_steps: int = 10000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Uses the unforced system (F=0) so SINDy can cleanly identify
    the damping and restoring force coefficients.
    """
    sim = _make_sim(
        mu=mu, alpha=alpha, beta=beta, F=0.0,
        x_0=0.5, y_0=0.0, dt=dt, n_steps=n_steps,
    )
    sim.reset()

    states_xy = [sim.observe()[:2].copy()]
    for _ in range(n_steps):
        sim.step()
        states_xy.append(sim.observe()[:2].copy())

    states_xy = np.array(states_xy)
    times = np.arange(n_steps + 1) * dt

    return {
        "time": times,
        "states": states_xy,
        "x": states_xy[:, 0],
        "v": states_xy[:, 1],
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "dt": dt,
    }


def generate_limit_cycle_data(
    n_mu: int = 15,
    n_beta: int = 10,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep mu and beta to measure unforced limit cycle amplitude.

    For F=0, the system should exhibit a Van der Pol-like limit cycle
    with amplitude near 2 when beta is small. As beta increases, the
    hardening spring modifies the amplitude.
    """
    mu_values = np.logspace(-0.5, 1.0, n_mu)  # ~0.32 to 10
    beta_values = np.linspace(0.0, 1.0, n_beta)

    all_mu = []
    all_beta = []
    all_amplitude = []

    for mu in mu_values:
        for beta in beta_values:
            sim = _make_sim(
                mu=mu, alpha=1.0, beta=beta, F=0.0,
                x_0=0.1, y_0=0.0, dt=dt, n_steps=1000,
            )
            sim.reset()
            amp = sim.compute_limit_cycle_amplitude(n_steps=30000)
            all_mu.append(mu)
            all_beta.append(beta)
            all_amplitude.append(amp)

    return {
        "mu": np.array(all_mu),
        "beta": np.array(all_beta),
        "amplitude": np.array(all_amplitude),
    }


def generate_forcing_bifurcation_data(
    n_F: int = 30,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep forcing amplitude F and record Poincare section statistics.

    Returns the spread (std) of Poincare x-values at each F, which
    distinguishes periodic (low spread) from chaotic (high spread) behavior.
    """
    F_values = np.linspace(0.0, 2.0, n_F)
    spreads = []
    max_xs = []

    for i, F_val in enumerate(F_values):
        sim = _make_sim(
            mu=1.0, alpha=1.0, beta=0.2, F=F_val, omega=1.0,
            x_0=0.1, y_0=0.0, dt=dt, n_steps=1000,
        )
        sim.reset()
        poincare = sim.poincare_section(n_periods=150, n_transient=100)
        spread = float(np.std(poincare["x"])) if len(poincare["x"]) > 0 else 0.0
        max_x = float(np.max(np.abs(poincare["x"]))) if len(poincare["x"]) > 0 else 0.0
        spreads.append(spread)
        max_xs.append(max_x)

        if (i + 1) % 10 == 0:
            logger.info(f"  F={F_val:.3f}: Poincare spread={spread:.6f}")

    return {
        "F": F_values,
        "poincare_spread": np.array(spreads),
        "max_x": np.array(max_xs),
    }


def generate_lyapunov_data(
    n_F: int = 20,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Compute Lyapunov exponent vs forcing amplitude F."""
    F_values = np.linspace(0.0, 2.0, n_F)
    lyap_values = []

    for i, F_val in enumerate(F_values):
        sim = _make_sim(
            mu=1.0, alpha=1.0, beta=0.2, F=F_val, omega=1.0,
            x_0=0.1, y_0=0.0, dt=dt, n_steps=1000,
        )
        sim.reset()
        lyap = sim.compute_lyapunov(n_steps=30000, n_transient=5000)
        lyap_values.append(lyap)

        if (i + 1) % 5 == 0:
            logger.info(f"  F={F_val:.3f}: Lyapunov={lyap:.4f}")

    return {
        "F": F_values,
        "lyapunov": np.array(lyap_values),
    }


def generate_frequency_response_data(
    n_omega: int = 25,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep forcing frequency omega and measure steady-state amplitude."""
    omega_values = np.linspace(0.3, 3.0, n_omega)
    sim = _make_sim(
        mu=1.0, alpha=1.0, beta=0.2, F=0.3, omega=1.0,
        x_0=0.1, y_0=0.0, dt=dt, n_steps=1000,
    )
    sim.reset()
    result = sim.frequency_response(omega_values)
    return result


def run_duffing_van_der_pol_rediscovery(
    output_dir: str | Path = "output/rediscovery/duffing_van_der_pol",
    n_iterations: int = 40,
) -> dict:
    """Run Duffing-Van der Pol rediscovery pipeline.

    1. Generate trajectory for SINDy ODE recovery (unforced)
    2. Measure unforced limit cycle amplitude vs mu, beta
    3. Forcing bifurcation diagram (Poincare spread vs F)
    4. Lyapunov exponent vs F
    5. Frequency response curve
    6. PySR: amplitude = f(mu, beta) for unforced limit cycle

    Args:
        output_dir: Directory to save results.
        n_iterations: Number of PySR iterations.

    Returns:
        Results dict with all discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "duffing_van_der_pol",
        "targets": {
            "ode": "x'' + mu*(x^2-1)*x' + alpha*x + beta*x^3 = F*cos(omega*t)",
            "limit_cycle": "A ~ 2 for unforced (F=0), small beta",
            "bifurcation": "Period-doubling route to chaos as F increases",
            "lyapunov": "Positive Lyapunov exponent in chaotic regime",
        },
    }

    # --- Part 1: SINDy ODE recovery (unforced) ---
    logger.info("Part 1: SINDy ODE recovery (unforced, mu=1, alpha=1, beta=0.2)...")
    data = generate_ode_data(mu=1.0, alpha=1.0, beta=0.2, n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=data["dt"],
            feature_names=["x", "v"],
            threshold=0.05,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries[:5]
            ],
        }
        if sindy_discoveries:
            best = sindy_discoveries[0]
            results["sindy_ode"]["best"] = best.expression
            results["sindy_ode"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  SINDy best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Limit cycle amplitude vs mu, beta ---
    logger.info("Part 2: Limit cycle amplitude sweep...")
    lc_data = generate_limit_cycle_data(n_mu=10, n_beta=8, dt=0.005)

    # Report average amplitude for beta ~ 0 (VdP limit)
    small_beta = lc_data["beta"] < 0.15
    if np.any(small_beta):
        mean_amp_vdp = float(np.mean(lc_data["amplitude"][small_beta]))
    else:
        mean_amp_vdp = float(np.mean(lc_data["amplitude"]))

    results["limit_cycle"] = {
        "n_samples": len(lc_data["mu"]),
        "mean_amplitude_small_beta": mean_amp_vdp,
        "mean_amplitude_all": float(np.mean(lc_data["amplitude"])),
        "amplitude_range": [
            float(np.min(lc_data["amplitude"])),
            float(np.max(lc_data["amplitude"])),
        ],
    }
    logger.info(
        f"  Mean amplitude (small beta): {mean_amp_vdp:.4f} (theory ~2.0)"
    )

    # PySR: amplitude = f(mu, beta)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = np.column_stack([lc_data["mu"], lc_data["beta"]])
        y = lc_data["amplitude"]
        valid = np.isfinite(y) & (y > 0)

        if np.sum(valid) > 5:
            logger.info("  Running PySR: A = f(mu, beta)...")
            discoveries = run_symbolic_regression(
                X[valid], y[valid],
                variable_names=["mu_", "b_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square"],
                max_complexity=12,
                populations=20,
                population_size=40,
            )
            results["amplitude_pysr"] = {
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
                results["amplitude_pysr"]["best"] = best.expression
                results["amplitude_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR amplitude failed: {e}")
        results["amplitude_pysr"] = {"error": str(e)}

    # --- Part 3: Forcing bifurcation ---
    logger.info("Part 3: Forcing bifurcation diagram...")
    bif_data = generate_forcing_bifurcation_data(n_F=25, dt=0.005)

    # Detect chaos onset
    chaos_threshold = 0.05
    above = bif_data["poincare_spread"] > chaos_threshold
    if np.any(above):
        idx = int(np.argmax(above))
        F_c = float(bif_data["F"][idx])
    else:
        F_c = float("inf")

    results["bifurcation"] = {
        "n_F": len(bif_data["F"]),
        "F_range": [float(bif_data["F"].min()), float(bif_data["F"].max())],
        "max_spread": float(np.max(bif_data["poincare_spread"])),
        "F_c_estimate": F_c,
    }
    logger.info(f"  Chaos onset estimate: F ~ {F_c:.3f}")

    # --- Part 4: Lyapunov exponent sweep ---
    logger.info("Part 4: Lyapunov exponent vs F...")
    lyap_data = generate_lyapunov_data(n_F=15, dt=0.005)

    positive_lyap = lyap_data["lyapunov"] > 0
    results["lyapunov"] = {
        "n_F": len(lyap_data["F"]),
        "n_chaotic": int(np.sum(positive_lyap)),
        "max_lyapunov": float(np.max(lyap_data["lyapunov"])),
        "lyapunov_at_F0": float(lyap_data["lyapunov"][0]),
    }
    logger.info(
        f"  Max Lyapunov: {results['lyapunov']['max_lyapunov']:.4f}, "
        f"chaotic F values: {results['lyapunov']['n_chaotic']}/{len(lyap_data['F'])}"
    )

    # --- Part 5: Frequency response ---
    logger.info("Part 5: Frequency response curve...")
    freq_data = generate_frequency_response_data(n_omega=20, dt=0.005)

    valid = np.isfinite(freq_data["amplitude"]) & (freq_data["amplitude"] > 0)
    if np.any(valid):
        peak_idx = int(np.argmax(freq_data["amplitude"]))
        results["frequency_response"] = {
            "n_samples": int(np.sum(valid)),
            "resonant_omega": float(freq_data["omega"][peak_idx]),
            "max_amplitude": float(freq_data["amplitude"][peak_idx]),
            "omega_range": [
                float(freq_data["omega"].min()),
                float(freq_data["omega"].max()),
            ],
        }
        logger.info(
            f"  Resonant omega: {results['frequency_response']['resonant_omega']:.3f}, "
            f"max A: {results['frequency_response']['max_amplitude']:.4f}"
        )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save raw data
    np.savez(
        output_path / "limit_cycle_data.npz",
        mu=lc_data["mu"],
        beta=lc_data["beta"],
        amplitude=lc_data["amplitude"],
    )
    np.savez(
        output_path / "bifurcation_data.npz",
        F=bif_data["F"],
        poincare_spread=bif_data["poincare_spread"],
    )
    np.savez(
        output_path / "lyapunov_data.npz",
        F=lyap_data["F"],
        lyapunov=lyap_data["lyapunov"],
    )
    np.savez(
        output_path / "frequency_response.npz",
        omega=freq_data["omega"],
        amplitude=freq_data["amplitude"],
    )

    return results
