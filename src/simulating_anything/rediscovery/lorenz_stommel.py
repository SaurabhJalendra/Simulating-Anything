"""Lorenz-Stommel coupled atmosphere-ocean rediscovery.

This is a NOVEL discovery domain -- no known analytical results exist for
this coupled system. The goals are to:

1. Characterize how coupling strength affects the Lyapunov exponent
2. Measure atmosphere-ocean synchronization
3. Recover the coupled ODEs via SINDy
4. Find coupling-dependent scaling laws via PySR

Targets:
- Lyapunov exponent as a function of coupling_strength
- Cross-correlation between Lorenz x and Stommel T, S
- SINDy recovery of the 5D coupled ODE system
- PySR expressions for coupling-dependent Lyapunov or mean flow rate
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.lorenz_stommel import (
    LorenzStommelSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use CUSTOM domain since LORENZ_STOMMEL is not in the Domain enum
_LORENZ_STOMMEL_DOMAIN = Domain.CUSTOM


def _make_config(
    coupling_strength: float = 0.1,
    ocean_feedback: float = 0.01,
    dt: float = 0.005,
    n_steps: int = 10000,
    **extra_params: float,
) -> SimulationConfig:
    """Create a SimulationConfig for the coupled Lorenz-Stommel system.

    Args:
        coupling_strength: Atmosphere -> ocean coupling.
        ocean_feedback: Ocean -> atmosphere feedback.
        dt: Integration timestep.
        n_steps: Number of integration steps.
        **extra_params: Additional parameter overrides.

    Returns:
        Configured SimulationConfig.
    """
    params: dict[str, float] = {
        "sigma": 10.0,
        "rho": 28.0,
        "beta": 8.0 / 3.0,
        "eta1": 3.0,
        "eta2": 1.0,
        "delta": 0.3,
        "coupling_strength": coupling_strength,
        "ocean_feedback": ocean_feedback,
        "x_0": 1.0,
        "y_0": 1.0,
        "z_0": 1.0,
        "T_0": 2.0,
        "S_0": 1.0,
    }
    params.update(extra_params)
    return SimulationConfig(
        domain=_LORENZ_STOMMEL_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


def generate_ode_data(
    coupling_strength: float = 0.1,
    n_steps: int = 10000,
    dt: float = 0.005,
) -> dict[str, np.ndarray | float]:
    """Generate a coupled trajectory for SINDy ODE recovery.

    Args:
        coupling_strength: Atmosphere-ocean coupling.
        n_steps: Number of integration steps.
        dt: Timestep.

    Returns:
        Dict with states array, parameters, and metadata.
    """
    config = _make_config(
        coupling_strength=coupling_strength, dt=dt, n_steps=n_steps
    )
    sim = LorenzStommelSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "coupling_strength": coupling_strength,
    }


def generate_lyapunov_vs_coupling(
    coupling_values: np.ndarray | None = None,
    n_steps: int = 30000,
    n_transient: int = 10000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep coupling_strength and compute Lyapunov exponent at each value.

    Args:
        coupling_values: Array of coupling strengths to sweep.
        n_steps: Steps for Lyapunov estimation.
        n_transient: Transient steps to discard.
        dt: Integration timestep.

    Returns:
        Dict with coupling values and Lyapunov exponents.
    """
    if coupling_values is None:
        coupling_values = np.array([0.0, 0.01, 0.05, 0.1, 0.5])

    lyapunov_exps = []

    for i, c in enumerate(coupling_values):
        config = _make_config(coupling_strength=c, dt=dt, n_steps=n_steps)
        sim = LorenzStommelSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(n_transient):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        logger.info(
            f"  coupling={c:.3f}: Lyapunov={lam:.4f}"
        )

    return {
        "coupling": coupling_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def generate_synchronization_data(
    coupling_values: np.ndarray | None = None,
    n_steps: int = 10000,
    n_transient: int = 3000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Measure atmosphere-ocean cross-correlations at each coupling.

    For each coupling strength, computes the Pearson correlation between
    the Lorenz x variable and the Stommel T and S variables, as well as
    the mean overturning flow rate q = |T - S|.

    Args:
        coupling_values: Array of coupling strengths.
        n_steps: Steps to measure.
        n_transient: Transient steps to discard.
        dt: Integration timestep.

    Returns:
        Dict with coupling values and correlation/flow data arrays.
    """
    if coupling_values is None:
        coupling_values = np.array([0.0, 0.01, 0.05, 0.1, 0.5])

    corr_x_T = []
    corr_x_S = []
    mean_q = []
    std_q = []

    for c in coupling_values:
        config = _make_config(coupling_strength=c, dt=dt, n_steps=n_steps)
        sim = LorenzStommelSimulation(config)
        corr_data = sim.atmosphere_ocean_correlation(
            n_steps=n_steps, n_transient=n_transient,
        )
        corr_x_T.append(corr_data["corr_x_T"])
        corr_x_S.append(corr_data["corr_x_S"])
        mean_q.append(corr_data["mean_q"])
        std_q.append(corr_data["std_q"])

        logger.info(
            f"  coupling={c:.3f}: corr(x,T)={corr_data['corr_x_T']:.4f}, "
            f"corr(x,S)={corr_data['corr_x_S']:.4f}, "
            f"mean_q={corr_data['mean_q']:.4f}"
        )

    return {
        "coupling": coupling_values,
        "corr_x_T": np.array(corr_x_T),
        "corr_x_S": np.array(corr_x_S),
        "mean_q": np.array(mean_q),
        "std_q": np.array(std_q),
    }


def run_lorenz_stommel_rediscovery(
    output_dir: str | Path = "output/rediscovery/lorenz_stommel",
    n_iterations: int = 40,
) -> dict:
    """Run the full Lorenz-Stommel coupled system rediscovery.

    1. Parameter sweep over coupling_strength: Lyapunov exponents
    2. Atmosphere-ocean synchronization measurement
    3. SINDy ODE recovery on the 5D coupled system
    4. PySR symbolic regression on coupling-dependent quantities

    Args:
        output_dir: Directory to save results and data files.
        n_iterations: Number of PySR evolution iterations.

    Returns:
        Dict with all discovery results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    coupling_values = np.array([0.0, 0.01, 0.05, 0.1, 0.5])

    results: dict = {
        "domain": "lorenz_stommel",
        "novel": True,
        "description": (
            "Coupled Lorenz atmosphere + Stommel thermohaline ocean. "
            "No known analytical results -- this is a discovery domain."
        ),
        "targets": {
            "lyapunov_vs_coupling": (
                "How does coupling strength affect chaotic dynamics?"
            ),
            "synchronization": (
                "Does ocean-atmosphere coupling produce synchronization?"
            ),
            "ode_recovery": (
                "Can SINDy recover the 5D coupled ODE from data?"
            ),
            "scaling_laws": (
                "PySR expressions for coupling-dependent quantities"
            ),
        },
    }

    # --- Part 1: Lyapunov exponent vs coupling strength ---
    logger.info(
        "Part 1: Lyapunov exponent sweep over coupling_strength..."
    )
    lyap_data = generate_lyapunov_vs_coupling(
        coupling_values=coupling_values,
        n_steps=30000,
        n_transient=10000,
        dt=0.005,
    )

    results["lyapunov_sweep"] = {
        "coupling_values": lyap_data["coupling"].tolist(),
        "lyapunov_exponents": lyap_data["lyapunov_exponent"].tolist(),
        "max_lyapunov": float(np.max(lyap_data["lyapunov_exponent"])),
        "min_lyapunov": float(np.min(lyap_data["lyapunov_exponent"])),
        "lyapunov_at_zero_coupling": float(
            lyap_data["lyapunov_exponent"][0]
        ),
    }

    # Check if coupling changes Lyapunov significantly
    lam_zero = lyap_data["lyapunov_exponent"][0]
    lam_max_coupling = lyap_data["lyapunov_exponent"][-1]
    results["lyapunov_sweep"]["coupling_effect"] = (
        "stabilizing" if lam_max_coupling < lam_zero
        else "destabilizing" if lam_max_coupling > lam_zero
        else "neutral"
    )
    logger.info(
        f"  Lyapunov at c=0: {lam_zero:.4f}, at c=0.5: "
        f"{lam_max_coupling:.4f} -> "
        f"{results['lyapunov_sweep']['coupling_effect']}"
    )

    # --- Part 2: Atmosphere-ocean synchronization ---
    logger.info("Part 2: Atmosphere-ocean synchronization...")
    sync_data = generate_synchronization_data(
        coupling_values=coupling_values,
        n_steps=10000,
        n_transient=3000,
        dt=0.005,
    )

    results["synchronization"] = {
        "coupling_values": sync_data["coupling"].tolist(),
        "corr_x_T": sync_data["corr_x_T"].tolist(),
        "corr_x_S": sync_data["corr_x_S"].tolist(),
        "mean_flow_rate": sync_data["mean_q"].tolist(),
        "std_flow_rate": sync_data["std_q"].tolist(),
    }

    # Measure if correlation increases with coupling
    corr_trend = np.corrcoef(
        sync_data["coupling"], np.abs(sync_data["corr_x_T"])
    )[0, 1]
    results["synchronization"]["correlation_trend"] = float(corr_trend)
    results["synchronization"]["trend_direction"] = (
        "increasing" if corr_trend > 0.3
        else "decreasing" if corr_trend < -0.3
        else "no_clear_trend"
    )

    # --- Part 3: SINDy ODE recovery ---
    logger.info("Part 3: SINDy ODE recovery on coupled system...")
    ode_data = generate_ode_data(
        coupling_strength=0.1, n_steps=10000, dt=0.005
    )

    try:
        from simulating_anything.analysis.equation_discovery import (
            run_sindy,
        )

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["x", "y", "z", "T", "S"],
            threshold=0.1,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "coupling_strength": ode_data["coupling_strength"],
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in sindy_discoveries
            ],
        }

        # Compute overall R-squared
        if sindy_discoveries:
            r2_vals = [
                d.evidence.fit_r_squared for d in sindy_discoveries
            ]
            results["sindy_ode"]["mean_r_squared"] = float(
                np.mean(r2_vals)
            )
            results["sindy_ode"]["min_r_squared"] = float(
                np.min(r2_vals)
            )

        for d in sindy_discoveries:
            logger.info(
                f"  SINDy: {d.expression} "
                f"(R2={d.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 4: PySR symbolic regression ---
    logger.info("Part 4: PySR on coupling-dependent quantities...")

    # Build dataset: for each coupling, the measured Lyapunov and mean q
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Finer coupling sweep for PySR
        fine_couplings = np.linspace(0.0, 0.5, 15)
        fine_lyap = generate_lyapunov_vs_coupling(
            coupling_values=fine_couplings,
            n_steps=20000,
            n_transient=5000,
            dt=0.005,
        )

        X_coupling = fine_couplings.reshape(-1, 1)
        y_lyap = fine_lyap["lyapunov_exponent"]

        logger.info(
            f"  Running PySR on {len(fine_couplings)} "
            f"coupling-Lyapunov pairs..."
        )
        discoveries = run_symbolic_regression(
            X_coupling,
            y_lyap,
            variable_names=["c_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "exp", "log", "square"],
            max_complexity=12,
            populations=20,
            population_size=40,
        )
        results["pysr_lyapunov"] = {
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
            results["pysr_lyapunov"]["best"] = best.expression
            results["pysr_lyapunov"]["best_r2"] = (
                best.evidence.fit_r_squared
            )
            logger.info(
                f"  PySR Lyapunov best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )

        # Also fit mean flow rate vs coupling
        fine_sync = generate_synchronization_data(
            coupling_values=fine_couplings,
            n_steps=5000,
            n_transient=2000,
            dt=0.005,
        )
        y_q = fine_sync["mean_q"]
        discoveries_q = run_symbolic_regression(
            X_coupling,
            y_q,
            variable_names=["c_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "exp", "square"],
            max_complexity=10,
            populations=20,
            population_size=40,
        )
        results["pysr_flow_rate"] = {
            "n_discoveries": len(discoveries_q),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in discoveries_q[:5]
            ],
        }
        if discoveries_q:
            best_q = discoveries_q[0]
            results["pysr_flow_rate"]["best"] = best_q.expression
            results["pysr_flow_rate"]["best_r2"] = (
                best_q.evidence.fit_r_squared
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["pysr_lyapunov"] = {"error": str(e)}
        results["pysr_flow_rate"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data arrays
    np.savez(
        output_path / "lyapunov_sweep.npz",
        coupling=lyap_data["coupling"],
        lyapunov_exponent=lyap_data["lyapunov_exponent"],
    )
    np.savez(
        output_path / "synchronization.npz",
        coupling=sync_data["coupling"],
        corr_x_T=sync_data["corr_x_T"],
        corr_x_S=sync_data["corr_x_S"],
        mean_q=sync_data["mean_q"],
    )
    np.savez(
        output_path / "ode_data.npz",
        states=ode_data["states"],
    )

    return results
