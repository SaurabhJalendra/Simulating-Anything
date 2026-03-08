"""Rediscovery for coupled neural-cardiac oscillator system.

Novel discovery domain: no known analytical solution for the coupled system.
Targets:
1. SINDy ODE recovery of the 4D coupled system
2. Phase locking ratio vs coupling strength
3. Critical coupling for frequency entrainment
4. PySR scaling laws for synchronization
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def generate_coupling_sweep_data(
    n_points: int = 20,
    n_steps: int = 8000,
    dt: float = 0.05,
) -> dict:
    """Sweep coupling strengths and measure synchronization.

    Args:
        n_points: Number of coupling values to sweep.
        n_steps: Steps per simulation.
        dt: Integration timestep.

    Returns:
        Dict with coupling values and sync metrics.
    """
    from simulating_anything.simulation.neural_cardiac import (
        NeuralCardiacSimulation,
    )
    from simulating_anything.types.simulation import Domain, SimulationConfig

    couplings = np.linspace(0.0, 0.5, n_points)
    phase_ratios = []
    correlations = []
    neural_freqs = []
    cardiac_freqs = []

    for c in couplings:
        config = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "coupling_nc": float(c),
                "coupling_cv": float(c * 0.5),
                "I_ext": 0.5,
                "mu_c": 1.0,
            },
        )
        sim = NeuralCardiacSimulation(config)
        sim.reset(seed=42)

        trajectory = []
        for _ in range(n_steps):
            trajectory.append(sim.step().copy())
        traj = np.array(trajectory)

        # Skip transient
        traj = traj[n_steps // 4:]

        v = traj[:, 0]
        x = traj[:, 2]

        # Frequency from zero crossings
        v_cross = np.sum(np.diff(np.sign(v)) > 0)
        x_cross = np.sum(np.diff(np.sign(x)) > 0)
        duration = len(traj) * dt

        v_freq = v_cross / duration if duration > 0 else 0
        x_freq = x_cross / duration if duration > 0 else 0
        neural_freqs.append(float(v_freq))
        cardiac_freqs.append(float(x_freq))

        if x_cross > 0:
            phase_ratios.append(float(v_cross / x_cross))
        else:
            phase_ratios.append(float("nan"))

        # Correlation between v and x
        if np.std(v) > 1e-10 and np.std(x) > 1e-10:
            correlations.append(float(np.corrcoef(v, x)[0, 1]))
        else:
            correlations.append(0.0)

    return {
        "couplings": couplings.tolist(),
        "phase_ratios": phase_ratios,
        "correlations": correlations,
        "neural_freqs": neural_freqs,
        "cardiac_freqs": cardiac_freqs,
    }


def generate_ode_data(
    coupling_nc: float = 0.1,
    coupling_cv: float = 0.05,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Args:
        coupling_nc: Neural -> cardiac coupling.
        coupling_cv: Cardiac -> neural coupling.
        n_steps: Number of timesteps.
        dt: Integration timestep.

    Returns:
        Tuple of (states, derivatives) arrays.
    """
    from simulating_anything.simulation.neural_cardiac import (
        NeuralCardiacSimulation,
    )
    from simulating_anything.types.simulation import Domain, SimulationConfig

    config = SimulationConfig(
        domain=Domain.CUSTOM,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "coupling_nc": coupling_nc,
            "coupling_cv": coupling_cv,
            "I_ext": 0.5,
            "mu_c": 1.0,
        },
    )
    sim = NeuralCardiacSimulation(config)
    sim.reset(seed=0)

    states = [sim.observe().copy()]
    for _ in range(n_steps - 1):
        states.append(sim.step().copy())

    X = np.array(states[:-1], dtype=np.float64)
    X_next = np.array(states[1:], dtype=np.float64)
    dXdt = (X_next - X) / dt

    return X, dXdt


def run_neural_cardiac_rediscovery(
    n_iterations: int = 40,
) -> dict:
    """Run full rediscovery analysis for coupled neural-cardiac system.

    Args:
        n_iterations: PySR iterations.

    Returns:
        Dict with all discovery results.
    """
    results = {
        "domain": "neural_cardiac",
        "novel": True,
        "description": (
            "Coupled FHN neural oscillator + VdP cardiac pacemaker "
            "with bidirectional autonomic coupling. No known analytical "
            "results for the coupled system."
        ),
        "targets": {
            "synchronization": "Phase locking and frequency entrainment",
            "critical_coupling": "Threshold for neural-cardiac sync",
            "ode_recovery": "SINDy recovery of 4D coupled ODE",
            "scaling_laws": "PySR scaling for coupling-dependent quantities",
        },
    }

    # 1. Coupling sweep
    logger.info("Running coupling sweep for neural-cardiac system...")
    sweep_data = generate_coupling_sweep_data()
    results["coupling_sweep"] = {
        "n_points": len(sweep_data["couplings"]),
        "coupling_range": [
            float(min(sweep_data["couplings"])),
            float(max(sweep_data["couplings"])),
        ],
        "phase_ratios": sweep_data["phase_ratios"],
        "correlations": sweep_data["correlations"],
        "neural_freqs": sweep_data["neural_freqs"],
        "cardiac_freqs": sweep_data["cardiac_freqs"],
        "max_correlation": float(
            max(abs(c) for c in sweep_data["correlations"])
        ),
    }

    # Detect frequency entrainment (phase ratio approaches integer)
    valid_ratios = [
        r for r in sweep_data["phase_ratios"] if np.isfinite(r)
    ]
    if valid_ratios:
        nearest_int_ratios = [abs(r - round(r)) for r in valid_ratios]
        results["coupling_sweep"]["entrainment_deviation"] = (
            float(min(nearest_int_ratios))
        )
        results["coupling_sweep"]["mean_phase_ratio"] = (
            float(np.mean(valid_ratios))
        )

    # 2. SINDy ODE recovery
    logger.info("Running SINDy ODE recovery...")
    try:
        import pysindy as ps

        X, dXdt = generate_ode_data()
        feature_names = ["v", "w", "x", "y"]

        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=0.05),
            feature_library=ps.PolynomialLibrary(degree=3),
        )
        model.fit(X, t=0.01, x_dot=dXdt, feature_names=feature_names)

        sindy_results = {"n_discoveries": len(feature_names), "discoveries": []}
        X_pred = model.predict(X)
        r2_vals = []
        for i, name in enumerate(feature_names):
            expr = model.equations()[i] if hasattr(model, "equations") else str(
                model.coefficients()[i]
            )
            ss_res = np.sum((dXdt[:, i] - X_pred[:, i]) ** 2)
            ss_tot = np.sum((dXdt[:, i] - np.mean(dXdt[:, i])) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
            r2_vals.append(r2)
            sindy_results["discoveries"].append({
                "expression": f"d({name})/dt = {expr}",
                "r_squared": float(r2),
            })

        sindy_results["mean_r_squared"] = float(np.mean(r2_vals))
        sindy_results["min_r_squared"] = float(np.min(r2_vals))
        results["sindy_ode"] = sindy_results
    except ImportError:
        logger.warning("PySINDy not available, skipping SINDy analysis")
    except Exception as e:
        logger.warning(f"SINDy analysis failed: {e}")

    # 3. PySR scaling: correlation vs coupling
    logger.info("Running PySR for scaling laws...")
    try:
        from pysr import PySRRegressor

        couplings = np.array(sweep_data["couplings"]).reshape(-1, 1)
        corrs = np.array([abs(c) for c in sweep_data["correlations"]])

        model_corr = PySRRegressor(
            niterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt", "exp", "tanh"],
            populations=10,
            population_size=30,
            maxsize=15,
            progress=False,
            verbosity=0,
        )
        model_corr.fit(couplings, corrs, variable_names=["c_"])

        pysr_corr = {
            "n_discoveries": len(model_corr.equations_),
            "discoveries": [],
        }
        for _, row in model_corr.equations_.iterrows():
            pysr_corr["discoveries"].append({
                "expression": str(row["equation"]),
            })

        y_pred = model_corr.predict(couplings)
        ss_res = np.sum((corrs - y_pred) ** 2)
        ss_tot = np.sum((corrs - np.mean(corrs)) ** 2)
        pysr_corr["best_r2"] = float(1.0 - ss_res / max(ss_tot, 1e-10))
        pysr_corr["best"] = str(
            model_corr.equations_.iloc[model_corr.equations_["loss"].idxmin()][
                "equation"
            ]
        )
        results["pysr_correlation"] = pysr_corr
    except ImportError:
        logger.warning("PySR not available, skipping")
    except Exception as e:
        logger.warning(f"PySR analysis failed: {e}")

    return results
