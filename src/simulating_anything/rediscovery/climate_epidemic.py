"""Rediscovery for coupled climate-epidemic system.

Novel discovery domain: no known analytical solutions for the coupled system.
Targets:
1. SINDy ODE recovery of the 6D coupled system
2. PySR scaling: epidemic peak timing vs ENSO amplitude
3. PySR scaling: coupling strength vs epidemic periodicity
4. Critical coupling for epidemic synchronization with ENSO
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def generate_coupling_sweep_data(
    n_points: int = 20,
    n_steps: int = 5000,
    dt: float = 0.05,
) -> dict:
    """Sweep coupling_TC and measure epidemic peak timing + amplitude.

    Args:
        n_points: Number of coupling values to sweep.
        n_steps: Simulation steps per coupling value.
        dt: Integration timestep.

    Returns:
        Dict with coupling values, peak timings, peak amplitudes.
    """
    from simulating_anything.simulation.climate_epidemic import (
        ClimateEpidemicSimulation,
    )
    from simulating_anything.types.simulation import Domain, SimulationConfig

    couplings = np.linspace(0.0, 1.0, n_points)
    peak_times = []
    peak_infected = []
    mean_infected = []
    enso_amplitudes = []

    for c in couplings:
        config = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "coupling_TC": float(c),
                "beta_0": 0.5,
                "gamma_epi": 0.1,
                "mu_pop": 0.01,
                "mu_c": 0.5,
                "omega": 2.0,
                "T_0": 0.5,
                "S_0": 0.95,
                "I_0": 0.04,
                "R_0_init": 0.01,
            },
        )
        sim = ClimateEpidemicSimulation(config)
        sim.reset(seed=42)

        trajectory = []
        for _ in range(n_steps):
            trajectory.append(sim.step().copy())
        traj = np.array(trajectory)

        T_series = traj[:, 0]
        I_series = traj[:, 4]

        # Find peak infected
        peak_idx = np.argmax(I_series)
        peak_times.append(float(peak_idx * dt))
        peak_infected.append(float(I_series[peak_idx]))
        mean_infected.append(float(np.mean(I_series)))
        enso_amplitudes.append(float(np.std(T_series)))

    return {
        "couplings": couplings.tolist(),
        "peak_times": peak_times,
        "peak_infected": peak_infected,
        "mean_infected": mean_infected,
        "enso_amplitudes": enso_amplitudes,
    }


def generate_ode_data(
    coupling_TC: float = 0.3,
    n_steps: int = 3000,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Args:
        coupling_TC: Climate-epidemic coupling strength.
        n_steps: Number of timesteps.
        dt: Integration timestep.

    Returns:
        Tuple of (states, derivatives) arrays.
    """
    from simulating_anything.simulation.climate_epidemic import (
        ClimateEpidemicSimulation,
    )
    from simulating_anything.types.simulation import Domain, SimulationConfig

    config = SimulationConfig(
        domain=Domain.CUSTOM,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "coupling_TC": coupling_TC,
            "beta_0": 0.5,
            "gamma_epi": 0.1,
            "mu_pop": 0.01,
            "T_0": 0.5,
            "S_0": 0.95,
            "I_0": 0.04,
            "R_0_init": 0.01,
        },
    )
    sim = ClimateEpidemicSimulation(config)
    sim.reset(seed=0)

    states = [sim.observe().copy()]
    for _ in range(n_steps - 1):
        states.append(sim.step().copy())

    X = np.array(states[:-1], dtype=np.float64)
    X_next = np.array(states[1:], dtype=np.float64)
    dXdt = (X_next - X) / dt

    return X, dXdt


def generate_synchronization_data(
    n_coupling_points: int = 15,
    n_steps: int = 8000,
    dt: float = 0.05,
) -> dict:
    """Measure ENSO-epidemic correlation as a function of coupling.

    Args:
        n_coupling_points: Number of coupling values to test.
        n_steps: Steps per simulation.
        dt: Timestep.

    Returns:
        Dict with coupling values and correlation metrics.
    """
    from simulating_anything.simulation.climate_epidemic import (
        ClimateEpidemicSimulation,
    )
    from simulating_anything.types.simulation import Domain, SimulationConfig

    couplings = np.linspace(0.0, 1.5, n_coupling_points)
    correlations_TI = []
    correlations_TdI = []

    for c in couplings:
        config = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "coupling_TC": float(c),
                "beta_0": 0.5,
                "gamma_epi": 0.1,
                "mu_pop": 0.01,
                "T_0": 0.5,
                "S_0": 0.95,
                "I_0": 0.04,
                "R_0_init": 0.01,
            },
        )
        sim = ClimateEpidemicSimulation(config)
        sim.reset(seed=42)

        trajectory = []
        for _ in range(n_steps):
            trajectory.append(sim.step().copy())
        traj = np.array(trajectory)

        # Skip transient
        traj = traj[n_steps // 4:]

        T = traj[:, 0]
        I = traj[:, 4]

        # Correlation between T and I
        if np.std(T) > 1e-10 and np.std(I) > 1e-10:
            corr = float(np.corrcoef(T, I)[0, 1])
        else:
            corr = 0.0
        correlations_TI.append(corr)

        # Correlation between T and dI/dt
        dI = np.diff(I)
        T_trim = T[:-1]
        if np.std(T_trim) > 1e-10 and np.std(dI) > 1e-10:
            corr_dI = float(np.corrcoef(T_trim, dI)[0, 1])
        else:
            corr_dI = 0.0
        correlations_TdI.append(corr_dI)

    return {
        "couplings": couplings.tolist(),
        "corr_T_I": correlations_TI,
        "corr_T_dI": correlations_TdI,
    }


def run_climate_epidemic_rediscovery(
    n_iterations: int = 40,
) -> dict:
    """Run full rediscovery analysis for coupled climate-epidemic system.

    Args:
        n_iterations: PySR iterations.

    Returns:
        Dict with all discovery results.
    """
    results = {
        "domain": "climate_epidemic",
        "novel": True,
        "description": (
            "Coupled ENSO climate oscillation + SIR epidemic. "
            "Temperature modulates disease transmission rate. "
            "No known analytical results for the coupled system."
        ),
        "targets": {
            "coupling_sweep": "How does coupling affect epidemic dynamics?",
            "synchronization": "Do epidemics synchronize with ENSO?",
            "ode_recovery": "Can SINDy recover the 6D coupled ODE?",
            "scaling_laws": "PySR expressions for coupling-dependent quantities",
        },
    }

    # 1. Coupling sweep
    logger.info("Running coupling sweep...")
    sweep_data = generate_coupling_sweep_data()
    results["coupling_sweep"] = {
        "n_points": len(sweep_data["couplings"]),
        "coupling_range": [
            float(min(sweep_data["couplings"])),
            float(max(sweep_data["couplings"])),
        ],
        "peak_time_range": [
            float(min(sweep_data["peak_times"])),
            float(max(sweep_data["peak_times"])),
        ],
        "peak_infected_range": [
            float(min(sweep_data["peak_infected"])),
            float(max(sweep_data["peak_infected"])),
        ],
    }

    # 2. Synchronization analysis
    logger.info("Running synchronization analysis...")
    sync_data = generate_synchronization_data()
    results["synchronization"] = {
        "couplings": sync_data["couplings"],
        "corr_T_I": sync_data["corr_T_I"],
        "corr_T_dI": sync_data["corr_T_dI"],
        "max_corr_T_I": float(max(np.abs(sync_data["corr_T_I"]))),
        "max_corr_T_dI": float(max(np.abs(sync_data["corr_T_dI"]))),
    }

    # 3. SINDy ODE recovery
    logger.info("Running SINDy ODE recovery...")
    try:
        import pysindy as ps

        X, dXdt = generate_ode_data(coupling_TC=0.3)
        feature_names = ["T", "h", "tau", "S", "I", "R"]

        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=0.05),
            feature_library=ps.PolynomialLibrary(degree=2),
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

    # 4. PySR scaling laws
    logger.info("Running PySR for scaling laws...")
    try:
        from pysr import PySRRegressor

        # Peak infected vs coupling strength
        couplings = np.array(sweep_data["couplings"]).reshape(-1, 1)
        peak_I = np.array(sweep_data["peak_infected"])

        model_peak = PySRRegressor(
            niterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt", "exp"],
            populations=10,
            population_size=30,
            maxsize=15,
            variable_names=["c_"],
            progress=False,
            verbosity=0,
        )
        model_peak.fit(couplings, peak_I)

        pysr_peak = {"n_discoveries": len(model_peak.equations_), "discoveries": []}
        for _, row in model_peak.equations_.iterrows():
            pysr_peak["discoveries"].append({
                "expression": str(row["equation"]),
                "r_squared": float(row["loss"]) if "loss" in row else 0.0,
            })
        best_idx = model_peak.equations_["loss"].idxmin()
        best_eq = model_peak.equations_.loc[best_idx]
        pysr_peak["best"] = str(best_eq["equation"])
        pysr_peak["best_r2"] = float(
            1.0 - best_eq["loss"] / max(
                np.var(peak_I) * len(peak_I), 1e-10
            )
        ) if "loss" in best_eq else 0.0

        # Recompute R² properly
        y_pred = model_peak.predict(couplings)
        ss_res = np.sum((peak_I - y_pred) ** 2)
        ss_tot = np.sum((peak_I - np.mean(peak_I)) ** 2)
        pysr_peak["best_r2"] = float(1.0 - ss_res / max(ss_tot, 1e-10))

        results["pysr_peak_infected"] = pysr_peak
    except ImportError:
        logger.warning("PySR not available, skipping symbolic regression")
    except Exception as e:
        logger.warning(f"PySR analysis failed: {e}")

    return results
