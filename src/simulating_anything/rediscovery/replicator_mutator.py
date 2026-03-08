"""Replicator-mutator equation rediscovery.

Targets:
- Rock-Paper-Scissors: heteroclinic cycles, interior FP at (1/3, 1/3, 1/3)
- Hawk-Dove: ESS frequency p* = V/C (mixed Nash equilibrium)
- Prisoner's Dilemma: cooperation decay rate (defect dominates)
- Mutation rate sweep: mutation stabilizes interior equilibria
- SINDy ODE recovery on 3-strategy replicator dynamics
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.replicator_mutator import (
    ReplicatorMutatorSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    n_strategies: int = 3,
    payoff: list[list[float]] | None = None,
    mutation_rate: float = 0.0,
    dt: float = 0.01,
    n_steps: int = 1000,
    x_init: list[float] | None = None,
) -> SimulationConfig:
    """Build a SimulationConfig for the replicator-mutator equation.

    Args:
        n_strategies: number of strategies.
        payoff: payoff matrix as list of lists. Defaults to RPS.
        mutation_rate: uniform mutation rate.
        dt: timestep.
        n_steps: number of steps.
        x_init: initial frequencies. Defaults to uniform.

    Returns:
        SimulationConfig with all parameters encoded.
    """
    params: dict[str, float] = {
        "n_strategies": float(n_strategies),
        "mutation_rate": mutation_rate,
    }
    if payoff is not None:
        for i, row in enumerate(payoff):
            for j, val in enumerate(row):
                params[f"A_{i}_{j}"] = val
    if x_init is not None:
        for i, val in enumerate(x_init):
            params[f"x_0_{i}"] = val

    return SimulationConfig(
        domain=Domain.CUSTOM,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


def generate_rps_trajectory(
    n_steps: int = 20000,
    dt: float = 0.01,
    x_init: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Generate Rock-Paper-Scissors trajectory for heteroclinic cycle analysis.

    The RPS payoff matrix A = [[0,-1,1],[1,0,-1],[-1,1,0]] produces
    neutrally stable cycles around the interior fixed point (1/3, 1/3, 1/3).

    Args:
        n_steps: number of timesteps.
        dt: timestep.
        x_init: initial frequencies (default: slightly off-center).

    Returns:
        Dict with time, states, and individual strategy traces.
    """
    if x_init is None:
        x_init = [0.4, 0.35, 0.25]

    config = _make_config(n_strategies=3, dt=dt, n_steps=n_steps, x_init=x_init)
    sim = ReplicatorMutatorSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "x0": states_arr[:, 0],
        "x1": states_arr[:, 1],
        "x2": states_arr[:, 2],
    }


def generate_hawk_dove_data(
    n_V: int = 20,
    n_steps: int = 50000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep V/C ratio in Hawk-Dove game and measure ESS frequency.

    Hawk-Dove payoff matrix (2-strategy):
        A = [[(V-C)/2, V], [0, V/2]]

    When V < C, mixed ESS at p* = V/C.
    When V >= C, Hawk is pure ESS.

    Args:
        n_V: number of V values to sweep.
        n_steps: steps per simulation (long for convergence).
        dt: timestep.

    Returns:
        Dict with V values, C, theoretical and measured ESS frequencies.
    """
    C = 10.0  # fixed cost
    V_values = np.linspace(1.0, 9.0, n_V)  # V < C regime
    p_star_theory = V_values / C
    p_star_measured = []

    for i, V in enumerate(V_values):
        payoff = [[(V - C) / 2.0, V], [0.0, V / 2.0]]
        config = _make_config(
            n_strategies=2,
            payoff=payoff,
            dt=dt,
            n_steps=n_steps,
            x_init=[0.5, 0.5],
        )
        sim = ReplicatorMutatorSimulation(config)
        sim.reset()

        # Run to steady state
        sim.simulate_to_steady_state(max_steps=n_steps)
        p_hawk = sim.observe()[0]
        p_star_measured.append(p_hawk)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  V={V:.1f}, C={C:.1f}: p*(theory)={V/C:.4f}, "
                f"p*(measured)={p_hawk:.4f}"
            )

    return {
        "V": V_values,
        "C": C,
        "p_star_theory": p_star_theory,
        "p_star_measured": np.array(p_star_measured),
    }


def generate_pd_trajectory(
    n_steps: int = 20000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate Prisoner's Dilemma trajectory showing cooperation collapse.

    Classic PD payoff matrix (2-strategy: Cooperate=0, Defect=1):
        A = [[3, 0], [5, 1]]  (R=3, S=0, T=5, P=1)

    Defect dominates: the population evolves toward all-defect.

    Args:
        n_steps: number of timesteps.
        dt: timestep.

    Returns:
        Dict with time, states, cooperation frequency, and decay info.
    """
    payoff = [[3.0, 0.0], [5.0, 1.0]]  # R=3, S=0, T=5, P=1
    config = _make_config(
        n_strategies=2,
        payoff=payoff,
        dt=dt,
        n_steps=n_steps,
        x_init=[0.9, 0.1],  # Start mostly cooperative
    )
    sim = ReplicatorMutatorSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    coop = states_arr[:, 0]

    # Estimate decay rate from log(coop) vs time (exponential decay regime)
    time = np.arange(n_steps + 1) * dt
    # Use middle portion where decay is approximately exponential
    mask = (coop > 0.05) & (coop < 0.85)
    decay_rate = np.nan
    if np.sum(mask) > 10:
        log_coop = np.log(coop[mask] / (1.0 - coop[mask]))
        t_masked = time[mask]
        # Linear fit of log(p/(1-p)) vs t gives the logistic decay rate
        coeffs = np.polyfit(t_masked, log_coop, 1)
        decay_rate = -coeffs[0]  # negative slope = decay rate

    return {
        "time": time,
        "states": states_arr,
        "cooperation": coop,
        "defection": states_arr[:, 1],
        "decay_rate": decay_rate,
    }


def generate_mutation_sweep_data(
    n_mu: int = 20,
    n_steps: int = 30000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep mutation rate in RPS game and measure equilibrium deviation.

    Without mutation, RPS has neutrally stable cycles.
    With mutation, the interior fixed point becomes asymptotically stable.

    Args:
        n_mu: number of mutation rate values.
        n_steps: steps per simulation.
        dt: timestep.

    Returns:
        Dict with mutation rates, final state, deviation from (1/3,1/3,1/3).
    """
    mu_values = np.linspace(0.0, 0.5, n_mu)
    deviations = []
    final_states = []

    for i, mu in enumerate(mu_values):
        config = _make_config(
            n_strategies=3,
            mutation_rate=mu,
            dt=dt,
            n_steps=n_steps,
            x_init=[0.5, 0.3, 0.2],
        )
        sim = ReplicatorMutatorSimulation(config)
        sim.reset()

        # Run to steady state
        sim.simulate_to_steady_state(max_steps=n_steps)
        final = sim.observe().copy()
        final_states.append(final)

        # Deviation from uniform (1/3, 1/3, 1/3)
        uniform = np.ones(3) / 3.0
        dev = np.linalg.norm(final - uniform)
        deviations.append(dev)

        if (i + 1) % 5 == 0:
            logger.info(f"  mu={mu:.3f}: deviation from uniform = {dev:.6f}")

    return {
        "mutation_rate": mu_values,
        "deviation": np.array(deviations),
        "final_states": np.array(final_states),
    }


def generate_ode_data(
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate 3-strategy RPS trajectory data for SINDy ODE recovery.

    Args:
        n_steps: number of timesteps.
        dt: timestep.

    Returns:
        Dict with time, states, and individual strategy traces.
    """
    config = _make_config(
        n_strategies=3,
        dt=dt,
        n_steps=n_steps,
        x_init=[0.4, 0.35, 0.25],
    )
    sim = ReplicatorMutatorSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "x0": states_arr[:, 0],
        "x1": states_arr[:, 1],
        "x2": states_arr[:, 2],
    }


def run_replicator_mutator_rediscovery(
    output_dir: str | Path = "output/rediscovery/replicator_mutator",
    n_iterations: int = 40,
) -> dict:
    """Run replicator-mutator equation rediscovery pipeline.

    Parts:
    1. Rock-Paper-Scissors: heteroclinic cycles, period measurement
    2. Hawk-Dove: ESS frequency p* = V/C via PySR
    3. Prisoner's Dilemma: cooperation decay rate
    4. Mutation rate sweep: stabilization of interior equilibrium
    5. SINDy ODE recovery on 3-strategy dynamics

    Args:
        output_dir: directory for saving results.
        n_iterations: PySR iterations.

    Returns:
        Results dict with all rediscovery outcomes.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "replicator_mutator",
        "targets": {
            "rps_heteroclinic": "interior FP at (1/3, 1/3, 1/3), cyclic orbits",
            "hawk_dove_ess": "p* = V/C for V < C",
            "pd_collapse": "defection dominates, cooperation decays",
            "mutation_stabilizes": "mutation breaks heteroclinic cycles",
            "ode_recovery": "dx_i/dt = x_i*(f_i - phi) via SINDy",
        },
    }

    # --- Part 1: Rock-Paper-Scissors heteroclinic cycles ---
    logger.info("Part 1: Rock-Paper-Scissors heteroclinic cycles...")
    rps_data = generate_rps_trajectory(n_steps=20000, dt=0.01)

    # Verify interior fixed point
    config = _make_config(n_strategies=3, dt=0.01, n_steps=1000)
    sim = ReplicatorMutatorSimulation(config)
    sim.reset()
    fp = sim.interior_fixed_point

    rps_result: dict = {
        "interior_fp": fp.tolist() if fp is not None else None,
    }

    # Check that frequencies stay on simplex and oscillate
    states = rps_data["states"]
    simplex_error = np.max(np.abs(np.sum(states, axis=1) - 1.0))
    rps_result["max_simplex_error"] = float(simplex_error)

    # Measure period
    sim2 = ReplicatorMutatorSimulation(
        _make_config(n_strategies=3, dt=0.005, n_steps=1000, x_init=[0.4, 0.35, 0.25])
    )
    sim2.reset()
    period = sim2.measure_period(n_periods=5)
    rps_result["period"] = float(period)

    # Check oscillation amplitude
    x0 = rps_data["x0"]
    rps_result["x0_range"] = [float(np.min(x0)), float(np.max(x0))]
    rps_result["x0_amplitude"] = float(np.max(x0) - np.min(x0))

    results["rps"] = rps_result
    logger.info(
        f"  Interior FP: {fp}, period: {period:.2f}, "
        f"simplex error: {simplex_error:.2e}"
    )

    # --- Part 2: Hawk-Dove ESS ---
    logger.info("Part 2: Hawk-Dove ESS frequency...")
    hd_data = generate_hawk_dove_data(n_V=20, n_steps=50000, dt=0.005)

    p_theory = hd_data["p_star_theory"]
    p_measured = hd_data["p_star_measured"]
    correlation = float(np.corrcoef(p_theory, p_measured)[0, 1])
    mae = float(np.mean(np.abs(p_theory - p_measured)))

    hd_result: dict = {
        "n_points": len(p_theory),
        "correlation": correlation,
        "mean_absolute_error": mae,
        "V_range": [float(hd_data["V"].min()), float(hd_data["V"].max())],
        "C": float(hd_data["C"]),
    }

    # PySR: find p* = f(V, C)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        V_arr = hd_data["V"]
        C_val = hd_data["C"]
        X = np.column_stack([V_arr, np.full_like(V_arr, C_val)])
        y = p_measured

        logger.info("  Running PySR: p* = f(V, C)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["V_", "C_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt"],
            max_complexity=8,
            populations=20,
            population_size=40,
        )
        hd_result["pysr"] = {
            "n_discoveries": len(discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in discoveries[:5]
            ],
        }
        if discoveries:
            best = discoveries[0]
            hd_result["pysr"]["best"] = best.expression
            hd_result["pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        hd_result["pysr"] = {"error": str(e)}

    results["hawk_dove"] = hd_result

    # --- Part 3: Prisoner's Dilemma cooperation collapse ---
    logger.info("Part 3: Prisoner's Dilemma cooperation collapse...")
    pd_data = generate_pd_trajectory(n_steps=20000, dt=0.01)

    coop = pd_data["cooperation"]
    pd_result: dict = {
        "initial_cooperation": float(coop[0]),
        "final_cooperation": float(coop[-1]),
        "cooperation_collapsed": bool(coop[-1] < 0.01),
        "decay_rate": float(pd_data["decay_rate"]) if np.isfinite(pd_data["decay_rate"]) else None,
    }
    results["prisoners_dilemma"] = pd_result
    logger.info(
        f"  Cooperation: {coop[0]:.3f} -> {coop[-1]:.6f}, "
        f"decay rate: {pd_data['decay_rate']:.4f}"
    )

    # --- Part 4: Mutation rate sweep ---
    logger.info("Part 4: Mutation rate sweep...")
    mu_data = generate_mutation_sweep_data(n_mu=15, n_steps=30000, dt=0.01)

    # High mutation should drive toward uniform
    deviations = mu_data["deviation"]
    mu_result: dict = {
        "n_points": len(deviations),
        "deviation_at_mu0": float(deviations[0]),
        "deviation_at_mu_max": float(deviations[-1]),
        "mutation_stabilizes": bool(deviations[-1] < deviations[0]),
    }

    # Check if deviation decreases with mutation rate
    if len(deviations) > 5:
        from numpy.polynomial import polynomial as P
        coeffs = P.polyfit(mu_data["mutation_rate"], deviations, 1)
        mu_result["deviation_slope"] = float(coeffs[1])
        mu_result["monotonically_decreasing"] = bool(coeffs[1] < 0)

    results["mutation_sweep"] = mu_result
    logger.info(
        f"  Deviation: mu=0 -> {deviations[0]:.4f}, "
        f"mu_max -> {deviations[-1]:.6f}"
    )

    # --- Part 5: SINDy ODE recovery ---
    logger.info("Part 5: SINDy ODE recovery on 3-strategy RPS...")
    ode_data = generate_ode_data(n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=0.005,
            feature_names=["x0", "x1", "x2"],
            threshold=0.01,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries[:6]
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

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
