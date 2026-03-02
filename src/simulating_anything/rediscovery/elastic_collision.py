"""Elastic collision chain rediscovery.

Targets:
- Momentum conservation: sum(m_i * v_i) = const
- Energy conservation (elastic, e=1.0): sum(0.5 * m_i * v_i^2) = const
- Newton's cradle: 1 ball hits N-1 stationary -> only last ball moves
- Energy loss vs restitution: KE_final/KE_initial = f(e)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.elastic_collision import (
    ElasticCollisionSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_sim(
    n_particles: int = 5,
    mass: float = 1.0,
    restitution: float = 1.0,
    spacing: float = 2.0,
    init_v0: float = 1.0,
    dt: float = 0.001,
    n_steps: int = 5000,
) -> ElasticCollisionSimulation:
    """Create an ElasticCollisionSimulation with the given parameters."""
    config = SimulationConfig(
        domain=Domain.ELASTIC_COLLISION,  
        dt=dt,
        n_steps=n_steps,
        parameters={
            "n_particles": float(n_particles),
            "mass": mass,
            "restitution": restitution,
            "spacing": spacing,
            "init_velocity_0": init_v0,
        },
    )
    return ElasticCollisionSimulation(config)


def generate_momentum_conservation_data(
    n_trials: int = 20,
    n_particles: int = 5,
    n_steps: int = 5000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Verify momentum conservation across multiple trials.

    Sweeps initial velocity and mass configurations, checking that
    total momentum is preserved throughout each trajectory.

    Returns dict with arrays: trial, p_initial, p_final, relative_drift.
    """
    rng = np.random.default_rng(42)
    p_initial = []
    p_final = []
    rel_drift = []

    for i in range(n_trials):
        # Random initial velocity for first particle
        v0 = rng.uniform(0.5, 5.0)
        mass = rng.uniform(0.5, 3.0)

        sim = _make_sim(
            n_particles=n_particles, mass=mass, init_v0=v0,
            dt=dt, n_steps=n_steps,
        )
        sim.reset()
        p0 = sim.compute_momentum()

        for _ in range(n_steps):
            sim.step()

        pf = sim.compute_momentum()
        drift = abs(pf - p0) / abs(p0) if abs(p0) > 1e-12 else 0.0

        p_initial.append(p0)
        p_final.append(pf)
        rel_drift.append(drift)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  Trial {i+1}/{n_trials}: p0={p0:.4f}, "
                f"pf={pf:.4f}, drift={drift:.2e}"
            )

    return {
        "p_initial": np.array(p_initial),
        "p_final": np.array(p_final),
        "relative_drift": np.array(rel_drift),
    }


def generate_energy_conservation_data(
    n_trials: int = 20,
    n_particles: int = 5,
    n_steps: int = 5000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Verify energy conservation for elastic collisions (e=1.0).

    Returns dict with arrays: trial, KE_initial, KE_final, relative_drift.
    """
    rng = np.random.default_rng(123)
    KE_initial = []
    KE_final = []
    rel_drift = []

    for i in range(n_trials):
        v0 = rng.uniform(0.5, 5.0)
        mass = rng.uniform(0.5, 3.0)

        sim = _make_sim(
            n_particles=n_particles, mass=mass, restitution=1.0,
            init_v0=v0, dt=dt, n_steps=n_steps,
        )
        sim.reset()
        KE0 = sim.compute_kinetic_energy()

        for _ in range(n_steps):
            sim.step()

        KEf = sim.compute_kinetic_energy()
        drift = abs(KEf - KE0) / abs(KE0) if KE0 > 1e-12 else 0.0

        KE_initial.append(KE0)
        KE_final.append(KEf)
        rel_drift.append(drift)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  Trial {i+1}/{n_trials}: KE0={KE0:.4f}, "
                f"KEf={KEf:.4f}, drift={drift:.2e}"
            )

    return {
        "KE_initial": np.array(KE_initial),
        "KE_final": np.array(KE_final),
        "relative_drift": np.array(rel_drift),
    }


def generate_newtons_cradle_data(
    n_particles: int = 5,
    n_steps: int = 10000,
    dt: float = 0.001,
) -> dict[str, np.ndarray | float]:
    """Simulate Newton's cradle: 1 ball hits N-1 stationary equal-mass balls.

    For equal masses, after the collision cascade, only the last ball
    should be moving with the original velocity.

    Returns dict with initial/final velocities and the velocity transfer ratio.
    """
    sim = _make_sim(
        n_particles=n_particles, mass=1.0, restitution=1.0,
        init_v0=1.0, dt=dt, n_steps=n_steps,
    )
    sim.reset()

    # Set up the Newton's cradle configuration
    sim.newtons_cradle_setup(n_moving=1)
    v_initial = sim.velocities.copy()
    v0_magnitude = abs(v_initial[0])

    for _ in range(n_steps):
        sim.step()

    v_final = sim.velocities.copy()

    # In ideal Newton's cradle, last particle should have all the velocity
    last_v = v_final[-1]
    transfer_ratio = abs(last_v) / v0_magnitude if v0_magnitude > 0 else 0.0

    # Check that intermediate particles are nearly stationary
    intermediate_v = v_final[:-1]
    max_intermediate = float(np.max(np.abs(intermediate_v)))

    logger.info(
        f"  Newton's cradle ({n_particles} balls): "
        f"v_last={last_v:.4f}, transfer_ratio={transfer_ratio:.4f}, "
        f"max_intermediate_v={max_intermediate:.2e}"
    )

    return {
        "v_initial": v_initial,
        "v_final": v_final,
        "transfer_ratio": transfer_ratio,
        "max_intermediate_v": max_intermediate,
        "n_particles": n_particles,
    }


def generate_energy_loss_vs_restitution_data(
    n_e_values: int = 25,
    n_particles: int = 5,
    n_steps: int = 5000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Sweep restitution coefficient and measure energy retained after collisions.

    For a single 1D collision of equal masses, the energy retained is:
        KE_final / KE_initial = (1 + e^2) / 2

    For a chain with multiple collisions, the relationship is more complex
    but KE_ratio should monotonically increase with e and equal 1.0 at e=1.0.

    Returns dict with arrays: e_values, KE_ratio, single_collision_theory.
    """
    e_values = np.linspace(0.1, 1.0, n_e_values)
    KE_ratios = []

    for i, e_val in enumerate(e_values):
        sim = _make_sim(
            n_particles=n_particles, mass=1.0, restitution=e_val,
            init_v0=2.0, dt=dt, n_steps=n_steps,
        )
        sim.reset()
        KE0 = sim.compute_kinetic_energy()

        for _ in range(n_steps):
            sim.step()

        KEf = sim.compute_kinetic_energy()
        ratio = KEf / KE0 if KE0 > 0 else 0.0
        KE_ratios.append(ratio)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  e={e_val:.3f}: KE_ratio={ratio:.4f}"
            )

    return {
        "e_values": e_values,
        "KE_ratio": np.array(KE_ratios),
        "single_collision_theory": (1.0 + e_values ** 2) / 2.0,
    }


def generate_two_body_collision_data(
    n_mass_ratios: int = 20,
    dt: float = 0.0005,
    n_steps: int = 10000,
) -> dict[str, np.ndarray]:
    """Verify the two-body elastic collision formula across mass ratios.

    For elastic collision (e=1):
        v1' = ((m1 - m2)*v1 + 2*m2*v2) / (m1 + m2)
        v2' = ((m2 - m1)*v2 + 2*m1*v1) / (m1 + m2)

    With v2=0 initially:
        v1' = (m1 - m2)*v1 / (m1 + m2)
        v2' = 2*m1*v1 / (m1 + m2)

    Returns dict with arrays: mass_ratio, v1_measured, v1_theory,
        v2_measured, v2_theory.
    """
    mass_ratios = np.logspace(-1, 1, n_mass_ratios)
    v1_measured = []
    v1_theory = []
    v2_measured = []
    v2_theory = []

    v0 = 2.0  # initial velocity of particle 1
    m2 = 1.0  # fixed mass of particle 2

    for i, ratio in enumerate(mass_ratios):
        m1 = ratio * m2

        sim = _make_sim(
            n_particles=2, mass=1.0, restitution=1.0,
            init_v0=v0, spacing=3.0, dt=dt, n_steps=n_steps,
        )
        sim.set_masses([m1, m2])
        sim.reset()
        # Re-set masses after reset since reset reinitializes from params
        sim.set_masses([m1, m2])
        sim._vel[0] = v0
        sim._vel[1] = 0.0
        sim._state = sim._pack_state()

        for _ in range(n_steps):
            sim.step()

        v1_meas = sim.velocities[0]
        v2_meas = sim.velocities[1]

        # Theory for elastic collision with v2=0
        M = m1 + m2
        v1_th = (m1 - m2) * v0 / M
        v2_th = 2.0 * m1 * v0 / M

        v1_measured.append(v1_meas)
        v1_theory.append(v1_th)
        v2_measured.append(v2_meas)
        v2_theory.append(v2_th)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  m1/m2={ratio:.2f}: v1={v1_meas:.4f} (theory={v1_th:.4f}), "
                f"v2={v2_meas:.4f} (theory={v2_th:.4f})"
            )

    return {
        "mass_ratio": mass_ratios,
        "v1_measured": np.array(v1_measured),
        "v1_theory": np.array(v1_theory),
        "v2_measured": np.array(v2_measured),
        "v2_theory": np.array(v2_theory),
    }


def run_elastic_collision_rediscovery(
    output_dir: str | Path = "output/rediscovery/elastic_collision",
    n_iterations: int = 40,
) -> dict:
    """Run the full elastic collision chain rediscovery.

    1. Momentum conservation verification
    2. Energy conservation verification (elastic)
    3. Newton's cradle effect
    4. Energy loss vs restitution coefficient
    5. Two-body collision formula verification

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "elastic_collision",
        "targets": {
            "momentum_conservation": "sum(m_i * v_i) = const",
            "energy_conservation": "sum(0.5 * m_i * v_i^2) = const (e=1)",
            "newtons_cradle": "1 ball -> last ball gets all velocity",
            "energy_vs_restitution": "KE_ratio = f(e)",
            "two_body_formula": "v1' = ((m1-m2)v1 + 2m2*v2)/(m1+m2)",
        },
    }

    # --- Part 1: Momentum conservation ---
    logger.info("Part 1: Momentum conservation verification...")
    mom_data = generate_momentum_conservation_data(
        n_trials=20, n_particles=5, n_steps=5000, dt=0.001,
    )
    results["momentum_conservation"] = {
        "n_trials": len(mom_data["relative_drift"]),
        "mean_relative_drift": float(np.mean(mom_data["relative_drift"])),
        "max_relative_drift": float(np.max(mom_data["relative_drift"])),
    }
    logger.info(
        f"  Momentum drift: mean={np.mean(mom_data['relative_drift']):.2e}, "
        f"max={np.max(mom_data['relative_drift']):.2e}"
    )

    # --- Part 2: Energy conservation (elastic) ---
    logger.info("Part 2: Energy conservation verification (e=1.0)...")
    energy_data = generate_energy_conservation_data(
        n_trials=20, n_particles=5, n_steps=5000, dt=0.001,
    )
    results["energy_conservation"] = {
        "n_trials": len(energy_data["relative_drift"]),
        "mean_relative_drift": float(np.mean(energy_data["relative_drift"])),
        "max_relative_drift": float(np.max(energy_data["relative_drift"])),
    }
    logger.info(
        f"  Energy drift: mean={np.mean(energy_data['relative_drift']):.2e}, "
        f"max={np.max(energy_data['relative_drift']):.2e}"
    )

    # --- Part 3: Newton's cradle ---
    logger.info("Part 3: Newton's cradle effect...")
    cradle_data = generate_newtons_cradle_data(
        n_particles=5, n_steps=10000, dt=0.001,
    )
    results["newtons_cradle"] = {
        "n_particles": int(cradle_data["n_particles"]),
        "transfer_ratio": float(cradle_data["transfer_ratio"]),
        "max_intermediate_v": float(cradle_data["max_intermediate_v"]),
    }

    # --- Part 4: Energy loss vs restitution ---
    logger.info("Part 4: Energy loss vs restitution coefficient...")
    eloss_data = generate_energy_loss_vs_restitution_data(
        n_e_values=25, n_particles=5, n_steps=5000, dt=0.001,
    )
    # Check monotonicity and e=1 gives ratio=1
    results["energy_vs_restitution"] = {
        "n_samples": len(eloss_data["e_values"]),
        "KE_ratio_at_e1": float(eloss_data["KE_ratio"][-1]),
        "KE_ratio_at_e01": float(eloss_data["KE_ratio"][0]),
        "is_monotonic": bool(
            np.all(np.diff(eloss_data["KE_ratio"]) >= -0.05)
        ),
    }
    logger.info(
        f"  KE_ratio at e=1.0: {eloss_data['KE_ratio'][-1]:.4f}, "
        f"at e=0.1: {eloss_data['KE_ratio'][0]:.4f}"
    )

    # --- Part 5: Two-body collision formula ---
    logger.info("Part 5: Two-body collision formula verification...")
    two_body_data = generate_two_body_collision_data(
        n_mass_ratios=20, dt=0.0005, n_steps=10000,
    )
    v1_err = np.abs(
        two_body_data["v1_measured"] - two_body_data["v1_theory"]
    )
    v2_err = np.abs(
        two_body_data["v2_measured"] - two_body_data["v2_theory"]
    )
    # Relative error using max(|theory|, 0.01) to avoid division by near-zero
    v1_denom = np.maximum(np.abs(two_body_data["v1_theory"]), 0.01)
    v2_denom = np.maximum(np.abs(two_body_data["v2_theory"]), 0.01)
    results["two_body_collision"] = {
        "n_mass_ratios": len(two_body_data["mass_ratio"]),
        "v1_mean_abs_error": float(np.mean(v1_err)),
        "v2_mean_abs_error": float(np.mean(v2_err)),
        "v1_mean_rel_error": float(np.mean(v1_err / v1_denom)),
        "v2_mean_rel_error": float(np.mean(v2_err / v2_denom)),
    }
    logger.info(
        f"  v1 mean abs error: {np.mean(v1_err):.4f}, "
        f"v2 mean abs error: {np.mean(v2_err):.4f}"
    )

    # --- Part 6: PySR on energy-vs-restitution ---
    logger.info("Part 6: PySR on KE_ratio = f(e)...")
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = eloss_data["e_values"].reshape(-1, 1)
        y = eloss_data["KE_ratio"]

        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["e_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt"],
            max_complexity=10,
            populations=15,
            population_size=30,
        )
        results["pysr_ke_ratio"] = {
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
            results["pysr_ke_ratio"]["best"] = best.expression
            results["pysr_ke_ratio"]["best_r2"] = (
                best.evidence.fit_r_squared
            )
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["pysr_ke_ratio"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save raw data
    np.savez(
        output_path / "momentum_data.npz",
        p_initial=mom_data["p_initial"],
        p_final=mom_data["p_final"],
        relative_drift=mom_data["relative_drift"],
    )
    np.savez(
        output_path / "energy_data.npz",
        KE_initial=energy_data["KE_initial"],
        KE_final=energy_data["KE_final"],
        relative_drift=energy_data["relative_drift"],
    )
    np.savez(
        output_path / "energy_vs_restitution.npz",
        e_values=eloss_data["e_values"],
        KE_ratio=eloss_data["KE_ratio"],
    )
    np.savez(
        output_path / "two_body_data.npz",
        mass_ratio=two_body_data["mass_ratio"],
        v1_measured=two_body_data["v1_measured"],
        v1_theory=two_body_data["v1_theory"],
        v2_measured=two_body_data["v2_measured"],
        v2_theory=two_body_data["v2_theory"],
    )

    return results
