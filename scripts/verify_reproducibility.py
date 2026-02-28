"""Verify all 14 simulation domains produce deterministic results with fixed seeds.

For each domain, runs the simulation twice with the same seed and compares
outputs via np.allclose. Prints a summary table and saves results to JSON.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulating_anything.simulation.rigid_body import ProjectileSimulation
from simulating_anything.simulation.agent_based import LotkaVolterraSimulation
from simulating_anything.simulation.reaction_diffusion import GrayScottSimulation
from simulating_anything.simulation.epidemiological import SIRSimulation
from simulating_anything.simulation.chaotic_ode import DoublePendulumSimulation
from simulating_anything.simulation.harmonic_oscillator import DampedHarmonicOscillator
from simulating_anything.simulation.lorenz import LorenzSimulation
from simulating_anything.simulation.navier_stokes import NavierStokes2DSimulation
from simulating_anything.simulation.van_der_pol import VanDerPolSimulation
from simulating_anything.simulation.kuramoto import KuramotoSimulation
from simulating_anything.simulation.brusselator import BrusselatorSimulation
from simulating_anything.simulation.fitzhugh_nagumo import FitzHughNagumoSimulation
from simulating_anything.simulation.heat_equation import HeatEquation1DSimulation
from simulating_anything.simulation.logistic_map import LogisticMapSimulation
from simulating_anything.simulation.template import DuffingOscillator
from simulating_anything.types.simulation import Domain, SimulationConfig, SimulationBackend

OUTPUT_DIR = Path("output/reproducibility")


def make_config(
    domain: Domain,
    parameters: dict[str, float],
    n_steps: int = 200,
    dt: float = 0.01,
    seed: int = 42,
    grid_resolution: tuple[int, ...] = (128, 128),
    domain_size: tuple[float, ...] = (1.0, 1.0),
    backend: SimulationBackend = SimulationBackend.CUSTOM_JAX,
) -> SimulationConfig:
    """Build a SimulationConfig with the given parameters."""
    return SimulationConfig(
        domain=domain,
        backend=backend,
        grid_resolution=grid_resolution,
        domain_size=domain_size,
        dt=dt,
        n_steps=n_steps,
        parameters=parameters,
        seed=seed,
    )


def run_simulation(sim_class: type, config: SimulationConfig) -> np.ndarray:
    """Instantiate a simulation, run it, and return the states array."""
    sim = sim_class(config)
    traj = sim.run(n_steps=config.n_steps)
    return traj.states


def verify_domain(
    name: str,
    sim_class: type,
    config: SimulationConfig,
) -> dict:
    """Run simulation twice and compare outputs for determinism."""
    t0 = time.perf_counter()

    states_a = run_simulation(sim_class, config)
    states_b = run_simulation(sim_class, config)

    elapsed = time.perf_counter() - t0

    max_diff = float(np.max(np.abs(states_a - states_b)))
    deterministic = bool(np.allclose(states_a, states_b, atol=0.0, rtol=0.0))
    obs_shape = list(states_a.shape[1:])  # Shape of a single observation

    return {
        "domain": name,
        "obs_shape": obs_shape,
        "n_steps": config.n_steps,
        "deterministic": deterministic,
        "max_diff": max_diff,
        "elapsed_s": round(elapsed, 4),
    }


def get_domain_specs() -> list[tuple[str, type, SimulationConfig]]:
    """Return (name, sim_class, config) for all 15 domains (14 + Duffing template)."""
    specs = []

    # 1. Projectile (rigid body)
    specs.append((
        "Projectile",
        ProjectileSimulation,
        make_config(
            domain=Domain.RIGID_BODY,
            parameters={
                "initial_speed": 30.0,
                "launch_angle": 45.0,
                "gravity": 9.81,
                "drag_coefficient": 0.1,
                "mass": 1.0,
            },
            n_steps=500,
            dt=0.01,
        ),
    ))

    # 2. Lotka-Volterra (agent-based)
    specs.append((
        "Lotka-Volterra",
        LotkaVolterraSimulation,
        make_config(
            domain=Domain.AGENT_BASED,
            parameters={
                "alpha": 1.1,
                "beta": 0.4,
                "gamma": 0.4,
                "delta": 0.1,
                "prey_0": 40.0,
                "predator_0": 9.0,
            },
            n_steps=500,
            dt=0.01,
        ),
    ))

    # 3. Gray-Scott (reaction-diffusion)
    specs.append((
        "Gray-Scott",
        GrayScottSimulation,
        make_config(
            domain=Domain.REACTION_DIFFUSION,
            parameters={
                "D_u": 0.16,
                "D_v": 0.08,
                "f": 0.035,
                "k": 0.065,
            },
            n_steps=100,
            dt=0.005,
            grid_resolution=(32, 32),
            domain_size=(2.5, 2.5),
            backend=SimulationBackend.JAX_FD,
        ),
    ))

    # 4. SIR (epidemiological)
    specs.append((
        "SIR",
        SIRSimulation,
        make_config(
            domain=Domain.EPIDEMIOLOGICAL,
            parameters={
                "beta": 0.3,
                "gamma": 0.1,
                "S_0": 0.99,
                "I_0": 0.01,
            },
            n_steps=500,
            dt=0.1,
        ),
    ))

    # 5. Double Pendulum (chaotic ODE)
    specs.append((
        "Double Pendulum",
        DoublePendulumSimulation,
        make_config(
            domain=Domain.CHAOTIC_ODE,
            parameters={
                "m1": 1.0,
                "m2": 1.0,
                "L1": 1.0,
                "L2": 1.0,
                "g": 9.81,
                "theta1_0": 2.0,
                "theta2_0": 2.5,
                "omega1_0": 0.0,
                "omega2_0": 0.0,
            },
            n_steps=500,
            dt=0.001,
        ),
    ))

    # 6. Harmonic Oscillator
    specs.append((
        "Harmonic Oscillator",
        DampedHarmonicOscillator,
        make_config(
            domain=Domain.HARMONIC_OSCILLATOR,
            parameters={
                "k": 4.0,
                "m": 1.0,
                "c": 0.4,
                "x_0": 2.0,
                "v_0": 0.0,
            },
            n_steps=500,
            dt=0.01,
        ),
    ))

    # 7. Lorenz Attractor
    specs.append((
        "Lorenz",
        LorenzSimulation,
        make_config(
            domain=Domain.LORENZ_ATTRACTOR,
            parameters={
                "sigma": 10.0,
                "rho": 28.0,
                "beta": 2.667,
            },
            n_steps=500,
            dt=0.01,
        ),
    ))

    # 8. Navier-Stokes 2D
    specs.append((
        "Navier-Stokes 2D",
        NavierStokes2DSimulation,
        make_config(
            domain=Domain.NAVIER_STOKES_2D,
            parameters={
                "nu": 0.01,
                "N": 32.0,
            },
            n_steps=100,
            dt=0.01,
        ),
    ))

    # 9. Van der Pol
    specs.append((
        "Van der Pol",
        VanDerPolSimulation,
        make_config(
            domain=Domain.VAN_DER_POL,
            parameters={
                "mu": 1.0,
                "x_0": 2.0,
                "v_0": 0.0,
            },
            n_steps=500,
            dt=0.01,
        ),
    ))

    # 10. Kuramoto Coupled Oscillators
    specs.append((
        "Kuramoto",
        KuramotoSimulation,
        make_config(
            domain=Domain.KURAMOTO,
            parameters={
                "N": 50.0,
                "K": 2.0,
                "omega_std": 1.0,
            },
            n_steps=200,
            dt=0.01,
        ),
    ))

    # 11. Brusselator
    specs.append((
        "Brusselator",
        BrusselatorSimulation,
        make_config(
            domain=Domain.BRUSSELATOR,
            parameters={
                "a": 1.0,
                "b": 3.0,
                "u_0": 1.0,
                "v_0": 1.0,
            },
            n_steps=500,
            dt=0.01,
        ),
    ))

    # 12. FitzHugh-Nagumo
    specs.append((
        "FitzHugh-Nagumo",
        FitzHughNagumoSimulation,
        make_config(
            domain=Domain.FITZHUGH_NAGUMO,
            parameters={
                "a": 0.7,
                "b": 0.8,
                "eps": 0.08,
                "I": 0.5,
                "v_0": -1.0,
                "w_0": -0.5,
            },
            n_steps=500,
            dt=0.1,
        ),
    ))

    # 13. Heat Equation 1D
    specs.append((
        "Heat Equation 1D",
        HeatEquation1DSimulation,
        make_config(
            domain=Domain.HEAT_EQUATION_1D,
            parameters={
                "D": 0.1,
                "N": 64.0,
            },
            n_steps=200,
            dt=0.01,
        ),
    ))

    # 14. Logistic Map
    specs.append((
        "Logistic Map",
        LogisticMapSimulation,
        make_config(
            domain=Domain.LOGISTIC_MAP,
            parameters={
                "r": 3.9,
                "x_0": 0.5,
            },
            n_steps=200,
            dt=1.0,  # Discrete map, dt is nominal
        ),
    ))

    # 15. Duffing Oscillator (template example)
    specs.append((
        "Duffing (template)",
        DuffingOscillator,
        make_config(
            domain=Domain.RIGID_BODY,  # Reuse RIGID_BODY since Duffing has no enum
            parameters={
                "alpha": 1.0,
                "beta": 1.0,
                "delta": 0.2,
                "gamma_f": 0.3,
                "omega": 1.0,
                "x_0": 0.5,
                "v_0": 0.0,
            },
            n_steps=500,
            dt=0.01,
        ),
    ))

    return specs


def print_summary_table(results: list[dict]) -> None:
    """Print a formatted summary table to stdout."""
    header = f"{'Domain':<22} {'obs_shape':<16} {'n_steps':>7} {'Deterministic':>14} {'max_diff':>14}"
    separator = "-" * len(header)
    print()
    print(separator)
    print(header)
    print(separator)

    for r in results:
        shape_str = str(tuple(r["obs_shape"])) if r["obs_shape"] else "(scalar)"
        det_str = "Yes" if r["deterministic"] else "NO"
        diff_str = f"{r['max_diff']:.2e}" if r["max_diff"] > 0 else "0.0"
        print(
            f"{r['domain']:<22} {shape_str:<16} {r['n_steps']:>7} {det_str:>14} {diff_str:>14}"
        )

    print(separator)

    n_pass = sum(1 for r in results if r["deterministic"])
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} domains are fully deterministic.")
    total_time = sum(r["elapsed_s"] for r in results)
    print(f"Total verification time: {total_time:.2f}s")


def save_results(results: list[dict]) -> Path:
    """Save results to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "verification_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "description": "Reproducibility verification for all simulation domains",
                "n_domains": len(results),
                "all_deterministic": all(r["deterministic"] for r in results),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")
    return output_path


def main() -> None:
    """Run reproducibility verification for all domains."""
    print("Verifying simulation reproducibility across all domains...")
    print("Each domain is run twice with the same seed; outputs must be identical.\n")

    specs = get_domain_specs()
    results = []

    for name, sim_class, config in specs:
        print(f"  Checking {name}...", end="", flush=True)
        try:
            result = verify_domain(name, sim_class, config)
            status = "OK" if result["deterministic"] else "MISMATCH"
            print(f" {status} (max_diff={result['max_diff']:.2e}, {result['elapsed_s']:.3f}s)")
            results.append(result)
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({
                "domain": name,
                "obs_shape": [],
                "n_steps": config.n_steps,
                "deterministic": False,
                "max_diff": float("inf"),
                "elapsed_s": 0.0,
                "error": str(e),
            })

    print_summary_table(results)
    save_results(results)


if __name__ == "__main__":
    main()
