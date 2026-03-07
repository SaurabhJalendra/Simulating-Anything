"""Registry of open/unsolved problems for new discovery attempts.

These are problems where the analytical solution is NOT fully known.
The pipeline will attempt to discover empirical scaling laws, equations,
and relationships from simulation data.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OpenProblem:
    """Description of an open scientific problem."""

    name: str
    domain: str
    description: str
    simulation_class: str  # Module path to simulation class
    parameter_sweeps: dict[str, tuple[float, float, int]] = field(default_factory=dict)
    observables: list[str] = field(default_factory=list)
    known_results: dict[str, str] = field(default_factory=dict)
    difficulty: str = "medium"  # easy/medium/hard


# Registry of open problems for new discovery
OPEN_PROBLEMS: dict[str, OpenProblem] = {
    "three_body_stability": OpenProblem(
        name="Three-Body Stability Boundaries",
        domain="celestial_mechanics",
        description=(
            "Find the stability boundary of orbits near L4/L5 Lagrange points "
            "as a function of mass ratio mu. Known: stable for mu < 0.0385 "
            "(Gascheau's value), but the detailed boundary shape and the "
            "transition dynamics are open questions."
        ),
        simulation_class="simulating_anything.simulation.three_body.ThreeBodySimulation",
        parameter_sweeps={
            "mu": (0.001, 0.05, 30),
            "x0_offset": (-0.1, 0.1, 20),
            "y0_offset": (-0.1, 0.1, 20),
        },
        observables=["jacobi_constant", "orbit_period", "max_displacement", "lyapunov"],
        known_results={
            "gascheau_limit": "mu_c = (1 - sqrt(23/27)) / 2 ~ 0.0385",
            "l4_location": "(0.5 - mu, sqrt(3)/2)",
        },
        difficulty="hard",
    ),
    "three_body_periodic_orbits": OpenProblem(
        name="Three-Body Periodic Orbit Families",
        domain="celestial_mechanics",
        description=(
            "Discover and classify families of periodic orbits in the CR3BP. "
            "The figure-8 orbit was found by Moore (1993) and proven by "
            "Chenciner-Montgomery (2000). Many families remain unexplored."
        ),
        simulation_class="simulating_anything.simulation.three_body.ThreeBodySimulation",
        parameter_sweeps={
            "mu": (0.001, 0.1, 20),
            "x0": (-1.5, 1.5, 30),
            "vy0": (-2.0, 2.0, 30),
        },
        observables=["period", "jacobi_constant", "orbit_classification"],
        known_results={
            "halo_orbits": "Exist near L1, L2 for all mu",
            "figure_eight": "Exists for equal masses (mu=0.5)",
        },
        difficulty="hard",
    ),
    "lorenz_transition_scaling": OpenProblem(
        name="Lorenz Chaos Transition Scaling Laws",
        domain="chaotic_dynamics",
        description=(
            "How do statistical properties (autocorrelation time, fractal "
            "dimension, mixing rate) scale with distance from the critical "
            "Rayleigh number rho_c ~ 24.74? The scaling exponents near the "
            "bifurcation are partially known but not fully characterized."
        ),
        simulation_class="simulating_anything.simulation.lorenz.LorenzSimulation",
        parameter_sweeps={
            "rho": (20.0, 35.0, 50),
            "sigma": (8.0, 12.0, 10),
        },
        observables=["lyapunov", "correlation_dim", "autocorrelation_time", "attractor_volume"],
        known_results={
            "critical_rho": "24.74 (subcritical Hopf)",
            "lyapunov_classic": "0.9056 at sigma=10, rho=28, beta=8/3",
        },
        difficulty="medium",
    ),
    "coupled_oscillator_sync_scaling": OpenProblem(
        name="Synchronization Scaling in Coupled Oscillators",
        domain="collective_dynamics",
        description=(
            "How does the synchronization transition scale with network "
            "topology? The Kuramoto model on complete graphs has K_c = 2/(pi*g(0)) "
            "but on complex networks the relationship between spectral gap, "
            "degree distribution, and critical coupling is open."
        ),
        simulation_class="simulating_anything.simulation.kuramoto.KuramotoSimulation",
        parameter_sweeps={
            "K": (0.1, 5.0, 40),
            "N": (10.0, 500.0, 10),
        },
        observables=["order_parameter", "sync_frequency", "coherence_time"],
        known_results={
            "mean_field_kc": "K_c = 2/(pi*g(0)) for complete graph",
            "finite_size_scaling": "K_c(N) ~ K_c(inf) + O(1/sqrt(N))",
        },
        difficulty="medium",
    ),
    "gray_scott_pattern_selection": OpenProblem(
        name="Gray-Scott Pattern Selection Rules",
        domain="reaction_diffusion",
        description=(
            "What determines whether Gray-Scott produces spots, stripes, "
            "or complex patterns? The pattern selection depends on (f, k, D_ratio) "
            "but the exact phase boundaries and transition mechanisms are "
            "not fully characterized analytically."
        ),
        simulation_class="simulating_anything.simulation.reaction_diffusion.ReactionDiffusionSimulation",
        parameter_sweeps={
            "f": (0.01, 0.08, 30),
            "k": (0.04, 0.07, 30),
        },
        observables=["pattern_type", "wavelength", "pattern_entropy", "spot_count"],
        known_results={
            "turing_condition": "D_u/D_v > threshold for instability",
            "pearson_classification": "alpha, beta, gamma, ... pattern types",
        },
        difficulty="medium",
    ),
    "ising_critical_exponents": OpenProblem(
        name="Ising Model Critical Exponents from Simulation",
        domain="statistical_mechanics",
        description=(
            "Rediscover the 2D Ising critical exponents (beta=1/8, gamma=7/4, "
            "nu=1) from Monte Carlo data. The exact solution (Onsager) is known, "
            "but discovering the exponents purely from data is a test of the "
            "discovery pipeline's ability to find universal scaling laws."
        ),
        simulation_class="simulating_anything.simulation.ising_model.IsingModelSimulation",
        parameter_sweeps={
            "temperature": (1.5, 3.5, 40),
            "N": (16.0, 128.0, 5),
        },
        observables=["magnetization", "susceptibility", "specific_heat", "correlation_length"],
        known_results={
            "T_c": "2 / ln(1 + sqrt(2)) ~ 2.269",
            "beta_exact": "1/8",
            "gamma_exact": "7/4",
            "nu_exact": "1",
        },
        difficulty="medium",
    ),
}


def get_problem(name: str) -> OpenProblem:
    """Get an open problem by name."""
    if name not in OPEN_PROBLEMS:
        available = ", ".join(OPEN_PROBLEMS.keys())
        raise KeyError(f"Unknown problem: {name}. Available: {available}")
    return OPEN_PROBLEMS[name]


def list_problems() -> list[str]:
    """List all registered open problems."""
    return list(OPEN_PROBLEMS.keys())
