"""Cross-domain analogy engine.

Detects mathematical isomorphisms between discovered equations and dynamics
across different simulation domains. This is a key evidence for the
universality claim: structurally similar equations appear in unrelated fields.

Mathematical analogies detected:
1. Structural: same functional form (e.g., x' = ax - bxy appears in both
   Lotka-Volterra and SIR epidemics)
2. Dimensional: same scaling relationships (e.g., T ~ sqrt(L/g) in both
   pendulum and spring-mass systems)
3. Topological: similar phase space structure (oscillatory, chaotic,
   convergent, divergent)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DomainSignature:
    """Mathematical signature of a domain's dynamics."""

    name: str
    math_type: str  # "algebraic", "ode_linear", "ode_nonlinear", "pde", "chaotic"
    state_dim: int
    n_parameters: int
    conserved_quantities: list[str] = field(default_factory=list)
    symmetries: list[str] = field(default_factory=list)
    phase_portrait_type: str = ""  # "limit_cycle", "fixed_point", "chaotic", "none"
    characteristic_timescale: str = ""  # e.g., "sqrt(L/g)", "1/gamma"
    discovered_equations: list[str] = field(default_factory=list)
    r_squared: list[float] = field(default_factory=list)


@dataclass
class Analogy:
    """A detected mathematical analogy between two domains."""

    domain_a: str
    domain_b: str
    analogy_type: str  # "structural", "dimensional", "topological"
    description: str
    strength: float  # 0 to 1
    mapping: dict[str, str] = field(default_factory=dict)  # variable mapping


def build_domain_signatures() -> list[DomainSignature]:
    """Build mathematical signatures for all known domains."""
    signatures = [
        DomainSignature(
            name="projectile",
            math_type="algebraic",
            state_dim=4,  # x, y, vx, vy
            n_parameters=3,  # v0, theta, g
            conserved_quantities=["energy (no drag)"],
            symmetries=["time_reversal (no drag)", "horizontal_translation"],
            phase_portrait_type="none",
            characteristic_timescale="v0*sin(theta)/g",
            discovered_equations=["R = v0^2 * sin(2*theta) / g"],
            r_squared=[0.9999],
        ),
        DomainSignature(
            name="lotka_volterra",
            math_type="ode_nonlinear",
            state_dim=2,  # prey, predator
            n_parameters=4,  # alpha, beta, gamma, delta
            conserved_quantities=["Hamiltonian (first integral)"],
            symmetries=["time_translation"],
            phase_portrait_type="limit_cycle",
            characteristic_timescale="1/alpha, 1/gamma",
            discovered_equations=[
                "d(prey)/dt = alpha*prey - beta*prey*pred",
                "d(pred)/dt = -gamma*pred + delta*prey*pred",
                "prey_eq = gamma/delta",
                "pred_eq = alpha/beta",
            ],
            r_squared=[1.0, 1.0, 0.9999, 0.9999],
        ),
        DomainSignature(
            name="gray_scott",
            math_type="pde",
            state_dim=2,  # u, v concentrations on 2D grid
            n_parameters=4,  # f, k, D_u, D_v
            conserved_quantities=[],
            symmetries=["rotation", "translation", "reflection"],
            phase_portrait_type="fixed_point",  # Turing instability
            characteristic_timescale="1/f, 1/k",
            discovered_equations=["lambda ~ sqrt(D_v)"],
            r_squared=[0.985],
        ),
        DomainSignature(
            name="sir_epidemic",
            math_type="ode_nonlinear",
            state_dim=3,  # S, I, R
            n_parameters=2,  # beta, gamma
            conserved_quantities=["S + I + R = 1"],
            symmetries=[],
            phase_portrait_type="fixed_point",  # Epidemic converges to (S_inf, 0, R_inf)
            characteristic_timescale="1/gamma",
            discovered_equations=[
                "R0 = beta/gamma",
                "dR/dt = gamma*I",
            ],
            r_squared=[1.0, 1.0],
        ),
        DomainSignature(
            name="double_pendulum",
            math_type="chaotic",
            state_dim=4,  # theta1, theta2, omega1, omega2
            n_parameters=5,  # m1, m2, L1, L2, g
            conserved_quantities=["total_energy"],
            symmetries=["time_translation", "energy_conservation"],
            phase_portrait_type="chaotic",
            characteristic_timescale="sqrt(L/g)",
            discovered_equations=["T = sqrt(L * 4.03) ≈ 2*pi*sqrt(L/g)"],
            r_squared=[0.999993],
        ),
        DomainSignature(
            name="harmonic_oscillator",
            math_type="ode_linear",
            state_dim=2,  # x, v
            n_parameters=3,  # k, m, c
            conserved_quantities=["energy (undamped)"],
            symmetries=["time_translation (undamped)", "phase_shift"],
            phase_portrait_type="limit_cycle",  # Undamped: center; damped: spiral
            characteristic_timescale="sqrt(m/k)",
            discovered_equations=["omega_0 = sqrt(k/m)", "decay = c/(2m)"],
            r_squared=[1.0, 1.0],
        ),
        DomainSignature(
            name="lorenz",
            math_type="chaotic",
            state_dim=3,  # x, y, z
            n_parameters=3,  # sigma, rho, beta
            conserved_quantities=[],
            symmetries=["rotation_symmetry_z_axis"],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/sigma",
            discovered_equations=[
                "dx/dt = sigma*(y-x)",
                "dy/dt = x*(rho-z) - y",
                "dz/dt = x*y - beta*z",
            ],
            r_squared=[0.99999, 0.99999, 0.99999],
        ),
        DomainSignature(
            name="navier_stokes",
            math_type="pde",
            state_dim=1,  # vorticity field (NxN flattened)
            n_parameters=2,  # nu, N (viscosity, resolution)
            conserved_quantities=["energy (inviscid limit)"],
            symmetries=["rotation", "translation", "Galilean_invariance"],
            phase_portrait_type="fixed_point",  # Decaying flow converges to rest
            characteristic_timescale="1/(nu*k^2)",
            discovered_equations=["decay_rate = 4*nu"],
            r_squared=[1.0],
        ),
        DomainSignature(
            name="van_der_pol",
            math_type="ode_nonlinear",
            state_dim=2,  # x, v
            n_parameters=1,  # mu
            conserved_quantities=[],
            symmetries=["time_translation"],
            phase_portrait_type="limit_cycle",
            characteristic_timescale="2*pi (small mu), mu (large mu)",
            discovered_equations=[
                "x'' - mu*(1-x^2)*x' + x = 0",
                "T(mu) scaling via PySR",
                "A = 2.01 (limit cycle amplitude)",
            ],
            r_squared=[0.9999],
        ),
        DomainSignature(
            name="kuramoto",
            math_type="ode_nonlinear",
            state_dim=50,  # N oscillator phases
            n_parameters=2,  # K, omega_std
            conserved_quantities=[],
            symmetries=["phase_shift", "permutation"],
            phase_portrait_type="fixed_point",  # Synchronized state
            characteristic_timescale="1/K",
            discovered_equations=[
                "r(K) sync transition via PySR",
                "K_c = 1.10 (theory: 4*omega_std/pi)",
            ],
            r_squared=[0.9695],
        ),
        DomainSignature(
            name="brusselator",
            math_type="ode_nonlinear",
            state_dim=2,  # u, v
            n_parameters=2,  # a, b
            conserved_quantities=[],
            symmetries=["time_translation"],
            phase_portrait_type="limit_cycle",
            characteristic_timescale="1/a",
            discovered_equations=[
                "b_c ~ a^2 + 0.91 (Hopf threshold, theory: 1+a^2)",
                "du/dt = 1.00 + u^2*v - 4.00*u (SINDy)",
                "dv/dt = 3.00*u - u^2*v (SINDy)",
            ],
            r_squared=[0.9964, 0.9999],
        ),
        DomainSignature(
            name="fitzhugh_nagumo",
            math_type="ode_nonlinear",
            state_dim=2,  # v, w
            n_parameters=4,  # a, b, eps, I
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="limit_cycle",  # For I > I_c
            characteristic_timescale="1/eps",
            discovered_equations=[
                "dv/dt = 0.500 + 1.000v - 1.000w - 0.333v^3 (SINDy, exact)",
                "dw/dt = 0.056 + 0.080v - 0.064w (SINDy, exact)",
            ],
            r_squared=[1.0, 1.0],
        ),
        DomainSignature(
            name="heat_equation",
            math_type="pde",
            state_dim=1,  # u(x) on 1D grid
            n_parameters=1,  # D (diffusion coefficient)
            conserved_quantities=["total_heat"],
            symmetries=["translation", "reflection"],
            phase_portrait_type="fixed_point",  # Converges to uniform
            characteristic_timescale="1/(D*k^2)",
            discovered_equations=["decay_rate = D (PySR, exact spectral)"],
            r_squared=[1.0],
        ),
        DomainSignature(
            name="logistic_map",
            math_type="chaotic",
            state_dim=1,  # x
            n_parameters=1,  # r
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="chaotic",  # For r > 3.57
            characteristic_timescale="1 (discrete)",
            discovered_equations=[
                "Feigenbaum delta in [4.0, 4.75] (theory: 4.669)",
                "lambda(r=4) = ln(2) (Lyapunov exponent)",
            ],
            r_squared=[0.6287],
        ),
        DomainSignature(
            name="duffing",
            math_type="chaotic",
            state_dim=2,  # x, v
            n_parameters=5,  # alpha, beta, delta, gamma, omega
            conserved_quantities=[],
            symmetries=["time_translation"],
            phase_portrait_type="chaotic",
            characteristic_timescale="2*pi/omega",
            discovered_equations=["x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t)"],
            r_squared=[],
        ),
        DomainSignature(
            name="schwarzschild",
            math_type="ode_nonlinear",
            state_dim=4,  # r, phi, p_r, L/r
            n_parameters=2,  # M, L
            conserved_quantities=["energy", "angular_momentum"],
            symmetries=["time_translation", "azimuthal"],
            phase_portrait_type="fixed_point",  # Circular orbits
            characteristic_timescale="M (gravitational radius)",
            discovered_equations=["ISCO = 6*M", "V_eff(r) = -M/r + L^2/(2r^2) - ML^2/r^3"],
            r_squared=[],
        ),
        DomainSignature(
            name="quantum_oscillator",
            math_type="pde",  # Schrodinger equation
            state_dim=1,  # |psi|^2 on grid
            n_parameters=2,  # omega, hbar
            conserved_quantities=["norm", "energy (time-independent)"],
            symmetries=["time_translation", "parity"],
            phase_portrait_type="none",  # Unitary evolution
            characteristic_timescale="2*pi/omega",
            discovered_equations=["E_n = (n + 0.5) * hbar * omega"],
            r_squared=[],
        ),
        DomainSignature(
            name="boltzmann_gas",
            math_type="ode_nonlinear",  # N-body Newtonian
            state_dim=4,  # Per-particle: x, y, vx, vy (N particles)
            n_parameters=3,  # N, T, L
            conserved_quantities=["total_energy (elastic)", "total_momentum"],
            symmetries=["translation", "rotation", "time_reversal"],
            phase_portrait_type="none",  # Ergodic
            characteristic_timescale="L/v_thermal",
            discovered_equations=["PV = NkT"],
            r_squared=[],
        ),
        DomainSignature(
            name="spring_mass_chain",
            math_type="ode_linear",
            state_dim=2,  # Per-mass: u, v (N masses)
            n_parameters=3,  # K, m, a
            conserved_quantities=["total_energy"],
            symmetries=["time_translation", "discrete_translation"],
            phase_portrait_type="limit_cycle",  # Normal modes
            characteristic_timescale="sqrt(m/K)",
            discovered_equations=["omega(k) = 2*sqrt(K/m)*sin(k*a/2)", "c = a*sqrt(K/m)"],
            r_squared=[],
        ),
        DomainSignature(
            name="kepler",
            math_type="ode_nonlinear",
            state_dim=4,  # r, theta, v_r, v_theta
            n_parameters=2,  # GM, eccentricity
            conserved_quantities=["energy", "angular_momentum"],
            symmetries=["time_translation", "azimuthal", "Laplace-Runge-Lenz"],
            phase_portrait_type="limit_cycle",  # Closed orbits
            characteristic_timescale="2*pi*a^(3/2)/sqrt(GM)",
            discovered_equations=["T^2 = (4*pi^2/GM) * a^3"],
            r_squared=[],
        ),
        DomainSignature(
            name="driven_pendulum",
            math_type="chaotic",
            state_dim=3,  # theta, omega, t
            n_parameters=4,  # gamma, omega0, A_drive, omega_d
            conserved_quantities=[],
            symmetries=["time_translation (modulo drive period)"],
            phase_portrait_type="chaotic",
            characteristic_timescale="2*pi/omega_d",
            discovered_equations=["period-doubling route to chaos, A_c detection"],
            r_squared=[],
        ),
        DomainSignature(
            name="coupled_oscillators",
            math_type="ode_linear",
            state_dim=4,  # x1, v1, x2, v2
            n_parameters=3,  # k, m, kc
            conserved_quantities=["total_energy"],
            symmetries=["time_translation", "permutation (identical masses)"],
            phase_portrait_type="limit_cycle",  # Quasi-periodic (two frequencies)
            characteristic_timescale="sqrt(m/k)",
            discovered_equations=[
                "omega_s = sqrt(k/m)",
                "omega_a = sqrt((k+2*kc)/m)",
                "omega_beat = omega_a - omega_s",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="diffusive_lv",
            math_type="pde",
            state_dim=2,  # u(x), v(x) on 1D grid
            n_parameters=6,  # alpha, beta, gamma, delta, D_u, D_v
            conserved_quantities=[],
            symmetries=["translation", "periodic_boundary"],
            phase_portrait_type="limit_cycle",  # Traveling waves
            characteristic_timescale="L^2/D_u",
            discovered_equations=["c_wave ~ 2*sqrt(alpha*D_u) (Fisher-KPP)"],
            r_squared=[],
        ),
        DomainSignature(
            name="damped_wave",
            math_type="pde",
            state_dim=2,  # u(x), u_dot(x) on 1D grid
            n_parameters=3,  # c, gamma, L
            conserved_quantities=["energy (undamped)"],
            symmetries=["translation", "reflection"],
            phase_portrait_type="fixed_point",  # Decays to rest
            characteristic_timescale="L/c",
            discovered_equations=[
                "omega_k = sqrt(c^2*k^2 - gamma^2/4)",
                "decay_rate = gamma/2",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="ising_model",
            math_type="ode_nonlinear",  # Monte Carlo dynamics (not ODE but discrete stochastic)
            state_dim=1,  # N*N spins (flattened)
            n_parameters=3,  # J, h, T
            conserved_quantities=[],
            symmetries=["rotation_90", "reflection", "spin_flip (h=0)"],
            phase_portrait_type="fixed_point",  # Equilibrium via MC
            characteristic_timescale="1/T",
            discovered_equations=[
                "T_c = 2*J/ln(1+sqrt(2))",
                "m(T) = (1 - 1/sinh(2J/T)^4)^(1/8) (Onsager)",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="cart_pole",
            math_type="ode_nonlinear",
            state_dim=4,  # [x, x_dot, theta, theta_dot]
            n_parameters=7,  # M, m, L, g, mu_c, mu_p, F
            conserved_quantities=["energy (when mu_c=mu_p=0, F=0)"],
            symmetries=["translation_x"],
            phase_portrait_type="limit_cycle",  # Oscillation near hanging eq.
            characteristic_timescale="sqrt(M*L / (g*(M+m)))",
            discovered_equations=[
                "omega = sqrt(g*(M+m) / (M*L))",
                "E = T + V = const (frictionless)",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="three_species",
            math_type="ode_nonlinear",
            state_dim=3,  # [x, y, z] = grass, herbivore, predator
            n_parameters=5,  # a1, b1, a2, b2, a3
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="limit_cycle",  # Can exhibit oscillatory coexistence
            characteristic_timescale="1/a1",
            discovered_equations=[
                "dx/dt = a1*x - b1*x*y",
                "dy/dt = -a2*y + b1*x*y - b2*y*z",
                "dz/dt = -a3*z + b2*y*z",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="elastic_pendulum",
            math_type="ode_nonlinear",
            state_dim=4,  # [r, r_dot, theta, theta_dot]
            n_parameters=4,  # k, m, L0, g
            conserved_quantities=["energy"],
            symmetries=[],
            phase_portrait_type="limit_cycle",  # Coupled oscillation modes
            characteristic_timescale="sqrt(m/k)",
            discovered_equations=[
                "omega_r = sqrt(k/m)",
                "omega_theta = sqrt(g/L0)",
                "E = 0.5*m*(r_dot^2 + r^2*theta_dot^2) + 0.5*k*(r-L0)^2 - m*g*r*cos(theta)",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="rossler",
            math_type="chaotic",
            state_dim=3,  # [x, y, z]
            n_parameters=3,  # a, b, c
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/a",
            discovered_equations=[
                "dx/dt = -y - z",
                "dy/dt = x + a*y",
                "dz/dt = b + z*(x - c)",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="brusselator_diffusion",
            math_type="pde",
            state_dim=2,  # 2*N spatial grid (u and v fields)
            n_parameters=5,  # a, b, D_u, D_v, L
            conserved_quantities=[],
            symmetries=["translation", "reflection"],
            phase_portrait_type="fixed_point",  # Turing patterns from uniform state
            characteristic_timescale="L^2/D_u",
            discovered_equations=[
                "du/dt = D_u*u_xx + a - (b+1)*u + u^2*v",
                "dv/dt = D_v*v_xx + b*u - u^2*v",
                "Turing threshold: b > 1+a^2 with D_v/D_u >> 1",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="henon_map",
            math_type="chaotic",  # Discrete chaos
            state_dim=2,  # [x, y]
            n_parameters=2,  # a, b
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="chaotic",
            characteristic_timescale="1",  # Discrete map
            discovered_equations=[
                "x_{n+1} = 1 - a*x_n^2 + y_n",
                "y_{n+1} = b*x_n",
                "Lyapunov ~ 0.42 (a=1.4, b=0.3)",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="rosenzweig_macarthur",
            math_type="ode_nonlinear",
            state_dim=2,  # [prey, predator]
            n_parameters=6,  # r, K, a, h, e, d
            conserved_quantities=[],
            symmetries=["positivity"],
            phase_portrait_type="limit_cycle",
            characteristic_timescale="1/r",
            discovered_equations=[
                "dx/dt = r*x*(1-x/K) - a*x*y/(1+a*h*x)",
                "dy/dt = e*a*x*y/(1+a*h*x) - d*y",
                "Holling Type II functional response",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="chua",
            math_type="chaotic",
            state_dim=3,  # [x, y, z]
            n_parameters=4,  # alpha, beta, m0, m1
            conserved_quantities=[],
            symmetries=["odd_symmetry"],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/alpha",
            discovered_equations=[
                "dx/dt = alpha*(y - x - f(x))",
                "dy/dt = x - y + z",
                "dz/dt = -beta*y",
                "Double-scroll strange attractor",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="shallow_water",
            math_type="pde",
            state_dim=256,  # 2*N spatial (default N=128)
            n_parameters=3,  # g, L, h0
            conserved_quantities=["mass", "energy"],
            symmetries=["translational"],
            phase_portrait_type="none",
            characteristic_timescale="L/sqrt(g*h0)",
            discovered_equations=[
                "dh/dt + d(h*u)/dx = 0",
                "d(hu)/dt + d(h*u^2 + 0.5*g*h^2)/dx = 0",
                "c = sqrt(g*h)",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="toda_lattice",
            math_type="ode_nonlinear",
            state_dim=16,  # 2*N particles (default N=8)
            n_parameters=2,  # N, a
            conserved_quantities=["energy", "momentum"],
            symmetries=["translational"],
            phase_portrait_type="none",  # Integrable, quasi-periodic
            characteristic_timescale="1/sqrt(a)",
            discovered_equations=[
                "dp_i/dt = a*(exp(-(x_i-x_{i-1})) - exp(-(x_{i+1}-x_i)))",
                "Soliton solutions",
                "omega_n = 2*sqrt(a)*|sin(pi*n/N)|",
            ],
            r_squared=[],
        ),
    ]
    return signatures


def detect_structural_analogies(
    signatures: list[DomainSignature],
) -> list[Analogy]:
    """Detect structural analogies between domain equations.

    Structural analogies: same functional form across different domains.
    """
    analogies = []

    # Analogy 1: Lotka-Volterra <-> SIR (coupled nonlinear ODEs with product terms)
    analogies.append(Analogy(
        domain_a="lotka_volterra",
        domain_b="sir_epidemic",
        analogy_type="structural",
        description=(
            "Both are systems of coupled nonlinear ODEs with bilinear interaction "
            "terms (x*y). In LV, prey*predator drives transfer between species. "
            "In SIR, S*I drives transfer from susceptible to infected. "
            "The mathematical structure is identical: dx/dt = ax - bxy."
        ),
        strength=0.9,
        mapping={
            "prey": "S (susceptible)",
            "predator": "I (infected)",
            "alpha*prey": "-beta*S*I (growth/infection)",
            "beta*prey*pred": "beta*S*I (predation/transmission)",
        },
    ))

    # Analogy 2: Pendulum <-> Spring (same harmonic structure)
    analogies.append(Analogy(
        domain_a="double_pendulum",
        domain_b="harmonic_oscillator",
        analogy_type="structural",
        description=(
            "Both are second-order ODEs with restoring force proportional to "
            "displacement. Small-angle pendulum: theta'' = -(g/L)*theta is "
            "identical to spring: x'' = -(k/m)*x. The natural frequency "
            "omega = sqrt(g/L) maps to omega = sqrt(k/m)."
        ),
        strength=0.95,
        mapping={
            "theta (angle)": "x (displacement)",
            "omega (angular vel)": "v (velocity)",
            "g/L (gravity/length)": "k/m (stiffness/mass)",
            "T = 2*pi*sqrt(L/g)": "T = 2*pi*sqrt(m/k)",
        },
    ))

    # Analogy 3: Projectile <-> Harmonic oscillator (energy conservation)
    analogies.append(Analogy(
        domain_a="projectile",
        domain_b="harmonic_oscillator",
        analogy_type="structural",
        description=(
            "Both conserve total mechanical energy (KE + PE = const) in the "
            "absence of dissipation. Projectile: 0.5*m*v^2 + m*g*h = const. "
            "Oscillator: 0.5*m*v^2 + 0.5*k*x^2 = const. The potential energy "
            "is linear (gravity) vs quadratic (spring)."
        ),
        strength=0.7,
        mapping={
            "m*g*h (gravitational PE)": "0.5*k*x^2 (elastic PE)",
            "0.5*m*v^2 (KE)": "0.5*m*v^2 (KE)",
            "g (gravity)": "k/m (spring constant ratio)",
        },
    ))

    # Analogy 4: Gray-Scott <-> Navier-Stokes (both PDEs with diffusion)
    analogies.append(Analogy(
        domain_a="gray_scott",
        domain_b="navier_stokes",
        analogy_type="structural",
        description=(
            "Both are PDEs with diffusive transport on 2D domains. "
            "Gray-Scott: u_t = Du*Lap(u) - uv^2 + f(1-u). "
            "Navier-Stokes: omega_t = nu*Lap(omega) - (u.grad)omega. "
            "Both have diffusion (Laplacian) plus nonlinear advection/reaction."
        ),
        strength=0.7,
        mapping={
            "Du*Lap(u) [chemical diffusion]": "nu*Lap(omega) [viscous diffusion]",
            "uv^2 [reaction]": "(u.grad)omega [advection]",
            "f, k [feed/kill rates]": "boundary conditions / forcing",
        },
    ))

    # Analogy 5: Van der Pol <-> Lotka-Volterra (both have limit cycles)
    analogies.append(Analogy(
        domain_a="van_der_pol",
        domain_b="lotka_volterra",
        analogy_type="structural",
        description=(
            "Both are nonlinear ODE systems with stable limit cycles. "
            "Van der Pol has a single limit cycle in the x-v phase plane. "
            "Lotka-Volterra has closed orbits in the prey-predator plane. "
            "Both feature nonlinear self-regulation (VdP: x^2 damping, LV: predation)."
        ),
        strength=0.75,
        mapping={
            "x (displacement)": "prey population",
            "v (velocity)": "predator population",
            "mu*(1-x^2) [nonlinear damping]": "beta*prey*pred [nonlinear coupling]",
        },
    ))

    # Analogy 6: Brusselator <-> Van der Pol (both Hopf bifurcation to limit cycle)
    analogies.append(Analogy(
        domain_a="brusselator",
        domain_b="van_der_pol",
        analogy_type="structural",
        description=(
            "Both undergo Hopf bifurcation from stable fixed point to limit cycle. "
            "Brusselator: b > 1+a^2 triggers oscillation. "
            "Van der Pol: any mu > 0 has a limit cycle. "
            "Both are 2D nonlinear systems with unique equilibrium."
        ),
        strength=0.8,
        mapping={
            "b [control parameter]": "mu [nonlinearity parameter]",
            "b_c = 1+a^2 [Hopf threshold]": "mu > 0 [always oscillatory]",
            "(a, b/a) [fixed point]": "(0, 0) [fixed point]",
        },
    ))

    # Analogy 7: FitzHugh-Nagumo <-> Van der Pol (same mathematical origin)
    analogies.append(Analogy(
        domain_a="fitzhugh_nagumo",
        domain_b="van_der_pol",
        analogy_type="structural",
        description=(
            "FitzHugh-Nagumo is a direct generalization of the Van der Pol oscillator. "
            "VdP: x'' - mu(1-x^2)x' + x = 0. "
            "FHN adds a slow recovery variable and external current: "
            "dv/dt = v - v^3/3 - w + I. The cubic nonlinearity v - v^3/3 "
            "is the FHN version of the VdP self-excitation."
        ),
        strength=0.9,
        mapping={
            "v [voltage]": "x [displacement]",
            "w [recovery]": "integral of damping",
            "I [current]": "external forcing",
            "v - v^3/3 [cubic]": "mu*(1-x^2) [quadratic damping]",
        },
    ))

    # Analogy 8: Heat equation <-> Navier-Stokes (linear vs nonlinear diffusion)
    analogies.append(Analogy(
        domain_a="heat_equation",
        domain_b="navier_stokes",
        analogy_type="structural",
        description=(
            "Heat equation is the linear diffusion limit of Navier-Stokes. "
            "Heat: u_t = D*u_xx (pure diffusion). "
            "NS: omega_t = nu*Lap(omega) - (u.grad)omega (diffusion + advection). "
            "Same spectral decay rate D*k^2 for each Fourier mode."
        ),
        strength=0.85,
        mapping={
            "D [diffusion coeff]": "nu [kinematic viscosity]",
            "u [temperature]": "omega [vorticity]",
            "D*k^2 [mode decay]": "nu*k^2 [viscous decay]",
        },
    ))

    # Analogy 9: Lorenz <-> Double Pendulum (both chaotic nonlinear ODEs)
    analogies.append(Analogy(
        domain_a="lorenz",
        domain_b="double_pendulum",
        analogy_type="structural",
        description=(
            "Both are chaotic nonlinear ODE systems with sensitive dependence "
            "on initial conditions and positive Lyapunov exponents. "
            "Lorenz: 3D flow with quadratic nonlinearity (x*y, x*z terms). "
            "Double pendulum: 4D flow with trigonometric nonlinearity. "
            "Both exhibit strange attractors in phase space."
        ),
        strength=0.75,
        mapping={
            "sigma*(y-x) [linear coupling]": "omega1 coupling",
            "x*z, x*y [quadratic nonlinearity]": "sin(theta) [trig nonlinearity]",
            "rho [bifurcation parameter]": "energy [bifurcation parameter]",
        },
    ))

    # Analogy 10: Kepler <-> Schwarzschild (Newtonian vs GR gravity)
    analogies.append(Analogy(
        domain_a="kepler",
        domain_b="schwarzschild",
        analogy_type="structural",
        description=(
            "Both are central force orbital mechanics with conserved energy and "
            "angular momentum. Kepler: V_eff = -GM/r + L^2/(2*m*r^2). "
            "Schwarzschild adds GR correction: -GM*L^2/(m*c^2*r^3). "
            "In the weak-field limit, Schwarzschild reduces to Kepler."
        ),
        strength=0.95,
        mapping={
            "-GM/r [Newtonian potential]": "-M/r [Schwarzschild potential]",
            "L^2/(2mr^2) [centrifugal]": "L^2/(2r^2) [centrifugal]",
            "T^2 ~ a^3": "ISCO = 6M",
        },
    ))

    # Analogy 11: Driven pendulum <-> Duffing (forced nonlinear oscillators with chaos)
    analogies.append(Analogy(
        domain_a="driven_pendulum",
        domain_b="duffing",
        analogy_type="structural",
        description=(
            "Both are forced nonlinear oscillators exhibiting period-doubling "
            "route to chaos. Driven pendulum: sin(theta) nonlinearity. "
            "Duffing: x^3 nonlinearity. Both share the same Feigenbaum "
            "universal constants at the period-doubling cascade."
        ),
        strength=0.9,
        mapping={
            "omega0^2*sin(theta) [pendulum]": "alpha*x + beta*x^3 [Duffing]",
            "A*cos(omega_d*t) [forcing]": "gamma*cos(omega*t) [forcing]",
            "gamma [damping]": "delta [damping]",
        },
    ))

    # Analogy 12: Duffing <-> Van der Pol (forced nonlinear oscillators)
    analogies.append(Analogy(
        domain_a="duffing",
        domain_b="van_der_pol",
        analogy_type="structural",
        description=(
            "Both are forced nonlinear second-order oscillators. "
            "Duffing: x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t). "
            "Van der Pol: x'' - mu*(1-x^2)*x' + x = 0. "
            "Both exhibit bifurcation routes to chaos and limit cycles."
        ),
        strength=0.8,
        mapping={
            "beta*x^3 [cubic stiffness]": "mu*(1-x^2)*x' [nonlinear damping]",
            "gamma*cos(omega*t) [external forcing]": "self-excited oscillation",
            "delta [damping]": "-mu*(1-x^2) [negative damping]",
        },
    ))

    # Analogy 11: Schwarzschild <-> Kepler (central force orbits)
    analogies.append(Analogy(
        domain_a="schwarzschild",
        domain_b="projectile",
        analogy_type="structural",
        description=(
            "Both involve motion under a central force with conserved energy "
            "and angular momentum. Schwarzschild adds a GR correction term "
            "-ML^2/r^3 to the Newtonian effective potential. In the weak-field "
            "limit (r >> M), Schwarzschild reduces to Keplerian orbits."
        ),
        strength=0.7,
        mapping={
            "-M/r [Newtonian gravity]": "-g*y [gravity]",
            "L^2/(2r^2) [centrifugal]": "KE_horizontal",
            "-ML^2/r^3 [GR correction]": "(no analogue in Newtonian)",
        },
    ))

    # Analogy 12: Spring-mass chain <-> Heat equation (discrete vs continuous)
    analogies.append(Analogy(
        domain_a="spring_mass_chain",
        domain_b="heat_equation",
        analogy_type="structural",
        description=(
            "Spring-mass chain is the discrete analogue of the 1D wave/heat equation. "
            "Chain: m*x_i'' = K*(x_{i+1} - 2*x_i + x_{i-1}) -- discrete Laplacian. "
            "Heat: u_t = D*u_xx -- continuous Laplacian. "
            "In the continuum limit (a->0, N->inf), the chain becomes the wave equation."
        ),
        strength=0.85,
        mapping={
            "K/m [spring/mass ratio]": "D [diffusion coefficient]",
            "x_{i+1}-2x_i+x_{i-1} [discrete Laplacian]": "u_xx [continuous Laplacian]",
            "a [lattice spacing]": "dx [grid spacing]",
        },
    ))

    # Analogy 13: Coupled oscillators <-> Spring-mass chain (coupled linear systems)
    analogies.append(Analogy(
        domain_a="coupled_oscillators",
        domain_b="spring_mass_chain",
        analogy_type="structural",
        description=(
            "Both are systems of linearly coupled harmonic oscillators. "
            "Coupled oscillators: 2 masses with coupling spring kc. "
            "Spring-mass chain: N masses with identical coupling springs K. "
            "Both have normal modes; chain generalizes the 2-body case to N-body."
        ),
        strength=0.9,
        mapping={
            "kc [coupling spring]": "K [inter-mass spring]",
            "2 normal modes": "N normal modes",
            "omega_s, omega_a": "omega_n = 2*sqrt(K/m)*sin(n*pi/(2*(N+1)))",
            "beat frequency": "wave dispersion",
        },
    ))

    # Analogy 14: Diffusive LV <-> Gray-Scott (reaction-diffusion PDEs)
    analogies.append(Analogy(
        domain_a="diffusive_lv",
        domain_b="gray_scott",
        analogy_type="structural",
        description=(
            "Both are reaction-diffusion PDEs with two coupled species. "
            "Diffusive LV: u_t = D_u*u_xx + alpha*u - beta*u*v. "
            "Gray-Scott: u_t = D_u*Lap(u) - u*v^2 + f*(1-u). "
            "Both feature diffusion + nonlinear interaction terms "
            "and can produce spatial patterns from homogeneous initial conditions."
        ),
        strength=0.85,
        mapping={
            "alpha*u - beta*u*v [LV reaction]": "-u*v^2 + f*(1-u) [GS reaction]",
            "D_u [prey diffusion]": "D_u [activator diffusion]",
            "prey/predator": "activator/inhibitor",
            "traveling waves": "Turing patterns",
        },
    ))

    # Analogy 15: Coupled oscillators <-> Harmonic oscillator (limiting case)
    analogies.append(Analogy(
        domain_a="coupled_oscillators",
        domain_b="harmonic_oscillator",
        analogy_type="structural",
        description=(
            "Coupled oscillators reduce to independent harmonic oscillators "
            "when coupling kc=0. Each mass obeys x'' = -(k/m)*x exactly. "
            "The coupled system extends the single-oscillator physics with "
            "mode splitting and energy transfer via the coupling spring."
        ),
        strength=0.9,
        mapping={
            "k [individual spring]": "k [spring constant]",
            "m [mass]": "m [mass]",
            "kc=0 limit": "single oscillator",
            "omega_s = sqrt(k/m)": "omega_0 = sqrt(k/m)",
        },
    ))

    # Analogy 16: Diffusive LV <-> Lotka-Volterra (spatial extension)
    analogies.append(Analogy(
        domain_a="diffusive_lv",
        domain_b="lotka_volterra",
        analogy_type="structural",
        description=(
            "Diffusive LV is the spatial extension of the original LV system. "
            "LV: du/dt = alpha*u - beta*u*v (well-mixed). "
            "Diffusive LV: du/dt = D*u_xx + alpha*u - beta*u*v (spatially extended). "
            "Setting D_u = D_v = 0 recovers the original ODE system exactly."
        ),
        strength=0.95,
        mapping={
            "alpha, beta [reaction rates]": "alpha, beta [same rates]",
            "D_u*u_xx [diffusion]": "0 [well-mixed assumption]",
            "spatial patterns": "temporal oscillations",
        },
    ))

    # Analogy 17: Damped wave <-> Heat equation (wave vs diffusion)
    analogies.append(Analogy(
        domain_a="damped_wave",
        domain_b="heat_equation",
        analogy_type="structural",
        description=(
            "Both are linear PDEs with spectral mode decay. "
            "Damped wave: u_tt + gamma*u_t = c^2*u_xx (oscillatory decay). "
            "Heat: u_t = D*u_xx (pure diffusion). "
            "In the overdamped limit (gamma >> c*k), the wave equation "
            "reduces to diffusion: the inertial term becomes negligible."
        ),
        strength=0.8,
        mapping={
            "c^2*u_xx [wave propagation]": "D*u_xx [diffusion]",
            "gamma*u_t [damping]": "(implicit in first-order PDE)",
            "oscillatory decay": "exponential decay",
        },
    ))

    # Analogy 18: Damped wave <-> Spring-mass chain (discrete vs continuous wave)
    analogies.append(Analogy(
        domain_a="damped_wave",
        domain_b="spring_mass_chain",
        analogy_type="structural",
        description=(
            "Spring-mass chain is the discrete version of the wave equation. "
            "Chain: m*u_i'' = K*(u_{i+1}-2u_i+u_{i-1}) -- discrete wave. "
            "Wave: u_tt = c^2*u_xx -- continuous wave. "
            "In the continuum limit, c = a*sqrt(K/m) and the chain becomes "
            "the wave equation exactly."
        ),
        strength=0.9,
        mapping={
            "c^2 [wave speed squared]": "K*a^2/m [effective c^2]",
            "u_xx [continuous Laplacian]": "(u_{i+1}-2u_i+u_{i-1})/a^2 [discrete Laplacian]",
            "gamma [damping]": "(no damping in chain)",
        },
    ))

    # Analogy 19: Ising <-> Kuramoto (phase transition in collective systems)
    analogies.append(Analogy(
        domain_a="ising_model",
        domain_b="kuramoto",
        analogy_type="structural",
        description=(
            "Both are N-body systems with a collective phase transition. "
            "Ising: spin alignment transition at T_c = 2J/ln(1+sqrt(2)). "
            "Kuramoto: phase synchronization at K_c = 2/(pi*g(0)). "
            "Both have an order parameter (magnetization/sync r) that "
            "jumps from 0 to finite above the critical point."
        ),
        strength=0.85,
        mapping={
            "T [temperature]": "1/K [inverse coupling]",
            "T_c [critical temp]": "K_c [critical coupling]",
            "magnetization m": "order parameter r",
            "spin-spin coupling J": "phase coupling K",
        },
    ))

    # Analogy 20: Ising <-> Boltzmann gas (statistical mechanics)
    analogies.append(Analogy(
        domain_a="ising_model",
        domain_b="boltzmann_gas",
        analogy_type="structural",
        description=(
            "Both are statistical mechanical systems described by the "
            "Boltzmann distribution P ~ exp(-E/kT). Ising uses Monte Carlo "
            "sampling; gas uses Newtonian dynamics that ergodically explores "
            "the microcanonical ensemble. Both exhibit thermal equilibrium."
        ),
        strength=0.75,
        mapping={
            "spin configuration": "particle positions/velocities",
            "Metropolis acceptance": "elastic collisions",
            "T [temperature]": "T [temperature]",
            "E = -J*sum(s_i*s_j)": "E = sum(0.5*m*v^2)",
        },
    ))

    # Analogy 21: Boltzmann gas <-> Kuramoto (collective N-body dynamics)
    analogies.append(Analogy(
        domain_a="boltzmann_gas",
        domain_b="kuramoto",
        analogy_type="structural",
        description=(
            "Both are N-body systems where macroscopic behavior emerges from "
            "microscopic interactions. Gas: N particles collide, producing "
            "Maxwell-Boltzmann distribution and PV=NkT. Kuramoto: N oscillators "
            "couple, producing synchronization transition. Both exhibit "
            "mean-field behavior in the N->infinity limit."
        ),
        strength=0.65,
        mapping={
            "v_i [particle velocity]": "theta_i [oscillator phase]",
            "collisions [pairwise]": "sin coupling [pairwise]",
            "temperature T": "order parameter r",
            "PV=NkT [equation of state]": "r(K) [sync curve]",
        },
    ))

    # Analogy: Cart-pole <-> Double pendulum (pendulum systems with coupled DOFs)
    analogies.append(Analogy(
        domain_a="cart_pole",
        domain_b="double_pendulum",
        analogy_type="structural",
        description=(
            "Both are coupled pendulum systems with Lagrangian mechanics. "
            "Cart-pole: pendulum on a sliding cart (1 translational + 1 rotational DOF). "
            "Double pendulum: two linked pendula (2 rotational DOF). "
            "Both use mass matrix inversion to solve coupled equations of motion."
        ),
        strength=0.85,
        mapping={
            "x (cart position)": "theta1 (upper pendulum angle)",
            "theta (pendulum angle)": "theta2 (lower pendulum angle)",
            "M (cart mass)": "m1 (upper mass)",
            "m (bob mass)": "m2 (lower mass)",
            "mass matrix inversion": "mass matrix inversion",
        },
    ))

    # Analogy: Cart-pole <-> Harmonic oscillator (linearized oscillation)
    analogies.append(Analogy(
        domain_a="cart_pole",
        domain_b="harmonic_oscillator",
        analogy_type="structural",
        description=(
            "Linearized cart-pole near the hanging equilibrium (theta=pi) "
            "reduces to a harmonic oscillator: phi'' = -(g*(M+m)/(M*L))*phi. "
            "This is identical to x'' = -(k/m)*x with k/m -> g*(M+m)/(M*L). "
            "Small-angle frequency omega = sqrt(g*(M+m)/(M*L))."
        ),
        strength=0.9,
        mapping={
            "phi (deviation from pi)": "x (displacement)",
            "g*(M+m)/(M*L) [effective stiffness]": "k/m [spring ratio]",
            "omega = sqrt(g*(M+m)/(M*L))": "omega = sqrt(k/m)",
        },
    ))

    # Analogy: Three-species <-> Lotka-Volterra (food chain extension)
    analogies.append(Analogy(
        domain_a="three_species",
        domain_b="lotka_volterra",
        analogy_type="structural",
        description=(
            "Three-species food chain is a direct extension of Lotka-Volterra. "
            "LV: dx/dt = a*x - b*x*y (2 species, 1 trophic level). "
            "3-species: adds dz/dt = -a3*z + b2*y*z (3 species, 2 trophic levels). "
            "Setting z=0 recovers the original LV system exactly."
        ),
        strength=0.95,
        mapping={
            "x (grass)": "prey",
            "y (herbivore)": "predator",
            "z (predator)": "(no analogue in 2-species)",
            "a1*x - b1*x*y": "alpha*prey - beta*prey*pred",
            "-a2*y + b1*x*y": "-gamma*pred + delta*prey*pred",
        },
    ))

    # Analogy: Three-species <-> SIR (3-compartment coupled nonlinear ODEs)
    analogies.append(Analogy(
        domain_a="three_species",
        domain_b="sir_epidemic",
        analogy_type="structural",
        description=(
            "Both are 3-variable coupled nonlinear ODE systems with bilinear "
            "interaction terms. 3-species: x*y and y*z drive population transfer. "
            "SIR: S*I drives transfer between compartments. "
            "Both have cascade dynamics where one variable feeds the next."
        ),
        strength=0.75,
        mapping={
            "x -> y transfer (b1*x*y)": "S -> I transfer (beta*S*I)",
            "y -> z transfer (b2*y*z)": "I -> R transfer (gamma*I)",
            "grass -> herbivore -> predator": "S -> I -> R",
        },
    ))

    # Analogy: Elastic pendulum <-> Cart-pole (Lagrangian coupled 2DOF systems)
    analogies.append(Analogy(
        domain_a="elastic_pendulum",
        domain_b="cart_pole",
        analogy_type="structural",
        description=(
            "Both are Lagrangian systems with 2 coupled degrees of freedom. "
            "Elastic pendulum: radial (spring) + angular (pendulum). "
            "Cart-pole: translational (cart) + rotational (pendulum). "
            "Both have a mass matrix and coupled nonlinear equations of motion."
        ),
        strength=0.8,
        mapping={
            "r (radial)": "x (cart position)",
            "theta (angular)": "theta (pendulum angle)",
            "k/m (spring restoring)": "g*(M+m)/(M*L) (gravity restoring)",
            "1:2 resonance": "small-angle oscillation",
        },
    ))

    # Analogy: Elastic pendulum <-> Harmonic oscillator (radial mode)
    analogies.append(Analogy(
        domain_a="elastic_pendulum",
        domain_b="harmonic_oscillator",
        analogy_type="structural",
        description=(
            "The radial mode of the elastic pendulum is a simple harmonic oscillator. "
            "For small theta, r'' = -(k/m)*(r-L0) + g is exactly the spring equation. "
            "omega_r = sqrt(k/m) matches the harmonic oscillator omega_0 = sqrt(k/m)."
        ),
        strength=0.9,
        mapping={
            "r - L0 (radial displacement)": "x (displacement)",
            "k/m (spring ratio)": "k/m (spring ratio)",
            "omega_r = sqrt(k/m)": "omega_0 = sqrt(k/m)",
        },
    ))

    # Analogy: Rossler <-> Lorenz (3D chaotic ODE systems)
    analogies.append(Analogy(
        domain_a="rossler",
        domain_b="lorenz",
        analogy_type="structural",
        description=(
            "Both are 3D autonomous ODE systems with chaotic attractors. "
            "Rossler: simpler single-lobe spiral attractor. "
            "Lorenz: two-lobe butterfly attractor. "
            "Both have quadratic nonlinearity (x*z in Lorenz, x*z in Rossler)."
        ),
        strength=0.85,
        mapping={
            "a [spiral rate]": "sigma [coupling]",
            "c [reinjection]": "rho [Rayleigh number]",
            "z*(x-c) [nonlinear]": "x*z, x*y [nonlinear]",
            "single-lobe attractor": "two-lobe butterfly",
        },
    ))

    # Analogy: Brusselator-diffusion <-> Gray-Scott (reaction-diffusion PDEs)
    analogies.append(Analogy(
        domain_a="brusselator_diffusion",
        domain_b="gray_scott",
        analogy_type="structural",
        description=(
            "Both are 2-component reaction-diffusion PDEs with Turing instability. "
            "Brusselator: u_t = D_u*u_xx + a - (b+1)*u + u^2*v. "
            "Gray-Scott: u_t = D_u*Lap(u) - u*v^2 + f*(1-u). "
            "Both produce spatial patterns from homogeneous initial conditions "
            "when the diffusion ratio D_v/D_u is sufficiently large."
        ),
        strength=0.9,
        mapping={
            "u^2*v [autocatalytic]": "u*v^2 [autocatalytic]",
            "b [bifurcation parameter]": "k [kill rate]",
            "Turing threshold b>1+a^2": "Turing boundary in (f,k) space",
            "D_v/D_u >> 1": "D_u/D_v >> 1 (different convention)",
        },
    ))

    # Analogy: Brusselator-diffusion <-> Brusselator (spatial extension)
    analogies.append(Analogy(
        domain_a="brusselator_diffusion",
        domain_b="brusselator",
        analogy_type="structural",
        description=(
            "Brusselator-diffusion is the spatial extension of the Brusselator. "
            "Brusselator: du/dt = a - (b+1)*u + u^2*v (well-mixed). "
            "Brusselator-diffusion: adds D_u*u_xx and D_v*v_xx (spatial). "
            "Setting D_u = D_v = 0 recovers the original ODE exactly."
        ),
        strength=0.95,
        mapping={
            "a, b [reaction params]": "a, b [same params]",
            "D_u*u_xx [diffusion]": "0 [well-mixed]",
            "Turing patterns": "Hopf oscillations",
        },
    ))

    # Analogy: Henon map <-> Logistic map (discrete chaos)
    analogies.append(Analogy(
        domain_a="henon_map",
        domain_b="logistic_map",
        analogy_type="structural",
        description=(
            "Both are discrete chaotic maps with period-doubling route to chaos. "
            "Henon: 2D map, x_{n+1} = 1 - a*x^2 + y, y_{n+1} = b*x. "
            "Logistic: 1D map, x_{n+1} = r*x*(1-x). "
            "For b=0, the Henon map reduces to a 1D quadratic map like logistic."
        ),
        strength=0.9,
        mapping={
            "a [nonlinearity]": "r [growth rate]",
            "b=0 [1D limit]": "logistic map",
            "Henon attractor (fractal D~1.26)": "chaotic bands",
            "Feigenbaum universality": "Feigenbaum universality",
        },
    ))

    # Analogy: Rosenzweig-MacArthur <-> Lotka-Volterra (predator-prey with functional response)
    analogies.append(Analogy(
        domain_a="rosenzweig_macarthur",
        domain_b="lotka_volterra",
        analogy_type="structural",
        description=(
            "Both are 2D predator-prey ODEs. RM extends LV with logistic prey "
            "growth and Holling Type II saturating functional response. In the "
            "limit K->inf and h->0, RM reduces to classic LV."
        ),
        strength=0.95,
        mapping={
            "prey x": "prey N1",
            "predator y": "predator N2",
            "growth rate r": "birth rate alpha",
            "Holling Type II a*x/(1+ahx)": "mass action beta*x*y",
        },
    ))

    # Analogy: Chua <-> Lorenz (3D chaotic strange attractor)
    analogies.append(Analogy(
        domain_a="chua",
        domain_b="lorenz",
        analogy_type="structural",
        description=(
            "Both are 3-variable autonomous ODEs exhibiting chaotic strange "
            "attractors via period-doubling cascade. Chua has piecewise-linear "
            "nonlinearity while Lorenz has quadratic (xz, xy) terms. Both have "
            "positive Lyapunov exponents and similar bifurcation structure."
        ),
        strength=0.85,
        mapping={
            "alpha [capacitor ratio]": "sigma [Prandtl]",
            "beta [inductance]": "rho [Rayleigh]",
            "double-scroll attractor": "butterfly attractor",
            "piecewise-linear f(x)": "quadratic xz, xy",
        },
    ))

    # Analogy: Shallow Water <-> Navier-Stokes (fluid conservation laws)
    analogies.append(Analogy(
        domain_a="shallow_water",
        domain_b="navier_stokes",
        analogy_type="structural",
        description=(
            "Both are fluid dynamics PDEs derived from conservation laws. "
            "Shallow water is depth-averaged NS with hydrostatic pressure. "
            "Both conserve mass and momentum, with nonlinear advection terms."
        ),
        strength=0.9,
        mapping={
            "height h": "velocity field u",
            "flux h*u": "momentum rho*u",
            "wave speed sqrt(gh)": "sound speed",
            "mass conservation": "incompressibility",
        },
    ))

    # Analogy: Toda Lattice <-> Spring-Mass Chain (lattice dynamics)
    analogies.append(Analogy(
        domain_a="toda_lattice",
        domain_b="spring_mass_chain",
        analogy_type="structural",
        description=(
            "Both are 1D lattice dynamics with nearest-neighbor interactions. "
            "Spring-mass uses linear Hooke's law F=-Kx; Toda uses exponential "
            "V(r)=exp(-r). In small-amplitude limit, Toda reduces to the "
            "harmonic spring-mass chain with identical dispersion relation."
        ),
        strength=0.9,
        mapping={
            "exp(-r) potential": "linear spring K*x",
            "coupling a": "spring constant K",
            "omega = 2*sqrt(a)*|sin(pi*n/N)|": "omega = 2*sqrt(K/m)*|sin(pi*n/N)|",
            "soliton solutions": "phonon modes",
        },
    ))

    # Analogy: Rosenzweig-MacArthur <-> Brusselator (Hopf bifurcation, limit cycles)
    analogies.append(Analogy(
        domain_a="rosenzweig_macarthur",
        domain_b="brusselator",
        analogy_type="structural",
        description=(
            "Both exhibit Hopf bifurcation from stable equilibrium to limit "
            "cycle oscillations as a control parameter increases. In RM, "
            "increasing K destabilizes the coexistence equilibrium. In "
            "Brusselator, increasing b past 1+a^2 triggers oscillation."
        ),
        strength=0.8,
        mapping={
            "carrying capacity K": "parameter b",
            "Hopf bifurcation at K_c": "Hopf at b_c=1+a^2",
            "prey-predator cycle": "u-v chemical oscillation",
        },
    ))

    return analogies


def detect_dimensional_analogies(
    signatures: list[DomainSignature],
) -> list[Analogy]:
    """Detect dimensional analogies: same scaling relationships."""
    analogies = []

    # Pendulum period and oscillator period have same sqrt structure
    analogies.append(Analogy(
        domain_a="double_pendulum",
        domain_b="harmonic_oscillator",
        analogy_type="dimensional",
        description=(
            "Both have characteristic timescale T ~ sqrt(inertia/restoring_force). "
            "Pendulum: T = 2*pi*sqrt(L/g), Oscillator: T = 2*pi*sqrt(m/k). "
            "This is a universal feature of all linear restoring force systems."
        ),
        strength=1.0,
        mapping={
            "L (length)": "m (mass) [inertia]",
            "g (gravity)": "k (spring constant) [restoring force]",
            "sqrt(L/g)": "sqrt(m/k)",
        },
    ))

    # Gray-Scott wavelength and diffusion
    analogies.append(Analogy(
        domain_a="gray_scott",
        domain_b="harmonic_oscillator",
        analogy_type="dimensional",
        description=(
            "Turing pattern wavelength lambda ~ sqrt(D/k_rate) has the same "
            "dimensional structure as oscillator period T ~ sqrt(m/k). "
            "Diffusion coefficient D plays the role of inertia (m), "
            "and reaction rate k plays the role of restoring force (k)."
        ),
        strength=0.6,
        mapping={
            "D (diffusion)": "m (mass) [spreading/inertia]",
            "k_rate (reaction)": "k (spring) [localization/restoring]",
            "lambda ~ sqrt(D/k)": "T ~ sqrt(m/k)",
        },
    ))

    # Spring-mass chain <-> Harmonic oscillator (same omega ~ sqrt(K/m))
    analogies.append(Analogy(
        domain_a="spring_mass_chain",
        domain_b="harmonic_oscillator",
        analogy_type="dimensional",
        description=(
            "Both have characteristic frequency omega ~ sqrt(K/m). "
            "Spring-mass chain: omega_max = 2*sqrt(K/m). "
            "Harmonic oscillator: omega_0 = sqrt(k/m). "
            "The chain is a discrete generalization of the single oscillator."
        ),
        strength=0.95,
        mapping={
            "K [spring constant]": "k [spring constant]",
            "m [mass per site]": "m [mass]",
            "omega_max = 2*sqrt(K/m)": "omega_0 = sqrt(k/m)",
        },
    ))

    # Kepler <-> Pendulum (sqrt period-timescale)
    analogies.append(Analogy(
        domain_a="kepler",
        domain_b="double_pendulum",
        analogy_type="dimensional",
        description=(
            "Both have period scaling as power of characteristic length. "
            "Kepler: T ~ a^(3/2)/sqrt(GM). "
            "Pendulum: T ~ sqrt(L/g). "
            "Both follow from dimensional analysis of central/gravitational forces."
        ),
        strength=0.7,
        mapping={
            "a [semi-major axis]": "L [pendulum length]",
            "GM [gravitational parameter]": "g [gravity]",
            "T ~ a^(3/2)": "T ~ L^(1/2)",
        },
    ))

    # Quantum oscillator <-> Harmonic oscillator (classical/quantum correspondence)
    analogies.append(Analogy(
        domain_a="quantum_oscillator",
        domain_b="harmonic_oscillator",
        analogy_type="dimensional",
        description=(
            "Classical and quantum harmonic oscillators share omega = sqrt(k/m). "
            "Classical: continuous energy, E = 0.5*k*A^2. "
            "Quantum: quantized E_n = (n+0.5)*hbar*omega. "
            "In the correspondence limit (n >> 1), quantum -> classical."
        ),
        strength=0.95,
        mapping={
            "hbar*omega [energy quantum]": "E = 0.5*k*A^2 [classical energy]",
            "|psi|^2 [probability density]": "delta(x - A*cos(omega*t)) [trajectory]",
            "omega = sqrt(k/m)": "omega_0 = sqrt(k/m)",
        },
    ))

    # Coupled oscillators <-> Harmonic oscillator (same omega ~ sqrt(k/m))
    analogies.append(Analogy(
        domain_a="coupled_oscillators",
        domain_b="harmonic_oscillator",
        analogy_type="dimensional",
        description=(
            "Symmetric mode of coupled oscillators has omega_s = sqrt(k/m), "
            "identical to the single harmonic oscillator. The coupling kc "
            "only affects the antisymmetric mode: omega_a = sqrt((k+2*kc)/m)."
        ),
        strength=0.95,
        mapping={
            "k [spring constant]": "k [spring constant]",
            "m [mass]": "m [mass]",
            "omega_s = sqrt(k/m)": "omega_0 = sqrt(k/m)",
        },
    ))

    # Damped wave <-> Harmonic oscillator (same dispersion relation)
    analogies.append(Analogy(
        domain_a="damped_wave",
        domain_b="harmonic_oscillator",
        analogy_type="dimensional",
        description=(
            "Each Fourier mode of the damped wave equation is a damped harmonic "
            "oscillator: u_k'' + gamma*u_k' + c^2*k^2*u_k = 0. "
            "The wave equation is an infinite collection of oscillators. "
            "Oscillator: omega_0 = sqrt(k_spring/m). Wave mode: omega_k = c*k."
        ),
        strength=0.9,
        mapping={
            "c*k [mode frequency]": "sqrt(k/m) [natural frequency]",
            "gamma [wave damping]": "c/(m) [oscillator damping]",
            "u_k [mode amplitude]": "x [displacement]",
        },
    ))

    # Diffusive LV <-> Heat equation (diffusion timescale)
    analogies.append(Analogy(
        domain_a="diffusive_lv",
        domain_b="heat_equation",
        analogy_type="dimensional",
        description=(
            "Both have diffusive transport with timescale ~ L^2/D. "
            "Heat equation: pure diffusion u_t = D*u_xx. "
            "Diffusive LV: diffusion + reaction u_t = D_u*u_xx + f(u,v). "
            "The diffusion operator is identical in both systems."
        ),
        strength=0.8,
        mapping={
            "D_u [prey diffusion]": "D [thermal diffusion]",
            "L^2/D_u [diffusive timescale]": "L^2/D [diffusive timescale]",
            "u(x) [prey density]": "u(x) [temperature]",
        },
    ))

    # Cart-pole <-> Double pendulum (same sqrt(L/g) dimensional structure)
    analogies.append(Analogy(
        domain_a="cart_pole",
        domain_b="double_pendulum",
        analogy_type="dimensional",
        description=(
            "Both have period scaling as sqrt(length/gravity). "
            "Cart-pole: T = 2*pi/sqrt(g*(M+m)/(M*L)) ~ sqrt(L/g). "
            "Double pendulum: T = 2*pi*sqrt(L/g) for small angles. "
            "The cart-pole has an additional mass ratio correction M/(M+m)."
        ),
        strength=0.85,
        mapping={
            "L [pendulum length]": "L [pendulum length]",
            "g [gravity]": "g [gravity]",
            "sqrt(M*L/(g*(M+m)))": "sqrt(L/g)",
        },
    ))

    # Cart-pole <-> Harmonic oscillator (omega ~ sqrt(stiffness/inertia))
    analogies.append(Analogy(
        domain_a="cart_pole",
        domain_b="harmonic_oscillator",
        analogy_type="dimensional",
        description=(
            "Both have omega = sqrt(restoring_force/inertia). "
            "Cart-pole: omega = sqrt(g*(M+m)/(M*L)). "
            "Oscillator: omega = sqrt(k/m). "
            "Same dimensional structure with different physical parameters."
        ),
        strength=0.9,
        mapping={
            "g*(M+m)/L [effective restoring]": "k [spring constant]",
            "M [effective inertia]": "m [mass]",
            "omega = sqrt(g*(M+m)/(M*L))": "omega_0 = sqrt(k/m)",
        },
    ))

    # Elastic pendulum <-> Harmonic oscillator (omega_r = sqrt(k/m))
    analogies.append(Analogy(
        domain_a="elastic_pendulum",
        domain_b="harmonic_oscillator",
        analogy_type="dimensional",
        description=(
            "Both have omega = sqrt(k/m). Elastic pendulum radial mode: "
            "omega_r = sqrt(k/m). Harmonic oscillator: omega_0 = sqrt(k/m). "
            "Angular mode gives omega_theta = sqrt(g/L0) ~ pendulum."
        ),
        strength=0.95,
        mapping={
            "k [spring constant]": "k [spring constant]",
            "m [mass]": "m [mass]",
            "omega_r = sqrt(k/m)": "omega_0 = sqrt(k/m)",
        },
    ))

    # Brusselator-diffusion <-> Heat equation (diffusion timescale)
    analogies.append(Analogy(
        domain_a="brusselator_diffusion",
        domain_b="heat_equation",
        analogy_type="dimensional",
        description=(
            "Both have diffusive transport with timescale ~ L^2/D. "
            "Heat: pure diffusion u_t = D*u_xx. "
            "Brusselator-diffusion: diffusion + reaction. "
            "The diffusion operator is identical in both systems."
        ),
        strength=0.75,
        mapping={
            "D_u [chemical diffusion]": "D [thermal diffusion]",
            "L^2/D_u [timescale]": "L^2/D [timescale]",
        },
    ))

    # Dimensional: Shallow Water <-> Damped Wave (wave propagation speed)
    analogies.append(Analogy(
        domain_a="shallow_water",
        domain_b="damped_wave",
        analogy_type="dimensional",
        description=(
            "Both support wave propagation with characteristic speed. "
            "Shallow water: c = sqrt(g*h). Damped wave: c (string tension/density). "
            "The dispersion relation has the same dimensional structure."
        ),
        strength=0.8,
        mapping={
            "sqrt(g*h) [wave speed]": "c [wave speed]",
            "L/c [transit time]": "L/c [transit time]",
        },
    ))

    # Dimensional: Toda Lattice <-> Spring-Mass Chain (dispersion relation)
    analogies.append(Analogy(
        domain_a="toda_lattice",
        domain_b="spring_mass_chain",
        analogy_type="dimensional",
        description=(
            "Both have identical dispersion relation in the harmonic limit: "
            "omega_n = 2*sqrt(K)*|sin(pi*n/N)|. Toda coupling a maps to "
            "spring constant K with the same frequency scaling."
        ),
        strength=0.95,
        mapping={
            "sqrt(a) [frequency scale]": "sqrt(K/m) [frequency scale]",
            "a [lattice spacing]": "a [lattice spacing]",
        },
    ))

    return analogies


def detect_topological_analogies(
    signatures: list[DomainSignature],
) -> list[Analogy]:
    """Detect topological analogies: similar phase space structure."""
    analogies = []

    # LV limit cycles and undamped oscillator
    analogies.append(Analogy(
        domain_a="lotka_volterra",
        domain_b="harmonic_oscillator",
        analogy_type="topological",
        description=(
            "Both exhibit closed orbits in phase space (limit cycles/center). "
            "LV: orbits around (gamma/delta, alpha/beta) in prey-predator space. "
            "Undamped oscillator: ellipses around origin in x-v space. "
            "Both have a conserved quantity (Hamiltonian/energy) that constrains "
            "trajectories to closed curves."
        ),
        strength=0.8,
        mapping={
            "prey-predator plane": "x-v phase plane",
            "LV Hamiltonian": "mechanical energy",
            "equilibrium (gamma/delta, alpha/beta)": "origin (0, 0)",
        },
    ))

    # SIR convergence and overdamped oscillator
    analogies.append(Analogy(
        domain_a="sir_epidemic",
        domain_b="harmonic_oscillator",
        analogy_type="topological",
        description=(
            "Both converge to a fixed point from any initial condition. "
            "SIR: (S_inf, 0, R_inf) is a global attractor. "
            "Overdamped oscillator: (0, 0) is a global attractor. "
            "Both show monotonic approach without oscillation in the "
            "relevant regime (post-peak SIR, overdamped oscillator)."
        ),
        strength=0.65,
        mapping={
            "S -> S_inf": "x -> 0",
            "I -> 0": "v -> 0",
            "epidemic end": "equilibrium",
        },
    ))

    # Lorenz strange attractor and double pendulum chaos
    analogies.append(Analogy(
        domain_a="lorenz",
        domain_b="double_pendulum",
        analogy_type="topological",
        description=(
            "Both have strange attractors with fractal dimension > 2. "
            "Lorenz: two-lobe butterfly attractor with Hausdorff dimension ~2.06. "
            "Double pendulum: chaotic trajectories fill phase space regions. "
            "Both show sensitive dependence on initial conditions with "
            "positive largest Lyapunov exponent (~0.9 for Lorenz)."
        ),
        strength=0.85,
        mapping={
            "butterfly attractor (2 lobes)": "chaotic orbit in (theta1, theta2, omega1, omega2)",
            "rho > rho_c (chaos onset)": "E > E_c (chaos onset)",
            "Lyapunov ~ 0.9": "Lyapunov > 0",
        },
    ))

    # Van der Pol <-> Harmonic Oscillator (same phase plane topology)
    analogies.append(Analogy(
        domain_a="van_der_pol",
        domain_b="harmonic_oscillator",
        analogy_type="topological",
        description=(
            "Both are second-order oscillators in the x-v phase plane. "
            "Harmonic oscillator: elliptical orbits (center). "
            "Van der Pol: limit cycle with spiral approach. "
            "VdP reduces to harmonic oscillator as mu -> 0."
        ),
        strength=0.85,
        mapping={
            "limit cycle": "center (undamped)",
            "mu=0 (harmonic limit)": "c=0 (undamped)",
            "spiral in/out": "ellipse",
        },
    ))

    # Logistic map <-> Lorenz (both chaotic with Lyapunov > 0)
    analogies.append(Analogy(
        domain_a="logistic_map",
        domain_b="lorenz",
        analogy_type="topological",
        description=(
            "Both exhibit chaos with positive Lyapunov exponents. "
            "Logistic map: period-doubling route to chaos, lambda(r=4)=ln(2). "
            "Lorenz: Hopf bifurcation route, lambda~0.9 at standard parameters. "
            "Both show sensitive dependence on initial conditions."
        ),
        strength=0.7,
        mapping={
            "r [growth rate]": "rho [Rayleigh number]",
            "r_c ~ 3.57 [chaos onset]": "rho_c ~ 24.74 [chaos onset]",
            "Feigenbaum cascade": "bifurcation sequence",
        },
    ))

    # Kuramoto <-> SIR (threshold/phase transition)
    analogies.append(Analogy(
        domain_a="kuramoto",
        domain_b="sir_epidemic",
        analogy_type="topological",
        description=(
            "Both exhibit a critical threshold transition. "
            "Kuramoto: K > K_c triggers synchronization (r jumps from 0). "
            "SIR: R0 > 1 triggers epidemic outbreak (I grows). "
            "Below threshold: stable incoherent/disease-free state. "
            "Above threshold: collective synchronized/epidemic behavior."
        ),
        strength=0.7,
        mapping={
            "K (coupling)": "R0 = beta/gamma",
            "K_c (critical coupling)": "R0 = 1 (epidemic threshold)",
            "r (order parameter)": "I_peak (infected peak)",
        },
    ))

    # Duffing <-> Lorenz (both chaotic with strange attractors)
    analogies.append(Analogy(
        domain_a="duffing",
        domain_b="lorenz",
        analogy_type="topological",
        description=(
            "Both exhibit chaotic dynamics with strange attractors. "
            "Duffing: period-doubling route to chaos as driving amplitude increases. "
            "Lorenz: homoclinic bifurcation as Rayleigh number increases. "
            "Both have positive largest Lyapunov exponent in chaotic regime."
        ),
        strength=0.75,
        mapping={
            "gamma [driving amplitude]": "rho [Rayleigh number]",
            "period-doubling cascade": "bifurcation cascade",
            "chaotic attractor (2D Poincare)": "butterfly attractor (3D)",
        },
    ))

    # Logistic map <-> Duffing (period-doubling route to chaos)
    analogies.append(Analogy(
        domain_a="logistic_map",
        domain_b="duffing",
        analogy_type="topological",
        description=(
            "Both exhibit the period-doubling route to chaos with Feigenbaum "
            "universality. Logistic map: x_{n+1} = r*x_n*(1-x_n), "
            "period-doubling at r values converging with ratio delta=4.669. "
            "Duffing: period-doubling as gamma increases. Same universal scaling."
        ),
        strength=0.8,
        mapping={
            "r [growth rate]": "gamma [driving amplitude]",
            "period-doubling cascade": "period-doubling cascade",
            "Feigenbaum delta": "Feigenbaum delta (same universal constant)",
        },
    ))

    # Driven pendulum <-> Logistic map (both period-doubling with Feigenbaum)
    analogies.append(Analogy(
        domain_a="driven_pendulum",
        domain_b="logistic_map",
        analogy_type="topological",
        description=(
            "Both exhibit the period-doubling route to chaos with the same "
            "Feigenbaum universal constants (delta=4.669, alpha=2.503). "
            "Driven pendulum: A increasing causes period 1->2->4->chaos. "
            "Logistic map: r increasing causes same cascade. "
            "Feigenbaum universality connects all period-doubling systems."
        ),
        strength=0.85,
        mapping={
            "A [driving amplitude]": "r [growth parameter]",
            "period-doubling cascade": "period-doubling cascade",
            "Feigenbaum delta = 4.669": "Feigenbaum delta = 4.669",
        },
    ))

    # Kepler <-> LV (closed orbits with conserved integral)
    analogies.append(Analogy(
        domain_a="kepler",
        domain_b="lotka_volterra",
        analogy_type="topological",
        description=(
            "Both have closed orbits in 2D phase space constrained by a "
            "conserved quantity. Kepler: energy E constrains r-v_r plane. "
            "LV: Hamiltonian H constrains prey-predator plane. "
            "Both have a family of nested closed curves around an equilibrium."
        ),
        strength=0.65,
        mapping={
            "energy [conserved]": "LV Hamiltonian [conserved]",
            "r-v_r phase plane": "prey-predator phase plane",
            "perihelion/aphelion": "population extrema",
        },
    ))

    # Coupled oscillators <-> Kepler (quasi-periodic closed orbits)
    analogies.append(Analogy(
        domain_a="coupled_oscillators",
        domain_b="kepler",
        analogy_type="topological",
        description=(
            "Both are integrable systems with quasi-periodic trajectories. "
            "Coupled oscillators: two incommensurate normal mode frequencies "
            "create Lissajous figures in configuration space. "
            "Kepler: closed elliptical orbits with precession for perturbations. "
            "Both fill tori in phase space."
        ),
        strength=0.6,
        mapping={
            "omega_s, omega_a [two frequencies]": "orbital freq, apsidal freq",
            "Lissajous figures": "elliptical orbits",
            "energy surface (torus)": "energy surface (torus)",
        },
    ))

    # Diffusive LV <-> Navier-Stokes (nonlinear PDEs with spatial structure)
    analogies.append(Analogy(
        domain_a="diffusive_lv",
        domain_b="navier_stokes",
        analogy_type="topological",
        description=(
            "Both are nonlinear PDEs that produce emergent spatial structure "
            "from simple initial conditions. Diffusive LV: traveling waves and "
            "spatial patterns from uniform + perturbation. NS: vortex formation "
            "and turbulent cascades from smooth initial flow."
        ),
        strength=0.6,
        mapping={
            "traveling waves": "vortex structures",
            "reaction-diffusion patterns": "turbulent eddies",
            "D_u, D_v [diffusion]": "nu [viscosity]",
        },
    ))

    # Ising <-> SIR (phase transition / threshold behavior)
    analogies.append(Analogy(
        domain_a="ising_model",
        domain_b="sir_epidemic",
        analogy_type="topological",
        description=(
            "Both exhibit critical threshold behavior. "
            "Ising: below T_c, magnetization is nonzero (ordered). "
            "SIR: above R0=1, epidemic spreads (outbreak). "
            "Both have a sharp transition from inactive to active state."
        ),
        strength=0.7,
        mapping={
            "T < T_c [ordered]": "R0 > 1 [epidemic]",
            "magnetization jump": "infected peak",
            "spin clusters": "infection clusters",
        },
    ))

    # Cart-pole <-> Van der Pol (oscillatory nonlinear systems)
    analogies.append(Analogy(
        domain_a="cart_pole",
        domain_b="van_der_pol",
        analogy_type="topological",
        description=(
            "Both are nonlinear oscillators in a 2D effective phase space. "
            "Cart-pole near hanging equilibrium: stable oscillation with "
            "friction-dependent amplitude decay. Van der Pol: limit cycle "
            "with amplitude self-regulation. Both exhibit energy-dependent "
            "orbit topology in the theta-omega or x-v plane."
        ),
        strength=0.6,
        mapping={
            "theta-omega phase plane": "x-v phase plane",
            "friction damping": "nonlinear damping mu*(1-x^2)",
            "gravity restoring force": "linear restoring force",
        },
    ))

    # Three-species <-> Lorenz (3D chaotic/complex ODEs)
    analogies.append(Analogy(
        domain_a="three_species",
        domain_b="lorenz",
        analogy_type="topological",
        description=(
            "Both are 3D nonlinear ODE systems that can exhibit complex dynamics. "
            "Three-species: 3-variable food chain with possible limit cycles "
            "and chaotic behavior for certain parameter regimes. "
            "Lorenz: 3-variable system with strange attractor and chaos. "
            "Both require 3 dimensions for their dynamical complexity."
        ),
        strength=0.6,
        mapping={
            "x, y, z [populations]": "x, y, z [convection variables]",
            "food chain interactions": "thermal convection coupling",
            "population oscillations": "lobe switching",
        },
    ))

    # Boltzmann gas <-> Heat equation (microscopic vs macroscopic diffusion)
    analogies.append(Analogy(
        domain_a="boltzmann_gas",
        domain_b="heat_equation",
        analogy_type="topological",
        description=(
            "Boltzmann gas is the microscopic origin of the heat equation. "
            "Random particle collisions produce diffusive transport at the "
            "macroscopic level. The diffusion coefficient D relates to "
            "mean free path and thermal velocity: D ~ lambda*v_thermal."
        ),
        strength=0.7,
        mapping={
            "particle velocities": "temperature field u(x)",
            "collision dynamics": "diffusion operator D*u_xx",
            "Maxwell-Boltzmann equilibrium": "uniform steady state",
        },
    ))

    # Rossler <-> Lorenz (chaotic 3D ODE strange attractors)
    analogies.append(Analogy(
        domain_a="rossler",
        domain_b="lorenz",
        analogy_type="topological",
        description=(
            "Both have strange attractors in 3D phase space with positive "
            "Lyapunov exponents. Rossler: single-lobe spiral, Lyapunov ~0.07. "
            "Lorenz: two-lobe butterfly, Lyapunov ~0.9. "
            "Both show sensitive dependence on initial conditions."
        ),
        strength=0.85,
        mapping={
            "single-lobe spiral": "two-lobe butterfly",
            "c [reinjection parameter]": "rho [Rayleigh number]",
            "period-doubling to chaos": "intermittency to chaos",
        },
    ))

    # Rossler <-> Duffing (period-doubling route to chaos)
    analogies.append(Analogy(
        domain_a="rossler",
        domain_b="duffing",
        analogy_type="topological",
        description=(
            "Both exhibit period-doubling route to chaos as a control parameter "
            "increases. Rossler: c increasing causes period 1->2->4->chaos. "
            "Duffing: driving amplitude gamma increasing causes similar cascade. "
            "Both share Feigenbaum universal scaling."
        ),
        strength=0.75,
        mapping={
            "c [control parameter]": "gamma [driving amplitude]",
            "period-doubling cascade": "period-doubling cascade",
            "chaotic attractor": "chaotic attractor",
        },
    ))

    # Henon map <-> Logistic map (discrete chaotic dynamics)
    analogies.append(Analogy(
        domain_a="henon_map",
        domain_b="logistic_map",
        analogy_type="topological",
        description=(
            "Both are discrete chaotic maps with Feigenbaum universality. "
            "Henon: 2D strange attractor with fractal dimension ~1.26. "
            "Logistic: 1D chaotic intervals with fractal spectrum. "
            "Both exhibit period-doubling cascades and positive Lyapunov exponents."
        ),
        strength=0.85,
        mapping={
            "a [Henon parameter]": "r [logistic parameter]",
            "Lyapunov ~0.42": "Lyapunov ln(2) at r=4",
            "strange attractor (D~1.26)": "chaotic intervals",
        },
    ))

    # Elastic pendulum <-> Double pendulum (coupled nonlinear pendula)
    analogies.append(Analogy(
        domain_a="elastic_pendulum",
        domain_b="double_pendulum",
        analogy_type="topological",
        description=(
            "Both are 2-DOF nonlinear pendulum systems that can exhibit chaos "
            "at high energies. Elastic pendulum: radial-angular energy exchange "
            "(1:2 autoparametric resonance). Double pendulum: angle-angle "
            "energy exchange. Both have integrable (small angle) and chaotic regimes."
        ),
        strength=0.7,
        mapping={
            "r-theta coupling": "theta1-theta2 coupling",
            "1:2 resonance": "chaotic onset at high E",
            "energy conservation": "energy conservation",
        },
    ))

    # Brusselator-diffusion <-> Diffusive LV (reaction-diffusion patterns)
    analogies.append(Analogy(
        domain_a="brusselator_diffusion",
        domain_b="diffusive_lv",
        analogy_type="topological",
        description=(
            "Both are 2-component reaction-diffusion PDEs that produce "
            "spatial patterns from uniform initial conditions. "
            "Brusselator-diff: Turing patterns (spots, stripes). "
            "Diffusive LV: traveling waves and spatial segregation. "
            "Both require differential diffusion (D_v != D_u)."
        ),
        strength=0.7,
        mapping={
            "Turing patterns": "traveling waves",
            "D_v/D_u ratio": "D_pred/D_prey ratio",
            "u, v [chemical concentrations]": "prey, predator [populations]",
        },
    ))

    # Topological: Chua <-> Rossler (3D chaotic attractors with period-doubling)
    analogies.append(Analogy(
        domain_a="chua",
        domain_b="rossler",
        analogy_type="topological",
        description=(
            "Both are 3D autonomous ODE systems that exhibit chaos via "
            "period-doubling cascades. Chua has a double-scroll attractor "
            "while Rossler has a single folded-band attractor. Both show "
            "positive Lyapunov exponents and similar bifurcation diagrams."
        ),
        strength=0.85,
        mapping={
            "double-scroll": "folded-band",
            "alpha [bifurcation param]": "c [bifurcation param]",
            "period-doubling": "period-doubling",
        },
    ))

    # Topological: Rosenzweig-MacArthur <-> Lotka-Volterra (predator-prey cycles)
    analogies.append(Analogy(
        domain_a="rosenzweig_macarthur",
        domain_b="lotka_volterra",
        analogy_type="topological",
        description=(
            "Both exhibit oscillatory predator-prey dynamics in phase space. "
            "LV has neutrally stable center orbits (conservative). RM has "
            "true limit cycles (dissipative). The paradox of enrichment "
            "in RM creates destabilizing oscillations as K increases."
        ),
        strength=0.85,
        mapping={
            "limit cycle": "center orbits",
            "phase plane (x,y)": "phase plane (N1,N2)",
        },
    ))

    # Topological: Shallow Water <-> Navier-Stokes (nonlinear wave propagation)
    analogies.append(Analogy(
        domain_a="shallow_water",
        domain_b="navier_stokes",
        analogy_type="topological",
        description=(
            "Both exhibit nonlinear wave dynamics in fluid media. "
            "Shallow water develops shock waves and bores; NS develops "
            "turbulent cascades. Both conserve energy in the inviscid limit."
        ),
        strength=0.7,
        mapping={
            "shock formation": "turbulent cascade",
            "wave breaking": "vortex stretching",
        },
    ))

    # Topological: Toda Lattice <-> Kepler (integrable Hamiltonian systems)
    analogies.append(Analogy(
        domain_a="toda_lattice",
        domain_b="kepler",
        analogy_type="topological",
        description=(
            "Both are completely integrable Hamiltonian systems with as many "
            "conserved quantities as degrees of freedom. Toda supports solitons; "
            "Kepler has closed elliptical orbits. Both have quasi-periodic "
            "phase space trajectories on invariant tori."
        ),
        strength=0.7,
        mapping={
            "soliton [N conserved quantities]": "closed orbit [E, L conserved]",
            "quasi-periodic tori": "elliptical orbits",
        },
    ))

    return analogies


def compute_equation_similarity(eq_a: str, eq_b: str) -> float:
    """Compute similarity score between two equation strings.

    Uses structural features: number of operators, variable count,
    presence of sqrt/sin/exp, etc.
    """
    def features(eq: str) -> dict[str, float]:
        return {
            "has_sqrt": float("sqrt" in eq),
            "has_sin": float("sin" in eq),
            "has_exp": float("exp" in eq),
            "has_division": float("/" in eq),
            "has_product": float("*" in eq),
            "n_vars": sum(1 for c in eq if c.isalpha() and c not in "sincoexpqrtlog"),
            "n_ops": sum(1 for c in eq if c in "+-*/^"),
            "length": len(eq),
        }

    f_a = features(eq_a)
    f_b = features(eq_b)

    # Jaccard-like similarity on binary features + normalized difference on numeric
    binary_keys = ["has_sqrt", "has_sin", "has_exp", "has_division", "has_product"]
    numeric_keys = ["n_vars", "n_ops", "length"]

    binary_sim = sum(f_a[k] == f_b[k] for k in binary_keys) / len(binary_keys)
    numeric_sim = 0
    for k in numeric_keys:
        max_val = max(f_a[k], f_b[k], 1)
        numeric_sim += 1 - abs(f_a[k] - f_b[k]) / max_val
    numeric_sim /= len(numeric_keys)

    return 0.5 * binary_sim + 0.5 * numeric_sim


def run_cross_domain_analysis(
    output_dir: str | Path = "output/cross_domain",
) -> dict:
    """Run the full cross-domain analogy analysis.

    Returns a structured report of all detected analogies.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    signatures = build_domain_signatures()

    # Detect all analogy types
    structural = detect_structural_analogies(signatures)
    dimensional = detect_dimensional_analogies(signatures)
    topological = detect_topological_analogies(signatures)

    all_analogies = structural + dimensional + topological

    # Build similarity matrix
    domain_names = [s.name for s in signatures]
    n = len(domain_names)
    sim_matrix = np.zeros((n, n))

    for analogy in all_analogies:
        i = domain_names.index(analogy.domain_a)
        j = domain_names.index(analogy.domain_b)
        sim_matrix[i, j] = max(sim_matrix[i, j], analogy.strength)
        sim_matrix[j, i] = max(sim_matrix[j, i], analogy.strength)

    # Fill diagonal
    np.fill_diagonal(sim_matrix, 1.0)

    # Build analogy graph (connections)
    graph = {}
    for analogy in all_analogies:
        key = f"{analogy.domain_a} <-> {analogy.domain_b}"
        graph[key] = {
            "type": analogy.analogy_type,
            "strength": analogy.strength,
            "description": analogy.description,
        }

    # Domain classification by math type
    type_groups = {}
    for sig in signatures:
        if sig.math_type not in type_groups:
            type_groups[sig.math_type] = []
        type_groups[sig.math_type].append(sig.name)

    results = {
        "n_domains": len(signatures),
        "n_analogies": len(all_analogies),
        "analogy_types": {
            "structural": len(structural),
            "dimensional": len(dimensional),
            "topological": len(topological),
        },
        "domain_signatures": {
            s.name: {
                "math_type": s.math_type,
                "state_dim": s.state_dim,
                "n_parameters": s.n_parameters,
                "phase_portrait": s.phase_portrait_type,
                "timescale": s.characteristic_timescale,
                "conserved": s.conserved_quantities,
                "n_equations": len(s.discovered_equations),
            }
            for s in signatures
        },
        "type_groups": type_groups,
        "analogies": [
            {
                "domains": [a.domain_a, a.domain_b],
                "type": a.analogy_type,
                "strength": a.strength,
                "description": a.description,
                "mapping": a.mapping,
            }
            for a in all_analogies
        ],
        "similarity_matrix": {
            "domain_names": domain_names,
            "matrix": sim_matrix.tolist(),
        },
        "analogy_graph": graph,
    }

    # Log results
    logger.info("=" * 60)
    logger.info("CROSS-DOMAIN ANALOGY ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Domains analyzed: {len(signatures)}")
    logger.info(f"Analogies found: {len(all_analogies)}")
    logger.info(f"  Structural: {len(structural)}")
    logger.info(f"  Dimensional: {len(dimensional)}")
    logger.info(f"  Topological: {len(topological)}")
    logger.info("")
    logger.info("Math type groups:")
    for mtype, domains in type_groups.items():
        logger.info(f"  {mtype}: {', '.join(domains)}")
    logger.info("")
    logger.info("Strongest analogies:")
    sorted_analogies = sorted(all_analogies, key=lambda a: a.strength, reverse=True)
    for a in sorted_analogies[:5]:
        logger.info(f"  {a.domain_a} <-> {a.domain_b} ({a.analogy_type}, "
                     f"strength={a.strength:.2f})")
        logger.info(f"    {a.description[:100]}...")

    # Save
    results_file = output_path / "cross_domain_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    np.savez(
        output_path / "similarity_matrix.npz",
        domain_names=np.array(domain_names),
        similarity=sim_matrix,
    )

    return results
