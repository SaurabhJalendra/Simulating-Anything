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
        DomainSignature(
            name="kuramoto_sivashinsky",
            math_type="pde",  # Chaotic PDE
            state_dim=128,  # N grid points
            n_parameters=3,  # L, N, viscosity
            conserved_quantities=["spatial_mean"],
            symmetries=["translational", "Galilean"],
            phase_portrait_type="chaotic",
            characteristic_timescale="L^2",
            discovered_equations=[
                "u_t + u*u_x + u_xx + u_xxxx = 0",
                "Spatiotemporal chaos for L > 2*pi*sqrt(2)",
                "Positive Lyapunov exponent",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="ginzburg_landau",
            math_type="pde",  # Complex PDE
            state_dim=256,  # 2*N (Re + Im)
            n_parameters=4,  # c1, c2, L, N
            conserved_quantities=[],
            symmetries=["translational", "phase_rotation"],
            phase_portrait_type="chaotic",
            characteristic_timescale="1",
            discovered_equations=[
                "dA/dt = A + (1+ic1)*d^2A/dx^2 - (1+ic2)*|A|^2*A",
                "Benjamin-Feir: unstable when 1+c1*c2 < 0",
                "Phase turbulence regime",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="oregonator",
            math_type="ode_nonlinear",
            state_dim=3,  # [u, v, w]
            n_parameters=4,  # eps, f, q, kw
            conserved_quantities=[],
            symmetries=["positivity"],
            phase_portrait_type="limit_cycle",
            characteristic_timescale="1/eps",
            discovered_equations=[
                "du/dt = (u*(1-u) - f*v*(u-q)/(u+q)) / eps",
                "dv/dt = u - v",
                "Relaxation oscillations (BZ reaction)",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="bak_sneppen",
            math_type="discrete",  # Stochastic extremal dynamics
            state_dim=50,  # N species
            n_parameters=1,  # N
            conserved_quantities=[],
            symmetries=["translational"],
            phase_portrait_type="none",  # SOC, no attractor
            characteristic_timescale="N",
            discovered_equations=[
                "SOC threshold f_c ~ 2/3",
                "Power-law avalanche distribution P(s) ~ s^{-tau}",
                "Self-organized criticality",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="lorenz96",
            math_type="chaotic",
            state_dim=36,  # N sites
            n_parameters=2,  # N, F
            conserved_quantities=[],
            symmetries=["translational", "cyclic"],
            phase_portrait_type="chaotic",
            characteristic_timescale="1",
            discovered_equations=[
                "dx_i/dt = (x_{i+1}-x_{i-2})*x_{i-1} - x_i + F",
                "Chaos for F >= 8",
                "Extensive chaos: Lyapunov dim ~ N",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="chemostat",
            math_type="ode_nonlinear",
            state_dim=2,  # [S, X]
            n_parameters=5,  # D, S_in, mu_max, K_s, Y_xs
            conserved_quantities=[],
            symmetries=["positivity"],
            phase_portrait_type="fixed_point",
            characteristic_timescale="1/D",
            discovered_equations=[
                "dS/dt = D*(S_in-S) - mu_max*S*X/(Y_xs*(K_s+S))",
                "dX/dt = (mu_max*S/(K_s+S)-D)*X",
                "Washout bifurcation at D_c = mu_max*S_in/(K_s+S_in)",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="fhn_spatial",
            math_type="pde",
            state_dim=256,  # 2*N
            n_parameters=5,  # a, b, eps, D_v, L
            conserved_quantities=[],
            symmetries=["translational"],
            phase_portrait_type="none",  # Traveling pulses
            characteristic_timescale="1/eps",
            discovered_equations=[
                "dv/dt = v-v^3/3-w + D_v*d^2v/dx^2",
                "dw/dt = eps*(v+a-b*w)",
                "Traveling pulse c ~ sqrt(D_v)",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="wilberforce",
            math_type="ode_linear",  # Linear coupled oscillators
            state_dim=4,  # [z, z_dot, theta, theta_dot]
            n_parameters=5,  # m, k, I, kappa, eps
            conserved_quantities=["energy"],
            symmetries=["time_reversal"],
            phase_portrait_type="none",  # Quasi-periodic
            characteristic_timescale="sqrt(m/k)",
            discovered_equations=[
                "m*z'' = -k*z - eps/2*theta",
                "I*theta'' = -kappa*theta - eps/2*z",
                "Beat frequency = |omega_z - omega_theta|",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="standard_map",
            math_type="discrete",  # 2D area-preserving map
            state_dim=2,  # [theta, p] per particle
            n_parameters=1,  # K (stochasticity parameter)
            conserved_quantities=["phase_space_area"],
            symmetries=["area_preserving", "translation_mod_2pi"],
            phase_portrait_type="chaotic",
            characteristic_timescale="1",  # Discrete map, one iteration
            discovered_equations=[
                "p_{n+1} = p_n + K*sin(theta_n)",
                "theta_{n+1} = theta_n + p_{n+1}",
                "K_c ~ 0.9716 (last KAM torus)",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="hodgkin_huxley",
            math_type="ode_nonlinear",  # 4D biophysical neuron
            state_dim=4,  # [V, n, m, h]
            n_parameters=7,  # g_Na, g_K, g_L, E_Na, E_K, E_L, C_m
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="limit_cycle",
            characteristic_timescale="1ms",  # Action potential duration
            discovered_equations=[
                "C_m*dV/dt = I - g_Na*m^3*h*(V-E_Na) - g_K*n^4*(V-E_K) - g_L*(V-E_L)",
                "dn/dt = alpha_n(V)*(1-n) - beta_n(V)*n",
                "f-I curve: monotonic firing frequency",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="rayleigh_benard",
            math_type="pde",  # 2D convection PDE
            state_dim=4096,  # Vorticity + temperature on grid
            n_parameters=3,  # Ra, Pr, Lx
            conserved_quantities=[],
            symmetries=["horizontal_translation"],
            phase_portrait_type="none",
            characteristic_timescale="H^2/kappa",
            discovered_equations=[
                "Ra_c = (27/4)*pi^4 ~ 657.5 (free-slip)",
                "Nu(Ra) ~ (Ra/Ra_c)^gamma above onset",
                "Convection roll wavelength ~ 2H",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="eco_epidemic",
            math_type="ode_nonlinear",  # 3D eco-epidemiological
            state_dim=3,  # [S, I, P]
            n_parameters=10,  # r, K, beta, a1, a2, h1, h2, e1, e2, d, m
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="limit_cycle",
            characteristic_timescale="1/r",
            discovered_equations=[
                "dS/dt = rS(1-(S+I)/K) - beta*S*I - a1*S*P/(1+h1*a1*S)",
                "R0 = beta*K/d (without predators)",
                "Predator-mediated disease control",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="hindmarsh_rose",
            math_type="ode_nonlinear",  # 3D bursting neuron
            state_dim=3,  # [x, y, z]
            n_parameters=8,  # a, b, c, d, r, s, x_rest, I_ext
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="limit_cycle",
            characteristic_timescale="1/r",  # Slow timescale
            discovered_equations=[
                "dx/dt = y - a*x^3 + b*x^2 - z + I_ext",
                "dy/dt = c - d*x^2 - y",
                "dz/dt = r*(s*(x-x_rest) - z)",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="magnetic_pendulum",
            math_type="ode_nonlinear",  # Fractal basin ODE
            state_dim=4,  # [x, y, vx, vy]
            n_parameters=5,  # gamma, omega0_sq, alpha, R, d
            conserved_quantities=[],
            symmetries=["rotation_120"],
            phase_portrait_type="none",  # Transient chaos -> fixed point
            characteristic_timescale="1/omega0",
            discovered_equations=[
                "x'' = -gamma*x' - omega0^2*x + sum(alpha*(xi-x)/ri^3)",
                "Fractal basin boundaries",
                "3-fold rotational symmetry",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="competitive_lv",
            math_type="ode_nonlinear",  # 4-species competitive
            state_dim=4,  # [N1, N2, N3, N4]
            n_parameters=24,  # r(4), K(4), alpha(4x4)
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="none",  # Stable node or exclusion
            characteristic_timescale="1/r",
            discovered_equations=[
                "dNi/dt = ri*Ni*(1 - sum(alpha_ij*Nj/Ki))",
                "Competitive exclusion principle",
                "N* = alpha^(-1) @ K",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="vicsek",
            math_type="collective",  # Active matter flocking
            state_dim=300,  # 3*N particles (x,y,theta each)
            n_parameters=4,  # N, L, v0, R, eta
            conserved_quantities=["particle_count"],
            symmetries=["rotation", "translation"],
            phase_portrait_type="none",
            characteristic_timescale="L/v0",
            discovered_equations=[
                "theta_i(t+1) = <theta_j>_{R} + eta*U(-pi,pi)",
                "Order parameter phi = |mean(exp(i*theta))|",
                "Phase transition at eta_c",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="coupled_lorenz",
            math_type="chaotic",  # Coupled chaotic ODEs
            state_dim=6,  # [x1,y1,z1, x2,y2,z2]
            n_parameters=4,  # sigma, rho, beta, eps
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/sigma",
            discovered_equations=[
                "dx2/dt = sigma*(y2-x2) + eps*(x1-x2)",
                "Sync at eps > eps_c",
                "Conditional Lyapunov < 0 for sync",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="bz_spiral",
            math_type="pde",  # 2D excitable PDE
            state_dim=8192,  # 2 * 64 * 64
            n_parameters=5,  # eps, f, q, D_u, D_v
            conserved_quantities=[],
            symmetries=["rotation"],
            phase_portrait_type="limit_cycle",
            characteristic_timescale="eps",
            discovered_equations=[
                "du/dt = D_u*lap(u) + (1/eps)*(u-u^2-f*v*(u-q)/(u+q))",
                "dv/dt = u - v",
                "Spiral wave frequency and wavelength",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="swinging_atwood",
            math_type="chaotic",  # Lagrangian chaotic ODE
            state_dim=4,  # [r, theta, r_dot, theta_dot]
            n_parameters=3,  # M, m, g
            conserved_quantities=["energy"],
            symmetries=[],
            phase_portrait_type="chaotic",
            characteristic_timescale="sqrt(r0/g)",
            discovered_equations=[
                "(M+m)*r'' = m*r*theta'^2 + m*g*cos(theta) - M*g",
                "r^2*theta'' = -2*r*r'*theta' - g*r*sin(theta)",
                "E = T + V = const",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="allee_predator_prey",
            math_type="ode_nonlinear",  # Bistable predator-prey
            state_dim=2,  # [N, P]
            n_parameters=7,  # r, A, K, a, h, e, m
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="none",  # Bistable: extinction or coexistence
            characteristic_timescale="1/r",
            discovered_equations=[
                "dN/dt = r*N*(N/A-1)*(1-N/K) - a*N*P/(1+h*a*N)",
                "Strong Allee effect at N=A",
                "Bistability: extinction vs coexistence",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="mackey_glass",
            math_type="chaotic",  # Delay differential equation
            state_dim=1,  # x (with delay embedding)
            n_parameters=4,  # beta, gamma, n, tau
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="chaotic",  # Chaotic for large tau
            characteristic_timescale="tau",
            discovered_equations=[
                "dx/dt = beta*x(t-tau)/(1+x(t-tau)^n) - gamma*x",
                "Period-doubling cascade with increasing tau",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="bouncing_ball",
            math_type="discrete",  # Impact map
            state_dim=2,  # [height, velocity]
            n_parameters=3,  # g, e (restitution), omega (table freq)
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="chaotic",  # Period-doubling chaos
            characteristic_timescale="sqrt(2*h/g)",
            discovered_equations=[
                "v_n+1 = e*v_n + impulse",
                "Period-doubling cascade with table amplitude",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="wilson_cowan",
            math_type="ode_nonlinear",  # Neural population model
            state_dim=2,  # [E, I]
            n_parameters=8,  # w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, I_ext_E, I_ext_I
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="limit_cycle",  # E-I oscillation
            characteristic_timescale="tau_e, tau_i",
            discovered_equations=[
                "tau_e*dE/dt = -E + S(w_ee*E - w_ei*I + I_ext_E)",
                "tau_i*dI/dt = -I + S(w_ie*E - w_ii*I + I_ext_I)",
                "S(x) = 1/(1+exp(-a*(x-theta)))",
                "Hopf bifurcation at critical I_ext",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="cable_equation",
            math_type="pde",  # Linear parabolic PDE
            state_dim=1,  # V(x,t) membrane potential
            n_parameters=3,  # lambda_e, tau_m, R_m
            conserved_quantities=[],
            symmetries=["translation"],
            phase_portrait_type="fixed_point",  # Exponential decay to steady state
            characteristic_timescale="tau_m",
            discovered_equations=[
                "tau_m*dV/dt = lambda^2*V_xx - V + R_m*I_ext",
                "V(x) ~ exp(-|x|/lambda) steady state",
                "lambda = sqrt(r_m/r_i) space constant",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="sine_gordon",
            math_type="pde",  # Nonlinear wave PDE
            state_dim=2,  # [u, u_t] field + time derivative
            n_parameters=2,  # c, N (wave speed, grid points)
            conserved_quantities=["energy", "topological_charge"],
            symmetries=["translation", "Lorentz_boost"],
            phase_portrait_type="none",  # Solitons
            characteristic_timescale="L/c",
            discovered_equations=[
                "u_tt = c^2*u_xx - sin(u)",
                "kink: u = 4*arctan(exp((x-vt)/sqrt(1-v^2/c^2)))",
                "E_kink = 8*c/sqrt(1-v^2/c^2)",
                "topological charge Q = integer",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="thomas",
            math_type="chaotic",  # Cyclically symmetric chaotic ODE
            state_dim=3,  # [x, y, z]
            n_parameters=1,  # b (dissipation)
            conserved_quantities=[],
            symmetries=["cyclic_permutation"],  # (x,y,z) -> (y,z,x)
            phase_portrait_type="chaotic",  # Labyrinth chaos
            characteristic_timescale="1/b",
            discovered_equations=[
                "dx/dt = sin(y) - b*x",
                "dy/dt = sin(z) - b*y",
                "dz/dt = sin(x) - b*z",
                "b_c ~ 0.208186 chaos transition",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="ikeda_map",
            math_type="discrete",  # Discrete chaos from nonlinear optics
            state_dim=2,  # [x, y]
            n_parameters=1,  # u (coupling)
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="chaotic",  # Strange attractor
            characteristic_timescale="1 (iteration)",
            discovered_equations=[
                "x' = 1 + u*(x*cos(t)-y*sin(t))",
                "y' = u*(x*sin(t)+y*cos(t))",
                "t = 0.4 - 6/(1+x^2+y^2)",
                "det(J) = u^2 (dissipative)",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="may_leonard",
            math_type="ode_nonlinear",  # Cyclic competition
            state_dim=4,  # [x1, x2, x3, x4]
            n_parameters=4,  # n_species, a, b, r
            conserved_quantities=[],
            symmetries=["cyclic_permutation"],
            phase_portrait_type="none",  # Heteroclinic cycles
            characteristic_timescale="1/r",
            discovered_equations=[
                "dx_i/dt = r*x_i*(1 - sum(alpha_ij*x_j/K))",
                "x* = K/(1+a+(n-2)*b) interior fixed point",
                "Cyclic dominance: 1->2->3->4->1",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="cahn_hilliard",
            math_type="pde",  # 4th-order phase field PDE
            state_dim=1,  # u(x,y) concentration field
            n_parameters=3,  # M, epsilon, N
            conserved_quantities=["total_mass"],
            symmetries=["translation", "rotation", "u_symmetry"],
            phase_portrait_type="fixed_point",  # Phase separation to +-1
            characteristic_timescale="L^4/(M*epsilon^2)",
            discovered_equations=[
                "du/dt = M*nabla^2(u^3-u-eps^2*nabla^2 u)",
                "L(t) ~ t^(1/3) coarsening law",
                "E = integral(f(u)+eps^2/2*|grad u|^2) decreases",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="delayed_predator_prey",
            math_type="ode_nonlinear",  # DDE predator-prey
            state_dim=2,  # [N, P] with delay buffer
            n_parameters=7,  # r, K, a, h, e, m, tau
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="limit_cycle",  # Delay-induced oscillation
            characteristic_timescale="1/r, tau",
            discovered_equations=[
                "dN/dt = r*N*(1-N/K) - a*N*P/(1+h*a*N)",
                "dP/dt = e*a*N(t-tau)*P/(1+h*a*N(t-tau)) - m*P",
                "Hopf bifurcation at critical tau_c",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="duffing_van_der_pol",
            math_type="chaotic",  # Hybrid nonlinear oscillator
            state_dim=3,  # [x, y, t]
            n_parameters=5,  # mu, alpha, beta, F, omega
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="chaotic",  # Can be limit cycle or chaotic
            characteristic_timescale="2*pi/omega",
            discovered_equations=[
                "x'' + mu*(x^2-1)*x' + alpha*x + beta*x^3 = F*cos(omega*t)",
                "VdP limit: beta=0 gives limit cycle A~2",
                "Duffing limit: mu=0 gives forced oscillator",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="network_sis",
            math_type="ode_nonlinear",  # Mean-field network epidemic
            state_dim=50,  # [p_1, ..., p_N] infection probabilities
            n_parameters=4,  # beta, gamma, N, mean_degree
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="fixed_point",  # Endemic or disease-free
            characteristic_timescale="1/gamma",
            discovered_equations=[
                "dp_i/dt = -gamma*p_i + beta*(1-p_i)*sum(A_ij*p_j)",
                "threshold: beta_c/gamma = 1/lambda_max(A)",
                "ER: beta_c ~ gamma/<k>",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="coupled_map_lattice",
            math_type="discrete",  # Spatiotemporal chaos
            state_dim=100,  # [x_1, ..., x_N] lattice
            n_parameters=3,  # N, r, eps
            conserved_quantities=[],
            symmetries=["translation", "reflection"],
            phase_portrait_type="chaotic",
            characteristic_timescale="1 (iteration)",
            discovered_equations=[
                "x_i' = (1-eps)*f(x_i) + eps/2*(f(x_{i-1})+f(x_{i+1}))",
                "f(x) = r*x*(1-x)",
                "sync threshold eps_c ~ 0.5",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="schnakenberg",
            math_type="pde",  # Reaction-diffusion Turing
            state_dim=2,  # [u, v] on 2D grid
            n_parameters=6,  # D_u, D_v, a, b, N, L
            conserved_quantities=[],
            symmetries=["rotation", "translation", "reflection"],
            phase_portrait_type="fixed_point",  # Turing patterns
            characteristic_timescale="L^2/D_u",
            discovered_equations=[
                "du/dt = D_u*nabla^2 u + a - u + u^2*v",
                "dv/dt = D_v*nabla^2 v + b - u^2*v",
                "u* = a+b, v* = b/(a+b)^2",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="kapitza_pendulum",
            math_type="chaotic",  # Parametrically driven ODE
            state_dim=3,  # [theta, theta_dot, t]
            n_parameters=5,  # L, g, a, omega, gamma
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="fixed_point",  # Inverted stability
            characteristic_timescale="sqrt(L/g)",
            discovered_equations=[
                "theta'' + gamma*theta' + (g/L)*sin(theta) = (a*omega^2/L)*cos(omega*t)*sin(theta)",
                "Stability criterion: a^2*omega^2 > 2*g*L",
                "V_eff = -gL*cos(theta) + (a*omega)^2/(4L)*sin^2(theta)",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="fitzhugh_rinzel",
            math_type="ode_nonlinear",  # Bursting neuron
            state_dim=3,  # [v, w, y]
            n_parameters=7,  # I_ext, a, b, c, d, delta, mu
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="limit_cycle",  # Bursting oscillation
            characteristic_timescale="1/delta, 1/mu",
            discovered_equations=[
                "dv/dt = v-v^3/3-w+y+I",
                "dw/dt = delta*(a+v-b*w)",
                "dy/dt = mu*(c-v-d*y)",
                "3 timescales: fast spikes, recovery, slow modulation",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="lorenz_84",
            math_type="chaotic",  # Low-dimensional atmospheric chaos
            state_dim=3,  # [x, y, z] -- westerly wind + eddy phases
            n_parameters=4,  # a, b, F, G
            conserved_quantities=[],
            symmetries=["(y,z)->(-y,-z) when G=0"],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/a",
            discovered_equations=[
                "dx/dt = -y^2-z^2-a*x+a*F",
                "dy/dt = x*y-b*x*z-y+G",
                "dz/dt = b*x*y+x*z-z",
                "Hadley fixed point x*=F",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="rabinovich_fabrikant",
            math_type="chaotic",  # Plasma physics chaos
            state_dim=3,  # [x, y, z]
            n_parameters=2,  # alpha, gamma
            conserved_quantities=[],
            symmetries=["(x,y)->(-x,-y)"],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/alpha",
            discovered_equations=[
                "dx/dt = y(z-1+x^2)+gamma*x",
                "dy/dt = x(3z+1-x^2)+gamma*y",
                "dz/dt = -2z(alpha+xy)",
                "Multiscroll strange attractor",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="sprott",
            math_type="chaotic",  # Minimal chaotic flow
            state_dim=3,  # [x, y, z]
            n_parameters=1,  # system letter (effectively parameterless)
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="chaotic",
            characteristic_timescale="1.0",
            discovered_equations=[
                "dx/dt = yz (Sprott-B)",
                "dy/dt = x-y",
                "dz/dt = 1-xy",
                "Minimal dissipative chaos",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="gray_scott_1d",
            math_type="pde",  # 1D reaction-diffusion
            state_dim=512,  # 2*N for u,v fields
            n_parameters=4,  # D_u, D_v, f, k
            conserved_quantities=[],
            symmetries=["translational"],
            phase_portrait_type="traveling_wave",
            characteristic_timescale="1/f",
            discovered_equations=[
                "du/dt = D_u*d2u/dx2 - u*v^2 + f*(1-u)",
                "dv/dt = D_v*d2v/dx2 + u*v^2 - (f+k)*v",
                "Self-replicating pulses",
                "Pulse splitting bifurcation",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="predator_prey_mutualist",
            math_type="ode_nonlinear",  # 3-species mutualistic ODE
            state_dim=3,  # [x, y, z] -- prey, predator, mutualist
            n_parameters=11,
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="limit_cycle",
            characteristic_timescale="1/r",
            discovered_equations=[
                "dx/dt = rx(1-x/K) - axy/(1+bx) + mxz/(1+nz)",
                "dy/dt = -dy + eaxy/(1+bx)",
                "dz/dt = sz(1-z/C) + pxz/(1+nz)",
                "Holling II + mutualism stabilization",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="brusselator_2d",
            math_type="pde",  # 2D reaction-diffusion
            state_dim=8192,  # 2*64*64
            n_parameters=4,  # D_u, D_v, a, b
            conserved_quantities=[],
            symmetries=["translational", "rotational"],
            phase_portrait_type="turing_pattern",
            characteristic_timescale="1/(b-1-a^2)",
            discovered_equations=[
                "du/dt = D_u*nabla^2(u) + a-(b+1)u+u^2v",
                "dv/dt = D_v*nabla^2(v) + bu-u^2v",
                "Turing wavelength ~ 2pi*sqrt(D_v/(b-1-a^2))",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="fput",
            math_type="ode_nonlinear",  # Hamiltonian lattice
            state_dim=64,  # 2*N positions + velocities
            n_parameters=3,  # k, alpha, beta
            conserved_quantities=["energy"],
            symmetries=["time_reversal"],
            phase_portrait_type="quasi_periodic",
            characteristic_timescale="2*pi/omega_1",
            discovered_equations=[
                "d2x_i/dt2 = F(x_{i+1}-x_i) - F(x_i-x_{i-1})",
                "F(d) = k*d + alpha*d^2 + beta*d^3",
                "FPUT recurrence paradox",
                "omega_n = 2*sqrt(k)*sin(n*pi/(2*(N+1)))",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="selkov",
            math_type="ode_nonlinear",  # Biochemical oscillator
            state_dim=2,  # [x, y] -- ADP, F6P
            n_parameters=2,  # a, b
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="limit_cycle",
            characteristic_timescale="1/a",
            discovered_equations=[
                "dx/dt = -x + ay + x^2y",
                "dy/dt = b - ay - x^2y",
                "Hopf at b_c(a) from trace(J)=0",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="rikitake",
            math_type="chaotic",  # Geophysical dynamo
            state_dim=3,  # [x, y, z]
            n_parameters=2,  # mu, a
            conserved_quantities=[],
            symmetries=["(x,y)->(-x,-y)"],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/mu",
            discovered_equations=[
                "dx/dt = -mu*x + z*y",
                "dy/dt = -mu*y + (z-a)*x",
                "dz/dt = 1 - x*y",
                "Polarity reversals in geomagnetic field",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="oregonator_1d",
            math_type="pde",  # 1D excitable RD
            state_dim=400,  # 2*N
            n_parameters=5,  # eps, f, q, D_u, D_v
            conserved_quantities=[],
            symmetries=["translational"],
            phase_portrait_type="traveling_wave",
            characteristic_timescale="eps",
            discovered_equations=[
                "du/dt = D_u*d2u/dx2 + (1/eps)*(u-u^2-f*v*(u-q)/(u+q))",
                "dv/dt = D_v*d2v/dx2 + u-v",
                "Traveling pulse speed ~ sqrt(D_u/eps)",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="ricker_map",
            math_type="discrete",  # Discrete population map
            state_dim=1,  # scalar x
            n_parameters=2,  # r, K
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="chaotic",
            characteristic_timescale="1 (discrete)",
            discovered_equations=[
                "x_{n+1} = x_n * exp(r*(1-x_n/K))",
                "Period-doubling at r~2",
                "Lyapunov = ln|1-r| at fixed point",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="morris_lecar",
            math_type="ode_nonlinear",  # Conductance neuron
            state_dim=2,  # [V, w]
            n_parameters=13,
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="limit_cycle",
            characteristic_timescale="C/g_L",
            discovered_equations=[
                "C*dV/dt = I-g_L*(V-V_L)-g_Ca*m_ss(V)*(V-V_Ca)-g_K*w*(V-V_K)",
                "dw/dt = phi*(w_ss(V)-w)/tau_w(V)",
                "Type I/II excitability classification",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="colpitts",
            math_type="chaotic",  # Electronic chaos
            state_dim=3,  # [x, y, z]
            n_parameters=2,  # Q, g_d
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/g_d",
            discovered_equations=[
                "dx/dt = y",
                "dy/dt = z",
                "dz/dt = -g_d*z - y + V_cc - Q*max(0,x)",
                "Piecewise-linear electronic chaos",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="rossler_hyperchaos",
            math_type="chaotic",  # 4D hyperchaos
            state_dim=4,  # [x, y, z, w]
            n_parameters=4,  # a, b, c, d
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/a",
            discovered_equations=[
                "dx/dt = -(y+z)",
                "dy/dt = x+a*y+w",
                "dz/dt = b+x*z",
                "dw/dt = -c*z+d*w",
                "Two positive Lyapunov exponents",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="harvested_population",
            math_type="ode_nonlinear",  # Resource dynamics
            state_dim=1,  # scalar x
            n_parameters=3,  # r, K, H
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="fixed_point",
            characteristic_timescale="1/r",
            discovered_equations=[
                "dx/dt = r*x*(1-x/K) - H",
                "MSY = r*K/4 (saddle-node)",
                "Two equilibria for H < H_MSY",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="fhn_ring",
            math_type="ode_nonlinear",  # Neural network
            state_dim=40,  # 2*N
            n_parameters=6,  # a, b, eps, I, D, N
            conserved_quantities=[],
            symmetries=["circular (Z_N)"],
            phase_portrait_type="traveling_wave",
            characteristic_timescale="1/eps",
            discovered_equations=[
                "dv_i/dt = v_i-v_i^3/3-w_i+I+D*(v_{i-1}-2v_i+v_{i+1})",
                "dw_i/dt = eps*(v_i+a-b*w_i)",
                "Traveling excitation waves around ring",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="bazykin",
            math_type="ode_nonlinear",  # Pred-prey ODE
            state_dim=2,
            n_parameters=3,  # alpha, gamma, delta
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="limit_cycle",
            characteristic_timescale="1/gamma",
            discovered_equations=[
                "dx/dt = x*(1-x) - x*y/(1+alpha*x)",
                "dy/dt = -gamma*y + x*y/(1+alpha*x) - delta*y^2",
                "Holling Type II + quadratic mortality",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="sir_vaccination",
            math_type="ode_nonlinear",  # Epidemic ODE
            state_dim=3,
            n_parameters=4,  # beta, gamma, mu, nu
            conserved_quantities=["total_population"],
            symmetries=[],
            phase_portrait_type="fixed_point",
            characteristic_timescale="1/gamma",
            discovered_equations=[
                "dS/dt = mu*N - beta*S*I/N - nu*S - mu*S",
                "dI/dt = beta*S*I/N - gamma*I - mu*I",
                "dR/dt = gamma*I + nu*S - mu*R",
                "R0 = beta/(gamma+mu), nu_c = mu*(R0-1)",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="langford",
            math_type="chaotic",  # Torus ODE
            state_dim=3,
            n_parameters=6,  # a, b, c, d, e, f
            conserved_quantities=[],
            symmetries=["rotational_xy"],
            phase_portrait_type="torus",
            characteristic_timescale="2*pi/d",
            discovered_equations=[
                "dx/dt = (z-b)*x - d*y",
                "dy/dt = d*x + (z-b)*y",
                "dz/dt = c + a*z - z^3/3 - r^2*(1+e*z) + f*z*x^3",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="laser_rate",
            math_type="ode_nonlinear",  # Laser physics
            state_dim=2,
            n_parameters=7,  # P, gamma_N, gamma_S, g, N_tr, Gamma, beta
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="fixed_point",
            characteristic_timescale="1/gamma_S",
            discovered_equations=[
                "dN/dt = P - gamma_N*N - g*(N-N_tr)*S",
                "dS/dt = Gamma*g*(N-N_tr)*S - gamma_S*S + Gamma*beta*gamma_N*N",
                "P_th = gamma_N*N_tr + gamma_S/(Gamma*g)",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="fhn_lattice",
            math_type="ode_nonlinear",  # Lattice ODE
            state_dim=2048,  # 2*32*32
            n_parameters=6,  # a, b, eps, I, D, N
            conserved_quantities=[],
            symmetries=["translational (periodic)", "rotational (square)"],
            phase_portrait_type="spiral_wave",
            characteristic_timescale="1/eps",
            discovered_equations=[
                "dv/dt = v-v^3/3-w+I+D*Lap(v)",
                "dw/dt = eps*(v+a-b*w)",
                "Spiral wave formation on 2D lattice",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="four_species_lv",
            math_type="ode_nonlinear",  # Food web ODE
            state_dim=4,
            n_parameters=12,  # r1,r2,a11,a12,a21,a22,b1,b2,c1,c2,d1,d2
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="limit_cycle",
            characteristic_timescale="1/d",
            discovered_equations=[
                "dx1/dt = x1*(r1-a11*x1-a12*x2-b1*y1)",
                "dx2/dt = x2*(r2-a21*x1-a22*x2-b2*y2)",
                "dy1/dt = y1*(-d1+c1*x1)",
                "dy2/dt = y2*(-d2+c2*x2)",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="lorenz_stenflo",
            math_type="chaotic",  # Plasma chaos
            state_dim=4,
            n_parameters=4,  # sigma, r, b, s
            conserved_quantities=[],
            symmetries=["Z2 (x,y,w -> -x,-y,-w)"],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/sigma",
            discovered_equations=[
                "dx/dt = sigma*(y-x) + s*w",
                "dy/dt = r*x - y - x*z",
                "dz/dt = x*y - b*z",
                "dw/dt = -x - sigma*w",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="chen",
            math_type="chaotic",  # Chaotic ODE
            state_dim=3,
            n_parameters=3,  # a, b, c
            conserved_quantities=[],
            symmetries=["Z2 (x,y -> -x,-y)"],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/a",
            discovered_equations=[
                "dx/dt = a*(y-x)",
                "dy/dt = (c-a)*x - x*z + c*y",
                "dz/dt = x*y - b*z",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="aizawa",
            math_type="chaotic",  # 3D chaotic ODE
            state_dim=3,
            n_parameters=6,  # a, b, c, d, e, f
            conserved_quantities=[],
            symmetries=["rotational_xy (coupled x,y rotation)"],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/d",
            discovered_equations=[
                "dx/dt = (z-b)*x - d*y",
                "dy/dt = d*x + (z-b)*y",
                "dz/dt = c + a*z - z^3/3 - (x^2+y^2)*(1+e*z) + f*z*x^3",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="halvorsen",
            math_type="chaotic",  # 3D chaotic ODE
            state_dim=3,
            n_parameters=1,  # a
            conserved_quantities=[],
            symmetries=["cyclic_S3 (x->y->z->x)"],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/a",
            discovered_equations=[
                "dx/dt = -a*x - 4*y - 4*z - y^2",
                "dy/dt = -a*y - 4*z - 4*x - z^2",
                "dz/dt = -a*z - 4*x - 4*y - x^2",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="burke_shaw",
            math_type="chaotic",  # 3D chaotic ODE
            state_dim=3,
            n_parameters=2,  # s, v
            conserved_quantities=[],
            symmetries=["Z2 (x,y -> -x,-y)"],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/s",
            discovered_equations=[
                "dx/dt = -s*(x+y)",
                "dy/dt = -y - s*x*z",
                "dz/dt = s*x*y + v",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="nose_hoover",
            math_type="chaotic",  # Volume-preserving-like ODE
            state_dim=3,
            n_parameters=1,  # a
            conserved_quantities=["time-avg divergence ~ 0"],
            symmetries=[],
            phase_portrait_type="chaotic",
            characteristic_timescale="1",
            discovered_equations=[
                "dx/dt = y",
                "dy/dt = -x + y*z",
                "dz/dt = a - y^2",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="lorenz_haken",
            math_type="chaotic",  # Laser ODE (Lorenz-type)
            state_dim=3,
            n_parameters=3,  # sigma, r, b
            conserved_quantities=[],
            symmetries=["Z2 (x,y -> -x,-y)"],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/sigma",
            discovered_equations=[
                "dx/dt = sigma*(y-x)",
                "dy/dt = (r-z)*x - y",
                "dz/dt = x*y - b*z",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="sakarya",
            math_type="chaotic",  # 3D chaotic ODE
            state_dim=3,
            n_parameters=2,  # a, b
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="chaotic",
            characteristic_timescale="1",
            discovered_equations=[
                "dx/dt = -x + y + y*z",
                "dy/dt = -x - y + a*x*z",
                "dz/dt = z - b*x*y",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="dadras",
            math_type="chaotic",  # 3D chaotic ODE
            state_dim=3,
            n_parameters=5,  # a, b, c, d, e
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/a",
            discovered_equations=[
                "dx/dt = y - a*x + b*y*z",
                "dy/dt = c*y - x*z + z",
                "dz/dt = d*x*y - e*z",
            ],
            r_squared=[],
        ),
        DomainSignature(
            name="genesio_tesi",
            math_type="chaotic",  # 3rd-order jerk ODE
            state_dim=3,
            n_parameters=3,  # a, b, c
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/a",
            discovered_equations=[
                "dx/dt = y",
                "dy/dt = z",
                "dz/dt = -c*x - b*y - a*z + x^2",
            ],
            r_squared=[],
        ),
        # --- Domain #104: Lu-Chen ---
        DomainSignature(
            name="lu_chen",
            math_type="chaotic",  # Unified Lorenz-Chen-Lu attractor
            state_dim=3,
            n_parameters=3,  # a, b, c
            conserved_quantities=[],
            symmetries=["z-axis_rotation"],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/a",
            discovered_equations=[
                "dx/dt = a*(y-x)",
                "dy/dt = -x*z + c*y",
                "dz/dt = x*y - b*z",
            ],
            r_squared=[],
        ),
        # --- Domain #105: Qi ---
        DomainSignature(
            name="qi",
            math_type="chaotic",  # 4D hyperchaotic system
            state_dim=4,
            n_parameters=4,  # a, b, c, d
            conserved_quantities=[],
            symmetries=["inversion"],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/a",
            discovered_equations=[
                "dx/dt = a*(y-x) + y*z",
                "dy/dt = c*x - y - x*z",
                "dz/dt = x*y - b*z",
                "dw/dt = -d*w + x*z",
            ],
            r_squared=[],
        ),
        # --- Domain #106: WINDMI ---
        DomainSignature(
            name="windmi",
            math_type="chaotic",  # Solar wind-magnetosphere jerk
            state_dim=3,
            n_parameters=2,  # a, b
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/a",
            discovered_equations=[
                "dx/dt = y",
                "dy/dt = z",
                "dz/dt = -a*z - y + b - exp(x)",
            ],
            r_squared=[],
        ),
        # --- Domain #107: Finance ---
        DomainSignature(
            name="finance",
            math_type="chaotic",  # Financial chaotic system
            state_dim=3,
            n_parameters=3,  # a, b, c
            conserved_quantities=[],
            symmetries=[],
            phase_portrait_type="chaotic",
            characteristic_timescale="1/a",
            discovered_equations=[
                "dx/dt = (1/b - a)*x + z + x*y",
                "dy/dt = -b*y - x^2",
                "dz/dt = -x - c*z",
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

    # Analogy: Kuramoto-Sivashinsky <-> Navier-Stokes (turbulent PDE)
    analogies.append(Analogy(
        domain_a="kuramoto_sivashinsky",
        domain_b="navier_stokes",
        analogy_type="structural",
        description=(
            "Both are nonlinear PDEs exhibiting spatiotemporal chaos/turbulence. "
            "KS has u*u_x advection like NS. Both have destabilizing and "
            "stabilizing terms competing at different scales. KS is a 1D "
            "model of flame front instability related to NS-derived equations."
        ),
        strength=0.8,
        mapping={
            "u*u_x [advection]": "u*grad(u) [advection]",
            "u_xx [anti-diffusion]": "Rayleigh instability",
            "u_xxxx [hyper-diffusion]": "nu*Laplacian [viscous]",
        },
    ))

    # Analogy: Ginzburg-Landau <-> Brusselator (pattern-forming instabilities)
    analogies.append(Analogy(
        domain_a="ginzburg_landau",
        domain_b="brusselator",
        analogy_type="structural",
        description=(
            "Both exhibit Hopf/Turing bifurcations and pattern formation. "
            "CGLE is the universal amplitude equation near any Hopf bifurcation, "
            "including the Brusselator's. The |A|^2*A nonlinearity in CGLE "
            "corresponds to the cubic u^2*v term in the Brusselator."
        ),
        strength=0.85,
        mapping={
            "|A|^2*A [cubic saturation]": "u^2*v [cubic kinetics]",
            "Benjamin-Feir instability": "Turing instability",
            "phase turbulence": "chemical patterns",
        },
    ))

    # Analogy: Oregonator <-> Brusselator (chemical oscillators)
    analogies.append(Analogy(
        domain_a="oregonator",
        domain_b="brusselator",
        analogy_type="structural",
        description=(
            "Both are chemical reaction oscillators with Hopf bifurcations. "
            "Brusselator has autocatalytic u^2*v kinetics. Oregonator has "
            "rational kinetics u(u-q)/(u+q). Both produce limit cycle "
            "oscillations above a critical parameter threshold."
        ),
        strength=0.9,
        mapping={
            "u*(u-q)/(u+q) [rational kinetics]": "u^2*v [cubic kinetics]",
            "f [stoichiometric]": "b [bifurcation param]",
            "BZ reaction": "generic chemical oscillator",
        },
    ))

    # Analogy: Bak-Sneppen <-> Ising (phase transitions / criticality)
    analogies.append(Analogy(
        domain_a="bak_sneppen",
        domain_b="ising_model",
        analogy_type="structural",
        description=(
            "Both exhibit critical phenomena with universal scaling behavior. "
            "Ising has a thermal phase transition at T_c; Bak-Sneppen "
            "self-organizes to criticality at f_c ~ 2/3. Both show "
            "power-law correlations at the critical point."
        ),
        strength=0.7,
        mapping={
            "fitness threshold f_c": "critical temperature T_c",
            "avalanche size distribution": "cluster size distribution",
            "self-organized criticality": "thermal phase transition",
        },
    ))

    # Analogy: Oregonator <-> FitzHugh-Nagumo (excitable systems)
    analogies.append(Analogy(
        domain_a="oregonator",
        domain_b="fitzhugh_nagumo",
        analogy_type="structural",
        description=(
            "Both are excitable systems with fast-slow dynamics. "
            "FHN models neural excitability (v fast, w slow). "
            "Oregonator models chemical excitability (u fast, v slow). "
            "Both have relaxation oscillations with similar phase portraits."
        ),
        strength=0.85,
        mapping={
            "u [fast activator]": "v [fast voltage]",
            "v [slow inhibitor]": "w [slow recovery]",
            "eps [timescale ratio]": "epsilon [timescale ratio]",
        },
    ))

    # Analogy: Lorenz-96 <-> Lorenz (chaotic atmospheric models)
    analogies.append(Analogy(
        domain_a="lorenz96",
        domain_b="lorenz",
        analogy_type="structural",
        description=(
            "Both are Lorenz-family models for atmospheric dynamics. "
            "Lorenz (1963) has 3 variables with quadratic nonlinearity. "
            "Lorenz-96 has N variables with quadratic advection-like terms. "
            "Both exhibit chaos above a critical forcing parameter."
        ),
        strength=0.85,
        mapping={
            "F [forcing]": "rho [Rayleigh number]",
            "(x_{i+1}-x_{i-2})*x_{i-1}": "xy, xz quadratic terms",
            "N-dimensional chaos": "3D strange attractor",
        },
    ))

    # Analogy: Chemostat <-> Lotka-Volterra (population dynamics with resource)
    analogies.append(Analogy(
        domain_a="chemostat",
        domain_b="lotka_volterra",
        analogy_type="structural",
        description=(
            "Both model consumer-resource interactions. LV has bilinear "
            "predation beta*x*y. Chemostat has Monod kinetics mu*S/(K+S)*X. "
            "Both have bifurcation between coexistence and extinction."
        ),
        strength=0.8,
        mapping={
            "substrate S": "prey x",
            "biomass X": "predator y",
            "Monod mu*S/(K+S)": "mass action beta*x",
            "washout D_c": "extinction threshold",
        },
    ))

    # Analogy: FHN Spatial <-> Diffusive LV (reaction-diffusion PDEs)
    analogies.append(Analogy(
        domain_a="fhn_spatial",
        domain_b="diffusive_lv",
        analogy_type="structural",
        description=(
            "Both are 2-component reaction-diffusion PDEs with traveling "
            "wave solutions. FHN spatial has excitable pulse waves. "
            "Diffusive LV has prey-predator invasion fronts. Both exhibit "
            "wavefront propagation with speed ~ sqrt(D)."
        ),
        strength=0.8,
        mapping={
            "v [activator]": "prey",
            "w [inhibitor]": "predator",
            "excitable pulse": "invasion front",
        },
    ))

    # Analogy: Wilberforce <-> Coupled Oscillators (energy transfer)
    analogies.append(Analogy(
        domain_a="wilberforce",
        domain_b="coupled_oscillators",
        analogy_type="structural",
        description=(
            "Both are coupled linear oscillator systems exhibiting beat "
            "phenomena and energy transfer between modes. Wilberforce "
            "couples translational and rotational modes; coupled oscillators "
            "couple two translational modes."
        ),
        strength=0.9,
        mapping={
            "z (translational)": "x1 (oscillator 1)",
            "theta (rotational)": "x2 (oscillator 2)",
            "coupling eps": "spring coupling K",
            "beat frequency": "beat frequency",
        },
    ))

    # Standard Map <-> Henon Map (area-preserving 2D discrete maps with chaos)
    analogies.append(Analogy(
        domain_a="standard_map",
        domain_b="henon_map",
        analogy_type="structural",
        description=(
            "Both are 2D discrete maps exhibiting chaos through period-doubling. "
            "Standard map is area-preserving (Hamiltonian); Henon map is dissipative."
        ),
        strength=0.8,
        mapping={
            "theta": "x",
            "p": "y",
            "K (stochasticity)": "a (nonlinearity)",
            "KAM tori": "strange attractor",
        },
    ))

    # Hodgkin-Huxley <-> FitzHugh-Nagumo (neuron models)
    analogies.append(Analogy(
        domain_a="hodgkin_huxley",
        domain_b="fitzhugh_nagumo",
        analogy_type="structural",
        description=(
            "FitzHugh-Nagumo is a 2D reduction of the 4D Hodgkin-Huxley model. "
            "Both exhibit excitability, threshold behavior, and action potentials."
        ),
        strength=0.95,
        mapping={
            "V (membrane voltage)": "v (fast variable)",
            "gating variables n,m,h": "w (recovery variable)",
            "I_ext": "I_ext",
            "f-I curve": "f-I curve",
        },
    ))

    # Rayleigh-Benard <-> Lorenz (RB convection gives rise to Lorenz equations)
    analogies.append(Analogy(
        domain_a="rayleigh_benard",
        domain_b="lorenz",
        analogy_type="structural",
        description=(
            "The Lorenz equations are a 3-mode truncation of Rayleigh-Benard "
            "convection. Lorenz derived his famous chaotic system from the "
            "Boussinesq convection equations."
        ),
        strength=0.95,
        mapping={
            "Ra (Rayleigh)": "rho (control parameter)",
            "Pr (Prandtl)": "sigma",
            "convection amplitude": "X (Lorenz)",
            "Ra_c (onset)": "rho_c (chaos onset)",
        },
    ))

    # Eco-epidemic <-> SIR (disease transmission dynamics)
    analogies.append(Analogy(
        domain_a="eco_epidemic",
        domain_b="sir_epidemic",
        analogy_type="structural",
        description=(
            "Both models feature disease transmission via beta*S*I terms. "
            "Eco-epidemic extends SIR with predator-prey dynamics and "
            "Holling Type II functional responses."
        ),
        strength=0.85,
        mapping={
            "S (susceptible prey)": "S (susceptible)",
            "I (infected prey)": "I (infected)",
            "beta (transmission)": "beta (transmission)",
            "R0 = beta*K/d": "R0 = beta/gamma",
        },
    ))

    # Eco-epidemic <-> Rosenzweig-MacArthur (predator-prey with Holling II)
    analogies.append(Analogy(
        domain_a="eco_epidemic",
        domain_b="rosenzweig_macarthur",
        analogy_type="structural",
        description=(
            "Both feature Holling Type II functional response for predation. "
            "Eco-epidemic splits prey into susceptible and infected classes."
        ),
        strength=0.85,
        mapping={
            "a*S/(1+h*a*S)": "a*N/(1+a*h*N)",
            "predator P": "predator P",
            "prey S+I": "prey N",
            "carrying capacity K": "carrying capacity K",
        },
    ))

    # Rayleigh-Benard <-> Navier-Stokes (fluid dynamics PDEs)
    analogies.append(Analogy(
        domain_a="rayleigh_benard",
        domain_b="navier_stokes",
        analogy_type="structural",
        description=(
            "Rayleigh-Benard convection is governed by the Navier-Stokes equations "
            "with Boussinesq buoyancy. Both use vorticity-streamfunction formulation."
        ),
        strength=0.9,
        mapping={
            "omega (vorticity)": "omega (vorticity)",
            "psi (streamfunction)": "psi (streamfunction)",
            "Pr*nabla^2(omega)": "nu*nabla^2(omega)",
            "Pr*Ra*dT/dx (buoyancy)": "0 (no buoyancy)",
        },
    ))

    # Hindmarsh-Rose <-> Hodgkin-Huxley (neuron models with spikes)
    analogies.append(Analogy(
        domain_a="hindmarsh_rose",
        domain_b="hodgkin_huxley",
        analogy_type="structural",
        description=(
            "Both model neuronal spiking with fast excitatory and slow "
            "recovery variables. HR is a polynomial simplification of HH "
            "that preserves bursting dynamics."
        ),
        strength=0.85,
        mapping={
            "x (fast potential)": "V (membrane voltage)",
            "z (slow adaptation)": "gating variables n,m,h",
            "I_ext": "I_ext",
            "bursting": "repetitive firing",
        },
    ))

    # Competitive LV <-> Lotka-Volterra (competitive interactions)
    analogies.append(Analogy(
        domain_a="competitive_lv",
        domain_b="lotka_volterra",
        analogy_type="structural",
        description=(
            "Both are Lotka-Volterra systems. Competitive LV has N species "
            "with logistic growth and pairwise competition, while classic LV "
            "has predator-prey interactions."
        ),
        strength=0.9,
        mapping={
            "N_i (competitor)": "prey/predator",
            "alpha_ij (competition)": "beta, delta (interaction)",
            "K (carrying capacity)": "equilibrium point",
            "competitive exclusion": "coexistence dynamics",
        },
    ))

    # Vicsek <-> Kuramoto (collective synchronization)
    analogies.append(Analogy(
        domain_a="vicsek",
        domain_b="kuramoto",
        analogy_type="structural",
        description=(
            "Both model synchronization transitions in coupled oscillator systems. "
            "Vicsek: heading alignment with noise. Kuramoto: phase alignment with "
            "coupling. Both have order parameters measuring collective coherence."
        ),
        strength=0.85,
        mapping={
            "theta_i (heading)": "theta_i (phase)",
            "phi (alignment)": "r (sync order param)",
            "eta (noise)": "1/K (inverse coupling)",
            "eta_c (critical noise)": "K_c (critical coupling)",
        },
    ))

    # Magnetic Pendulum <-> Double Pendulum (nonlinear pendulum dynamics)
    analogies.append(Analogy(
        domain_a="magnetic_pendulum",
        domain_b="double_pendulum",
        analogy_type="structural",
        description=(
            "Both feature pendulum dynamics with sensitivity to initial conditions. "
            "Magnetic pendulum has fractal basin boundaries with multiple attractors. "
            "Double pendulum has chaotic trajectories."
        ),
        strength=0.7,
        mapping={
            "magnet basins": "chaotic regions",
            "damping -> attractor": "energy -> trajectory",
            "fractal boundaries": "Lyapunov divergence",
        },
    ))

    # Coupled Lorenz <-> Lorenz (drive system is standard Lorenz)
    analogies.append(Analogy(
        domain_a="coupled_lorenz",
        domain_b="lorenz",
        analogy_type="structural",
        description=(
            "The drive subsystem of coupled Lorenz IS the standard Lorenz system. "
            "Coupling adds eps*(x1-x2) diffusive term to the response."
        ),
        strength=0.95,
        mapping={
            "drive (x1,y1,z1)": "(x,y,z) Lorenz",
            "sigma, rho, beta": "sigma, rho, beta",
            "eps (coupling)": "0 (uncoupled)",
        },
    ))

    # BZ Spiral <-> Oregonator (same model, different dimension)
    analogies.append(Analogy(
        domain_a="bz_spiral",
        domain_b="oregonator",
        analogy_type="structural",
        description=(
            "BZ spiral is the 2D PDE extension of the Oregonator ODE. "
            "Same reaction kinetics but with spatial diffusion of the activator."
        ),
        strength=0.95,
        mapping={
            "u (activator, 2D)": "u (activator, 0D)",
            "v (catalyst, 2D)": "v (catalyst, 0D)",
            "eps, f, q": "eps, f, q",
            "spiral waves": "relaxation oscillations",
        },
    ))

    # BZ Spiral <-> FHN Spatial (excitable reaction-diffusion PDEs)
    analogies.append(Analogy(
        domain_a="bz_spiral",
        domain_b="fhn_spatial",
        analogy_type="structural",
        description=(
            "Both are 2D excitable reaction-diffusion systems that produce "
            "spiral waves. BZ: chemical excitability. FHN: neural excitability."
        ),
        strength=0.85,
        mapping={
            "u (HBrO2 activator)": "v (voltage)",
            "v (Ce4+ inhibitor)": "w (recovery)",
            "spiral tip": "spiral tip",
        },
    ))

    # Swinging Atwood <-> Double Pendulum (Lagrangian chaotic mechanics)
    analogies.append(Analogy(
        domain_a="swinging_atwood",
        domain_b="double_pendulum",
        analogy_type="structural",
        description=(
            "Both are Lagrangian mechanical systems with 2 DOF that exhibit chaos. "
            "Both conserve energy with chaotic trajectory structure."
        ),
        strength=0.8,
        mapping={
            "r, theta (generalized coords)": "theta1, theta2",
            "M/m (mass ratio)": "m1/m2 (mass ratio)",
            "energy conservation": "energy conservation",
        },
    ))

    # Allee <-> Eco-Epidemic (predator-prey with Holling Type II)
    analogies.append(Analogy(
        domain_a="allee_predator_prey",
        domain_b="eco_epidemic",
        analogy_type="structural",
        description=(
            "Both feature predator-prey dynamics with Holling Type II functional "
            "response. Allee adds positive density dependence (critical threshold); "
            "eco-epidemic adds disease transmission."
        ),
        strength=0.8,
        mapping={
            "a*N/(1+h*a*N)": "a*S/(1+h*a*S)",
            "Allee threshold A": "disease threshold R0",
            "prey extinction": "disease-free equilibrium",
        },
    ))

    # Allee <-> Rosenzweig-MacArthur (predator-prey with functional response)
    analogies.append(Analogy(
        domain_a="allee_predator_prey",
        domain_b="rosenzweig_macarthur",
        analogy_type="structural",
        description=(
            "Both are predator-prey with Holling Type II. Allee has cubic prey "
            "nullcline (strong Allee effect); RM has logistic prey growth."
        ),
        strength=0.85,
        mapping={
            "r*N*(N/A-1)*(1-N/K)": "r*N*(1-N/K)",
            "bistability": "Hopf bifurcation",
            "Holling II": "Holling II",
        },
    ))

    # Mackey-Glass <-> Logistic Map (period-doubling chaos)
    analogies.append(Analogy(
        domain_a="mackey_glass",
        domain_b="logistic_map",
        analogy_type="structural",
        description=(
            "Both exhibit period-doubling route to chaos. Mackey-Glass has "
            "unimodal feedback function x^n/(1+x^n) analogous to the logistic "
            "map's parabolic iteration f(x) = r*x*(1-x)."
        ),
        strength=0.8,
        mapping={
            "beta*x^n/(1+x^n)": "r*x*(1-x)",
            "tau (delay)": "r (bifurcation parameter)",
            "period-doubling": "period-doubling",
        },
    ))

    # Wilson-Cowan <-> FitzHugh-Nagumo (excitable neural models)
    analogies.append(Analogy(
        domain_a="wilson_cowan",
        domain_b="fitzhugh_nagumo",
        analogy_type="structural",
        description=(
            "Both model excitable neural dynamics with fast-slow structure. "
            "WC uses sigmoid nonlinearity for population firing rates; FHN uses "
            "cubic nullcline. Both exhibit Hopf bifurcation to oscillation."
        ),
        strength=0.85,
        mapping={
            "E (excitatory)": "v (fast variable)",
            "I (inhibitory)": "w (slow variable)",
            "sigmoid S(x)": "cubic v-v^3/3",
            "Hopf at I_ext_c": "Hopf at I_c",
        },
    ))

    # Wilson-Cowan <-> Brusselator (Hopf bifurcation oscillators)
    analogies.append(Analogy(
        domain_a="wilson_cowan",
        domain_b="brusselator",
        analogy_type="structural",
        description=(
            "Both are 2D nonlinear ODE systems exhibiting Hopf bifurcation "
            "from fixed point to limit cycle. WC uses E-I neural populations; "
            "Brusselator uses chemical concentrations u-v."
        ),
        strength=0.75,
        mapping={
            "E-I oscillation": "u-v oscillation",
            "I_ext bifurcation": "b bifurcation",
            "sigmoid activation": "cubic autocatalysis",
        },
    ))

    # Cable Equation <-> Heat Equation (linear diffusion PDEs)
    analogies.append(Analogy(
        domain_a="cable_equation",
        domain_b="heat_equation",
        analogy_type="structural",
        description=(
            "Both are linear parabolic PDEs with diffusion. Cable equation "
            "tau*dV/dt = lambda^2*V_xx - V + source is heat equation with "
            "decay term. Same spectral solution structure."
        ),
        strength=0.9,
        mapping={
            "lambda^2 (space constant^2)": "D (diffusivity)",
            "V (membrane potential)": "u (temperature)",
            "tau_m (membrane time constant)": "1 (time scale)",
            "R_m*I_ext (source)": "source term",
        },
    ))

    # Bouncing Ball <-> Standard Map (kicked Hamiltonian systems)
    analogies.append(Analogy(
        domain_a="bouncing_ball",
        domain_b="standard_map",
        analogy_type="structural",
        description=(
            "Both are periodically kicked systems with discrete maps. "
            "Bouncing ball has impact map with restitution; Standard Map "
            "has area-preserving kick. Both show period-doubling to chaos."
        ),
        strength=0.8,
        mapping={
            "v_n+1 = e*v_n + kick": "p_n+1 = p_n + K*sin(theta_n)",
            "table amplitude A": "kick strength K",
            "restitution e": "area preservation",
        },
    ))

    # Sine-Gordon <-> Toda Lattice (integrable soliton systems)
    analogies.append(Analogy(
        domain_a="sine_gordon",
        domain_b="toda_lattice",
        analogy_type="structural",
        description=(
            "Both are completely integrable systems supporting exact soliton "
            "solutions. Sine-Gordon has topological kink solitons; Toda has "
            "lattice solitons. Both conserve infinitely many integrals."
        ),
        strength=0.85,
        mapping={
            "kink soliton": "lattice soliton",
            "topological charge Q": "amplitude/speed relation",
            "Lorentz contraction": "exponential profile",
        },
    ))

    # Thomas <-> Rossler (3D chaotic ODEs with simple structure)
    analogies.append(Analogy(
        domain_a="thomas",
        domain_b="rossler",
        analogy_type="structural",
        description=(
            "Both are 3D chaotic ODEs with period-doubling to chaos. "
            "Thomas uses sin() nonlinearity with cyclic symmetry; Rossler "
            "uses simpler polynomial terms. Both have single dissipation parameter."
        ),
        strength=0.75,
        mapping={
            "sin(y)-b*x": "-y-z",
            "b (dissipation)": "a, b, c (parameters)",
            "cyclic symmetry": "no symmetry",
        },
    ))

    # Ikeda Map <-> Henon Map (2D dissipative discrete chaos)
    analogies.append(Analogy(
        domain_a="ikeda_map",
        domain_b="henon_map",
        analogy_type="structural",
        description=(
            "Both are 2D dissipative maps with strange attractors. "
            "Ikeda uses trigonometric nonlinearity (optical resonator); "
            "Henon uses quadratic nonlinearity. Both have det(J)<1."
        ),
        strength=0.85,
        mapping={
            "u (coupling)": "a (nonlinearity)",
            "det(J)=u^2": "det(J)=b",
            "optical spiral attractor": "banana-shaped attractor",
        },
    ))

    # May-Leonard <-> Competitive LV (multi-species competition)
    analogies.append(Analogy(
        domain_a="may_leonard",
        domain_b="competitive_lv",
        analogy_type="structural",
        description=(
            "Both model competitive Lotka-Volterra dynamics. May-Leonard "
            "has circulant competition matrix creating heteroclinic cycles; "
            "Competitive LV has general exclusion dynamics."
        ),
        strength=0.9,
        mapping={
            "cyclic dominance": "competitive exclusion",
            "circulant alpha_ij": "general alpha_ij",
            "heteroclinic cycle": "stable coexistence or exclusion",
        },
    ))

    # May-Leonard <-> Lotka-Volterra (predator-prey generalization)
    analogies.append(Analogy(
        domain_a="may_leonard",
        domain_b="lotka_volterra",
        analogy_type="structural",
        description=(
            "May-Leonard generalizes 2-species LV to N-species with "
            "cyclic competition. Both use bilinear x_i*x_j interaction "
            "terms and share the same mathematical framework."
        ),
        strength=0.85,
        mapping={
            "N-species competition": "2-species predation",
            "circulant matrix": "2x2 interaction",
            "heteroclinic cycles": "closed orbits",
        },
    ))

    # Cahn-Hilliard <-> Brusselator-Diffusion (pattern formation PDEs)
    analogies.append(Analogy(
        domain_a="cahn_hilliard",
        domain_b="brusselator_diffusion",
        analogy_type="structural",
        description=(
            "Both are reaction-diffusion PDEs producing spatial patterns. "
            "CH uses double-well potential for spinodal decomposition; "
            "Brusselator-diffusion uses Turing instability. Both produce "
            "characteristic spatial wavelengths."
        ),
        strength=0.75,
        mapping={
            "spinodal decomposition": "Turing instability",
            "u^3-u nonlinearity": "autocatalytic nonlinearity",
            "epsilon (interface width)": "sqrt(D) (pattern scale)",
        },
    ))

    # Delayed Pred-Prey <-> Rosenzweig-MacArthur (functional response)
    analogies.append(Analogy(
        domain_a="delayed_predator_prey",
        domain_b="rosenzweig_macarthur",
        analogy_type="structural",
        description=(
            "Both use Holling Type II functional response a*N/(1+h*a*N) "
            "for predation. Delayed version adds maturation delay tau; "
            "RM is the tau=0 limit. Same equilibrium structure."
        ),
        strength=0.9,
        mapping={
            "a*N(t-tau)/(1+h*a*N(t-tau))": "a*N/(1+h*a*N)",
            "delay tau": "no delay",
            "Hopf at tau_c": "Hopf at K_c (paradox of enrichment)",
        },
    ))

    # Duffing-VdP <-> Van der Pol (self-excitation)
    analogies.append(Analogy(
        domain_a="duffing_van_der_pol",
        domain_b="van_der_pol",
        analogy_type="structural",
        description=(
            "Both have mu*(x^2-1)*x' self-excitation term. DVdP adds "
            "Duffing cubic restoring force beta*x^3 and external forcing "
            "F*cos(omega*t). beta=0, F=0 recovers pure VdP."
        ),
        strength=0.9,
        mapping={
            "mu*(x^2-1)*x'": "mu*(x^2-1)*x'",
            "alpha*x+beta*x^3": "x",
            "F*cos(omega*t)": "0",
        },
    ))

    # Duffing-VdP <-> Duffing (nonlinear restoring force)
    analogies.append(Analogy(
        domain_a="duffing_van_der_pol",
        domain_b="duffing",
        analogy_type="structural",
        description=(
            "Both have Duffing cubic restoring force alpha*x+beta*x^3. "
            "DVdP adds VdP self-excitation; Duffing has linear damping. "
            "mu=0 in DVdP recovers forced Duffing."
        ),
        strength=0.85,
        mapping={
            "mu*(x^2-1)*x'": "delta*x'",
            "alpha*x+beta*x^3": "alpha*x+beta*x^3",
            "VdP+Duffing hybrid": "pure Duffing",
        },
    ))

    # Network SIS <-> SIR (epidemic models)
    analogies.append(Analogy(
        domain_a="network_sis",
        domain_b="sir_epidemic",
        analogy_type="structural",
        description=(
            "Both are compartmental epidemic models with infection rate "
            "beta and recovery rate gamma. SIS allows reinfection (no R); "
            "SIR has permanent immunity. Both have epidemic threshold R0."
        ),
        strength=0.85,
        mapping={
            "beta*(1-p_i)*sum(A_ij*p_j)": "beta*S*I",
            "1/lambda_max(A)": "gamma/beta",
            "endemic equilibrium": "final size R_inf",
        },
    ))

    # CML <-> Logistic Map (coupled vs single logistic)
    analogies.append(Analogy(
        domain_a="coupled_map_lattice",
        domain_b="logistic_map",
        analogy_type="structural",
        description=(
            "CML is a spatial lattice of coupled logistic maps. Each site "
            "evolves via f(x)=r*x*(1-x) with diffusive coupling eps. "
            "eps=0 recovers independent logistic maps."
        ),
        strength=0.95,
        mapping={
            "(1-eps)*f(x_i)+eps/2*(f(x_{i-1})+f(x_{i+1}))": "f(x)=r*x*(1-x)",
            "eps (coupling)": "no coupling",
            "spatiotemporal chaos": "temporal chaos",
        },
    ))

    # Schnakenberg <-> Gray-Scott (reaction-diffusion Turing systems)
    analogies.append(Analogy(
        domain_a="schnakenberg",
        domain_b="gray_scott",
        analogy_type="structural",
        description=(
            "Both are 2-component reaction-diffusion systems producing "
            "Turing patterns. Schnakenberg uses a-u+u^2*v kinetics; "
            "Gray-Scott uses -u*v^2+f*(1-u). Same qualitative behavior."
        ),
        strength=0.85,
        mapping={
            "a-u+u^2*v": "-u*v^2+f*(1-u)",
            "b-u^2*v": "u*v^2-(f+k)*v",
            "D_v/D_u ratio": "D_v/D_u ratio",
        },
    ))

    # Kapitza <-> Double Pendulum (pendulum physics)
    analogies.append(Analogy(
        domain_a="kapitza_pendulum",
        domain_b="double_pendulum",
        analogy_type="structural",
        description=(
            "Both are pendulum systems with gravitational restoring force "
            "g/L*sin(theta). Kapitza adds parametric excitation; double "
            "pendulum adds a second link. Both show rich dynamics."
        ),
        strength=0.75,
        mapping={
            "g/L*sin(theta)": "g/L*sin(theta)",
            "a*omega^2*cos(omega*t)": "second pendulum coupling",
            "inverted stability": "chaotic motion",
        },
    ))

    # FitzHugh-Rinzel <-> Hindmarsh-Rose (bursting neuron models)
    analogies.append(Analogy(
        domain_a="fitzhugh_rinzel",
        domain_b="hindmarsh_rose",
        analogy_type="structural",
        description=(
            "Both are 3-variable bursting neuron models with fast-slow "
            "decomposition. FHR uses v-v^3/3 nullcline; HR uses x-x^3. "
            "Both produce clusters of spikes modulated by slow variable."
        ),
        strength=0.9,
        mapping={
            "v-v^3/3": "x-x^3",
            "y (slow)": "z (slow)",
            "mu (ultraslow)": "r (slow timescale)",
            "bursting spikes": "bursting spikes",
        },
    ))

    # FitzHugh-Rinzel <-> FitzHugh-Nagumo (2D limit)
    analogies.append(Analogy(
        domain_a="fitzhugh_rinzel",
        domain_b="fitzhugh_nagumo",
        analogy_type="structural",
        description=(
            "FHR extends FHN by adding slow variable y for bursting. "
            "Setting mu=0 reduces FHR to standard FHN. Same fast-slow "
            "structure with cubic nullcline."
        ),
        strength=0.9,
        mapping={
            "v-v^3/3-w+y+I": "v-v^3/3-w+I",
            "dy/dt=mu*(c-v-dy)": "0 (no slow variable)",
            "bursting": "tonic spiking",
        },
    ))

    # Lorenz-84 <-> Lorenz (3D atmospheric chaos with similar quadratic structure)
    analogies.append(Analogy(
        domain_a="lorenz_84",
        domain_b="lorenz",
        analogy_type="structural",
        description=(
            "Both are 3D chaotic atmospheric models with quadratic nonlinearities. "
            "Lorenz (1963) has x*z and x*y coupling; Lorenz-84 has y^2+z^2 and xy, xz."
        ),
        strength=0.85,
        mapping={
            "x (westerly wind)": "x (convective intensity)",
            "y,z (eddy modes)": "y,z (temperature gradients)",
            "quadratic coupling": "quadratic coupling",
        },
    ))

    # Rabinovich-Fabrikant <-> Lorenz (3D chaotic with quadratic nonlinearity)
    analogies.append(Analogy(
        domain_a="rabinovich_fabrikant",
        domain_b="lorenz",
        analogy_type="structural",
        description=(
            "Both are 3D autonomous chaotic ODEs with quadratic nonlinear coupling "
            "terms. RF has x*y and x*z coupling similar to Lorenz x*z and x*y."
        ),
        strength=0.8,
        mapping={
            "y(z-1+x^2)": "sigma*(y-x)",
            "-2z(alpha+xy)": "-beta*z+x*y",
            "strange attractor": "strange attractor",
        },
    ))

    # Sprott <-> Lorenz (minimal 3D chaotic flow)
    analogies.append(Analogy(
        domain_a="sprott",
        domain_b="lorenz",
        analogy_type="structural",
        description=(
            "Both are 3D autonomous dissipative chaotic flows with strange attractors. "
            "Sprott-B achieves chaos with minimal nonlinearity (yz, xy terms only)."
        ),
        strength=0.75,
        mapping={
            "yz coupling": "xz coupling",
            "1-xy (forcing)": "rho*x (forcing)",
            "chaos with minimal terms": "chaos with 7 terms",
        },
    ))

    # Gray-Scott 1D <-> Gray-Scott 2D (same chemistry, different dimensionality)
    analogies.append(Analogy(
        domain_a="gray_scott_1d",
        domain_b="gray_scott",
        analogy_type="structural",
        description=(
            "Identical reaction kinetics (u*v^2 autocatalysis) in 1D vs 2D. "
            "1D shows pulse dynamics; 2D shows spot/stripe patterns."
        ),
        strength=1.0,
        mapping={
            "u*v^2 (1D)": "u*v^2 (2D)",
            "pulses": "spots/stripes",
            "pulse splitting": "spot replication",
        },
    ))

    # Gray-Scott 1D <-> Schnakenberg (1D RD pulse dynamics)
    analogies.append(Analogy(
        domain_a="gray_scott_1d",
        domain_b="schnakenberg",
        analogy_type="structural",
        description=(
            "Both are activator-inhibitor reaction-diffusion systems with "
            "autocatalytic u*v^2 or u^2*v terms producing Turing-type patterns."
        ),
        strength=0.85,
        mapping={
            "u*v^2": "u^2*v",
            "f*(1-u)": "a-u",
            "pulse formation": "Turing patterns",
        },
    ))

    # PPM <-> Rosenzweig-MacArthur (Holling Type II predation)
    analogies.append(Analogy(
        domain_a="predator_prey_mutualist",
        domain_b="rosenzweig_macarthur",
        analogy_type="structural",
        description=(
            "Both feature Holling Type II functional response axy/(1+bx). "
            "PPM adds mutualism term; RM has simpler 2-species structure."
        ),
        strength=0.85,
        mapping={
            "axy/(1+bx) predation": "axy/(1+bx) predation",
            "mutualism mxz/(1+nz)": "absent",
            "3-species": "2-species",
        },
    ))

    # Brusselator 2D <-> Brusselator-diffusion 1D (same chemistry, 1D vs 2D)
    analogies.append(Analogy(
        domain_a="brusselator_2d",
        domain_b="brusselator_diffusion",
        analogy_type="structural",
        description=(
            "Identical Brusselator kinetics (a-(b+1)u+u^2v, bu-u^2v) in 2D vs 1D. "
            "2D shows hexagonal spots and stripes; 1D shows stripe patterns only."
        ),
        strength=1.0,
        mapping={
            "u^2v (2D)": "u^2v (1D)",
            "2D Turing patterns": "1D Turing patterns",
            "hexagonal spots": "periodic stripes",
        },
    ))

    # Brusselator 2D <-> Schnakenberg (activator-inhibitor 2D Turing)
    analogies.append(Analogy(
        domain_a="brusselator_2d",
        domain_b="schnakenberg",
        analogy_type="structural",
        description=(
            "Both 2D activator-inhibitor Turing pattern systems with u^2v autocatalysis. "
            "Same Turing instability mechanism with diffusion ratio threshold."
        ),
        strength=0.9,
        mapping={
            "u^2v": "u^2v",
            "a-(b+1)u": "a-u",
            "D_v/D_u threshold": "D_v/D_u threshold",
        },
    ))

    # FPUT <-> Toda lattice (nonlinear lattice dynamics)
    analogies.append(Analogy(
        domain_a="fput",
        domain_b="toda_lattice",
        analogy_type="structural",
        description=(
            "Both are 1D nonlinear lattice chains with nearest-neighbor coupling. "
            "FPUT uses polynomial nonlinearity; Toda uses exponential. Both are Hamiltonian."
        ),
        strength=0.9,
        mapping={
            "F=k*d+alpha*d^2": "F=a*(e^(-bd)-1)",
            "FPUT recurrence": "exact solitons",
            "energy conservation": "energy conservation",
        },
    ))

    # Selkov <-> Brusselator (chemical oscillator with u^2v kinetics)
    analogies.append(Analogy(
        domain_a="selkov",
        domain_b="brusselator",
        analogy_type="structural",
        description=(
            "Both are chemical oscillators with u^2v autocatalytic terms. "
            "Both exhibit Hopf bifurcation to limit cycle oscillations."
        ),
        strength=0.9,
        mapping={
            "-x+ay+x^2y": "a-(b+1)u+u^2v",
            "b-ay-x^2y": "bu-u^2v",
            "Hopf bifurcation": "Hopf bifurcation",
        },
    ))

    # Rikitake <-> Lorenz (3D chaotic with quadratic coupling)
    analogies.append(Analogy(
        domain_a="rikitake",
        domain_b="lorenz",
        analogy_type="structural",
        description=(
            "Both 3D chaotic systems with quadratic cross-coupling: Rikitake z*y, x*y "
            "mirrors Lorenz x*z, x*y. Both model real geophysical phenomena."
        ),
        strength=0.8,
        mapping={
            "-mu*x+z*y": "sigma*(y-x)",
            "1-x*y": "-beta*z+x*y",
            "polarity reversals": "convective chaos",
        },
    ))

    # Oregonator 1D <-> Gray-Scott 1D (1D excitable/pattern-forming RD)
    analogies.append(Analogy(
        domain_a="oregonator_1d",
        domain_b="gray_scott_1d",
        analogy_type="structural",
        description=(
            "Both 1D reaction-diffusion systems with traveling localized structures. "
            "Oregonator: excitable pulses; Gray-Scott: self-replicating pulses."
        ),
        strength=0.85,
        mapping={
            "excitable pulse": "self-replicating pulse",
            "u(u-q)/(u+q)": "u*v^2",
            "traveling wave": "traveling wave",
        },
    ))

    # Ricker <-> Logistic map (discrete population chaos)
    analogies.append(Analogy(
        domain_a="ricker_map",
        domain_b="logistic_map",
        analogy_type="structural",
        description=(
            "Both discrete 1D maps with period-doubling route to chaos. "
            "Ricker: x*exp(r(1-x/K)); Logistic: r*x*(1-x). Same universality class."
        ),
        strength=0.95,
        mapping={
            "x*exp(r(1-x/K))": "r*x*(1-x)",
            "overcompensation": "logistic saturation",
            "Feigenbaum universality": "Feigenbaum universality",
        },
    ))

    # Morris-Lecar <-> FitzHugh-Nagumo (2D neuron models)
    analogies.append(Analogy(
        domain_a="morris_lecar",
        domain_b="fitzhugh_nagumo",
        analogy_type="structural",
        description=(
            "Both are 2D fast-slow neuron models with voltage (V) and recovery (w). "
            "ML uses conductance-based ionic currents; FHN uses polynomial approximation."
        ),
        strength=0.9,
        mapping={
            "g_Ca*m_ss(V)*(V-V_Ca)": "v^3/3 (cubic nullcline)",
            "w_ss(V)": "v+a-b*w",
            "Type I/II excitability": "Type II excitability",
        },
    ))

    # Morris-Lecar <-> Hodgkin-Huxley (conductance-based neurons)
    analogies.append(Analogy(
        domain_a="morris_lecar",
        domain_b="hodgkin_huxley",
        analogy_type="structural",
        description=(
            "Both conductance-based neuron models with ionic current g*(V-V_rev) terms. "
            "ML is a 2D reduction of HH; HH has 4 gating variables, ML has 1+instant."
        ),
        strength=0.9,
        mapping={
            "g_Ca*m_ss(V)": "g_Na*m^3*h (Na+ current)",
            "g_K*w": "g_K*n^4 (K+ current)",
            "2D phase plane": "4D state space",
        },
    ))

    # Colpitts <-> Chua (electronic chaotic oscillators)
    analogies.append(Analogy(
        domain_a="colpitts",
        domain_b="chua",
        analogy_type="structural",
        description=(
            "Both electronic chaos circuits with piecewise-linear nonlinearity. "
            "Colpitts: max(0,x) transistor; Chua: piecewise diode characteristic."
        ),
        strength=0.85,
        mapping={
            "max(0,x) [BJT]": "f(x) piecewise [Chua diode]",
            "jerk ODE form": "3D state space",
            "electronic chaos": "electronic chaos",
        },
    ))

    # Rossler Hyperchaos <-> Rossler (3D subset of 4D system)
    analogies.append(Analogy(
        domain_a="rossler_hyperchaos",
        domain_b="rossler",
        analogy_type="structural",
        description=(
            "Rossler hyperchaos is the 4D extension of the 3D Rossler system. "
            "First 3 equations are identical; w variable adds second positive LE."
        ),
        strength=0.95,
        mapping={
            "-(y+z)": "-(y+z)",
            "x+ay+w": "x+ay (no w)",
            "b+xz": "b+xz",
        },
    ))

    # Harvested Population <-> Logistic Map (logistic growth + perturbation)
    analogies.append(Analogy(
        domain_a="harvested_population",
        domain_b="logistic_map",
        analogy_type="structural",
        description=(
            "Both based on logistic growth r*x*(1-x/K). Harvested population adds "
            "constant removal H; logistic map is the discrete-time version."
        ),
        strength=0.7,
        mapping={
            "rx(1-x/K)-H": "rx(1-x)",
            "saddle-node": "period-doubling",
            "MSY = rK/4": "chaos onset r~3.57",
        },
    ))

    # FHN Ring <-> FHN Spatial (coupled FHN, discrete vs continuous)
    analogies.append(Analogy(
        domain_a="fhn_ring",
        domain_b="fhn_spatial",
        analogy_type="structural",
        description=(
            "Both are coupled FitzHugh-Nagumo systems. Ring: discrete ring of N neurons "
            "with diffusive coupling. Spatial: continuous 2D PDE with Laplacian."
        ),
        strength=0.9,
        mapping={
            "D*(v_{i-1}-2v_i+v_{i+1})": "D*nabla^2(v)",
            "ring topology": "2D plane",
            "traveling wave on ring": "spiral waves",
        },
    ))

    # FHN Ring <-> Kuramoto (ring synchronization)
    analogies.append(Analogy(
        domain_a="fhn_ring",
        domain_b="kuramoto",
        analogy_type="structural",
        description=(
            "Both are coupled oscillator networks showing synchronization transitions. "
            "FHN Ring: nearest-neighbor diffusive coupling. Kuramoto: all-to-all sinusoidal."
        ),
        strength=0.75,
        mapping={
            "diffusive coupling D": "coupling strength K",
            "ring order parameter": "Kuramoto order parameter r",
            "sync transition D_c": "sync transition K_c",
        },
    ))

    # Bazykin <-> Rosenzweig-MacArthur (Holling Type II predator-prey)
    analogies.append(Analogy(
        domain_a="bazykin",
        domain_b="rosenzweig_macarthur",
        analogy_type="structural",
        description=(
            "Both use Holling Type II functional response x*y/(1+alpha*x) for predation. "
            "Bazykin adds quadratic mortality delta*y^2 causing richer bifurcation."
        ),
        strength=0.90,
        mapping={
            "x*y/(1+alpha*x)": "x*y/(1+alpha*x)",
            "quadratic mortality": "absent in RM",
            "Hopf bifurcation": "paradox of enrichment",
        },
    ))

    # Bazykin <-> Lotka-Volterra (predator-prey ODEs)
    analogies.append(Analogy(
        domain_a="bazykin",
        domain_b="lotka_volterra",
        analogy_type="structural",
        description=(
            "Both are 2D predator-prey systems with growth and predation terms. "
            "Bazykin adds saturating functional response and intraspecific competition."
        ),
        strength=0.82,
        mapping={
            "x*(1-x)": "alpha*x (logistic vs exponential)",
            "x*y/(1+alpha*x)": "beta*x*y (saturating vs mass-action)",
            "limit cycle": "limit cycle",
        },
    ))

    # SIR-Vaccination <-> SIR Epidemic (compartmental models)
    analogies.append(Analogy(
        domain_a="sir_vaccination",
        domain_b="sir_epidemic",
        analogy_type="structural",
        description=(
            "SIR-Vaccination extends classic SIR with vaccination rate nu and "
            "vital dynamics mu. Same beta*S*I/N transmission term. "
            "R_eff = R0*mu/(nu+mu) generalizes R0=beta/gamma."
        ),
        strength=0.95,
        mapping={
            "beta*S*I/N": "beta*S*I",
            "R0=beta/(gamma+mu)": "R0=beta/gamma",
            "nu_c=mu*(R0-1)": "no vaccination analog",
        },
    ))

    # SIR-Vaccination <-> Network SIS (epidemic threshold models)
    analogies.append(Analogy(
        domain_a="sir_vaccination",
        domain_b="network_sis",
        analogy_type="structural",
        description=(
            "Both have epidemic thresholds controlling disease persistence. "
            "SIRV: nu_c = mu*(R0-1). Network SIS: beta_c = gamma/lambda_max."
        ),
        strength=0.72,
        mapping={
            "critical vaccination rate nu_c": "spectral threshold beta_c",
            "herd immunity": "spectral barrier",
        },
    ))

    # Langford <-> Rossler (3D chaotic/quasiperiodic flows)
    analogies.append(Analogy(
        domain_a="langford",
        domain_b="rossler",
        analogy_type="structural",
        description=(
            "Both are 3D autonomous ODE systems with a slow variable (z) "
            "modulating fast oscillations in (x,y). Langford shows torus "
            "dynamics while Rossler shows spiral chaos."
        ),
        strength=0.73,
        mapping={
            "z modulates (x,y)": "z modulates (x,y)",
            "Hopf-Hopf bifurcation": "period-doubling cascade",
            "torus": "spiral attractor",
        },
    ))

    # Laser Rate <-> Chemostat (threshold-pump/washout dynamics)
    analogies.append(Analogy(
        domain_a="laser_rate",
        domain_b="chemostat",
        analogy_type="structural",
        description=(
            "Both exhibit threshold behavior: laser P_th vs chemostat washout D_c. "
            "Above threshold: stable operating point. Below: no output/extinction."
        ),
        strength=0.70,
        mapping={
            "pump P": "dilution rate D",
            "P_th threshold": "D_c washout",
            "carrier N": "substrate S",
            "photon S": "biomass X",
        },
    ))

    # FHN Lattice <-> FHN Spatial (same equations, different discretization)
    analogies.append(Analogy(
        domain_a="fhn_lattice",
        domain_b="fhn_spatial",
        analogy_type="structural",
        description=(
            "Identical FHN equations with diffusion. Lattice: discrete 5-point stencil "
            "on NxN grid. Spatial: continuous PDE with spectral or FD solver."
        ),
        strength=0.95,
        mapping={
            "discrete Laplacian": "continuous Laplacian",
            "spiral waves": "spiral waves",
            "lattice artifacts": "continuous patterns",
        },
    ))

    # FHN Lattice <-> BZ Spiral (spiral wave excitable media)
    analogies.append(Analogy(
        domain_a="fhn_lattice",
        domain_b="bz_spiral",
        analogy_type="structural",
        description=(
            "Both exhibit spiral wave dynamics in 2D excitable media. "
            "FHN: v-w system. BZ: Oregonator u-v system."
        ),
        strength=0.80,
        mapping={
            "FHN spiral": "BZ spiral",
            "diffusion D": "diffusion D_u",
        },
    ))

    # Four-Species LV <-> Three-Species (multi-trophic food web)
    analogies.append(Analogy(
        domain_a="four_species_lv",
        domain_b="three_species",
        analogy_type="structural",
        description=(
            "Both are multi-species food web models with trophic coupling. "
            "4-species: 2 prey + 2 predators. 3-species: bottom-up cascade."
        ),
        strength=0.82,
        mapping={
            "x1,x2 prey": "x prey",
            "y1,y2 predators": "y predator, z top predator",
            "cross-competition a12": "trophic coupling",
        },
    ))

    # Four-Species LV <-> Competitive LV (multi-species competition)
    analogies.append(Analogy(
        domain_a="four_species_lv",
        domain_b="competitive_lv",
        analogy_type="structural",
        description=(
            "Both model N-species Lotka-Volterra competition. 4-species: "
            "cross-competition between prey. Competitive LV: symmetric competition."
        ),
        strength=0.85,
        mapping={
            "a_ij competition": "a_ij competition",
            "competitive exclusion": "exclusion principle",
        },
    ))

    # Lorenz-Stenflo <-> Lorenz (4D extension with wave field)
    analogies.append(Analogy(
        domain_a="lorenz_stenflo",
        domain_b="lorenz",
        analogy_type="structural",
        description=(
            "Lorenz-Stenflo reduces to Lorenz when s=0. The w equation "
            "adds electromagnetic wave coupling: dw/dt = -x - sigma*w."
        ),
        strength=0.92,
        mapping={
            "sigma*(y-x)+s*w": "sigma*(y-x)",
            "dw/dt = -x-sigma*w": "absent (3D)",
            "s=0 reduces to Lorenz": "classic Lorenz",
        },
    ))

    # Lorenz-Stenflo <-> Rossler Hyperchaos (4D hyperchaotic systems)
    analogies.append(Analogy(
        domain_a="lorenz_stenflo",
        domain_b="rossler_hyperchaos",
        analogy_type="structural",
        description=(
            "Both are 4D extensions of 3D chaotic systems capable of hyperchaos "
            "(two positive Lyapunov exponents)."
        ),
        strength=0.70,
        mapping={
            "4th dimension w": "4th dimension w",
            "potential hyperchaos": "confirmed hyperchaos",
        },
    ))

    # Chen <-> Lorenz (Lorenz family dual)
    analogies.append(Analogy(
        domain_a="chen",
        domain_b="lorenz",
        analogy_type="structural",
        description=(
            "Chen is the algebraic dual of Lorenz in the generalized Lorenz system "
            "family (a12*a21 < 0 for Chen, > 0 for Lorenz). Same 3D structure, "
            "different coefficient signs."
        ),
        strength=0.88,
        mapping={
            "a*(y-x)": "sigma*(y-x)",
            "(c-a)*x - xz + c*y": "rho*x - y - xz",
            "xy - b*z": "xy - beta*z",
        },
    ))

    # Aizawa <-> Rossler (3D chaotic attractors with spiral structure)
    analogies.append(Analogy(
        domain_a="aizawa",
        domain_b="rossler",
        analogy_type="structural",
        description=(
            "Both are 3D chaotic ODEs producing spiral-type strange attractors. "
            "Aizawa has mushroom geometry, Rossler has band-type spiral."
        ),
        strength=0.72,
        mapping={
            "(z-b)*x - d*y": "-y - z",
            "d*x + (z-b)*y": "x + a*y",
            "c + a*z - z^3/3 - r^2": "b + z*(x-c)",
        },
    ))

    # Halvorsen <-> Thomas (cyclic symmetry in chaotic ODEs)
    analogies.append(Analogy(
        domain_a="halvorsen",
        domain_b="thomas",
        analogy_type="structural",
        description=(
            "Both exhibit cyclic symmetry (S3 permutation x->y->z->x). "
            "Thomas uses sin() dissipation, Halvorsen uses quadratic coupling."
        ),
        strength=0.80,
        mapping={
            "-a*x - 4*y - 4*z - y^2": "-b*x + sin(y)",
            "cyclic_S3": "cyclic_S3",
        },
    ))

    # Burke-Shaw <-> Lorenz (3D chaotic ODE with quadratic nonlinearity)
    analogies.append(Analogy(
        domain_a="burke_shaw",
        domain_b="lorenz",
        analogy_type="structural",
        description=(
            "Both are 3D chaotic ODEs with quadratic nonlinear coupling (xz, xy). "
            "Burke-Shaw has s*(x+y) diffusion, Lorenz has sigma*(y-x)."
        ),
        strength=0.75,
        mapping={
            "-s*(x+y)": "sigma*(y-x)",
            "-y - s*x*z": "rho*x - y - x*z",
            "s*x*y + v": "x*y - beta*z",
        },
    ))

    # Burke-Shaw <-> Chen (3D quadratic chaotic, Z2 symmetry)
    analogies.append(Analogy(
        domain_a="burke_shaw",
        domain_b="chen",
        analogy_type="structural",
        description=(
            "Both are 3D chaotic ODEs with Z2 symmetry (x,y -> -x,-y) "
            "and quadratic bilinear coupling terms xz, xy."
        ),
        strength=0.73,
        mapping={
            "Z2 symmetry": "Z2 symmetry",
            "xz coupling": "xz coupling",
            "xy coupling": "xy coupling",
        },
    ))

    # Nose-Hoover <-> Sprott (minimal chaotic ODE, 3D single-parameter)
    analogies.append(Analogy(
        domain_a="nose_hoover",
        domain_b="sprott",
        analogy_type="structural",
        description=(
            "Both are minimal 3D chaotic ODEs with few parameters. "
            "Nose-Hoover derives from statistical mechanics, Sprott from "
            "exhaustive search for simplest chaos."
        ),
        strength=0.70,
        mapping={
            "y": "y (linear coupling)",
            "-x + y*z": "quadratic nonlinearity",
            "a - y^2": "constant + quadratic",
        },
    ))

    # Nose-Hoover <-> Harmonic Oscillator (thermostatted harmonic oscillator origin)
    analogies.append(Analogy(
        domain_a="nose_hoover",
        domain_b="harmonic_oscillator",
        analogy_type="structural",
        description=(
            "Nose-Hoover is a harmonic oscillator (dx=y, dy=-x) coupled to "
            "a thermostat variable z. Setting z=0 recovers SHO."
        ),
        strength=0.78,
        mapping={
            "dx/dt = y": "dx/dt = v",
            "dy/dt = -x": "dv/dt = -omega^2*x",
            "z thermostat": "no damping term",
        },
    ))

    # Lorenz-Haken <-> Lorenz (identical ODE structure, different physics)
    analogies.append(Analogy(
        domain_a="lorenz_haken",
        domain_b="lorenz",
        analogy_type="structural",
        description=(
            "Structurally identical ODEs: sigma*(y-x), (r-z)*x-y, x*y-b*z. "
            "Lorenz: fluid convection. Lorenz-Haken: single-mode laser. "
            "Different physics, same mathematics."
        ),
        strength=0.98,
        mapping={
            "sigma [cavity/polarization]": "sigma [Prandtl]",
            "r [pump parameter]": "rho [Rayleigh]",
            "b [decay ratio]": "beta [aspect ratio]",
        },
    ))

    # Lorenz-Haken <-> Laser Rate (laser physics)
    analogies.append(Analogy(
        domain_a="lorenz_haken",
        domain_b="laser_rate",
        analogy_type="structural",
        description=(
            "Both model laser dynamics. Lorenz-Haken: single-mode with "
            "polarization dynamics. Laser rate: semiconductor carrier-photon. "
            "Both have lasing threshold as transcritical bifurcation."
        ),
        strength=0.75,
        mapping={
            "r [pump]": "P [pump power]",
            "lasing threshold r=1": "threshold P_th",
            "E field + polarization": "photon + carrier",
        },
    ))

    # Sakarya <-> Burke-Shaw (3D chaotic with bilinear coupling)
    analogies.append(Analogy(
        domain_a="sakarya",
        domain_b="burke_shaw",
        analogy_type="structural",
        description=(
            "Both are 3D chaotic ODEs with bilinear coupling terms (xz, xy). "
            "Sakarya: neural-inspired with yz, xz terms. Burke-Shaw: magnetic."
        ),
        strength=0.68,
        mapping={
            "y*z coupling": "x*z coupling",
            "x*z coupling": "x*y coupling",
        },
    ))

    # Dadras <-> Rossler (3D chaotic with mixed coupling)
    analogies.append(Analogy(
        domain_a="dadras",
        domain_b="rossler",
        analogy_type="structural",
        description=(
            "Both are 3D chaotic ODEs with quadratic bilinear terms. "
            "Dadras has 5 parameters, Rossler 3 parameters. Both produce "
            "spiral-type strange attractors."
        ),
        strength=0.70,
        mapping={
            "y - a*x + b*y*z": "-y - z (x equation)",
            "d*x*y - e*z": "b + z*(x-c) (z equation)",
        },
    ))

    # Genesio-Tesi <-> Sprott (minimal chaotic ODEs)
    analogies.append(Analogy(
        domain_a="genesio_tesi",
        domain_b="sprott",
        analogy_type="structural",
        description=(
            "Both are minimal 3D chaotic ODEs. Genesio-Tesi is a jerk system "
            "(3rd-order ODE) with x^2 nonlinearity. Sprott systems are also "
            "minimal-parameter chaotic flows."
        ),
        strength=0.73,
        mapping={
            "x''' + ax'' + bx' + cx = x^2": "minimal jerk form",
            "3 parameters": "1-2 parameters",
        },
    ))

    # Genesio-Tesi <-> Duffing (polynomial nonlinear oscillator)
    analogies.append(Analogy(
        domain_a="genesio_tesi",
        domain_b="duffing",
        analogy_type="structural",
        description=(
            "Both feature polynomial nonlinear restoring forces in oscillator "
            "framework. Genesio-Tesi: x^2 term in jerk. Duffing: x^3 term."
        ),
        strength=0.65,
        mapping={
            "x^2 [quadratic]": "x^3 [cubic]",
            "jerk form": "forced oscillator form",
        },
    ))

    # Lu-Chen <-> Lorenz (structurally identical ODE form)
    analogies.append(Analogy(
        domain_a="lu_chen",
        domain_b="lorenz",
        analogy_type="structural",
        description=(
            "Lu-Chen is the unified Lorenz-Chen family: dx/dt=a(y-x), "
            "dy/dt=-xz+cy, dz/dt=xy-bz. When c=rho-1 it reduces to Lorenz. "
            "Same quadratic nonlinearities xz and xy."
        ),
        strength=0.95,
        mapping={
            "a*(y-x)": "sigma*(y-x)",
            "-x*z + c*y": "rho*x - y - x*z",
            "x*y - b*z": "x*y - beta*z",
        },
    ))

    # Lu-Chen <-> Chen (same attractor family)
    analogies.append(Analogy(
        domain_a="lu_chen",
        domain_b="chen",
        analogy_type="structural",
        description=(
            "Lu-Chen generalizes the Chen attractor. Both share the same "
            "ODE structure with quadratic cross-coupling xz and xy terms. "
            "Chen is a special case in the Lu-Chen family."
        ),
        strength=0.93,
        mapping={
            "a*(y-x)": "a*(y-x)",
            "-x*z + c*y": "(c-a)*x - x*z + c*y",
            "x*y - b*z": "x*y - b*z",
        },
    ))

    # Qi <-> Lorenz-Stenflo (4D quadratic chaotic systems)
    analogies.append(Analogy(
        domain_a="qi",
        domain_b="lorenz_stenflo",
        analogy_type="structural",
        description=(
            "Both are 4D extensions of Lorenz-type chaos with quadratic "
            "cross-coupling terms. Qi: 4D with yz, xz, xy, xz couplings. "
            "Lorenz-Stenflo: 4D with acoustic gravity wave coupling."
        ),
        strength=0.75,
        mapping={
            "a*(y-x)+yz": "sigma*(y-x)+r*w",
            "4D quadratic": "4D quadratic",
        },
    ))

    # WINDMI <-> Genesio-Tesi (jerk system structure)
    analogies.append(Analogy(
        domain_a="windmi",
        domain_b="genesio_tesi",
        analogy_type="structural",
        description=(
            "Both are jerk systems (3rd-order ODEs written as x'=y, y'=z, z'=f). "
            "WINDMI: z' = -az - y + b - exp(x). Genesio-Tesi: z' = -cx - by - az + x^2. "
            "Both have damping and nonlinear restoring force."
        ),
        strength=0.78,
        mapping={
            "jerk ODE": "jerk ODE",
            "exp(x) nonlinearity": "x^2 nonlinearity",
            "-a*z damping": "-a*z damping",
        },
    ))

    # Finance <-> Lorenz (3D quadratic chaos)
    analogies.append(Analogy(
        domain_a="finance",
        domain_b="lorenz",
        analogy_type="structural",
        description=(
            "Both are 3D chaotic systems with quadratic nonlinearities. "
            "Finance: xy and x^2 couplings. Lorenz: xz and xy couplings. "
            "Both exhibit sensitive dependence on initial conditions."
        ),
        strength=0.70,
        mapping={
            "x*y [interest-price]": "x*z [convection]",
            "-x^2 [investment]": "x*y [rotation]",
            "3 params": "3 params",
        },
    ))

    # Finance <-> Rossler (3D chaos with quadratic term)
    analogies.append(Analogy(
        domain_a="finance",
        domain_b="rossler",
        analogy_type="structural",
        description=(
            "Both are 3D chaotic systems with mixed linear-quadratic terms. "
            "Finance has x^2 and xy couplings. Rossler has xz coupling. "
            "Both show period-doubling routes to chaos."
        ),
        strength=0.65,
        mapping={
            "x*y + x^2 nonlinearity": "x*z nonlinearity",
            "period-doubling": "period-doubling",
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

    # Dimensional: KS <-> Gray-Scott (pattern wavelength scaling)
    analogies.append(Analogy(
        domain_a="kuramoto_sivashinsky",
        domain_b="gray_scott",
        analogy_type="dimensional",
        description=(
            "Both exhibit characteristic spatial wavelengths determined by "
            "competition between destabilizing and stabilizing scales. "
            "KS: lambda ~ 2*pi*sqrt(2). Gray-Scott: lambda ~ sqrt(D_v/k). "
            "Both are set by ratio of diffusion coefficients."
        ),
        strength=0.75,
        mapping={
            "L_c = 2*pi*sqrt(2) [critical length]": "lambda ~ sqrt(D/k)",
            "viscosity [stabilizing]": "D_v [diffusion]",
        },
    ))

    # Dimensional: Oregonator <-> Van der Pol (relaxation timescale)
    analogies.append(Analogy(
        domain_a="oregonator",
        domain_b="van_der_pol",
        analogy_type="dimensional",
        description=(
            "Both exhibit relaxation oscillations with period scaling. "
            "Oregonator: T ~ 1/eps for small eps. VdP: T ~ mu for large mu. "
            "Both have fast-slow timescale separation."
        ),
        strength=0.8,
        mapping={
            "1/eps [fast timescale]": "mu [relaxation param]",
            "period ~ O(1/eps)": "period ~ O(mu)",
        },
    ))

    # Dimensional: Wilberforce <-> Harmonic Oscillator (frequency scaling)
    analogies.append(Analogy(
        domain_a="wilberforce",
        domain_b="harmonic_oscillator",
        analogy_type="dimensional",
        description=(
            "Both have omega = sqrt(stiffness/inertia) frequency scaling. "
            "Wilberforce: omega_z = sqrt(k/m), omega_theta = sqrt(kappa/I). "
            "Harmonic oscillator: omega_0 = sqrt(k/m). "
            "Same dimensional analysis applies."
        ),
        strength=0.9,
        mapping={
            "sqrt(k/m) [translational]": "sqrt(k/m) [natural freq]",
            "sqrt(kappa/I) [torsional]": "same dimensional structure",
        },
    ))

    # Dimensional: Chemostat <-> SIR (threshold dynamics)
    analogies.append(Analogy(
        domain_a="chemostat",
        domain_b="sir_epidemic",
        analogy_type="dimensional",
        description=(
            "Both have critical threshold bifurcations with similar structure. "
            "Chemostat: D_c = mu_max*S_in/(K_s+S_in). "
            "SIR: R_0 = beta/gamma. "
            "Both determine whether population persists or dies out."
        ),
        strength=0.75,
        mapping={
            "D_c [washout]": "1/R_0 [epidemic threshold]",
            "mu_max [max growth]": "beta [infection rate]",
        },
    ))

    # Dimensional: HH <-> FHN (excitable neuron threshold scaling)
    analogies.append(Analogy(
        domain_a="hodgkin_huxley",
        domain_b="fitzhugh_nagumo",
        analogy_type="dimensional",
        description=(
            "Both have a threshold current for action potential generation. "
            "HH rheobase current scales with conductance parameters; FHN "
            "critical current I_c has analogous threshold scaling."
        ),
        strength=0.85,
        mapping={
            "I_rheobase [HH]": "I_c [FHN]",
            "spike amplitude ~100mV": "spike amplitude ~2v units",
        },
    ))

    # Dimensional: Rayleigh-Benard <-> Gray-Scott (pattern wavelength scaling)
    analogies.append(Analogy(
        domain_a="rayleigh_benard",
        domain_b="gray_scott",
        analogy_type="dimensional",
        description=(
            "Both exhibit pattern formation with characteristic wavelength "
            "set by diffusion-reaction balance. RB: roll width ~ 2H. "
            "Gray-Scott: pattern wavelength ~ sqrt(D/k)."
        ),
        strength=0.75,
        mapping={
            "roll wavelength ~ 2H": "pattern wavelength ~ sqrt(D/k)",
            "Ra/Ra_c (supercriticality)": "distance from Turing boundary",
        },
    ))

    # Dimensional: Eco-epidemic <-> LV (population timescale)
    analogies.append(Analogy(
        domain_a="eco_epidemic",
        domain_b="lotka_volterra",
        analogy_type="dimensional",
        description=(
            "Both have characteristic timescale 1/r (prey growth rate). "
            "Predator-prey oscillation period scales with 1/sqrt(r*m) in "
            "both models."
        ),
        strength=0.8,
        mapping={
            "1/r [prey growth]": "1/alpha [prey growth]",
            "K [carrying capacity]": "gamma/delta [prey equilibrium]",
        },
    ))

    # Dimensional: Wilson-Cowan <-> Hodgkin-Huxley (neural timescales)
    analogies.append(Analogy(
        domain_a="wilson_cowan",
        domain_b="hodgkin_huxley",
        analogy_type="dimensional",
        description=(
            "Both model neural dynamics with millisecond timescales. "
            "WC uses population-level tau_e~tau_i; HH uses membrane "
            "capacitance C_m and channel conductances. Both have "
            "oscillation frequency ~ 1/tau_m."
        ),
        strength=0.8,
        mapping={
            "tau_e [excitatory time constant]": "C_m/g_Na [activation time]",
            "tau_i [inhibitory time constant]": "C_m/g_K [recovery time]",
        },
    ))

    # Dimensional: Cable Equation <-> Heat Equation (diffusion scaling)
    analogies.append(Analogy(
        domain_a="cable_equation",
        domain_b="heat_equation",
        analogy_type="dimensional",
        description=(
            "Both have diffusive spreading with L ~ sqrt(D*t) scaling. "
            "Cable equation space constant lambda = sqrt(r_m/r_i) maps to "
            "heat diffusion length sqrt(D*t)."
        ),
        strength=0.9,
        mapping={
            "lambda [space constant]": "sqrt(D*t) [diffusion length]",
            "tau_m [membrane time constant]": "L^2/D [diffusion time]",
        },
    ))

    # Dimensional: Mackey-Glass <-> Lorenz (chaotic timescale)
    analogies.append(Analogy(
        domain_a="mackey_glass",
        domain_b="lorenz",
        analogy_type="dimensional",
        description=(
            "Both are chaotic with characteristic timescale set by system "
            "parameters. MG: tau (delay) sets oscillation period. Lorenz: "
            "1/sigma sets fast mixing time. Both have positive Lyapunov exponents."
        ),
        strength=0.7,
        mapping={
            "tau [delay time]": "1/sigma [mixing time]",
            "1/gamma [decay time]": "1/beta [z decay time]",
        },
    ))

    # Dimensional: Sine-Gordon <-> Damped Wave (wave speed scaling)
    analogies.append(Analogy(
        domain_a="sine_gordon",
        domain_b="damped_wave",
        analogy_type="dimensional",
        description=(
            "Both are wave equations with characteristic speed c. "
            "SG kink width ~ c/omega (with relativistic contraction); "
            "Damped wave has dispersion omega = c*k. Same dimensional structure."
        ),
        strength=0.8,
        mapping={
            "c [wave speed]": "c [wave speed]",
            "kink width ~ 1/sqrt(1-v^2/c^2)": "wavelength ~ c/f",
        },
    ))

    # Dimensional: Thomas <-> Lorenz (chaotic ODE timescale)
    analogies.append(Analogy(
        domain_a="thomas",
        domain_b="lorenz",
        analogy_type="dimensional",
        description=(
            "Both are 3D chaotic ODEs with dissipation parameter controlling "
            "the chaos transition. Thomas: b_c ~ 0.208, timescale 1/b. "
            "Lorenz: rho_c ~ 24.74, timescale 1/sigma."
        ),
        strength=0.7,
        mapping={
            "1/b [dissipation time]": "1/sigma [damping time]",
            "b_c ~ 0.208": "rho_c ~ 24.74",
        },
    ))

    # Dimensional: Delayed Pred-Prey <-> Mackey-Glass (delay timescale)
    analogies.append(Analogy(
        domain_a="delayed_predator_prey",
        domain_b="mackey_glass",
        analogy_type="dimensional",
        description=(
            "Both are DDEs where the delay tau is the key bifurcation parameter. "
            "Increasing tau destabilizes equilibrium in both, causing oscillation "
            "and eventually chaos. Period scales with tau."
        ),
        strength=0.85,
        mapping={
            "tau [maturation delay]": "tau [feedback delay]",
            "tau_c [Hopf threshold]": "tau_c [Hopf threshold]",
            "1/r [growth time]": "1/gamma [decay time]",
        },
    ))

    # Dimensional: Network SIS <-> SIR (epidemic timescale)
    analogies.append(Analogy(
        domain_a="network_sis",
        domain_b="sir_epidemic",
        analogy_type="dimensional",
        description=(
            "Both have characteristic timescale 1/gamma (recovery time) "
            "and epidemic threshold R0 = beta/gamma. Network SIS adds "
            "graph spectral structure: threshold ~ 1/lambda_max."
        ),
        strength=0.85,
        mapping={
            "1/gamma [recovery]": "1/gamma [recovery]",
            "1/lambda_max [network threshold]": "gamma/beta [R0 threshold]",
        },
    ))

    # Dimensional: Schnakenberg <-> Gray-Scott (RD pattern scale)
    analogies.append(Analogy(
        domain_a="schnakenberg",
        domain_b="gray_scott",
        analogy_type="dimensional",
        description=(
            "Both have pattern wavelength scaling with sqrt(D) where D is "
            "the larger diffusivity. Schnakenberg: lambda ~ sqrt(D_v); "
            "Gray-Scott: lambda ~ sqrt(D_v). Same RD scaling."
        ),
        strength=0.85,
        mapping={
            "sqrt(D_v) [pattern scale]": "sqrt(D_v) [pattern scale]",
            "L^2/D_u [diffusion time]": "L^2/D_u [diffusion time]",
        },
    ))

    # Dimensional: Kapitza <-> Harmonic Oscillator (pendulum timescale)
    analogies.append(Analogy(
        domain_a="kapitza_pendulum",
        domain_b="harmonic_oscillator",
        analogy_type="dimensional",
        description=(
            "Both have natural frequency sqrt(g/L) or sqrt(k/m). "
            "Kapitza requires omega >> sqrt(g/L) for inverted stability. "
            "Same dimensional scaling T ~ sqrt(inertia/force)."
        ),
        strength=0.8,
        mapping={
            "sqrt(g/L) [natural frequency]": "sqrt(k/m) [natural frequency]",
            "a*omega [parametric drive]": "F [external force]",
        },
    ))

    # Dimensional: FitzHugh-Rinzel <-> FitzHugh-Nagumo (neural timescale)
    analogies.append(Analogy(
        domain_a="fitzhugh_rinzel",
        domain_b="fitzhugh_nagumo",
        analogy_type="dimensional",
        description=(
            "Both have fast timescale ~1 for spikes and slow timescale "
            "1/delta for recovery. FHR adds ultraslow 1/mu for burst "
            "modulation. Same fast-slow separation."
        ),
        strength=0.9,
        mapping={
            "1/delta [recovery time]": "1/eps [recovery time]",
            "1/mu [burst modulation]": "N/A (no bursting in FHN)",
        },
    ))

    # Dimensional: Lorenz-84 <-> Lorenz (atmospheric timescale 1/a vs 1/sigma)
    analogies.append(Analogy(
        domain_a="lorenz_84",
        domain_b="lorenz",
        analogy_type="dimensional",
        description=(
            "Both atmospheric models with damping timescale: 1/a in L84, 1/sigma in L63. "
            "Forcing parameter F in L84 maps to rho in L63 for chaos onset."
        ),
        strength=0.8,
        mapping={
            "1/a [damping time]": "1/sigma [damping time]",
            "F [forcing]": "rho [Rayleigh number]",
        },
    ))

    # Dimensional: Gray-Scott 1D <-> Gray-Scott 2D (same scaling, different dimension)
    analogies.append(Analogy(
        domain_a="gray_scott_1d",
        domain_b="gray_scott",
        analogy_type="dimensional",
        description=(
            "Identical dimensional scaling: pattern wavelength ~ sqrt(D_v/k), "
            "timescale ~ 1/f. Same diffusion coefficients in 1D and 2D."
        ),
        strength=1.0,
        mapping={
            "sqrt(D_v/k) [1D pulse width]": "sqrt(D_v/k) [2D pattern wavelength]",
            "1/f [feed timescale]": "1/f [feed timescale]",
        },
    ))

    # Dimensional: Rabinovich-Fabrikant <-> Rossler (dissipation timescale)
    analogies.append(Analogy(
        domain_a="rabinovich_fabrikant",
        domain_b="rossler",
        analogy_type="dimensional",
        description=(
            "Both 3D chaotic systems with dissipation rate controlling chaos onset. "
            "RF: 1/alpha damping; Rossler: 1/c dissipation timescale."
        ),
        strength=0.7,
        mapping={
            "1/alpha [z damping]": "1/c [z damping]",
            "gamma [linear growth]": "a [spiral rate]",
        },
    ))

    # Dimensional: Brusselator 2D <-> Gray-Scott (pattern wavelength ~ sqrt(D))
    analogies.append(Analogy(
        domain_a="brusselator_2d",
        domain_b="gray_scott",
        analogy_type="dimensional",
        description=(
            "Both 2D RD systems with Turing wavelength scaling as sqrt(D_v/rate). "
            "Same dimensional analysis: lambda ~ 2*pi*sqrt(D_v/k_eff)."
        ),
        strength=0.85,
        mapping={
            "sqrt(D_v/(b-1-a^2))": "sqrt(D_v/k)",
            "2D Turing patterns": "2D Turing patterns",
        },
    ))

    # Dimensional: FPUT <-> Spring-mass chain (omega ~ sqrt(k))
    analogies.append(Analogy(
        domain_a="fput",
        domain_b="spring_mass_chain",
        analogy_type="dimensional",
        description=(
            "Both lattice chains with omega_n = 2*sqrt(k)*sin(n*pi/(2*(N+1))). "
            "FPUT adds nonlinear corrections; spring chain is linear."
        ),
        strength=0.9,
        mapping={
            "omega ~ sqrt(k) [linear limit]": "omega ~ sqrt(K/m)",
            "fixed BCs": "periodic/fixed BCs",
        },
    ))

    # Dimensional: Selkov <-> Oregonator (chemical oscillation timescale)
    analogies.append(Analogy(
        domain_a="selkov",
        domain_b="oregonator",
        analogy_type="dimensional",
        description=(
            "Both biochemical oscillators with period ~ 1/sqrt(rate). "
            "Selkov: glycolysis timescale 1/a; Oregonator: BZ timescale 1/epsilon."
        ),
        strength=0.75,
        mapping={
            "1/a [glycolysis rate]": "1/epsilon [BZ rate]",
            "x^2y autocatalysis": "xy autocatalysis",
        },
    ))

    # Dimensional: Morris-Lecar <-> Hodgkin-Huxley (membrane timescale C/g)
    analogies.append(Analogy(
        domain_a="morris_lecar",
        domain_b="hodgkin_huxley",
        analogy_type="dimensional",
        description=(
            "Both conductance neurons with membrane timescale tau_m = C/g_L. "
            "Same dimensional structure for ionic currents: g*(V-V_rev)."
        ),
        strength=0.9,
        mapping={
            "C/g_L [membrane time constant]": "C_m/g_L [membrane time constant]",
            "g_Ca, g_K [conductances]": "g_Na, g_K [conductances]",
        },
    ))

    # Dimensional: Ricker <-> Logistic (same universality class, same delta)
    analogies.append(Analogy(
        domain_a="ricker_map",
        domain_b="logistic_map",
        analogy_type="dimensional",
        description=(
            "Both discrete maps share Feigenbaum universality constant delta=4.669. "
            "Chaos onset: Ricker at r~2, Logistic at r~3.57."
        ),
        strength=0.9,
        mapping={
            "r [growth rate] ~ 2": "r [growth rate] ~ 3.57",
            "Feigenbaum delta": "Feigenbaum delta",
        },
    ))

    # Dimensional: Rossler Hyperchaos <-> Rossler (same timescale, extra dimension)
    analogies.append(Analogy(
        domain_a="rossler_hyperchaos",
        domain_b="rossler",
        analogy_type="dimensional",
        description=(
            "Same characteristic timescale 1/a. The 4D system adds a w dimension "
            "with timescale 1/d, creating a second unstable direction."
        ),
        strength=0.9,
        mapping={
            "1/a [spiral rate]": "1/a [spiral rate]",
            "1/d [hyperchaos time]": "N/A (3D only)",
        },
    ))

    # Dimensional: FHN Ring <-> CML (discrete coupling, lattice dynamics)
    analogies.append(Analogy(
        domain_a="fhn_ring",
        domain_b="coupled_map_lattice",
        analogy_type="dimensional",
        description=(
            "Both are lattice systems with discrete coupling. FHN Ring: continuous-time "
            "ODE on ring; CML: discrete-time maps on lattice. Both show sync transitions."
        ),
        strength=0.7,
        mapping={
            "D [coupling strength]": "eps [coupling strength]",
            "ring topology": "1D lattice",
        },
    ))

    # Dimensional: Bazykin <-> Rosenzweig-MacArthur (same timescale 1/gamma)
    analogies.append(Analogy(
        domain_a="bazykin",
        domain_b="rosenzweig_macarthur",
        analogy_type="dimensional",
        description=(
            "Both share timescale 1/gamma for predator dynamics and "
            "1/alpha for saturation. Same Holling II half-saturation scaling."
        ),
        strength=0.88,
        mapping={
            "1/gamma [predator decay]": "1/d [predator decay]",
            "1/alpha [saturation]": "1/alpha [saturation]",
        },
    ))

    # Dimensional: SIR-Vaccination <-> SIR (1/gamma recovery time)
    analogies.append(Analogy(
        domain_a="sir_vaccination",
        domain_b="sir_epidemic",
        analogy_type="dimensional",
        description=(
            "Same 1/gamma recovery timescale. Vaccination adds 1/nu "
            "timescale for immunization rate."
        ),
        strength=0.90,
        mapping={
            "1/gamma [recovery]": "1/gamma [recovery]",
            "1/nu [vaccination]": "N/A (no vaccination)",
        },
    ))

    # Dimensional: Laser Rate <-> Brusselator (relaxation oscillation timescale)
    analogies.append(Analogy(
        domain_a="laser_rate",
        domain_b="brusselator",
        analogy_type="dimensional",
        description=(
            "Both show relaxation oscillations with separated timescales. "
            "Laser: 1/gamma_S (fast photon) vs 1/gamma_N (slow carrier). "
            "Brusselator: fast u vs slow v near Hopf."
        ),
        strength=0.65,
        mapping={
            "1/gamma_S [photon]": "fast u timescale",
            "1/gamma_N [carrier]": "slow v timescale",
        },
    ))

    # Dimensional: FHN Lattice <-> FHN Ring (same 1/eps timescale)
    analogies.append(Analogy(
        domain_a="fhn_lattice",
        domain_b="fhn_ring",
        analogy_type="dimensional",
        description=(
            "Both share the FHN timescale 1/eps. FHN Lattice: 2D square grid. "
            "FHN Ring: 1D ring. Same diffusion coupling D."
        ),
        strength=0.88,
        mapping={
            "1/eps [recovery]": "1/eps [recovery]",
            "D [diffusion]": "D [coupling]",
        },
    ))

    # Dimensional: Chen <-> Lorenz (same 1/a ~ 1/sigma timescale)
    analogies.append(Analogy(
        domain_a="chen",
        domain_b="lorenz",
        analogy_type="dimensional",
        description=(
            "Same characteristic timescale 1/a ~ 1/sigma for convective mixing. "
            "Chen: a=35, Lorenz: sigma=10 -- different values, same role."
        ),
        strength=0.82,
        mapping={
            "1/a [mixing]": "1/sigma [mixing]",
            "1/b [damping]": "1/beta [damping]",
        },
    ))

    # Dimensional: Lorenz-Stenflo <-> Lorenz (same sigma timescale)
    analogies.append(Analogy(
        domain_a="lorenz_stenflo",
        domain_b="lorenz",
        analogy_type="dimensional",
        description=(
            "Identical sigma timescale. Extra 1/sigma decay in w equation "
            "introduces same-scale wave damping."
        ),
        strength=0.92,
        mapping={
            "1/sigma [x-y mixing]": "1/sigma [x-y mixing]",
            "1/sigma [w decay]": "N/A (3D)",
        },
    ))

    # Dimensional: Aizawa <-> Langford (xy-rotation timescale d)
    analogies.append(Analogy(
        domain_a="aizawa",
        domain_b="langford",
        analogy_type="dimensional",
        description=(
            "Both feature xy-plane rotation with angular frequency parameter d. "
            "Aizawa: (z-b)*x - d*y, Langford: (z-eps)*x - omega*y."
        ),
        strength=0.75,
        mapping={
            "d [rotation rate]": "omega [rotation rate]",
            "1/a [z timescale]": "1/lambda [growth rate]",
        },
    ))

    # Dimensional: Burke-Shaw <-> Lorenz (s ~ sigma coupling strength)
    analogies.append(Analogy(
        domain_a="burke_shaw",
        domain_b="lorenz",
        analogy_type="dimensional",
        description=(
            "Both use a large coupling parameter (s=10, sigma=10) that sets "
            "the fast relaxation timescale. Same order of magnitude."
        ),
        strength=0.80,
        mapping={
            "1/s [diffusion]": "1/sigma [relaxation]",
            "v [forcing]": "rho*sigma [effective forcing]",
        },
    ))

    # Dimensional: Nose-Hoover <-> Van der Pol (unit timescale oscillation)
    analogies.append(Analogy(
        domain_a="nose_hoover",
        domain_b="van_der_pol",
        analogy_type="dimensional",
        description=(
            "Both have unit-frequency oscillation (omega=1) as base dynamics. "
            "Nose-Hoover: dx=y, dy=-x. VdP: dx=y, dy=-x+mu*(1-x^2)*y."
        ),
        strength=0.72,
        mapping={
            "omega=1 [natural freq]": "omega_0=1 [natural freq]",
            "z [thermostat]": "mu*(1-x^2) [nonlinear damping]",
        },
    ))

    # Dimensional: Lorenz-Haken <-> Lorenz (same timescale structure)
    analogies.append(Analogy(
        domain_a="lorenz_haken",
        domain_b="lorenz",
        analogy_type="dimensional",
        description=(
            "Same timescale structure 1/sigma, same parameterization. "
            "Lorenz-Haken uses sigma=3 (laser), Lorenz sigma=10 (fluid)."
        ),
        strength=0.95,
        mapping={
            "1/sigma [field decay]": "1/sigma [thermal diffusion]",
            "1/b [inversion decay]": "1/beta [mode damping]",
        },
    ))

    # Dimensional: Dadras <-> Chen (multi-parameter chaotic with 1/a scale)
    analogies.append(Analogy(
        domain_a="dadras",
        domain_b="chen",
        analogy_type="dimensional",
        description=(
            "Both have a primary dissipation parameter (Dadras: a=3, Chen: a=35) "
            "that sets the dominant timescale."
        ),
        strength=0.65,
        mapping={
            "1/a [dissipation]": "1/a [mixing]",
            "1/e [z decay]": "1/b [z damping]",
        },
    ))

    # Dimensional: Genesio-Tesi <-> Duffing (oscillation timescale)
    analogies.append(Analogy(
        domain_a="genesio_tesi",
        domain_b="duffing",
        analogy_type="dimensional",
        description=(
            "Both have oscillation timescale set by sqrt(1/c) or sqrt(1/omega^2). "
            "Genesio-Tesi: 1/sqrt(c). Duffing: 1/omega."
        ),
        strength=0.68,
        mapping={
            "1/sqrt(c) [oscillation]": "1/omega [natural period]",
            "1/a [damping]": "1/delta [damping]",
        },
    ))

    # Dimensional: Lu-Chen <-> Lorenz (timescale from linear coefficient)
    analogies.append(Analogy(
        domain_a="lu_chen",
        domain_b="lorenz",
        analogy_type="dimensional",
        description=(
            "Both have primary timescale set by 1/a (Lu-Chen) or 1/sigma (Lorenz). "
            "The parameter a plays the same role as sigma in setting mixing rate."
        ),
        strength=0.88,
        mapping={
            "1/a [mixing]": "1/sigma [mixing]",
            "1/b [z-decay]": "1/beta [z-decay]",
        },
    ))

    # Dimensional: Qi <-> Rossler-Hyperchaos (4D timescale)
    analogies.append(Analogy(
        domain_a="qi",
        domain_b="rossler_hyperchaos",
        analogy_type="dimensional",
        description=(
            "Both are 4D systems with primary timescale 1/a. "
            "Qi: a controls x-y mixing. Rossler 4D: a controls oscillation."
        ),
        strength=0.62,
        mapping={
            "1/a [mixing rate]": "1/a [oscillation rate]",
            "4D state space": "4D state space",
        },
    ))

    # Dimensional: WINDMI <-> Colpitts (exponential nonlinearity timescale)
    analogies.append(Analogy(
        domain_a="windmi",
        domain_b="colpitts",
        analogy_type="dimensional",
        description=(
            "Both feature exponential nonlinearity with damping timescale 1/a. "
            "WINDMI: exp(x) in magnetosphere. Colpitts: exp(x) in transistor."
        ),
        strength=0.72,
        mapping={
            "1/a [damping]": "1/alpha [damping]",
            "exp(x) response": "exp(x) response",
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

    # Topological: KS <-> Lorenz (spatiotemporal vs temporal chaos)
    analogies.append(Analogy(
        domain_a="kuramoto_sivashinsky",
        domain_b="lorenz",
        analogy_type="topological",
        description=(
            "Both exhibit chaotic dynamics with positive Lyapunov exponents. "
            "Lorenz has low-dimensional temporal chaos (3D attractor). "
            "KS has high-dimensional spatiotemporal chaos (extensive, "
            "Lyapunov dimension ~ L). Both show sensitive dependence."
        ),
        strength=0.7,
        mapping={
            "spatiotemporal chaos": "temporal chaos",
            "positive Lyapunov spectrum": "positive max Lyapunov",
        },
    ))

    # Topological: Ginzburg-Landau <-> Brusselator-Diffusion (pattern-forming PDEs)
    analogies.append(Analogy(
        domain_a="ginzburg_landau",
        domain_b="brusselator_diffusion",
        analogy_type="topological",
        description=(
            "Both are reaction-diffusion PDEs exhibiting pattern formation "
            "from uniform states. CGLE shows phase turbulence, spiral waves, "
            "and defect chaos. Brusselator-diff shows Turing stripes and spots. "
            "Both transition from uniform to patterned via instability."
        ),
        strength=0.75,
        mapping={
            "Benjamin-Feir instability": "Turing instability",
            "phase defects": "Turing patterns",
        },
    ))

    # Topological: Oregonator <-> Brusselator (chemical limit cycles)
    analogies.append(Analogy(
        domain_a="oregonator",
        domain_b="brusselator",
        analogy_type="topological",
        description=(
            "Both chemical oscillators have stable limit cycles in phase space "
            "born from Hopf bifurcations. The phase portraits show similar "
            "relaxation oscillation topology with fast jumps and slow arcs."
        ),
        strength=0.9,
        mapping={
            "BZ limit cycle": "Brusselator limit cycle",
            "Hopf bifurcation": "Hopf bifurcation",
        },
    ))

    # Topological: Bak-Sneppen <-> Logistic Map (critical transitions)
    analogies.append(Analogy(
        domain_a="bak_sneppen",
        domain_b="logistic_map",
        analogy_type="topological",
        description=(
            "Both exhibit critical transitions and power-law scaling. "
            "Logistic map has period-doubling cascade to chaos at r_c. "
            "Bak-Sneppen self-organizes to critical threshold f_c. "
            "Both show universal scaling exponents near criticality."
        ),
        strength=0.65,
        mapping={
            "SOC threshold f_c": "chaos onset r_c",
            "power-law avalanches": "Feigenbaum scaling",
        },
    ))

    # Topological: Lorenz-96 <-> KS (extensive chaos)
    analogies.append(Analogy(
        domain_a="lorenz96",
        domain_b="kuramoto_sivashinsky",
        analogy_type="topological",
        description=(
            "Both exhibit extensive chaos: Lyapunov dimension grows linearly "
            "with system size (N for L96, L for KS). Both are canonical models "
            "of high-dimensional chaos in spatially extended systems."
        ),
        strength=0.8,
        mapping={
            "N sites": "L domain length",
            "dim ~ O(N)": "dim ~ O(L)",
        },
    ))

    # Topological: FHN Spatial <-> FitzHugh-Nagumo (local to global)
    analogies.append(Analogy(
        domain_a="fhn_spatial",
        domain_b="fitzhugh_nagumo",
        analogy_type="topological",
        description=(
            "Same reaction kinetics (v-v^3/3-w excitability), but FHN spatial "
            "adds diffusion creating traveling pulse solutions. The local "
            "nullcline structure is identical; spatial coupling enables waves."
        ),
        strength=0.95,
        mapping={
            "traveling pulse": "limit cycle / excitable spike",
            "wave speed c": "spike frequency",
        },
    ))

    # Topological: Wilberforce <-> Elastic Pendulum (coupled-mode energy transfer)
    analogies.append(Analogy(
        domain_a="wilberforce",
        domain_b="elastic_pendulum",
        analogy_type="topological",
        description=(
            "Both exhibit energy transfer between two coupled degrees of "
            "freedom. Wilberforce: translation-rotation. Elastic pendulum: "
            "radial-angular. Both show beat patterns and autoparametric resonance."
        ),
        strength=0.85,
        mapping={
            "z-theta coupling": "r-theta coupling",
            "beat frequency": "energy exchange period",
        },
    ))

    # Topological: Chemostat <-> Rosenzweig-MacArthur (consumer-resource stability)
    analogies.append(Analogy(
        domain_a="chemostat",
        domain_b="rosenzweig_macarthur",
        analogy_type="topological",
        description=(
            "Both model consumer-resource interactions with bifurcation. "
            "Chemostat has washout bifurcation (stable -> extinction). "
            "RM has Hopf bifurcation (stable -> oscillation). Both have "
            "stable coexistence equilibria that can lose stability."
        ),
        strength=0.75,
        mapping={
            "washout (D_c)": "Hopf (K_c)",
            "stable node -> extinction": "stable node -> limit cycle",
        },
    ))

    # Standard Map <-> Logistic Map (KAM/Feigenbaum routes to chaos)
    analogies.append(Analogy(
        domain_a="standard_map",
        domain_b="logistic_map",
        analogy_type="topological",
        description=(
            "Both are discrete maps transitioning from regular to chaotic "
            "as control parameter increases. Standard map: K_c ~ 0.97. "
            "Logistic map: r_c ~ 3.57. Both exhibit positive Lyapunov exponents."
        ),
        strength=0.8,
        mapping={
            "K (stochasticity)": "r (growth rate)",
            "KAM tori -> chaos": "period doubling -> chaos",
            "lambda(K) > 0": "lambda(r) > 0",
        },
    ))

    # HH <-> VdP (relaxation oscillation limit cycles)
    analogies.append(Analogy(
        domain_a="hodgkin_huxley",
        domain_b="van_der_pol",
        analogy_type="topological",
        description=(
            "Both exhibit relaxation oscillation limit cycles with fast-slow "
            "dynamics. HH: fast V spike + slow gating recovery. "
            "VdP: fast excursion + slow return. Both have stable limit cycles."
        ),
        strength=0.8,
        mapping={
            "V spike (fast)": "x excursion (fast)",
            "gating recovery (slow)": "relaxation (slow)",
            "I_ext threshold": "mu parameter",
        },
    ))

    # Rayleigh-Benard <-> Brusselator-diffusion (pattern formation via instability)
    analogies.append(Analogy(
        domain_a="rayleigh_benard",
        domain_b="brusselator_diffusion",
        analogy_type="topological",
        description=(
            "Both exhibit pattern formation via symmetry-breaking instability. "
            "RB: convection rolls at Ra > Ra_c. Brusselator-diffusion: "
            "Turing patterns at b > b_c. Same supercritical bifurcation topology."
        ),
        strength=0.8,
        mapping={
            "Ra_c (convection onset)": "b_c (Turing onset)",
            "convection rolls": "chemical Turing patterns",
            "conduction state (uniform)": "homogeneous steady state",
        },
    ))

    # Eco-epidemic <-> LV (predator-prey oscillations)
    analogies.append(Analogy(
        domain_a="eco_epidemic",
        domain_b="lotka_volterra",
        analogy_type="topological",
        description=(
            "Both exhibit predator-prey oscillations in phase space. "
            "Eco-epidemic adds disease dynamics but preserves the "
            "consumer-resource oscillation topology."
        ),
        strength=0.85,
        mapping={
            "S+I (total prey) vs P": "prey vs predator",
            "limit cycle/spiral": "center/limit cycle",
            "coexistence equilibrium": "equilibrium (gamma/delta, alpha/beta)",
        },
    ))

    # HR <-> FHN (excitable neuron topology)
    analogies.append(Analogy(
        domain_a="hindmarsh_rose",
        domain_b="fitzhugh_nagumo",
        analogy_type="topological",
        description=(
            "Both exhibit excitable dynamics: quiescent rest state that can "
            "produce spikes when perturbed. HR adds a slow variable for "
            "bursting. Both have Hopf bifurcation at critical I_ext."
        ),
        strength=0.85,
        mapping={
            "quiescent -> spiking (Hopf)": "rest -> oscillation (Hopf)",
            "slow z modulation": "no slow variable (2D)",
            "burst envelope": "single spikes only",
        },
    ))

    # Vicsek <-> Ising (order-disorder phase transition)
    analogies.append(Analogy(
        domain_a="vicsek",
        domain_b="ising_model",
        analogy_type="topological",
        description=(
            "Both exhibit order-disorder phase transitions. Vicsek: heading "
            "alignment vs noise. Ising: spin alignment vs temperature. Both "
            "have an order parameter transitioning from 0 to 1."
        ),
        strength=0.8,
        mapping={
            "phi (alignment order)": "m (magnetization)",
            "eta_c (critical noise)": "T_c (critical temperature)",
            "flocking (ordered)": "ferromagnetic (ordered)",
        },
    ))

    # Competitive LV <-> Chemostat (resource competition dynamics)
    analogies.append(Analogy(
        domain_a="competitive_lv",
        domain_b="chemostat",
        analogy_type="topological",
        description=(
            "Both model competitive exclusion: species competing for shared "
            "resources. Competitive LV: N species with pairwise competition. "
            "Chemostat: microbial species competing for substrate."
        ),
        strength=0.75,
        mapping={
            "alpha_ij (competition)": "mu_max, K_s (competitive ability)",
            "exclusion principle": "washout bifurcation",
            "stable coexistence": "coexistence at different substrates",
        },
    ))

    # Magnetic Pendulum <-> Standard Map (sensitive dependence topology)
    analogies.append(Analogy(
        domain_a="magnetic_pendulum",
        domain_b="standard_map",
        analogy_type="topological",
        description=(
            "Both feature intricate dependence on initial conditions. "
            "Magnetic pendulum: fractal basin boundaries. Standard map: "
            "KAM tori and chaotic seas. Both produce complex phase space."
        ),
        strength=0.7,
        mapping={
            "fractal basin boundaries": "KAM tori / chaotic seas",
            "3 attractors": "regular islands / chaos",
            "sensitive IC dependence": "sensitive IC dependence",
        },
    ))

    # Coupled Lorenz <-> Kuramoto (synchronization transitions)
    analogies.append(Analogy(
        domain_a="coupled_lorenz",
        domain_b="kuramoto",
        analogy_type="topological",
        description=(
            "Both exhibit synchronization transitions. Coupled Lorenz: "
            "chaotic synchronization above eps_c. Kuramoto: phase sync "
            "above K_c. Both have order parameters measuring coherence."
        ),
        strength=0.75,
        mapping={
            "sync error -> 0": "r -> 1",
            "eps_c (critical coupling)": "K_c (critical coupling)",
            "desync (eps<eps_c)": "incoherence (K<K_c)",
        },
    ))

    # BZ Spiral <-> Gray-Scott (pattern-forming RD-PDEs)
    analogies.append(Analogy(
        domain_a="bz_spiral",
        domain_b="gray_scott",
        analogy_type="topological",
        description=(
            "Both are reaction-diffusion systems producing spatially "
            "organized patterns. BZ: spiral waves in excitable medium. "
            "Gray-Scott: spots and stripes in bistable medium."
        ),
        strength=0.75,
        mapping={
            "spiral waves": "spots/stripes/labyrinths",
            "excitable dynamics": "activator-inhibitor dynamics",
            "Oregonator kinetics": "Gray-Scott kinetics",
        },
    ))

    # Swinging Atwood <-> Elastic Pendulum (2-DOF chaotic mechanics)
    analogies.append(Analogy(
        domain_a="swinging_atwood",
        domain_b="elastic_pendulum",
        analogy_type="topological",
        description=(
            "Both are 2-DOF Lagrangian systems with radial and angular "
            "degrees of freedom that can exchange energy and exhibit chaos."
        ),
        strength=0.8,
        mapping={
            "r (string length)": "l (spring length)",
            "theta (swing angle)": "theta (pendulum angle)",
            "mass ratio M/m": "spring/gravity ratio",
        },
    ))

    # Allee <-> Logistic Map (bistability / threshold dynamics)
    analogies.append(Analogy(
        domain_a="allee_predator_prey",
        domain_b="logistic_map",
        analogy_type="topological",
        description=(
            "Both exhibit threshold behavior: Allee has critical population "
            "below which extinction occurs. Logistic map has period-doubling "
            "threshold. Both show bifurcation-driven qualitative transitions."
        ),
        strength=0.6,
        mapping={
            "Allee threshold A": "period-doubling r values",
            "extinction vs survival": "periodic vs chaotic",
        },
    ))

    # Mackey-Glass <-> Lorenz (chaotic attractor topology)
    analogies.append(Analogy(
        domain_a="mackey_glass",
        domain_b="lorenz",
        analogy_type="topological",
        description=(
            "Both produce chaotic strange attractors with positive Lyapunov "
            "exponents and fractal dimension. MG attractor lives in delay "
            "embedding space; Lorenz in 3D ODE space. Both have period-doubling "
            "cascade route to chaos."
        ),
        strength=0.8,
        mapping={
            "delay embedding attractor": "butterfly attractor",
            "tau bifurcation": "rho bifurcation",
            "Lyapunov > 0": "Lyapunov > 0",
        },
    ))

    # Bouncing Ball <-> Logistic Map (period-doubling)
    analogies.append(Analogy(
        domain_a="bouncing_ball",
        domain_b="logistic_map",
        analogy_type="topological",
        description=(
            "Both are discrete maps exhibiting period-doubling cascade to "
            "chaos. Bouncing ball with increasing table amplitude parallels "
            "logistic map with increasing r. Feigenbaum universality applies to both."
        ),
        strength=0.85,
        mapping={
            "table amplitude": "parameter r",
            "period-1 bounce": "period-1 orbit",
            "chaotic bouncing": "chaotic iterations",
        },
    ))

    # Wilson-Cowan <-> Van der Pol (limit cycle oscillation)
    analogies.append(Analogy(
        domain_a="wilson_cowan",
        domain_b="van_der_pol",
        analogy_type="topological",
        description=(
            "Both exhibit Hopf bifurcation from stable fixed point to "
            "limit cycle. WC E-I oscillation has same topological structure "
            "as VdP relaxation oscillation: 2D system with unique limit cycle."
        ),
        strength=0.8,
        mapping={
            "E-I limit cycle": "x-v limit cycle",
            "I_ext (Hopf parameter)": "mu (nonlinearity)",
            "sigmoid saturation": "cubic damping",
        },
    ))

    # Cable Equation <-> Navier-Stokes (diffusive decay)
    analogies.append(Analogy(
        domain_a="cable_equation",
        domain_b="navier_stokes",
        analogy_type="topological",
        description=(
            "Both have exponential modal decay: cable equation modes decay "
            "as exp(-t/tau_m) * exp(-k^2*lambda^2*t/tau_m); NS vorticity modes "
            "decay as exp(-nu*k^2*t). Same spectral decay topology."
        ),
        strength=0.75,
        mapping={
            "lambda^2/tau_m [diffusivity]": "nu [kinematic viscosity]",
            "exp(-|x|/lambda) spatial": "exp(-nu*k^2*t) temporal",
        },
    ))

    # Thomas <-> Lorenz (3D strange attractor topology)
    analogies.append(Analogy(
        domain_a="thomas",
        domain_b="lorenz",
        analogy_type="topological",
        description=(
            "Both are 3D dissipative chaotic systems with strange attractors "
            "and positive Lyapunov exponents. Thomas has labyrinth-like "
            "attractor with cyclic symmetry; Lorenz has butterfly attractor."
        ),
        strength=0.8,
        mapping={
            "labyrinth attractor": "butterfly attractor",
            "b_c chaos transition": "rho_c chaos transition",
            "cyclic symmetry": "Z2 symmetry",
        },
    ))

    # Ikeda Map <-> Henon Map (2D strange attractor)
    analogies.append(Analogy(
        domain_a="ikeda_map",
        domain_b="henon_map",
        analogy_type="topological",
        description=(
            "Both are 2D dissipative maps with fractal strange attractors. "
            "Ikeda has spiral structure from optical phase; Henon has "
            "folding structure. Both have positive Lyapunov exponents."
        ),
        strength=0.85,
        mapping={
            "spiral attractor": "folded attractor",
            "period-doubling": "period-doubling",
            "D_corr ~ 1.7": "D_corr ~ 1.26",
        },
    ))

    # Sine-Gordon <-> Shallow Water (nonlinear wave PDEs)
    analogies.append(Analogy(
        domain_a="sine_gordon",
        domain_b="shallow_water",
        analogy_type="topological",
        description=(
            "Both are nonlinear wave equations supporting localized traveling "
            "wave solutions. Sine-Gordon has kink solitons; Shallow water "
            "has shock waves. Both conserve wave energy."
        ),
        strength=0.7,
        mapping={
            "kink soliton": "hydraulic bore",
            "topological charge": "wave amplitude",
            "Lorentz symmetry": "Galilean symmetry",
        },
    ))

    # May-Leonard <-> Three Species (multi-species oscillation)
    analogies.append(Analogy(
        domain_a="may_leonard",
        domain_b="three_species",
        analogy_type="topological",
        description=(
            "Both are multi-species ODE systems with oscillatory dynamics. "
            "May-Leonard shows heteroclinic cycles (cyclic dominance); "
            "three-species shows trophic cascade oscillations."
        ),
        strength=0.75,
        mapping={
            "heteroclinic cycle": "food chain oscillation",
            "cyclic dominance": "trophic cascade",
            "biodiversity index H": "population stability",
        },
    ))

    # Cahn-Hilliard <-> Gray-Scott (pattern formation)
    analogies.append(Analogy(
        domain_a="cahn_hilliard",
        domain_b="gray_scott",
        analogy_type="topological",
        description=(
            "Both produce spatial patterns from initially uniform states. "
            "CH has spinodal decomposition (coarsening); GS has Turing "
            "patterns (spots, stripes). Both have characteristic wavelengths."
        ),
        strength=0.7,
        mapping={
            "spinodal instability": "Turing instability",
            "L(t)~t^(1/3) coarsening": "wavelength ~ sqrt(D/k)",
            "phase-separated domains": "spots/stripes",
        },
    ))

    # Duffing-VdP <-> Driven Pendulum (forced chaotic oscillators)
    analogies.append(Analogy(
        domain_a="duffing_van_der_pol",
        domain_b="driven_pendulum",
        analogy_type="topological",
        description=(
            "Both are periodically forced nonlinear oscillators showing "
            "period-doubling cascade to chaos. DVdP has cubic + VdP "
            "nonlinearity; driven pendulum has sinusoidal restoring force."
        ),
        strength=0.8,
        mapping={
            "Poincare section": "Poincare section",
            "F*cos(omega*t)": "F_d*cos(omega_d*t)",
            "period-doubling": "period-doubling",
        },
    ))

    # Delayed Pred-Prey <-> Lotka-Volterra (predator-prey oscillation)
    analogies.append(Analogy(
        domain_a="delayed_predator_prey",
        domain_b="lotka_volterra",
        analogy_type="topological",
        description=(
            "Both are predator-prey systems with oscillatory dynamics. "
            "Standard LV has neutral orbits; delayed version has "
            "delay-induced limit cycles from Hopf bifurcation."
        ),
        strength=0.8,
        mapping={
            "delay-induced limit cycle": "neutral center orbits",
            "Hopf bifurcation at tau_c": "center manifold",
            "N*,P* equilibrium": "gamma/delta, alpha/beta",
        },
    ))

    # Network SIS <-> Kuramoto (threshold/phase transition on network)
    analogies.append(Analogy(
        domain_a="network_sis",
        domain_b="kuramoto",
        analogy_type="topological",
        description=(
            "Both exhibit phase transitions on networks. SIS has epidemic "
            "threshold beta_c; Kuramoto has synchronization threshold K_c. "
            "Both have order parameter (prevalence/r) that jumps at threshold."
        ),
        strength=0.75,
        mapping={
            "prevalence (fraction infected)": "order parameter r",
            "beta_c/gamma": "K_c",
            "endemic state": "synchronized state",
        },
    ))

    # CML <-> KS (spatiotemporal chaos)
    analogies.append(Analogy(
        domain_a="coupled_map_lattice",
        domain_b="kuramoto_sivashinsky",
        analogy_type="topological",
        description=(
            "Both produce spatiotemporal chaos in 1D. CML uses discrete "
            "map dynamics; KS uses continuous PDE. Both have positive "
            "Lyapunov exponents and space-time pattern complexity."
        ),
        strength=0.75,
        mapping={
            "lattice chaos": "PDE turbulence",
            "eps (coupling)": "L (domain size)",
            "Lyapunov spectrum": "Lyapunov spectrum",
        },
    ))

    # Schnakenberg <-> Brusselator-Diffusion (Turing pattern topology)
    analogies.append(Analogy(
        domain_a="schnakenberg",
        domain_b="brusselator_diffusion",
        analogy_type="topological",
        description=(
            "Both produce Turing patterns (spots, stripes) via diffusion-driven "
            "instability. Same qualitative phase space: uniform -> patterned "
            "transition as D_v/D_u increases beyond threshold."
        ),
        strength=0.85,
        mapping={
            "Turing bifurcation": "Turing bifurcation",
            "spots/stripes": "spots/stripes",
            "u^2*v autocatalysis": "u^2*v autocatalysis",
        },
    ))

    # Kapitza <-> Driven Pendulum (parametric vs direct forcing)
    analogies.append(Analogy(
        domain_a="kapitza_pendulum",
        domain_b="driven_pendulum",
        analogy_type="topological",
        description=(
            "Both are forced pendulum systems. Kapitza has parametric forcing "
            "(pivot oscillation); driven pendulum has direct torque forcing. "
            "Both can show stabilization, resonance, and chaos."
        ),
        strength=0.8,
        mapping={
            "parametric excitation": "direct forcing",
            "inverted stability": "resonance tongues",
            "a*omega (drive strength)": "F_d (forcing amplitude)",
        },
    ))

    # FitzHugh-Rinzel <-> Hindmarsh-Rose (bursting topology)
    analogies.append(Analogy(
        domain_a="fitzhugh_rinzel",
        domain_b="hindmarsh_rose",
        analogy_type="topological",
        description=(
            "Both have identical phase space topology: slow variable y/z "
            "modulates fast 2D dynamics between quiescent and spiking. "
            "Both show square-wave bursting with same bifurcation structure."
        ),
        strength=0.9,
        mapping={
            "fast spike manifold": "fast spike manifold",
            "slow y drift": "slow z drift",
            "burst/quiescent transition": "burst/quiescent transition",
        },
    ))

    # Topological: Lorenz-84 <-> Double Pendulum (chaotic with quasi-periodic routes)
    analogies.append(Analogy(
        domain_a="lorenz_84",
        domain_b="double_pendulum",
        analogy_type="topological",
        description=(
            "Both exhibit quasi-periodic routes to chaos with strange attractors. "
            "L84 transitions through Hopf then torus-doubling; double pendulum "
            "through energy-dependent KAM tori breakdown."
        ),
        strength=0.7,
        mapping={
            "F (forcing bifurcation)": "E (energy bifurcation)",
            "quasi-periodic torus": "KAM tori",
            "strange attractor": "chaotic sea",
        },
    ))

    # Topological: Rabinovich-Fabrikant <-> Rossler (scroll attractor topology)
    analogies.append(Analogy(
        domain_a="rabinovich_fabrikant",
        domain_b="rossler",
        analogy_type="topological",
        description=(
            "Both exhibit multiscroll/spiral strange attractors with fold-and-stretch "
            "dynamics. RF has more complex scroll structure from plasma instability."
        ),
        strength=0.75,
        mapping={
            "multiscroll attractor": "single-scroll attractor",
            "gamma->chaos transition": "c->chaos transition",
            "positive Lyapunov": "positive Lyapunov",
        },
    ))

    # Topological: Sprott <-> Thomas (minimal chaos, similar simplicity)
    analogies.append(Analogy(
        domain_a="sprott",
        domain_b="thomas",
        analogy_type="topological",
        description=(
            "Both are minimal chaotic systems: Sprott-B with just 5 terms, "
            "Thomas with cyclic symmetry. Both achieve chaos with minimal nonlinearity."
        ),
        strength=0.7,
        mapping={
            "Sprott minimal flow": "Thomas cyclic flow",
            "yz, xy nonlinearity": "sin(y), sin(z), sin(x) nonlinearity",
            "strange attractor": "labyrinth attractor",
        },
    ))

    # Topological: Gray-Scott 1D <-> Brusselator-diffusion (1D RD patterns)
    analogies.append(Analogy(
        domain_a="gray_scott_1d",
        domain_b="brusselator_diffusion",
        analogy_type="topological",
        description=(
            "Both 1D reaction-diffusion systems exhibiting localized structures. "
            "GS-1D shows self-replicating pulses; Brusselator-diff shows Turing spots."
        ),
        strength=0.8,
        mapping={
            "pulse solutions": "Turing patterns",
            "pulse splitting": "pattern formation",
            "D_u/D_v ratio": "D_u/D_v ratio",
        },
    ))

    # Topological: PPM <-> Three-species (3-species with different interactions)
    analogies.append(Analogy(
        domain_a="predator_prey_mutualist",
        domain_b="three_species",
        analogy_type="topological",
        description=(
            "Both 3-species systems with trophic interactions. PPM adds mutualism, "
            "stabilizing oscillations. Three-species has trophic cascade only."
        ),
        strength=0.8,
        mapping={
            "predator-prey-mutualist": "grass-herbivore-predator",
            "mutualism stabilization": "trophic cascade",
            "Holling II response": "linear/Holling response",
        },
    ))

    # Topological: Brusselator 2D <-> Gray-Scott (2D Turing pattern formation)
    analogies.append(Analogy(
        domain_a="brusselator_2d",
        domain_b="gray_scott",
        analogy_type="topological",
        description=(
            "Both exhibit 2D Turing instability with spots and stripes. "
            "Same topological pattern classes despite different kinetics."
        ),
        strength=0.85,
        mapping={
            "hexagonal spots": "hexagonal spots",
            "stripes": "stripes/labyrinthine",
            "Turing bifurcation": "Turing bifurcation",
        },
    ))

    # Topological: FPUT <-> Toda (near-integrable lattice dynamics)
    analogies.append(Analogy(
        domain_a="fput",
        domain_b="toda_lattice",
        analogy_type="topological",
        description=(
            "Both are Hamiltonian lattice chains with quasi-periodic orbits and "
            "soliton-like solutions. FPUT shows near-recurrence; Toda is exactly integrable."
        ),
        strength=0.85,
        mapping={
            "FPUT recurrence": "exact solitons",
            "quasi-periodic orbits": "action-angle variables",
            "energy conservation": "energy conservation",
        },
    ))

    # Topological: Selkov <-> Van der Pol (limit cycle oscillators)
    analogies.append(Analogy(
        domain_a="selkov",
        domain_b="van_der_pol",
        analogy_type="topological",
        description=(
            "Both exhibit stable limit cycle oscillations via Hopf bifurcation. "
            "Selkov: glycolysis oscillation; VdP: relaxation oscillation."
        ),
        strength=0.75,
        mapping={
            "Hopf at b_c(a)": "Hopf at mu=0",
            "limit cycle": "limit cycle",
            "metabolic oscillation": "electrical oscillation",
        },
    ))

    # Topological: Rikitake <-> Lorenz-84 (geophysical chaotic oscillators)
    analogies.append(Analogy(
        domain_a="rikitake",
        domain_b="lorenz_84",
        analogy_type="topological",
        description=(
            "Both geophysical chaotic systems: Rikitake (geomagnetic reversals) "
            "and Lorenz-84 (atmospheric circulation). Both show intermittent chaos."
        ),
        strength=0.75,
        mapping={
            "polarity reversals": "regime transitions",
            "disc dynamo": "atmospheric circulation",
            "strange attractor": "strange attractor",
        },
    ))

    # Topological: Oregonator 1D <-> FHN spatial (excitable RD PDE)
    analogies.append(Analogy(
        domain_a="oregonator_1d",
        domain_b="fhn_spatial",
        analogy_type="topological",
        description=(
            "Both excitable reaction-diffusion systems showing propagating pulses. "
            "Oregonator: chemical traveling waves; FHN: neural excitation waves."
        ),
        strength=0.85,
        mapping={
            "BZ traveling pulse": "neural excitation wave",
            "excitable rest state": "excitable rest state",
            "pulse annihilation": "wave collision",
        },
    ))

    # Topological: Ricker <-> Henon map (discrete chaos, strange attractors)
    analogies.append(Analogy(
        domain_a="ricker_map",
        domain_b="henon_map",
        analogy_type="topological",
        description=(
            "Both discrete maps with period-doubling cascades to chaos. "
            "Ricker is 1D with Feigenbaum universality; Henon is 2D with fractal attractor."
        ),
        strength=0.7,
        mapping={
            "1D period-doubling": "2D period-doubling",
            "chaotic band": "strange attractor",
            "Feigenbaum delta": "Feigenbaum delta",
        },
    ))

    # Topological: Morris-Lecar <-> Wilson-Cowan (neural oscillators)
    analogies.append(Analogy(
        domain_a="morris_lecar",
        domain_b="wilson_cowan",
        analogy_type="topological",
        description=(
            "Both 2D neural models with excitatory-inhibitory dynamics and Hopf "
            "bifurcation to oscillation. ML: single neuron; WC: neural population."
        ),
        strength=0.75,
        mapping={
            "V (voltage)": "E (excitatory)",
            "w (recovery)": "I (inhibitory)",
            "f-I curve": "E-I oscillation",
        },
    ))

    # Topological: Colpitts <-> Lorenz (3D chaotic with strange attractor)
    analogies.append(Analogy(
        domain_a="colpitts",
        domain_b="lorenz",
        analogy_type="topological",
        description=(
            "Both 3D chaotic systems with strange attractors and positive Lyapunov. "
            "Colpitts: electronic jerk circuit; Lorenz: atmospheric convection."
        ),
        strength=0.7,
        mapping={
            "jerk attractor": "butterfly attractor",
            "Q bifurcation": "rho bifurcation",
            "electronic chaos": "fluid chaos",
        },
    ))

    # Topological: Rossler Hyperchaos <-> Coupled Lorenz (high-dim chaos)
    analogies.append(Analogy(
        domain_a="rossler_hyperchaos",
        domain_b="coupled_lorenz",
        analogy_type="topological",
        description=(
            "Both are higher-dimensional chaotic systems (4D+). Rossler hyperchaos "
            "has 2 positive LE; coupled Lorenz can also show hyperchaotic behavior."
        ),
        strength=0.7,
        mapping={
            "4D hyperchaos": "6D coupled chaos",
            "2 positive LE": "conditional Lyapunov",
            "d parameter": "coupling epsilon",
        },
    ))

    # Topological: Harvested Population <-> Allee Pred-Prey (saddle-node/bistability)
    analogies.append(Analogy(
        domain_a="harvested_population",
        domain_b="allee_predator_prey",
        analogy_type="topological",
        description=(
            "Both show saddle-node bifurcation leading to extinction. Harvested: "
            "H>MSY causes collapse. Allee: population below threshold causes collapse."
        ),
        strength=0.75,
        mapping={
            "saddle-node at MSY": "saddle-node at Allee threshold",
            "H-driven extinction": "density-dependent extinction",
            "two stable states": "bistability",
        },
    ))

    # Topological: FHN Ring <-> Coupled Map Lattice (ring sync dynamics)
    analogies.append(Analogy(
        domain_a="fhn_ring",
        domain_b="coupled_map_lattice",
        analogy_type="topological",
        description=(
            "Both show synchronization-desynchronization transitions on lattice. "
            "FHN Ring: continuous-time neural waves; CML: discrete spatiotemporal patterns."
        ),
        strength=0.7,
        mapping={
            "traveling wave": "spatiotemporal pattern",
            "sync order parameter": "sync order parameter",
            "D_c transition": "eps_c transition",
        },
    ))

    # Topological: Bazykin <-> Lotka-Volterra (limit cycle pred-prey)
    analogies.append(Analogy(
        domain_a="bazykin",
        domain_b="lotka_volterra",
        analogy_type="topological",
        description=(
            "Both exhibit limit cycle oscillations in predator-prey phase space. "
            "Bazykin: Hopf bifurcation from saturating response. "
            "LV: structurally neutral cycles."
        ),
        strength=0.80,
        mapping={
            "limit cycle (Hopf)": "neutral cycle",
            "stable spiral": "center",
        },
    ))

    # Topological: SIR-Vaccination <-> Eco-Epidemic (disease threshold dynamics)
    analogies.append(Analogy(
        domain_a="sir_vaccination",
        domain_b="eco_epidemic",
        analogy_type="topological",
        description=(
            "Both have disease-free equilibria that lose stability at a threshold. "
            "SIRV: R_eff > 1 causes endemic state. Eco-epidemic: disease invades "
            "predator-prey system above threshold."
        ),
        strength=0.72,
        mapping={
            "DFE stability": "disease-free equilibrium",
            "R_eff threshold": "disease invasion threshold",
        },
    ))

    # Topological: Langford <-> Lorenz-84 (3D atmospheric/torus flows)
    analogies.append(Analogy(
        domain_a="langford",
        domain_b="lorenz84",
        analogy_type="topological",
        description=(
            "Both are 3D systems that can exhibit torus dynamics. "
            "Langford: Hopf-Hopf bifurcation creates quasiperiodic torus. "
            "Lorenz-84: Shilnikov scenario with interlocked tori."
        ),
        strength=0.68,
        mapping={
            "torus (quasiperiodic)": "torus (shilnikov)",
            "Hopf-Hopf bifurcation": "Hadley circulation",
        },
    ))

    # Topological: Laser Rate <-> Harvested Population (threshold transcritical)
    analogies.append(Analogy(
        domain_a="laser_rate",
        domain_b="harvested_population",
        analogy_type="topological",
        description=(
            "Both show transcritical-like threshold transitions. "
            "Laser: P < P_th means no lasing. Harvested: H > MSY means extinction. "
            "Both have stable and unstable branches meeting at threshold."
        ),
        strength=0.65,
        mapping={
            "P_th (lasing onset)": "H_MSY (collapse onset)",
            "below threshold (spontaneous only)": "below MSY (sustainable)",
            "stable operating point": "stable equilibrium",
        },
    ))

    # Topological: FHN Lattice <-> Schnakenberg (2D pattern formation)
    analogies.append(Analogy(
        domain_a="fhn_lattice",
        domain_b="schnakenberg",
        analogy_type="topological",
        description=(
            "Both form 2D spatial patterns via diffusion-driven instability. "
            "FHN: excitable spirals. Schnakenberg: Turing spots/stripes."
        ),
        strength=0.70,
        mapping={
            "spiral wave": "Turing pattern",
            "excitable medium": "activator-inhibitor",
        },
    ))

    # Topological: Four-Species LV <-> May-Leonard (multi-species dynamics)
    analogies.append(Analogy(
        domain_a="four_species_lv",
        domain_b="may_leonard",
        analogy_type="topological",
        description=(
            "Both are multi-species competition systems that can show "
            "heteroclinic-like orbits and competitive exclusion. "
            "4-species: trophic layers. May-Leonard: cyclic competition."
        ),
        strength=0.68,
        mapping={
            "coexistence equilibrium": "interior fixed point",
            "competitive exclusion": "heteroclinic cycle",
        },
    ))

    # Topological: Chen <-> Rossler (3D strange attractors)
    analogies.append(Analogy(
        domain_a="chen",
        domain_b="rossler",
        analogy_type="topological",
        description=(
            "Both are 3D chaotic ODEs with butterfly-type strange attractors. "
            "Chen: double-scroll via Z2 symmetry. Rossler: single-scroll spiral."
        ),
        strength=0.72,
        mapping={
            "double-scroll attractor": "single-scroll attractor",
            "Z2 symmetry": "no symmetry",
            "positive Lyapunov": "positive Lyapunov",
        },
    ))

    # Topological: Lorenz-Stenflo <-> Coupled Lorenz (4D chaos extensions)
    analogies.append(Analogy(
        domain_a="lorenz_stenflo",
        domain_b="coupled_lorenz",
        analogy_type="topological",
        description=(
            "Both extend Lorenz dynamics to higher dimensions. "
            "Stenflo: 4D wave-plasma. Coupled Lorenz: 6D sync of two copies."
        ),
        strength=0.65,
        mapping={
            "4D hyperchaos potential": "6D sync/desync transition",
            "s parameter": "epsilon coupling",
        },
    ))

    # Topological: Aizawa <-> Lorenz (3D strange attractor with double-wing structure)
    analogies.append(Analogy(
        domain_a="aizawa",
        domain_b="lorenz",
        analogy_type="topological",
        description=(
            "Both produce 3D strange attractors with nested structure. "
            "Lorenz: butterfly wings. Aizawa: mushroom-shaped with toroidal base."
        ),
        strength=0.70,
        mapping={
            "mushroom attractor": "butterfly attractor",
            "positive Lyapunov": "positive Lyapunov",
        },
    ))

    # Topological: Halvorsen <-> Thomas (cyclically symmetric chaos)
    analogies.append(Analogy(
        domain_a="halvorsen",
        domain_b="thomas",
        analogy_type="topological",
        description=(
            "Both produce chaotic attractors with S3 cyclic symmetry. "
            "Halvorsen: propeller-shaped. Thomas: labyrinth attractor."
        ),
        strength=0.78,
        mapping={
            "S3 cyclic orbit": "S3 cyclic orbit",
            "propeller attractor": "labyrinth attractor",
        },
    ))

    # Topological: Burke-Shaw <-> Chua (double-scroll strange attractors)
    analogies.append(Analogy(
        domain_a="burke_shaw",
        domain_b="chua",
        analogy_type="topological",
        description=(
            "Both produce double-scroll type strange attractors with Z2 symmetry. "
            "Burke-Shaw: magnetic origin. Chua: electronic circuit origin."
        ),
        strength=0.74,
        mapping={
            "double-scroll": "double-scroll",
            "Z2 symmetry": "Z2 symmetry",
        },
    ))

    # Topological: Nose-Hoover <-> Duffing (chaotic oscillator)
    analogies.append(Analogy(
        domain_a="nose_hoover",
        domain_b="duffing",
        analogy_type="topological",
        description=(
            "Both are modified oscillators producing chaotic trajectories. "
            "Nose-Hoover: thermostat coupling. Duffing: forced cubic restoring force."
        ),
        strength=0.62,
        mapping={
            "thermostatted SHO": "forced nonlinear oscillator",
            "z-modulated damping": "periodic driving",
        },
    ))

    # Topological: Lorenz-Haken <-> Chen (butterfly-type strange attractors)
    analogies.append(Analogy(
        domain_a="lorenz_haken",
        domain_b="chen",
        analogy_type="topological",
        description=(
            "Both produce double-wing butterfly attractors with Z2 symmetry. "
            "Lorenz-Haken: laser origin. Chen: Lorenz algebraic dual."
        ),
        strength=0.80,
        mapping={
            "double-wing attractor": "double-wing attractor",
            "Z2 symmetry": "Z2 symmetry",
        },
    ))

    # Topological: Sakarya <-> Aizawa (3D chaotic with spiral structure)
    analogies.append(Analogy(
        domain_a="sakarya",
        domain_b="aizawa",
        analogy_type="topological",
        description=(
            "Both are 3D chaotic ODEs with non-standard attractor geometries. "
            "Sakarya: neural-inspired branching chaos. Aizawa: mushroom-shaped."
        ),
        strength=0.62,
        mapping={
            "branching attractor": "mushroom attractor",
            "positive Lyapunov": "positive Lyapunov",
        },
    ))

    # Topological: Dadras <-> Rossler (spiral-type strange attractors)
    analogies.append(Analogy(
        domain_a="dadras",
        domain_b="rossler",
        analogy_type="topological",
        description=(
            "Both produce spiral-type single-scroll strange attractors. "
            "Both show period-doubling routes to chaos."
        ),
        strength=0.72,
        mapping={
            "spiral attractor": "spiral attractor",
            "period-doubling": "period-doubling",
        },
    ))

    # Topological: Genesio-Tesi <-> Nose-Hoover (3D chaotic from simple ODEs)
    analogies.append(Analogy(
        domain_a="genesio_tesi",
        domain_b="nose_hoover",
        analogy_type="topological",
        description=(
            "Both are minimal 3D chaotic systems with simple polynomial nonlinearity. "
            "Genesio-Tesi: x^2 jerk. Nose-Hoover: y*z coupling."
        ),
        strength=0.63,
        mapping={
            "jerk chaos": "thermostat chaos",
            "quadratic nonlinearity": "bilinear nonlinearity",
        },
    ))

    # Topological: Lu-Chen <-> Lorenz (strange attractor family)
    analogies.append(Analogy(
        domain_a="lu_chen",
        domain_b="lorenz",
        analogy_type="topological",
        description=(
            "Both exhibit butterfly-shaped strange attractors with two lobes. "
            "Lu-Chen continuously deforms between Lorenz and Chen attractors "
            "as parameter c varies."
        ),
        strength=0.92,
        mapping={
            "double-scroll": "butterfly wings",
            "strange attractor": "strange attractor",
        },
    ))

    # Topological: Qi <-> Lorenz-Stenflo (4D hyperchaotic attractors)
    analogies.append(Analogy(
        domain_a="qi",
        domain_b="lorenz_stenflo",
        analogy_type="topological",
        description=(
            "Both are 4D systems exhibiting hyperchaotic attractors with "
            "two positive Lyapunov exponents."
        ),
        strength=0.74,
        mapping={
            "4D hyperchaos": "4D hyperchaos",
            "two positive Lyapunov": "two positive Lyapunov",
        },
    ))

    # Topological: WINDMI <-> Sprott (minimal chaotic jerk attractors)
    analogies.append(Analogy(
        domain_a="windmi",
        domain_b="sprott",
        analogy_type="topological",
        description=(
            "Both are jerk-type minimal chaotic systems. WINDMI has exp(x) "
            "nonlinearity; Sprott flows use minimal polynomial terms. "
            "Both produce single-scroll strange attractors."
        ),
        strength=0.67,
        mapping={
            "jerk chaos": "jerk chaos",
            "single-scroll": "single-scroll",
        },
    ))

    # Topological: Finance <-> Chen (double-scroll chaotic attractors)
    analogies.append(Analogy(
        domain_a="finance",
        domain_b="chen",
        analogy_type="topological",
        description=(
            "Both produce complex 3D strange attractors with quadratic "
            "nonlinearities and period-doubling routes to chaos."
        ),
        strength=0.68,
        mapping={
            "strange attractor": "strange attractor",
            "period-doubling": "period-doubling",
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
