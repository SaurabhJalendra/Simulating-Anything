"""Domain statistics: quick summary metrics for all simulation domains.

Computes runtime performance, trajectory statistics, and domain properties
for all 14 domains. Useful for benchmarking and paper reporting.
"""
from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Domain registry: module, class, domain enum, default params
DOMAIN_REGISTRY: dict[str, dict[str, Any]] = {
    "projectile": {
        "module": "simulating_anything.simulation.rigid_body",
        "cls": "ProjectileSimulation",
        "domain": Domain.RIGID_BODY,
        "params": {
            "initial_speed": 30.0, "launch_angle": 45.0,
            "gravity": 9.81, "drag_coefficient": 0.1, "mass": 1.0,
        },
        "dt": 0.01, "n_steps": 500, "math_class": "Algebraic",
    },
    "lotka_volterra": {
        "module": "simulating_anything.simulation.agent_based",
        "cls": "LotkaVolterraSimulation",
        "domain": Domain.AGENT_BASED,
        "params": {
            "alpha": 1.1, "beta": 0.4, "gamma": 0.4, "delta": 0.1,
            "prey_0": 40.0, "predator_0": 9.0,
        },
        "dt": 0.01, "n_steps": 500, "math_class": "Nonlinear ODE",
    },
    "sir_epidemic": {
        "module": "simulating_anything.simulation.epidemiological",
        "cls": "SIRSimulation",
        "domain": Domain.EPIDEMIOLOGICAL,
        "params": {"beta": 0.3, "gamma": 0.1, "S_0": 0.99, "I_0": 0.01},
        "dt": 0.1, "n_steps": 500, "math_class": "Nonlinear ODE",
    },
    "double_pendulum": {
        "module": "simulating_anything.simulation.chaotic_ode",
        "cls": "DoublePendulumSimulation",
        "domain": Domain.CHAOTIC_ODE,
        "params": {
            "m1": 1.0, "m2": 1.0, "L1": 1.0, "L2": 1.0, "g": 9.81,
            "theta1_0": 1.0, "theta2_0": 1.5,
            "omega1_0": 0.0, "omega2_0": 0.0,
        },
        "dt": 0.001, "n_steps": 500, "math_class": "Chaotic ODE",
    },
    "harmonic_oscillator": {
        "module": "simulating_anything.simulation.harmonic_oscillator",
        "cls": "DampedHarmonicOscillator",
        "domain": Domain.HARMONIC_OSCILLATOR,
        "params": {"k": 4.0, "m": 1.0, "c": 0.4, "x_0": 2.0, "v_0": 0.0},
        "dt": 0.01, "n_steps": 500, "math_class": "Linear ODE",
    },
    "lorenz": {
        "module": "simulating_anything.simulation.lorenz",
        "cls": "LorenzSimulation",
        "domain": Domain.LORENZ_ATTRACTOR,
        "params": {"sigma": 10.0, "rho": 28.0, "beta": 2.667},
        "dt": 0.01, "n_steps": 500, "math_class": "Chaotic ODE",
    },
    "navier_stokes": {
        "module": "simulating_anything.simulation.navier_stokes",
        "cls": "NavierStokes2DSimulation",
        "domain": Domain.NAVIER_STOKES_2D,
        "params": {"nu": 0.01, "N": 32},
        "dt": 0.01, "n_steps": 50, "math_class": "PDE",
    },
    "van_der_pol": {
        "module": "simulating_anything.simulation.van_der_pol",
        "cls": "VanDerPolSimulation",
        "domain": Domain.VAN_DER_POL,
        "params": {"mu": 1.0, "x_0": 2.0, "v_0": 0.0},
        "dt": 0.01, "n_steps": 500, "math_class": "Nonlinear ODE",
    },
    "kuramoto": {
        "module": "simulating_anything.simulation.kuramoto",
        "cls": "KuramotoSimulation",
        "domain": Domain.KURAMOTO,
        "params": {"N": 20, "K": 2.0, "omega_std": 1.0},
        "dt": 0.01, "n_steps": 500, "math_class": "Collective ODE",
    },
    "brusselator": {
        "module": "simulating_anything.simulation.brusselator",
        "cls": "BrusselatorSimulation",
        "domain": Domain.BRUSSELATOR,
        "params": {"a": 1.0, "b": 3.0, "u_0": 1.0, "v_0": 1.0},
        "dt": 0.01, "n_steps": 500, "math_class": "Nonlinear ODE",
    },
    "fitzhugh_nagumo": {
        "module": "simulating_anything.simulation.fitzhugh_nagumo",
        "cls": "FitzHughNagumoSimulation",
        "domain": Domain.FITZHUGH_NAGUMO,
        "params": {
            "a": 0.7, "b": 0.8, "eps": 0.08,
            "I_ext": 0.5, "v_0": -1.0, "w_0": -0.5,
        },
        "dt": 0.1, "n_steps": 500, "math_class": "Nonlinear ODE",
    },
    "heat_equation": {
        "module": "simulating_anything.simulation.heat_equation",
        "cls": "HeatEquation1DSimulation",
        "domain": Domain.HEAT_EQUATION_1D,
        "params": {"D": 0.1, "N": 64},
        "dt": 0.01, "n_steps": 200, "math_class": "Linear PDE",
    },
    "logistic_map": {
        "module": "simulating_anything.simulation.logistic_map",
        "cls": "LogisticMapSimulation",
        "domain": Domain.LOGISTIC_MAP,
        "params": {"r": 3.9, "x_0": 0.5},
        "dt": 1.0, "n_steps": 500, "math_class": "Discrete Chaos",
    },
    "boltzmann_gas": {
        "module": "simulating_anything.simulation.boltzmann_gas",
        "cls": "BoltzmannGas2D",
        "domain": Domain.BOLTZMANN_GAS,
        "params": {
            "N": 50.0, "L": 10.0, "T": 1.0,
            "particle_radius": 0.05, "m": 1.0,
        },
        "dt": 0.005, "n_steps": 500, "math_class": "Statistical Mechanics",
    },
    "duffing": {
        "module": "simulating_anything.simulation.duffing",
        "cls": "DuffingOscillator",
        "domain": Domain.DUFFING,
        "params": {
            "alpha": 1.0, "beta": 1.0, "delta": 0.2,
            "gamma_f": 0.3, "omega": 1.0, "x_0": 0.5, "v_0": 0.0,
        },
        "dt": 0.005, "n_steps": 500, "math_class": "Nonlinear ODE",
    },
    "quantum_oscillator": {
        "module": "simulating_anything.simulation.quantum_oscillator",
        "cls": "QuantumHarmonicOscillator",
        "domain": Domain.QUANTUM_OSCILLATOR,
        "params": {
            "m": 1.0, "omega": 1.0, "hbar": 1.0,
            "N": 128.0, "x_max": 10.0, "x_0": 2.0, "p_0": 0.0,
        },
        "dt": 0.01, "n_steps": 500, "math_class": "Quantum PDE",
    },
    "schwarzschild": {
        "module": "simulating_anything.simulation.schwarzschild",
        "cls": "SchwarzschildGeodesic",
        "domain": Domain.SCHWARZSCHILD,
        "params": {"M": 1.0, "L": 4.0, "r_0": 10.0, "pr_0": 0.0},
        "dt": 0.01, "n_steps": 500, "math_class": "GR Geodesic",
    },
    "spring_mass_chain": {
        "module": "simulating_anything.simulation.spring_mass_chain",
        "cls": "SpringMassChain",
        "domain": Domain.SPRING_MASS_CHAIN,
        "params": {
            "N": 20.0, "K": 4.0, "m": 1.0, "a": 1.0,
            "mode": 1.0, "amplitude": 0.1,
        },
        "dt": 0.001, "n_steps": 500, "math_class": "Coupled ODE",
    },
    "kepler": {
        "module": "simulating_anything.simulation.kepler",
        "cls": "KeplerOrbit",
        "domain": Domain.KEPLER,
        "params": {
            "GM": 1.0, "initial_r": 1.0, "eccentricity": 0.5,
        },
        "dt": 0.001, "n_steps": 500, "math_class": "Celestial Mechanics",
    },
    "driven_pendulum": {
        "module": "simulating_anything.simulation.driven_pendulum",
        "cls": "DrivenPendulum",
        "domain": Domain.DRIVEN_PENDULUM,
        "params": {
            "gamma": 0.5, "omega0": 1.5, "A_drive": 1.2,
            "omega_d": 0.6667, "theta_0": 0.1, "omega_init": 0.0,
        },
        "dt": 0.005, "n_steps": 500, "math_class": "Chaotic ODE",
    },
    "coupled_oscillators": {
        "module": "simulating_anything.simulation.coupled_oscillators",
        "cls": "CoupledOscillators",
        "domain": Domain.COUPLED_OSCILLATORS,
        "params": {
            "k": 4.0, "m": 1.0, "kc": 0.5,
            "x1_0": 1.0, "v1_0": 0.0, "x2_0": 0.0, "v2_0": 0.0,
        },
        "dt": 0.001, "n_steps": 500, "math_class": "Coupled ODE",
    },
    "diffusive_lv": {
        "module": "simulating_anything.simulation.diffusive_lv",
        "cls": "DiffusiveLotkaVolterra",
        "domain": Domain.DIFFUSIVE_LV,
        "params": {
            "alpha": 1.0, "beta": 0.5, "gamma": 0.5, "delta": 0.2,
            "D_u": 0.1, "D_v": 0.05, "N_grid": 64.0, "L_domain": 20.0,
        },
        "dt": 0.005, "n_steps": 500, "math_class": "Reaction-Diffusion PDE",
    },
    "damped_wave": {
        "module": "simulating_anything.simulation.damped_wave",
        "cls": "DampedWave1D",
        "domain": Domain.DAMPED_WAVE,
        "params": {
            "c": 1.0, "gamma": 0.1, "N": 64.0, "L": 6.283185307179586,
        },
        "dt": 0.001, "n_steps": 500, "math_class": "Wave PDE",
    },
    "ising_model": {
        "module": "simulating_anything.simulation.ising_model",
        "cls": "IsingModel2D",
        "domain": Domain.ISING_MODEL,
        "params": {
            "N": 16.0, "J": 1.0, "h": 0.0, "T": 2.0,
        },
        "dt": 1.0, "n_steps": 100, "math_class": "Statistical Mechanics",
    },
    "three_species": {
        "module": "simulating_anything.simulation.three_species",
        "cls": "ThreeSpecies",
        "domain": Domain.THREE_SPECIES,
        "params": {
            "a1": 1.0, "b1": 0.5, "a2": 0.5, "b2": 0.2, "a3": 0.3,
            "x0": 1.0, "y0": 0.5, "z0": 0.5,
        },
        "dt": 0.01, "n_steps": 500, "math_class": "Nonlinear ODE",
    },
    "cart_pole": {
        "module": "simulating_anything.simulation.cart_pole",
        "cls": "CartPole",
        "domain": Domain.CART_POLE,
        "params": {
            "M": 1.0, "m": 0.1, "L": 0.5, "g": 9.81,
            "mu_c": 0.0, "mu_p": 0.0, "F": 0.0,
            "x_0": 0.0, "x_dot_0": 0.0,
            "theta_0": 0.1, "theta_dot_0": 0.0,
        },
        "dt": 0.001, "n_steps": 500, "math_class": "Nonlinear ODE",
    },
    "elastic_pendulum": {
        "module": "simulating_anything.simulation.elastic_pendulum",
        "cls": "ElasticPendulum",
        "domain": Domain.ELASTIC_PENDULUM,
        "params": {
            "k": 10.0, "m": 1.0, "L0": 1.0, "g": 9.81,
            "r_0": 1.981, "r_dot_0": 0.0,
            "theta_0": 0.1, "theta_dot_0": 0.0,
        },
        "dt": 0.001, "n_steps": 500, "math_class": "Nonlinear ODE",
    },
    "rossler": {
        "module": "simulating_anything.simulation.rossler",
        "cls": "RosslerSimulation",
        "domain": Domain.ROSSLER,
        "params": {
            "a": 0.2, "b": 0.2, "c": 5.7,
            "x_0": 1.0, "y_0": 1.0, "z_0": 0.0,
        },
        "dt": 0.005, "n_steps": 500, "math_class": "Chaotic ODE",
    },
    "henon_map": {
        "module": "simulating_anything.simulation.henon_map",
        "cls": "HenonMapSimulation",
        "domain": Domain.HENON_MAP,
        "params": {"a": 1.4, "b": 0.3, "x_0": 0.0, "y_0": 0.0},
        "dt": 1.0, "n_steps": 500, "math_class": "Discrete Chaos",
    },
    "brusselator_diffusion": {
        "module": "simulating_anything.simulation.brusselator_diffusion",
        "cls": "BrusselatorDiffusion",
        "domain": Domain.BRUSSELATOR_DIFFUSION,
        "params": {
            "a": 1.0, "b": 3.0, "D_u": 0.01, "D_v": 0.1,
            "N_grid": 64.0, "L_domain": 20.0,
        },
        "dt": 0.01, "n_steps": 500, "math_class": "Reaction-Diffusion PDE",
    },
    "rosenzweig_macarthur": {
        "module": "simulating_anything.simulation.rosenzweig_macarthur",
        "cls": "RosenzweigMacArthur",
        "domain": Domain.ROSENZWEIG_MACARTHUR,
        "params": {
            "r": 1.0, "K": 10.0, "a": 0.5, "h": 0.5,
            "e": 0.5, "d": 0.1, "x_0": 1.0, "y_0": 1.0,
        },
        "dt": 0.01, "n_steps": 500, "math_class": "Nonlinear ODE",
    },
    "toda_lattice": {
        "module": "simulating_anything.simulation.toda_lattice",
        "cls": "TodaLattice",
        "domain": Domain.TODA_LATTICE,
        "params": {
            "N": 8.0, "a": 1.0, "mode": 1.0, "amplitude": 0.1,
        },
        "dt": 0.001, "n_steps": 500, "math_class": "Integrable Chain",
    },
    "chua": {
        "module": "simulating_anything.simulation.chua",
        "cls": "ChuaCircuit",
        "domain": Domain.CHUA,
        "params": {
            "alpha": 15.6, "beta": 28.0,
            "m0": -1.143, "m1": -0.714,
            "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
        },
        "dt": 0.001, "n_steps": 500, "math_class": "Chaotic ODE",
    },
    "shallow_water": {
        "module": "simulating_anything.simulation.shallow_water",
        "cls": "ShallowWater",
        "domain": Domain.SHALLOW_WATER,
        "params": {
            "g": 9.81, "h0": 1.0, "N": 128.0, "L": 10.0,
            "perturbation_amplitude": 0.1,
        },
        "dt": 0.001, "n_steps": 500, "math_class": "Hyperbolic PDE",
    },
    "ginzburg_landau": {
        "module": "simulating_anything.simulation.ginzburg_landau",
        "cls": "GinzburgLandau",
        "domain": Domain.GINZBURG_LANDAU,
        "params": {
            "c1": 1.0, "c2": -1.2, "L": 50.0, "N": 128.0,
        },
        "dt": 0.05, "n_steps": 500, "math_class": "Complex PDE",
    },
    "bak_sneppen": {
        "module": "simulating_anything.simulation.bak_sneppen",
        "cls": "BakSneppen",
        "domain": Domain.BAK_SNEPPEN,
        "params": {"N": 50.0},
        "dt": 1.0, "n_steps": 500, "math_class": "SOC / Extremal",
    },
    "kuramoto_sivashinsky": {
        "module": "simulating_anything.simulation.kuramoto_sivashinsky",
        "cls": "KuramotoSivashinsky",
        "domain": Domain.KURAMOTO_SIVASHINSKY,
        "params": {
            "L": 100.53096491487338, "N": 128.0, "viscosity": 1.0,
        },
        "dt": 0.05, "n_steps": 500, "math_class": "Chaotic PDE",
    },
    "oregonator": {
        "module": "simulating_anything.simulation.oregonator",
        "cls": "Oregonator",
        "domain": Domain.OREGONATOR,
        "params": {
            "eps": 0.04, "f": 1.0, "q": 0.002, "kw": 0.5,
            "u_0": 0.5, "v_0": 0.5, "w_0": 0.5,
        },
        "dt": 0.001, "n_steps": 500, "math_class": "Nonlinear ODE",
    },
    "lorenz96": {
        "module": "simulating_anything.simulation.lorenz96",
        "cls": "Lorenz96",
        "domain": Domain.LORENZ96,
        "params": {"N": 36.0, "F": 8.0},
        "dt": 0.01, "n_steps": 500, "math_class": "Chaotic ODE",
    },
    "chemostat": {
        "module": "simulating_anything.simulation.chemostat",
        "cls": "Chemostat",
        "domain": Domain.CHEMOSTAT,
        "params": {
            "D": 0.1, "S_in": 10.0, "mu_max": 0.5,
            "K_s": 2.0, "Y_xs": 0.5, "S_0": 5.0, "X_0": 1.0,
        },
        "dt": 0.01, "n_steps": 500, "math_class": "Nonlinear ODE",
    },
    "wilberforce": {
        "module": "simulating_anything.simulation.wilberforce",
        "cls": "Wilberforce",
        "domain": Domain.WILBERFORCE,
        "params": {
            "m": 0.5, "k": 5.0, "I": 1e-4, "kappa": 1e-3, "eps": 1e-3,
            "z_0": 0.1, "z_dot_0": 0.0, "theta_0": 0.0, "theta_dot_0": 0.0,
        },
        "dt": 0.001, "n_steps": 500, "math_class": "Coupled ODE",
    },
    "standard_map": {
        "module": "simulating_anything.simulation.standard_map",
        "cls": "StandardMapSimulation",
        "domain": Domain.STANDARD_MAP,
        "params": {"K": 0.9716, "n_particles": 100.0},
        "dt": 1.0, "n_steps": 500, "math_class": "Hamiltonian",
    },
    "hodgkin_huxley": {
        "module": "simulating_anything.simulation.hodgkin_huxley",
        "cls": "HodgkinHuxleySimulation",
        "domain": Domain.HODGKIN_HUXLEY,
        "params": {
            "g_Na": 120.0, "g_K": 36.0, "g_L": 0.3,
            "E_Na": 50.0, "E_K": -77.0, "E_L": -54.387,
            "C_m": 1.0, "I_ext": 10.0,
        },
        "dt": 0.01, "n_steps": 500, "math_class": "Biophysical ODE",
    },
    "eco_epidemic": {
        "module": "simulating_anything.simulation.eco_epidemic",
        "cls": "EcoEpidemicSimulation",
        "domain": Domain.ECO_EPIDEMIC,
        "params": {
            "r": 1.0, "K": 100.0, "beta": 0.01,
            "a1": 0.1, "a2": 0.3, "h1": 0.1, "h2": 0.1,
            "e1": 0.5, "e2": 0.3, "d_disease": 0.2, "m": 0.3,
            "S_0": 50.0, "I_0": 10.0, "P_0": 5.0,
        },
        "dt": 0.01, "n_steps": 500, "math_class": "Eco-Epidemiological",
    },
    "rayleigh_benard": {
        "module": "simulating_anything.simulation.rayleigh_benard",
        "cls": "RayleighBenardSimulation",
        "domain": Domain.RAYLEIGH_BENARD,
        "params": {
            "Ra": 1000.0, "Pr": 1.0,
            "Lx": 2.0, "H": 1.0,
            "Nx": 32.0, "Nz": 16.0,
            "perturbation_amp": 0.01,
        },
        "dt": 5e-5, "n_steps": 200, "math_class": "Convection PDE",
    },
    "hindmarsh_rose": {
        "module": "simulating_anything.simulation.hindmarsh_rose",
        "cls": "HindmarshRoseSimulation",
        "domain": Domain.HINDMARSH_ROSE,
        "params": {
            "a": 1.0, "b": 3.0, "c": 1.0, "d": 5.0,
            "r": 0.001, "s": 4.0, "x_rest": -1.6, "I_ext": 3.25,
            "x_0": -1.5, "y_0": -10.0, "z_0": 2.0,
        },
        "dt": 0.05, "n_steps": 500, "math_class": "Bursting ODE",
    },
    "competitive_lv": {
        "module": "simulating_anything.simulation.competitive_lv",
        "cls": "CompetitiveLVSimulation",
        "domain": Domain.COMPETITIVE_LV,
        "params": {
            "n_species": 4.0,
            "r_0": 1.0, "r_1": 0.72, "r_2": 1.53, "r_3": 1.27,
            "K_0": 100.0, "K_1": 100.0, "K_2": 100.0, "K_3": 100.0,
            "alpha_0_0": 1.0, "alpha_0_1": 0.5, "alpha_0_2": 0.4, "alpha_0_3": 0.3,
            "alpha_1_0": 0.4, "alpha_1_1": 1.0, "alpha_1_2": 0.6, "alpha_1_3": 0.3,
            "alpha_2_0": 0.3, "alpha_2_1": 0.4, "alpha_2_2": 1.0, "alpha_2_3": 0.5,
            "alpha_3_0": 0.5, "alpha_3_1": 0.3, "alpha_3_2": 0.4, "alpha_3_3": 1.0,
        },
        "dt": 0.01, "n_steps": 500, "math_class": "Competitive ODE",
    },
    "vicsek": {
        "module": "simulating_anything.simulation.vicsek",
        "cls": "VicsekSimulation",
        "domain": Domain.VICSEK,
        "params": {
            "N": 100.0, "L": 10.0, "v0": 0.5, "R": 1.0, "eta": 0.3,
        },
        "dt": 1.0, "n_steps": 500, "math_class": "Active Matter",
    },
    "magnetic_pendulum": {
        "module": "simulating_anything.simulation.magnetic_pendulum",
        "cls": "MagneticPendulumSimulation",
        "domain": Domain.MAGNETIC_PENDULUM,
        "params": {
            "gamma": 0.1, "omega0_sq": 0.5, "alpha": 1.0,
            "R": 1.0, "d": 0.3, "n_magnets": 3.0,
            "x_0": 0.5, "y_0": 0.5, "vx_0": 0.0, "vy_0": 0.0,
        },
        "dt": 0.01, "n_steps": 500, "math_class": "Fractal Basin",
    },
    "bz_spiral": {
        "module": "simulating_anything.simulation.bz_spiral",
        "cls": "BZSpiralSimulation",
        "domain": Domain.BZ_SPIRAL,
        "params": {
            "eps": 0.01, "f": 1.0, "q": 0.002,
            "D_u": 1.0, "D_v": 0.0,
            "Nx": 64.0, "Ny": 64.0, "dx": 0.5,
        },
        "dt": 0.01, "n_steps": 200, "math_class": "Excitable PDE",
    },
    "coupled_lorenz": {
        "module": "simulating_anything.simulation.coupled_lorenz",
        "cls": "CoupledLorenzSimulation",
        "domain": Domain.COUPLED_LORENZ,
        "params": {
            "sigma": 10.0, "rho": 28.0, "beta": 2.667,
            "eps": 5.0,
            "x1_0": 1.0, "y1_0": 1.0, "z1_0": 1.0,
            "x2_0": -5.0, "y2_0": 5.0, "z2_0": 25.0,
        },
        "dt": 0.005, "n_steps": 500, "math_class": "Chaos Sync",
    },
    "swinging_atwood": {
        "module": "simulating_anything.simulation.swinging_atwood",
        "cls": "SwingingAtwoodSimulation",
        "domain": Domain.SWINGING_ATWOOD,
        "params": {
            "M": 3.0, "m": 1.0, "g": 9.81,
            "r_min": 0.1, "r_0": 1.0, "theta_0": 0.5,
            "r_dot_0": 0.0, "theta_dot_0": 0.0,
        },
        "dt": 0.001, "n_steps": 500, "math_class": "Lagrangian ODE",
    },
    "allee_predator_prey": {
        "module": "simulating_anything.simulation.allee_predator_prey",
        "cls": "AlleePredatorPreySimulation",
        "domain": Domain.ALLEE_PREDATOR_PREY,
        "params": {
            "r": 1.0, "A": 10.0, "K": 100.0,
            "a": 0.01, "h": 0.1, "e": 0.5, "m": 0.3,
            "N_0": 50.0, "P_0": 5.0,
        },
        "dt": 0.01, "n_steps": 500, "math_class": "Bistable ODE",
    },
}


@dataclass
class DomainStats:
    """Statistics for a single domain."""

    name: str
    math_class: str
    obs_dim: int
    n_steps: int
    run_time_ms: float
    state_mean: float
    state_std: float
    state_min: float
    state_max: float
    is_finite: bool
    is_deterministic: bool


def compute_domain_stats(domain_name: str) -> DomainStats:
    """Compute statistics for a single domain."""
    spec = DOMAIN_REGISTRY[domain_name]
    mod = importlib.import_module(spec["module"])
    cls = getattr(mod, spec["cls"])

    config = SimulationConfig(
        domain=spec["domain"], dt=spec["dt"], n_steps=spec["n_steps"],
        parameters=spec["params"],
    )
    sim = cls(config)

    # Benchmark
    t0 = time.perf_counter()
    traj = sim.run(n_steps=spec["n_steps"])
    run_time = (time.perf_counter() - t0) * 1000  # ms

    states = traj.states
    obs_dim = states.shape[1] if states.ndim > 1 else 1

    # Determinism check
    sim2 = cls(config)
    traj2 = sim2.run(n_steps=spec["n_steps"])
    is_det = np.allclose(states, traj2.states, atol=1e-10)

    return DomainStats(
        name=domain_name,
        math_class=spec["math_class"],
        obs_dim=obs_dim,
        n_steps=spec["n_steps"],
        run_time_ms=run_time,
        state_mean=float(np.mean(states)),
        state_std=float(np.std(states)),
        state_min=float(np.min(states)),
        state_max=float(np.max(states)),
        is_finite=bool(np.all(np.isfinite(states))),
        is_deterministic=is_det,
    )


def compute_all_stats(
    skip_kuramoto: bool = True,
) -> list[DomainStats]:
    """Compute statistics for all domains."""
    results = []
    for name in DOMAIN_REGISTRY:
        if skip_kuramoto and name == "kuramoto":
            # Kuramoto uses random frequencies, not deterministic
            continue
        try:
            stats = compute_domain_stats(name)
            results.append(stats)
            logger.info(
                f"  {name:25s}: dim={stats.obs_dim:4d}, "
                f"time={stats.run_time_ms:8.1f}ms, "
                f"range=[{stats.state_min:.3f}, {stats.state_max:.3f}]"
            )
        except Exception as e:
            logger.warning(f"  {name}: FAILED - {e}")
    return results


def print_stats_table(stats: list[DomainStats]) -> str:
    """Format stats as a text table."""
    lines = []
    lines.append(f"{'Domain':25s} {'Class':15s} {'Dim':>4s} {'Steps':>5s} "
                 f"{'Time(ms)':>8s} {'Range':>20s} {'Det':>4s}")
    lines.append("-" * 90)
    total_time = 0.0
    for s in stats:
        total_time += s.run_time_ms
        det = "Y" if s.is_deterministic else "N"
        lines.append(
            f"{s.name:25s} {s.math_class:15s} {s.obs_dim:4d} {s.n_steps:5d} "
            f"{s.run_time_ms:8.1f} [{s.state_min:8.3f}, {s.state_max:8.3f}] "
            f"{det:>4s}"
        )
    lines.append("-" * 90)
    lines.append(f"{'Total':25s} {'':15s} {'':4s} {'':5s} {total_time:8.1f}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Computing domain statistics...")
    stats = compute_all_stats(skip_kuramoto=False)
    print()
    print(print_stats_table(stats))
