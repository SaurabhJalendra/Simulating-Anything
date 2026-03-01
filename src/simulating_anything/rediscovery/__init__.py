"""Rediscovery module: data generation + equation discovery for each domain."""

from __future__ import annotations

from simulating_anything.rediscovery.brusselator import run_brusselator_rediscovery
from simulating_anything.rediscovery.double_pendulum import run_double_pendulum_rediscovery
from simulating_anything.rediscovery.duffing import run_duffing_rediscovery
from simulating_anything.rediscovery.fitzhugh_nagumo import run_fitzhugh_nagumo_rediscovery
from simulating_anything.rediscovery.quantum_oscillator import run_quantum_oscillator_rediscovery
from simulating_anything.rediscovery.schwarzschild import run_schwarzschild_rediscovery
from simulating_anything.rediscovery.gray_scott import run_gray_scott_analysis
from simulating_anything.rediscovery.harmonic_oscillator import run_harmonic_oscillator_rediscovery
from simulating_anything.rediscovery.heat_equation import run_heat_equation_rediscovery
from simulating_anything.rediscovery.kuramoto import run_kuramoto_rediscovery
from simulating_anything.rediscovery.logistic_map import run_logistic_map_rediscovery
from simulating_anything.rediscovery.lorenz import run_lorenz_rediscovery
from simulating_anything.rediscovery.lotka_volterra import run_lotka_volterra_rediscovery
from simulating_anything.rediscovery.navier_stokes import run_navier_stokes_rediscovery
from simulating_anything.rediscovery.projectile import run_projectile_rediscovery
from simulating_anything.rediscovery.sir_epidemic import run_sir_rediscovery
from simulating_anything.rediscovery.van_der_pol import run_van_der_pol_rediscovery

__all__ = [
    "run_projectile_rediscovery",
    "run_lotka_volterra_rediscovery",
    "run_gray_scott_analysis",
    "run_sir_rediscovery",
    "run_double_pendulum_rediscovery",
    "run_harmonic_oscillator_rediscovery",
    "run_lorenz_rediscovery",
    "run_navier_stokes_rediscovery",
    "run_van_der_pol_rediscovery",
    "run_kuramoto_rediscovery",
    "run_brusselator_rediscovery",
    "run_fitzhugh_nagumo_rediscovery",
    "run_heat_equation_rediscovery",
    "run_logistic_map_rediscovery",
    "run_duffing_rediscovery",
    "run_schwarzschild_rediscovery",
    "run_quantum_oscillator_rediscovery",
]
