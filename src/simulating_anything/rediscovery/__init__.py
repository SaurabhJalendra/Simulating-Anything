"""Rediscovery module: data generation + equation discovery for each domain."""

from __future__ import annotations

from simulating_anything.rediscovery.projectile import run_projectile_rediscovery
from simulating_anything.rediscovery.lotka_volterra import run_lotka_volterra_rediscovery
from simulating_anything.rediscovery.gray_scott import run_gray_scott_analysis
from simulating_anything.rediscovery.sir_epidemic import run_sir_rediscovery
from simulating_anything.rediscovery.double_pendulum import run_double_pendulum_rediscovery
from simulating_anything.rediscovery.harmonic_oscillator import run_harmonic_oscillator_rediscovery
from simulating_anything.rediscovery.lorenz import run_lorenz_rediscovery
from simulating_anything.rediscovery.navier_stokes import run_navier_stokes_rediscovery

__all__ = [
    "run_projectile_rediscovery",
    "run_lotka_volterra_rediscovery",
    "run_gray_scott_analysis",
    "run_sir_rediscovery",
    "run_double_pendulum_rediscovery",
    "run_harmonic_oscillator_rediscovery",
    "run_lorenz_rediscovery",
    "run_navier_stokes_rediscovery",
]
