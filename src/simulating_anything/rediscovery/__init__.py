"""Rediscovery module: data generation + equation discovery for each domain."""

from __future__ import annotations

from simulating_anything.rediscovery.bak_sneppen import run_bak_sneppen_rediscovery
from simulating_anything.rediscovery.boltzmann_gas import run_boltzmann_gas_rediscovery
from simulating_anything.rediscovery.brusselator import run_brusselator_rediscovery
from simulating_anything.rediscovery.brusselator_diffusion import (
    run_brusselator_diffusion_rediscovery,
)
from simulating_anything.rediscovery.cart_pole import run_cart_pole_rediscovery
from simulating_anything.rediscovery.chemostat import run_chemostat_rediscovery
from simulating_anything.rediscovery.chua import run_chua_rediscovery
from simulating_anything.rediscovery.coupled_oscillators import run_coupled_oscillators_rediscovery
from simulating_anything.rediscovery.damped_wave import run_damped_wave_rediscovery
from simulating_anything.rediscovery.diffusive_lv import run_diffusive_lv_rediscovery
from simulating_anything.rediscovery.double_pendulum import run_double_pendulum_rediscovery
from simulating_anything.rediscovery.driven_pendulum import run_driven_pendulum_rediscovery
from simulating_anything.rediscovery.duffing import run_duffing_rediscovery
from simulating_anything.rediscovery.elastic_pendulum import run_elastic_pendulum_rediscovery
from simulating_anything.rediscovery.fhn_spatial import run_fhn_spatial_rediscovery
from simulating_anything.rediscovery.fitzhugh_nagumo import run_fitzhugh_nagumo_rediscovery
from simulating_anything.rediscovery.ginzburg_landau import run_ginzburg_landau_rediscovery
from simulating_anything.rediscovery.gray_scott import run_gray_scott_analysis
from simulating_anything.rediscovery.harmonic_oscillator import run_harmonic_oscillator_rediscovery
from simulating_anything.rediscovery.heat_equation import run_heat_equation_rediscovery
from simulating_anything.rediscovery.henon_map import run_henon_map_rediscovery
from simulating_anything.rediscovery.ising_model import run_ising_model_rediscovery
from simulating_anything.rediscovery.kepler import run_kepler_rediscovery
from simulating_anything.rediscovery.kuramoto import run_kuramoto_rediscovery
from simulating_anything.rediscovery.kuramoto_sivashinsky import (
    run_kuramoto_sivashinsky_rediscovery,
)
from simulating_anything.rediscovery.logistic_map import run_logistic_map_rediscovery
from simulating_anything.rediscovery.lorenz import run_lorenz_rediscovery
from simulating_anything.rediscovery.lorenz96 import run_lorenz96_rediscovery
from simulating_anything.rediscovery.lotka_volterra import run_lotka_volterra_rediscovery
from simulating_anything.rediscovery.navier_stokes import run_navier_stokes_rediscovery
from simulating_anything.rediscovery.oregonator import run_oregonator_rediscovery
from simulating_anything.rediscovery.projectile import run_projectile_rediscovery
from simulating_anything.rediscovery.quantum_oscillator import run_quantum_oscillator_rediscovery
from simulating_anything.rediscovery.rosenzweig_macarthur import (
    run_rosenzweig_macarthur_rediscovery,
)
from simulating_anything.rediscovery.rossler import run_rossler_rediscovery
from simulating_anything.rediscovery.schwarzschild import run_schwarzschild_rediscovery
from simulating_anything.rediscovery.sir_epidemic import run_sir_rediscovery
from simulating_anything.rediscovery.spring_mass_chain import run_spring_mass_chain_rediscovery
from simulating_anything.rediscovery.three_species import run_three_species_rediscovery
from simulating_anything.rediscovery.shallow_water import run_shallow_water_rediscovery
from simulating_anything.rediscovery.toda_lattice import run_toda_lattice_rediscovery
from simulating_anything.rediscovery.van_der_pol import run_van_der_pol_rediscovery
from simulating_anything.rediscovery.wilberforce import run_wilberforce_rediscovery

__all__ = [
    "run_boltzmann_gas_rediscovery",
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
    "run_kepler_rediscovery",
    "run_duffing_rediscovery",
    "run_schwarzschild_rediscovery",
    "run_quantum_oscillator_rediscovery",
    "run_spring_mass_chain_rediscovery",
    "run_driven_pendulum_rediscovery",
    "run_coupled_oscillators_rediscovery",
    "run_damped_wave_rediscovery",
    "run_diffusive_lv_rediscovery",
    "run_ising_model_rediscovery",
    "run_three_species_rediscovery",
    "run_cart_pole_rediscovery",
    "run_elastic_pendulum_rediscovery",
    "run_rossler_rediscovery",
    "run_henon_map_rediscovery",
    "run_brusselator_diffusion_rediscovery",
    "run_rosenzweig_macarthur_rediscovery",
    "run_shallow_water_rediscovery",
    "run_chua_rediscovery",
    "run_toda_lattice_rediscovery",
    "run_bak_sneppen_rediscovery",
    "run_kuramoto_sivashinsky_rediscovery",
    "run_oregonator_rediscovery",
    "run_ginzburg_landau_rediscovery",
    "run_lorenz96_rediscovery",
    "run_chemostat_rediscovery",
    "run_fhn_spatial_rediscovery",
    "run_wilberforce_rediscovery",
]
