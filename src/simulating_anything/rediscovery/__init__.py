"""Rediscovery module: data generation + equation discovery for each domain."""

from __future__ import annotations

from simulating_anything.rediscovery.aizawa import run_aizawa_rediscovery
from simulating_anything.rediscovery.allee_predator_prey import (
    run_allee_predator_prey_rediscovery,
)
from simulating_anything.rediscovery.bak_sneppen import run_bak_sneppen_rediscovery
from simulating_anything.rediscovery.bazykin import run_bazykin_rediscovery
from simulating_anything.rediscovery.boltzmann_gas import run_boltzmann_gas_rediscovery
from simulating_anything.rediscovery.burke_shaw import run_burke_shaw_rediscovery
from simulating_anything.rediscovery.bouncing_ball import run_bouncing_ball_rediscovery
from simulating_anything.rediscovery.brusselator import run_brusselator_rediscovery
from simulating_anything.rediscovery.chen import run_chen_rediscovery
from simulating_anything.rediscovery.dadras import run_dadras_rediscovery
from simulating_anything.rediscovery.brusselator_2d import run_brusselator_2d_rediscovery
from simulating_anything.rediscovery.brusselator_diffusion import (
    run_brusselator_diffusion_rediscovery,
)
from simulating_anything.rediscovery.bz_spiral import run_bz_spiral_rediscovery
from simulating_anything.rediscovery.cable_equation import run_cable_equation_rediscovery
from simulating_anything.rediscovery.cahn_hilliard import run_cahn_hilliard_rediscovery
from simulating_anything.rediscovery.cart_pole import run_cart_pole_rediscovery
from simulating_anything.rediscovery.chemostat import run_chemostat_rediscovery
from simulating_anything.rediscovery.chua import run_chua_rediscovery
from simulating_anything.rediscovery.colpitts import run_colpitts_rediscovery
from simulating_anything.rediscovery.competitive_lv import run_competitive_lv_rediscovery
from simulating_anything.rediscovery.coupled_lorenz import run_coupled_lorenz_rediscovery
from simulating_anything.rediscovery.coupled_map_lattice import (
    run_coupled_map_lattice_rediscovery,
)
from simulating_anything.rediscovery.coupled_oscillators import run_coupled_oscillators_rediscovery
from simulating_anything.rediscovery.cubic_map import run_cubic_map_rediscovery
from simulating_anything.rediscovery.damped_wave import run_damped_wave_rediscovery
from simulating_anything.rediscovery.delayed_predator_prey import (
    run_delayed_predator_prey_rediscovery,
)
from simulating_anything.rediscovery.diffusive_lv import run_diffusive_lv_rediscovery
from simulating_anything.rediscovery.double_pendulum import run_double_pendulum_rediscovery
from simulating_anything.rediscovery.driven_pendulum import run_driven_pendulum_rediscovery
from simulating_anything.rediscovery.duffing import run_duffing_rediscovery
from simulating_anything.rediscovery.duffing_van_der_pol import (
    run_duffing_van_der_pol_rediscovery,
)
from simulating_anything.rediscovery.eco_epidemic import run_eco_epidemic_rediscovery
from simulating_anything.rediscovery.elastic_collision import run_elastic_collision_rediscovery
from simulating_anything.rediscovery.elastic_pendulum import run_elastic_pendulum_rediscovery
from simulating_anything.rediscovery.fhn_ring import run_fhn_ring_rediscovery
from simulating_anything.rediscovery.fhn_lattice import run_fhn_lattice_rediscovery
from simulating_anything.rediscovery.fhn_spatial import run_fhn_spatial_rediscovery
from simulating_anything.rediscovery.fitzhugh_nagumo import run_fitzhugh_nagumo_rediscovery
from simulating_anything.rediscovery.fitzhugh_rinzel import run_fitzhugh_rinzel_rediscovery
from simulating_anything.rediscovery.four_species_lv import run_four_species_lv_rediscovery
from simulating_anything.rediscovery.genesio_tesi import run_genesio_tesi_rediscovery
from simulating_anything.rediscovery.fput import run_fput_rediscovery
from simulating_anything.rediscovery.ginzburg_landau import run_ginzburg_landau_rediscovery
from simulating_anything.rediscovery.gray_scott import run_gray_scott_analysis
from simulating_anything.rediscovery.gray_scott_1d import run_gray_scott_1d_rediscovery
from simulating_anything.rediscovery.halvorsen import run_halvorsen_rediscovery
from simulating_anything.rediscovery.harmonic_oscillator import run_harmonic_oscillator_rediscovery
from simulating_anything.rediscovery.harvested_population import (
    run_harvested_population_rediscovery,
)
from simulating_anything.rediscovery.heat_equation import run_heat_equation_rediscovery
from simulating_anything.rediscovery.henon_map import run_henon_map_rediscovery
from simulating_anything.rediscovery.hindmarsh_rose import run_hindmarsh_rose_rediscovery
from simulating_anything.rediscovery.hodgkin_huxley import run_hodgkin_huxley_rediscovery
from simulating_anything.rediscovery.ikeda_map import run_ikeda_map_rediscovery
from simulating_anything.rediscovery.ising_model import run_ising_model_rediscovery
from simulating_anything.rediscovery.kapitza_pendulum import run_kapitza_pendulum_rediscovery
from simulating_anything.rediscovery.kepler import run_kepler_rediscovery
from simulating_anything.rediscovery.langford import run_langford_rediscovery
from simulating_anything.rediscovery.laser_rate import run_laser_rate_rediscovery
from simulating_anything.rediscovery.kuramoto import run_kuramoto_rediscovery
from simulating_anything.rediscovery.kuramoto_sivashinsky import (
    run_kuramoto_sivashinsky_rediscovery,
)
from simulating_anything.rediscovery.logistic_map import run_logistic_map_rediscovery
from simulating_anything.rediscovery.lorenz import run_lorenz_rediscovery
from simulating_anything.rediscovery.lorenz_haken import run_lorenz_haken_rediscovery
from simulating_anything.rediscovery.lorenz84 import run_lorenz84_rediscovery
from simulating_anything.rediscovery.lorenz96 import run_lorenz96_rediscovery
from simulating_anything.rediscovery.lorenz_stenflo import run_lorenz_stenflo_rediscovery
from simulating_anything.rediscovery.lotka_volterra import run_lotka_volterra_rediscovery
from simulating_anything.rediscovery.mackey_glass import run_mackey_glass_rediscovery
from simulating_anything.rediscovery.magnetic_pendulum import run_magnetic_pendulum_rediscovery
from simulating_anything.rediscovery.may_leonard import run_may_leonard_rediscovery
from simulating_anything.rediscovery.morris_lecar import run_morris_lecar_rediscovery
from simulating_anything.rediscovery.navier_stokes import run_navier_stokes_rediscovery
from simulating_anything.rediscovery.network_sis import run_network_sis_rediscovery
from simulating_anything.rediscovery.nose_hoover import run_nose_hoover_rediscovery
from simulating_anything.rediscovery.oregonator import run_oregonator_rediscovery
from simulating_anything.rediscovery.oregonator_1d import run_oregonator_1d_rediscovery
from simulating_anything.rediscovery.predator_prey_mutualist import (
    run_predator_prey_mutualist_rediscovery,
)
from simulating_anything.rediscovery.projectile import run_projectile_rediscovery
from simulating_anything.rediscovery.quantum_oscillator import run_quantum_oscillator_rediscovery
from simulating_anything.rediscovery.rabinovich_fabrikant import (
    run_rabinovich_fabrikant_rediscovery,
)
from simulating_anything.rediscovery.rayleigh_benard import run_rayleigh_benard_rediscovery
from simulating_anything.rediscovery.ricker_map import run_ricker_map_rediscovery
from simulating_anything.rediscovery.rikitake import run_rikitake_rediscovery
from simulating_anything.rediscovery.rosenzweig_macarthur import (
    run_rosenzweig_macarthur_rediscovery,
)
from simulating_anything.rediscovery.rossler import run_rossler_rediscovery
from simulating_anything.rediscovery.sakarya import run_sakarya_rediscovery
from simulating_anything.rediscovery.rossler_hyperchaos import (
    run_rossler_hyperchaos_rediscovery,
)
from simulating_anything.rediscovery.schnakenberg import run_schnakenberg_rediscovery
from simulating_anything.rediscovery.schwarzschild import run_schwarzschild_rediscovery
from simulating_anything.rediscovery.selkov import run_selkov_rediscovery
from simulating_anything.rediscovery.shallow_water import run_shallow_water_rediscovery
from simulating_anything.rediscovery.sine_gordon import run_sine_gordon_rediscovery
from simulating_anything.rediscovery.sir_epidemic import run_sir_rediscovery
from simulating_anything.rediscovery.sir_vaccination import (
    run_sir_vaccination_rediscovery,
)
from simulating_anything.rediscovery.spring_mass_chain import run_spring_mass_chain_rediscovery
from simulating_anything.rediscovery.sprott import run_sprott_rediscovery
from simulating_anything.rediscovery.standard_map import run_standard_map_rediscovery
from simulating_anything.rediscovery.swinging_atwood import run_swinging_atwood_rediscovery
from simulating_anything.rediscovery.thomas import run_thomas_rediscovery
from simulating_anything.rediscovery.three_species import run_three_species_rediscovery
from simulating_anything.rediscovery.toda_lattice import run_toda_lattice_rediscovery
from simulating_anything.rediscovery.van_der_pol import run_van_der_pol_rediscovery
from simulating_anything.rediscovery.vicsek import run_vicsek_rediscovery
from simulating_anything.rediscovery.wilberforce import run_wilberforce_rediscovery
from simulating_anything.rediscovery.windmi import run_windmi_rediscovery
from simulating_anything.rediscovery.wilson_cowan import run_wilson_cowan_rediscovery
from simulating_anything.rediscovery.finance import run_finance_rediscovery
from simulating_anything.rediscovery.lu_chen import run_lu_chen_rediscovery
from simulating_anything.rediscovery.qi import run_qi_rediscovery
from simulating_anything.rediscovery.shimizu_morioka import run_shimizu_morioka_rediscovery
from simulating_anything.rediscovery.newton_leipnik import run_newton_leipnik_rediscovery
from simulating_anything.rediscovery.wang import run_wang_rediscovery
from simulating_anything.rediscovery.arneodo import run_arneodo_rediscovery
from simulating_anything.rediscovery.rucklidge import run_rucklidge_rediscovery
from simulating_anything.rediscovery.liu import run_liu_rediscovery
from simulating_anything.rediscovery.hadley import run_hadley_rediscovery
from simulating_anything.rediscovery.vallis import run_vallis_rediscovery
from simulating_anything.rediscovery.tigan import run_tigan_rediscovery
from simulating_anything.rediscovery.predator_two_prey import (
    run_predator_two_prey_rediscovery,
)
from simulating_anything.rediscovery.autocatalator import run_autocatalator_rediscovery
from simulating_anything.rediscovery.seir import run_seir_rediscovery
from simulating_anything.rediscovery.ueda import run_ueda_rediscovery
from simulating_anything.rediscovery.zombie_sir import run_zombie_sir_rediscovery

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
    "run_standard_map_rediscovery",
    "run_hodgkin_huxley_rediscovery",
    "run_eco_epidemic_rediscovery",
    "run_rayleigh_benard_rediscovery",
    "run_wilberforce_rediscovery",
    "run_hindmarsh_rose_rediscovery",
    "run_magnetic_pendulum_rediscovery",
    "run_competitive_lv_rediscovery",
    "run_vicsek_rediscovery",
    "run_bz_spiral_rediscovery",
    "run_coupled_lorenz_rediscovery",
    "run_swinging_atwood_rediscovery",
    "run_allee_predator_prey_rediscovery",
    "run_mackey_glass_rediscovery",
    "run_bouncing_ball_rediscovery",
    "run_wilson_cowan_rediscovery",
    "run_cable_equation_rediscovery",
    "run_may_leonard_rediscovery",
    "run_sine_gordon_rediscovery",
    "run_ikeda_map_rediscovery",
    "run_thomas_rediscovery",
    "run_cahn_hilliard_rediscovery",
    "run_network_sis_rediscovery",
    "run_delayed_predator_prey_rediscovery",
    "run_duffing_van_der_pol_rediscovery",
    "run_coupled_map_lattice_rediscovery",
    "run_schnakenberg_rediscovery",
    "run_kapitza_pendulum_rediscovery",
    "run_fitzhugh_rinzel_rediscovery",
    "run_lorenz84_rediscovery",
    "run_gray_scott_1d_rediscovery",
    "run_rabinovich_fabrikant_rediscovery",
    "run_sprott_rediscovery",
    "run_brusselator_2d_rediscovery",
    "run_predator_prey_mutualist_rediscovery",
    "run_selkov_rediscovery",
    "run_fput_rediscovery",
    "run_oregonator_1d_rediscovery",
    "run_ricker_map_rediscovery",
    "run_rikitake_rediscovery",
    "run_morris_lecar_rediscovery",
    "run_colpitts_rediscovery",
    "run_harvested_population_rediscovery",
    "run_rossler_hyperchaos_rediscovery",
    "run_fhn_ring_rediscovery",
    "run_laser_rate_rediscovery",
    "run_bazykin_rediscovery",
    "run_langford_rediscovery",
    "run_sir_vaccination_rediscovery",
    "run_fhn_lattice_rediscovery",
    "run_four_species_lv_rediscovery",
    "run_lorenz_stenflo_rediscovery",
    "run_chen_rediscovery",
    "run_aizawa_rediscovery",
    "run_halvorsen_rediscovery",
    "run_burke_shaw_rediscovery",
    "run_nose_hoover_rediscovery",
    "run_lorenz_haken_rediscovery",
    "run_sakarya_rediscovery",
    "run_dadras_rediscovery",
    "run_genesio_tesi_rediscovery",
    "run_lu_chen_rediscovery",
    "run_qi_rediscovery",
    "run_windmi_rediscovery",
    "run_finance_rediscovery",
    "run_shimizu_morioka_rediscovery",
    "run_newton_leipnik_rediscovery",
    "run_wang_rediscovery",
    "run_arneodo_rediscovery",
    "run_rucklidge_rediscovery",
    "run_liu_rediscovery",
    "run_hadley_rediscovery",
    "run_vallis_rediscovery",
    "run_tigan_rediscovery",
    "run_predator_two_prey_rediscovery",
    "run_autocatalator_rediscovery",
    "run_seir_rediscovery",
    "run_ueda_rediscovery",
    "run_cubic_map_rediscovery",
    "run_zombie_sir_rediscovery",
    "run_elastic_collision_rediscovery",
]
