"""Simulation configuration, domain classification, and world model checkpoint types."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class Domain(str, Enum):
    REACTION_DIFFUSION = "reaction_diffusion"
    RIGID_BODY = "rigid_body"
    AGENT_BASED = "agent_based"
    EPIDEMIOLOGICAL = "epidemiological"
    CHAOTIC_ODE = "chaotic_ode"
    HARMONIC_OSCILLATOR = "harmonic_oscillator"
    LORENZ_ATTRACTOR = "lorenz_attractor"
    NAVIER_STOKES_2D = "navier_stokes_2d"
    VAN_DER_POL = "van_der_pol"
    KURAMOTO = "kuramoto"
    BRUSSELATOR = "brusselator"
    FITZHUGH_NAGUMO = "fitzhugh_nagumo"
    HEAT_EQUATION_1D = "heat_equation_1d"
    LOGISTIC_MAP = "logistic_map"
    BOLTZMANN_GAS = "boltzmann_gas"
    DUFFING = "duffing"
    SCHWARZSCHILD = "schwarzschild"
    QUANTUM_OSCILLATOR = "quantum_oscillator"
    SPRING_MASS_CHAIN = "spring_mass_chain"
    KEPLER = "kepler"
    DRIVEN_PENDULUM = "driven_pendulum"
    COUPLED_OSCILLATORS = "coupled_oscillators"
    DIFFUSIVE_LV = "diffusive_lv"
    DAMPED_WAVE = "damped_wave"
    ISING_MODEL = "ising_model"
    THREE_SPECIES = "three_species"
    CART_POLE = "cart_pole"
    ELASTIC_PENDULUM = "elastic_pendulum"
    ROSSLER = "rossler"
    HENON_MAP = "henon_map"
    BRUSSELATOR_DIFFUSION = "brusselator_diffusion"
    ROSENZWEIG_MACARTHUR = "rosenzweig_macarthur"
    SHALLOW_WATER = "shallow_water"
    CHUA = "chua"
    TODA_LATTICE = "toda_lattice"
    KURAMOTO_SIVASHINSKY = "kuramoto_sivashinsky"
    OREGONATOR = "oregonator"
    BAK_SNEPPEN = "bak_sneppen"
    GINZBURG_LANDAU = "ginzburg_landau"
    LORENZ96 = "lorenz96"
    FHN_SPATIAL = "fhn_spatial"
    CHEMOSTAT = "chemostat"
    WILBERFORCE = "wilberforce"
    STANDARD_MAP = "standard_map"
    HODGKIN_HUXLEY = "hodgkin_huxley"
    ECO_EPIDEMIC = "eco_epidemic"
    RAYLEIGH_BENARD = "rayleigh_benard"
    HINDMARSH_ROSE = "hindmarsh_rose"
    MAGNETIC_PENDULUM = "magnetic_pendulum"
    VICSEK = "vicsek"
    COMPETITIVE_LV = "competitive_lv"
    BZ_SPIRAL = "bz_spiral"
    COUPLED_LORENZ = "coupled_lorenz"
    SWINGING_ATWOOD = "swinging_atwood"
    ALLEE_PREDATOR_PREY = "allee_predator_prey"
    MACKEY_GLASS = "mackey_glass"
    BOUNCING_BALL = "bouncing_ball"
    WILSON_COWAN = "wilson_cowan"
    CABLE_EQUATION = "cable_equation"
    MAY_LEONARD = "may_leonard"
    SINE_GORDON = "sine_gordon"
    IKEDA_MAP = "ikeda_map"
    THOMAS = "thomas"
    CAHN_HILLIARD = "cahn_hilliard"
    DELAYED_PREDATOR_PREY = "delayed_predator_prey"
    NETWORK_SIS = "network_sis"
    DUFFING_VAN_DER_POL = "duffing_van_der_pol"
    FITZHUGH_RINZEL = "fitzhugh_rinzel"
    COUPLED_MAP_LATTICE = "coupled_map_lattice"
    SCHNAKENBERG = "schnakenberg"
    KAPITZA_PENDULUM = "kapitza_pendulum"
    LORENZ_84 = "lorenz_84"
    GRAY_SCOTT_1D = "gray_scott_1d"
    RABINOVICH_FABRIKANT = "rabinovich_fabrikant"
    SPROTT = "sprott"
    BRUSSELATOR_2D = "brusselator_2d"
    SELKOV = "selkov"
    PREDATOR_PREY_MUTUALIST = "predator_prey_mutualist"
    FPUT = "fput"


class SimulationBackend(str, Enum):
    PHIFLOW = "phiflow"
    JAX_FD = "jax_finite_differences"
    BRAX = "brax"
    MJX = "mjx"
    CUSTOM_JAX = "custom_jax"
    DIFFRAX = "diffrax"


class SimulationConfig(BaseModel):
    """Configuration for instantiating and running a simulation."""

    domain: Domain
    backend: SimulationBackend = SimulationBackend.JAX_FD
    grid_resolution: tuple[int, ...] = (128, 128)
    domain_size: tuple[float, ...] = (1.0, 1.0)
    dt: float = 0.01
    n_steps: int = 1000
    parameters: dict[str, float] = Field(default_factory=dict)
    boundary_conditions: str = "periodic"
    initial_conditions: str = "default"
    seed: int = 42


class DomainClassification(BaseModel):
    """Output of the Domain Classifier Agent."""

    domain: Domain
    confidence: float = 1.0
    backend: SimulationBackend = SimulationBackend.JAX_FD
    template: str = ""
    rationale: str = ""


class Provenance(BaseModel):
    """Full provenance chain for reproducibility."""

    code_version: str = ""
    random_seed: int = 0
    hardware: str = ""
    wall_clock_time: float = 0.0
    config_hash: str = ""


class TrainingConfig(BaseModel):
    """World model training hyperparameters."""

    learning_rate: float = 1e-4
    batch_size: int = 16
    sequence_length: int = 50
    n_epochs: int = 300
    warmup_steps: int = 1000
    grad_clip_norm: float = 100.0
    kl_free_bits: float = 1.0
    seed: int = 42


class ValidationMetrics(BaseModel):
    """Metrics from world model validation."""

    reconstruction_mse: float = 0.0
    kl_divergence: float = 0.0
    prediction_error_1step: float = 0.0
    prediction_error_50step: float = 0.0
    best_epoch: int = 0


class WorldModelCheckpoint(BaseModel):
    """A saved world model with metadata."""

    model_id: str = ""
    path: str = ""
    domain: Domain = Domain.REACTION_DIFFUSION
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)
    validation_metrics: ValidationMetrics = Field(default_factory=ValidationMetrics)
    provenance: Provenance = Field(default_factory=Provenance)
    epoch: int = 0

    def checkpoint_path(self) -> Path:
        return Path(self.path)
