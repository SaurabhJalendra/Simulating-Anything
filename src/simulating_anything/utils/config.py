"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from simulating_anything.types.simulation import Domain, SimulationBackend

# Default config directory relative to package root
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CONFIGS_DIR = _PACKAGE_ROOT / "configs"


class DomainConfig(BaseModel):
    """Per-domain simulation defaults."""

    domain: Domain
    backend: SimulationBackend = SimulationBackend.JAX_FD
    grid_resolution: tuple[int, ...] = (128, 128)
    domain_size: tuple[float, ...] = (1.0, 1.0)
    dt: float = 0.01
    n_steps: int = 1000
    boundary_conditions: str = "periodic"
    default_parameters: dict[str, float] = Field(default_factory=dict)
    sweep_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)


class WorldModelConfig(BaseModel):
    """World model training defaults."""

    learning_rate: float = 1e-4
    batch_size: int = 16
    sequence_length: int = 50
    n_epochs: int = 300
    warmup_steps: int = 1000
    grad_clip_norm: float = 100.0
    kl_free_bits: float = 1.0
    hidden_size: int = 512
    stochastic_classes: int = 32
    stochastic_vars: int = 32
    seed: int = 42


class ExplorationConfig(BaseModel):
    """Exploration loop defaults."""

    n_rounds: int = 5
    trajectories_per_round: int = 20
    uncertainty_threshold: float = 0.3
    novelty_weight: float = 0.5
    mc_dropout_samples: int = 10


class SimulatingAnythingConfig(BaseModel):
    """Top-level configuration for the entire pipeline."""

    output_dir: str = "output"
    log_level: str = "INFO"
    device: str = "cpu"
    dtype: str = "float32"
    seed: int = 42
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)
    exploration: ExplorationConfig = Field(default_factory=ExplorationConfig)
    domain_configs: dict[str, DomainConfig] = Field(default_factory=dict)


def load_config(path: str | Path | None = None) -> SimulatingAnythingConfig:
    """Load global config from a YAML file.

    Falls back to configs/default.yaml if no path is given.
    """
    if path is None:
        path = _CONFIGS_DIR / "default.yaml"
    path = Path(path)

    if not path.exists():
        return SimulatingAnythingConfig()

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return SimulatingAnythingConfig(**raw)


def load_domain_config(domain: Domain | str, path: str | Path | None = None) -> DomainConfig:
    """Load domain-specific config from a YAML file.

    Falls back to configs/domains/{domain}.yaml if no path is given.
    """
    if isinstance(domain, Domain):
        domain_str = domain.value
    else:
        domain_str = domain

    if path is None:
        path = _CONFIGS_DIR / "domains" / f"{domain_str}.yaml"
    path = Path(path)

    if not path.exists():
        return DomainConfig(domain=Domain(domain_str))

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    if "domain" not in raw:
        raw["domain"] = domain_str

    return DomainConfig(**raw)
