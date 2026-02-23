"""Tests for the configuration system."""

import pytest
import yaml

from simulating_anything.types.simulation import Domain, SimulationBackend
from simulating_anything.utils.config import (
    DomainConfig,
    ExplorationConfig,
    SimulatingAnythingConfig,
    WorldModelConfig,
    load_config,
    load_domain_config,
)


class TestWorldModelConfig:
    def test_defaults(self):
        config = WorldModelConfig()
        assert config.learning_rate == 1e-4
        assert config.hidden_size == 512
        assert config.stochastic_classes == 32
        assert config.stochastic_vars == 32

    def test_custom_values(self):
        config = WorldModelConfig(learning_rate=3e-4, batch_size=32)
        assert config.learning_rate == 3e-4
        assert config.batch_size == 32


class TestExplorationConfig:
    def test_defaults(self):
        config = ExplorationConfig()
        assert config.n_rounds == 5
        assert config.mc_dropout_samples == 10


class TestSimulatingAnythingConfig:
    def test_defaults(self):
        config = SimulatingAnythingConfig()
        assert config.output_dir == "output"
        assert config.device == "cpu"
        assert config.seed == 42

    def test_nested_configs(self):
        config = SimulatingAnythingConfig()
        assert isinstance(config.world_model, WorldModelConfig)
        assert isinstance(config.exploration, ExplorationConfig)


class TestDomainConfig:
    def test_reaction_diffusion(self):
        config = DomainConfig(
            domain=Domain.REACTION_DIFFUSION,
            default_parameters={"D_u": 0.16, "D_v": 0.08},
        )
        assert config.domain == Domain.REACTION_DIFFUSION
        assert config.default_parameters["D_u"] == 0.16

    def test_rigid_body(self):
        config = DomainConfig(domain=Domain.RIGID_BODY, dt=0.01)
        assert config.dt == 0.01


class TestLoadConfig:
    def test_load_default_config(self):
        config = load_config()
        assert isinstance(config, SimulatingAnythingConfig)
        assert config.log_level == "INFO"

    def test_load_nonexistent_returns_defaults(self, tmp_path):
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config.output_dir == "output"

    def test_load_from_yaml(self, tmp_path):
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump({
            "output_dir": "custom_output",
            "seed": 123,
            "world_model": {"learning_rate": 3e-4},
        }))
        config = load_config(yaml_path)
        assert config.output_dir == "custom_output"
        assert config.seed == 123
        assert config.world_model.learning_rate == 3e-4


class TestLoadDomainConfig:
    def test_load_reaction_diffusion(self):
        config = load_domain_config(Domain.REACTION_DIFFUSION)
        assert config.domain == Domain.REACTION_DIFFUSION
        assert "D_u" in config.default_parameters

    def test_load_rigid_body(self):
        config = load_domain_config("rigid_body")
        assert config.domain == Domain.RIGID_BODY
        assert "gravity" in config.default_parameters

    def test_load_agent_based(self):
        config = load_domain_config(Domain.AGENT_BASED)
        assert config.domain == Domain.AGENT_BASED
        assert "alpha" in config.default_parameters

    def test_load_nonexistent_domain(self, tmp_path):
        # Should return defaults with the specified domain
        config = load_domain_config(Domain.REACTION_DIFFUSION, tmp_path / "nope.yaml")
        assert config.domain == Domain.REACTION_DIFFUSION
        assert config.default_parameters == {}
