"""Tests for simulation engines."""

import numpy as np
import pytest

from simulating_anything.simulation.agent_based import LotkaVolterraSimulation
from simulating_anything.simulation.reaction_diffusion import GrayScottSimulation
from simulating_anything.simulation.rigid_body import ProjectileSimulation
from simulating_anything.types.simulation import Domain, SimulationBackend, SimulationConfig


def _gray_scott_config(**overrides) -> SimulationConfig:
    defaults = dict(
        domain=Domain.REACTION_DIFFUSION,
        backend=SimulationBackend.JAX_FD,
        grid_resolution=(32, 32),
        domain_size=(2.5, 2.5),
        dt=1.0,
        n_steps=10,
        parameters={"D_u": 0.16, "D_v": 0.08, "f": 0.035, "k": 0.065},
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def _projectile_config(**overrides) -> SimulationConfig:
    defaults = dict(
        domain=Domain.RIGID_BODY,
        grid_resolution=(1,),
        dt=0.01,
        n_steps=500,
        parameters={
            "gravity": 9.81,
            "drag_coefficient": 0.1,
            "mass": 1.0,
            "initial_speed": 30.0,
            "launch_angle": 45.0,
        },
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def _lotka_volterra_config(**overrides) -> SimulationConfig:
    defaults = dict(
        domain=Domain.AGENT_BASED,
        backend=SimulationBackend.DIFFRAX,
        grid_resolution=(1,),
        dt=0.01,
        n_steps=100,
        parameters={
            "alpha": 1.1,
            "beta": 0.4,
            "gamma": 0.4,
            "delta": 0.1,
            "prey_0": 40.0,
            "predator_0": 9.0,
        },
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


class TestGrayScott:
    def test_reset_shape(self):
        sim = GrayScottSimulation(_gray_scott_config())
        state = sim.reset()
        assert state.shape == (32, 32, 2)

    def test_step_preserves_shape(self):
        sim = GrayScottSimulation(_gray_scott_config())
        sim.reset()
        state = sim.step()
        assert state.shape == (32, 32, 2)

    def test_run_trajectory(self):
        sim = GrayScottSimulation(_gray_scott_config(n_steps=5))
        traj = sim.run()
        assert traj.states.shape == (6, 32, 32, 2)  # 5 steps + initial
        assert traj.timestamps.shape == (6,)

    def test_initial_conditions(self):
        sim = GrayScottSimulation(_gray_scott_config())
        state = sim.reset(seed=42)
        u = state[:, :, 0]
        v = state[:, :, 1]
        # u should be ~1.0 everywhere except center
        assert np.mean(u) > 0.8
        # v should be ~0.0 everywhere except center
        assert np.mean(v) < 0.2

    def test_values_stay_finite(self):
        # CFL condition for diffusion: dt < dx^2/(4*D_max)
        # dx = 2.5/32 ~ 0.078, D_max = 0.16, so dt < 0.0095
        sim = GrayScottSimulation(_gray_scott_config(n_steps=50, dt=0.005))
        traj = sim.run()
        assert np.all(np.isfinite(traj.states))


class TestProjectile:
    def test_reset_state(self):
        sim = ProjectileSimulation(_projectile_config())
        state = sim.reset()
        assert state.shape == (4,)
        assert state[0] == 0.0  # x = 0
        assert state[1] == 0.0  # y = 0

    def test_step(self):
        sim = ProjectileSimulation(_projectile_config())
        sim.reset()
        state = sim.step()
        assert state[0] > 0  # x increases
        assert state[1] > 0  # y increases initially

    def test_projectile_lands(self):
        sim = ProjectileSimulation(_projectile_config(n_steps=2000))
        traj = sim.run()
        # Find where y returns to 0
        y_vals = traj.states[:, 1]
        # At some point y should be at or near 0 after initial rise
        landed = np.any(y_vals[10:] <= 0.01)
        assert landed or y_vals[-1] < y_vals[len(y_vals) // 2]

    def test_no_drag_goes_further(self):
        config_drag = _projectile_config(n_steps=1000)
        config_no_drag = _projectile_config(
            n_steps=1000,
            parameters={**config_drag.parameters, "drag_coefficient": 0.0},
        )

        sim_drag = ProjectileSimulation(config_drag)
        sim_no_drag = ProjectileSimulation(config_no_drag)

        traj_drag = sim_drag.run()
        traj_no_drag = sim_no_drag.run()

        max_x_drag = np.max(traj_drag.states[:, 0])
        max_x_no_drag = np.max(traj_no_drag.states[:, 0])
        assert max_x_no_drag >= max_x_drag


class TestLotkaVolterra:
    def test_reset_state(self):
        sim = LotkaVolterraSimulation(_lotka_volterra_config())
        state = sim.reset()
        assert state.shape == (2,)
        assert state[0] == 40.0  # prey_0
        assert state[1] == 9.0  # predator_0

    def test_step(self):
        sim = LotkaVolterraSimulation(_lotka_volterra_config())
        sim.reset()
        state = sim.step()
        assert state.shape == (2,)
        # Populations should change but stay positive
        assert state[0] >= 0
        assert state[1] >= 0

    def test_run_trajectory(self):
        sim = LotkaVolterraSimulation(_lotka_volterra_config(n_steps=50))
        traj = sim.run()
        assert traj.states.shape == (51, 2)  # 50 steps + initial
        assert np.all(traj.states >= 0)  # Positivity enforced

    def test_populations_oscillate(self):
        # Need enough time steps to see full oscillation (~period of ~5-15 time units)
        sim = LotkaVolterraSimulation(_lotka_volterra_config(n_steps=5000))
        traj = sim.run()
        prey = traj.states[:, 0]
        # Check for oscillation: prey should have at least one local max and min
        diffs = np.diff(prey)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        assert sign_changes >= 2  # At least one full oscillation

    def test_observe(self):
        sim = LotkaVolterraSimulation(_lotka_volterra_config())
        sim.reset()
        sim.step()
        obs = sim.observe()
        assert obs.shape == (2,)
