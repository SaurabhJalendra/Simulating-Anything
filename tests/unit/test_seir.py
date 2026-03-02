"""Tests for the SEIR epidemic model simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.seir import SEIRSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    dt: float = 0.1,
    n_steps: int = 1000,
    **param_overrides: float,
) -> SimulationConfig:
    """Create a SimulationConfig for SEIR with optional overrides."""
    params: dict[str, float] = {
        "beta": 0.5,
        "sigma": 0.2,
        "gamma": 0.1,
        "N": 1000.0,
        "S_0": 990.0,
        "E_0": 5.0,
        "I_0": 5.0,
        "R_0_init": 0.0,
    }
    params.update(param_overrides)
    return SimulationConfig(
        domain=Domain.SEIR,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestSEIRSimulation:
    """Core simulation tests."""

    def test_initial_state_shape(self):
        """State vector should be [S, E, I, R] with 4 components."""
        sim = SEIRSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (4,)

    def test_initial_state_values(self):
        """Initial conditions should match parameters."""
        sim = SEIRSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state, [990.0, 5.0, 5.0, 0.0])

    def test_initial_conservation(self):
        """S + E + I + R should equal N at initialization."""
        sim = SEIRSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state.sum(), 1000.0, atol=1e-10)

    def test_conservation_throughout(self):
        """S + E + I + R = N should hold at all times."""
        sim = SEIRSimulation(_make_config())
        sim.reset()
        for _ in range(500):
            state = sim.step()
            np.testing.assert_allclose(
                state.sum(), 1000.0, atol=1e-6,
                err_msg="Population conservation violated",
            )

    def test_non_negative(self):
        """All compartments should remain non-negative."""
        sim = SEIRSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert state[0] >= 0, f"S became negative: {state[0]}"
            assert state[1] >= 0, f"E became negative: {state[1]}"
            assert state[2] >= 0, f"I became negative: {state[2]}"
            assert state[3] >= 0, f"R became negative: {state[3]}"

    def test_step_advances(self):
        """State should change after a step."""
        sim = SEIRSimulation(_make_config())
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_returns_state(self):
        """Observe should return the current state."""
        sim = SEIRSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_deterministic(self):
        """Same config produces same trajectory."""
        config = _make_config()
        sim1 = SEIRSimulation(config)
        sim1.reset()
        for _ in range(100):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = SEIRSimulation(config)
        sim2.reset()
        for _ in range(100):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)

    def test_epidemic_occurs(self):
        """With R0 > 1, infectious should rise then fall."""
        sim = SEIRSimulation(_make_config(beta=0.5, gamma=0.1))  # R0 = 5
        sim.reset()
        max_I = 0
        for _ in range(10000):
            state = sim.step()
            max_I = max(max_I, state[2])
        # Significant epidemic peak
        assert max_I > 50.0, f"Peak I too low: {max_I}"
        # Epidemic should end
        assert state[2] < 1.0, f"I should decay: {state[2]}"

    def test_no_epidemic(self):
        """With R0 < 1, infectious should decay."""
        sim = SEIRSimulation(
            _make_config(beta=0.05, gamma=0.1, E_0=0.0)
        )  # R0 = 0.5
        sim.reset()
        initial_I = sim.observe()[2]
        for _ in range(1000):
            state = sim.step()
        assert state[2] < initial_I, (
            f"I should decay when R0 < 1: I={state[2]}, I_0={initial_I}"
        )

    def test_exposed_precedes_infectious(self):
        """Exposed should peak before infectious (latent delay)."""
        sim = SEIRSimulation(
            _make_config(beta=0.5, sigma=0.1, gamma=0.1, E_0=0.0)
        )
        sim.reset()
        peak_E_time = 0
        peak_I_time = 0
        peak_E = 0.0
        peak_I = 0.0
        for step in range(10000):
            state = sim.step()
            if state[1] > peak_E:
                peak_E = state[1]
                peak_E_time = step
            if state[2] > peak_I:
                peak_I = state[2]
                peak_I_time = step
        assert peak_E_time < peak_I_time, (
            f"Exposed peak ({peak_E_time}) should precede "
            f"infectious peak ({peak_I_time})"
        )

    def test_long_run_stability(self):
        """No NaN after many steps."""
        sim = SEIRSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN or Inf in state: {state}"

    def test_run_method(self):
        """run() should produce a TrajectoryData object."""
        sim = SEIRSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 4)
        assert len(traj.timestamps) == 101

    def test_normalization_on_reset(self):
        """Initial conditions should normalize to N even if sum differs."""
        sim = SEIRSimulation(_make_config(
            S_0=500.0, E_0=100.0, I_0=100.0, R_0_init=100.0, N=1000.0,
        ))
        state = sim.reset()
        # Sum should be N = 1000 (input sums to 800, so scale by 1.25)
        np.testing.assert_allclose(state.sum(), 1000.0, atol=1e-10)


class TestSEIRProperties:
    """Tests for analytical properties and computed quantities."""

    def test_basic_reproduction_number(self):
        """R0 = beta / gamma."""
        sim = SEIRSimulation(_make_config(beta=0.5, gamma=0.1))
        assert sim.basic_reproduction_number == pytest.approx(5.0)

    def test_basic_reproduction_number_subcritical(self):
        """R0 < 1 when gamma > beta."""
        sim = SEIRSimulation(_make_config(beta=0.05, gamma=0.1))
        assert sim.basic_reproduction_number == pytest.approx(0.5)

    def test_latent_period(self):
        """Latent period = 1 / sigma."""
        sim = SEIRSimulation(_make_config(sigma=0.2))
        assert sim.latent_period == pytest.approx(5.0)

    def test_infectious_period(self):
        """Infectious period = 1 / gamma."""
        sim = SEIRSimulation(_make_config(gamma=0.1))
        assert sim.infectious_period == pytest.approx(10.0)

    def test_generation_time(self):
        """Generation time = 1/sigma + 1/gamma."""
        sim = SEIRSimulation(_make_config(sigma=0.2, gamma=0.1))
        assert sim.generation_time == pytest.approx(15.0)

    def test_jacobian_shape(self):
        """Jacobian should be 4x4."""
        sim = SEIRSimulation(_make_config())
        sim.reset()
        J = sim.jacobian()
        assert J.shape == (4, 4)

    def test_jacobian_at_dfe(self):
        """Jacobian at disease-free equilibrium should have known structure."""
        sim = SEIRSimulation(_make_config(beta=0.5, sigma=0.2, gamma=0.1))
        dfe = sim.disease_free_equilibrium()
        J = sim.jacobian(dfe)
        # At DFE (N, 0, 0, 0): force_of_infection terms vanish for I=0
        # J[2,1] should be sigma
        assert J[2, 1] == pytest.approx(0.2)
        # J[2,2] should be -gamma
        assert J[2, 2] == pytest.approx(-0.1)
        # J[3,2] should be gamma
        assert J[3, 2] == pytest.approx(0.1)

    def test_compute_divergence(self):
        """Divergence should be computable and finite."""
        sim = SEIRSimulation(_make_config())
        sim.reset()
        div = sim.compute_divergence()
        assert np.isfinite(div)

    def test_disease_free_equilibrium(self):
        """DFE should be [N, 0, 0, 0]."""
        sim = SEIRSimulation(_make_config(N=1000.0))
        dfe = sim.disease_free_equilibrium()
        np.testing.assert_allclose(dfe, [1000.0, 0.0, 0.0, 0.0])

    def test_total_population_property(self):
        """total_population should return sum of compartments."""
        sim = SEIRSimulation(_make_config(N=1000.0))
        sim.reset()
        assert sim.total_population == pytest.approx(1000.0)

    def test_final_size_supercritical(self):
        """Final size should be positive when R0 > 1."""
        sim = SEIRSimulation(_make_config(beta=0.5, gamma=0.1))
        fs = sim.final_size()
        assert fs > 0.5, f"Final size too low for R0=5: {fs}"

    def test_final_size_subcritical(self):
        """Final size should be 0 when R0 < 1."""
        sim = SEIRSimulation(_make_config(beta=0.05, gamma=0.1))
        assert sim.final_size() == 0.0


class TestSEIRRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_generate_sweep_data(self):
        """Sweep data generation should produce valid arrays."""
        from simulating_anything.rediscovery.seir import (
            generate_seir_sweep_data,
        )
        data = generate_seir_sweep_data(n_samples=5, n_steps=500, dt=0.1)
        assert "beta" in data
        assert "sigma" in data
        assert "gamma" in data
        assert "R0" in data
        assert "peak_I" in data
        assert "peak_E" in data
        assert "final_size" in data
        assert "time_to_peak" in data
        assert len(data["beta"]) == 5
        assert np.all(data["R0"] > 0)
        assert np.all(np.isfinite(data["peak_I"]))

    def test_generate_ode_data(self):
        """ODE data generation should produce valid trajectory."""
        from simulating_anything.rediscovery.seir import (
            generate_seir_ode_data,
        )
        data = generate_seir_ode_data(n_steps=100, dt=0.1)
        assert data["states"].shape == (101, 4)
        # Population should be conserved
        totals = data["states"].sum(axis=1)
        np.testing.assert_allclose(totals, 1000.0, atol=1.0)

    def test_sweep_r0_range(self):
        """Sweep should cover both sub- and super-critical R0."""
        from simulating_anything.rediscovery.seir import (
            generate_seir_sweep_data,
        )
        data = generate_seir_sweep_data(n_samples=50, n_steps=500, dt=0.1)
        assert np.any(data["R0"] < 1.0), "Should have some R0 < 1 cases"
        assert np.any(data["R0"] > 1.0), "Should have some R0 > 1 cases"

    def test_ode_data_parameters(self):
        """ODE data should include ground truth parameters."""
        from simulating_anything.rediscovery.seir import (
            generate_seir_ode_data,
        )
        data = generate_seir_ode_data(n_steps=50, dt=0.1)
        assert data["beta"] == 0.5
        assert data["sigma"] == 0.2
        assert data["gamma"] == 0.1
        assert data["N"] == 1000.0
