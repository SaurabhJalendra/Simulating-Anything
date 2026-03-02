"""Tests for the FitzHugh-Nagumo coupled pair simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.fhn_coupled_pair import FHNCoupledPairSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    g: float = 0.1,
    I1: float = 0.5,
    I2: float = 0.5,
    dt: float = 0.05,
    n_steps: int = 2000,
    **overrides,
) -> SimulationConfig:
    params = {
        "a": 0.7, "b": 0.8, "eps": 0.08,
        "I1": I1, "I2": I2, "g": g,
        "v1_0": -1.0, "w1_0": -0.5,
        "v2_0": 0.5, "w2_0": 0.0,
    }
    params.update(overrides)
    return SimulationConfig(
        domain=Domain.FHN_COUPLED_PAIR,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestFHNCoupledPairBasics:
    """Basic simulation tests: shape, determinism, stability."""

    def test_observe_shape(self):
        """State should be a 4-element vector [v1, w1, v2, w2]."""
        sim = FHNCoupledPairSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (4,)
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_state_dimensionality(self):
        """State must be exactly 4D: two (v, w) pairs."""
        sim = FHNCoupledPairSimulation(_make_config())
        state = sim.reset()
        assert state.ndim == 1
        assert len(state) == 4

    def test_initial_conditions(self):
        """Reset should set the specified initial conditions."""
        sim = FHNCoupledPairSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state, [-1.0, -0.5, 0.5, 0.0])

    def test_custom_initial_conditions(self):
        """Custom ICs should be respected."""
        config = _make_config(v1_0=1.5, w1_0=0.2, v2_0=-1.5, w2_0=-0.2)
        sim = FHNCoupledPairSimulation(config)
        state = sim.reset()
        np.testing.assert_allclose(state, [1.5, 0.2, -1.5, -0.2])

    def test_step_advances(self):
        """State should change after one step."""
        sim = FHNCoupledPairSimulation(_make_config())
        s0 = sim.reset().copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1)

    def test_deterministic(self):
        """Same config should produce identical trajectories."""
        config = _make_config()
        sim1 = FHNCoupledPairSimulation(config)
        sim1.reset()
        for _ in range(200):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = FHNCoupledPairSimulation(config)
        sim2.reset()
        for _ in range(200):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_array_equal(state1, state2)

    def test_rk4_stability(self):
        """No NaN or Inf should appear during integration."""
        sim = FHNCoupledPairSimulation(_make_config())
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State became NaN/Inf: {state}"

    def test_run_returns_trajectory(self):
        """run() should return TrajectoryData with correct shape."""
        config = _make_config(n_steps=100)
        sim = FHNCoupledPairSimulation(config)
        traj = sim.run(100)
        assert traj.states.shape == (101, 4)
        assert traj.timestamps.shape == (101,)

    def test_default_parameters(self):
        """Default parameters should match expected values."""
        config = SimulationConfig(
            domain=Domain.FHN_COUPLED_PAIR,
            dt=0.05,
            n_steps=100,
            parameters={},
        )
        sim = FHNCoupledPairSimulation(config)
        assert sim.a == 0.7
        assert sim.b_param == 0.8
        assert sim.eps == 0.08
        assert sim.I1 == 0.5
        assert sim.I2 == 0.5
        assert sim.g == 0.1


class TestFHNCoupledPairDynamics:
    """Tests for core physical behavior: oscillations and synchronization."""

    def test_trajectory_bounded(self):
        """Both neurons should remain bounded during integration."""
        sim = FHNCoupledPairSimulation(_make_config(g=0.5))
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            # FHN voltage stays in roughly [-3, 3] range
            assert np.all(np.abs(state) < 20), (
                f"State diverged: {state}"
            )

    def test_zero_coupling_independent(self):
        """With g=0, neurons evolve independently and can have large error."""
        config = _make_config(
            g=0.0,
            v1_0=-1.0, w1_0=-0.5,
            v2_0=0.5, w2_0=0.0,
        )
        sim = FHNCoupledPairSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        # Both should be oscillating but potentially out of phase
        errors = []
        for _ in range(5000):
            sim.step()
            errors.append(sim.compute_sync_error())
        # Time-averaged error should be nonzero for different ICs
        mean_err = np.mean(errors)
        assert mean_err > 0.01, (
            f"Mean error {mean_err:.4f} unexpectedly small for g=0"
        )

    def test_strong_coupling_synchronization(self):
        """Strong coupling should synchronize neurons (v1 ~ v2)."""
        config = _make_config(
            g=1.0,
            v1_0=-1.0, w1_0=-0.5,
            v2_0=0.5, w2_0=0.0,
        )
        sim = FHNCoupledPairSimulation(config)
        sim.reset()
        # Run through transient -- strong coupling needs time to sync
        for _ in range(20000):
            sim.step()
        # Measure sync error after transient
        errors = []
        for _ in range(5000):
            sim.step()
            errors.append(sim.compute_sync_error())
        mean_err = np.mean(errors)
        assert mean_err < 0.5, (
            f"Strong coupling sync error {mean_err:.4f} too large"
        )

    def test_repulsive_coupling_anti_phase(self):
        """With g < 0, neurons should tend toward anti-phase behavior."""
        # Start with slightly different ICs to break the sync manifold
        # symmetry -- identical ICs stay on the manifold even with g < 0
        config = _make_config(
            g=-0.3,
            v1_0=-1.0, w1_0=-0.5,
            v2_0=-0.9, w2_0=-0.45,
        )
        sim = FHNCoupledPairSimulation(config)
        sim.reset()
        for _ in range(15000):
            sim.step()
        # With repulsive coupling from near-identical ICs, states should diverge
        errors = []
        for _ in range(5000):
            sim.step()
            errors.append(sim.compute_sync_error())
        mean_err = np.mean(errors)
        # Should not stay perfectly synchronized
        assert mean_err > 0.01, (
            f"Repulsive coupling error {mean_err:.4f} too small"
        )

    def test_identical_neurons_same_ic_converge(self):
        """Identical neurons with same IC should stay synchronized."""
        config = _make_config(
            g=0.5,
            I1=0.5, I2=0.5,
            v1_0=-1.0, w1_0=-0.5,
            v2_0=-1.0, w2_0=-0.5,
        )
        sim = FHNCoupledPairSimulation(config)
        sim.reset()
        # On the sync manifold, should remain exactly synced
        for _ in range(5000):
            sim.step()
        err = sim.compute_sync_error()
        assert err < 1e-10, (
            f"Identical IC sync error {err:.2e} should be ~0"
        )

    def test_sync_error_decreases_with_coupling(self):
        """Sync error should generally decrease as coupling increases."""
        errors_by_g = {}
        for g_val in [0.0, 0.5]:
            config = _make_config(
                g=g_val,
                v1_0=-1.0, w1_0=-0.5,
                v2_0=0.5, w2_0=0.0,
            )
            sim = FHNCoupledPairSimulation(config)
            sim.reset()
            for _ in range(15000):
                sim.step()
            err_sum = 0.0
            n_meas = 5000
            for _ in range(n_meas):
                sim.step()
                err_sum += sim.compute_sync_error()
            errors_by_g[g_val] = err_sum / n_meas

        assert errors_by_g[0.5] < errors_by_g[0.0], (
            f"Error at g=0.5 ({errors_by_g[0.5]:.4f}) not less than "
            f"at g=0 ({errors_by_g[0.0]:.4f})"
        )

    def test_coupling_effect_on_dynamics(self):
        """Coupling should affect the dynamics -- states differ from g=0."""
        config0 = _make_config(g=0.0)
        sim0 = FHNCoupledPairSimulation(config0)
        sim0.reset()
        for _ in range(500):
            sim0.step()
        state0 = sim0.observe().copy()

        config1 = _make_config(g=0.5)
        sim1 = FHNCoupledPairSimulation(config1)
        sim1.reset()
        for _ in range(500):
            sim1.step()
        state1 = sim1.observe().copy()

        assert not np.allclose(state0, state1), (
            "Coupling has no effect on dynamics"
        )

    def test_oscillatory_behavior(self):
        """With I=0.5 (above Hopf threshold), neurons should oscillate."""
        sim = FHNCoupledPairSimulation(_make_config(I1=0.5, I2=0.5))
        assert sim.is_oscillatory()


class TestFHNCoupledPairDerivatives:
    """Tests for the derivative computation."""

    def test_derivatives_decoupled(self):
        """With g=0, each neuron follows standard FHN dynamics."""
        sim = FHNCoupledPairSimulation(_make_config(g=0.0))
        sim.reset()
        state = np.array([1.0, 0.5, 2.0, -0.3])
        derivs = sim._derivatives(state)
        # dv1/dt = v1 - v1^3/3 - w1 + I1 = 1 - 1/3 - 0.5 + 0.5 = 0.6667
        assert np.isclose(derivs[0], 1.0 - 1.0 / 3 - 0.5 + 0.5)
        # dw1/dt = eps*(v1 + a - b*w1) = 0.08*(1.0 + 0.7 - 0.8*0.5)
        assert np.isclose(derivs[1], 0.08 * (1.0 + 0.7 - 0.8 * 0.5))
        # dv2/dt = v2 - v2^3/3 - w2 + I2 = 2 - 8/3 + 0.3 + 0.5
        assert np.isclose(derivs[2], 2.0 - 8.0 / 3 + 0.3 + 0.5)
        # dw2/dt = eps*(v2 + a - b*w2) = 0.08*(2.0 + 0.7 - 0.8*(-0.3))
        assert np.isclose(derivs[3], 0.08 * (2.0 + 0.7 + 0.24))

    def test_coupling_term_appears(self):
        """Coupling g*(v2-v1) should modify dv1 and g*(v1-v2) dv2."""
        sim0 = FHNCoupledPairSimulation(_make_config(g=0.0))
        sim0.reset()
        sim2 = FHNCoupledPairSimulation(_make_config(g=0.5))
        sim2.reset()

        state = np.array([1.0, 0.5, 3.0, -0.3])
        d0 = sim0._derivatives(state)
        d2 = sim2._derivatives(state)

        # dw1/dt and dw2/dt should be identical (coupling is in dv)
        assert np.isclose(d0[1], d2[1])
        assert np.isclose(d0[3], d2[3])

        # dv1 changes by g*(v2-v1) = 0.5*(3-1) = 1.0
        assert np.isclose(d2[0] - d0[0], 1.0)
        # dv2 changes by g*(v1-v2) = 0.5*(1-3) = -1.0
        assert np.isclose(d2[2] - d0[2], -1.0)

    def test_derivatives_at_sync_manifold(self):
        """On v1=v2, w1=w2, the coupling terms vanish (for I1=I2)."""
        sim = FHNCoupledPairSimulation(_make_config(g=5.0, I1=0.5, I2=0.5))
        sim.reset()
        state = np.array([1.5, 0.3, 1.5, 0.3])
        derivs = sim._derivatives(state)
        # On the sync manifold with identical params, derivatives are equal
        np.testing.assert_allclose(derivs[:2], derivs[2:])

    def test_coupling_antisymmetric(self):
        """Coupling is antisymmetric: g*(v2-v1) = -g*(v1-v2)."""
        sim = FHNCoupledPairSimulation(_make_config(g=1.0))
        sim.reset()
        state = np.array([1.0, 0.0, 3.0, 0.0])
        derivs = sim._derivatives(state)
        # dv1 coupling contribution: g*(v2-v1) = 1*(3-1) = +2
        # dv2 coupling contribution: g*(v1-v2) = 1*(1-3) = -2
        # Without coupling: dv1 = 1-1/3-0+0.5 = 1.1667
        #                    dv2 = 3-9+0+0.5 = -5.5
        dv1_uncoupled = 1.0 - 1.0 / 3 - 0.0 + 0.5
        dv2_uncoupled = 3.0 - 27.0 / 3 - 0.0 + 0.5
        assert np.isclose(derivs[0], dv1_uncoupled + 2.0)
        assert np.isclose(derivs[2], dv2_uncoupled - 2.0)


class TestSyncErrorAndPhase:
    """Tests for sync error and phase difference computations."""

    def test_sync_error_zero_on_manifold(self):
        """Sync error should be 0 when both neurons are identical."""
        config = _make_config(v1_0=1.0, w1_0=0.5, v2_0=1.0, w2_0=0.5)
        sim = FHNCoupledPairSimulation(config)
        sim.reset()
        assert sim.compute_sync_error() == pytest.approx(0.0, abs=1e-15)

    def test_sync_error_positive_when_different(self):
        """Sync error should be positive when neurons differ."""
        sim = FHNCoupledPairSimulation(_make_config())
        sim.reset()
        err = sim.compute_sync_error()
        assert err > 0

    def test_sync_error_with_explicit_state(self):
        """compute_sync_error should accept an explicit state argument."""
        sim = FHNCoupledPairSimulation(_make_config())
        sim.reset()
        state = np.array([1.0, 0.0, 4.0, 0.0])
        err = sim.compute_sync_error(state=state)
        # sqrt((1-4)^2 + (0-0)^2) = 3.0
        assert err == pytest.approx(3.0)

    def test_phase_difference_zero_in_phase(self):
        """Phase diff should be ~0 for in-phase states."""
        config = _make_config(v1_0=1.0, w1_0=0.5, v2_0=1.0, w2_0=0.5)
        sim = FHNCoupledPairSimulation(config)
        sim.reset()
        dphi = sim.compute_phase_difference()
        assert dphi < 0.01 or dphi > (2 * np.pi - 0.01)

    def test_phase_difference_detection(self):
        """Phase difference should be nonzero for out-of-phase states."""
        config = _make_config(v1_0=1.0, w1_0=0.0, v2_0=-1.0, w2_0=0.0)
        sim = FHNCoupledPairSimulation(config)
        sim.reset()
        dphi = sim.compute_phase_difference()
        # Should be near pi for opposite-sign voltages
        assert abs(dphi - np.pi) < 0.01

    def test_sync_error_trajectory_shape(self):
        """sync_error_trajectory should return correct-shaped arrays."""
        sim = FHNCoupledPairSimulation(_make_config())
        sim.reset()
        result = sim.sync_error_trajectory(n_steps=100)
        assert result["time"].shape == (101,)
        assert result["error"].shape == (101,)
        assert result["phase_diff"].shape == (101,)
        assert result["time"][0] == 0.0

    def test_sync_error_trajectory_positive(self):
        """All sync errors in trajectory should be non-negative."""
        sim = FHNCoupledPairSimulation(_make_config())
        sim.reset()
        result = sim.sync_error_trajectory(n_steps=200)
        assert np.all(result["error"] >= 0)

    def test_is_synchronized_true(self):
        """is_synchronized should return True for identical states."""
        config = _make_config(v1_0=1.0, w1_0=0.5, v2_0=1.0, w2_0=0.5)
        sim = FHNCoupledPairSimulation(config)
        sim.reset()
        assert sim.is_synchronized(threshold=0.01)

    def test_is_synchronized_false(self):
        """is_synchronized should return False for different states."""
        config = _make_config(v1_0=-1.0, w1_0=-0.5, v2_0=1.0, w2_0=0.5)
        sim = FHNCoupledPairSimulation(config)
        sim.reset()
        assert not sim.is_synchronized(threshold=0.1)

    def test_is_in_phase_detection(self):
        """is_in_phase should correctly detect in-phase state."""
        config = _make_config(v1_0=1.0, w1_0=0.5, v2_0=1.0, w2_0=0.5)
        sim = FHNCoupledPairSimulation(config)
        sim.reset()
        assert sim.is_in_phase(threshold=0.1)
        assert not sim.is_anti_phase(threshold=0.1)

    def test_is_anti_phase_detection(self):
        """is_anti_phase should correctly detect anti-phase state."""
        config = _make_config(v1_0=1.0, w1_0=0.0, v2_0=-1.0, w2_0=0.0)
        sim = FHNCoupledPairSimulation(config)
        sim.reset()
        assert sim.is_anti_phase(threshold=0.1)
        assert not sim.is_in_phase(threshold=0.1)


class TestFHNCoupledPairSweep:
    """Tests for the coupling sweep and measurement methods."""

    def test_coupling_sweep_shape(self):
        """coupling_sweep should return arrays of correct length."""
        sim = FHNCoupledPairSimulation(_make_config())
        sim.reset()
        g_vals = np.array([0.0, 0.5, 1.0])
        result = sim.coupling_sweep(g_vals, n_steps=500, n_transient=300)
        assert len(result["g"]) == 3
        assert len(result["mean_error"]) == 3
        assert len(result["mean_phase_diff"]) == 3

    def test_coupling_sweep_errors_positive(self):
        """All sweep errors should be non-negative."""
        sim = FHNCoupledPairSimulation(_make_config())
        sim.reset()
        g_vals = np.array([0.0, 0.1, 0.5, 1.0])
        result = sim.coupling_sweep(g_vals, n_steps=500, n_transient=300)
        assert np.all(result["mean_error"] >= 0)

    def test_firing_frequency_positive(self):
        """Firing frequency should be positive for oscillatory regime."""
        sim = FHNCoupledPairSimulation(_make_config(I1=0.5, I2=0.5))
        sim.reset()
        freq = sim.measure_firing_frequency(neuron=1, n_spikes=3)
        assert freq > 0, f"Expected positive freq, got {freq}"

    def test_firing_frequency_both_neurons(self):
        """Both neurons should have measurable firing frequency."""
        sim = FHNCoupledPairSimulation(_make_config(I1=0.5, I2=0.5, g=0.1))
        sim.reset()
        freq1 = sim.measure_firing_frequency(neuron=1, n_spikes=3)
        # Reset to measure neuron 2 independently
        sim.reset()
        freq2 = sim.measure_firing_frequency(neuron=2, n_spikes=3)
        assert freq1 > 0
        assert freq2 > 0


class TestFHNCoupledPairRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.fhn_coupled_pair import (
            generate_ode_data,
        )
        data = generate_ode_data(g=0.1, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 4)
        assert len(data["time"]) == 501
        assert data["g"] == 0.1
        assert data["I1"] == 0.5
        assert data["I2"] == 0.5

    def test_coupling_sweep_data_generation(self):
        from simulating_anything.rediscovery.fhn_coupled_pair import (
            generate_coupling_sweep_data,
        )
        data = generate_coupling_sweep_data(
            n_g=5, g_min=0.0, g_max=1.0,
            n_steps=500, n_transient=200, dt=0.05,
        )
        assert len(data["g"]) == 5
        assert len(data["mean_error"]) == 5
        assert data["g"][0] == pytest.approx(0.0)
        assert data["g"][-1] == pytest.approx(1.0)

    def test_heterogeneity_data_generation(self):
        from simulating_anything.rediscovery.fhn_coupled_pair import (
            generate_heterogeneity_data,
        )
        data = generate_heterogeneity_data(
            g=0.2, I1=0.5,
            n_I2=5, I2_min=0.3, I2_max=0.7,
            n_steps=500, n_transient=200, dt=0.05,
        )
        assert len(data["I2"]) == 5
        assert len(data["delta_I"]) == 5
        assert len(data["mean_error"]) == 5
        assert data["I2"][0] == pytest.approx(0.3)
        assert data["I2"][-1] == pytest.approx(0.7)
        # delta_I should be I2 - 0.5
        np.testing.assert_allclose(data["delta_I"], data["I2"] - 0.5)

    def test_critical_coupling_detection(self):
        from simulating_anything.rediscovery.fhn_coupled_pair import (
            _detect_critical_coupling,
        )
        g_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        # Simulated error that drops from 1.0 to near 0
        mean_errors = np.array([1.0, 0.5, 0.2, 0.05, 0.01])
        g_c = _detect_critical_coupling(g_values, mean_errors, threshold_fraction=0.1)
        assert np.isfinite(g_c)
        # Should be between 0.5 and 0.75 where error drops below 0.1
        assert 0.5 < g_c < 0.75

    def test_run_function_exists(self):
        """Ensure the main rediscovery runner function is importable."""
        from simulating_anything.rediscovery.fhn_coupled_pair import (
            run_fhn_coupled_pair_rediscovery,
        )
        assert callable(run_fhn_coupled_pair_rediscovery)
