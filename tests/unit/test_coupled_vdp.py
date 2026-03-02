"""Tests for the coupled Van der Pol oscillators simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.coupled_vdp import CoupledVdPSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    mu: float = 1.0,
    k: float = 0.5,
    dt: float = 0.005,
    n_steps: int = 10000,
    **overrides,
) -> SimulationConfig:
    params = {
        "mu": mu, "k": k,
        "x1_0": 0.1, "y1_0": 0.0,
        "x2_0": -0.1, "y2_0": 0.0,
    }
    params.update(overrides)
    return SimulationConfig(
        domain=Domain.COUPLED_VDP,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestCoupledVdPBasics:
    """Basic simulation tests: shape, determinism, stability."""

    def test_observe_shape(self):
        """State should be a 4-element vector [x1, y1, x2, y2]."""
        sim = CoupledVdPSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (4,)
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_initial_conditions(self):
        """Reset should set the specified initial conditions."""
        sim = CoupledVdPSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state, [0.1, 0.0, -0.1, 0.0])

    def test_custom_initial_conditions(self):
        """Custom ICs should be respected."""
        config = _make_config(x1_0=1.5, y1_0=0.2, x2_0=-1.5, y2_0=-0.2)
        sim = CoupledVdPSimulation(config)
        state = sim.reset()
        np.testing.assert_allclose(state, [1.5, 0.2, -1.5, -0.2])

    def test_step_advances(self):
        """State should change after one step."""
        sim = CoupledVdPSimulation(_make_config())
        s0 = sim.reset().copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1)

    def test_deterministic(self):
        """Same config should produce identical trajectories."""
        config = _make_config()
        sim1 = CoupledVdPSimulation(config)
        sim1.reset()
        for _ in range(200):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = CoupledVdPSimulation(config)
        sim2.reset()
        for _ in range(200):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_array_equal(state1, state2)

    def test_rk4_stability(self):
        """No NaN or Inf should appear during integration."""
        sim = CoupledVdPSimulation(_make_config())
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State became NaN/Inf: {state}"

    def test_trajectory_bounded(self):
        """Both oscillators should remain bounded on their limit cycles."""
        sim = CoupledVdPSimulation(_make_config(k=0.5))
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            # VdP limit cycle amplitude is ~2, with coupling should stay bounded
            assert np.all(np.abs(state) < 20), (
                f"State diverged: {state}"
            )

    def test_run_returns_trajectory(self):
        """run() should return TrajectoryData with correct shape."""
        config = _make_config(n_steps=100)
        sim = CoupledVdPSimulation(config)
        traj = sim.run(100)
        assert traj.states.shape == (101, 4)
        assert traj.timestamps.shape == (101,)

    def test_default_parameters(self):
        """Default parameters should match mu=1, k=0.5."""
        config = SimulationConfig(
            domain=Domain.COUPLED_VDP,
            dt=0.005,
            n_steps=100,
            parameters={},
        )
        sim = CoupledVdPSimulation(config)
        assert sim.mu == 1.0
        assert sim.k == 0.5


class TestCoupledVdPDynamics:
    """Tests for core physical behavior: limit cycles and synchronization."""

    def test_limit_cycle_exists(self):
        """Each oscillator should settle onto a limit cycle."""
        sim = CoupledVdPSimulation(_make_config(k=0.0))
        sim.reset()
        # Run through transient
        for _ in range(20000):
            sim.step()
        # Record peaks of x1
        x1_peaks = []
        prev_y1 = sim._state[1]
        for _ in range(10000):
            sim.step()
            y1 = sim._state[1]
            # Peak when velocity crosses zero downward
            if prev_y1 > 0 and y1 <= 0:
                x1_peaks.append(abs(sim._state[0]))
            prev_y1 = y1
        assert len(x1_peaks) >= 2, "No peaks detected for oscillator 1"
        # Peaks should be consistent (limit cycle)
        assert np.std(x1_peaks) < 0.1, (
            f"Peak amplitude not stable: std={np.std(x1_peaks):.4f}"
        )

    def test_limit_cycle_amplitude_near_two(self):
        """Single VdP limit cycle amplitude should be close to 2."""
        config = _make_config(k=0.0, x1_0=0.1, y1_0=0.0)
        sim = CoupledVdPSimulation(config)
        sim.reset()
        # Skip transient
        for _ in range(20000):
            sim.step()
        x1_max = 0.0
        for _ in range(10000):
            sim.step()
            x1_max = max(x1_max, abs(sim._state[0]))
        # VdP limit cycle amplitude is 2.0 for any mu > 0
        assert abs(x1_max - 2.0) < 0.2, (
            f"Amplitude {x1_max:.4f} not close to 2.0"
        )

    def test_zero_coupling_independent(self):
        """With k=0, oscillators evolve independently and can have large error."""
        config = _make_config(
            k=0.0,
            x1_0=2.0, y1_0=0.0,
            x2_0=-2.0, y2_0=0.0,
        )
        sim = CoupledVdPSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        # Both should be on limit cycles but potentially out of phase
        errors = []
        for _ in range(5000):
            sim.step()
            errors.append(sim.compute_sync_error())
        # Time-averaged error should be nonzero for anti-phase ICs
        mean_err = np.mean(errors)
        # Independent oscillators in anti-phase have avg error >> 0
        assert mean_err > 0.1, (
            f"Mean error {mean_err:.4f} unexpectedly small for k=0"
        )

    def test_strong_coupling_reduces_error(self):
        """Strong coupling with in-phase ICs should keep sync error low.

        With in-phase initial conditions and nonzero coupling, the oscillators
        stay synchronized. At k=0 with a small perturbation, they can drift
        apart over time.
        """
        errors_by_k = {}
        for k_val in [0.0, 2.0]:
            config = _make_config(
                k=k_val,
                x1_0=2.0, y1_0=0.0,
                x2_0=1.9, y2_0=0.05,
            )
            sim = CoupledVdPSimulation(config)
            sim.reset()
            for _ in range(15000):
                sim.step()
            err_sum = 0.0
            n_meas = 5000
            for _ in range(n_meas):
                sim.step()
                err_sum += sim.compute_sync_error()
            errors_by_k[k_val] = err_sum / n_meas

        assert errors_by_k[2.0] < errors_by_k[0.0], (
            f"Error at k=2 ({errors_by_k[2.0]:.4f}) not less than "
            f"at k=0 ({errors_by_k[0.0]:.4f})"
        )

    def test_in_phase_sync_preserved(self):
        """In-phase ICs with coupling should stay in phase."""
        config = _make_config(
            k=1.0,
            x1_0=2.0, y1_0=0.0,
            x2_0=2.0, y2_0=0.0,
        )
        sim = CoupledVdPSimulation(config)
        sim.reset()
        for _ in range(15000):
            sim.step()
        # Should remain nearly synchronized
        err = sim.compute_sync_error()
        assert err < 0.1, (
            f"In-phase sync error {err:.6f} too large"
        )

    def test_anti_phase_at_zero_coupling(self):
        """With k=0 and anti-phase ICs, phase difference should remain near pi."""
        config = _make_config(
            k=0.0,
            x1_0=2.0, y1_0=0.0,
            x2_0=-2.0, y2_0=0.0,
        )
        sim = CoupledVdPSimulation(config)
        sim.reset()
        # Skip transient
        for _ in range(10000):
            sim.step()
        # Measure phase differences
        phases = []
        for _ in range(5000):
            sim.step()
            phases.append(sim.compute_phase_difference())
        # Should cluster near pi for anti-phase
        mean_phase = np.mean(phases)
        assert abs(mean_phase - np.pi) < 1.0, (
            f"Mean phase diff {mean_phase:.4f} not near pi for anti-phase ICs"
        )

    def test_coupling_effect_on_dynamics(self):
        """Coupling should affect the dynamics -- states differ from k=0."""
        config0 = _make_config(k=0.0)
        sim0 = CoupledVdPSimulation(config0)
        sim0.reset()
        for _ in range(500):
            sim0.step()
        state0 = sim0.observe().copy()

        config1 = _make_config(k=1.0)
        sim1 = CoupledVdPSimulation(config1)
        sim1.reset()
        for _ in range(500):
            sim1.step()
        state1 = sim1.observe().copy()

        assert not np.allclose(state0, state1), (
            "Coupling has no effect on dynamics"
        )


class TestCoupledVdPDerivatives:
    """Tests for the derivative computation."""

    def test_derivatives_decoupled(self):
        """With k=0, each oscillator follows standard VdP dynamics."""
        sim = CoupledVdPSimulation(_make_config(k=0.0, mu=1.0))
        sim.reset()
        state = np.array([1.0, 0.5, 2.0, -0.3])
        derivs = sim._derivatives(state)
        # dx1/dt = y1 = 0.5
        assert np.isclose(derivs[0], 0.5)
        # dy1/dt = mu*(1-x1^2)*y1 - x1 = 1*(1-1)*0.5 - 1 = -1
        assert np.isclose(derivs[1], -1.0)
        # dx2/dt = y2 = -0.3
        assert np.isclose(derivs[2], -0.3)
        # dy2/dt = mu*(1-x2^2)*y2 - x2 = 1*(1-4)*(-0.3) - 2 = 0.9 - 2 = -1.1
        assert np.isclose(derivs[3], -1.1)

    def test_coupling_term_appears(self):
        """Coupling k*(x2-x1) should modify dy1 and k*(x1-x2) should modify dy2."""
        sim0 = CoupledVdPSimulation(_make_config(k=0.0, mu=1.0))
        sim0.reset()
        sim2 = CoupledVdPSimulation(_make_config(k=2.0, mu=1.0))
        sim2.reset()

        state = np.array([1.0, 0.5, 3.0, -0.3])
        d0 = sim0._derivatives(state)
        d2 = sim2._derivatives(state)

        # dx1/dt and dx2/dt should be identical (coupling is in dy)
        assert np.isclose(d0[0], d2[0])
        assert np.isclose(d0[2], d2[2])

        # dy1 changes by k*(x2-x1) = 2*(3-1) = 4
        assert np.isclose(d2[1] - d0[1], 4.0)
        # dy2 changes by k*(x1-x2) = 2*(1-3) = -4
        assert np.isclose(d2[3] - d0[3], -4.0)

    def test_derivatives_at_sync_manifold(self):
        """On x1=x2, y1=y2, the coupling terms vanish."""
        sim = CoupledVdPSimulation(_make_config(k=5.0, mu=1.0))
        sim.reset()
        state = np.array([1.5, 0.3, 1.5, 0.3])
        derivs = sim._derivatives(state)
        # On the sync manifold, derivatives of osc 1 and 2 should be equal
        np.testing.assert_allclose(derivs[:2], derivs[2:])

    def test_coupling_antisymmetric(self):
        """Coupling is antisymmetric: k*(x2-x1) = -k*(x1-x2)."""
        sim = CoupledVdPSimulation(_make_config(k=1.0, mu=1.0))
        sim.reset()
        state = np.array([1.0, 0.0, 3.0, 0.0])
        derivs = sim._derivatives(state)
        # dy1 coupling contribution: k*(x2-x1) = 1*(3-1) = +2
        # dy2 coupling contribution: k*(x1-x2) = 1*(1-3) = -2
        # Without coupling: dy1 = mu*(1-1)*0 - 1 = -1
        #                    dy2 = mu*(1-9)*0 - 3 = -3
        assert np.isclose(derivs[1], -1.0 + 2.0)  # -1 + k*(x2-x1)
        assert np.isclose(derivs[3], -3.0 - 2.0)  # -3 + k*(x1-x2)


class TestSyncErrorAndPhase:
    """Tests for sync error and phase difference computations."""

    def test_sync_error_zero_on_manifold(self):
        """Sync error should be 0 when both oscillators are identical."""
        config = _make_config(x1_0=1.0, y1_0=0.5, x2_0=1.0, y2_0=0.5)
        sim = CoupledVdPSimulation(config)
        sim.reset()
        assert sim.compute_sync_error() == pytest.approx(0.0, abs=1e-15)

    def test_sync_error_positive_when_different(self):
        """Sync error should be positive when oscillators differ."""
        sim = CoupledVdPSimulation(_make_config())
        sim.reset()
        err = sim.compute_sync_error()
        assert err > 0

    def test_sync_error_with_explicit_state(self):
        """compute_sync_error should accept an explicit state argument."""
        sim = CoupledVdPSimulation(_make_config())
        sim.reset()
        state = np.array([1.0, 0.0, 4.0, 0.0])
        err = sim.compute_sync_error(state=state)
        # sqrt((1-4)^2 + (0-0)^2) = 3.0
        assert err == pytest.approx(3.0)

    def test_phase_difference_zero_in_phase(self):
        """Phase diff should be ~0 for in-phase states."""
        config = _make_config(x1_0=1.0, y1_0=0.5, x2_0=1.0, y2_0=0.5)
        sim = CoupledVdPSimulation(config)
        sim.reset()
        dphi = sim.compute_phase_difference()
        assert dphi < 0.01 or dphi > (2 * np.pi - 0.01)

    def test_phase_difference_pi_anti_phase(self):
        """Phase diff should be ~pi for anti-phase states."""
        config = _make_config(x1_0=1.0, y1_0=0.0, x2_0=-1.0, y2_0=0.0)
        sim = CoupledVdPSimulation(config)
        sim.reset()
        dphi = sim.compute_phase_difference()
        assert abs(dphi - np.pi) < 0.01

    def test_sync_error_trajectory_shape(self):
        """sync_error_trajectory should return correct-shaped arrays."""
        sim = CoupledVdPSimulation(_make_config())
        sim.reset()
        result = sim.sync_error_trajectory(n_steps=100)
        assert result["time"].shape == (101,)
        assert result["error"].shape == (101,)
        assert result["phase_diff"].shape == (101,)
        assert result["time"][0] == 0.0

    def test_sync_error_trajectory_positive(self):
        """All sync errors in trajectory should be non-negative."""
        sim = CoupledVdPSimulation(_make_config())
        sim.reset()
        result = sim.sync_error_trajectory(n_steps=200)
        assert np.all(result["error"] >= 0)


class TestCoupledVdPEnergy:
    """Tests for energy computation and helper properties."""

    def test_energy_positive(self):
        """Energy should be non-negative."""
        sim = CoupledVdPSimulation(_make_config())
        sim.reset()
        E = sim.compute_energy()
        assert E >= 0

    def test_energy_formula(self):
        """Energy should match expected formula."""
        sim = CoupledVdPSimulation(_make_config(k=2.0))
        sim.reset()
        state = np.array([1.0, 2.0, 3.0, 4.0])
        sim._state = state
        E = sim.compute_energy()
        # KE = 0.5*(4 + 16) = 10
        # PE_indiv = 0.5*(1 + 9) = 5
        # PE_coupling = 0.5*2*(1-3)^2 = 4
        expected = 10.0 + 5.0 + 4.0
        assert E == pytest.approx(expected)

    def test_energy_with_explicit_state(self):
        """compute_energy should accept explicit state argument."""
        sim = CoupledVdPSimulation(_make_config(k=1.0))
        sim.reset()
        state = np.array([0.0, 1.0, 0.0, 1.0])
        E = sim.compute_energy(state=state)
        # KE = 0.5*(1+1) = 1, PE_indiv = 0, PE_coupling = 0
        assert E == pytest.approx(1.0)

    def test_is_in_phase_detection(self):
        """is_in_phase should correctly detect in-phase state."""
        config = _make_config(x1_0=1.0, y1_0=0.5, x2_0=1.0, y2_0=0.5)
        sim = CoupledVdPSimulation(config)
        sim.reset()
        assert sim.is_in_phase(threshold=0.1)
        assert not sim.is_anti_phase(threshold=0.1)

    def test_is_anti_phase_detection(self):
        """is_anti_phase should correctly detect anti-phase state."""
        config = _make_config(x1_0=1.0, y1_0=0.0, x2_0=-1.0, y2_0=0.0)
        sim = CoupledVdPSimulation(config)
        sim.reset()
        assert sim.is_anti_phase(threshold=0.1)
        assert not sim.is_in_phase(threshold=0.1)


class TestCoupledVdPSweep:
    """Tests for the coupling sweep and measurement methods."""

    def test_coupling_sweep_shape(self):
        """coupling_sweep should return arrays of correct length."""
        sim = CoupledVdPSimulation(_make_config())
        sim.reset()
        k_vals = np.array([0.0, 1.0, 2.0])
        result = sim.coupling_sweep(k_vals, n_steps=500, n_transient=300)
        assert len(result["k"]) == 3
        assert len(result["mean_error"]) == 3
        assert len(result["mean_phase_diff"]) == 3

    def test_coupling_sweep_errors_positive(self):
        """All sweep errors should be non-negative."""
        sim = CoupledVdPSimulation(_make_config())
        sim.reset()
        k_vals = np.array([0.0, 0.5, 1.0, 2.0])
        result = sim.coupling_sweep(k_vals, n_steps=500, n_transient=300)
        assert np.all(result["mean_error"] >= 0)

    def test_measure_period_finite(self):
        """Oscillation period should be finite and positive."""
        config = _make_config(k=0.0, x1_0=0.1, y1_0=0.0)
        sim = CoupledVdPSimulation(config)
        sim.reset()
        T = sim.measure_period(n_periods=3)
        assert np.isfinite(T)
        assert T > 0
        # Period should be near 2*pi for mu=1 (slightly larger)
        assert 4.0 < T < 10.0, f"Period {T:.4f} outside expected range"

    def test_measure_amplitudes(self):
        """Both amplitude measurements should be near 2 for uncoupled VdP."""
        config = _make_config(k=0.0, x1_0=0.1, y1_0=0.0, x2_0=-0.1, y2_0=0.0)
        sim = CoupledVdPSimulation(config)
        sim.reset()
        a1, a2 = sim.measure_amplitudes(n_periods=3)
        assert abs(a1 - 2.0) < 0.3, f"Amplitude 1 = {a1:.4f}, expected ~2.0"
        assert abs(a2 - 2.0) < 0.3, f"Amplitude 2 = {a2:.4f}, expected ~2.0"


class TestCoupledVdPRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.coupled_vdp import generate_ode_data
        data = generate_ode_data(mu=1.0, k=0.5, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 4)
        assert len(data["time"]) == 501
        assert data["mu"] == 1.0
        assert data["k"] == 0.5

    def test_coupling_sweep_data_generation(self):
        from simulating_anything.rediscovery.coupled_vdp import (
            generate_coupling_sweep_data,
        )
        data = generate_coupling_sweep_data(
            n_k=5, k_min=0.0, k_max=2.0,
            n_steps=500, n_transient=200, dt=0.01,
        )
        assert len(data["k"]) == 5
        assert len(data["mean_error"]) == 5
        assert data["k"][0] == pytest.approx(0.0)
        assert data["k"][-1] == pytest.approx(2.0)

    def test_mode_selection_data(self):
        from simulating_anything.rediscovery.coupled_vdp import (
            generate_mode_selection_data,
        )
        data = generate_mode_selection_data(
            k=0.5, mu=1.0, n_steps=5000, dt=0.01,
        )
        assert "in_phase_ic" in data
        assert "anti_phase_ic" in data
        assert "final_sync_error" in data["in_phase_ic"]
        assert "is_in_phase" in data["in_phase_ic"]

    def test_run_function_exists(self):
        """Ensure the main rediscovery runner function is importable."""
        from simulating_anything.rediscovery.coupled_vdp import (
            run_coupled_vdp_rediscovery,
        )
        assert callable(run_coupled_vdp_rediscovery)
