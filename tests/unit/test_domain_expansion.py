"""Tests demonstrating domain expansion -- adding a new domain in ~50 lines.

The Duffing oscillator is implemented as an example in simulation/template.py.
These tests prove it integrates seamlessly with the pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.template import DuffingOscillator
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(**kwargs) -> SimulationConfig:
    params = {
        "alpha": 1.0, "beta": 1.0, "delta": 0.2,
        "gamma_f": 0.3, "omega": 1.0,
        "x_0": 0.5, "v_0": 0.0,
    }
    params.update(kwargs)
    return SimulationConfig(
        domain=Domain.RIGID_BODY,  # Reuse existing enum for template
        dt=0.01,
        n_steps=1000,
        parameters=params,
    )


class TestDuffingOscillator:
    """Test the example domain expansion (Duffing oscillator)."""

    def test_reset_returns_initial_state(self):
        sim = DuffingOscillator(_make_config(x_0=1.0, v_0=0.0))
        state = sim.reset()
        assert state.shape == (2,)
        assert state[0] == pytest.approx(1.0)
        assert state[1] == pytest.approx(0.0)

    def test_step_advances_state(self):
        sim = DuffingOscillator(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_matches_step(self):
        sim = DuffingOscillator(_make_config())
        sim.reset()
        step_result = sim.step()
        obs_result = sim.observe()
        np.testing.assert_array_equal(step_result, obs_result)

    def test_unforced_undamped_oscillation(self):
        """With no forcing and no damping, energy should be conserved."""
        sim = DuffingOscillator(_make_config(
            delta=0.0, gamma_f=0.0, x_0=1.0, v_0=0.0
        ))
        sim.reset()
        e0 = sim.total_energy
        for _ in range(10000):
            sim.step()
        e1 = sim.total_energy
        assert abs(e1 - e0) / abs(e0) < 1e-6, f"Energy drift: {abs(e1 - e0) / abs(e0)}"

    def test_damped_amplitude_decreases(self):
        """With damping and no forcing, amplitude should decrease."""
        sim = DuffingOscillator(_make_config(
            delta=0.5, gamma_f=0.0, x_0=2.0, v_0=0.0
        ))
        sim.reset()
        max_early = 0.0
        for _ in range(500):
            sim.step()
            max_early = max(max_early, abs(sim.observe()[0]))

        max_late = 0.0
        for _ in range(500):
            sim.step()
            max_late = max(max_late, abs(sim.observe()[0]))

        assert max_late < max_early

    def test_trajectory_collection(self):
        """run() should produce a TrajectoryData object."""
        sim = DuffingOscillator(_make_config())
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 2)  # 100 steps + initial
        assert len(traj.timestamps) == 101

    def test_bounded_for_small_forcing(self):
        """Solution should remain bounded for small forcing."""
        sim = DuffingOscillator(_make_config(
            gamma_f=0.1, delta=0.3, x_0=0.1
        ))
        sim.reset()
        for _ in range(5000):
            sim.step()
            x = sim.observe()[0]
            assert abs(x) < 100, f"Unbounded: x={x}"

    def test_small_linear_limit(self):
        """With beta=0 (no cubic term), should behave as damped harmonic oscillator."""
        sim = DuffingOscillator(_make_config(
            alpha=4.0, beta=0.0, delta=0.0, gamma_f=0.0,
            x_0=1.0, v_0=0.0,
        ))
        sim.reset()
        # Undamped linear: x(t) = cos(omega_0 * t), omega_0 = sqrt(alpha) = 2
        for _ in range(100):
            sim.step()
        t = 100 * 0.01
        expected_x = np.cos(2.0 * t)
        assert sim.observe()[0] == pytest.approx(expected_x, abs=0.01)

    def test_lines_of_code(self):
        """Verify the implementation is truly compact."""
        import inspect
        source = inspect.getsource(DuffingOscillator)
        n_lines = len([l for l in source.split("\n") if l.strip()])
        assert n_lines < 60, f"Duffing implementation is {n_lines} lines (should be < 60)"

    def test_works_with_pysr_data_prep(self):
        """Verify simulation output is suitable for PySR/SINDy analysis."""
        sim = DuffingOscillator(_make_config(x_0=1.0, gamma_f=0.3))
        traj = sim.run(n_steps=500)

        # PySR needs X (features) and y (target)
        states = traj.states
        assert states.shape[1] == 2  # [x, v]
        assert np.all(np.isfinite(states))

        # SINDy needs continuous trajectory + dt
        dt = 0.01
        assert len(traj.timestamps) == 501
        assert np.allclose(np.diff(traj.timestamps), dt)
