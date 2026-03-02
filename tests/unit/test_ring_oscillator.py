"""Tests for the ring oscillator simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.ring_oscillator import RingOscillatorSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    gain: float = 5.0,
    dt: float = 0.01,
    n_steps: int = 1000,
    x1_0: float = 0.01,
    x2_0: float = 0.0,
    x3_0: float = 0.0,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.RING_OSCILLATOR,  # placeholder
        dt=dt,
        n_steps=n_steps,
        parameters={
            "gain": gain,
            "x1_0": x1_0,
            "x2_0": x2_0,
            "x3_0": x3_0,
        },
    )


class TestRingOscillatorConstruction:
    def test_default_parameters(self):
        sim = RingOscillatorSimulation(_make_config())
        assert sim.gain == 5.0
        assert sim.x1_0 == 0.01
        assert sim.x2_0 == 0.0
        assert sim.x3_0 == 0.0

    def test_custom_gain(self):
        sim = RingOscillatorSimulation(_make_config(gain=3.0))
        assert sim.gain == 3.0

    def test_custom_initial_conditions(self):
        sim = RingOscillatorSimulation(
            _make_config(x1_0=0.5, x2_0=-0.3, x3_0=0.1)
        )
        assert sim.x1_0 == 0.5
        assert sim.x2_0 == -0.3
        assert sim.x3_0 == 0.1

    def test_initial_state_shape(self):
        sim = RingOscillatorSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (3,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        sim = RingOscillatorSimulation(
            _make_config(x1_0=0.5, x2_0=-0.2, x3_0=0.3)
        )
        state = sim.reset()
        np.testing.assert_allclose(state, [0.5, -0.2, 0.3])


class TestRingOscillatorProperties:
    def test_is_oscillatory_high_gain(self):
        sim = RingOscillatorSimulation(_make_config(gain=5.0))
        assert sim.is_oscillatory is True

    def test_is_oscillatory_low_gain(self):
        sim = RingOscillatorSimulation(_make_config(gain=0.5))
        assert sim.is_oscillatory is False

    def test_is_oscillatory_at_threshold(self):
        sim = RingOscillatorSimulation(_make_config(gain=1.0))
        assert sim.is_oscillatory is False

    def test_is_oscillatory_just_above(self):
        sim = RingOscillatorSimulation(_make_config(gain=1.01))
        assert sim.is_oscillatory is True


class TestRingOscillatorDynamics:
    def test_step_advances_state(self):
        sim = RingOscillatorSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_3d_state(self):
        sim = RingOscillatorSimulation(_make_config())
        sim.reset()
        sim.step()
        obs = sim.observe()
        assert obs.shape == (3,)

    def test_run_returns_trajectory_data(self):
        sim = RingOscillatorSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert len(traj.timestamps) == 101

    def test_trajectory_timestamps_correct(self):
        dt = 0.01
        sim = RingOscillatorSimulation(_make_config(dt=dt, n_steps=50))
        traj = sim.run(n_steps=50)
        expected_times = np.arange(51) * dt
        np.testing.assert_allclose(traj.timestamps, expected_times, atol=1e-12)

    def test_derivatives_at_origin(self):
        """At origin, all derivatives should be zero (fixed point)."""
        sim = RingOscillatorSimulation(_make_config(gain=5.0))
        y = np.array([0.0, 0.0, 0.0])
        dy = sim._derivatives(y)
        # dx1/dt = -0 + tanh(5*(-0)) = 0
        np.testing.assert_allclose(dy, [0.0, 0.0, 0.0], atol=1e-15)

    def test_derivatives_nonzero_for_perturbation(self):
        """With nonzero state, derivatives should be nonzero."""
        sim = RingOscillatorSimulation(_make_config(gain=5.0))
        y = np.array([0.1, 0.0, 0.0])
        dy = sim._derivatives(y)
        # dx1/dt = -0.1 + tanh(5*0) = -0.1
        # dx2/dt = -0.0 + tanh(5*(-0.1)) = tanh(-0.5) ~ -0.462
        # dx3/dt = -0.0 + tanh(5*0) = 0
        assert abs(dy[0] - (-0.1)) < 1e-10
        assert dy[1] < 0  # tanh of negative is negative
        assert abs(dy[2]) < 1e-10

    def test_derivatives_cyclic_structure(self):
        """Verify the cyclic coupling: x1->x2->x3->x1."""
        sim = RingOscillatorSimulation(_make_config(gain=5.0))
        # State where only x1 is nonzero
        y = np.array([0.5, 0.0, 0.0])
        dy = sim._derivatives(y)

        # dx2 depends on x1, dx3 depends on x2 (=0), dx1 depends on x3 (=0)
        assert abs(dy[0] - (-0.5)) < 1e-10  # -x1 + tanh(g*(-x3)) = -0.5 + 0
        assert abs(dy[1] - np.tanh(5.0 * (-0.5))) < 1e-10  # tanh(g*(-x1))
        assert abs(dy[2]) < 1e-10  # -x2 + tanh(g*(-x2)) = 0


class TestRingOscillatorOscillation:
    def test_oscillation_for_high_gain(self):
        """With gain > 1, the system should oscillate."""
        sim = RingOscillatorSimulation(_make_config(gain=5.0, dt=0.01))
        sim.reset()

        # Run for a while and check x1 crosses zero
        has_positive = False
        has_negative = False
        for _ in range(5000):
            sim.step()
            x1 = sim._state[0]
            if x1 > 0.1:
                has_positive = True
            if x1 < -0.1:
                has_negative = True

        assert has_positive and has_negative, "x1 should oscillate"

    def test_no_oscillation_for_low_gain(self):
        """With gain < 1, the system should converge to origin."""
        sim = RingOscillatorSimulation(
            _make_config(gain=0.5, dt=0.01, x1_0=0.5)
        )
        sim.reset()

        for _ in range(5000):
            sim.step()

        # All state variables should be near zero
        state = sim.observe()
        assert np.max(np.abs(state)) < 0.01, (
            f"State should decay to origin: {state}"
        )

    def test_trajectory_bounded(self):
        """All state variables should remain bounded (|x_i| < 1.5)."""
        sim = RingOscillatorSimulation(_make_config(gain=10.0, dt=0.005))
        sim.reset()

        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.max(np.abs(state)) < 1.5, (
                f"State diverged: {state}"
            )

    def test_amplitude_bounded_by_one(self):
        """For any gain, amplitude should be bounded by ~ tanh(g) < 1."""
        for g in [2.0, 5.0, 10.0, 20.0]:
            sim = RingOscillatorSimulation(_make_config(gain=g, dt=0.005))
            sim.reset()

            # Skip transient
            for _ in range(5000):
                sim.step()

            # Check amplitude
            x_max = 0.0
            for _ in range(3000):
                sim.step()
                x_max = max(x_max, np.max(np.abs(sim._state)))

            assert x_max < 1.1, f"Amplitude {x_max} too large for gain={g}"

    def test_amplitude_increases_with_gain(self):
        """Higher gain should yield larger oscillation amplitude."""
        gains = [2.0, 3.0, 5.0, 8.0]
        amps = []
        for g in gains:
            sim = RingOscillatorSimulation(_make_config(gain=g, dt=0.01))
            sim.reset()
            amp = sim.measure_amplitude(n_transient=2000, n_measure=2000)
            amps.append(amp)

        for i in range(len(amps) - 1):
            assert amps[i + 1] >= amps[i] - 0.01, (
                f"Amplitude not monotonic: gains={gains}, amps={amps}"
            )


class TestRingOscillatorPeriod:
    def test_period_finite_for_oscillating(self):
        sim = RingOscillatorSimulation(_make_config(gain=5.0, dt=0.01))
        sim.reset()
        period = sim.compute_period(n_transient=2000, n_measure=4000)
        assert np.isfinite(period), "Period should be finite for oscillating system"

    def test_period_infinite_for_non_oscillating(self):
        # Start from zero IC so there is no perturbation to cross zero
        sim = RingOscillatorSimulation(
            _make_config(gain=0.5, dt=0.01, x1_0=0.0, x2_0=0.0, x3_0=0.0)
        )
        sim.reset()
        period = sim.compute_period(n_transient=2000, n_measure=3000)
        assert period == float("inf"), "Period should be inf for non-oscillating"

    def test_period_approximately_six_tau(self):
        """Period should be approximately 6*tau where tau=1."""
        sim = RingOscillatorSimulation(_make_config(gain=5.0, dt=0.005))
        sim.reset()
        period = sim.compute_period(n_transient=3000, n_measure=5000)
        # Theoretical: T ~ 6*tau = 6.0, but exact value depends on gain
        # For moderate to high gain, period should be in the range 4-8
        assert 3.0 < period < 10.0, (
            f"Period {period} outside expected range [3, 10]"
        )

    def test_period_consistent_across_runs(self):
        """Period should be reproducible."""
        periods = []
        for _ in range(3):
            sim = RingOscillatorSimulation(
                _make_config(gain=5.0, dt=0.005, x1_0=0.01)
            )
            sim.reset()
            p = sim.compute_period(n_transient=3000, n_measure=5000)
            periods.append(p)

        assert np.std(periods) < 0.1, (
            f"Period not reproducible: {periods}"
        )


class TestRingOscillatorPhaseSymmetry:
    def test_three_phase_shifts(self):
        """Phase shifts between stages should be near 120 or 240 degrees.

        The ring oscillator can rotate in either direction, so the phase
        shift is either ~120 or ~240 (= 360 - 120), both reflecting Z3 symmetry.
        """
        sim = RingOscillatorSimulation(_make_config(gain=5.0, dt=0.005))
        sim.reset()
        shift_12, shift_23 = sim.compute_phase_shifts(
            n_transient=3000, n_measure=7000
        )
        # Accept either 120 or 240 (both valid for Z3 cyclic symmetry)
        shift_12_min = min(abs(shift_12 - 120.0), abs(shift_12 - 240.0))
        shift_23_min = min(abs(shift_23 - 120.0), abs(shift_23 - 240.0))
        assert shift_12_min < 30.0, (
            f"Shift 1->2 = {shift_12} deg, expected ~120 or ~240"
        )
        assert shift_23_min < 30.0, (
            f"Shift 2->3 = {shift_23} deg, expected ~120 or ~240"
        )

    def test_amplitude_symmetry(self):
        """All three stages should have similar amplitudes (Z3 symmetry)."""
        sim = RingOscillatorSimulation(_make_config(gain=5.0, dt=0.005))
        sim.reset()
        a1, a2, a3 = sim.measure_amplitudes_all(
            n_transient=3000, n_measure=5000
        )
        spread = max(a1, a2, a3) - min(a1, a2, a3)
        assert spread < 0.1, (
            f"Amplitudes not symmetric: [{a1:.4f}, {a2:.4f}, {a3:.4f}]"
        )


class TestRingOscillatorGainSweep:
    def test_sweep_returns_correct_keys(self):
        sim = RingOscillatorSimulation(_make_config())
        result = sim.gain_sweep(
            gain_range=(0.5, 5.0), n_gains=5, n_steps=2000
        )
        assert "gain" in result
        assert "amplitude" in result
        assert "period" in result

    def test_sweep_correct_length(self):
        sim = RingOscillatorSimulation(_make_config())
        result = sim.gain_sweep(
            gain_range=(0.5, 5.0), n_gains=8, n_steps=2000
        )
        assert len(result["gain"]) == 8
        assert len(result["amplitude"]) == 8
        assert len(result["period"]) == 8

    def test_sweep_low_gain_no_oscillation(self):
        """Gains below 1 should show near-zero amplitude."""
        sim = RingOscillatorSimulation(_make_config())
        result = sim.gain_sweep(
            gain_range=(0.3, 0.9), n_gains=5, n_steps=3000
        )
        assert np.all(result["amplitude"] < 0.1), (
            f"Sub-threshold gains should not oscillate: {result['amplitude']}"
        )

    def test_sweep_high_gain_oscillates(self):
        """Gains well above 1 should show significant amplitude."""
        sim = RingOscillatorSimulation(_make_config())
        result = sim.gain_sweep(
            gain_range=(3.0, 8.0), n_gains=5, n_steps=3000
        )
        assert np.all(result["amplitude"] > 0.3), (
            f"High-gain should oscillate: {result['amplitude']}"
        )


class TestRingOscillatorReproducibility:
    def test_deterministic_without_seed(self):
        """Same initial conditions produce same trajectory."""
        sim1 = RingOscillatorSimulation(_make_config(x1_0=0.01))
        sim2 = RingOscillatorSimulation(_make_config(x1_0=0.01))
        sim1.reset()
        sim2.reset()

        for _ in range(500):
            sim1.step()
            sim2.step()

        np.testing.assert_allclose(sim1.observe(), sim2.observe())

    def test_reset_with_seed_adds_perturbation(self):
        """Reset with seed adds random perturbation to initial conditions."""
        sim = RingOscillatorSimulation(_make_config(x1_0=0.0, x2_0=0.0, x3_0=0.0))
        state = sim.reset(seed=42)
        # With seed, perturbation is added so state should not be exactly zero
        assert not np.allclose(state, [0.0, 0.0, 0.0])

    def test_different_seeds_give_different_starts(self):
        """Different seeds produce different initial perturbations."""
        sim = RingOscillatorSimulation(_make_config())
        s1 = sim.reset(seed=42)
        s2 = sim.reset(seed=123)
        assert not np.allclose(s1, s2)


class TestRingOscillatorFixedPoint:
    def test_origin_is_fixed_point(self):
        """Origin [0,0,0] is a fixed point for any gain."""
        for g in [0.5, 1.0, 5.0, 10.0]:
            sim = RingOscillatorSimulation(_make_config(gain=g))
            dy = sim._derivatives(np.array([0.0, 0.0, 0.0]))
            np.testing.assert_allclose(
                dy, [0.0, 0.0, 0.0], atol=1e-15,
                err_msg=f"Origin not fixed for gain={g}",
            )

    def test_origin_stable_for_low_gain(self):
        """With gain < 1, trajectories starting near origin converge to it."""
        sim = RingOscillatorSimulation(
            _make_config(gain=0.5, x1_0=0.5, x2_0=-0.3, x3_0=0.2, dt=0.01)
        )
        sim.reset()

        for _ in range(10000):
            sim.step()

        state = sim.observe()
        assert np.max(np.abs(state)) < 0.01, (
            f"Origin should be stable for gain=0.5: {state}"
        )

    def test_origin_unstable_for_high_gain(self):
        """With gain > 1, small perturbation from origin grows."""
        sim = RingOscillatorSimulation(
            _make_config(gain=5.0, x1_0=0.001, dt=0.01)
        )
        sim.reset()

        # Track amplitude growth
        initial_amp = np.max(np.abs(sim.observe()))
        for _ in range(2000):
            sim.step()

        final_amp = np.max(np.abs(sim.observe()))
        assert final_amp > 10 * initial_amp, (
            f"Perturbation should grow: initial={initial_amp}, final={final_amp}"
        )


class TestRingOscillatorRediscovery:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.ring_oscillator import generate_ode_data

        data = generate_ode_data(gain=5.0, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 3)
        assert len(data["time"]) == 501
        assert data["gain"] == 5.0
        assert data["dt"] == 0.01

    def test_ode_data_contains_oscillation(self):
        from simulating_anything.rediscovery.ring_oscillator import generate_ode_data

        data = generate_ode_data(gain=5.0, n_steps=5000, dt=0.005)
        x1 = data["x1"]
        # x1 should oscillate: both positive and negative values
        assert np.any(x1 > 0.1) and np.any(x1 < -0.1)

    def test_gain_sweep_data_generation(self):
        from simulating_anything.rediscovery.ring_oscillator import (
            generate_gain_sweep_data,
        )

        data = generate_gain_sweep_data(n_gains=5, dt=0.01, n_steps=2000)
        assert len(data["gain"]) == 5
        assert len(data["amplitude"]) == 5
        assert len(data["period"]) == 5

    def test_phase_symmetry_data_generation(self):
        from simulating_anything.rediscovery.ring_oscillator import (
            generate_phase_symmetry_data,
        )

        data = generate_phase_symmetry_data(gain=5.0, dt=0.005)
        assert "shift_12_degrees" in data
        assert "shift_23_degrees" in data
        assert "ideal_shift" in data
        assert data["ideal_shift"] == 120.0


class TestRingOscillatorTanhNonlinearity:
    def test_tanh_saturation(self):
        """Verify tanh saturates at +/-1 for large inputs."""
        sim = RingOscillatorSimulation(_make_config(gain=10.0))
        # Large x3 -> tanh(g*(-x3)) -> tanh(-100) ~ -1
        y = np.array([0.0, 0.0, 10.0])
        dy = sim._derivatives(y)
        assert abs(dy[0] - (-1.0)) < 0.01  # -x1 + tanh(-g*x3) ~ 0 + (-1) = -1

    def test_tanh_linearity_for_small_input(self):
        """For small x, tanh(g*x) ~ g*x."""
        sim = RingOscillatorSimulation(_make_config(gain=5.0))
        eps = 0.001
        y = np.array([eps, 0.0, 0.0])
        dy = sim._derivatives(y)
        # dx2/dt = -0 + tanh(g*(-eps)) ~ -g*eps for small eps
        expected_dx2 = -5.0 * eps
        assert abs(dy[1] - expected_dx2) < 1e-6

    def test_inverting_nature(self):
        """Each stage inverts: positive input produces negative output tendency."""
        sim = RingOscillatorSimulation(_make_config(gain=5.0))
        y = np.array([0.5, 0.0, 0.0])
        dy = sim._derivatives(y)
        # x1 > 0 drives dx2 negative (inverting through tanh(g*(-x1)))
        assert dy[1] < 0, "Stage 2 should be driven negative by positive x1"
