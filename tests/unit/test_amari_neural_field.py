"""Tests for the Amari neural field simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.amari_neural_field import (
    AmariNeuralFieldSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    h: float = -0.5,
    A: float = 5.0,
    B: float = 2.0,
    sigma_e: float = 1.0,
    sigma_i: float = 3.0,
    beta: float = 50.0,
    theta: float = 0.5,
    tau: float = 1.0,
    N: int = 128,
    L_half: float = 10.0,
    dt: float = 0.02,
    n_steps: int = 2000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.AMARI_NEURAL_FIELD,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "tau": tau,
            "A": A,
            "B": B,
            "sigma_e": sigma_e,
            "sigma_i": sigma_i,
            "beta": beta,
            "theta": theta,
            "h": h,
            "N": float(N),
            "L_half": L_half,
        },
    )


class TestAmariCreation:
    def test_create_simulation(self):
        """Simulation can be instantiated with default config."""
        sim = AmariNeuralFieldSimulation(_make_config())
        assert sim is not None
        assert sim.N == 128

    def test_parameters_loaded(self):
        """All parameters are loaded from config."""
        sim = AmariNeuralFieldSimulation(_make_config(
            A=2.0, B=1.0, sigma_e=1.5, h=-0.3,
        ))
        assert sim.A == 2.0
        assert sim.B == 1.0
        assert sim.sigma_e == 1.5
        assert sim.h == -0.3

    def test_grid_setup(self):
        """Spatial grid is correctly configured."""
        sim = AmariNeuralFieldSimulation(_make_config(N=64, L_half=5.0))
        assert sim.N == 64
        assert sim.L == 10.0
        assert len(sim.x) == 64
        assert sim.x[0] == pytest.approx(-5.0)
        assert sim.x[-1] < 5.0  # endpoint=False

    def test_custom_grid_size(self):
        """Different grid sizes work correctly."""
        for N in [32, 64, 128, 256]:
            sim = AmariNeuralFieldSimulation(_make_config(N=N))
            assert sim.N == N
            assert len(sim.x) == N


class TestAmariReset:
    def test_reset_returns_array(self):
        """Reset returns a numpy array."""
        sim = AmariNeuralFieldSimulation(_make_config())
        state = sim.reset()
        assert isinstance(state, np.ndarray)

    def test_reset_state_shape(self):
        """Reset state has shape (N,)."""
        sim = AmariNeuralFieldSimulation(_make_config(N=128))
        state = sim.reset()
        assert state.shape == (128,)

    def test_reset_state_shape_different_N(self):
        """State shape matches grid size N."""
        sim = AmariNeuralFieldSimulation(_make_config(N=64))
        state = sim.reset()
        assert state.shape == (64,)

    def test_reset_initial_condition(self):
        """Initial condition has a localized perturbation near center."""
        sim = AmariNeuralFieldSimulation(_make_config())
        state = sim.reset(seed=42)
        center_idx = sim.N // 2
        # Center should have the peak (Gaussian perturbation on top of h)
        assert state[center_idx] > sim.h + 0.5

    def test_reset_deterministic_with_seed(self):
        """Same seed produces same initial condition."""
        sim = AmariNeuralFieldSimulation(_make_config())
        state1 = sim.reset(seed=123)
        state2 = sim.reset(seed=123)
        np.testing.assert_allclose(state1, state2, atol=1e-12)


class TestAmariStep:
    def test_step_returns_array(self):
        """Step returns a numpy array."""
        sim = AmariNeuralFieldSimulation(_make_config())
        sim.reset()
        state = sim.step()
        assert isinstance(state, np.ndarray)
        assert state.shape == (128,)

    def test_step_changes_state(self):
        """State changes after stepping."""
        sim = AmariNeuralFieldSimulation(_make_config())
        state0 = sim.reset(seed=42)
        state0 = state0.copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_step_count_increments(self):
        """Step counter increments correctly."""
        sim = AmariNeuralFieldSimulation(_make_config())
        sim.reset()
        assert sim._step_count == 0
        sim.step()
        assert sim._step_count == 1
        sim.step()
        assert sim._step_count == 2

    def test_no_nan_after_many_steps(self):
        """No NaN or Inf values after extended simulation."""
        sim = AmariNeuralFieldSimulation(_make_config(dt=0.05))
        sim.reset(seed=42)
        for _ in range(2000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"Non-finite state at step {sim._step_count}"

    def test_deterministic_evolution(self):
        """Same initial conditions produce same trajectory."""
        config = _make_config()
        sim1 = AmariNeuralFieldSimulation(config)
        sim1.reset(seed=42)
        for _ in range(100):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = AmariNeuralFieldSimulation(config)
        sim2.reset(seed=42)
        for _ in range(100):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2, atol=1e-12)


class TestAmariObserve:
    def test_observe_returns_state(self):
        """Observe returns current state."""
        sim = AmariNeuralFieldSimulation(_make_config())
        sim.reset(seed=42)
        sim.step()
        obs = sim.observe()
        assert obs.shape == (128,)
        np.testing.assert_array_equal(obs, sim._state)


class TestAmariRun:
    def test_run_returns_trajectory(self):
        """run() returns TrajectoryData with correct shapes."""
        sim = AmariNeuralFieldSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 128)  # 51 = initial + 50 steps
        assert traj.timestamps.shape == (51,)

    def test_run_default_steps(self):
        """run() uses config.n_steps by default."""
        sim = AmariNeuralFieldSimulation(_make_config(n_steps=30))
        traj = sim.run()
        assert traj.states.shape[0] == 31  # initial + 30 steps


class TestAmariSigmoid:
    def test_sigmoid_at_threshold(self):
        """f(theta) = 0.5 at the threshold."""
        sim = AmariNeuralFieldSimulation(_make_config(theta=0.5))
        assert sim.sigmoid(0.5) == pytest.approx(0.5)

    def test_sigmoid_large_input(self):
        """f(large) approaches 1."""
        sim = AmariNeuralFieldSimulation(_make_config())
        assert sim.sigmoid(100.0) == pytest.approx(1.0, abs=1e-10)

    def test_sigmoid_small_input(self):
        """f(small) approaches 0."""
        sim = AmariNeuralFieldSimulation(_make_config())
        assert sim.sigmoid(-100.0) == pytest.approx(0.0, abs=1e-10)

    def test_sigmoid_monotonic(self):
        """Sigmoid is monotonically increasing."""
        sim = AmariNeuralFieldSimulation(_make_config())
        x = np.linspace(-5, 5, 100)
        fx = sim.sigmoid(x)
        assert np.all(np.diff(fx) >= 0)

    def test_sigmoid_steepness(self):
        """Higher beta gives steeper sigmoid."""
        sim_steep = AmariNeuralFieldSimulation(_make_config(beta=20.0))
        sim_flat = AmariNeuralFieldSimulation(_make_config(beta=2.0))
        # Both equal 0.5 at theta
        assert sim_steep.sigmoid(0.5) == pytest.approx(0.5)
        assert sim_flat.sigmoid(0.5) == pytest.approx(0.5)
        # Steep should be closer to 1 above threshold
        x = 0.8
        assert sim_steep.sigmoid(x) > sim_flat.sigmoid(x)

    def test_sigmoid_array_input(self):
        """Sigmoid works with array inputs."""
        sim = AmariNeuralFieldSimulation(_make_config())
        x = np.array([0.0, 0.5, 1.0])
        fx = sim.sigmoid(x)
        assert fx.shape == (3,)
        assert fx[1] == pytest.approx(0.5)


class TestAmariKernel:
    def test_kernel_shape(self):
        """Kernel has correct shape."""
        sim = AmariNeuralFieldSimulation(_make_config(N=128))
        sim.reset()
        kernel = sim.get_kernel()
        assert kernel.shape == (128,)

    def test_kernel_mexican_hat(self):
        """Kernel is Mexican hat: positive center, negative surround."""
        sim = AmariNeuralFieldSimulation(_make_config())
        sim.reset()
        kernel = sim.get_kernel()

        # The center of the kernel (x=0) is near index N//2 due to the grid
        # layout from -L_half to L_half
        center_idx = sim.N // 2
        # Center should be positive (excitation dominates locally)
        assert kernel[center_idx] > 0, f"Center kernel value: {kernel[center_idx]}"

    def test_kernel_surround_negative(self):
        """Kernel has negative regions at moderate distances."""
        sim = AmariNeuralFieldSimulation(_make_config(
            A=5.0, B=2.0, sigma_e=1.0, sigma_i=3.0,
        ))
        sim.reset()
        kernel = sim.get_kernel()
        # There should be negative values somewhere (surround inhibition)
        assert np.min(kernel) < 0, "Kernel should have negative (inhibitory) values"

    def test_kernel_symmetry(self):
        """Kernel is symmetric around center (x=0)."""
        sim = AmariNeuralFieldSimulation(_make_config())
        sim.reset()
        kernel = sim.get_kernel()
        # On the periodic grid, kernel should be symmetric
        # kernel[i] should approximately equal kernel[N-i] for i > 0
        for i in range(1, sim.N // 4):
            assert kernel[i] == pytest.approx(kernel[-i], abs=1e-10), (
                f"Kernel not symmetric at offset {i}"
            )

    def test_kernel_peak_at_zero(self):
        """Kernel peak is at x=0 (or nearest grid point)."""
        sim = AmariNeuralFieldSimulation(_make_config())
        sim.reset()
        kernel = sim.get_kernel()
        peak_idx = np.argmax(kernel)
        # Peak should be at or very near x=0
        assert abs(sim.x[peak_idx]) < sim.dx + 1e-10

    def test_kernel_scales_with_A_B(self):
        """Kernel amplitude scales with A and B parameters."""
        sim1 = AmariNeuralFieldSimulation(_make_config(A=5.0, B=2.0))
        sim1.reset()
        kernel1 = sim1.get_kernel()

        sim2 = AmariNeuralFieldSimulation(_make_config(A=10.0, B=4.0))
        sim2.reset()
        kernel2 = sim2.get_kernel()

        # Doubling A and B should double the kernel
        np.testing.assert_allclose(kernel2, 2.0 * kernel1, rtol=1e-10)


class TestAmariBumpFormation:
    def test_bump_forms_with_default_params(self):
        """A stable bump forms with default parameters."""
        sim = AmariNeuralFieldSimulation(_make_config(h=-0.5))
        sim.reset(seed=42)
        for _ in range(3000):
            sim.step()

        width = sim.compute_bump_width()
        amplitude = sim.compute_bump_amplitude()
        assert width > 0.5, f"Expected bump, got width={width}"
        assert amplitude > 0.3, f"Expected bump amplitude, got {amplitude}"

    def test_no_bump_very_negative_h(self):
        """No bump forms when h is very negative."""
        sim = AmariNeuralFieldSimulation(_make_config(h=-5.0))
        sim.reset(seed=42)
        for _ in range(3000):
            sim.step()

        width = sim.compute_bump_width()
        assert width < 0.5, f"Should have no bump with h=-5.0, got width={width}"

    def test_bump_localized(self):
        """Bump activity is localized, not space-filling."""
        sim = AmariNeuralFieldSimulation(_make_config(h=-0.5))
        sim.reset(seed=42)
        for _ in range(3000):
            sim.step()

        state = sim.observe()
        above = np.sum(state > sim.theta)
        # Bump should occupy less than half the grid
        assert above < sim.N // 2, f"Bump too wide: {above}/{sim.N} points above threshold"

    def test_bump_stability(self):
        """Established bump remains stable over time."""
        sim = AmariNeuralFieldSimulation(_make_config(h=-0.5))
        sim.reset(seed=42)
        # Form bump
        for _ in range(3000):
            sim.step()

        stable = sim.is_bump_stable(n_check_steps=500)
        assert stable, "Bump should be stable"

    def test_bump_width_method(self):
        """compute_bump_width returns reasonable values."""
        sim = AmariNeuralFieldSimulation(_make_config(h=-0.5))
        sim.reset(seed=42)
        for _ in range(3000):
            sim.step()

        width = sim.compute_bump_width()
        # Width should be positive and less than domain length
        assert 0 < width < sim.L

    def test_bump_amplitude_method(self):
        """compute_bump_amplitude returns peak minus resting level."""
        sim = AmariNeuralFieldSimulation(_make_config(h=-0.5))
        sim.reset(seed=42)
        for _ in range(3000):
            sim.step()

        amp = sim.compute_bump_amplitude()
        # Amplitude should be positive for a bump
        assert amp > 0


class TestAmariSpatialSymmetry:
    def test_symmetric_ic_short_evolution(self):
        """Symmetric initial condition preserves symmetry over short time.

        Tests that the numerical scheme does not introduce large asymmetries
        during a few integration steps. Uses only a few steps to avoid
        the bump collapsing or growing to fill the domain.
        """
        sim = AmariNeuralFieldSimulation(_make_config(N=128))
        # Set a symmetric initial condition (no noise)
        sim._state = sim.h + 2.0 * np.exp(-sim.x ** 2 / 2.0)
        sim._step_count = 0

        state_before = sim._state.copy()
        # Only a few steps to check scheme symmetry (not long-term dynamics)
        for _ in range(5):
            sim.step()

        state = sim.observe()
        # The field should have changed from initial
        assert not np.allclose(state, state_before)
        # Check that field near x=0 has approximately symmetric values
        # index 64 = x=0.0 for N=128
        center_idx = sim.N // 2
        # Values just left and right of center should be similar
        assert abs(state[center_idx - 1] - state[center_idx + 1]) < 0.1, (
            f"Left ({state[center_idx - 1]:.4f}) and right ({state[center_idx + 1]:.4f}) "
            f"of center should be similar"
        )

    def test_bump_activity_near_origin(self):
        """Bump formed from centered IC has activity near x=0."""
        sim = AmariNeuralFieldSimulation(_make_config(h=-0.5))
        sim.reset(seed=42)
        for _ in range(3000):
            sim.step()

        state = sim.observe()
        if sim.compute_bump_width() > 0.5:
            # The center of the domain (x=0, index 64) should be above threshold
            center_idx = sim.N // 2
            assert state[center_idx] > sim.theta, (
                f"State at x=0 should be above threshold, got {state[center_idx]:.4f}"
            )


class TestAmariExternalInput:
    def test_set_external_input(self):
        """External input can be set."""
        sim = AmariNeuralFieldSimulation(_make_config())
        sim.reset()
        I_ext = np.zeros(sim.N)
        I_ext[sim.N // 2] = 1.0
        sim.set_external_input(I_ext)
        np.testing.assert_array_equal(sim._I_ext, I_ext)

    def test_set_localized_input(self):
        """Localized Gaussian input is correctly set."""
        sim = AmariNeuralFieldSimulation(_make_config())
        sim.reset()
        sim.set_localized_input(center=0.0, width=1.0, amplitude=2.0)
        # Peak should be near center
        peak_idx = np.argmax(sim._I_ext)
        assert abs(sim.x[peak_idx]) < sim.dx + 0.1
        assert sim._I_ext[peak_idx] == pytest.approx(2.0, abs=0.1)

    def test_external_input_affects_dynamics(self):
        """External input changes the field evolution."""
        config = _make_config(h=-2.0, n_steps=500)

        # Without input
        sim1 = AmariNeuralFieldSimulation(config)
        sim1.reset(seed=42)
        for _ in range(500):
            sim1.step()
        state1 = sim1.observe().copy()

        # With strong localized input
        sim2 = AmariNeuralFieldSimulation(config)
        sim2.reset(seed=42)
        sim2.set_localized_input(center=0.0, width=1.0, amplitude=5.0)
        for _ in range(500):
            sim2.step()
        state2 = sim2.observe().copy()

        assert not np.allclose(state1, state2), "External input should change dynamics"


class TestAmariTotalActivity:
    def test_total_activity_positive(self):
        """Total firing rate activity is always non-negative."""
        sim = AmariNeuralFieldSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(100):
            sim.step()
        activity = sim.compute_total_activity()
        assert activity >= 0.0

    def test_total_activity_bounded(self):
        """Total activity is bounded by domain length (sigmoid max is 1)."""
        sim = AmariNeuralFieldSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(100):
            sim.step()
        activity = sim.compute_total_activity()
        # Max total activity = integral of 1 over domain = L
        assert activity <= sim.L + 0.1


class TestAmariSweeps:
    def test_bump_existence_sweep(self):
        """Bump existence sweep returns valid data structure."""
        sim = AmariNeuralFieldSimulation(_make_config())
        sim.reset()
        result = sim.bump_existence_sweep(
            h_values=np.array([-2.0, -1.0, -0.5, 0.0]),
            n_steps=500,
        )
        assert "h_values" in result
        assert "bump_widths" in result
        assert "bump_amplitudes" in result
        assert len(result["h_values"]) == 4
        assert len(result["bump_widths"]) == 4

    def test_sigma_ratio_sweep(self):
        """Sigma ratio sweep returns valid data structure."""
        sim = AmariNeuralFieldSimulation(_make_config())
        sim.reset()
        result = sim.sigma_ratio_sweep(
            sigma_e_values=np.array([0.5, 1.0, 1.5, 2.0]),
            n_steps=500,
        )
        assert "sigma_e_values" in result
        assert "sigma_ratios" in result
        assert "bump_widths" in result
        assert len(result["sigma_e_values"]) == 4

    def test_bump_width_increases_with_h(self):
        """Bump width should generally increase as h increases (less inhibited)."""
        sim = AmariNeuralFieldSimulation(_make_config())
        result = sim.bump_existence_sweep(
            h_values=np.array([-2.0, -1.5, -1.0, -0.5, 0.0]),
            n_steps=1000,
        )
        widths = result["bump_widths"]
        # At least the last few h values should have wider bumps
        # than the most negative h values
        assert widths[-1] >= widths[0], (
            f"Width at h=0.0 ({widths[-1]}) should be >= width at h=-2.0 ({widths[0]})"
        )


class TestAmariRediscovery:
    def test_bump_existence_data_generation(self):
        from simulating_anything.rediscovery.amari_neural_field import (
            generate_bump_existence_data,
        )
        data = generate_bump_existence_data(n_h=5, n_steps=200, dt=0.05)
        assert len(data["h_values"]) == 5
        assert len(data["bump_widths"]) == 5
        assert len(data["bump_amplitudes"]) == 5

    def test_sigma_ratio_data_generation(self):
        from simulating_anything.rediscovery.amari_neural_field import (
            generate_sigma_ratio_data,
        )
        data = generate_sigma_ratio_data(n_sigma=5, n_steps=200, dt=0.05)
        assert len(data["sigma_e_values"]) == 5
        assert len(data["sigma_ratios"]) == 5
        assert len(data["bump_widths"]) == 5

    def test_kernel_data_generation(self):
        from simulating_anything.rediscovery.amari_neural_field import (
            generate_kernel_data,
        )
        data = generate_kernel_data()
        assert "x" in data
        assert "kernel" in data
        assert "excitatory" in data
        assert "inhibitory" in data
        assert len(data["x"]) == 128
        # Excitatory should be all positive
        assert np.all(data["excitatory"] > 0)
        # Inhibitory should be all negative
        assert np.all(data["inhibitory"] < 0)
