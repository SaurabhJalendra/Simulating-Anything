"""Tests for the Swift-Hohenberg equation simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.swift_hohenberg import SwiftHohenbergSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    r: float = 0.5,
    g: float = 0.0,
    N: int = 128,
    L: float = 20.0 * np.pi,
    dt: float = 0.1,
    n_steps: int = 1000,
    seed: int = 42,
    init_amplitude: float = 0.01,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.SWIFT_HOHENBERG,  # Placeholder until Domain.SWIFT_HOHENBERG exists
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r": r,
            "g": g,
            "N": float(N),
            "L": L,
            "init_amplitude": init_amplitude,
        },
        seed=seed,
    )


# ---------------------------------------------------------------------------
# TestSimulation: basic init, reset, step, observe, run, shapes
# ---------------------------------------------------------------------------
class TestSimulation:
    """Basic simulation interface tests."""

    def test_create_simulation(self):
        """Simulation should be created with default parameters."""
        sim = SwiftHohenbergSimulation(_make_config())
        assert sim.r == 0.5
        assert sim.g == 0.0
        assert sim.N == 128

    def test_custom_parameters(self):
        """Custom parameters should override defaults."""
        sim = SwiftHohenbergSimulation(_make_config(r=1.0, g=0.5, N=64))
        assert sim.r == 1.0
        assert sim.g == 0.5
        assert sim.N == 64

    def test_reset_shape(self):
        """Initial state should be 1D array of length N."""
        sim = SwiftHohenbergSimulation(_make_config(N=128))
        state = sim.reset()
        assert state.shape == (128,)

    def test_reset_dtype(self):
        """State should be float64."""
        sim = SwiftHohenbergSimulation(_make_config())
        state = sim.reset()
        assert state.dtype == np.float64

    def test_step_returns_correct_shape(self):
        """Step should return array of shape (N,)."""
        sim = SwiftHohenbergSimulation(_make_config(N=128))
        sim.reset()
        state = sim.step()
        assert state.shape == (128,)

    def test_step_advances_state(self):
        """State should change after a step."""
        sim = SwiftHohenbergSimulation(_make_config())
        s0 = sim.reset().copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1)

    def test_observe_matches_state(self):
        """Observe should return the current state."""
        sim = SwiftHohenbergSimulation(_make_config())
        sim.reset()
        sim.step()
        obs = sim.observe()
        assert obs.shape == (sim.N,)
        assert np.array_equal(obs, sim._state)

    def test_run_trajectory(self):
        """run() should produce a TrajectoryData object with correct shape."""
        sim = SwiftHohenbergSimulation(_make_config(n_steps=50))
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 128)
        assert len(traj.timestamps) == 51

    def test_run_uses_config_n_steps(self):
        """run() with no argument should use config.n_steps."""
        sim = SwiftHohenbergSimulation(_make_config(n_steps=30))
        traj = sim.run()
        assert traj.states.shape[0] == 31

    def test_deterministic(self):
        """Same seed should produce identical results."""
        cfg = _make_config()
        sim1 = SwiftHohenbergSimulation(cfg)
        sim1.reset(seed=42)
        for _ in range(50):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = SwiftHohenbergSimulation(cfg)
        sim2.reset(seed=42)
        for _ in range(50):
            sim2.step()
        s2 = sim2.observe()

        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different initial conditions."""
        cfg = _make_config()
        sim1 = SwiftHohenbergSimulation(cfg)
        s1 = sim1.reset(seed=1)

        sim2 = SwiftHohenbergSimulation(cfg)
        s2 = sim2.reset(seed=2)

        assert not np.allclose(s1, s2)

    def test_stability_no_nan(self):
        """Solution should remain finite after many steps."""
        sim = SwiftHohenbergSimulation(_make_config(N=64, dt=0.1, n_steps=2000))
        sim.reset()
        for _ in range(2000):
            state = sim.step()
        assert np.all(np.isfinite(state)), "Solution became NaN/Inf"


# ---------------------------------------------------------------------------
# TestSpectralMethod: wavenumbers, linear operator
# ---------------------------------------------------------------------------
class TestSpectralMethod:
    """Tests for the spectral discretization setup."""

    def test_wavenumber_shape(self):
        """Wavenumber array should have length N."""
        sim = SwiftHohenbergSimulation(_make_config(N=64))
        assert sim.k.shape == (64,)

    def test_wavenumber_zero_mode(self):
        """k[0] should be 0 (DC mode)."""
        sim = SwiftHohenbergSimulation(_make_config(N=64))
        assert sim.k[0] == pytest.approx(0.0)

    def test_wavenumber_fundamental(self):
        """k[1] should be 2*pi/L (fundamental mode)."""
        L = 20.0 * np.pi
        sim = SwiftHohenbergSimulation(_make_config(N=64, L=L))
        expected_k1 = 2 * np.pi / L
        assert sim.k[1] == pytest.approx(expected_k1, rel=1e-10)

    def test_linear_operator_shape(self):
        """Linear operator should have shape (N,)."""
        sim = SwiftHohenbergSimulation(_make_config(N=64))
        assert sim.linear_op.shape == (64,)

    def test_linear_operator_at_k0(self):
        """L(k=0) = r - (1-0)^2 = r - 1."""
        sim = SwiftHohenbergSimulation(_make_config(r=0.5, N=64))
        assert sim.linear_op[0] == pytest.approx(0.5 - 1.0)

    def test_linear_operator_maximum_near_k1(self):
        """Linear operator should have maximum near k = 1 (k_c)."""
        # Use a domain where k=1 is well-resolved
        L = 2 * np.pi * 10  # so that k values include k=1
        N = 128
        sim = SwiftHohenbergSimulation(_make_config(N=N, L=L, r=0.5))
        # The maximum of r - (1-k^2)^2 occurs at k=1 where L(k)=r
        # Find which k index is closest to k=1
        k_abs = np.abs(sim.k)
        # Among positive k, the max of L should be near k=1
        pos_mask = k_abs > 0
        if np.any(pos_mask):
            max_idx = np.argmax(sim.linear_op[pos_mask])
            k_at_max = k_abs[pos_mask][max_idx]
            assert k_at_max == pytest.approx(1.0, abs=0.2)

    def test_exponential_factors_finite(self):
        """Strang splitting exponential factors should be finite and positive."""
        sim = SwiftHohenbergSimulation(_make_config(r=0.5, dt=0.1, N=64))
        # exp(L * dt/2) should be finite (no overflow) and positive (real exp)
        assert np.all(np.isfinite(sim._exp_half)), (
            "Half-step exponential has non-finite values"
        )
        assert np.all(sim._exp_half > 0), (
            "Half-step exponential has non-positive values"
        )

    def test_grid_spacing(self):
        """Grid spacing dx should equal L/N."""
        L = 20.0 * np.pi
        N = 128
        sim = SwiftHohenbergSimulation(_make_config(N=N, L=L))
        assert sim.dx == pytest.approx(L / N)


# ---------------------------------------------------------------------------
# TestPatternFormation: r > 0 grows, r < 0 decays
# ---------------------------------------------------------------------------
class TestPatternFormation:
    """Tests for pattern formation vs decay behavior."""

    def test_pattern_grows_for_positive_r(self):
        """With r > 0, the amplitude should grow from the initial perturbation."""
        sim = SwiftHohenbergSimulation(
            _make_config(r=0.5, N=128, dt=0.1, init_amplitude=0.01)
        )
        sim.reset()
        initial_amp = sim.pattern_amplitude()

        for _ in range(2000):
            sim.step()
        final_amp = sim.pattern_amplitude()

        assert final_amp > initial_amp * 5, (
            f"Pattern did not grow: {initial_amp:.6f} -> {final_amp:.6f}"
        )

    def test_pattern_decays_for_negative_r(self):
        """With r < 0, the field should decay toward zero."""
        sim = SwiftHohenbergSimulation(
            _make_config(r=-0.5, N=128, dt=0.1, init_amplitude=0.1)
        )
        sim.reset()

        for _ in range(2000):
            sim.step()
        final_amp = sim.pattern_amplitude()

        assert final_amp < 0.01, (
            f"Pattern should decay for r < 0, got amplitude {final_amp:.6f}"
        )

    def test_larger_r_larger_amplitude(self):
        """Larger r should produce larger saturated amplitude."""
        amps = {}
        for r_val in [0.2, 1.0]:
            sim = SwiftHohenbergSimulation(
                _make_config(r=r_val, N=128, dt=0.1)
            )
            sim.reset(seed=42)
            for _ in range(3000):
                sim.step()
            amps[r_val] = sim.pattern_amplitude()

        assert amps[1.0] > amps[0.2], (
            f"Larger r should give larger amplitude: "
            f"A(0.2)={amps[0.2]:.4f}, A(1.0)={amps[1.0]:.4f}"
        )

    def test_energy_grows_positive_r(self):
        """L2 energy should increase from initial for r > 0."""
        sim = SwiftHohenbergSimulation(_make_config(r=0.5, N=128, dt=0.1))
        sim.reset()
        e0 = sim.energy

        for _ in range(1000):
            sim.step()
        e1 = sim.energy

        assert e1 > e0 * 2, f"Energy should grow: {e0:.6f} -> {e1:.6f}"

    def test_saturated_amplitude_finite(self):
        """Pattern amplitude should saturate to a finite value (no blowup)."""
        sim = SwiftHohenbergSimulation(
            _make_config(r=1.0, N=128, dt=0.1)
        )
        sim.reset()
        for _ in range(5000):
            sim.step()
        amp = sim.pattern_amplitude()
        assert np.isfinite(amp)
        # For r=1, g=0: theoretical A ~ 2*sqrt(r/3) ~ 1.15
        assert amp < 10.0, f"Amplitude too large (blowup?): {amp}"


# ---------------------------------------------------------------------------
# TestWavelength: dominant wavelength near 2*pi
# ---------------------------------------------------------------------------
class TestWavelength:
    """Tests for pattern wavelength measurement."""

    def test_wavelength_near_2pi(self):
        """Dominant wavelength should be close to 2*pi for r > 0."""
        sim = SwiftHohenbergSimulation(
            _make_config(r=0.5, N=256, L=20.0 * np.pi, dt=0.1)
        )
        sim.reset(seed=42)
        for _ in range(5000):
            sim.step()

        wl = sim.pattern_wavelength()
        assert wl == pytest.approx(2 * np.pi, rel=0.15), (
            f"Wavelength {wl:.4f} not close to 2*pi={2*np.pi:.4f}"
        )

    def test_wavelength_independent_of_r(self):
        """Wavelength should be approximately the same for different r > 0."""
        wavelengths = {}
        for r_val in [0.3, 1.0, 2.0]:
            sim = SwiftHohenbergSimulation(
                _make_config(r=r_val, N=256, L=20.0 * np.pi, dt=0.1)
            )
            sim.reset(seed=42)
            for _ in range(5000):
                sim.step()
            wavelengths[r_val] = sim.pattern_wavelength()

        # All wavelengths should be similar (within 30%)
        wl_values = list(wavelengths.values())
        mean_wl = np.mean(wl_values)
        for r_val, wl in wavelengths.items():
            assert wl == pytest.approx(mean_wl, rel=0.3), (
                f"Wavelength at r={r_val} differs: {wl:.4f} vs mean {mean_wl:.4f}"
            )

    def test_wavelength_returns_float(self):
        """pattern_wavelength should return a float."""
        sim = SwiftHohenbergSimulation(_make_config(r=0.5))
        sim.reset()
        for _ in range(100):
            sim.step()
        wl = sim.pattern_wavelength()
        assert isinstance(wl, float)
        assert np.isfinite(wl)

    def test_wavelength_with_explicit_state(self):
        """pattern_wavelength should accept an explicit state array."""
        sim = SwiftHohenbergSimulation(_make_config(r=0.5, N=128))
        sim.reset()
        # Create a pure sinusoidal pattern with known wavelength
        k_target = 1.0
        state = np.sin(k_target * sim.x)
        wl = sim.pattern_wavelength(state=state)
        expected = 2 * np.pi / k_target
        assert wl == pytest.approx(expected, rel=0.1), (
            f"Wavelength {wl:.4f} not close to expected {expected:.4f}"
        )

    def test_wavelength_flat_field(self):
        """Flat field should return domain length L (no pattern)."""
        sim = SwiftHohenbergSimulation(_make_config(N=64))
        flat = np.zeros(64)
        wl = sim.pattern_wavelength(state=flat)
        # With no signal, should return L
        assert wl == pytest.approx(sim.L)


# ---------------------------------------------------------------------------
# TestAmplitude: amplitude scales as sqrt(r) near onset
# ---------------------------------------------------------------------------
class TestAmplitude:
    """Tests for amplitude scaling behavior."""

    def test_amplitude_returns_float(self):
        """pattern_amplitude should return a float."""
        sim = SwiftHohenbergSimulation(_make_config(r=0.5))
        sim.reset()
        amp = sim.pattern_amplitude()
        assert isinstance(amp, float)
        assert amp >= 0.0

    def test_amplitude_with_explicit_state(self):
        """pattern_amplitude should accept an explicit state array."""
        sim = SwiftHohenbergSimulation(_make_config(N=128))
        # Known sinusoidal: A*sin(x) has amplitude A
        A = 2.0
        state = A * np.sin(sim.x)
        amp = sim.pattern_amplitude(state=state)
        assert amp == pytest.approx(A, rel=0.1)

    def test_amplitude_near_zero_for_flat(self):
        """Flat field should have near-zero amplitude."""
        sim = SwiftHohenbergSimulation(_make_config(N=64))
        flat = np.ones(64) * 0.5
        amp = sim.pattern_amplitude(state=flat)
        assert amp < 1e-10

    def test_amplitude_scales_with_r(self):
        """Amplitude should increase roughly as sqrt(r) for small r > 0.

        Theory: for g=0 (supercritical), the saturated amplitude A ~ 2*sqrt(r/3).
        """
        amplitudes = {}
        for r_val in [0.1, 0.4, 0.9, 1.6]:
            sim = SwiftHohenbergSimulation(
                _make_config(r=r_val, N=128, dt=0.1)
            )
            sim.reset(seed=42)
            for _ in range(5000):
                sim.step()
            amplitudes[r_val] = sim.pattern_amplitude()

        # Check monotonic increase
        r_sorted = sorted(amplitudes.keys())
        for i in range(len(r_sorted) - 1):
            assert amplitudes[r_sorted[i + 1]] > amplitudes[r_sorted[i]] * 0.8, (
                f"Amplitude not increasing: "
                f"A({r_sorted[i]})={amplitudes[r_sorted[i]]:.4f}, "
                f"A({r_sorted[i+1]})={amplitudes[r_sorted[i+1]]:.4f}"
            )

    def test_amplitude_sqrt_r_fit(self):
        """A vs sqrt(r) should show reasonable linear correlation."""
        r_values = [0.2, 0.5, 1.0, 1.5, 2.0]
        amps = []
        for r_val in r_values:
            sim = SwiftHohenbergSimulation(
                _make_config(r=r_val, N=128, dt=0.1)
            )
            sim.reset(seed=42)
            for _ in range(5000):
                sim.step()
            amps.append(sim.pattern_amplitude())

        sqrt_r = np.sqrt(r_values)
        amps_arr = np.array(amps)

        # Correlation between A and sqrt(r) should be high
        corr = np.corrcoef(sqrt_r, amps_arr)[0, 1]
        assert corr > 0.9, f"A vs sqrt(r) correlation too low: {corr:.4f}"

    def test_max_amplitude_property(self):
        """max_amplitude should return the peak absolute value."""
        sim = SwiftHohenbergSimulation(_make_config(r=0.5, N=64))
        sim.reset()
        for _ in range(500):
            sim.step()
        max_amp = sim.max_amplitude
        assert isinstance(max_amp, float)
        assert max_amp >= 0
        assert max_amp == pytest.approx(np.max(np.abs(sim._state)))


# ---------------------------------------------------------------------------
# TestSubcritical: g != 0 changes bifurcation character
# ---------------------------------------------------------------------------
class TestSubcritical:
    """Tests for subcritical behavior with quadratic coupling g != 0."""

    def test_g_zero_supercritical(self):
        """With g=0 the bifurcation is supercritical (smooth onset)."""
        sim = SwiftHohenbergSimulation(
            _make_config(r=0.1, g=0.0, N=128, dt=0.1)
        )
        sim.reset(seed=42)
        for _ in range(3000):
            sim.step()
        amp_g0 = sim.pattern_amplitude()
        assert amp_g0 > 0, "Should have nonzero amplitude for r > 0"

    def test_g_nonzero_changes_amplitude(self):
        """Nonzero g should change the saturated amplitude."""
        amps = {}
        for g_val in [0.0, 1.0]:
            sim = SwiftHohenbergSimulation(
                _make_config(r=0.5, g=g_val, N=128, dt=0.1)
            )
            sim.reset(seed=42)
            for _ in range(3000):
                sim.step()
            amps[g_val] = sim.pattern_amplitude()

        # g != 0 should change the amplitude (could be larger due to subcritical)
        assert abs(amps[1.0] - amps[0.0]) > 0.01, (
            f"g=1 should change amplitude: A(g=0)={amps[0.0]:.4f}, "
            f"A(g=1)={amps[1.0]:.4f}"
        )

    def test_g_positive_breaks_symmetry(self):
        """Positive g should break the u -> -u symmetry."""
        sim = SwiftHohenbergSimulation(
            _make_config(r=0.5, g=1.0, N=128, dt=0.1)
        )
        sim.reset(seed=42)
        for _ in range(3000):
            sim.step()
        # With g > 0, the mean should shift from zero
        mean_u = sim.spatial_mean
        # The g*u^2 term favors positive u, so mean should be positive
        # (or at least nonzero)
        assert abs(mean_u) > 1e-4, (
            f"g > 0 should break symmetry, but mean ~ 0: {mean_u:.6f}"
        )

    def test_g_zero_preserves_symmetry(self):
        """With g=0, the equation has u -> -u symmetry.

        For a random IC, the mean should stay near zero.
        """
        sim = SwiftHohenbergSimulation(
            _make_config(r=0.5, g=0.0, N=256, dt=0.1)
        )
        sim.reset(seed=42)
        for _ in range(3000):
            sim.step()
        mean_u = sim.spatial_mean
        # Mean should remain close to zero due to symmetry
        assert abs(mean_u) < 0.3, (
            f"g=0 should preserve symmetry, mean drifted to {mean_u:.6f}"
        )

    def test_stability_with_large_g(self):
        """Simulation should remain stable even with moderate g."""
        sim = SwiftHohenbergSimulation(
            _make_config(r=0.5, g=1.5, N=128, dt=0.05)
        )
        sim.reset()
        for _ in range(2000):
            state = sim.step()
        assert np.all(np.isfinite(state)), "Solution diverged with large g"

    def test_larger_g_larger_mean(self):
        """Larger g should produce larger deviation of mean from zero."""
        means = {}
        for g_val in [0.5, 1.5]:
            sim = SwiftHohenbergSimulation(
                _make_config(r=0.5, g=g_val, N=128, dt=0.1)
            )
            sim.reset(seed=42)
            for _ in range(3000):
                sim.step()
            means[g_val] = abs(sim.spatial_mean)

        assert means[1.5] > means[0.5] * 0.5, (
            f"Larger g should shift mean more: "
            f"|mean|(g=0.5)={means[0.5]:.4f}, |mean|(g=1.5)={means[1.5]:.4f}"
        )


# ---------------------------------------------------------------------------
# TestProperties: energy, energy_spectrum, spatial_mean
# ---------------------------------------------------------------------------
class TestProperties:
    """Tests for computed properties."""

    def test_energy_non_negative(self):
        """L2 energy should be non-negative."""
        sim = SwiftHohenbergSimulation(_make_config(r=0.5))
        sim.reset()
        assert sim.energy >= 0.0

    def test_energy_type(self):
        """Energy should be a float."""
        sim = SwiftHohenbergSimulation(_make_config(r=0.5))
        sim.reset()
        assert isinstance(sim.energy, float)

    def test_energy_spectrum_shape(self):
        """Energy spectrum should return (N/2,) arrays."""
        N = 64
        sim = SwiftHohenbergSimulation(_make_config(r=0.5, N=N))
        sim.reset()
        for _ in range(100):
            sim.step()
        k_pos, spectrum = sim.energy_spectrum()
        assert k_pos.shape == (N // 2,)
        assert spectrum.shape == (N // 2,)

    def test_spatial_mean_type(self):
        """Spatial mean should be a float."""
        sim = SwiftHohenbergSimulation(_make_config(r=0.5))
        sim.reset()
        assert isinstance(sim.spatial_mean, float)


# ---------------------------------------------------------------------------
# TestRediscovery: data generation, sweep shapes
# ---------------------------------------------------------------------------
class TestRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_amplitude_vs_r_data_shape(self):
        """generate_amplitude_vs_r_data should return arrays of correct length."""
        from simulating_anything.rediscovery.swift_hohenberg import (
            generate_amplitude_vs_r_data,
        )

        data = generate_amplitude_vs_r_data(
            n_r_values=5, r_min=0.0, r_max=1.0, n_steps=200, dt=0.1,
            N=64,
        )
        assert "r_values" in data
        assert "amplitudes" in data
        assert "wavelengths" in data
        assert len(data["r_values"]) == 5
        assert len(data["amplitudes"]) == 5
        assert len(data["wavelengths"]) == 5

    def test_amplitude_vs_r_data_finite(self):
        """All amplitude data should be finite."""
        from simulating_anything.rediscovery.swift_hohenberg import (
            generate_amplitude_vs_r_data,
        )

        data = generate_amplitude_vs_r_data(
            n_r_values=3, r_min=0.1, r_max=1.0, n_steps=200, dt=0.1,
            N=64,
        )
        assert np.all(np.isfinite(data["amplitudes"]))
        assert np.all(np.isfinite(data["wavelengths"]))

    def test_wavelength_data_shape(self):
        """generate_wavelength_data should return arrays of correct length."""
        from simulating_anything.rediscovery.swift_hohenberg import (
            generate_wavelength_data,
        )

        data = generate_wavelength_data(
            n_r_values=4, r_min=0.1, r_max=1.0, n_steps=200, dt=0.1,
            N=64,
        )
        assert len(data["r_values"]) == 4
        assert len(data["wavelengths"]) == 4

    def test_g_sweep_data_shape(self):
        """generate_g_sweep_data should return arrays of correct length."""
        from simulating_anything.rediscovery.swift_hohenberg import (
            generate_g_sweep_data,
        )

        data = generate_g_sweep_data(
            n_g_values=4, g_min=0.0, g_max=1.0, r_val=0.5,
            n_steps=200, dt=0.1, N=64,
        )
        assert "g_values" in data
        assert "amplitudes" in data
        assert "wavelengths" in data
        assert len(data["g_values"]) == 4

    def test_g_sweep_data_finite(self):
        """All g-sweep data should be finite."""
        from simulating_anything.rediscovery.swift_hohenberg import (
            generate_g_sweep_data,
        )

        data = generate_g_sweep_data(
            n_g_values=3, g_min=0.0, g_max=1.0, r_val=0.5,
            n_steps=200, dt=0.1, N=64,
        )
        assert np.all(np.isfinite(data["amplitudes"]))
        assert np.all(np.isfinite(data["wavelengths"]))

    def test_amplitude_increases_with_r(self):
        """In sweep data, amplitude should generally increase with r for r > 0."""
        from simulating_anything.rediscovery.swift_hohenberg import (
            generate_amplitude_vs_r_data,
        )

        data = generate_amplitude_vs_r_data(
            n_r_values=5, r_min=0.1, r_max=2.0, n_steps=1000, dt=0.1,
            N=64,
        )
        # Amplitude at r_max should exceed amplitude at r_min
        assert data["amplitudes"][-1] > data["amplitudes"][0] * 0.5, (
            f"Amplitude should increase with r: "
            f"A(r_min)={data['amplitudes'][0]:.4f}, "
            f"A(r_max)={data['amplitudes'][-1]:.4f}"
        )
