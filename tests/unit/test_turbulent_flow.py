"""Tests for the 2D turbulent flow simulation."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.turbulent_flow import TurbulentFlow2DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestTurbulentFlow:
    """Tests for the spectral 2D turbulent flow solver."""

    def _make_sim(self, N: int = 32, **kwargs) -> TurbulentFlow2DSimulation:
        defaults: dict[str, float] = {
            "nu": 1e-3,
            "N": float(N),
            "L": 2 * np.pi,
            "forcing_k": 4.0,
            "forcing_amplitude": 0.1,
            "hyper_nu": 1e-10,
            "hyper_order": 4.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.TURBULENT_FLOW,
            dt=0.005,
            n_steps=100,
            parameters=defaults,
            seed=42,
        )
        return TurbulentFlow2DSimulation(config)

    # --- Instantiation ---

    def test_instantiation_defaults(self):
        """Simulation instantiates with default parameters."""
        sim = self._make_sim()
        assert sim.N == 32
        assert sim.nu == 1e-3
        assert sim.L == 2 * np.pi
        assert sim.forcing_k == 4
        assert sim.forcing_amp == 0.1

    def test_instantiation_custom_params(self):
        """Simulation instantiates with custom nu and N."""
        sim = self._make_sim(N=16, nu=5e-4)
        assert sim.N == 16
        assert sim.nu == 5e-4

    # --- Reset ---

    def test_reset_returns_correct_shape(self):
        """Reset returns a flattened array of shape (N*N,)."""
        sim = self._make_sim(N=32)
        state = sim.reset()
        assert state.shape == (32 * 32,)

    def test_reset_no_nan(self):
        """Reset state contains no NaN values."""
        sim = self._make_sim()
        state = sim.reset()
        assert not np.any(np.isnan(state))

    def test_reset_nonzero(self):
        """Reset state is not all zeros (random IC is injected)."""
        sim = self._make_sim()
        state = sim.reset()
        assert np.max(np.abs(state)) > 0

    # --- Step ---

    def test_step_returns_correct_shape(self):
        """Step returns the same flattened shape as reset."""
        sim = self._make_sim()
        sim.reset()
        state = sim.step()
        assert state.shape == (32 * 32,)

    def test_step_no_nan_10_steps(self):
        """10 consecutive steps produce no NaN values."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10):
            state = sim.step()
            assert not np.any(np.isnan(state)), "NaN detected during stepping"

    def test_step_changes_state(self):
        """A single step changes the vorticity field."""
        sim = self._make_sim()
        state0 = sim.reset()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_multiple_steps_no_crash(self):
        """50 consecutive steps complete without error."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(50):
            sim.step()
        # If we reach here, no crash occurred.
        assert sim._step_count == 50

    # --- Observe ---

    def test_observe_matches_state(self):
        """observe() returns a copy consistent with the current state."""
        sim = self._make_sim()
        sim.reset()
        sim.step()
        obs = sim.observe()
        assert obs.shape == (32 * 32,)
        np.testing.assert_array_equal(obs, sim._state)

    def test_observe_returns_copy(self):
        """observe() returns a copy, not a reference."""
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        obs[:] = 0.0
        assert np.max(np.abs(sim._state)) > 0

    # --- Determinism ---

    def test_determinism_same_seed(self):
        """Same seed produces identical trajectories."""
        sim = self._make_sim()
        state_a = sim.reset(seed=123)
        for _ in range(5):
            state_a = sim.step()

        state_b = sim.reset(seed=123)
        for _ in range(5):
            state_b = sim.step()

        np.testing.assert_array_equal(state_a, state_b)

    def test_different_seeds_differ(self):
        """Different seeds produce different initial states."""
        sim = self._make_sim()
        state_a = sim.reset(seed=1)
        state_b = sim.reset(seed=2)
        assert not np.allclose(state_a, state_b)

    # --- Energy ---

    def test_total_energy_positive(self):
        """Total kinetic energy is positive after initialization."""
        sim = self._make_sim()
        sim.reset()
        E = sim.total_energy()
        assert E > 0, f"Energy should be positive, got {E}"
        assert np.isfinite(E)

    def test_total_energy_after_steps(self):
        """Total energy remains positive and finite after stepping."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10):
            sim.step()
        E = sim.total_energy()
        assert E > 0
        assert np.isfinite(E)

    # --- Enstrophy ---

    def test_total_enstrophy_positive(self):
        """Total enstrophy is positive after initialization."""
        sim = self._make_sim()
        sim.reset()
        Z = sim.total_enstrophy()
        assert Z > 0, f"Enstrophy should be positive, got {Z}"
        assert np.isfinite(Z)

    def test_total_enstrophy_after_steps(self):
        """Enstrophy remains positive and finite after stepping."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10):
            sim.step()
        Z = sim.total_enstrophy()
        assert Z > 0
        assert np.isfinite(Z)

    # --- Reynolds number ---

    def test_reynolds_number_positive_finite(self):
        """Reynolds number is positive and finite."""
        sim = self._make_sim()
        sim.reset()
        Re = sim.reynolds_number()
        assert Re > 0, f"Reynolds number should be positive, got {Re}"
        assert np.isfinite(Re)

    def test_reynolds_scales_with_viscosity(self):
        """Lower viscosity should yield higher Reynolds number (same IC seed)."""
        sim_high_nu = self._make_sim(nu=1e-2)
        sim_high_nu.reset(seed=42)
        Re_high = sim_high_nu.reynolds_number()

        sim_low_nu = self._make_sim(nu=1e-4)
        sim_low_nu.reset(seed=42)
        Re_low = sim_low_nu.reynolds_number()

        # Same IC -> same velocity field -> Re ~ 1/nu
        assert Re_low > Re_high, (
            f"Lower nu should give higher Re: got {Re_low} vs {Re_high}"
        )

    # --- Energy spectrum ---

    def test_energy_spectrum_shapes(self):
        """energy_spectrum returns (k_bins, E_k) with matching lengths."""
        sim = self._make_sim(N=32)
        sim.reset()
        k_bins, E_k = sim.energy_spectrum()
        assert k_bins.shape == E_k.shape
        assert len(k_bins) == 32 // 2  # k_max = N // 2

    def test_energy_spectrum_nonnegative(self):
        """Energy spectrum values are non-negative."""
        sim = self._make_sim()
        sim.reset()
        _, E_k = sim.energy_spectrum()
        assert np.all(E_k >= 0)

    def test_energy_spectrum_decays_at_high_k(self):
        """Energy at highest wavenumbers should be less than at forcing scale."""
        sim = self._make_sim(N=32)
        sim.reset()
        # Run a few steps so the spectrum is shaped by forcing
        for _ in range(20):
            sim.step()
        k_bins, E_k = sim.energy_spectrum()
        # Compare energy near forcing wavenumber (k~4) to high k (last few bins)
        forcing_idx = int(sim.forcing_k) - 1  # k_bins starts at 1
        high_k_energy = np.mean(E_k[-3:])
        forcing_energy = E_k[forcing_idx]
        # With forcing at k~4, energy there should exceed the tail
        if forcing_energy > 0:
            assert high_k_energy < forcing_energy, (
                f"High-k energy {high_k_energy} should be less than "
                f"forcing-scale energy {forcing_energy}"
            )

    def test_energy_spectrum_wavenumber_bins(self):
        """Wavenumber bins start at 1 and are consecutive integers."""
        sim = self._make_sim(N=32)
        sim.reset()
        k_bins, _ = sim.energy_spectrum()
        np.testing.assert_array_equal(k_bins, np.arange(1, 17, dtype=np.float64))

    # --- Internal consistency ---

    def test_omega_grid_shape(self):
        """Internal vorticity field has shape (N, N)."""
        sim = self._make_sim(N=32)
        sim.reset()
        assert sim._omega.shape == (32, 32)

    def test_dissipation_array_shape(self):
        """Dissipation operator has shape (N, N)."""
        sim = self._make_sim(N=32)
        assert sim._dissipation.shape == (32, 32)

    def test_dealias_mask_shape(self):
        """Dealiasing mask has shape (N, N)."""
        sim = self._make_sim(N=32)
        assert sim._dealias.shape == (32, 32)

    def test_dealias_mask_zeros_high_k(self):
        """Dealiasing mask zeros out modes above 2/3 Nyquist."""
        sim = self._make_sim(N=32)
        kmax = 32 // 3  # = 10
        # All entries where |kx| > kmax should be zero
        mask = sim._dealias
        kx = np.fft.fftfreq(32, d=sim.L / (32 * 2 * np.pi))
        _, ky_grid = np.meshgrid(kx, kx, indexing="ij")
        kx_grid = np.meshgrid(kx, kx, indexing="ij")[0]
        high_k = (np.abs(kx_grid) > kmax) | (np.abs(ky_grid) > kmax)
        assert np.all(mask[high_k] == 0.0)

    # --- Custom grid size ---

    def test_small_grid_n16(self):
        """Simulation works with N=16."""
        sim = self._make_sim(N=16)
        state = sim.reset()
        assert state.shape == (16 * 16,)
        state = sim.step()
        assert state.shape == (16 * 16,)
        assert not np.any(np.isnan(state))

    def test_larger_grid_n64(self):
        """Simulation works with N=64."""
        sim = self._make_sim(N=64)
        state = sim.reset()
        assert state.shape == (64 * 64,)
        state = sim.step()
        assert state.shape == (64 * 64,)
        assert not np.any(np.isnan(state))

    # --- Forcing ---

    def test_forcing_mask_shape(self):
        """Forcing mask has shape (N, N)."""
        sim = self._make_sim(N=32)
        assert sim._forcing_mask.shape == (32, 32)

    def test_forcing_mask_nonzero_in_band(self):
        """Forcing mask is nonzero in the forcing wavenumber band."""
        sim = self._make_sim(N=32)
        assert np.sum(sim._forcing_mask) > 0

    def test_zero_forcing_amplitude(self):
        """With zero forcing amplitude, energy decays (purely dissipative)."""
        sim = self._make_sim(forcing_amplitude=0.0, nu=1e-2)
        sim.reset(seed=42)
        E0 = sim.total_energy()
        for _ in range(30):
            sim.step()
        E1 = sim.total_energy()
        assert E1 < E0, (
            f"Energy should decay without forcing: E0={E0}, E1={E1}"
        )

    # --- Hyperviscosity ---

    def test_dissipation_increases_with_k(self):
        """Dissipation operator grows with wavenumber magnitude."""
        sim = self._make_sim(N=32)
        # Check that dissipation at high k is larger than at low k
        # k2 at (1,0) vs (10,0) in frequency space
        low_k_diss = sim._dissipation[1, 0]
        high_k_diss = sim._dissipation[10, 0]
        assert high_k_diss > low_k_diss

    # --- Step count ---

    def test_step_count_increments(self):
        """Step counter increments correctly."""
        sim = self._make_sim()
        sim.reset()
        assert sim._step_count == 0
        sim.step()
        assert sim._step_count == 1
        for _ in range(4):
            sim.step()
        assert sim._step_count == 5
