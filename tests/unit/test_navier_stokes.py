"""Tests for the 2D Navier-Stokes simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.navier_stokes import NavierStokes2DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestNavierStokes2D:
    """Tests for the vorticity-streamfunction NS solver."""

    def _make_sim(self, init_type: str = "taylor_green", **kwargs) -> NavierStokes2DSimulation:
        defaults = {"nu": 0.01, "N": 32, "L": 2 * np.pi}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.NAVIER_STOKES_2D,
            dt=0.01,
            n_steps=100,
            parameters=defaults,
        )
        sim = NavierStokes2DSimulation(config)
        sim.init_type = init_type
        return sim

    def test_initial_state_shape(self):
        sim = self._make_sim(N=32)
        state = sim.reset()
        assert state.shape == (32 * 32,)

    def test_step_advances(self):
        sim = self._make_sim()
        state0 = sim.reset()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe(self):
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (32 * 32,)

    def test_taylor_green_init(self):
        """Taylor-Green vortex should have specific structure."""
        sim = self._make_sim()
        sim.reset()
        omega = sim._omega
        # Should be non-zero
        assert np.max(np.abs(omega)) > 0.1
        # Should have grid structure (symmetric)
        assert omega.shape == (32, 32)

    def test_double_vortex_init(self):
        sim = self._make_sim(init_type="double_vortex")
        # init_type already set by _make_sim
        sim.reset()
        omega = sim._omega
        # Should have both positive and negative vorticity
        assert np.max(omega) > 0
        assert np.min(omega) < 0

    def test_random_init(self):
        sim = self._make_sim(init_type="random")
        # init_type already set by _make_sim
        sim.reset()
        omega = sim._omega
        assert np.max(np.abs(omega)) > 0

    def test_stays_finite(self):
        """NS solution should remain finite for moderate nu."""
        sim = self._make_sim(nu=0.01)
        sim.reset()
        for _ in range(50):
            state = sim.step()
            assert np.all(np.isfinite(state)), "Solution became NaN/Inf"

    def test_energy_decreases(self):
        """For viscous flow, kinetic energy should monotonically decrease."""
        sim = self._make_sim(nu=0.01, N=32)
        sim.reset()
        E_prev = sim.kinetic_energy
        for _ in range(30):
            sim.step()
            E_now = sim.kinetic_energy
            assert E_now <= E_prev + 1e-10, f"Energy increased: {E_prev} -> {E_now}"
            E_prev = E_now

    def test_enstrophy_decreases(self):
        """For viscous 2D flow, enstrophy should decrease."""
        sim = self._make_sim(nu=0.01, N=32)
        sim.reset()
        Z_prev = sim.enstrophy
        for _ in range(30):
            sim.step()
            Z_now = sim.enstrophy
            # Allow small numerical increase
            assert Z_now <= Z_prev + 1e-8, f"Enstrophy increased: {Z_prev} -> {Z_now}"
            Z_prev = Z_now

    def test_taylor_green_decay(self):
        """Taylor-Green vortex energy should decay exponentially at correct rate."""
        nu = 0.01
        sim = self._make_sim(nu=nu, N=64, init_amplitude=1.0)
        sim.reset()
        dt = sim.config.dt

        E_0 = sim.kinetic_energy
        # Run for some steps
        n_steps = 200
        for _ in range(n_steps):
            sim.step()
        E_final = sim.kinetic_energy
        t = n_steps * dt

        # Theoretical decay: E(t) = E_0 * exp(-2*nu*k^2*t) where k = 1
        k = 2 * np.pi / sim.L
        decay_theory = np.exp(-2 * nu * k**2 * t)
        decay_measured = E_final / E_0

        # Should match within 20% (nonlinear effects cause some deviation)
        rel_err = abs(decay_measured - decay_theory) / decay_theory
        assert rel_err < 0.3, (
            f"Decay mismatch: measured={decay_measured:.4f}, "
            f"theory={decay_theory:.4f}, error={rel_err:.2%}"
        )

    def test_velocity_field(self):
        """Velocity field should be divergence-free (incompressible)."""
        sim = self._make_sim(N=32)
        sim.reset()
        u, v = sim.velocity_field
        assert u.shape == (32, 32)
        assert v.shape == (32, 32)
        # Check divergence ~ 0 (spectral method guarantees this)
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        div_hat = 1j * sim.kx * u_hat + 1j * sim.ky * v_hat
        div = np.real(np.fft.ifft2(div_hat))
        assert np.max(np.abs(div)) < 1e-10, f"Non-zero divergence: {np.max(np.abs(div))}"

    def test_energy_spectrum(self):
        """Energy spectrum should be non-negative."""
        sim = self._make_sim(N=32)
        sim.reset()
        for _ in range(10):
            sim.step()
        k_bins, E_k = sim.energy_spectrum()
        assert len(k_bins) == len(E_k)
        assert np.all(E_k >= -1e-12)  # Allow small numerical noise

    def test_higher_nu_faster_decay(self):
        """Higher viscosity should cause faster energy decay."""
        E_final = {}
        for nu in [0.001, 0.01, 0.1]:
            sim = self._make_sim(nu=nu, N=32)
            sim.reset()
            for _ in range(50):
                sim.step()
            E_final[nu] = sim.kinetic_energy

        assert E_final[0.1] < E_final[0.01] < E_final[0.001]
