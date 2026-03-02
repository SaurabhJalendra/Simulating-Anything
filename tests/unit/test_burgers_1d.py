"""Tests for the 1D viscous Burgers equation simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.burgers_1d import Burgers1DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    nu: float = 0.01,
    N: int = 128,
    L: float = 2 * np.pi,
    amplitude: float = 1.0,
    dt: float = 0.001,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.BURGERS_1D,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "nu": nu,
            "N": float(N),
            "L": L,
            "amplitude": amplitude,
        },
    )


def _make_sim(**kwargs) -> Burgers1DSimulation:
    return Burgers1DSimulation(_make_config(**kwargs))


# ---------------------------------------------------------------------------
# Creation and basic functionality
# ---------------------------------------------------------------------------
class TestBurgersCreation:
    def test_creation(self):
        sim = _make_sim()
        assert sim.nu == 0.01
        assert sim.N == 128
        assert sim.L == pytest.approx(2 * np.pi)
        assert sim.amplitude == 1.0

    def test_custom_params(self):
        sim = _make_sim(nu=0.1, N=256, L=10.0, amplitude=2.0)
        assert sim.nu == 0.1
        assert sim.N == 256
        assert sim.L == 10.0
        assert sim.amplitude == 2.0

    def test_default_params(self):
        """Default parameters are used when not specified."""
        config = SimulationConfig(
            domain=Domain.BURGERS_1D,
            dt=0.001,
            n_steps=100,
            parameters={},
        )
        sim = Burgers1DSimulation(config)
        assert sim.nu == 0.01
        assert sim.N == 256
        assert sim.L == pytest.approx(2 * np.pi)
        assert sim.amplitude == 1.0

    def test_grid_setup(self):
        """Grid should have N equally spaced points on [0, L)."""
        N = 64
        L = 4.0
        sim = _make_sim(N=N, L=L)
        assert len(sim.x) == N
        assert sim.x[0] == pytest.approx(0.0)
        assert sim.dx == pytest.approx(L / N)
        # Last point should be L - dx (endpoint=False)
        assert sim.x[-1] == pytest.approx(L - L / N)


class TestBurgersState:
    def test_reset_state_shape(self):
        """Reset returns state with shape (N,)."""
        N = 128
        sim = _make_sim(N=N)
        state = sim.reset()
        assert state.shape == (N,)

    def test_observe_shape(self):
        """Observe returns state with shape (N,)."""
        N = 128
        sim = _make_sim(N=N)
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (N,)

    def test_step_advances_state(self):
        sim = _make_sim()
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_step_returns_correct_shape(self):
        N = 64
        sim = _make_sim(N=N)
        sim.reset()
        state = sim.step()
        assert state.shape == (N,)

    def test_trajectory_shape(self):
        """Run returns trajectory with correct state dimension."""
        N = 64
        n_steps = 50
        sim = _make_sim(N=N, n_steps=n_steps)
        traj = sim.run(n_steps=n_steps)
        assert traj.states.shape == (n_steps + 1, N)

    def test_trajectory_timestamps(self):
        """Trajectory timestamps should be evenly spaced."""
        n_steps = 100
        dt = 0.001
        sim = _make_sim(dt=dt, n_steps=n_steps)
        traj = sim.run(n_steps=n_steps)
        assert len(traj.timestamps) == n_steps + 1
        assert traj.timestamps[0] == pytest.approx(0.0)
        assert traj.timestamps[-1] == pytest.approx(n_steps * dt)


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------
class TestBurgersInitialCondition:
    def test_ic_is_sinusoidal(self):
        """Initial condition should be A*sin(2*pi*x/L)."""
        N = 128
        L = 2 * np.pi
        amp = 1.0
        sim = _make_sim(N=N, L=L, amplitude=amp)
        state = sim.reset()
        expected = amp * np.sin(2 * np.pi * sim.x / L)
        np.testing.assert_allclose(state, expected, atol=1e-14)

    def test_ic_amplitude_scaling(self):
        """Amplitude parameter should scale the initial condition."""
        sim1 = _make_sim(amplitude=1.0)
        sim2 = _make_sim(amplitude=2.0)
        s1 = sim1.reset()
        s2 = sim2.reset()
        np.testing.assert_allclose(s2, 2.0 * s1, atol=1e-14)

    def test_ic_zero_mean(self):
        """sin(x) IC should have zero mean."""
        sim = _make_sim(N=256)
        state = sim.reset()
        assert abs(np.mean(state)) < 1e-12


# ---------------------------------------------------------------------------
# Conservation and dissipation
# ---------------------------------------------------------------------------
class TestBurgersConservation:
    def test_mass_conservation(self):
        """Total mass (integral u dx) should be conserved for periodic BC.

        For sin(x) IC the initial mass is zero and should remain zero.
        """
        sim = _make_sim(nu=0.01, N=128, dt=0.001)
        sim.reset()
        mass0 = sim.total_mass
        assert abs(mass0) < 1e-10  # sin(x) has zero integral

        for _ in range(500):
            sim.step()

        mass_final = sim.total_mass
        assert abs(mass_final - mass0) < 1e-8, (
            f"Mass not conserved: {mass0:.2e} -> {mass_final:.2e}"
        )

    def test_energy_decreases_with_viscosity(self):
        """Energy should decrease monotonically when nu > 0."""
        sim = _make_sim(nu=0.05, N=128, dt=0.001)
        sim.reset()
        E_prev = sim.total_energy

        decreasing = True
        for step in range(500):
            sim.step()
            if (step + 1) % 50 == 0:
                E = sim.total_energy
                if E > E_prev + 1e-10:
                    decreasing = False
                E_prev = E

        assert decreasing, "Energy should decrease with viscous dissipation"

    def test_energy_nonnegative(self):
        """Energy 0.5*integral(u^2) should always be non-negative."""
        sim = _make_sim(nu=0.01, N=128, dt=0.001)
        sim.reset()
        for _ in range(200):
            sim.step()
        assert sim.total_energy >= 0

    def test_higher_viscosity_faster_decay(self):
        """Higher viscosity should cause faster energy decay."""
        E_final = []
        for nu_val in [0.01, 0.1]:
            sim = _make_sim(nu=nu_val, N=128, dt=0.001)
            sim.reset()
            for _ in range(500):
                sim.step()
            E_final.append(sim.total_energy)

        # Higher viscosity -> less energy remaining
        assert E_final[1] < E_final[0]

    def test_dissipation_rate_positive(self):
        """Energy dissipation rate should be non-negative."""
        sim = _make_sim(nu=0.05, N=128, dt=0.001)
        sim.reset()
        for _ in range(100):
            sim.step()
        assert sim.energy_dissipation_rate >= 0


# ---------------------------------------------------------------------------
# Shock formation physics
# ---------------------------------------------------------------------------
class TestBurgersShock:
    def test_gradient_increases_initially(self):
        """For small nu, the gradient should increase initially (steepening)."""
        sim = _make_sim(nu=0.001, N=256, dt=0.0005)
        sim.reset()
        grad0 = sim.max_gradient

        # Run a fraction of the inviscid shock time
        t_s = sim.shock_formation_time_inviscid()
        n_steps = int(0.5 * t_s / sim.config.dt)
        for _ in range(n_steps):
            sim.step()

        grad_mid = sim.max_gradient
        assert grad_mid > grad0, (
            f"Gradient should increase: {grad0:.2f} -> {grad_mid:.2f}"
        )

    def test_shock_formation_time_inviscid(self):
        """Theoretical shock time: t_s = 1/(A*k) for u = A*sin(kx)."""
        L = 2 * np.pi
        A = 2.0
        sim = _make_sim(amplitude=A, L=L)
        sim.reset()
        t_s = sim.shock_formation_time_inviscid()
        k0 = 2 * np.pi / L
        assert t_s == pytest.approx(1.0 / (A * k0))

    def test_lower_viscosity_sharper_gradient(self):
        """Lower viscosity should produce sharper gradients at fixed time."""
        max_grads = []
        for nu_val in [0.1, 0.01, 0.001]:
            sim = _make_sim(nu=nu_val, N=512, dt=0.0005, amplitude=1.0)
            sim.reset()
            # Run to ~half the inviscid shock time
            n_steps = int(0.5 / sim.config.dt)
            for _ in range(n_steps):
                sim.step()
            max_grads.append(sim.max_gradient)

        # Gradients should increase as nu decreases
        assert max_grads[1] > max_grads[0], (
            f"Expected sharper gradient for lower nu: {max_grads}"
        )
        assert max_grads[2] > max_grads[1], (
            f"Expected sharper gradient for lower nu: {max_grads}"
        )


# ---------------------------------------------------------------------------
# Viscous decay (long time)
# ---------------------------------------------------------------------------
class TestBurgersViscousDecay:
    def test_solution_decays_to_zero(self):
        """With viscosity, the solution should decay toward zero."""
        sim = _make_sim(nu=0.5, N=128, dt=0.001)
        sim.reset()
        E0 = sim.total_energy
        for _ in range(5000):
            sim.step()

        # After long time with high viscosity, energy should be very small
        assert sim.total_energy < 0.01 * E0, (
            f"Solution should decay: E={sim.total_energy}, E0={E0}"
        )

    def test_max_value_bounded(self):
        """Maximum value should not exceed initial amplitude (max principle)."""
        amp = 1.0
        sim = _make_sim(nu=0.01, N=128, amplitude=amp, dt=0.001)
        sim.reset()

        for _ in range(500):
            sim.step()
            # Allow small numerical overshoot
            assert sim.max_value < amp + 0.1, (
                f"Max value {sim.max_value} exceeds amplitude {amp}"
            )


# ---------------------------------------------------------------------------
# CFL condition
# ---------------------------------------------------------------------------
class TestBurgersCFL:
    def test_cfl_advection_defined(self):
        """CFL advection number should be computable."""
        sim = _make_sim(nu=0.01, N=128, dt=0.001)
        sim.reset()
        cfl = sim.cfl_advection()
        assert isinstance(cfl, float)
        assert cfl >= 0

    def test_cfl_diffusion_defined(self):
        """CFL diffusion number should be computable."""
        sim = _make_sim(nu=0.01, N=128, dt=0.001)
        sim.reset()
        cfl = sim.cfl_diffusion()
        assert isinstance(cfl, float)
        assert cfl >= 0

    def test_cfl_diffusion_formula(self):
        """CFL diffusion = nu * dt / dx^2."""
        nu = 0.05
        dt = 0.001
        N = 128
        L = 2 * np.pi
        dx = L / N
        sim = _make_sim(nu=nu, N=N, L=L, dt=dt)
        sim.reset()
        expected = nu * dt / (dx ** 2)
        assert sim.cfl_diffusion() == pytest.approx(expected, rel=1e-10)

    def test_stable_parameters_no_nan(self):
        """With reasonable CFL, solution should not contain NaN."""
        sim = _make_sim(nu=0.01, N=128, dt=0.001)
        sim.reset()
        for _ in range(500):
            sim.step()
        assert np.all(np.isfinite(sim.observe()))


# ---------------------------------------------------------------------------
# Cole-Hopf exact solution
# ---------------------------------------------------------------------------
class TestBurgersColeHopf:
    def test_cole_hopf_import(self):
        """Cole-Hopf method should be callable (requires scipy)."""
        sim = _make_sim(nu=0.1, N=64)
        sim.reset()
        try:
            u_exact = sim.cole_hopf_exact(0.5)
            assert u_exact.shape == (64,)
            assert np.all(np.isfinite(u_exact))
        except ImportError:
            pytest.skip("scipy not installed")

    def test_cole_hopf_accuracy_moderate_nu(self):
        """Simulation should match Cole-Hopf for moderate viscosity."""
        nu = 0.1
        N = 256
        dt = 0.001
        t_eval = 1.0
        n_steps = int(t_eval / dt)

        sim = _make_sim(nu=nu, N=N, dt=dt)
        sim.reset()
        for _ in range(n_steps):
            sim.step()

        try:
            u_exact = sim.cole_hopf_exact(t_eval)
            l2_error = np.sqrt(np.sum((sim.observe() - u_exact) ** 2) * sim.dx)
            # For moderate nu and fine grid, error should be small
            assert l2_error < 0.05, (
                f"Cole-Hopf L2 error {l2_error:.6f} too large"
            )
        except ImportError:
            pytest.skip("scipy not installed")

    def test_cole_hopf_at_t0_matches_ic(self):
        """Cole-Hopf at t=0 should match the initial condition."""
        sim = _make_sim(nu=0.1, N=128)
        sim.reset()
        try:
            u_exact_t0 = sim.cole_hopf_exact(1e-10)
            ic = sim.observe()
            np.testing.assert_allclose(u_exact_t0, ic, atol=0.01)
        except ImportError:
            pytest.skip("scipy not installed")


# ---------------------------------------------------------------------------
# Properties and diagnostics
# ---------------------------------------------------------------------------
class TestBurgersDiagnostics:
    def test_enstrophy_nonnegative(self):
        """Enstrophy (integral u_x^2) should be non-negative."""
        sim = _make_sim(nu=0.01, N=128, dt=0.001)
        sim.reset()
        for _ in range(100):
            sim.step()
        assert sim.enstrophy >= 0

    def test_reynolds_number(self):
        """Reynolds number = A * L / (2*pi*nu)."""
        nu = 0.01
        amp = 1.0
        L = 2 * np.pi
        sim = _make_sim(nu=nu, amplitude=amp, L=L)
        sim.reset()
        expected_re = amp * L / (2 * np.pi * nu)
        assert sim.reynolds_number() == pytest.approx(expected_re)

    def test_reynolds_number_infinite_inviscid(self):
        """Reynolds number should be infinite for nu=0."""
        sim = _make_sim(nu=0.0)
        sim.reset()
        assert sim.reynolds_number() == float("inf")

    def test_mode_amplitudes_shape(self):
        """mode_amplitudes should return N complex coefficients."""
        N = 128
        sim = _make_sim(N=N)
        sim.reset()
        amps = sim.mode_amplitudes()
        assert amps.shape == (N,)
        assert np.iscomplexobj(amps)

    def test_mode_amplitudes_ic(self):
        """For sin(x) IC, mode 1 should dominate."""
        N = 128
        sim = _make_sim(N=N)
        sim.reset()
        amps = np.abs(sim.mode_amplitudes())
        # Mode 1 and mode N-1 (conjugate) should dominate
        assert amps[1] > amps[0]  # larger than DC component
        assert amps[1] > amps[2]  # larger than mode 2


# ---------------------------------------------------------------------------
# Determinism and reproducibility
# ---------------------------------------------------------------------------
class TestBurgersDeterminism:
    def test_deterministic_runs(self):
        """Two runs with same parameters should produce identical results."""
        sim1 = _make_sim(nu=0.01, N=64, dt=0.001)
        sim2 = _make_sim(nu=0.01, N=64, dt=0.001)

        state1 = sim1.reset()
        state2 = sim2.reset()
        np.testing.assert_array_equal(state1, state2)

        for _ in range(100):
            state1 = sim1.step()
            state2 = sim2.step()

        np.testing.assert_array_equal(state1, state2)

    def test_reset_restores_initial(self):
        """Resetting should restore the exact initial condition."""
        sim = _make_sim()
        state0 = sim.reset().copy()
        for _ in range(50):
            sim.step()
        state_after_reset = sim.reset()
        np.testing.assert_array_equal(state0, state_after_reset)


# ---------------------------------------------------------------------------
# Parameter sweep physics
# ---------------------------------------------------------------------------
class TestBurgersParameterSweep:
    def test_viscosity_sweep_energy(self):
        """Higher viscosity should leave less energy after fixed time."""
        nu_values = [0.005, 0.05, 0.5]
        final_energies = []
        for nu_val in nu_values:
            sim = _make_sim(nu=nu_val, N=128, dt=0.001)
            sim.reset()
            for _ in range(500):
                sim.step()
            final_energies.append(sim.total_energy)

        # Monotonically decreasing energy with increasing viscosity
        for i in range(len(final_energies) - 1):
            assert final_energies[i + 1] < final_energies[i], (
                f"Energy should decrease with nu: {final_energies}"
            )

    def test_amplitude_sweep_gradient(self):
        """Higher amplitude should produce larger gradients at fixed time."""
        max_grads = []
        for amp in [0.5, 1.0, 2.0]:
            sim = _make_sim(nu=0.01, N=256, dt=0.001, amplitude=amp)
            sim.reset()
            for _ in range(200):
                sim.step()
            max_grads.append(sim.max_gradient)

        # Monotonically increasing max gradient with amplitude
        for i in range(len(max_grads) - 1):
            assert max_grads[i + 1] > max_grads[i], (
                f"Gradient should increase with amplitude: {max_grads}"
            )


# ---------------------------------------------------------------------------
# Rediscovery data generation
# ---------------------------------------------------------------------------
class TestBurgersRediscovery:
    def test_shock_formation_data(self):
        from simulating_anything.rediscovery.burgers_1d import (
            generate_shock_formation_data,
        )
        data = generate_shock_formation_data(
            n_amp=5, N=128, dt=0.001, nu=0.01,
        )
        assert len(data["amplitudes"]) == 5
        assert len(data["t_shock_measured"]) == 5
        assert len(data["t_shock_theory"]) == 5
        # All should be finite and positive
        assert np.all(np.isfinite(data["t_shock_measured"]))
        assert np.all(data["t_shock_theory"] > 0)

    def test_viscosity_sweep_data(self):
        from simulating_anything.rediscovery.burgers_1d import (
            generate_viscosity_sweep_data,
        )
        data = generate_viscosity_sweep_data(
            n_nu=5, N=128, dt=0.001, t_eval=0.5,
        )
        assert len(data["nu_values"]) == 5
        assert len(data["max_gradients"]) == 5
        assert len(data["energies"]) == 5
        assert np.all(np.isfinite(data["max_gradients"]))

    def test_energy_dissipation_data(self):
        from simulating_anything.rediscovery.burgers_1d import (
            generate_energy_dissipation_data,
        )
        data = generate_energy_dissipation_data(
            nu=0.05, N=64, dt=0.001, n_steps=500, sample_interval=10,
        )
        assert len(data["times"]) > 0
        assert len(data["energies"]) == len(data["times"])
        assert len(data["dissipation_rates"]) == len(data["times"])
        # Energy should be decreasing
        assert data["energies"][-1] < data["energies"][0]

    def test_cole_hopf_comparison_data(self):
        try:
            from simulating_anything.rediscovery.burgers_1d import (
                generate_cole_hopf_comparison,
            )
            data = generate_cole_hopf_comparison(
                nu=0.1, N=64, dt=0.001, n_snapshots=5, t_final=1.0,
            )
            assert len(data["times"]) > 0
            assert len(data["l2_errors"]) == len(data["times"])
            assert np.all(data["l2_errors"] >= 0)
        except ImportError:
            pytest.skip("scipy not installed")
