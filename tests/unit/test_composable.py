"""Tests for the composable dynamics module library."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.composable import (
    BrusselatorKinetics,
    ComposedSimulation,
    CubicForce,
    GravitationalForce,
    HarmonicForce,
    LinearDamping,
    LogisticGrowth,
    NewtonianDynamics,
    PendulumForce,
    PeriodicForcing,
    PredatorPreyInteraction,
    SIRDynamics,
    VanDerPolDamping,
    make_duffing_oscillator,
    make_harmonic_oscillator,
    make_lotka_volterra,
    make_pendulum,
    make_sir,
    make_van_der_pol,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestDynamicsModules:
    """Test individual dynamics modules."""

    def test_harmonic_force(self):
        mod = HarmonicForce(var="x", accel_var="a", param_k="k")
        result = mod.compute({"x": 2.0}, {"k": 3.0}, 0.0)
        assert result["a"] == pytest.approx(-6.0)

    def test_cubic_force(self):
        mod = CubicForce(var="x", accel_var="a", param_beta="b")
        result = mod.compute({"x": 2.0}, {"b": 1.0}, 0.0)
        assert result["a"] == pytest.approx(-8.0)

    def test_periodic_forcing(self):
        mod = PeriodicForcing(accel_var="a", param_gamma="g", param_omega="w")
        result = mod.compute({}, {"g": 1.0, "w": 0.0}, 0.0)
        assert result["a"] == pytest.approx(1.0)  # cos(0) = 1

    def test_gravitational_force(self):
        mod = GravitationalForce(accel_var="a", param_g="g")
        result = mod.compute({}, {"g": 9.81}, 0.0)
        assert result["a"] == pytest.approx(-9.81)

    def test_pendulum_force(self):
        mod = PendulumForce(angle_var="th", accel_var="a", param_g="g", param_L="L")
        result = mod.compute({"th": 0.0}, {"g": 9.81, "L": 1.0}, 0.0)
        assert result["a"] == pytest.approx(0.0)  # sin(0) = 0

    def test_linear_damping(self):
        mod = LinearDamping(vel_var="v", accel_var="a", param_c="c")
        result = mod.compute({"v": 5.0}, {"c": 2.0}, 0.0)
        assert result["a"] == pytest.approx(-10.0)

    def test_vdp_damping(self):
        mod = VanDerPolDamping(pos_var="x", vel_var="v", accel_var="a", param_mu="mu")
        # At x=0: mu*(1-0)*v = mu*v
        result = mod.compute({"x": 0.0, "v": 1.0}, {"mu": 2.0}, 0.0)
        assert result["a"] == pytest.approx(2.0)
        # At x=1: mu*(1-1)*v = 0
        result2 = mod.compute({"x": 1.0, "v": 1.0}, {"mu": 2.0}, 0.0)
        assert result2["a"] == pytest.approx(0.0)

    def test_newtonian_dynamics(self):
        mod = NewtonianDynamics(pos_var="x", vel_var="v", accel_var="a", param_m="m")
        result = mod.compute({"x": 0.0, "v": 3.0, "a": 6.0}, {"m": 2.0}, 0.0)
        assert result["x"] == pytest.approx(3.0)  # dx/dt = v
        assert result["v"] == pytest.approx(3.0)  # dv/dt = a/m = 6/2

    def test_logistic_growth(self):
        mod = LogisticGrowth(var="N", param_r="r", param_K="K")
        result = mod.compute({"N": 50.0}, {"r": 0.5, "K": 100.0}, 0.0)
        assert result["N"] == pytest.approx(12.5)  # 0.5*50*(1-50/100)

    def test_predator_prey_interaction(self):
        mod = PredatorPreyInteraction(
            prey_var="N", pred_var="P",
            param_beta="b", param_delta="d", param_gamma="g",
        )
        result = mod.compute({"N": 10.0, "P": 5.0}, {"b": 0.4, "d": 0.1, "g": 0.3}, 0.0)
        assert result["N"] == pytest.approx(-20.0)  # -0.4*10*5
        assert result["P"] == pytest.approx(3.5)  # 0.1*10*5 - 0.3*5

    def test_sir_dynamics(self):
        mod = SIRDynamics(param_beta="b", param_gamma="g", param_N="N")
        state = {"S": 990.0, "I": 10.0, "R": 0.0}
        result = mod.compute(state, {"b": 0.3, "g": 0.1, "N": 1000.0}, 0.0)
        assert result["S"] < 0  # S decreasing
        assert result["R"] > 0  # R increasing
        # dS + dI + dR should sum to 0 (conservation)
        total = result["S"] + result["I"] + result["R"]
        assert total == pytest.approx(0.0, abs=1e-10)

    def test_brusselator_kinetics(self):
        mod = BrusselatorKinetics(param_a="a", param_b="b")
        # At steady state (a, b/a): du/dt = 0, dv/dt = 0
        result = mod.compute({"u": 1.0, "v": 3.0}, {"a": 1.0, "b": 3.0}, 0.0)
        assert result["u"] == pytest.approx(0.0, abs=1e-10)
        assert result["v"] == pytest.approx(0.0, abs=1e-10)


class TestComposedSimulation:
    """Test composed simulation creation and execution."""

    def test_from_modules_basic(self):
        sim = make_harmonic_oscillator(k=4.0, m=1.0, c=0.0, x_0=1.0)
        assert isinstance(sim, ComposedSimulation)
        assert "x" in sim.get_variable_names()
        assert "v" in sim.get_variable_names()

    def test_reset(self):
        sim = make_harmonic_oscillator(k=1.0, x_0=2.0, v_0=0.5)
        state = sim.reset()
        assert state[0] == pytest.approx(2.0)
        assert state[1] == pytest.approx(0.5)

    def test_step_deterministic(self):
        sim = make_harmonic_oscillator(k=4.0, m=1.0, c=0.0, x_0=1.0, v_0=0.0)
        sim.reset()
        s1 = sim.step().copy()
        sim.reset()
        s2 = sim.step().copy()
        np.testing.assert_array_equal(s1, s2)

    def test_observe_matches_step(self):
        sim = make_harmonic_oscillator()
        sim.reset()
        state = sim.step()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_harmonic_oscillator_frequency(self):
        """Composed oscillator should have omega = sqrt(k/m)."""
        k, m = 9.0, 1.0
        sim = make_harmonic_oscillator(k=k, m=m, c=0.0, x_0=1.0, v_0=0.0, dt=0.001, n_steps=10000)
        sim.reset()
        positions = []
        for _ in range(10000):
            state = sim.step()
            positions.append(state[0])

        # Find period via zero crossings
        positions = np.array(positions)
        crossings = np.where(np.diff(np.sign(positions)))[0]
        if len(crossings) >= 4:
            half_periods = np.diff(crossings) * 0.001
            period = 2 * np.mean(half_periods)
            omega_measured = 2 * np.pi / period
            omega_theory = np.sqrt(k / m)
            assert omega_measured == pytest.approx(omega_theory, rel=0.02)

    def test_harmonic_energy_conservation(self):
        """Undamped oscillator should conserve energy."""
        k = 4.0
        sim = make_harmonic_oscillator(k=k, m=1.0, c=0.0, x_0=1.0, v_0=0.0, dt=0.001, n_steps=5000)
        sim.reset()
        initial_energy = 0.5 * k * 1.0 ** 2  # KE + PE

        for _ in range(5000):
            state = sim.step()

        x, v = state
        final_energy = 0.5 * k * x ** 2 + 0.5 * v ** 2
        assert final_energy == pytest.approx(initial_energy, rel=0.01)

    def test_damped_oscillator_decay(self):
        """Damped oscillator amplitude should decrease."""
        sim = make_harmonic_oscillator(k=4.0, m=1.0, c=0.5, x_0=1.0, v_0=0.0, dt=0.01, n_steps=2000)
        sim.reset()
        for _ in range(2000):
            state = sim.step()
        # Amplitude should be much smaller than initial
        assert abs(state[0]) < 0.5

    def test_get_variable_names(self):
        sim = make_harmonic_oscillator()
        names = sim.get_variable_names()
        assert "x" in names
        assert "v" in names

    def test_get_module_names(self):
        sim = make_harmonic_oscillator()
        names = sim.get_module_names()
        assert "HarmonicForce" in names
        assert "NewtonianDynamics" in names


class TestRecipes:
    """Test pre-built composed simulation recipes."""

    def test_duffing_runs(self):
        sim = make_duffing_oscillator(dt=0.01, n_steps=100)
        sim.reset()
        for _ in range(100):
            state = sim.step()
        assert np.all(np.isfinite(state))

    def test_van_der_pol_limit_cycle(self):
        """VdP should converge to limit cycle with amplitude ~2."""
        sim = make_van_der_pol(mu=1.0, x_0=0.1, v_0=0.0, dt=0.01, n_steps=5000)
        sim.reset()
        for _ in range(3000):
            sim.step()
        # After transient, check amplitude
        max_x = 0.0
        for _ in range(2000):
            state = sim.step()
            max_x = max(max_x, abs(state[0]))
        assert max_x == pytest.approx(2.0, abs=0.3)

    def test_pendulum_small_angle(self):
        """Small-angle pendulum should match harmonic oscillator."""
        sim = make_pendulum(g=9.81, L=1.0, c=0.0, theta_0=0.1, dt=0.001, n_steps=5000)
        sim.reset()
        positions = []
        for _ in range(5000):
            state = sim.step()
            positions.append(state[0])
        positions = np.array(positions)
        crossings = np.where(np.diff(np.sign(positions)))[0]
        if len(crossings) >= 4:
            half_periods = np.diff(crossings) * 0.001
            period = 2 * np.mean(half_periods)
            period_theory = 2 * np.pi * np.sqrt(1.0 / 9.81)
            assert period == pytest.approx(period_theory, rel=0.02)

    def test_lotka_volterra_runs(self):
        sim = make_lotka_volterra(dt=0.01, n_steps=500)
        sim.reset()
        for _ in range(500):
            state = sim.step()
        assert np.all(np.isfinite(state))
        assert np.all(state > 0)  # Populations stay positive

    def test_sir_conservation(self):
        """SIR should conserve total population."""
        sim = make_sir(N_pop=1000.0, S_0=990.0, I_0=10.0, R_0=0.0, dt=0.1, n_steps=500)
        sim.reset()
        for _ in range(500):
            state = sim.step()
        total = sum(state)
        assert total == pytest.approx(1000.0, rel=0.01)

    def test_duffing_matches_monolithic(self):
        """Composed Duffing should produce similar results to monolithic."""
        from simulating_anything.simulation.duffing import DuffingOscillator

        dt = 0.005
        n_steps = 500
        params = {
            "alpha": 1.0, "beta": 1.0, "delta": 0.2,
            "gamma_f": 0.3, "omega": 1.0, "x_0": 0.5, "v_0": 0.0,
        }

        # Composed version
        composed = make_duffing_oscillator(**params, dt=dt, n_steps=n_steps)
        composed.reset()
        for _ in range(n_steps):
            c_state = composed.step()

        # Monolithic version
        mono_config = SimulationConfig(
            domain=Domain.DUFFING, parameters=params, dt=dt, n_steps=n_steps,
        )
        mono = DuffingOscillator(mono_config)
        mono.reset()
        for _ in range(n_steps):
            m_state = mono.step()

        # Should be close (not exact due to implementation details)
        assert c_state[0] == pytest.approx(m_state[0], abs=0.5)

    def test_composed_run_trajectory(self):
        """ComposedSimulation.run() should return TrajectoryData."""
        sim = make_harmonic_oscillator(n_steps=100)
        traj = sim.run(100)
        assert traj.states.shape == (101, 2)
        assert len(traj.timestamps) == 101

    def test_composed_custom_domain(self):
        """Composed simulations use Domain.CUSTOM."""
        sim = make_harmonic_oscillator()
        assert sim.config.domain == Domain.CUSTOM


class TestComposability:
    """Test that modules compose correctly."""

    def test_additive_composition(self):
        """Adding a zero-strength force should not change dynamics."""
        sim1 = make_harmonic_oscillator(k=4.0, c=0.0, x_0=1.0, dt=0.01, n_steps=100)
        sim1.reset()

        # Add a zero-amplitude forcing
        sim2 = ComposedSimulation.from_modules(
            modules=[
                HarmonicForce(var="x", accel_var="a_x", param_k="k"),
                PeriodicForcing(accel_var="a_x", param_gamma="gamma_f", param_omega="omega"),
                NewtonianDynamics(pos_var="x", vel_var="v", accel_var="a_x", param_m="m"),
            ],
            defaults={"k": 4.0, "m": 1.0, "gamma_f": 0.0, "omega": 1.0, "x_0": 1.0, "v_0": 0.0},
            accel_vars={"a_x"},
            dt=0.01, n_steps=100,
        )
        sim2.reset()

        for _ in range(100):
            s1 = sim1.step().copy()
            s2 = sim2.step().copy()

        np.testing.assert_array_almost_equal(s1, s2, decimal=10)

    def test_module_list_immutable(self):
        """Modifying the original module list should not affect the sim."""
        modules = [
            HarmonicForce(var="x", accel_var="a_x", param_k="k"),
            NewtonianDynamics(pos_var="x", vel_var="v", accel_var="a_x", param_m="m"),
        ]
        sim = ComposedSimulation.from_modules(
            modules=modules,
            defaults={"k": 1.0, "m": 1.0, "x_0": 1.0, "v_0": 0.0},
            accel_vars={"a_x"},
        )
        modules.clear()
        assert len(sim._modules) == 2
