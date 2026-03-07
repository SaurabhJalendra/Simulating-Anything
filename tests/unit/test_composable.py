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

# ---------------------------------------------------------------------------
# 1. Individual module tests
# ---------------------------------------------------------------------------

class TestHarmonicForce:
    """Tests for the HarmonicForce module."""

    def test_variables(self):
        mod = HarmonicForce(var="x", accel_var="a", param_k="k")
        assert mod.variables() == ["a"]

    def test_compute_positive_displacement(self):
        mod = HarmonicForce(var="x", accel_var="a", param_k="k")
        result = mod.compute({"x": 2.0}, {"k": 3.0}, 0.0)
        assert result["a"] == pytest.approx(-6.0)

    def test_compute_negative_displacement(self):
        mod = HarmonicForce(var="x", accel_var="a", param_k="k")
        result = mod.compute({"x": -2.0}, {"k": 3.0}, 0.0)
        assert result["a"] == pytest.approx(6.0)

    def test_zero_displacement(self):
        mod = HarmonicForce()
        result = mod.compute({"x": 0.0}, {"k": 5.0}, 0.0)
        assert result["a_x"] == pytest.approx(0.0)


class TestCubicForce:
    """Tests for the CubicForce module."""

    def test_variables(self):
        mod = CubicForce(var="x", accel_var="a", param_beta="b")
        assert mod.variables() == ["a"]

    def test_compute(self):
        mod = CubicForce(var="x", accel_var="a", param_beta="b")
        result = mod.compute({"x": 2.0}, {"b": 1.0}, 0.0)
        assert result["a"] == pytest.approx(-8.0)

    def test_odd_symmetry(self):
        """Cubic force should be an odd function of displacement."""
        mod = CubicForce(var="x", accel_var="a", param_beta="b")
        r_pos = mod.compute({"x": 1.5}, {"b": 2.0}, 0.0)["a"]
        r_neg = mod.compute({"x": -1.5}, {"b": 2.0}, 0.0)["a"]
        assert r_pos == pytest.approx(-r_neg)


class TestPeriodicForcing:
    """Tests for the PeriodicForcing module."""

    def test_variables(self):
        mod = PeriodicForcing(accel_var="a")
        assert mod.variables() == ["a"]

    def test_compute_at_t_zero(self):
        mod = PeriodicForcing(accel_var="a", param_gamma="g", param_omega="w")
        result = mod.compute({}, {"g": 5.0, "w": 1.0}, 0.0)
        assert result["a"] == pytest.approx(5.0)

    def test_compute_at_quarter_period(self):
        """cos(pi/2) = 0, so force should vanish."""
        mod = PeriodicForcing(accel_var="a", param_gamma="g", param_omega="w")
        result = mod.compute({}, {"g": 5.0, "w": 1.0}, np.pi / 2)
        assert result["a"] == pytest.approx(0.0, abs=1e-12)

    def test_defaults_zero_amplitude(self):
        """Missing gamma_f defaults to 0."""
        mod = PeriodicForcing()
        result = mod.compute({}, {}, 0.0)
        assert result["a_x"] == pytest.approx(0.0)


class TestGravitationalForce:
    """Tests for the GravitationalForce module."""

    def test_variables(self):
        mod = GravitationalForce(accel_var="a")
        assert mod.variables() == ["a"]

    def test_compute(self):
        mod = GravitationalForce(accel_var="a", param_g="g")
        result = mod.compute({}, {"g": 9.81}, 0.0)
        assert result["a"] == pytest.approx(-9.81)

    def test_default_g(self):
        mod = GravitationalForce()
        result = mod.compute({}, {}, 0.0)
        assert result["a_y"] == pytest.approx(-9.81)


class TestPendulumForce:
    """Tests for the PendulumForce module."""

    def test_variables(self):
        mod = PendulumForce(angle_var="th", accel_var="a")
        assert mod.variables() == ["a"]

    def test_at_zero_angle(self):
        mod = PendulumForce(angle_var="th", accel_var="a", param_g="g", param_L="L")
        result = mod.compute({"th": 0.0}, {"g": 9.81, "L": 1.0}, 0.0)
        assert result["a"] == pytest.approx(0.0)

    def test_at_small_angle(self):
        mod = PendulumForce(angle_var="th", accel_var="a", param_g="g", param_L="L")
        theta = 0.1
        result = mod.compute({"th": theta}, {"g": 9.81, "L": 2.0}, 0.0)
        expected = -9.81 / 2.0 * np.sin(theta)
        assert result["a"] == pytest.approx(expected)


class TestLinearDamping:
    """Tests for the LinearDamping module."""

    def test_variables(self):
        mod = LinearDamping(vel_var="v", accel_var="a")
        assert mod.variables() == ["a"]

    def test_compute(self):
        mod = LinearDamping(vel_var="v", accel_var="a", param_c="c")
        result = mod.compute({"v": 5.0}, {"c": 2.0}, 0.0)
        assert result["a"] == pytest.approx(-10.0)

    def test_default_zero_damping(self):
        mod = LinearDamping(vel_var="v", accel_var="a", param_c="c")
        result = mod.compute({"v": 5.0}, {}, 0.0)
        assert result["a"] == pytest.approx(0.0)


class TestVanDerPolDamping:
    """Tests for the VanDerPolDamping module."""

    def test_variables(self):
        mod = VanDerPolDamping(pos_var="x", vel_var="v", accel_var="a")
        assert mod.variables() == ["a"]

    def test_at_origin(self):
        """At x=0: mu*(1 - 0)*v = mu*v."""
        mod = VanDerPolDamping(pos_var="x", vel_var="v", accel_var="a", param_mu="mu")
        result = mod.compute({"x": 0.0, "v": 1.0}, {"mu": 2.0}, 0.0)
        assert result["a"] == pytest.approx(2.0)

    def test_at_unit_displacement(self):
        """At x=1: mu*(1 - 1)*v = 0."""
        mod = VanDerPolDamping(pos_var="x", vel_var="v", accel_var="a", param_mu="mu")
        result = mod.compute({"x": 1.0, "v": 1.0}, {"mu": 2.0}, 0.0)
        assert result["a"] == pytest.approx(0.0)

    def test_negative_damping_inside(self):
        """For |x| < 1 the damping is negative (energy injection)."""
        mod = VanDerPolDamping(pos_var="x", vel_var="v", accel_var="a", param_mu="mu")
        result = mod.compute({"x": 0.5, "v": 1.0}, {"mu": 1.0}, 0.0)
        assert result["a"] > 0  # positive contribution when v > 0

    def test_positive_damping_outside(self):
        """For |x| > 1 the damping is positive (energy dissipation)."""
        mod = VanDerPolDamping(pos_var="x", vel_var="v", accel_var="a", param_mu="mu")
        result = mod.compute({"x": 2.0, "v": 1.0}, {"mu": 1.0}, 0.0)
        assert result["a"] < 0  # negative contribution when v > 0


class TestNewtonianDynamics:
    """Tests for the NewtonianDynamics module."""

    def test_variables(self):
        mod = NewtonianDynamics(pos_var="x", vel_var="v")
        assert "x" in mod.variables()
        assert "v" in mod.variables()

    def test_compute(self):
        mod = NewtonianDynamics(pos_var="x", vel_var="v", accel_var="a", param_m="m")
        result = mod.compute({"x": 0.0, "v": 3.0, "a": 6.0}, {"m": 2.0}, 0.0)
        assert result["x"] == pytest.approx(3.0)  # dx/dt = v
        assert result["v"] == pytest.approx(3.0)  # dv/dt = a/m = 6/2

    def test_default_mass(self):
        """Default mass should be 1."""
        mod = NewtonianDynamics(pos_var="x", vel_var="v", accel_var="a", param_m="m")
        result = mod.compute({"x": 0.0, "v": 0.0, "a": 5.0}, {}, 0.0)
        assert result["v"] == pytest.approx(5.0)

    def test_zero_accel_when_missing(self):
        """If accel_var is missing from state, should default to 0."""
        mod = NewtonianDynamics(pos_var="x", vel_var="v", accel_var="a", param_m="m")
        result = mod.compute({"x": 0.0, "v": 2.0}, {"m": 1.0}, 0.0)
        assert result["x"] == pytest.approx(2.0)
        assert result["v"] == pytest.approx(0.0)


class TestLogisticGrowth:
    """Tests for the LogisticGrowth module."""

    def test_variables(self):
        mod = LogisticGrowth(var="N")
        assert mod.variables() == ["N"]

    def test_compute_half_capacity(self):
        mod = LogisticGrowth(var="N", param_r="r", param_K="K")
        result = mod.compute({"N": 50.0}, {"r": 0.5, "K": 100.0}, 0.0)
        assert result["N"] == pytest.approx(12.5)  # 0.5*50*(1-50/100)

    def test_at_capacity(self):
        """At carrying capacity, growth rate should be zero."""
        mod = LogisticGrowth(var="N", param_r="r", param_K="K")
        result = mod.compute({"N": 100.0}, {"r": 1.0, "K": 100.0}, 0.0)
        assert result["N"] == pytest.approx(0.0)


class TestPredatorPreyInteraction:
    """Tests for the PredatorPreyInteraction module."""

    def test_variables(self):
        mod = PredatorPreyInteraction()
        assert "N" in mod.variables()
        assert "P" in mod.variables()

    def test_compute(self):
        mod = PredatorPreyInteraction(
            prey_var="N", pred_var="P",
            param_beta="b", param_delta="d", param_gamma="g",
        )
        result = mod.compute({"N": 10.0, "P": 5.0}, {"b": 0.4, "d": 0.1, "g": 0.3}, 0.0)
        assert result["N"] == pytest.approx(-20.0)  # -0.4*10*5
        assert result["P"] == pytest.approx(3.5)  # 0.1*10*5 - 0.3*5


class TestSIRDynamics:
    """Tests for the SIRDynamics module."""

    def test_variables(self):
        mod = SIRDynamics()
        assert mod.variables() == ["S", "I", "R"]

    def test_compute_conservation(self):
        """dS + dI + dR should sum to zero."""
        mod = SIRDynamics(param_beta="b", param_gamma="g", param_N="N")
        state = {"S": 990.0, "I": 10.0, "R": 0.0}
        result = mod.compute(state, {"b": 0.3, "g": 0.1, "N": 1000.0}, 0.0)
        total = result["S"] + result["I"] + result["R"]
        assert total == pytest.approx(0.0, abs=1e-10)

    def test_no_infection_without_infected(self):
        """If I=0, dS should be zero."""
        mod = SIRDynamics(param_beta="b", param_gamma="g", param_N="N")
        state = {"S": 1000.0, "I": 0.0, "R": 0.0}
        result = mod.compute(state, {"b": 0.3, "g": 0.1, "N": 1000.0}, 0.0)
        assert result["S"] == pytest.approx(0.0)
        assert result["I"] == pytest.approx(0.0)
        assert result["R"] == pytest.approx(0.0)

    def test_compute_values(self):
        mod = SIRDynamics(param_beta="b", param_gamma="g", param_N="N")
        state = {"S": 990.0, "I": 10.0, "R": 0.0}
        result = mod.compute(state, {"b": 0.3, "g": 0.1, "N": 1000.0}, 0.0)
        infection = 0.3 * 990.0 * 10.0 / 1000.0  # 2.97
        recovery = 0.1 * 10.0  # 1.0
        assert result["S"] == pytest.approx(-infection)
        assert result["I"] == pytest.approx(infection - recovery)
        assert result["R"] == pytest.approx(recovery)


class TestBrusselatorKinetics:
    """Tests for the BrusselatorKinetics module."""

    def test_variables(self):
        mod = BrusselatorKinetics()
        assert "u" in mod.variables()
        assert "v" in mod.variables()

    def test_steady_state_rates_zero(self):
        """At steady state (a, b/a) both rates should vanish."""
        mod = BrusselatorKinetics(param_a="a", param_b="b")
        result = mod.compute({"u": 1.0, "v": 3.0}, {"a": 1.0, "b": 3.0}, 0.0)
        assert result["u"] == pytest.approx(0.0, abs=1e-10)
        assert result["v"] == pytest.approx(0.0, abs=1e-10)

    def test_compute_off_steady_state(self):
        mod = BrusselatorKinetics(param_a="a", param_b="b")
        # du/dt = 1.0 - (3+1)*2.0 + (2.0)^2*1.5 = 1 - 8 + 6 = -1
        # dv/dt = 3.0*2.0 - (2.0)^2*1.5 = 6 - 6 = 0
        result = mod.compute({"u": 2.0, "v": 1.5}, {"a": 1.0, "b": 3.0}, 0.0)
        assert result["u"] == pytest.approx(-1.0)
        assert result["v"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. ComposedSimulation.from_modules() factory
# ---------------------------------------------------------------------------

class TestFromModulesFactory:
    """Tests for ComposedSimulation.from_modules()."""

    def test_basic_construction(self):
        sim = ComposedSimulation.from_modules(
            modules=[LogisticGrowth(var="N", param_r="r", param_K="K")],
            defaults={"r": 0.5, "K": 100.0, "N_0": 10.0},
        )
        assert isinstance(sim, ComposedSimulation)

    def test_initial_values_from_defaults_suffix(self):
        """Keys ending in _0 in defaults become initial values."""
        sim = ComposedSimulation.from_modules(
            modules=[LogisticGrowth(var="N", param_r="r", param_K="K")],
            defaults={"r": 0.5, "K": 100.0, "N_0": 25.0},
        )
        state = sim.reset()
        assert state[0] == pytest.approx(25.0)

    def test_explicit_initial_values_override(self):
        """Explicit initial_values should override _0 defaults."""
        sim = ComposedSimulation.from_modules(
            modules=[LogisticGrowth(var="N", param_r="r", param_K="K")],
            defaults={"r": 0.5, "K": 100.0, "N_0": 25.0},
            initial_values={"N": 50.0},
        )
        state = sim.reset()
        assert state[0] == pytest.approx(50.0)

    def test_accel_vars_excluded_from_state(self):
        """Intermediate acceleration variables should not appear in state."""
        sim = make_harmonic_oscillator()
        names = sim.get_variable_names()
        assert "a_x" not in names
        assert len(names) == 2

    def test_domain_is_custom(self):
        sim = make_harmonic_oscillator()
        assert sim.config.domain == Domain.CUSTOM

    def test_parameters_exclude_initial_values(self):
        """_0 keys should not end up in config.parameters."""
        sim = ComposedSimulation.from_modules(
            modules=[LogisticGrowth(var="N", param_r="r", param_K="K")],
            defaults={"r": 0.5, "K": 100.0, "N_0": 25.0},
        )
        assert "N_0" not in sim.config.parameters
        assert "r" in sim.config.parameters
        assert "K" in sim.config.parameters

    def test_module_list_is_copied(self):
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


# ---------------------------------------------------------------------------
# 3. Variable names and module names accessors
# ---------------------------------------------------------------------------

class TestAccessors:
    """Tests for get_variable_names() and get_module_names()."""

    def test_variable_names_harmonic(self):
        sim = make_harmonic_oscillator()
        names = sim.get_variable_names()
        assert "x" in names
        assert "v" in names

    def test_variable_names_sir(self):
        sim = make_sir()
        names = sim.get_variable_names()
        assert names == ["S", "I", "R"]

    def test_variable_names_pendulum(self):
        sim = make_pendulum()
        names = sim.get_variable_names()
        assert "theta" in names
        assert "omega" in names

    def test_module_names_harmonic(self):
        sim = make_harmonic_oscillator()
        names = sim.get_module_names()
        assert "HarmonicForce" in names
        assert "LinearDamping" in names
        assert "NewtonianDynamics" in names

    def test_module_names_duffing(self):
        sim = make_duffing_oscillator()
        names = sim.get_module_names()
        assert "CubicForce" in names
        assert "PeriodicForcing" in names

    def test_module_names_vdp(self):
        sim = make_van_der_pol()
        names = sim.get_module_names()
        assert "VanDerPolDamping" in names

    def test_module_names_lotka_volterra(self):
        sim = make_lotka_volterra()
        names = sim.get_module_names()
        assert "LogisticGrowth" in names
        assert "PredatorPreyInteraction" in names


# ---------------------------------------------------------------------------
# 4. Reset/step/observe interface
# ---------------------------------------------------------------------------

class TestSimulationInterface:
    """Tests for the reset/step/observe interface."""

    def test_reset_returns_initial_state(self):
        sim = make_harmonic_oscillator(x_0=3.0, v_0=-1.0)
        state = sim.reset()
        assert state[0] == pytest.approx(3.0)
        assert state[1] == pytest.approx(-1.0)

    def test_step_changes_state(self):
        sim = make_harmonic_oscillator()
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_matches_step(self):
        sim = make_harmonic_oscillator()
        sim.reset()
        step_result = sim.step()
        obs_result = sim.observe()
        np.testing.assert_array_equal(step_result, obs_result)

    def test_reset_restores_initial_after_stepping(self):
        sim = make_harmonic_oscillator(x_0=1.0, v_0=0.0)
        sim.reset()
        for _ in range(50):
            sim.step()
        state = sim.reset()
        assert state[0] == pytest.approx(1.0)
        assert state[1] == pytest.approx(0.0)

    def test_step_deterministic(self):
        sim = make_harmonic_oscillator(k=4.0, m=1.0, c=0.0, x_0=1.0, v_0=0.0)
        sim.reset()
        s1 = sim.step().copy()
        sim.reset()
        s2 = sim.step().copy()
        np.testing.assert_array_equal(s1, s2)

    def test_run_trajectory_shape(self):
        sim = make_harmonic_oscillator(n_steps=100)
        traj = sim.run(100)
        assert traj.states.shape == (101, 2)  # initial + 100 steps
        assert len(traj.timestamps) == 101


# ---------------------------------------------------------------------------
# 5. Recipe: no-NaN for 100+ steps
# ---------------------------------------------------------------------------

class TestRecipesNoNaN:
    """Each recipe runs without NaN for at least 200 steps."""

    def test_harmonic_oscillator_no_nan(self):
        sim = make_harmonic_oscillator()
        sim.reset()
        for _ in range(200):
            state = sim.step()
            assert np.all(np.isfinite(state)), "NaN in harmonic oscillator"

    def test_duffing_oscillator_no_nan(self):
        sim = make_duffing_oscillator()
        sim.reset()
        for _ in range(200):
            state = sim.step()
            assert np.all(np.isfinite(state)), "NaN in Duffing"

    def test_van_der_pol_no_nan(self):
        sim = make_van_der_pol()
        sim.reset()
        for _ in range(200):
            state = sim.step()
            assert np.all(np.isfinite(state)), "NaN in VdP"

    def test_pendulum_no_nan(self):
        sim = make_pendulum()
        sim.reset()
        for _ in range(200):
            state = sim.step()
            assert np.all(np.isfinite(state)), "NaN in pendulum"

    def test_lotka_volterra_no_nan(self):
        sim = make_lotka_volterra()
        sim.reset()
        for _ in range(200):
            state = sim.step()
            assert np.all(np.isfinite(state)), "NaN in LV"

    def test_sir_no_nan(self):
        sim = make_sir()
        sim.reset()
        for _ in range(200):
            state = sim.step()
            assert np.all(np.isfinite(state)), "NaN in SIR"


# ---------------------------------------------------------------------------
# 6. Harmonic oscillator energy conservation
# ---------------------------------------------------------------------------

class TestHarmonicEnergyConservation:
    """Undamped harmonic oscillator should approximately conserve energy."""

    def test_energy_conserved_short_run(self):
        k, m = 4.0, 1.0
        sim = make_harmonic_oscillator(k=k, m=m, c=0.0, x_0=1.0, v_0=0.0, dt=0.001)
        sim.reset()
        s = sim.observe()
        E0 = 0.5 * k * s[0] ** 2 + 0.5 * m * s[1] ** 2

        for _ in range(5000):
            sim.step()

        s = sim.observe()
        E_final = 0.5 * k * s[0] ** 2 + 0.5 * m * s[1] ** 2
        assert E_final == pytest.approx(E0, rel=1e-6)

    def test_energy_conserved_long_run(self):
        k, m = 9.0, 2.0
        sim = make_harmonic_oscillator(k=k, m=m, c=0.0, x_0=2.0, v_0=1.0, dt=0.001)
        sim.reset()
        s = sim.observe()
        E0 = 0.5 * k * s[0] ** 2 + 0.5 * m * s[1] ** 2

        for _ in range(20000):
            sim.step()

        s = sim.observe()
        E_final = 0.5 * k * s[0] ** 2 + 0.5 * m * s[1] ** 2
        rel_drift = abs(E_final - E0) / E0
        assert rel_drift < 1e-5, f"Energy drift {rel_drift:.2e}"


# ---------------------------------------------------------------------------
# 7. Composed vs monolithic agreement
# ---------------------------------------------------------------------------

class TestComposedVsMonolithic:
    """Composed simulations should agree with monolithic implementations."""

    def test_harmonic_frequency_matches_omega(self):
        """Oscillation frequency should match omega = sqrt(k/m)."""
        k, m = 9.0, 1.0
        sim = make_harmonic_oscillator(
            k=k, m=m, c=0.0, x_0=1.0, v_0=0.0, dt=0.001,
        )
        sim.reset()
        positions = []
        for _ in range(10000):
            state = sim.step()
            positions.append(state[0])

        positions = np.array(positions)
        crossings = np.where(np.diff(np.sign(positions)))[0]
        assert len(crossings) >= 4, f"Too few crossings: {len(crossings)}"
        half_periods = np.diff(crossings) * 0.001
        period = 2 * np.mean(half_periods)
        omega_measured = 2 * np.pi / period
        omega_theory = np.sqrt(k / m)
        assert omega_measured == pytest.approx(omega_theory, rel=0.02)

    def test_duffing_matches_monolithic(self):
        """Composed Duffing should produce similar results to monolithic."""
        from simulating_anything.simulation.duffing import DuffingOscillator

        dt = 0.005
        n_steps = 500
        params = {
            "alpha": 1.0, "beta": 1.0, "delta": 0.2,
            "gamma_f": 0.3, "omega": 1.0, "x_0": 0.5, "v_0": 0.0,
        }

        composed = make_duffing_oscillator(**params, dt=dt, n_steps=n_steps)
        composed.reset()
        for _ in range(n_steps):
            c_state = composed.step()

        mono_config = SimulationConfig(
            domain=Domain.DUFFING, parameters=params, dt=dt, n_steps=n_steps,
        )
        mono = DuffingOscillator(mono_config)
        mono.reset()
        for _ in range(n_steps):
            m_state = mono.step()

        # Should be close (not exact due to possible implementation detail differences)
        assert c_state[0] == pytest.approx(m_state[0], abs=0.5)

    def test_damped_oscillator_amplitude_decays(self):
        """With damping, amplitude should decrease over time."""
        sim = make_harmonic_oscillator(
            k=4.0, m=1.0, c=0.5, x_0=2.0, v_0=0.0, dt=0.01,
        )
        sim.reset()

        early_max = 0.0
        for _ in range(200):
            state = sim.step()
            early_max = max(early_max, abs(state[0]))

        for _ in range(2000):
            sim.step()

        late_max = 0.0
        for _ in range(200):
            state = sim.step()
            late_max = max(late_max, abs(state[0]))

        assert late_max < early_max


# ---------------------------------------------------------------------------
# 8. VdP limit cycle amplitude ~2
# ---------------------------------------------------------------------------

class TestVanDerPolLimitCycle:
    """Van der Pol oscillator should settle onto a limit cycle with amplitude ~2."""

    def test_limit_cycle_amplitude_mu_1(self):
        sim = make_van_der_pol(mu=1.0, x_0=0.1, v_0=0.0, dt=0.005, n_steps=50000)
        sim.reset()
        for _ in range(20000):
            sim.step()
        x_max = 0.0
        for _ in range(10000):
            state = sim.step()
            x_max = max(x_max, abs(state[0]))
        assert x_max == pytest.approx(2.0, abs=0.3)

    def test_limit_cycle_from_outside(self):
        """Starting outside the limit cycle should also converge to amplitude ~2."""
        sim = make_van_der_pol(mu=1.0, x_0=5.0, v_0=0.0, dt=0.005, n_steps=50000)
        sim.reset()
        for _ in range(20000):
            sim.step()
        x_max = 0.0
        for _ in range(10000):
            state = sim.step()
            x_max = max(x_max, abs(state[0]))
        assert x_max == pytest.approx(2.0, abs=0.4)

    def test_bounded_trajectory(self):
        sim = make_van_der_pol(mu=1.0, dt=0.005)
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(np.abs(state) < 30), f"VdP diverged: {state}"


# ---------------------------------------------------------------------------
# 9. SIR conservation S+I+R=const
# ---------------------------------------------------------------------------

class TestSIRConservation:
    """SIR model should conserve total population."""

    def test_conservation_throughout(self):
        N_pop = 1000.0
        sim = make_sir(
            beta=0.3, gamma=0.1, N_pop=N_pop,
            S_0=990.0, I_0=10.0, R_0=0.0, dt=0.1,
        )
        sim.reset()
        for _ in range(500):
            state = sim.step()
            total = state[0] + state[1] + state[2]
            assert total == pytest.approx(N_pop, rel=1e-6)

    def test_epidemic_peak_with_r0_gt_1(self):
        """With R0>1, I should rise then fall."""
        sim = make_sir(
            beta=0.3, gamma=0.1, S_0=990.0, I_0=10.0, R_0=0.0, dt=0.1,
        )
        sim.reset()
        infected = []
        for _ in range(2000):
            state = sim.step()
            infected.append(state[1])

        infected = np.array(infected)
        peak_idx = np.argmax(infected)
        assert 10 < peak_idx < len(infected) - 10
        assert infected[-1] < infected[peak_idx]

    def test_no_epidemic_with_r0_lt_1(self):
        """With R0<1, I should decline monotonically."""
        sim = make_sir(
            beta=0.05, gamma=0.1,  # R0 = 0.5
            S_0=990.0, I_0=10.0, R_0=0.0, dt=0.1,
        )
        sim.reset()
        max_infected = 0.0
        for _ in range(1000):
            state = sim.step()
            max_infected = max(max_infected, state[1])
        # Should stay near or below initial infected
        assert max_infected < 15.0


# ---------------------------------------------------------------------------
# 10. Advanced composition tests
# ---------------------------------------------------------------------------

class TestAdvancedComposition:
    """Advanced tests for composition mechanics."""

    def test_additive_zero_force_identity(self):
        """Adding a zero-strength force should not change dynamics."""
        sim1 = make_harmonic_oscillator(
            k=4.0, c=0.0, x_0=1.0, v_0=0.0, dt=0.01, n_steps=100,
        )
        sim1.reset()

        sim2 = ComposedSimulation.from_modules(
            modules=[
                HarmonicForce(var="x", accel_var="a_x", param_k="k"),
                PeriodicForcing(
                    accel_var="a_x", param_gamma="gamma_f", param_omega="omega",
                ),
                LinearDamping(vel_var="v", accel_var="a_x", param_c="c"),
                NewtonianDynamics(
                    pos_var="x", vel_var="v", accel_var="a_x", param_m="m",
                ),
            ],
            defaults={
                "k": 4.0, "m": 1.0, "c": 0.0,
                "gamma_f": 0.0, "omega": 1.0,
                "x_0": 1.0, "v_0": 0.0,
            },
            accel_vars={"a_x"},
            dt=0.01, n_steps=100,
        )
        sim2.reset()

        for _ in range(100):
            s1 = sim1.step().copy()
            s2 = sim2.step().copy()
        np.testing.assert_array_almost_equal(s1, s2, decimal=10)

    def test_brusselator_steady_state_convergence(self):
        """Brusselator with b < 1+a^2 should converge to steady state (a, b/a)."""
        a_val, b_val = 1.0, 1.5  # b < 1 + a^2 = 2
        sim = ComposedSimulation.from_modules(
            modules=[BrusselatorKinetics(param_a="a", param_b="b")],
            defaults={"a": a_val, "b": b_val, "u_0": a_val, "v_0": b_val / a_val},
            dt=0.01,
        )
        state = sim.reset()
        for _ in range(5000):
            state = sim.step()
        assert state[0] == pytest.approx(a_val, abs=0.05)
        assert state[1] == pytest.approx(b_val / a_val, abs=0.05)

    def test_brusselator_oscillation_above_hopf(self):
        """Brusselator with b > 1+a^2 should exhibit oscillation."""
        a_val, b_val = 1.0, 3.0  # b > 1 + a^2 = 2
        sim = ComposedSimulation.from_modules(
            modules=[BrusselatorKinetics(param_a="a", param_b="b")],
            defaults={"a": a_val, "b": b_val, "u_0": 1.1, "v_0": 3.1},
            dt=0.005,
        )
        sim.reset()
        for _ in range(5000):
            sim.step()

        u_values = []
        for _ in range(3000):
            state = sim.step()
            u_values.append(state[0])

        amplitude = max(u_values) - min(u_values)
        assert amplitude > 0.1

    def test_logistic_growth_saturation(self):
        """Pure logistic growth should saturate at K."""
        sim = ComposedSimulation.from_modules(
            modules=[LogisticGrowth(var="N", param_r="r", param_K="K")],
            defaults={"r": 1.0, "K": 100.0, "N_0": 1.0},
            dt=0.01,
        )
        sim.reset()
        for _ in range(5000):
            state = sim.step()
        assert state[0] == pytest.approx(100.0, rel=0.01)

    def test_lotka_volterra_populations_positive(self):
        """LV populations should remain positive."""
        sim = make_lotka_volterra(N_0=10.0, P_0=5.0, dt=0.005)
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert state[0] > 0, f"Prey went non-positive: {state[0]}"
            assert state[1] > 0, f"Predator went non-positive: {state[1]}"

    def test_lotka_volterra_oscillatory(self):
        """LV should produce oscillatory dynamics."""
        sim = make_lotka_volterra(dt=0.01)
        sim.reset()
        prey_values = []
        for _ in range(5000):
            state = sim.step()
            prey_values.append(state[0])

        diffs = np.diff(prey_values)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        assert sign_changes >= 4

    def test_pendulum_small_angle_period(self):
        """Small-angle pendulum period should match T = 2*pi*sqrt(L/g)."""
        g, L = 9.81, 1.0
        sim = make_pendulum(
            g=g, L=L, c=0.0, theta_0=0.1, omega_0=0.0,
            dt=0.001, n_steps=50000,
        )
        sim.reset()

        prev_theta = sim.observe()[0]
        crossings = []
        for step_i in range(10000):
            state = sim.step()
            if prev_theta < 0 and state[0] >= 0:
                frac = -prev_theta / (state[0] - prev_theta)
                crossings.append((step_i + frac) * 0.001)
            prev_theta = state[0]

        assert len(crossings) >= 2, f"Too few crossings: {len(crossings)}"
        periods = np.diff(crossings)
        mean_period = np.mean(periods)
        expected = 2 * np.pi * np.sqrt(L / g)
        assert mean_period == pytest.approx(expected, rel=0.02)

    def test_deterministic_across_resets(self):
        """Same initial conditions should produce identical trajectories."""
        sim = make_harmonic_oscillator(x_0=1.0, v_0=0.5, dt=0.01)

        sim.reset()
        states_1 = [sim.step().copy() for _ in range(50)]

        sim.reset()
        states_2 = [sim.step().copy() for _ in range(50)]

        for s1, s2 in zip(states_1, states_2):
            np.testing.assert_array_equal(s1, s2)

    def test_duffing_bounded(self):
        """Duffing trajectory should remain bounded."""
        sim = make_duffing_oscillator()
        sim.reset()
        for _ in range(2000):
            state = sim.step()
            assert np.all(np.abs(state) < 100), f"Duffing diverged: {state}"
