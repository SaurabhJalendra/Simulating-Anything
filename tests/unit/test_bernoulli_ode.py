"""Tests for the Bernoulli ODE simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.bernoulli_ode import BernoulliODESimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    a: float = 1.0,
    b: float = -1.0,
    n: float = 2.0,
    y_0: float = 0.1,
    dt: float = 0.01,
    n_steps: int = 500,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.BERNOULLI_ODE,
        dt=dt,
        n_steps=n_steps,
        parameters={"a": a, "b": b, "n": n, "y_0": y_0},
    )


# ---------------------------------------------------------------------------
# Basic construction and interface
# ---------------------------------------------------------------------------


class TestBernoulliBasic:
    def test_default_parameters(self):
        sim = BernoulliODESimulation(_make_config())
        assert sim.a == 1.0
        assert sim.b == -1.0
        assert sim.n == 2.0
        assert sim.y_0 == 0.1

    def test_custom_parameters(self):
        sim = BernoulliODESimulation(_make_config(a=2.0, b=-0.5, n=3.0, y_0=0.8))
        assert sim.a == 2.0
        assert sim.b == -0.5
        assert sim.n == 3.0
        assert sim.y_0 == 0.8

    def test_empty_parameters_use_defaults(self):
        config = SimulationConfig(
            domain=Domain.BERNOULLI_ODE, dt=0.01, n_steps=10, parameters={},
        )
        sim = BernoulliODESimulation(config)
        assert sim.a == 1.0
        assert sim.b == -1.0
        assert sim.n == 2.0
        assert sim.y_0 == 0.1

    def test_reset_returns_initial_state(self):
        sim = BernoulliODESimulation(_make_config(y_0=0.7))
        state = sim.reset()
        assert state.shape == (1,)
        assert state[0] == pytest.approx(0.7)

    def test_reset_clears_step_count(self):
        sim = BernoulliODESimulation(_make_config())
        sim.reset()
        sim.step()
        sim.step()
        sim.reset()
        assert sim._step_count == 0

    def test_step_advances_state(self):
        sim = BernoulliODESimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_step_increments_step_count(self):
        sim = BernoulliODESimulation(_make_config())
        sim.reset()
        sim.step()
        assert sim._step_count == 1
        sim.step()
        assert sim._step_count == 2

    def test_observe_returns_state(self):
        sim = BernoulliODESimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (1,)
        assert obs.dtype == np.float64

    def test_run_returns_trajectory_data(self):
        sim = BernoulliODESimulation(_make_config())
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 1)
        assert len(traj.timestamps) == 101

    def test_run_timestamps_correct(self):
        sim = BernoulliODESimulation(_make_config(dt=0.01))
        traj = sim.run(n_steps=50)
        np.testing.assert_allclose(
            traj.timestamps, np.arange(51) * 0.01, atol=1e-12,
        )

    def test_state_is_float64(self):
        sim = BernoulliODESimulation(_make_config())
        state = sim.reset()
        assert state.dtype == np.float64

    def test_dt_used_correctly(self):
        """Smaller dt should give more accurate results."""
        # a=1.0, b=-1.0, n=2, y_0=0.1 -> stable, converges to y_ss=1.0
        config_coarse = _make_config(dt=0.1, n_steps=50)
        config_fine = _make_config(dt=0.001, n_steps=5000)

        sim_c = BernoulliODESimulation(config_coarse)
        sim_f = BernoulliODESimulation(config_fine)

        sim_c.reset()
        for _ in range(50):
            sim_c.step()

        sim_f.reset()
        for _ in range(5000):
            sim_f.step()

        y_exact = float(sim_f.analytical_solution(5.0))
        err_c = abs(sim_c.observe()[0] - y_exact)
        err_f = abs(sim_f.observe()[0] - y_exact)
        assert err_f < err_c


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


class TestBernoulliDynamics:
    def test_derivatives_at_known_point(self):
        """dy/dt = a*y + b*y^n at y=1: dy/dt = 1.0 + (-1.0)*1 = 0.0 (steady state)."""
        sim = BernoulliODESimulation(_make_config(a=1.0, b=-1.0, n=2.0))
        dy = sim._derivatives(np.array([1.0]))
        assert dy[0] == pytest.approx(0.0)

    def test_derivatives_at_zero(self):
        """At y=0, dy/dt = 0 (trivial steady state)."""
        sim = BernoulliODESimulation(_make_config(a=1.0, b=-1.0, n=2.0))
        dy = sim._derivatives(np.array([0.0]))
        assert dy[0] == pytest.approx(0.0)

    def test_derivatives_positive_below_steady_state(self):
        """For a > 0, b < 0, n=2: below y_ss, dy/dt > 0 (growing)."""
        sim = BernoulliODESimulation(_make_config(a=1.0, b=-1.0, n=2.0))
        dy = sim._derivatives(np.array([0.5]))
        # dy/dt = 1.0*0.5 + (-1.0)*0.25 = 0.5 - 0.25 = 0.25 > 0
        assert dy[0] == pytest.approx(0.25)
        assert dy[0] > 0

    def test_derivatives_negative_above_steady_state(self):
        """For a > 0, b < 0, n=2: above y_ss, dy/dt < 0 (decaying)."""
        sim = BernoulliODESimulation(_make_config(a=1.0, b=-1.0, n=2.0))
        dy = sim._derivatives(np.array([2.0]))
        # dy/dt = 1.0*2.0 + (-1.0)*4.0 = 2.0 - 4.0 = -2.0 < 0
        assert dy[0] == pytest.approx(-2.0)
        assert dy[0] < 0

    def test_derivatives_at_steady_state(self):
        """At y_ss = -a/b (n=2), dy/dt should be zero."""
        a, b = 2.0, -1.0
        y_ss = -a / b  # = 2.0
        sim = BernoulliODESimulation(_make_config(a=a, b=b, n=2.0))
        dy = sim._derivatives(np.array([y_ss]))
        assert dy[0] == pytest.approx(0.0, abs=1e-12)

    def test_converges_to_steady_state_n2(self):
        """For a > 0, b < 0, n=2: solution should converge to y_ss = -a/b."""
        a, b = 2.0, -1.0
        y_ss = -a / b  # = 2.0
        sim = BernoulliODESimulation(
            _make_config(a=a, b=b, n=2.0, y_0=0.1, dt=0.01, n_steps=2000)
        )
        sim.reset()
        for _ in range(2000):
            sim.step()
        assert sim.observe()[0] == pytest.approx(y_ss, abs=0.01)

    def test_converges_to_steady_state_n3(self):
        """For a > 0, b < 0, n=3: y_ss = (-a/b)^(1/2) = sqrt(a/(-b))."""
        a, b = 2.0, -1.0
        y_ss = np.sqrt(-a / b)  # = sqrt(2) ~ 1.414
        sim = BernoulliODESimulation(
            _make_config(a=a, b=b, n=3.0, y_0=0.5, dt=0.005, n_steps=5000)
        )
        sim.reset()
        for _ in range(5000):
            sim.step()
        assert sim.observe()[0] == pytest.approx(y_ss, abs=0.02)

    def test_solution_bounded_stable_case(self):
        """For a > 0, b < 0, solution should remain bounded."""
        sim = BernoulliODESimulation(
            _make_config(a=1.0, b=-1.0, n=2.0, y_0=0.1, dt=0.01, n_steps=5000)
        )
        sim.reset()
        for _ in range(5000):
            sim.step()
            y = sim.observe()[0]
            assert abs(y) < 100, f"y diverged: {y}"

    def test_different_ic_converge_to_same_steady_state(self):
        """Different ICs should converge to same y_ss (for a > 0, b < 0)."""
        a, b = 1.0, -1.0
        y_ss = -a / b  # = 1.0

        finals = []
        for y_0 in [0.01, 0.1, 0.5, 2.0, 5.0]:
            sim = BernoulliODESimulation(
                _make_config(a=a, b=b, n=2.0, y_0=y_0, dt=0.01, n_steps=3000)
            )
            sim.reset()
            for _ in range(3000):
                sim.step()
            finals.append(sim.observe()[0])

        for y_f in finals:
            assert y_f == pytest.approx(y_ss, abs=0.05)

    def test_monotone_approach_from_below(self):
        """Starting below y_ss with a > 0, b < 0: solution increases monotonically."""
        a, b = 1.0, -1.0
        sim = BernoulliODESimulation(
            _make_config(a=a, b=b, n=2.0, y_0=0.1, dt=0.01)
        )
        sim.reset()
        prev = sim.observe()[0]
        for _ in range(200):
            sim.step()
            curr = sim.observe()[0]
            assert curr >= prev - 1e-10, "Solution should increase toward y_ss"
            prev = curr

    def test_monotone_approach_from_above(self):
        """Starting above y_ss with a > 0, b < 0: solution decreases monotonically."""
        a, b = 1.0, -1.0
        sim = BernoulliODESimulation(
            _make_config(a=a, b=b, n=2.0, y_0=3.0, dt=0.01)
        )
        sim.reset()
        prev = sim.observe()[0]
        for _ in range(200):
            sim.step()
            curr = sim.observe()[0]
            assert curr <= prev + 1e-10, "Solution should decrease toward y_ss"
            prev = curr


# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------


class TestBernoulliAnalytical:
    def test_analytical_at_t0(self):
        """At t=0, analytical solution should equal y_0."""
        sim = BernoulliODESimulation(_make_config(y_0=0.7))
        y_0_computed = sim.analytical_solution(0.0)
        assert float(y_0_computed) == pytest.approx(0.7, abs=1e-10)

    def test_analytical_matches_numerical_n2(self):
        """Analytical and numerical should agree for n=2 stable case."""
        sim = BernoulliODESimulation(
            _make_config(a=1.0, b=-1.0, n=2.0, y_0=0.1, dt=0.001, n_steps=5000)
        )
        sim.reset()
        for _ in range(5000):
            sim.step()

        t_final = 5.0
        y_num = sim.observe()[0]
        y_ana = float(sim.analytical_solution(t_final))
        assert abs(y_num - y_ana) < 1e-4, (
            f"Numerical={y_num:.6f}, Analytical={y_ana:.6f}"
        )

    def test_analytical_matches_numerical_n3(self):
        """Analytical and numerical should agree for n=3 stable case."""
        sim = BernoulliODESimulation(
            _make_config(a=1.0, b=-0.5, n=3.0, y_0=0.8, dt=0.001, n_steps=3000)
        )
        sim.reset()
        for _ in range(3000):
            sim.step()

        t_final = 3.0
        y_num = sim.observe()[0]
        y_ana = float(sim.analytical_solution(t_final))
        assert np.isfinite(y_ana)
        assert abs(y_num - y_ana) < 0.01, (
            f"Numerical={y_num:.6f}, Analytical={y_ana:.6f}"
        )

    def test_analytical_long_time_steady_state(self):
        """At large t, analytical should approach y_ss = -a/b for n=2."""
        sim = BernoulliODESimulation(_make_config(a=1.0, b=-2.0, n=2.0, y_0=0.1))
        y_long = sim.analytical_solution(100.0)
        y_ss = -1.0 / (-2.0)  # = 0.5
        assert float(y_long) == pytest.approx(y_ss, abs=1e-6)

    def test_analytical_vectorized(self):
        """analytical_solution should accept array input."""
        sim = BernoulliODESimulation(_make_config(a=1.0, b=-1.0, n=2.0, y_0=0.1))
        times = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
        result = sim.analytical_solution(times)
        assert result.shape == (5,)
        assert all(np.isfinite(result))

    def test_analytical_a_zero_special_case(self):
        """For a=0, dy/dt = b*y^n. The substitution gives dv/dt = (1-n)*b."""
        sim = BernoulliODESimulation(
            _make_config(a=0.0, b=1.0, n=2.0, y_0=0.5)
        )
        # v = 1/y, v_0 = 2, dv/dt = -(1)*1 = -1, v(t) = 2 - t
        # y(t) = 1/(2-t), valid for t < 2
        y_1 = sim.analytical_solution(1.0)
        assert float(y_1) == pytest.approx(1.0, abs=1e-10)  # 1/(2-1) = 1.0

        y_15 = sim.analytical_solution(1.5)
        assert float(y_15) == pytest.approx(2.0, abs=1e-10)  # 1/(2-1.5) = 2.0


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


class TestBernoulliParameters:
    def test_different_a_changes_steady_state(self):
        """Changing a should change the steady state."""
        ss_1 = BernoulliODESimulation(_make_config(a=1.0, b=-1.0, n=2.0)).steady_state
        ss_2 = BernoulliODESimulation(_make_config(a=2.0, b=-1.0, n=2.0)).steady_state
        assert ss_1 is not None and ss_2 is not None
        assert ss_1 != pytest.approx(ss_2)
        # y_ss = -a/b = a for b=-1, so ss_1 = 1.0, ss_2 = 2.0
        assert ss_1 == pytest.approx(1.0)
        assert ss_2 == pytest.approx(2.0)

    def test_different_b_changes_steady_state(self):
        """Changing b should change the steady state."""
        ss_1 = BernoulliODESimulation(_make_config(a=1.0, b=-1.0, n=2.0)).steady_state
        ss_2 = BernoulliODESimulation(_make_config(a=1.0, b=-2.0, n=2.0)).steady_state
        assert ss_1 is not None and ss_2 is not None
        assert ss_1 == pytest.approx(1.0)   # -1/(-1) = 1.0
        assert ss_2 == pytest.approx(0.5)   # -1/(-2) = 0.5

    def test_different_n_changes_steady_state(self):
        """y_ss = (-a/b)^(1/(n-1)): different n gives different y_ss."""
        ss_2 = BernoulliODESimulation(_make_config(a=4.0, b=-1.0, n=2.0)).steady_state
        ss_3 = BernoulliODESimulation(_make_config(a=4.0, b=-1.0, n=3.0)).steady_state
        assert ss_2 is not None and ss_3 is not None
        assert ss_2 == pytest.approx(4.0)   # (-4/(-1))^1 = 4
        assert ss_3 == pytest.approx(2.0)   # (-4/(-1))^(1/2) = 2

    def test_different_y0(self):
        """Different initial values are set correctly."""
        for y_0 in [0.1, 0.5, 1.0, 3.0]:
            sim = BernoulliODESimulation(_make_config(y_0=y_0))
            state = sim.reset()
            assert state[0] == pytest.approx(y_0)

    def test_steady_state_positive_ratio(self):
        """Steady state exists when -a/b > 0 (a > 0, b < 0)."""
        sim = BernoulliODESimulation(_make_config(a=1.0, b=-1.0, n=2.0))
        assert sim.steady_state is not None
        assert sim.steady_state == pytest.approx(1.0)

    def test_steady_state_also_exists_negative_a_positive_b(self):
        """Steady state also exists when a < 0, b > 0 (ratio -a/b > 0)."""
        sim = BernoulliODESimulation(_make_config(a=-1.0, b=1.0, n=2.0))
        assert sim.steady_state is not None
        assert sim.steady_state == pytest.approx(1.0)

    def test_steady_state_none_when_ratio_negative(self):
        """No positive steady state when -a/b < 0 (same sign a and b)."""
        sim = BernoulliODESimulation(_make_config(a=1.0, b=1.0, n=2.0))
        assert sim.steady_state is None

    def test_steady_state_formula_n2(self):
        """For n=2, y_ss = -a/b (positive a, negative b)."""
        for a_val in [0.5, 1.0, 2.0, 3.0]:
            for b_val in [-0.5, -1.0, -2.0]:
                sim = BernoulliODESimulation(
                    _make_config(a=a_val, b=b_val, n=2.0)
                )
                assert sim.steady_state == pytest.approx(-a_val / b_val)


# ---------------------------------------------------------------------------
# Linearization
# ---------------------------------------------------------------------------


class TestBernoulliLinearization:
    def test_linearized_trajectory_shape(self):
        """compute_linearized_trajectory returns correct shapes."""
        sim = BernoulliODESimulation(_make_config(n_steps=100))
        result = sim.compute_linearized_trajectory(n_steps=100)
        assert result["time"].shape == (101,)
        assert result["y"].shape == (101,)
        assert result["v"].shape == (101,)
        assert result["v_analytical"].shape == (101,)

    def test_v_matches_y_transform(self):
        """v should equal y^(1-n) pointwise."""
        sim = BernoulliODESimulation(_make_config(n_steps=200, dt=0.01))
        result = sim.compute_linearized_trajectory(n_steps=200, dt=0.01)

        valid = result["y"] > 0
        v_from_y = result["y"][valid] ** (1.0 - sim.n)
        np.testing.assert_allclose(result["v"][valid], v_from_y, rtol=1e-10)

    def test_v_numerical_matches_v_analytical(self):
        """Numerical v and analytical v should closely match."""
        sim = BernoulliODESimulation(
            _make_config(a=1.0, b=-1.0, n=2.0, y_0=0.1, dt=0.001, n_steps=2000)
        )
        result = sim.compute_linearized_trajectory(n_steps=2000, dt=0.001)

        valid = np.isfinite(result["v"]) & np.isfinite(result["v_analytical"])
        err = np.abs(result["v"][valid] - result["v_analytical"][valid])
        assert np.max(err) < 1e-3, f"Max v-space error: {np.max(err)}"

    def test_linearized_v_approaches_steady_state(self):
        """For n=2, v -> v_ss = -b/a = 1.0 at steady state."""
        sim = BernoulliODESimulation(
            _make_config(a=1.0, b=-1.0, n=2.0, y_0=0.1, dt=0.001, n_steps=10000)
        )
        result = sim.compute_linearized_trajectory(n_steps=10000, dt=0.001)

        # v_ss = -b/a = -(-1)/1 = 1.0 (since y_ss=1 and v=1/y for n=2)
        v_final = result["v"][-1]
        assert v_final == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Blowup
# ---------------------------------------------------------------------------


class TestBernoulliBlowup:
    def test_blowup_time_none_for_stable(self):
        """No blowup for a > 0, b < 0, n=2 (stable nontrivial fixed point)."""
        sim = BernoulliODESimulation(_make_config(a=1.0, b=-1.0, n=2.0, y_0=0.1))
        assert sim.blowup_time is None

    def test_blowup_time_none_for_n_not_2(self):
        """blowup_time is only implemented for n=2."""
        sim = BernoulliODESimulation(_make_config(a=0.5, b=1.0, n=3.0, y_0=0.5))
        assert sim.blowup_time is None

    def test_blowup_time_positive_for_positive_a_positive_b(self):
        """For a > 0, b > 0, n=2: both terms are positive, blowup occurs."""
        # a=1.0, b=1.0, y_0=0.5: v_0=2, v(t) = (2+1)*exp(-t) - 1 = 3*exp(-t) - 1
        # blowup when v(t) = 0: 3*exp(-t) = 1, t = ln(3) ~ 1.099
        sim = BernoulliODESimulation(_make_config(a=1.0, b=1.0, n=2.0, y_0=0.5))
        t_blow = sim.blowup_time
        assert t_blow is not None
        assert t_blow == pytest.approx(np.log(3.0), abs=0.01)


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------


class TestBernoulliSweep:
    def test_parameter_sweep_returns_arrays(self):
        sim = BernoulliODESimulation(_make_config())
        result = sim.parameter_sweep_steady_state(
            a_values=np.array([1.0, 2.0]),
            b_values=np.array([-1.0, -0.5]),
        )
        assert "a" in result
        assert "b" in result
        assert "y_final" in result
        assert "y_ss_theory" in result
        assert len(result["a"]) == 4  # 2 * 2

    def test_sweep_matches_theory(self):
        """Swept y_final should approximate y_ss = -a/b for n=2 (stable case)."""
        sim = BernoulliODESimulation(_make_config(dt=0.01))
        result = sim.parameter_sweep_steady_state(
            a_values=np.array([0.5, 1.0, 2.0]),
            b_values=np.array([-1.0, -0.5]),
        )
        valid = np.isfinite(result["y_ss_theory"])
        errors = np.abs(result["y_final"][valid] - result["y_ss_theory"][valid])
        assert np.max(errors) < 0.1, f"Max error: {np.max(errors)}"


# ---------------------------------------------------------------------------
# Rediscovery data generation
# ---------------------------------------------------------------------------


class TestBernoulliRediscovery:
    def test_generate_solution_data(self):
        from simulating_anything.rediscovery.bernoulli_ode import (
            generate_solution_data,
        )
        data = generate_solution_data(n_samples=25, dt=0.01, n_steps=200)
        assert data["n_samples"] > 0
        assert len(data["a"]) == data["n_samples"]
        assert len(data["b"]) == data["n_samples"]
        assert len(data["y_final_numerical"]) == data["n_samples"]
        assert len(data["y_final_analytical"]) == data["n_samples"]
        assert data["n"] == 2.0

    def test_solution_data_numerical_matches_analytical(self):
        from simulating_anything.rediscovery.bernoulli_ode import (
            generate_solution_data,
        )
        data = generate_solution_data(n_samples=16, dt=0.005, n_steps=500)
        valid = np.isfinite(data["y_final_analytical"]) & (
            data["y_final_numerical"] > 0.01
        )
        if np.sum(valid) > 0:
            errors = np.abs(
                data["y_final_numerical"][valid]
                - data["y_final_analytical"][valid]
            )
            rel_errors = errors / (
                np.abs(data["y_final_analytical"][valid]) + 1e-15
            )
            assert np.mean(rel_errors) < 0.1

    def test_generate_ode_data(self):
        from simulating_anything.rediscovery.bernoulli_ode import generate_ode_data
        data = generate_ode_data(a=1.0, b=-1.0, n=2.0, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 1)
        assert len(data["time"]) == 501
        assert data["a"] == 1.0
        assert data["b"] == -1.0
        assert data["n"] == 2.0

    def test_ode_data_trajectory_finite(self):
        from simulating_anything.rediscovery.bernoulli_ode import generate_ode_data
        data = generate_ode_data(a=1.0, b=-1.0, n=2.0, n_steps=1000, dt=0.01)
        assert np.all(np.isfinite(data["y"]))

    def test_generate_linearization_data(self):
        from simulating_anything.rediscovery.bernoulli_ode import (
            generate_linearization_data,
        )
        data = generate_linearization_data(
            a=1.0, b=-1.0, n=2.0, n_steps=200, dt=0.01,
        )
        assert data["time"].shape == (201,)
        assert data["y"].shape == (201,)
        assert data["v"].shape == (201,)
        assert data["v_analytical"].shape == (201,)
        assert data["a"] == 1.0

    def test_linearization_data_v_matches_analytical(self):
        from simulating_anything.rediscovery.bernoulli_ode import (
            generate_linearization_data,
        )
        data = generate_linearization_data(
            a=1.0, b=-1.0, n=2.0, n_steps=500, dt=0.001,
        )
        valid = np.isfinite(data["v"]) & np.isfinite(data["v_analytical"])
        if np.sum(valid) > 0:
            err = np.abs(data["v"][valid] - data["v_analytical"][valid])
            assert np.mean(err) < 0.01

    def test_generate_steady_state_data(self):
        from simulating_anything.rediscovery.bernoulli_ode import (
            generate_steady_state_data,
        )
        data = generate_steady_state_data(n_a=5, n_b=5, n=2.0, dt=0.01)
        assert len(data["a"]) == 25  # 5 * 5
        assert len(data["b"]) == 25
        assert len(data["y_final"]) == 25
        assert len(data["y_ss_theory"]) == 25

    def test_steady_state_data_matches_theory(self):
        from simulating_anything.rediscovery.bernoulli_ode import (
            generate_steady_state_data,
        )
        data = generate_steady_state_data(n_a=5, n_b=5, n=2.0, dt=0.01)
        valid = np.isfinite(data["y_ss_theory"]) & (data["y_final"] > 0.01)
        if np.sum(valid) > 0:
            correlation = np.corrcoef(
                data["y_final"][valid], data["y_ss_theory"][valid],
            )[0, 1]
            assert correlation > 0.95
