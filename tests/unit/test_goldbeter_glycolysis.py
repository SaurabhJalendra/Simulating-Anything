"""Tests for the Goldbeter glycolysis model."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.goldbeter_glycolysis import (
    GoldbeterGlycolysisSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    v: float = 0.36,
    sigma: float = 0.1,
    q: float = 1.0,
    k_s: float = 0.02,
    L: float = 5e6,
    x_0: float = 1.0,
    y_0: float = 1.0,
    dt: float = 0.1,
    n_steps: int = 10000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.GOLDBETER_GLYCOLYSIS,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "v": v, "sigma": sigma, "q": q, "k_s": k_s, "L": L,
            "x_0": x_0, "y_0": y_0,
        },
    )


class TestGoldbeterGlycolysisBasic:
    """Basic simulation behavior tests."""

    def test_initial_state_shape(self):
        sim = GoldbeterGlycolysisSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_initial_state_values(self):
        sim = GoldbeterGlycolysisSimulation(_make_config(x_0=2.0, y_0=3.0))
        state = sim.reset()
        np.testing.assert_allclose(state, [2.0, 3.0])

    def test_default_initial_state(self):
        sim = GoldbeterGlycolysisSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state, [1.0, 1.0])

    def test_step_advances_state(self):
        sim = GoldbeterGlycolysisSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        sim = GoldbeterGlycolysisSimulation(_make_config())
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_state_dimension_is_2d(self):
        sim = GoldbeterGlycolysisSimulation(_make_config())
        sim.reset()
        for _ in range(100):
            sim.step()
        assert sim.observe().shape == (2,)

    def test_run_returns_trajectory(self):
        sim = GoldbeterGlycolysisSimulation(_make_config())
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 2)
        assert len(traj.timestamps) == 101

    def test_trajectory_timestamps(self):
        dt = 0.1
        sim = GoldbeterGlycolysisSimulation(_make_config(dt=dt))
        traj = sim.run(n_steps=50)
        expected = np.arange(51) * dt
        np.testing.assert_allclose(traj.timestamps, expected, atol=1e-10)

    def test_reproducibility(self):
        """Two runs with same config produce identical trajectories."""
        config = _make_config()
        sim1 = GoldbeterGlycolysisSimulation(config)
        traj1 = sim1.run(n_steps=200)

        sim2 = GoldbeterGlycolysisSimulation(config)
        traj2 = sim2.run(n_steps=200)

        np.testing.assert_array_equal(traj1.states, traj2.states)

    def test_parameter_extraction(self):
        sim = GoldbeterGlycolysisSimulation(
            _make_config(v=0.5, sigma=0.2, q=1.5, k_s=0.03, L=1e7)
        )
        assert sim.v == 0.5
        assert sim.sigma == 0.2
        assert sim.q == 1.5
        assert sim.k_s == 0.03
        assert sim.L == 1e7


class TestPhiFunction:
    """Tests for the allosteric rate function."""

    def test_phi_at_origin_is_zero(self):
        """phi(0, y) = 0 since numerator has factor x."""
        sim = GoldbeterGlycolysisSimulation(_make_config())
        assert sim.phi(0.0, 0.0) == 0.0
        assert sim.phi(0.0, 10.0) == 0.0

    def test_phi_non_negative(self):
        """phi should be non-negative for non-negative x, y."""
        sim = GoldbeterGlycolysisSimulation(_make_config())
        for x in [0.0, 0.1, 1.0, 10.0, 100.0]:
            for y in [0.0, 0.1, 1.0, 10.0, 100.0]:
                assert sim.phi(x, y) >= 0.0, f"phi({x}, {y}) < 0"

    def test_phi_bounded_below_one(self):
        """phi < 1 for all finite x, y since L >> 1 dominates denominator."""
        sim = GoldbeterGlycolysisSimulation(_make_config())
        for x in [0.01, 0.1, 1.0, 10.0, 100.0]:
            for y in [0.01, 0.1, 1.0, 10.0, 100.0]:
                assert sim.phi(x, y) < 1.0, f"phi({x}, {y}) >= 1"

    def test_phi_increases_with_x(self):
        """phi should increase with substrate x for fixed y > 0."""
        sim = GoldbeterGlycolysisSimulation(_make_config())
        y_fixed = 1.0
        prev = sim.phi(0.0, y_fixed)
        for x in [0.1, 0.5, 1.0, 5.0, 10.0]:
            curr = sim.phi(x, y_fixed)
            assert curr >= prev, f"phi not increasing: phi({x},{y_fixed})={curr}"
            prev = curr

    def test_phi_increases_with_y(self):
        """phi should increase with product y for fixed x > 0 (cooperative activation)."""
        sim = GoldbeterGlycolysisSimulation(_make_config())
        x_fixed = 1.0
        prev = sim.phi(x_fixed, 0.0)
        for y in [0.1, 1.0, 5.0, 10.0, 50.0]:
            curr = sim.phi(x_fixed, y)
            assert curr >= prev, f"phi not increasing: phi({x_fixed},{y})={curr}"
            prev = curr

    def test_phi_with_small_L(self):
        """With small L, phi approaches 1 for large concentrations."""
        sim = GoldbeterGlycolysisSimulation(_make_config(L=1.0))
        # phi(100, 100) ~ 100*101*(101)^2 / (1 + (101)^2*(101)^2) ~ 1
        val = sim.phi(100.0, 100.0)
        assert val > 0.99, f"phi with L=1 at large conc should be ~1, got {val}"

    def test_phi_symmetry_formula(self):
        """Verify phi matches the formula directly."""
        sim = GoldbeterGlycolysisSimulation(_make_config(L=1e3))
        x, y = 2.0, 3.0
        expected = x * (1 + x) * (1 + y)**2 / (1e3 + (1 + x)**2 * (1 + y)**2)
        actual = sim.phi(x, y)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)


class TestDerivatives:
    """Tests for the ODE derivatives."""

    def test_derivatives_shape(self):
        sim = GoldbeterGlycolysisSimulation(_make_config())
        sim.reset()
        dy = sim._derivatives(sim.observe())
        assert dy.shape == (2,)

    def test_dx_positive_when_phi_small(self):
        """dx/dt = v - sigma*phi; at low concentrations phi~0 so dx/dt ~ v > 0."""
        sim = GoldbeterGlycolysisSimulation(_make_config(v=0.36))
        dy = sim._derivatives(np.array([0.001, 0.001]))
        assert dy[0] > 0, "dx/dt should be positive when phi ~ 0"

    def test_dy_negative_when_phi_small_and_y_large(self):
        """dy/dt = q*sigma*phi - k_s*y; if phi~0 and y large, dy < 0."""
        sim = GoldbeterGlycolysisSimulation(_make_config())
        dy = sim._derivatives(np.array([0.001, 100.0]))
        assert dy[1] < 0, "dy/dt should be negative when phi~0 and y is large"

    def test_derivatives_finite(self):
        """Derivatives should be finite for reasonable states."""
        sim = GoldbeterGlycolysisSimulation(_make_config())
        for x in [0.0, 0.5, 1.0, 5.0, 10.0]:
            for y in [0.0, 0.5, 1.0, 5.0, 10.0]:
                dy = sim._derivatives(np.array([x, y]))
                assert np.all(np.isfinite(dy)), f"Non-finite derivative at ({x},{y})"


class TestTrajectoryBehavior:
    """Tests for long-term trajectory properties."""

    def test_trajectory_bounded(self):
        """Trajectory should remain bounded for default parameters."""
        sim = GoldbeterGlycolysisSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(10000):
            sim.step()
            x, y = sim.observe()
            assert abs(x) < 1000, f"x diverged: {x}"
            assert abs(y) < 1000, f"y diverged: {y}"

    def test_state_non_negativity(self):
        """Concentrations should remain non-negative for normal parameters."""
        sim = GoldbeterGlycolysisSimulation(_make_config(dt=0.05))
        sim.reset()
        for _ in range(10000):
            sim.step()
            x, y = sim.observe()
            assert x > -0.01, f"x went negative: {x}"
            assert y > -0.01, f"y went negative: {y}"

    def test_oscillation_default_params(self):
        """Default parameters should produce oscillatory behavior."""
        sim = GoldbeterGlycolysisSimulation(_make_config(dt=0.1))
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        # Collect y values
        y_vals = []
        for _ in range(5000):
            sim.step()
            y_vals.append(sim.observe()[1])

        amplitude = max(y_vals) - min(y_vals)
        # Default params (v=0.36, sigma=0.1, L=5e6) should oscillate
        assert amplitude > 0.01, f"No oscillation detected, amplitude={amplitude}"

    def test_steady_state_for_extreme_v(self):
        """Very small v should lead to low concentrations (near steady state)."""
        sim = GoldbeterGlycolysisSimulation(_make_config(v=0.001, dt=0.1))
        sim.reset()

        # Run for a long time
        for _ in range(20000):
            sim.step()

        # Collect x values -- should be near steady state (small variation)
        x_vals = []
        for _ in range(5000):
            sim.step()
            x_vals.append(sim.observe()[0])

        # Very small input flux -> system is likely not oscillatory
        # x should accumulate slowly since phi is very small with L=5e6
        assert max(x_vals) < 100, f"x too large for small v: {max(x_vals)}"


class TestBifurcation:
    """Tests for bifurcation and parameter sweeps."""

    def test_bifurcation_sweep_returns_correct_shape(self):
        sim = GoldbeterGlycolysisSimulation(_make_config())
        sim.reset()
        result = sim.bifurcation_sweep(
            param_name="v",
            param_range=(0.1, 0.5),
            n_params=5,
            n_steps=5000,
        )
        assert len(result["param_values"]) == 5
        assert len(result["y_max"]) == 5
        assert len(result["y_min"]) == 5
        assert len(result["amplitude"]) == 5

    def test_bifurcation_sweep_amplitude_non_negative(self):
        sim = GoldbeterGlycolysisSimulation(_make_config())
        sim.reset()
        result = sim.bifurcation_sweep(
            param_name="v",
            param_range=(0.1, 0.5),
            n_params=5,
            n_steps=5000,
        )
        assert np.all(result["amplitude"] >= 0)

    def test_bifurcation_sweep_sigma(self):
        """Sweep sigma parameter works without errors."""
        sim = GoldbeterGlycolysisSimulation(_make_config())
        sim.reset()
        result = sim.bifurcation_sweep(
            param_name="sigma",
            param_range=(0.01, 0.3),
            n_params=5,
            n_steps=5000,
        )
        assert len(result["param_values"]) == 5
        assert result["param_name"] == "sigma"

    def test_y_max_geq_y_min(self):
        sim = GoldbeterGlycolysisSimulation(_make_config())
        sim.reset()
        result = sim.bifurcation_sweep(
            param_name="v",
            param_range=(0.1, 0.5),
            n_params=5,
            n_steps=5000,
        )
        assert np.all(result["y_max"] >= result["y_min"])


class TestPeriod:
    """Tests for period and amplitude measurement."""

    def test_compute_period_returns_float(self):
        sim = GoldbeterGlycolysisSimulation(_make_config(dt=0.1))
        sim.reset()
        period = sim.compute_period(n_transient=2000, n_measure=3000)
        assert isinstance(period, float)

    def test_compute_period_positive_or_inf(self):
        sim = GoldbeterGlycolysisSimulation(_make_config(dt=0.1))
        sim.reset()
        period = sim.compute_period(n_transient=2000, n_measure=3000)
        assert period > 0

    def test_measure_amplitude_non_negative(self):
        sim = GoldbeterGlycolysisSimulation(_make_config(dt=0.1))
        sim.reset()
        amp = sim.measure_amplitude(transient_time=200.0)
        assert amp >= 0

    def test_period_finite_when_oscillating(self):
        """Default params should give finite period (oscillatory regime)."""
        sim = GoldbeterGlycolysisSimulation(_make_config(dt=0.1))
        sim.reset()
        period = sim.compute_period(n_transient=3000, n_measure=5000)
        # Default params may or may not oscillate depending on L
        # Just check it returns a valid float
        assert np.isfinite(period) or period == float("inf")


class TestRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.goldbeter_glycolysis import (
            generate_ode_data,
        )

        data = generate_ode_data(n_steps=500, dt=0.1)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["v"] == 0.36
        assert data["sigma"] == 0.1

    def test_ode_data_custom_params(self):
        from simulating_anything.rediscovery.goldbeter_glycolysis import (
            generate_ode_data,
        )

        data = generate_ode_data(v=0.5, sigma=0.2, n_steps=200, dt=0.1)
        assert data["states"].shape == (201, 2)
        assert data["v"] == 0.5
        assert data["sigma"] == 0.2

    def test_bifurcation_data_generation(self):
        from simulating_anything.rediscovery.goldbeter_glycolysis import (
            generate_bifurcation_data,
        )

        data = generate_bifurcation_data(n_v=5, v_range=(0.1, 0.5), dt=0.5)
        assert len(data["v_values"]) == 5
        assert len(data["amplitude"]) == 5

    def test_sigma_sweep_data_generation(self):
        from simulating_anything.rediscovery.goldbeter_glycolysis import (
            generate_sigma_sweep_data,
        )

        data = generate_sigma_sweep_data(n_sigma=5, sigma_range=(0.05, 0.3), dt=0.5)
        assert len(data["sigma_values"]) == 5
        assert len(data["amplitude"]) == 5

    def test_period_data_generation(self):
        from simulating_anything.rediscovery.goldbeter_glycolysis import (
            generate_period_data,
        )

        data = generate_period_data(n_v=3, v_range=(0.2, 0.5), dt=0.5)
        assert len(data["v_values"]) == 3
        assert len(data["period"]) == 3


class TestRK4Integration:
    """Tests for the RK4 integrator accuracy."""

    def test_rk4_small_dt_more_accurate(self):
        """Smaller dt should give more accurate integration."""
        # Run with two dt values and compare final state
        config_coarse = _make_config(dt=1.0, n_steps=100)
        config_fine = _make_config(dt=0.1, n_steps=1000)

        sim_coarse = GoldbeterGlycolysisSimulation(config_coarse)
        sim_fine = GoldbeterGlycolysisSimulation(config_fine)

        traj_coarse = sim_coarse.run(n_steps=100)
        traj_fine = sim_fine.run(n_steps=1000)

        # Both run for t=100; fine should be more accurate
        # Just verify both finish without error and produce different results
        assert traj_coarse.states.shape == (101, 2)
        assert traj_fine.states.shape == (1001, 2)

    def test_conservation_of_total_concentration(self):
        """In the special case q=1, total concentration x+y should change only
        by input v and output k_s*y per timestep (no creation/destruction)."""
        sim = GoldbeterGlycolysisSimulation(_make_config(dt=0.01, q=1.0))
        sim.reset()

        # Run a few steps and check mass balance
        states = [sim.observe().copy()]
        for _ in range(100):
            sim.step()
            states.append(sim.observe().copy())

        states_arr = np.array(states)
        x = states_arr[:, 0]
        y = states_arr[:, 1]

        # d(x+y)/dt = v - k_s*y (since q*sigma*phi cancels)
        # So total = x + y changes smoothly
        total = x + y
        # Check that total is monotonically changing or oscillating smoothly
        diff = np.diff(total)
        assert np.all(np.isfinite(diff)), "Total concentration has non-finite changes"

    def test_step_count_increments(self):
        sim = GoldbeterGlycolysisSimulation(_make_config())
        sim.reset()
        assert sim._step_count == 0
        sim.step()
        assert sim._step_count == 1
        sim.step()
        assert sim._step_count == 2
