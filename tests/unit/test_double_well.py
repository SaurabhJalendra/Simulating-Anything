"""Tests for the double well potential simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.double_well import DoubleWellSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    gamma: float = 0.5,
    sigma: float = 0.0,
    x_0: float = 0.5,
    v_0: float = 0.0,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.DOUBLE_WELL,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "gamma": gamma,
            "sigma": sigma,
            "x_0": x_0,
            "v_0": v_0,
        },
    )


class TestDoubleWellCreation:
    """Tests for construction and default parameters."""

    def test_creation_default(self):
        """DoubleWellSimulation can be created with default parameters."""
        sim = DoubleWellSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)
        np.testing.assert_allclose(state, [0.5, 0.0])

    def test_default_parameters(self):
        """Default parameters should match the specification."""
        config = SimulationConfig(
            domain=Domain.DOUBLE_WELL, dt=0.01, n_steps=100, parameters={}
        )
        sim = DoubleWellSimulation(config)
        assert sim.gamma == 0.5
        assert sim.sigma == 0.0
        assert sim.x_0 == 0.5
        assert sim.v_0 == 0.0

    def test_custom_parameters(self):
        """Custom parameters should override defaults."""
        sim = DoubleWellSimulation(
            _make_config(gamma=1.0, sigma=0.1, x_0=-0.8, v_0=0.3)
        )
        assert sim.gamma == 1.0
        assert sim.sigma == 0.1
        assert sim.x_0 == -0.8
        assert sim.v_0 == 0.3

    def test_state_shape(self):
        """State should be 2D: [x, v]."""
        sim = DoubleWellSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_observe_equals_state(self):
        """observe() should return the current state."""
        sim = DoubleWellSimulation(_make_config())
        sim.reset()
        for _ in range(10):
            state = sim.step()
            observed = sim.observe()
            np.testing.assert_array_equal(state, observed)


class TestDoubleWellEquilibria:
    """Tests for equilibrium points and potential."""

    def test_equilibria_exist(self):
        """find_equilibria should return three fixed points."""
        eq = DoubleWellSimulation.find_equilibria()
        assert len(eq) == 3
        assert "x_minus" in eq
        assert "x_zero" in eq
        assert "x_plus" in eq

    def test_equilibrium_positions(self):
        """Equilibria are at x = -1, 0, +1."""
        eq = DoubleWellSimulation.find_equilibria()
        assert eq["x_minus"]["x"] == -1.0
        assert eq["x_zero"]["x"] == 0.0
        assert eq["x_plus"]["x"] == 1.0

    def test_equilibrium_stability(self):
        """x=0 is unstable, x=+/-1 are stable."""
        eq = DoubleWellSimulation.find_equilibria()
        assert eq["x_minus"]["stable"] is True
        assert eq["x_zero"]["stable"] is False
        assert eq["x_plus"]["stable"] is True

    def test_potential_at_minima(self):
        """V(+/-1) = 1/4 - 1/2 = -1/4."""
        assert np.isclose(DoubleWellSimulation.compute_potential(1.0), -0.25)
        assert np.isclose(DoubleWellSimulation.compute_potential(-1.0), -0.25)

    def test_potential_at_origin(self):
        """V(0) = 0."""
        assert np.isclose(DoubleWellSimulation.compute_potential(0.0), 0.0)

    def test_potential_at_large_x(self):
        """V(x) should be large and positive for large |x|."""
        V_3 = DoubleWellSimulation.compute_potential(3.0)
        assert V_3 > 0, f"V(3) = {V_3} should be positive"
        V_neg3 = DoubleWellSimulation.compute_potential(-3.0)
        assert V_neg3 > 0, f"V(-3) = {V_neg3} should be positive"

    def test_potential_symmetry(self):
        """V(x) = V(-x) -- symmetric double well."""
        for x in [0.3, 0.7, 1.5, 2.0]:
            V_pos = DoubleWellSimulation.compute_potential(x)
            V_neg = DoubleWellSimulation.compute_potential(-x)
            assert np.isclose(V_pos, V_neg), f"V({x}) != V(-{x})"

    def test_potential_array(self):
        """compute_potential should work on numpy arrays."""
        x = np.array([-1.0, 0.0, 1.0])
        V = DoubleWellSimulation.compute_potential(x)
        np.testing.assert_allclose(V, [-0.25, 0.0, -0.25])


class TestDoubleWellBarrier:
    """Tests for barrier height and crossing behavior."""

    def test_barrier_height(self):
        """Barrier height should be 1/4."""
        assert np.isclose(DoubleWellSimulation.barrier_height(), 0.25)

    def test_barrier_height_from_potential(self):
        """Barrier height = V(0) - V(1) = 0 - (-1/4) = 1/4."""
        V_top = DoubleWellSimulation.compute_potential(0.0)
        V_min = DoubleWellSimulation.compute_potential(1.0)
        barrier = V_top - V_min
        assert np.isclose(barrier, 0.25)

    def test_trapped_low_energy(self):
        """Particle with low energy should be trapped in one well."""
        # Start near right minimum with small velocity
        sim = DoubleWellSimulation(_make_config(
            gamma=0.0, x_0=1.0, v_0=0.1, dt=0.005
        ))
        sim.reset()
        assert sim.is_trapped_in_well(), (
            f"E = {sim.total_energy} should be < 0 (trapped)"
        )

        # Verify it stays in the right well
        for _ in range(5000):
            sim.step()
            assert sim.observe()[0] > 0, "Trapped particle should not cross x=0"

    def test_crossing_high_energy(self):
        """Particle with high energy should cross the barrier."""
        # Start near right minimum with large velocity
        sim = DoubleWellSimulation(_make_config(
            gamma=0.0, x_0=1.0, v_0=1.5, dt=0.005
        ))
        sim.reset()
        assert not sim.is_trapped_in_well(), (
            f"E = {sim.total_energy} should be > 0 (can cross)"
        )

        # Verify it crosses to the left well at some point
        crossed = False
        for _ in range(20000):
            sim.step()
            if sim.observe()[0] < 0:
                crossed = True
                break
        assert crossed, "High-energy particle should cross the barrier"


class TestDoubleWellEnergy:
    """Tests for energy properties."""

    def test_energy_decomposition(self):
        """KE + PE should equal total energy."""
        sim = DoubleWellSimulation(_make_config(x_0=0.8, v_0=0.5))
        sim.reset()
        assert np.isclose(
            sim.kinetic_energy + sim.potential_energy,
            sim.total_energy,
            rtol=1e-12,
        )

    def test_energy_conservation_undamped(self):
        """Energy should be conserved for gamma=0, sigma=0."""
        sim = DoubleWellSimulation(_make_config(
            gamma=0.0, sigma=0.0, x_0=0.8, v_0=0.3, dt=0.001
        ))
        sim.reset()
        E0 = sim.total_energy

        for _ in range(10000):
            sim.step()

        E_final = sim.total_energy
        rel_drift = abs(E_final - E0) / max(abs(E0), 1e-10)
        assert rel_drift < 1e-6, f"Energy drift {rel_drift:.2e} too large"

    def test_energy_dissipation_damped(self):
        """Energy should decrease with damping (gamma > 0)."""
        sim = DoubleWellSimulation(_make_config(
            gamma=0.5, sigma=0.0, x_0=1.0, v_0=0.5, dt=0.005
        ))
        sim.reset()
        E0 = sim.total_energy

        for _ in range(5000):
            sim.step()

        E_final = sim.total_energy
        assert E_final < E0, (
            f"Energy should decrease: E0={E0:.6f}, E_final={E_final:.6f}"
        )

    def test_energy_at_minimum(self):
        """Energy at x=1, v=0 should be V(1) = -1/4."""
        sim = DoubleWellSimulation(_make_config(x_0=1.0, v_0=0.0))
        sim.reset()
        assert np.isclose(sim.total_energy, -0.25, atol=1e-12)

    def test_energy_at_origin(self):
        """Energy at x=0, v=0 should be V(0) = 0."""
        sim = DoubleWellSimulation(_make_config(x_0=0.0, v_0=0.0))
        sim.reset()
        assert np.isclose(sim.total_energy, 0.0, atol=1e-12)


class TestDoubleWellDynamics:
    """Tests for trajectory and dynamics."""

    def test_step_changes_state(self):
        """A single step should change the state."""
        sim = DoubleWellSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_trajectory_shape(self):
        """Run should produce a trajectory with correct shape."""
        sim = DoubleWellSimulation(_make_config(n_steps=500))
        traj = sim.run(n_steps=500)
        assert traj.states.shape == (501, 2)
        assert len(traj.timestamps) == 501

    def test_trajectory_finite(self):
        """All trajectory states should be finite."""
        sim = DoubleWellSimulation(_make_config(dt=0.005, n_steps=5000))
        traj = sim.run(n_steps=5000)
        assert np.all(np.isfinite(traj.states)), "Trajectory contains non-finite values"

    def test_reproducibility(self):
        """Two runs with same config should produce identical trajectories."""
        config = _make_config()
        sim1 = DoubleWellSimulation(config)
        traj1 = sim1.run(n_steps=200)

        sim2 = DoubleWellSimulation(config)
        traj2 = sim2.run(n_steps=200)

        np.testing.assert_allclose(traj1.states, traj2.states, atol=1e-12)

    def test_damped_settles_to_minimum(self):
        """With damping, particle should settle to the nearest well minimum."""
        sim = DoubleWellSimulation(_make_config(
            gamma=1.0, x_0=0.8, v_0=0.0, dt=0.005, n_steps=20000
        ))
        sim.reset()
        for _ in range(20000):
            sim.step()

        x_final = sim.observe()[0]
        # Should settle to x ~ 1.0 (right well)
        assert abs(x_final - 1.0) < 0.01, (
            f"Should settle to right minimum: x_final={x_final:.4f}"
        )

    def test_damped_settles_left_well(self):
        """Starting in left well with damping should settle to x = -1."""
        sim = DoubleWellSimulation(_make_config(
            gamma=1.0, x_0=-0.8, v_0=0.0, dt=0.005, n_steps=20000
        ))
        sim.reset()
        for _ in range(20000):
            sim.step()

        x_final = sim.observe()[0]
        assert abs(x_final + 1.0) < 0.01, (
            f"Should settle to left minimum: x_final={x_final:.4f}"
        )

    def test_which_well(self):
        """which_well should correctly identify left vs right well."""
        sim = DoubleWellSimulation(_make_config(x_0=0.5))
        sim.reset()
        assert sim.which_well() == 1

        sim2 = DoubleWellSimulation(_make_config(x_0=-0.5))
        sim2.reset()
        assert sim2.which_well() == -1

    def test_static_at_minimum(self):
        """Starting exactly at x=1, v=0 should remain stationary."""
        sim = DoubleWellSimulation(_make_config(
            gamma=0.0, x_0=1.0, v_0=0.0, dt=0.001
        ))
        sim.reset()
        for _ in range(1000):
            sim.step()

        state = sim.observe()
        assert np.isclose(state[0], 1.0, atol=1e-8), (
            f"x deviated from minimum: {state[0]}"
        )
        assert np.isclose(state[1], 0.0, atol=1e-8), (
            f"v deviated from zero: {state[1]}"
        )


class TestDoubleWellFrequency:
    """Tests for oscillation frequency."""

    def test_well_frequency_value(self):
        """Well frequency should be sqrt(2)."""
        assert np.isclose(DoubleWellSimulation.well_frequency(), np.sqrt(2.0))

    def test_small_oscillation_period(self):
        """Measured period should match 2*pi/sqrt(2) for small oscillations."""
        sim = DoubleWellSimulation(_make_config(
            gamma=0.0, x_0=1.01, v_0=0.0, dt=0.001, n_steps=50000
        ))
        sim.reset()
        T = sim.compute_oscillation_period(n_periods=5)
        T_theory = 2 * np.pi / np.sqrt(2.0)
        assert abs(T - T_theory) / T_theory < 0.02, (
            f"Period mismatch: measured={T:.4f}, theory={T_theory:.4f}"
        )

    def test_frequency_measurement_from_data(self):
        """Directly measure frequency from trajectory zero crossings."""
        dt = 0.001
        n_steps = 30000
        sim = DoubleWellSimulation(_make_config(
            gamma=0.0, x_0=1.02, v_0=0.0, dt=dt, n_steps=n_steps
        ))
        sim.reset()

        prev_dx = sim.observe()[0] - 1.0
        crossings: list[float] = []
        for step_idx in range(n_steps):
            state = sim.step()
            dx = state[0] - 1.0
            if prev_dx < 0 and dx >= 0:
                frac = -prev_dx / (dx - prev_dx)
                crossings.append((step_idx + frac) * dt)
            prev_dx = dx

        assert len(crossings) >= 3, f"Not enough crossings: {len(crossings)}"
        periods = np.diff(crossings)
        T_measured = float(np.median(periods))
        omega_measured = 2 * np.pi / T_measured
        omega_theory = np.sqrt(2.0)

        rel_error = abs(omega_measured - omega_theory) / omega_theory
        assert rel_error < 0.02, (
            f"Frequency mismatch: measured={omega_measured:.4f}, "
            f"theory={omega_theory:.4f}, error={rel_error:.4%}"
        )


class TestDoubleWellRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.double_well import generate_ode_data

        data = generate_ode_data(gamma=0.5, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["gamma"] == 0.5
        assert np.all(np.isfinite(data["states"]))

    def test_energy_conservation_data(self):
        from simulating_anything.rediscovery.double_well import (
            generate_energy_conservation_data,
        )

        data = generate_energy_conservation_data(
            n_trajectories=3, n_steps=1000, dt=0.001
        )
        assert "final_drift" in data
        assert "max_drift" in data
        assert len(data["final_drift"]) == 3
        assert np.max(data["max_drift"]) < 1e-4, (
            f"Energy drift too large: {np.max(data['max_drift']):.2e}"
        )

    def test_well_frequency_data(self):
        from simulating_anything.rediscovery.double_well import (
            generate_well_frequency_data,
        )

        data = generate_well_frequency_data(n_samples=5, n_steps=20000, dt=0.001)
        assert "omega_measured" in data
        assert "omega_theory" in data
        assert data["omega_theory"] == np.sqrt(2.0)
        if len(data["omega_measured"]) > 0:
            rel_err = np.abs(
                data["omega_measured"] - data["omega_theory"]
            ) / data["omega_theory"]
            assert np.mean(rel_err) < 0.05, (
                f"Frequency error too large: {np.mean(rel_err):.4%}"
            )

    def test_barrier_verification_data(self):
        from simulating_anything.rediscovery.double_well import (
            generate_barrier_verification_data,
        )

        data = generate_barrier_verification_data(
            n_energies=10, n_steps=10000, dt=0.005
        )
        assert "initial_energy" in data
        assert "crossed" in data
        assert len(data["initial_energy"]) == 10
        assert len(data["crossed"]) == 10
        # At least some should cross (high energy) and some should not (low energy)
        assert np.any(data["crossed"]), "Some particles should cross the barrier"
        assert np.any(~data["crossed"]), "Some particles should be trapped"
