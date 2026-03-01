"""Tests for the Kepler two-body orbit simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.kepler import KeplerOrbit
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestKeplerCreation:
    """Tests for KeplerOrbit creation and configuration."""

    def _make_sim(self, **kwargs) -> KeplerOrbit:
        defaults = {"GM": 1.0, "initial_r": 1.0, "eccentricity": 0.5}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.KEPLER,
            dt=0.001,
            n_steps=10000,
            parameters=defaults,
        )
        return KeplerOrbit(config)

    def test_creation(self):
        """Simulation object should be created without error."""
        sim = self._make_sim()
        assert sim is not None
        assert sim.GM == 1.0
        assert sim.initial_r == 1.0
        assert sim.ecc == 0.5

    def test_default_params(self):
        """Default parameters should match specification."""
        config = SimulationConfig(
            domain=Domain.KEPLER,
            dt=0.001,
            n_steps=1000,
            parameters={},
        )
        sim = KeplerOrbit(config)
        assert sim.GM == 1.0
        assert sim.initial_r == 1.0
        assert sim.ecc == 0.5

    def test_custom_params(self):
        """Custom parameters should be correctly applied."""
        sim = self._make_sim(GM=2.0, initial_r=3.0, eccentricity=0.2)
        assert sim.GM == 2.0
        assert sim.initial_r == 3.0
        assert sim.ecc == 0.2

    def test_observe_shape(self):
        """Observe should return shape (4,) = [r, theta, v_r, v_theta]."""
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_step(self):
        """Step should advance the state."""
        sim = self._make_sim()
        state0 = sim.reset()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_trajectory_shape(self):
        """Running a trajectory should return correct shape."""
        sim = self._make_sim()
        sim.reset()
        states = [sim.observe().copy()]
        for _ in range(100):
            states.append(sim.step().copy())
        states = np.array(states)
        assert states.shape == (101, 4)
        assert np.all(np.isfinite(states))


class TestKeplerPhysics:
    """Tests for Kepler orbit physics accuracy."""

    def _make_sim(self, **kwargs) -> KeplerOrbit:
        defaults = {"GM": 1.0, "initial_r": 1.0, "eccentricity": 0.5}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.KEPLER,
            dt=0.001,
            n_steps=10000,
            parameters=defaults,
        )
        return KeplerOrbit(config)

    def test_energy_conservation(self):
        """Energy should be conserved to high precision over 10 orbits."""
        sim = self._make_sim(eccentricity=0.5)
        sim.reset()
        E0 = sim.energy

        # Run for ~10 orbital periods (T = 2*pi for a=1, GM=1)
        T = sim.period
        n_steps = int(10 * T / 0.001)
        for _ in range(n_steps):
            sim.step()

        E_final = sim.energy
        rel_drift = abs(E_final - E0) / abs(E0)
        assert rel_drift < 1e-6, (
            f"Energy drift {rel_drift:.2e} exceeds tolerance"
        )

    def test_angular_momentum_conservation(self):
        """Angular momentum should be conserved to high precision."""
        sim = self._make_sim(eccentricity=0.5)
        sim.reset()
        L0 = sim.angular_momentum

        T = sim.period
        n_steps = int(10 * T / 0.001)
        for _ in range(n_steps):
            sim.step()

        L_final = sim.angular_momentum
        rel_drift = abs(L_final - L0) / abs(L0)
        assert rel_drift < 1e-6, (
            f"Angular momentum drift {rel_drift:.2e} exceeds tolerance"
        )

    def test_circular_orbit(self):
        """Circular orbit (e=0): r should remain constant."""
        sim = self._make_sim(eccentricity=0.0, initial_r=2.0)
        sim.reset()

        # Run one full orbit
        T = sim.period
        n_steps = int(T / 0.001)
        r_values = []
        for _ in range(n_steps):
            state = sim.step()
            r_values.append(state[0])

        r_values = np.array(r_values)
        # r should stay very close to initial value
        rel_variation = np.std(r_values) / np.mean(r_values)
        assert rel_variation < 1e-6, (
            f"Circular orbit r variation {rel_variation:.2e} too large"
        )

    def test_elliptical_orbit(self):
        """Elliptical orbit (e=0.5): r should oscillate between perihelion and aphelion."""
        a = 1.0
        e = 0.5
        sim = self._make_sim(eccentricity=e, initial_r=a)
        sim.reset()

        T = sim.period
        n_steps = int(T / 0.001)
        r_values = []
        for _ in range(n_steps):
            state = sim.step()
            r_values.append(state[0])

        r_min = np.min(r_values)
        r_max = np.max(r_values)

        r_peri_theory = a * (1 - e)
        r_apo_theory = a * (1 + e)

        assert abs(r_min - r_peri_theory) / r_peri_theory < 0.01, (
            f"Perihelion {r_min:.4f} vs theory {r_peri_theory:.4f}"
        )
        assert abs(r_max - r_apo_theory) / r_apo_theory < 0.01, (
            f"Aphelion {r_max:.4f} vs theory {r_apo_theory:.4f}"
        )

    def test_perihelion_aphelion(self):
        """Perihelion and aphelion properties should match theory."""
        a = 2.0
        e = 0.3
        sim = self._make_sim(eccentricity=e, initial_r=a)
        sim.reset()

        r_peri = sim.perihelion
        r_apo = sim.aphelion

        assert abs(r_peri - a * (1 - e)) < 0.01, (
            f"Perihelion {r_peri:.4f} vs {a * (1 - e):.4f}"
        )
        assert abs(r_apo - a * (1 + e)) < 0.01, (
            f"Aphelion {r_apo:.4f} vs {a * (1 + e):.4f}"
        )

    def test_period_kepler_third_law(self):
        """T^2 should be proportional to a^3 (Kepler's third law)."""
        GM = 1.0
        dt = 0.001
        results = []

        for a in [0.5, 1.0, 2.0, 3.0]:
            config = SimulationConfig(
                domain=Domain.KEPLER,
                dt=dt,
                n_steps=1,
                parameters={
                    "GM": GM, "initial_r": a, "eccentricity": 0.2,
                },
            )
            sim = KeplerOrbit(config)
            sim.reset()

            # Measure period from theta completing 2*pi
            total_theta = 0.0
            prev_theta = sim.observe()[1]
            n_step = 0
            max_steps = int(200.0 / dt)

            while total_theta < 2.0 * np.pi and n_step < max_steps:
                state = sim.step()
                n_step += 1
                d_theta = state[1] - prev_theta
                total_theta += d_theta
                prev_theta = state[1]

            T_measured = n_step * dt
            T_theory = 2.0 * np.pi * a**1.5 / np.sqrt(GM)
            results.append((a, T_measured, T_theory))

        # Check T^2 ~ a^3
        for a, T_meas, T_theo in results:
            rel_err = abs(T_meas - T_theo) / T_theo
            assert rel_err < 0.01, (
                f"a={a}: T_meas={T_meas:.4f}, T_theo={T_theo:.4f}, "
                f"error={rel_err:.4%}"
            )

    def test_semi_major_axis_property(self):
        """Semi-major axis computed from state should match input."""
        a_input = 2.5
        sim = self._make_sim(initial_r=a_input, eccentricity=0.3)
        sim.reset()

        a_computed = sim.semi_major_axis
        assert abs(a_computed - a_input) / a_input < 1e-10, (
            f"SMA {a_computed:.6f} vs input {a_input:.6f}"
        )

    def test_eccentricity_from_state(self):
        """Eccentricity computed from state should match input."""
        e_input = 0.7
        sim = self._make_sim(eccentricity=e_input)
        sim.reset()

        e_computed = sim.eccentricity_from_state
        assert abs(e_computed - e_input) < 1e-6, (
            f"Eccentricity {e_computed:.6f} vs input {e_input:.6f}"
        )

    def test_initial_conditions_at_perihelion(self):
        """Reset should place the orbit at perihelion with v_r = 0."""
        a = 1.0
        e = 0.5
        sim = self._make_sim(initial_r=a, eccentricity=e)
        state = sim.reset()

        # At perihelion
        r_peri = a * (1 - e)
        assert abs(state[0] - r_peri) < 1e-10, f"r = {state[0]} vs {r_peri}"
        assert abs(state[1]) < 1e-10, f"theta = {state[1]} should be 0"
        assert abs(state[2]) < 1e-10, f"v_r = {state[2]} should be 0"
        assert state[3] > 0, "v_theta should be positive"

    def test_trajectory_stays_finite(self):
        """Trajectory should remain finite and r should stay positive."""
        sim = self._make_sim(eccentricity=0.8)
        sim.reset()

        T = sim.period
        n_steps = int(5 * T / 0.001)
        for _ in range(n_steps):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State not finite: {state}"
            assert state[0] > 0, f"r went negative: {state[0]}"


class TestKeplerRediscovery:
    """Tests for Kepler rediscovery data generation."""

    def test_period_data_generation(self):
        from simulating_anything.rediscovery.kepler import (
            generate_period_vs_sma_data,
        )
        data = generate_period_vs_sma_data(
            n_samples=5, dt=0.001, GM=1.0, eccentricity=0.2,
        )
        assert "a_values" in data
        assert "T_measured" in data
        assert "T_theory" in data
        assert len(data["a_values"]) > 0
        # Measured periods should be close to theory
        rel_err = np.abs(
            data["T_measured"] - data["T_theory"]
        ) / data["T_theory"]
        assert np.mean(rel_err) < 0.02, (
            f"Mean period error {np.mean(rel_err):.4%} too large"
        )

    def test_energy_conservation_data(self):
        from simulating_anything.rediscovery.kepler import (
            generate_energy_conservation_data,
        )
        data = generate_energy_conservation_data(
            n_orbits=2, dt=0.001, GM=1.0, a=1.0, eccentricity=0.3,
        )
        assert "times" in data
        assert "energies" in data
        assert data["max_drift"] < 1e-5

    def test_angular_momentum_data(self):
        from simulating_anything.rediscovery.kepler import (
            generate_angular_momentum_data,
        )
        data = generate_angular_momentum_data(
            n_orbits=2, dt=0.001, GM=1.0, a=1.0, eccentricity=0.3,
        )
        assert "times" in data
        assert "angular_momenta" in data
        assert data["max_drift"] < 1e-5
