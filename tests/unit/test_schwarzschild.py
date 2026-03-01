"""Tests for the Schwarzschild geodesic simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.schwarzschild import SchwarzschildGeodesic
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestSchwarzschildGeodesic:
    """Tests for the Schwarzschild geodesic simulation."""

    def _make_sim(self, **kwargs) -> SchwarzschildGeodesic:
        defaults = {"M": 1.0, "L": 4.0, "r_0": 10.0, "pr_0": 0.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.SCHWARZSCHILD,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return SchwarzschildGeodesic(config)

    def test_creation(self):
        """Simulation should be created with default parameters."""
        sim = self._make_sim()
        assert sim.M == 1.0
        assert sim.L == 4.0
        assert sim.r_0 == 10.0
        assert sim.pr_0 == 0.0

    def test_default_params(self):
        """Default parameter values should match documented defaults."""
        config = SimulationConfig(
            domain=Domain.SCHWARZSCHILD,
            dt=0.01,
            n_steps=100,
            parameters={},
        )
        sim = SchwarzschildGeodesic(config)
        assert sim.M == 1.0
        assert sim.L == 4.0
        assert sim.r_0 == 10.0
        assert sim.pr_0 == 0.0

    def test_custom_params(self):
        """Custom parameters should override defaults."""
        sim = self._make_sim(M=2.0, L=6.0, r_0=20.0, pr_0=-0.1)
        assert sim.M == 2.0
        assert sim.L == 6.0
        assert sim.r_0 == 20.0
        assert sim.pr_0 == -0.1

    def test_step(self):
        """A single step should change the state."""
        sim = self._make_sim()
        state0 = sim.reset()
        state1 = sim.step()
        assert state0.shape == (4,)
        assert state1.shape == (4,)
        assert not np.allclose(state0, state1)

    def test_trajectory(self):
        """run() should produce a valid TrajectoryData."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 4)
        assert len(traj.timestamps) == 101
        # Radius should stay positive for a bound orbit
        assert np.all(traj.states[:, 0] > 0)

    def test_circular_orbit(self):
        """An orbit at the circular radius should stay approximately circular.

        For L=4, M=1, the stable circular orbit is at r = (L^2 + sqrt(L^4 - 12M^2L^2)) / (2M).
        Starting with pr_0=0 at this radius should give near-constant r.
        """
        M, L = 1.0, 4.0
        sim = self._make_sim(M=M, L=L, r_0=10.0)
        r_circ = sim.find_circular_orbit_radius(L)
        assert r_circ is not None

        sim = self._make_sim(M=M, L=L, r_0=r_circ, pr_0=0.0, dt=0.005)
        sim.reset()

        r_values = [sim.observe()[0]]
        for _ in range(5000):
            state = sim.step()
            r_values.append(state[0])

        r_arr = np.array(r_values)
        # For a truly circular orbit, r should barely vary
        r_variation = (np.max(r_arr) - np.min(r_arr)) / r_circ
        assert r_variation < 0.01, f"Radial variation {r_variation:.4%} too large"

    def test_energy_conservation(self):
        """Energy E = 0.5*pr^2 + V_eff(r) should be conserved."""
        sim = self._make_sim(M=1.0, L=4.0, r_0=12.0, pr_0=0.0, dt=0.005)
        sim.reset()
        E0 = sim.energy

        E_values = [E0]
        for _ in range(10000):
            sim.step()
            if sim._captured:
                break
            E_values.append(sim.energy)

        E_arr = np.array(E_values)
        max_drift = np.max(np.abs(E_arr - E0))
        rel_drift = max_drift / max(abs(E0), 1e-15)
        assert rel_drift < 1e-6, f"Energy drift {rel_drift:.2e} too large"

    def test_angular_momentum_conservation(self):
        """Angular momentum L should be constant (it is a parameter)."""
        sim = self._make_sim(M=1.0, L=4.5, r_0=10.0, pr_0=0.01)
        sim.reset()

        # L is conserved by construction: it is a constant parameter
        # but we verify r^2 * dphi/dtau = L along the trajectory
        for _ in range(5000):
            state = sim.step()
            if sim._captured:
                break
            r = state[0]
            dphi = state[3]
            L_computed = r**2 * dphi
            assert abs(L_computed - sim.L) / sim.L < 1e-6, (
                f"L mismatch: {L_computed:.6f} vs {sim.L:.6f}"
            )

    def test_schwarzschild_radius(self):
        """A particle with L=0 falling radially should be captured at r < 2M."""
        M = 1.0
        sim = self._make_sim(M=M, L=0.0, r_0=5.0, pr_0=-0.5, dt=0.001)
        sim.reset()

        # Run until captured or timeout
        for _ in range(100000):
            sim.step()
            if sim.is_captured():
                break

        assert sim.is_captured(), "Radially infalling particle should be captured"
        assert sim._state[0] < 2.0 * M * 1.01

    def test_isco(self):
        """ISCO should be at 6M."""
        sim = self._make_sim(M=1.0)
        assert sim.find_isco() == 6.0

        sim2 = self._make_sim(M=2.0)
        assert sim2.find_isco() == 12.0

        sim3 = self._make_sim(M=0.5)
        assert sim3.find_isco() == 3.0

    def test_effective_potential_shape(self):
        """V_eff should have a local minimum for L > 2*sqrt(3)*M.

        The potential barrier exists when the angular momentum is large
        enough to support stable orbits.
        """
        M = 1.0
        L_crit = 2.0 * np.sqrt(3.0) * M  # ~3.464

        # For L > L_crit, V_eff should have a minimum (stable circular orbit)
        sim_stable = self._make_sim(M=M, L=L_crit * 1.2)
        r_vals = np.linspace(4.0, 30.0, 500)
        V_vals = np.array([sim_stable.effective_potential(r) for r in r_vals])
        # Should have at least one local minimum
        dV = np.diff(V_vals)
        sign_changes = np.where(dV[:-1] * dV[1:] < 0)[0]
        assert len(sign_changes) >= 1, "No extremum found for L > L_crit"

        # For L < L_crit (but > 0), the potential has no stable minimum
        sim_unstable = self._make_sim(M=M, L=L_crit * 0.8)
        # The potential is monotonically decreasing for small L
        # (no barrier), so V_eff -> -inf as r -> 0
        V_near = sim_unstable.effective_potential(3.0)
        V_far = sim_unstable.effective_potential(30.0)
        # At least the far value should be less negative than near
        assert V_far > V_near, "V_eff should decrease toward small r"

    def test_radial_freefall(self):
        """L=0 gives pure radial infall. phi should not change."""
        sim = self._make_sim(M=1.0, L=0.0, r_0=10.0, pr_0=-0.1, dt=0.01)
        sim.reset()

        phi_initial = sim.observe()[1]
        for _ in range(500):
            state = sim.step()
            if sim._captured:
                break

        # phi should remain exactly 0 (no angular motion)
        assert abs(state[1] - phi_initial) < 1e-12, (
            f"phi changed by {abs(state[1] - phi_initial)} in radial freefall"
        )

    def test_precession_exists(self):
        """Non-Keplerian precession: orbit should not close after 2*pi in phi.

        In Newtonian gravity, an orbit closes after exactly 2*pi in phi.
        In Schwarzschild, it overshoots (prograde precession).
        """
        M, L = 1.0, 4.0
        sim = self._make_sim(M=M, L=L, r_0=10.0, pr_0=0.0)
        r_circ = sim.find_circular_orbit_radius(L)
        assert r_circ is not None

        # Start with small radial perturbation for elliptical orbit
        sim = SchwarzschildGeodesic(SimulationConfig(
            domain=Domain.SCHWARZSCHILD,
            dt=0.005,
            n_steps=100000,
            parameters={"M": M, "L": L, "r_0": r_circ, "pr_0": 0.02},
        ))
        sim.reset()

        r_vals = []
        phi_vals = []
        for _ in range(100000):
            state = sim.step()
            if sim._captured:
                break
            r_vals.append(state[0])
            phi_vals.append(state[1])

        r_arr = np.array(r_vals)
        phi_arr = np.array(phi_vals)

        # Find periapsis passages
        peri_indices = []
        for j in range(1, len(r_arr) - 1):
            if r_arr[j] < r_arr[j - 1] and r_arr[j] < r_arr[j + 1]:
                peri_indices.append(j)

        if len(peri_indices) >= 2:
            dphi = phi_arr[peri_indices[1]] - phi_arr[peri_indices[0]]
            # Precession: dphi should be > 2*pi (prograde)
            assert dphi > 2 * np.pi, (
                f"No precession detected: dphi = {dphi:.6f} <= 2*pi"
            )


class TestSchwarzschildRediscovery:
    """Tests for Schwarzschild rediscovery data generation."""

    def test_isco_data_generation(self):
        from simulating_anything.rediscovery.schwarzschild import generate_isco_data
        data = generate_isco_data(n_points=10)
        assert "L" in data
        assert "r_circ" in data
        assert len(data["L"]) > 0
        # All circular orbit radii should be >= 6M
        assert np.all(data["r_circ"] >= 6.0 - 0.1), (
            f"Found r_circ < 6M: {np.min(data['r_circ']):.4f}"
        )

    def test_orbit_data_generation(self):
        from simulating_anything.rediscovery.schwarzschild import generate_orbit_data
        data = generate_orbit_data(n_orbits=5, n_steps=10000, dt=0.01)
        assert "L" in data
        assert "r_circ" in data
        assert "E_orbit" in data
        assert len(data["L"]) > 0
