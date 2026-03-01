"""Tests for the cable equation (passive neurite) simulation."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.cable_equation import CableEquationSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    tau_m: float = 10.0,
    lambda_e: float = 0.5,
    L: float = 5.0,
    N: int = 100,
    dt: float = 0.01,
    I0: float = 1.0,
    inject_x: float = 0.5,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.CABLE_EQUATION,
        dt=dt,
        n_steps=1000,
        parameters={
            "tau_m": tau_m,
            "lambda_e": lambda_e,
            "L": L,
            "N": float(N),
            "R_m": 1.0,
            "I0": I0,
            "inject_x": inject_x,
        },
    )


class TestCableEquation:
    def test_initial_rest(self):
        """V = 0 everywhere at t=0."""
        sim = CableEquationSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_array_equal(state, np.zeros(100))

    def test_current_injection(self):
        """V increases at injection site after steps with current."""
        sim = CableEquationSimulation(_make_config())
        sim.reset()
        for _ in range(100):
            sim.step()
        V = sim.observe()
        inject_idx = sim._inject_idx
        assert V[inject_idx] > 0.0

    def test_spatial_decay(self):
        """V decreases away from injection point."""
        sim = CableEquationSimulation(_make_config(N=100))
        sim.reset()
        for _ in range(5000):
            sim.step()
        V = sim.observe()
        inject_idx = sim._inject_idx
        # V should decrease monotonically from injection toward right boundary
        for i in range(inject_idx + 1, min(inject_idx + 20, sim.N - 1)):
            assert V[i] <= V[i - 1] + 1e-12, (
                f"V not decaying: V[{i}]={V[i]:.6f} > V[{i-1}]={V[i-1]:.6f}"
            )

    def test_exponential_profile(self):
        """Steady-state profile approximately matches exp(-|x|/lambda)."""
        sim = CableEquationSimulation(_make_config(
            lambda_e=1.0, L=10.0, N=200, dt=0.01,
        ))
        sim.reset()
        # Run to near steady state
        for _ in range(20000):
            sim.step()

        V = sim.observe()
        inject_idx = sim._inject_idx
        V_peak = V[inject_idx]
        assert V_peak > 0

        # Check a few points on the right side
        x_offset = sim.x[inject_idx + 10] - sim.x[inject_idx]
        V_at_offset = V[inject_idx + 10]
        expected_ratio = np.exp(-x_offset / 1.0)
        actual_ratio = V_at_offset / V_peak
        # Allow finite cable boundary effects to cause some deviation
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=0.15)

    def test_time_constant(self):
        """Response approaches steady state with timescale ~ tau_m."""
        sim = CableEquationSimulation(_make_config(tau_m=10.0))
        sim.reset()
        dt = 0.01
        inject_idx = sim._inject_idx

        # Sample voltage at injection point over time
        voltages = []
        for _ in range(3000):
            sim.step()
            voltages.append(sim.observe()[inject_idx])

        voltages = np.array(voltages)
        V_final = voltages[-1]

        # At t = tau_m, should have reached ~63% of steady state
        t_tau_idx = int(10.0 / dt)  # tau_m = 10
        if t_tau_idx < len(voltages) and V_final > 0:
            fraction = voltages[t_tau_idx] / V_final
            # Should be roughly 1 - exp(-1) ~ 0.632
            # Cable equation is not pure exponential at injection point
            # due to spatial spreading, so allow wide tolerance
            assert 0.3 < fraction < 0.95, (
                f"Fraction at t=tau_m: {fraction:.3f}"
            )

    def test_sealed_ends(self):
        """dV/dx ~ 0 at boundaries (Neumann BC)."""
        sim = CableEquationSimulation(_make_config(N=100))
        sim.reset()
        for _ in range(5000):
            sim.step()
        V = sim.observe()
        # Boundary gradient should be small
        grad_left = abs(V[1] - V[0]) / sim.dx
        grad_right = abs(V[-1] - V[-2]) / sim.dx
        # Compare to interior gradient near injection
        inject_idx = sim._inject_idx
        grad_interior = abs(V[inject_idx + 1] - V[inject_idx]) / sim.dx
        if grad_interior > 1e-10:
            assert grad_left < grad_interior
            assert grad_right < grad_interior

    def test_observe_shape(self):
        """observe() returns N-element array."""
        sim = CableEquationSimulation(_make_config(N=80))
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (80,)

    def test_deterministic(self):
        """Same parameters produce same result."""
        config = _make_config()
        sim1 = CableEquationSimulation(config)
        sim1.reset()
        for _ in range(500):
            sim1.step()
        V1 = sim1.observe().copy()

        sim2 = CableEquationSimulation(config)
        sim2.reset()
        for _ in range(500):
            sim2.step()
        V2 = sim2.observe()

        np.testing.assert_array_equal(V1, V2)

    def test_stability(self):
        """No NaN or Inf after many steps."""
        sim = CableEquationSimulation(_make_config())
        sim.reset()
        for _ in range(10000):
            sim.step()
        V = sim.observe()
        assert np.all(np.isfinite(V))

    def test_step_advances(self):
        """State changes after current injection and stepping."""
        sim = CableEquationSimulation(_make_config())
        s0 = sim.reset().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_reset(self):
        """Reset returns V=0 everywhere."""
        sim = CableEquationSimulation(_make_config())
        sim.reset()
        for _ in range(100):
            sim.step()
        state = sim.reset()
        np.testing.assert_array_equal(state, np.zeros(sim.N))

    def test_symmetry(self):
        """Point injection at center gives symmetric profile."""
        sim = CableEquationSimulation(_make_config(
            N=101, inject_x=0.5, L=5.0,
        ))
        sim.reset()
        for _ in range(5000):
            sim.step()
        V = sim.observe()
        inject_idx = sim._inject_idx
        # Compare symmetric pairs around injection point
        n_check = min(inject_idx, sim.N - 1 - inject_idx, 20)
        for offset in range(1, n_check):
            np.testing.assert_allclose(
                V[inject_idx - offset],
                V[inject_idx + offset],
                rtol=1e-6,
                err_msg=f"Asymmetry at offset {offset}",
            )

    def test_lambda_effect(self):
        """Larger lambda -> broader spatial spread."""
        def get_spread(lam: float) -> float:
            config = _make_config(lambda_e=lam, L=10.0, N=200, dt=0.01)
            sim = CableEquationSimulation(config)
            sim.reset()
            for _ in range(15000):
                sim.step()
            V = sim.observe()
            # Width at half-max
            V_half = V[sim._inject_idx] / 2.0
            above_half = np.where(V > V_half)[0]
            if len(above_half) > 0:
                return float(
                    (above_half[-1] - above_half[0]) * sim.dx
                )
            return 0.0

        spread_small = get_spread(0.3)
        spread_large = get_spread(1.5)
        assert spread_large > spread_small

    def test_tau_effect(self):
        """Larger tau -> slower approach to steady state."""
        def get_rise_fraction(tau: float) -> float:
            config = _make_config(tau_m=tau, dt=0.01)
            sim = CableEquationSimulation(config)
            sim.reset()
            # Run fixed number of steps
            for _ in range(500):
                sim.step()
            V_500 = sim.observe()[sim._inject_idx]
            # Run many more to get closer to steady state
            for _ in range(20000):
                sim.step()
            V_ss = sim.observe()[sim._inject_idx]
            return V_500 / V_ss if V_ss > 0 else 0.0

        frac_fast = get_rise_fraction(5.0)
        frac_slow = get_rise_fraction(50.0)
        # Faster tau should reach higher fraction in fixed time
        assert frac_fast > frac_slow

    def test_superposition(self):
        """Two injections sum linearly (linear system)."""
        N = 100
        L = 5.0
        dt = 0.01
        n_steps = 5000

        # Single injection at x=1.0
        config1 = _make_config(N=N, L=L, dt=dt, inject_x=0.2)
        sim1 = CableEquationSimulation(config1)
        sim1.reset()
        for _ in range(n_steps):
            sim1.step()
        V1 = sim1.observe().copy()

        # Single injection at x=4.0
        config2 = _make_config(N=N, L=L, dt=dt, inject_x=0.8)
        sim2 = CableEquationSimulation(config2)
        sim2.reset()
        for _ in range(n_steps):
            sim2.step()
        V2 = sim2.observe().copy()

        # Both injections simultaneously
        config_both = _make_config(N=N, L=L, dt=dt, inject_x=0.2)
        sim_both = CableEquationSimulation(config_both)
        # Add second injection
        I_ext = np.zeros(N)
        idx1 = int(0.2 * (N - 1))
        idx2 = int(0.8 * (N - 1))
        I_ext[idx1] = 1.0 / sim_both.dx
        I_ext[idx2] = 1.0 / sim_both.dx
        sim_both.set_current(I_ext)
        sim_both.reset()
        # Reset sets V=0 but does not touch I_ext, re-set it after reset
        sim_both.set_current(I_ext)
        for _ in range(n_steps):
            sim_both.step()
        V_both = sim_both.observe()

        np.testing.assert_allclose(V_both, V1 + V2, rtol=1e-4)

    def test_bounded(self):
        """V stays finite and bounded."""
        sim = CableEquationSimulation(_make_config())
        sim.reset()
        for _ in range(10000):
            sim.step()
        V = sim.observe()
        assert np.all(np.isfinite(V))
        assert np.max(np.abs(V)) < 1e6

    def test_space_constant_measurement(self):
        """Measured lambda matches parameter within tolerance."""
        config = _make_config(
            lambda_e=0.5, L=10.0, N=200, dt=0.01,
        )
        sim = CableEquationSimulation(config)
        sim.reset()
        for _ in range(20000):
            sim.step()
        lam_meas = sim.measure_space_constant()
        # Allow 20% tolerance due to finite cable + boundary effects
        np.testing.assert_allclose(lam_meas, 0.5, rtol=0.20)


class TestCableEquationRediscovery:
    def test_rediscovery_data(self):
        """Data generation works and produces valid sweep."""
        from simulating_anything.rediscovery.cable_equation import (
            generate_space_constant_data,
        )
        data = generate_space_constant_data(
            n_lambda=5, n_steps=3000, dt=0.01, N=60,
        )
        assert len(data["lambda_set"]) == 5
        assert len(data["lambda_measured"]) == 5
        valid = (data["lambda_measured"] > 0) & np.isfinite(
            data["lambda_measured"]
        )
        assert np.sum(valid) >= 2
