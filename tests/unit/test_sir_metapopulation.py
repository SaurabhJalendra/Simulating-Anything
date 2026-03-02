"""Tests for the SIR Metapopulation (multi-patch SIR with migration) model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.sir_metapopulation import (
    SIRMetapopulationSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    dt: float = 0.1,
    n_steps: int = 2000,
    **param_overrides: float,
) -> SimulationConfig:
    """Create a SimulationConfig for SIR metapopulation with optional overrides."""
    params: dict[str, float] = {
        "n_patches": 5.0,
        "beta": 0.5,
        "gamma": 0.1,
        "m": 0.01,
        "N_per_patch": 1000.0,
    }
    params.update(param_overrides)
    return SimulationConfig(
        domain=Domain.SIR_METAPOPULATION,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestSIRMetapopulationCreation:
    """Tests for simulation creation and initialization."""

    def test_create_simulation(self):
        """Simulation should be created with default parameters."""
        sim = SIRMetapopulationSimulation(_make_config())
        assert sim.n_patches == 5
        assert sim.beta == 0.5
        assert sim.gamma == 0.1
        assert sim.m == 0.01
        assert sim.N_per_patch == 1000.0

    def test_initial_state_shape(self):
        """State vector should be 3 * n_patches."""
        sim = SIRMetapopulationSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (15,)

    def test_initial_state_shape_3_patches(self):
        """State should be (9,) for 3 patches."""
        sim = SIRMetapopulationSimulation(_make_config(n_patches=3.0))
        state = sim.reset()
        assert state.shape == (9,)

    def test_initial_state_shape_10_patches(self):
        """State should be (30,) for 10 patches."""
        sim = SIRMetapopulationSimulation(_make_config(n_patches=10.0))
        state = sim.reset()
        assert state.shape == (30,)

    def test_initial_patch0_seeded(self):
        """Patch 0 should have 10 infected initially."""
        sim = SIRMetapopulationSimulation(_make_config())
        state = sim.reset()
        # Patch 0: [S, I, R] = [990, 10, 0]
        assert state[0] == pytest.approx(990.0)
        assert state[1] == pytest.approx(10.0)
        assert state[2] == pytest.approx(0.0)

    def test_initial_other_patches_uninfected(self):
        """Patches 1..n should have 0 infected initially."""
        sim = SIRMetapopulationSimulation(_make_config())
        state = sim.reset()
        for i in range(1, 5):
            assert state[3 * i] == pytest.approx(1000.0)
            assert state[3 * i + 1] == pytest.approx(0.0)
            assert state[3 * i + 2] == pytest.approx(0.0)

    def test_custom_parameters(self):
        """Custom parameters should override defaults."""
        sim = SIRMetapopulationSimulation(_make_config(
            beta=0.3, gamma=0.2, m=0.05, n_patches=3.0,
        ))
        assert sim.beta == 0.3
        assert sim.gamma == 0.2
        assert sim.m == 0.05
        assert sim.n_patches == 3


class TestSIRMetapopulationBasicSimulation:
    """Core simulation mechanics tests."""

    def test_step_advances(self):
        """State should change after a step."""
        sim = SIRMetapopulationSimulation(_make_config())
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_returns_state(self):
        """Observe should return the current state."""
        sim = SIRMetapopulationSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (15,)

    def test_run_method(self):
        """run() should produce a TrajectoryData object."""
        sim = SIRMetapopulationSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 15)
        assert len(traj.timestamps) == 101

    def test_deterministic(self):
        """Same config produces same trajectory."""
        config = _make_config()
        sim1 = SIRMetapopulationSimulation(config)
        sim1.reset()
        for _ in range(200):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = SIRMetapopulationSimulation(config)
        sim2.reset()
        for _ in range(200):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)

    def test_long_run_stability(self):
        """No NaN after many steps."""
        sim = SIRMetapopulationSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(10000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN or Inf in state: {state}"


class TestSIRMetapopulationConservation:
    """Conservation law tests -- total population S+I+R should be conserved per patch
    in a mean-field migration model (migration is mass-conserving)."""

    def test_total_population_conservation(self):
        """Total population across all patches should be approximately conserved."""
        sim = SIRMetapopulationSimulation(_make_config())
        sim.reset()
        total_N = sim.N_per_patch * sim.n_patches
        for _ in range(500):
            state = sim.step()
            np.testing.assert_allclose(
                state.sum(), total_N, atol=1e-4,
                err_msg="Total population conservation violated",
            )

    def test_initial_total_population(self):
        """Total population should equal n_patches * N_per_patch at initialization."""
        sim = SIRMetapopulationSimulation(_make_config())
        state = sim.reset()
        expected = 5 * 1000.0
        np.testing.assert_allclose(state.sum(), expected, atol=1e-10)

    def test_conservation_long_run(self):
        """Conservation should hold over long simulation."""
        sim = SIRMetapopulationSimulation(_make_config())
        sim.reset()
        total_N = sim.N_per_patch * sim.n_patches
        for _ in range(5000):
            state = sim.step()
        np.testing.assert_allclose(
            state.sum(), total_N, atol=0.1,
            err_msg="Conservation violated over long run",
        )

    def test_per_patch_population_approximately_conserved(self):
        """With zero migration, per-patch population should be exactly conserved."""
        sim = SIRMetapopulationSimulation(_make_config(m=0.0))
        sim.reset()
        for _ in range(500):
            state = sim.step()
            for i in range(5):
                patch_pop = state[3 * i] + state[3 * i + 1] + state[3 * i + 2]
                np.testing.assert_allclose(
                    patch_pop, 1000.0, atol=1e-6,
                    err_msg=f"Patch {i} population not conserved: {patch_pop}",
                )


class TestSIRMetapopulationNonNegativity:
    """All compartments must remain non-negative."""

    def test_non_negative_all_compartments(self):
        """All S, I, R compartments should remain >= 0."""
        sim = SIRMetapopulationSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative compartment: {state[state < 0]}"

    def test_non_negative_high_beta(self):
        """Non-negativity should hold with high transmission rate."""
        sim = SIRMetapopulationSimulation(_make_config(beta=2.0))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative compartment: {state}"

    def test_non_negative_high_migration(self):
        """Non-negativity should hold with high migration rate."""
        sim = SIRMetapopulationSimulation(_make_config(m=0.5))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative compartment: {state}"


class TestSIRMetapopulationEpidemicSpread:
    """Tests for epidemic spreading between patches."""

    def test_epidemic_spreads_to_other_patches(self):
        """With migration, infection should reach all patches."""
        sim = SIRMetapopulationSimulation(_make_config(m=0.01))
        sim.reset()
        for _ in range(2000):
            sim.step()
        # All patches should have some recovered (indicating infection occurred)
        R_all = sim.get_all_recovered()
        for i in range(5):
            assert R_all[i] > 1.0, (
                f"Patch {i} recovered too low: {R_all[i]}"
            )

    def test_zero_migration_isolates_patches(self):
        """Without migration, only patch 0 should have an epidemic."""
        sim = SIRMetapopulationSimulation(_make_config(m=0.0))
        sim.reset()
        for _ in range(3000):
            sim.step()
        # Patch 0 should have recovered individuals
        R_all = sim.get_all_recovered()
        assert R_all[0] > 100.0, f"Patch 0 should have epidemic: R={R_all[0]}"
        # Other patches should have no recovered (no infection spread)
        for i in range(1, 5):
            assert R_all[i] == pytest.approx(0.0, abs=1e-10), (
                f"Patch {i} should be isolated: R={R_all[i]}"
            )

    def test_patch0_infected_rises_then_falls(self):
        """Patch 0 (seeded) should show epidemic curve."""
        sim = SIRMetapopulationSimulation(_make_config())
        sim.reset()
        max_I = 0.0
        for _ in range(2000):
            sim.step()
            I0 = sim.get_patch_state(0)[1]
            max_I = max(max_I, I0)
        # Peak should be significant
        assert max_I > 50.0, f"Peak I in patch 0 too low: {max_I}"
        # Should decay at the end
        final_I0 = sim.get_patch_state(0)[1]
        assert final_I0 < max_I, "Infected should decrease after peak"

    def test_epidemic_wave_delay(self):
        """Remote patches should peak later than the seeded patch."""
        sim = SIRMetapopulationSimulation(_make_config(m=0.01))
        peaks = sim.peak_infected_per_patch(n_steps=3000)
        # Patch 0 should peak first
        patch0_time = peaks["peak_time"][0]
        # At least one other patch should peak later
        other_peak_times = peaks["peak_time"][1:]
        assert np.any(other_peak_times > patch0_time), (
            f"No wave delay detected: patch0={patch0_time}, "
            f"others={other_peak_times}"
        )


class TestSIRMetapopulationMigrationSynchronization:
    """Tests for migration-induced synchronization."""

    def test_high_migration_synchronizes_peaks(self):
        """With high migration, all patches should peak at similar times."""
        sim = SIRMetapopulationSimulation(_make_config(m=1.0))
        peaks = sim.peak_infected_per_patch(n_steps=3000)
        peak_times = peaks["peak_time"]
        # High migration: peak time std should be small
        std = np.std(peak_times)
        assert std < 5.0, (
            f"High migration should synchronize peaks, but std={std:.2f}"
        )

    def test_low_migration_less_synchronized(self):
        """Low migration should show more spread in peak times."""
        # Run with low migration
        sim_low = SIRMetapopulationSimulation(_make_config(m=0.001))
        peaks_low = sim_low.peak_infected_per_patch(n_steps=5000)
        std_low = np.std(peaks_low["peak_time"])

        # Run with high migration
        sim_high = SIRMetapopulationSimulation(_make_config(m=1.0))
        peaks_high = sim_high.peak_infected_per_patch(n_steps=3000)
        std_high = np.std(peaks_high["peak_time"])

        assert std_low > std_high, (
            f"Low m should be less synchronized: std_low={std_low:.2f}, "
            f"std_high={std_high:.2f}"
        )

    def test_migration_increases_total_epidemic_size(self):
        """Migration should lead to larger total epidemic than isolated patches."""
        # With migration
        sim_with = SIRMetapopulationSimulation(_make_config(m=0.01))
        sim_with.reset()
        for _ in range(3000):
            sim_with.step()
        R_with = float(np.sum(sim_with.get_all_recovered()))

        # Without migration
        sim_without = SIRMetapopulationSimulation(_make_config(m=0.0))
        sim_without.reset()
        for _ in range(3000):
            sim_without.step()
        R_without = float(np.sum(sim_without.get_all_recovered()))

        assert R_with > R_without, (
            f"Migration should increase total R: with={R_with:.1f}, "
            f"without={R_without:.1f}"
        )


class TestSIRMetapopulationR0:
    """Tests for the basic reproduction number."""

    def test_r0_formula(self):
        """R0 should equal beta / gamma."""
        sim = SIRMetapopulationSimulation(_make_config(beta=0.5, gamma=0.1))
        assert sim.R0 == pytest.approx(5.0)

    def test_r0_different_params(self):
        """R0 should vary correctly with parameters."""
        sim = SIRMetapopulationSimulation(_make_config(beta=0.3, gamma=0.15))
        assert sim.R0 == pytest.approx(2.0)

    def test_r0_subcritical_no_spread(self):
        """With R0 < 1, epidemic should die out even in seeded patch."""
        sim = SIRMetapopulationSimulation(_make_config(beta=0.05, gamma=0.1, m=0.01))
        sim.reset()
        initial_I = sim.total_infected()
        for _ in range(3000):
            sim.step()
        final_I = sim.total_infected()
        assert final_I < initial_I, (
            f"R0 < 1 should cause decay: initial={initial_I}, final={final_I}"
        )


class TestSIRMetapopulationHelperMethods:
    """Tests for helper/accessor methods."""

    def test_get_patch_state(self):
        """get_patch_state should return [S, I, R] for a given patch."""
        sim = SIRMetapopulationSimulation(_make_config())
        sim.reset()
        ps = sim.get_patch_state(0)
        assert ps.shape == (3,)
        assert ps[0] == pytest.approx(990.0)
        assert ps[1] == pytest.approx(10.0)
        assert ps[2] == pytest.approx(0.0)

    def test_get_all_infected(self):
        """get_all_infected should return array of shape (n_patches,)."""
        sim = SIRMetapopulationSimulation(_make_config())
        sim.reset()
        I_all = sim.get_all_infected()
        assert I_all.shape == (5,)
        assert I_all[0] == pytest.approx(10.0)
        for i in range(1, 5):
            assert I_all[i] == pytest.approx(0.0)

    def test_get_all_susceptible(self):
        """get_all_susceptible should return array of shape (n_patches,)."""
        sim = SIRMetapopulationSimulation(_make_config())
        sim.reset()
        S_all = sim.get_all_susceptible()
        assert S_all.shape == (5,)
        assert S_all[0] == pytest.approx(990.0)

    def test_get_all_recovered(self):
        """get_all_recovered should return array of shape (n_patches,)."""
        sim = SIRMetapopulationSimulation(_make_config())
        sim.reset()
        R_all = sim.get_all_recovered()
        assert R_all.shape == (5,)
        assert np.allclose(R_all, 0.0)

    def test_total_infected(self):
        """total_infected should return sum of all I compartments."""
        sim = SIRMetapopulationSimulation(_make_config())
        sim.reset()
        assert sim.total_infected() == pytest.approx(10.0)

    def test_total_population(self):
        """total_population should return sum of all compartments."""
        sim = SIRMetapopulationSimulation(_make_config())
        sim.reset()
        assert sim.total_population() == pytest.approx(5000.0)

    def test_patch_population(self):
        """patch_population should return S+I+R for a specific patch."""
        sim = SIRMetapopulationSimulation(_make_config())
        sim.reset()
        for i in range(5):
            assert sim.patch_population(i) == pytest.approx(1000.0)

    def test_peak_infected_per_patch(self):
        """peak_infected_per_patch should return peak info for all patches."""
        sim = SIRMetapopulationSimulation(_make_config())
        peaks = sim.peak_infected_per_patch(n_steps=1000)
        assert peaks["peak_I"].shape == (5,)
        assert peaks["peak_time"].shape == (5,)
        # Patch 0 should have significant peak
        assert peaks["peak_I"][0] > 10.0


class TestSIRMetapopulationParameterSweeps:
    """Tests for parameter variation behavior."""

    def test_higher_beta_faster_epidemic(self):
        """Higher beta should lead to earlier peak in seeded patch."""
        sim_low = SIRMetapopulationSimulation(_make_config(beta=0.3))
        peaks_low = sim_low.peak_infected_per_patch(n_steps=3000)

        sim_high = SIRMetapopulationSimulation(_make_config(beta=0.8))
        peaks_high = sim_high.peak_infected_per_patch(n_steps=3000)

        assert peaks_high["peak_time"][0] < peaks_low["peak_time"][0], (
            f"Higher beta should peak earlier: "
            f"beta=0.8 at {peaks_high['peak_time'][0]}, "
            f"beta=0.3 at {peaks_low['peak_time'][0]}"
        )

    def test_higher_gamma_lower_peak(self):
        """Higher recovery rate should produce lower peak infected."""
        sim_low_gamma = SIRMetapopulationSimulation(_make_config(gamma=0.05))
        peaks_low = sim_low_gamma.peak_infected_per_patch(n_steps=3000)

        sim_high_gamma = SIRMetapopulationSimulation(_make_config(gamma=0.3))
        peaks_high = sim_high_gamma.peak_infected_per_patch(n_steps=3000)

        assert peaks_high["peak_I"][0] < peaks_low["peak_I"][0], (
            f"Higher gamma should lower peak: "
            f"gamma=0.3 peak={peaks_high['peak_I'][0]:.1f}, "
            f"gamma=0.05 peak={peaks_low['peak_I'][0]:.1f}"
        )

    def test_more_patches_same_dynamics(self):
        """Changing n_patches shouldn't change patch 0 dynamics (for m=0)."""
        sim_3 = SIRMetapopulationSimulation(_make_config(n_patches=3.0, m=0.0))
        sim_3.reset()
        for _ in range(500):
            sim_3.step()
        state_3 = sim_3.get_patch_state(0)

        sim_10 = SIRMetapopulationSimulation(_make_config(n_patches=10.0, m=0.0))
        sim_10.reset()
        for _ in range(500):
            sim_10.step()
        state_10 = sim_10.get_patch_state(0)

        np.testing.assert_allclose(state_3, state_10, atol=1e-8)

    def test_n_per_patch_scales_counts(self):
        """Doubling N_per_patch should approximately double peak infected count."""
        sim_small = SIRMetapopulationSimulation(
            _make_config(N_per_patch=500.0, m=0.0)
        )
        peaks_small = sim_small.peak_infected_per_patch(n_steps=2000)

        sim_large = SIRMetapopulationSimulation(
            _make_config(N_per_patch=1000.0, m=0.0)
        )
        peaks_large = sim_large.peak_infected_per_patch(n_steps=2000)

        ratio = peaks_large["peak_I"][0] / peaks_small["peak_I"][0]
        assert 1.5 < ratio < 2.5, (
            f"Peak ratio should be ~2x: {ratio:.2f}"
        )


class TestSIRMetapopulationTrajectoryData:
    """Tests for trajectory data output."""

    def test_trajectory_states_shape(self):
        """Trajectory states should have correct shape."""
        sim = SIRMetapopulationSimulation(_make_config(n_steps=50))
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 15)

    def test_trajectory_timestamps(self):
        """Timestamps should be evenly spaced."""
        sim = SIRMetapopulationSimulation(_make_config(n_steps=50, dt=0.1))
        traj = sim.run(n_steps=50)
        np.testing.assert_allclose(traj.timestamps[0], 0.0)
        np.testing.assert_allclose(traj.timestamps[-1], 5.0)
        dt_diffs = np.diff(traj.timestamps)
        np.testing.assert_allclose(dt_diffs, 0.1, atol=1e-10)

    def test_trajectory_parameters(self):
        """Trajectory parameters should match config."""
        sim = SIRMetapopulationSimulation(_make_config())
        traj = sim.run(n_steps=10)
        assert traj.parameters["beta"] == 0.5
        assert traj.parameters["gamma"] == 0.1
        assert traj.parameters["m"] == 0.01

    def test_trajectory_states_non_negative(self):
        """All states in trajectory should be non-negative."""
        sim = SIRMetapopulationSimulation(_make_config(n_steps=500))
        traj = sim.run(n_steps=500)
        assert np.all(traj.states >= 0), "Negative values in trajectory"


class TestSIRMetapopulationRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_generate_migration_sweep(self):
        """Migration sweep should produce valid arrays."""
        from simulating_anything.rediscovery.sir_metapopulation import (
            generate_migration_sweep_data,
        )
        data = generate_migration_sweep_data(
            n_m_values=5, n_patches=3, n_steps=500, dt=0.1,
        )
        assert "m_values" in data
        assert "peak_time_std" in data
        assert "total_final_size" in data
        assert "peak_times" in data
        assert len(data["m_values"]) == 5
        assert data["peak_times"].shape == (5, 3)
        assert np.all(np.isfinite(data["peak_time_std"]))

    def test_generate_epidemic_wave(self):
        """Epidemic wave data should have correct shape."""
        from simulating_anything.rediscovery.sir_metapopulation import (
            generate_epidemic_wave_data,
        )
        data = generate_epidemic_wave_data(
            n_patches=3, n_steps=200, dt=0.1,
        )
        assert data["I_timeseries"].shape == (201, 3)
        assert data["peak_times"].shape == (3,)
        assert data["peak_delays"].shape == (3,)
        assert data["peak_delays"][0] == pytest.approx(0.0)

    def test_compare_with_without_migration(self):
        """Comparison function should return valid results."""
        from simulating_anything.rediscovery.sir_metapopulation import (
            compare_with_without_migration,
        )
        result = compare_with_without_migration(
            n_patches=3, n_steps=1000, dt=0.1,
        )
        assert "total_R_with_migration" in result
        assert "total_R_without_migration" in result
        assert "fraction_with" in result
        assert "fraction_without" in result
        assert result["fraction_with"] >= 0
        assert result["fraction_without"] >= 0

    def test_migration_sweep_synchronization_trend(self):
        """Higher migration should produce lower peak time std."""
        from simulating_anything.rediscovery.sir_metapopulation import (
            generate_migration_sweep_data,
        )
        data = generate_migration_sweep_data(
            n_m_values=8, n_patches=3, n_steps=1500, dt=0.1,
        )
        # General trend: peak_time_std should decrease with increasing m
        low_m_std = np.mean(data["peak_time_std"][:3])
        high_m_std = np.mean(data["peak_time_std"][-3:])
        assert high_m_std <= low_m_std + 1.0, (
            f"High m should synchronize: low_m_std={low_m_std:.2f}, "
            f"high_m_std={high_m_std:.2f}"
        )

    def test_wave_data_parameters(self):
        """Wave data should include ground truth parameters."""
        from simulating_anything.rediscovery.sir_metapopulation import (
            generate_epidemic_wave_data,
        )
        data = generate_epidemic_wave_data(
            n_patches=3, beta=0.4, gamma=0.15, m=0.02, n_steps=200, dt=0.1,
        )
        assert data["beta"] == 0.4
        assert data["gamma"] == 0.15
        assert data["m"] == 0.02
        assert data["n_patches"] == 3
