"""Delayed predator-prey simulation with Holling Type II functional response.

The delayed Lotka-Volterra equations with logistic prey growth:
    dN/dt = r*N*(1 - N/K) - a*N*P/(1 + h*a*N)
    dP/dt = e*a*N(t-tau)*P/(1 + h*a*N(t-tau)) - m*P

The delay tau in predator reproduction models the maturation lag between
prey consumption and predator birth. For tau=0 this reduces to the standard
Rosenzweig-MacArthur model.

Key dynamical properties:
- Delay destabilizes the interior equilibrium via Hopf bifurcation
- Critical delay tau_c: above which sustained oscillations emerge
- Period of oscillations increases with tau
- Can show period-doubling cascades for very large tau

Target rediscoveries:
- Critical delay tau_c for Hopf bifurcation onset
- Period scaling T(tau)
- Equilibrium (N*, P*) for the no-delay case
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class DelayedPredatorPreySimulation(SimulationEnvironment):
    """Delayed predator-prey model with Holling Type II functional response.

    State: [N, P] where N = prey density, P = predator density.
    Internal: ring buffer storing past prey values for delayed term N(t - tau).

    Parameters:
        r: prey intrinsic growth rate (default 1.0)
        K: prey carrying capacity (default 3.0)
        a: attack rate (default 0.5)
        h: handling time (default 0.1)
        e: conversion efficiency (default 0.6)
        m: predator mortality rate (default 0.4)
        tau: maturation delay (default 2.0)
        N_0: initial prey density (default 2.0)
        P_0: initial predator density (default 1.0)

    Note on K: A moderate carrying capacity (K=3) keeps N*/K ~ 0.48, which
    ensures the no-delay Rosenzweig-MacArthur system is stable while allowing
    the delay to cleanly destabilize the equilibrium via Hopf bifurcation.
    Large K (e.g. K=10) triggers the "paradox of enrichment" where even
    tau=0 produces violent oscillations that drive populations to numerical zero.
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 1.0)
        self.K = p.get("K", 3.0)
        self.a = p.get("a", 0.5)
        self.h = p.get("h", 0.1)
        self.e = p.get("e", 0.6)
        self.m = p.get("m", 0.4)
        self.tau = p.get("tau", 2.0)
        self.N_0 = p.get("N_0", 2.0)
        self.P_0 = p.get("P_0", 1.0)

        # Ring buffer for delayed prey values
        self._buf_size = max(int(self.tau / config.dt) + 1, 1)
        self._buffer: np.ndarray = np.empty(0)
        self._buf_idx = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize populations and fill history buffer with N_0."""
        self._state = np.array([self.N_0, self.P_0], dtype=np.float64)
        self._step_count = 0

        # Fill history buffer with initial prey value
        self._buffer = np.full(self._buf_size, self.N_0, dtype=np.float64)
        self._buf_idx = 0
        return self._state.copy()

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with fixed delayed term.

        The delayed prey value N(t - tau) is read from the ring buffer and
        held constant during the RK4 substeps (standard for DDE solvers).
        RK4 provides much better accuracy for the stiff predator-prey dynamics
        than Euler, preventing spurious extinction from overshoot.
        """
        dt = self.config.dt

        # Delayed prey value from ring buffer (fixed for this step)
        N_delayed = self._buffer[self._buf_idx]
        fr_delayed = self.functional_response(N_delayed)

        y = self._state
        k1 = self._derivatives(y, fr_delayed)
        k2 = self._derivatives(np.maximum(y + 0.5 * dt * k1, 0.0), fr_delayed)
        k3 = self._derivatives(np.maximum(y + 0.5 * dt * k2, 0.0), fr_delayed)
        k4 = self._derivatives(np.maximum(y + dt * k3, 0.0), fr_delayed)

        new_state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Ensure non-negative populations
        new_state = np.maximum(new_state, 0.0)

        # Store new prey value in ring buffer (overwrite oldest)
        self._buffer[self._buf_idx] = new_state[0]
        self._buf_idx = (self._buf_idx + 1) % self._buf_size

        self._state = new_state
        self._step_count += 1
        return self._state.copy()

    def _derivatives(
        self, y: np.ndarray, fr_delayed: float,
    ) -> np.ndarray:
        """Right-hand side of the delayed predator-prey system.

        Args:
            y: Current state [N, P].
            fr_delayed: Functional response evaluated at N(t - tau).
        """
        N, P = y
        dN = self.r * N * (1.0 - N / self.K) - self.functional_response(N) * P
        dP = self.e * fr_delayed * P - self.m * P
        return np.array([dN, dP])

    def observe(self) -> np.ndarray:
        """Return current state [N, P]."""
        return self._state.copy()

    def functional_response(self, N: float) -> float:
        """Holling Type II functional response: a*N / (1 + h*a*N).

        Saturates at 1/h for large N, representing handling-time limitation.
        """
        return self.a * N / (1.0 + self.h * self.a * N)

    def find_equilibrium(self) -> tuple[float, float]:
        """Find the interior equilibrium (N*, P*) numerically.

        At equilibrium with no delay effect (steady state):
            r*N*(1 - N/K) = a*N*P/(1 + h*a*N)
            e*a*N/(1 + h*a*N) = m

        From the predator equation: N* = m / (a*(e - m*h))
        Then P* from the prey equation.

        Returns:
            Tuple (N_star, P_star). Returns (0, 0) if no valid equilibrium.
        """
        denom = self.a * (self.e - self.m * self.h)
        if denom <= 0:
            return (0.0, 0.0)

        N_star = self.m / denom
        if N_star <= 0 or N_star >= self.K:
            return (0.0, 0.0)

        # From prey equation at equilibrium:
        # r*N*(1 - N/K) = (a*N/(1+h*a*N)) * P
        fr_star = self.functional_response(N_star)
        if fr_star <= 0:
            return (0.0, 0.0)

        P_star = self.r * N_star * (1.0 - N_star / self.K) / fr_star
        if P_star < 0:
            return (0.0, 0.0)

        return (float(N_star), float(P_star))

    def delay_sweep(
        self,
        tau_values: np.ndarray,
        n_steps: int = 50000,
        n_transient: int = 20000,
    ) -> dict[str, np.ndarray]:
        """Sweep delay values and measure oscillation amplitude and period.

        For each tau, runs the system, discards transient, then measures
        the peak-to-trough amplitude of prey and the dominant period via FFT.

        Args:
            tau_values: Array of delay values to sweep.
            n_steps: Steps after transient for measurement.
            n_transient: Steps to discard as transient.

        Returns:
            Dict with tau, amplitude, and period arrays.
        """
        amplitudes = []
        periods = []

        for tau_val in tau_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    "r": self.r, "K": self.K, "a": self.a,
                    "h": self.h, "e": self.e, "m": self.m,
                    "tau": tau_val, "N_0": self.N_0, "P_0": self.P_0,
                },
            )
            sim = DelayedPredatorPreySimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient):
                sim.step()

            # Collect prey values
            N_vals = []
            for _ in range(n_steps):
                state = sim.step()
                N_vals.append(state[0])

            N_arr = np.array(N_vals)
            amp = float(np.max(N_arr) - np.min(N_arr))
            amplitudes.append(amp)

            # Period via FFT
            T = self._fft_period(N_arr, self.config.dt)
            periods.append(T)

        return {
            "tau": tau_values,
            "amplitude": np.array(amplitudes),
            "period": np.array(periods),
        }

    def hopf_bifurcation_detect(
        self,
        tau_values: np.ndarray,
        n_steps: int = 50000,
        n_transient: int = 20000,
        amplitude_threshold: float = 0.05,
    ) -> float | None:
        """Detect critical delay tau_c where Hopf bifurcation occurs.

        Sweeps tau values and finds the first value where oscillation
        amplitude exceeds the threshold.

        Args:
            tau_values: Sorted array of delay values to test.
            n_steps: Steps after transient for measurement.
            n_transient: Steps to skip as transient.
            amplitude_threshold: Min amplitude to count as oscillating.

        Returns:
            Estimated tau_c, or None if no bifurcation detected.
        """
        for tau_val in tau_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    "r": self.r, "K": self.K, "a": self.a,
                    "h": self.h, "e": self.e, "m": self.m,
                    "tau": tau_val, "N_0": self.N_0, "P_0": self.P_0,
                },
            )
            sim = DelayedPredatorPreySimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient):
                sim.step()

            # Measure amplitude
            N_vals = []
            for _ in range(n_steps):
                state = sim.step()
                N_vals.append(state[0])

            N_arr = np.array(N_vals)
            amp = float(np.max(N_arr) - np.min(N_arr))

            if amp > amplitude_threshold:
                return float(tau_val)

        return None

    def compute_period(self, n_steps: int = 50000) -> float:
        """Measure dominant oscillation period via FFT.

        Runs the current simulation for n_steps (skipping transient),
        then extracts the dominant frequency from the prey time series.

        Returns:
            Period in time units. Returns 0.0 if no clear oscillation.
        """
        dt = self.config.dt

        # Transient
        for _ in range(20000):
            self.step()

        # Collect data
        N_vals = []
        for _ in range(n_steps):
            state = self.step()
            N_vals.append(state[0])

        return self._fft_period(np.array(N_vals), dt)

    def no_delay_comparison(self, n_steps: int = 50000) -> dict[str, np.ndarray]:
        """Run the system with tau=0 for comparison (Rosenzweig-MacArthur).

        Returns trajectory data and final state for the no-delay case.
        """
        config = SimulationConfig(
            domain=self.config.domain,
            dt=self.config.dt,
            n_steps=n_steps,
            parameters={
                "r": self.r, "K": self.K, "a": self.a,
                "h": self.h, "e": self.e, "m": self.m,
                "tau": 0.0, "N_0": self.N_0, "P_0": self.P_0,
            },
        )
        sim = DelayedPredatorPreySimulation(config)
        sim.reset()

        N_vals = []
        P_vals = []
        for _ in range(n_steps):
            state = sim.step()
            N_vals.append(state[0])
            P_vals.append(state[1])

        return {
            "N": np.array(N_vals),
            "P": np.array(P_vals),
            "final_state": sim.observe(),
        }

    @staticmethod
    def _fft_period(signal: np.ndarray, dt: float) -> float:
        """Extract dominant period from a time series using FFT.

        Returns 0.0 if no clear oscillation is detected.
        """
        # Remove mean
        signal = signal - np.mean(signal)
        if np.std(signal) < 1e-10:
            return 0.0

        n = len(signal)
        fft_vals = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(n, d=dt)

        # Skip DC component (index 0)
        if len(fft_vals) < 2:
            return 0.0

        peak_idx = np.argmax(fft_vals[1:]) + 1
        if freqs[peak_idx] <= 0:
            return 0.0

        return float(1.0 / freqs[peak_idx])
