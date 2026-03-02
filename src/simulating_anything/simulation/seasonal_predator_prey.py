"""Seasonal predator-prey model with periodic forcing on prey birth rate.

Lotka-Volterra with carrying capacity (logistic prey growth), Holling Type II
functional response, and seasonal modulation of the prey intrinsic growth rate:

    dN/dt = r(t)*N*(1 - N/K) - a*N*P/(1 + h*N)
    dP/dt = e*a*N*P/(1 + h*N) - d*P

where r(t) = r0*(1 + epsilon*sin(2*pi*t/T)) is the seasonally forced prey
birth rate. The forcing amplitude epsilon controls the transition from:
  - epsilon=0: standard Rosenzweig-MacArthur (fixed point or limit cycle)
  - small epsilon: entrained periodic orbit at period T
  - moderate epsilon: quasi-periodic dynamics (two incommensurate frequencies)
  - large epsilon: chaotic dynamics via period-doubling

Target rediscoveries:
- Seasonal forcing frequency from FFT of population time series
- Transition from limit cycle to quasi-periodic to chaos as epsilon increases
- Equilibrium recovery at epsilon=0 (non-seasonal LV with carrying capacity)
- Dominant frequency matches 1/T for entrained regime

Default parameters: r0=1.0, K=100.0, a=0.5, h=0.02, e=0.4, d=0.2,
                    epsilon=0.3, T=12.0 (monthly period)
Initial conditions: N=50.0, P=10.0
"""

from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SeasonalPredatorPreySimulation(SimulationEnvironment):
    """Lotka-Volterra predator-prey with seasonal forcing.

    State vector: [N, P] where N = prey density, P = predator density.

    The prey birth rate is periodically modulated:
        r(t) = r0 * (1 + epsilon * sin(2*pi*t/T))

    Combined with logistic prey growth and Holling Type II predation:
        dN/dt = r(t)*N*(1 - N/K) - a*N*P/(1 + h*N)
        dP/dt = e*a*N*P/(1 + h*N) - d*P

    Parameters:
        r0: base intrinsic growth rate (default 1.0)
        K: prey carrying capacity (default 100.0)
        a: predator attack rate (default 0.5)
        h: handling time per prey item (default 0.02)
        e: conversion efficiency, prey to predator biomass (default 0.4)
        d: predator death rate (default 0.2)
        epsilon: seasonal forcing amplitude, 0 <= epsilon <= 1 (default 0.3)
        T: seasonal period (default 12.0, representing months)
        N_0: initial prey density (default 50.0)
        P_0: initial predator density (default 10.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r0 = p.get("r0", 1.0)
        self.K = p.get("K", 100.0)
        self.a = p.get("a", 0.5)
        self.h = p.get("h", 0.02)
        self.e = p.get("e", 0.4)
        self.d = p.get("d", 0.2)
        self.epsilon = p.get("epsilon", 0.3)
        self.T = p.get("T", 12.0)
        self.N_0 = p.get("N_0", 50.0)
        self.P_0 = p.get("P_0", 10.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize populations [N, P]."""
        self._state = np.array([self.N_0, self.P_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with non-negativity enforcement."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current populations [N, P]."""
        return self._state

    def current_time(self) -> float:
        """Return the current simulation time."""
        return self._step_count * self.config.dt

    def seasonal_growth_rate(self, t: float) -> float:
        """Compute the seasonally modulated prey growth rate at time t.

        r(t) = r0 * (1 + epsilon * sin(2*pi*t/T))

        Args:
            t: Current time.

        Returns:
            The instantaneous prey growth rate.
        """
        return self.r0 * (1.0 + self.epsilon * np.sin(2.0 * np.pi * t / self.T))

    def holling_type2(self, N: float | np.ndarray) -> float | np.ndarray:
        """Holling Type II functional response: a*N/(1 + h*N).

        Saturates at a/h for large N.

        Args:
            N: Prey density (scalar or array).

        Returns:
            Per-predator consumption rate.
        """
        return self.a * N / (1.0 + self.h * N)

    def _rk4_step(self) -> None:
        """Classical Runge-Kutta 4th order step."""
        dt = self.config.dt
        t = self.current_time()
        y = self._state

        k1 = self._derivatives(y, t)
        k2 = self._derivatives(y + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self._derivatives(y + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self._derivatives(y + dt * k3, t + dt)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Ensure non-negative populations
        self._state = np.maximum(self._state, 0.0)

    def _derivatives(self, y: np.ndarray, t: float) -> np.ndarray:
        """Right-hand side of the seasonal predator-prey system.

        dN/dt = r(t)*N*(1 - N/K) - a*N*P/(1 + h*N)
        dP/dt = e*a*N*P/(1 + h*N) - d*P

        Args:
            y: State vector [N, P].
            t: Current time (for seasonal forcing).

        Returns:
            Derivatives [dN/dt, dP/dt].
        """
        N, P = y
        r_t = self.seasonal_growth_rate(t)
        functional_response = self.holling_type2(N)

        dN = r_t * N * (1.0 - N / self.K) - functional_response * P
        dP = self.e * functional_response * P - self.d * P

        return np.array([dN, dP])

    def coexistence_equilibrium(self) -> dict[str, float]:
        """Compute the coexistence equilibrium of the unforced (epsilon=0) system.

        From dP/dt = 0: e*a*N*/(1+h*N*) - d = 0
            => N* = d / (e*a - h*d)

        From dN/dt = 0 with r(t) = r0:
            => P* = r0*(1 - N*/K)*(1 + h*N*) / a

        Returns:
            Dict with N_star, P_star, and whether the equilibrium is valid.
        """
        denom = self.e * self.a - self.h * self.d
        if denom <= 0:
            return {"N_star": float("nan"), "P_star": float("nan"), "valid": False}

        N_star = self.d / denom
        if N_star <= 0 or N_star >= self.K:
            return {"N_star": float(N_star), "P_star": float("nan"), "valid": False}

        P_star = self.r0 * (1.0 - N_star / self.K) * (1.0 + self.h * N_star) / self.a
        valid = P_star > 0

        return {
            "N_star": float(N_star),
            "P_star": float(P_star),
            "valid": bool(valid),
        }

    def detect_dominant_frequency(
        self,
        trajectory: np.ndarray,
        dt: float | None = None,
    ) -> dict[str, float]:
        """Detect the dominant frequency in a prey time series using FFT.

        Args:
            trajectory: Array of shape (n_steps, 2) with [N, P] columns,
                or 1D array of prey densities.
            dt: Time step (defaults to self.config.dt).

        Returns:
            Dict with dominant_freq, dominant_period, and power_spectrum_peak.
        """
        if dt is None:
            dt = self.config.dt

        if trajectory.ndim == 2:
            signal = trajectory[:, 0]  # Prey population
        else:
            signal = trajectory

        # Remove mean to focus on oscillations
        signal_centered = signal - np.mean(signal)
        n = len(signal_centered)

        # FFT
        fft_vals = np.fft.rfft(signal_centered)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n, d=dt)

        # Exclude DC component (index 0)
        if len(power) > 1:
            peak_idx = np.argmax(power[1:]) + 1
            dominant_freq = float(freqs[peak_idx])
            dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else float("inf")
            peak_power = float(power[peak_idx])
        else:
            dominant_freq = 0.0
            dominant_period = float("inf")
            peak_power = 0.0

        return {
            "dominant_freq": dominant_freq,
            "dominant_period": dominant_period,
            "peak_power": peak_power,
        }
