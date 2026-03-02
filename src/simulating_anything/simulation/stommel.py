"""Stommel box model simulation (ocean thermohaline circulation).

Stommel's 2-box model describes the competition between thermal and haline
forcing on the Atlantic meridional overturning circulation (AMOC):

    dT/dt = eta1 - T*(1 + |T - S|)
    dS/dt = eta2*eta3 - S*(eta3 + |T - S|)

where T = temperature difference, S = salinity difference between
equatorial and polar boxes, and the flow rate is q = |T - S|.

Target rediscoveries:
- ODE recovery via SINDy (piecewise-smooth system)
- Bistability: thermally-driven (T > S) vs salinity-driven (S > T) equilibria
- Saddle-node bifurcation as eta1 varies -- hysteresis loop
- Flow rate q = |T - S| characterization
- Equilibria finding and stability
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class StommelSimulation(SimulationEnvironment):
    """Stommel 2-box model for thermohaline ocean circulation.

    State vector: [T, S] where T = temperature difference, S = salinity
    difference between equatorial and polar ocean boxes.

    The overturning flow rate q = |T - S| couples to both variables.
    Thermal forcing has a fast relaxation timescale (1), while haline
    forcing relaxes on a slower timescale (eta3 < 1).

    This creates bistability: one equilibrium with thermally-driven
    circulation (T > S, poleward warm surface flow) and another with
    salinity-driven circulation (S > T, equatorward warm surface flow).
    As eta1 varies, the system exhibits saddle-node bifurcations and
    hysteresis -- a key model for abrupt climate change (AMOC shutdown).

    Parameters:
        eta1: thermal forcing parameter (default 3.0)
        eta2: salinity forcing ratio (default 1.0)
        eta3: ratio of haline-to-thermal relaxation times (default 0.3)
        T_0: initial temperature difference (default 1.0)
        S_0: initial salinity difference (default 0.5)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.eta1 = p.get("eta1", 3.0)
        self.eta2 = p.get("eta2", 1.0)
        self.eta3 = p.get("eta3", 0.3)
        self.T_0 = p.get("T_0", 1.0)
        self.S_0 = p.get("S_0", 0.5)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize temperature and salinity differences.

        A small random perturbation is added when a seed is provided.
        """
        rng = np.random.default_rng(seed)
        perturbation = rng.normal(0, 1e-4, size=2)
        self._state = np.array([self.T_0, self.S_0], dtype=np.float64) + perturbation
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [T, S]."""
        return self._state

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta integration."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute dT/dt, dS/dt for the Stommel box model.

        Args:
            y: State vector [T, S].

        Returns:
            Derivatives [dT/dt, dS/dt].
        """
        T, S = y
        q = abs(T - S)
        dT = self.eta1 - T * (1.0 + q)
        dS = self.eta2 * self.eta3 - S * (self.eta3 + q)
        return np.array([dT, dS])

    def flow_rate(self, T: float | None = None, S: float | None = None) -> float:
        """Compute the overturning circulation strength q = |T - S|.

        Args:
            T: Temperature difference. Uses current state if None.
            S: Salinity difference. Uses current state if None.

        Returns:
            Overturning flow rate (non-negative).
        """
        if T is None or S is None:
            T, S = self._state
        return float(abs(T - S))

    def find_equilibria(
        self,
        n_guesses: int = 50,
        tol: float = 1e-10,
    ) -> list[tuple[float, float]]:
        """Find equilibria of the Stommel system numerically.

        Uses fsolve from multiple initial guesses to find all fixed points
        where dT/dt = 0, dS/dt = 0 simultaneously.

        Args:
            n_guesses: Number of random initial guesses to try.
            tol: Tolerance for considering a point an equilibrium.

        Returns:
            List of (T, S) equilibrium tuples, deduplicated.
        """
        rng = np.random.default_rng(12345)
        equilibria: list[tuple[float, float]] = []

        def residual(y: np.ndarray) -> np.ndarray:
            return self._derivatives(y)

        # Try a grid of initial guesses covering both circulation regimes
        for _ in range(n_guesses):
            guess = rng.uniform(low=[0.0, 0.0], high=[5.0, 5.0])
            try:
                sol, info, ier, _ = fsolve(residual, guess, full_output=True)
                if ier == 1 and np.max(np.abs(info["fvec"])) < tol:
                    T_eq, S_eq = sol
                    # Only keep physical solutions (positive T, S)
                    if T_eq > -0.1 and S_eq > -0.1:
                        # Check if this equilibrium is already found
                        is_new = True
                        for T_prev, S_prev in equilibria:
                            if abs(T_eq - T_prev) < 1e-6 and abs(S_eq - S_prev) < 1e-6:
                                is_new = False
                                break
                        if is_new:
                            equilibria.append((float(T_eq), float(S_eq)))
            except Exception:
                continue

        # Sort by T value for consistent ordering
        equilibria.sort(key=lambda x: x[0])
        return equilibria

    def classify_equilibrium(self, T: float, S: float) -> str:
        """Classify an equilibrium as thermally-driven or salinity-driven.

        Args:
            T: Temperature difference at equilibrium.
            S: Salinity difference at equilibrium.

        Returns:
            'thermal' if T > S (warm poleward surface flow),
            'haline' if S > T (cold deep flow dominates),
            'degenerate' if T == S.
        """
        if abs(T - S) < 1e-10:
            return "degenerate"
        return "thermal" if T > S else "haline"

    def eta1_sweep(
        self,
        eta1_range: tuple[float, float] = (0.5, 5.0),
        n_eta: int = 40,
        n_steps: int = 10000,
    ) -> dict:
        """Sweep eta1 from two different ICs to capture hysteresis.

        Starting from thermally-driven (high T) and salinity-driven (high S)
        initial conditions, sweeps eta1 and records final steady states.
        Hysteresis appears as different final states for the same eta1.

        Args:
            eta1_range: (min, max) range for eta1 sweep.
            n_eta: Number of eta1 values in sweep.
            n_steps: Steps to integrate at each eta1 value.

        Returns:
            Dict with eta1 values and final T, S from both ICs.
        """
        eta1_values = np.linspace(eta1_range[0], eta1_range[1], n_eta)
        dt = self.config.dt

        # Forward sweep from thermally-driven IC (high T, low S)
        T_thermal = []
        S_thermal = []
        for eta1 in eta1_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=dt,
                n_steps=n_steps,
                parameters={
                    "eta1": float(eta1),
                    "eta2": self.eta2,
                    "eta3": self.eta3,
                    "T_0": 3.0,
                    "S_0": 0.5,
                },
                seed=42,
            )
            sim = StommelSimulation(config)
            sim.reset(seed=42)
            for _ in range(n_steps):
                sim.step()
            T_final, S_final = sim.observe()
            T_thermal.append(float(T_final))
            S_thermal.append(float(S_final))

        # Reverse sweep from salinity-driven IC (low T, high S)
        T_haline = []
        S_haline = []
        for eta1 in eta1_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=dt,
                n_steps=n_steps,
                parameters={
                    "eta1": float(eta1),
                    "eta2": self.eta2,
                    "eta3": self.eta3,
                    "T_0": 0.5,
                    "S_0": 3.0,
                },
                seed=42,
            )
            sim = StommelSimulation(config)
            sim.reset(seed=42)
            for _ in range(n_steps):
                sim.step()
            T_final, S_final = sim.observe()
            T_haline.append(float(T_final))
            S_haline.append(float(S_final))

        return {
            "eta1": eta1_values,
            "T_thermal": np.array(T_thermal),
            "S_thermal": np.array(S_thermal),
            "T_haline": np.array(T_haline),
            "S_haline": np.array(S_haline),
            "q_thermal": np.abs(np.array(T_thermal) - np.array(S_thermal)),
            "q_haline": np.abs(np.array(T_haline) - np.array(S_haline)),
        }
