"""Gompertz growth model simulation.

The Gompertz equation is a saturating growth ODE commonly used in tumor growth
modeling:

    dN/dt = r * N * ln(K / N)

where N = population or tumor size, r = growth rate, K = carrying capacity.

Unlike logistic growth, the Gompertz model has an asymmetric sigmoid curve:
growth decelerates earlier, with the inflection point at N = K/e rather than
N = K/2 for the logistic equation.

Analytical solution:
    N(t) = K * (N_0 / K)^exp(-r * t)
         = K * exp(ln(N_0 / K) * exp(-r * t))

Key properties:
- Inflection point at N = K/e (where dN/dt is maximal)
- Maximum growth rate: r * K / e at the inflection point
- Asymptotic approach to K as t -> infinity
- For N_0 < K/e: growth accelerates then decelerates (sigmoid)
- For K/e < N_0 < K: growth decelerates monotonically

Target rediscoveries:
- Analytical solution match: N(t) = K * (N_0/K)^exp(-r*t)
- Inflection point: N_inflection = K / e
- Maximum growth rate: dN/dt_max = r * K / e
- ODE recovery via SINDy: dN/dt = r * N * ln(K/N)
- PySR: inflection point and growth rate scaling
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class GompertzSimulation(SimulationEnvironment):
    """Gompertz growth model: dN/dt = r * N * ln(K / N).

    State vector: [N] (single scalar population/tumor size, stored as 1D array).

    The population is clamped to N > 0 (Gompertz is undefined at N = 0) and
    N <= K (cannot exceed carrying capacity due to numerical overshoot).

    Parameters:
        r: growth rate (default 0.5)
        K: carrying capacity (default 100.0)
        N_0: initial population/tumor size (default 1.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 0.5)
        self.K = p.get("K", 100.0)
        self.N_0 = p.get("N_0", 1.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize population to N_0."""
        self._state = np.array([self.N_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4, clamping N to (0, K]."""
        self._rk4_step()
        # Clamp to prevent numerical issues (N must be positive for ln(K/N))
        self._state[0] = np.clip(self._state[0], 1e-15, self.K)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [N]."""
        return self._state

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta integrator."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute dN/dt = r * N * ln(K / N).

        For safety, clamp N > 0 before taking the logarithm.
        """
        N = max(y[0], 1e-15)
        dNdt = self.r * N * np.log(self.K / N)
        return np.array([dNdt])

    def analytical_solution(self, t: float | np.ndarray) -> np.ndarray:
        """Exact Gompertz solution: N(t) = K * (N_0/K)^exp(-r*t).

        Args:
            t: time or array of times.

        Returns:
            Population at each time point.
        """
        t = np.asarray(t, dtype=np.float64)
        return self.K * (self.N_0 / self.K) ** np.exp(-self.r * t)

    @property
    def inflection_point(self) -> float:
        """Population at the inflection point: N_inflection = K / e."""
        return self.K / np.e

    @property
    def max_growth_rate(self) -> float:
        """Maximum growth rate: dN/dt_max = r * K / e.

        Occurs at the inflection point N = K/e.
        """
        return self.r * self.K / np.e

    def inflection_time(self) -> float:
        """Time at which population reaches K/e.

        From N(t) = K * (N_0/K)^exp(-r*t), setting N(t) = K/e:
            exp(-r*t) * ln(N_0/K) = -1
            t_inflection = -ln(-1 / ln(N_0/K)) / r

        Only defined when N_0 < K/e (so inflection is reached during growth).
        Returns float('inf') if inflection point is below N_0.
        """
        if self.N_0 >= self.K / np.e:
            return 0.0  # Already past inflection
        if self.N_0 <= 0:
            return float("inf")
        log_ratio = np.log(self.N_0 / self.K)
        # We need exp(-r*t) = -1 / log_ratio = 1 / |log_ratio|
        # So -r*t = ln(1/|log_ratio|) = -ln(|log_ratio|)
        # t = ln(|log_ratio|) / r
        return np.log(abs(log_ratio)) / self.r

    def half_saturation_time(self) -> float:
        """Time at which population reaches K/2.

        From N(t) = K * (N_0/K)^exp(-r*t), setting N(t) = K/2:
            exp(-r*t) * ln(N_0/K) = ln(1/2) = -ln(2)
            exp(-r*t) = ln(2) / |ln(N_0/K)|
            t = -ln(ln(2) / |ln(N_0/K)|) / r

        Returns float('inf') if not reachable.
        """
        if self.N_0 >= self.K / 2:
            return 0.0
        if self.N_0 <= 0:
            return float("inf")
        log_ratio = np.log(self.N_0 / self.K)  # negative
        # exp(-r*t) = -ln(2) / log_ratio = ln(2) / |log_ratio|
        ratio = np.log(2) / abs(log_ratio)
        if ratio <= 0 or ratio > 1:
            return float("inf")
        return -np.log(ratio) / self.r

    def growth_rate_at(self, N: float) -> float:
        """Compute dN/dt at a given population level.

        Args:
            N: population size (must be in (0, K)).

        Returns:
            Growth rate dN/dt = r * N * ln(K/N).
        """
        if N <= 0 or N >= self.K:
            return 0.0
        return self.r * N * np.log(self.K / N)

    def r_sweep(
        self,
        r_values: np.ndarray | None = None,
        dt: float | None = None,
        n_steps: int = 500,
    ) -> dict[str, np.ndarray]:
        """Sweep growth rate r and measure final population + time to inflection.

        Returns dict with 'r', 'final_N', 'inflection_time' arrays.
        """
        if r_values is None:
            r_values = np.linspace(0.1, 2.0, 20)
        if dt is None:
            dt = self.config.dt

        final_N = []
        inf_times = []

        for r_val in r_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=dt,
                n_steps=n_steps,
                parameters={
                    "r": float(r_val),
                    "K": self.K,
                    "N_0": self.N_0,
                },
            )
            sim = GompertzSimulation(config)
            sim.reset()

            for _ in range(n_steps):
                sim.step()

            final_N.append(float(sim.observe()[0]))
            inf_times.append(sim.inflection_time())

        return {
            "r": r_values,
            "final_N": np.array(final_N),
            "inflection_time": np.array(inf_times),
        }
