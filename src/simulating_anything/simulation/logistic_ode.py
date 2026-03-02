"""Continuous-time logistic (Verhulst) growth model simulation.

The classic logistic ODE for population growth:

    dN/dt = r * N * (1 - N / K)

where N = population size, r = intrinsic growth rate, K = carrying capacity.

Unlike the Gompertz model (inflection at K/e), the logistic model has a
symmetric sigmoid growth curve with the inflection point at N = K/2.

Analytical solution:
    N(t) = K / (1 + (K/N_0 - 1) * exp(-r * t))

Key properties:
- Inflection point at N = K/2 (where dN/dt is maximal)
- Maximum growth rate: r * K / 4 at the inflection point
- Two fixed points: N=0 (unstable for r>0), N=K (stable for r>0)
- Sigmoid (S-shaped) growth curve
- Monotonic approach to K from any positive N_0

Target rediscoveries:
- Analytical solution match: N(t) = K / (1 + (K/N_0 - 1)*exp(-r*t))
- Inflection point: N_inflection = K / 2
- Maximum growth rate: dN/dt_max = r * K / 4
- ODE recovery via SINDy: dN/dt = r*N*(1 - N/K)
- PySR: inflection point and growth rate scaling
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class LogisticODESimulation(SimulationEnvironment):
    """Continuous logistic (Verhulst) growth: dN/dt = r * N * (1 - N / K).

    State vector: [N] (single scalar population, stored as 1D array).

    The population is clamped to N >= 0 to prevent numerical issues.

    Parameters:
        r: intrinsic growth rate (default 1.0)
        K: carrying capacity (default 100.0)
        N_0: initial population (default 1.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 1.0)
        self.K = p.get("K", 100.0)
        self.N_0 = p.get("N_0", 1.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize population to N_0."""
        self._state = np.array([self.N_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4, clamping N >= 0."""
        self._rk4_step()
        if self._state[0] < 0.0:
            self._state[0] = 0.0
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
        """Compute dN/dt = r * N * (1 - N / K)."""
        N = y[0]
        dNdt = self.r * N * (1.0 - N / self.K)
        return np.array([dNdt])

    def analytical_solution(self, t: float | np.ndarray) -> np.ndarray:
        """Exact logistic solution: N(t) = K / (1 + (K/N_0 - 1)*exp(-r*t)).

        Args:
            t: time or array of times.

        Returns:
            Population at each time point.
        """
        t = np.asarray(t, dtype=np.float64)
        return self.K / (1.0 + (self.K / self.N_0 - 1.0) * np.exp(-self.r * t))

    @property
    def inflection_point(self) -> float:
        """Population at the inflection point: N_inflection = K / 2."""
        return self.K / 2.0

    @property
    def max_growth_rate(self) -> float:
        """Maximum growth rate: dN/dt_max = r * K / 4.

        Occurs at the inflection point N = K/2.
        """
        return self.r * self.K / 4.0

    def inflection_time(self) -> float:
        """Time at which population reaches K/2.

        From N(t) = K / (1 + (K/N_0 - 1)*exp(-r*t)), setting N(t) = K/2:
            2 = 1 + (K/N_0 - 1)*exp(-r*t)
            exp(-r*t) = 1 / (K/N_0 - 1)
            t = ln(K/N_0 - 1) / r

        Only defined when N_0 < K/2 (so inflection is reached during growth).
        Returns 0.0 if N_0 >= K/2.
        """
        if self.N_0 >= self.K / 2.0:
            return 0.0
        if self.N_0 <= 0:
            return float("inf")
        return np.log(self.K / self.N_0 - 1.0) / self.r

    def growth_rate_at(self, N: float) -> float:
        """Compute dN/dt at a given population level.

        Args:
            N: population size.

        Returns:
            Growth rate dN/dt = r * N * (1 - N/K).
        """
        if N <= 0 or N >= self.K:
            return 0.0
        return self.r * N * (1.0 - N / self.K)

    def find_fixed_points(self) -> list[dict[str, float | str]]:
        """Find the two fixed points of the logistic ODE.

        Returns:
            List of dicts with 'N' and 'stability' keys.
            N=0 is unstable (for r>0), N=K is stable (for r>0).
        """
        fixed_points: list[dict[str, float | str]] = []

        # N=0: eigenvalue is r (unstable for r>0)
        eig_0 = self.r
        fixed_points.append({
            "N": 0.0,
            "stability": "unstable" if eig_0 > 0 else "stable",
            "eigenvalue": eig_0,
        })

        # N=K: eigenvalue is -r (stable for r>0)
        eig_K = -self.r
        fixed_points.append({
            "N": self.K,
            "stability": "stable" if eig_K < 0 else "unstable",
            "eigenvalue": eig_K,
        })

        return fixed_points

    def r_sweep(
        self,
        r_values: np.ndarray | None = None,
        dt: float | None = None,
        n_steps: int = 500,
    ) -> dict[str, np.ndarray]:
        """Sweep growth rate r and measure final population and inflection time.

        Returns dict with 'r', 'final_N', 'inflection_time' arrays.
        """
        if r_values is None:
            r_values = np.linspace(0.1, 3.0, 20)
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
            sim = LogisticODESimulation(config)
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
