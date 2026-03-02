"""Goldbeter minimal glycolysis model simulation.

The Goldbeter model describes allosteric regulation of the PFK enzyme
in the glycolytic pathway. The 2D minimal version captures the essential
oscillatory behavior driven by cooperative product activation.

ODE system:
    dx/dt = v - sigma * phi(x, y)
    dy/dt = q * sigma * phi(x, y) - k_s * y

Rate function (allosteric):
    phi(x, y) = x * (1 + x) * (1 + y)^2 / (L + (1 + x)^2 * (1 + y)^2)

where x = substrate (F6P), y = product (FBP).

Target rediscoveries:
- Hopf bifurcation as v varies (oscillation onset/offset)
- Limit cycle oscillations in the glycolytic pathway
- Allosteric rate function phi(x, y) properties
- SINDy ODE recovery (approximate)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class GoldbeterGlycolysisSimulation(SimulationEnvironment):
    """Goldbeter minimal glycolysis model with allosteric PFK regulation.

    State vector: [x, y] where x = F6P (substrate), y = FBP (product).

    The allosteric rate function phi(x, y) captures the cooperative
    activation of phosphofructokinase (PFK) by its product FBP, which
    is the key mechanism driving glycolytic oscillations.

    Parameters:
        v: substrate input flux (default 0.36)
        sigma: maximum PFK rate (default 0.1)
        q: stoichiometric ratio (default 1.0)
        k_s: product efflux rate (default 0.02)
        L: allosteric constant (default 5e6)
        x_0: initial substrate concentration (default 1.0)
        y_0: initial product concentration (default 1.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.v = p.get("v", 0.36)
        self.sigma = p.get("sigma", 0.1)
        self.q = p.get("q", 1.0)
        self.k_s = p.get("k_s", 0.02)
        self.L = p.get("L", 5e6)
        self.x_0 = p.get("x_0", 1.0)
        self.y_0 = p.get("y_0", 1.0)

    def phi(self, x: float, y: float) -> float:
        """Allosteric rate function for PFK enzyme.

        phi(x, y) = x * (1 + x) * (1 + y)^2 / (L + (1 + x)^2 * (1 + y)^2)

        This function is bounded in [0, 1) and captures:
        - Substrate activation (x terms in numerator)
        - Product activation (cooperative (1+y)^2 terms)
        - Allosteric inhibition at low concentrations (large L denominator)
        """
        numer = x * (1.0 + x) * (1.0 + y) ** 2
        denom = self.L + (1.0 + x) ** 2 * (1.0 + y) ** 2
        return numer / denom

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize concentrations to [x_0, y_0]."""
        self._state = np.array([self.x_0, self.y_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, y]."""
        return self._state

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta integration step."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Compute dx/dt and dy/dt."""
        x, y = state
        rate = self.phi(x, y)
        dx = self.v - self.sigma * rate
        dy = self.q * self.sigma * rate - self.k_s * y
        return np.array([dx, dy])

    def compute_period(
        self,
        n_transient: int = 3000,
        n_measure: int = 5000,
    ) -> float:
        """Measure the oscillation period via zero crossings of y - y_mean.

        Args:
            n_transient: Number of steps to skip as transient.
            n_measure: Number of steps to use for measurement.

        Returns:
            Estimated period in time units, or inf if no oscillation detected.
        """
        dt = self.config.dt

        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Collect y values to compute mean
        y_vals: list[float] = []
        for _ in range(n_measure):
            self.step()
            y_vals.append(self._state[1])

        y_arr = np.array(y_vals)
        y_mean = np.mean(y_arr)

        # Detect upward zero crossings of y - y_mean
        crossings: list[int] = []
        for i in range(1, len(y_arr)):
            if y_arr[i - 1] < y_mean and y_arr[i] >= y_mean:
                crossings.append(i)

        if len(crossings) < 2:
            return float("inf")

        # Average period from crossing intervals
        periods = np.diff(crossings) * dt
        return float(np.mean(periods))

    def measure_amplitude(self, transient_time: float = 500.0) -> float:
        """Measure peak-to-peak amplitude of y after transient.

        Args:
            transient_time: Time to skip as transient.

        Returns:
            Peak-to-peak amplitude of y coordinate.
        """
        dt = self.config.dt

        # Skip transient
        for _ in range(int(transient_time / dt)):
            self.step()

        # Collect y values
        y_vals: list[float] = []
        for _ in range(int(200 / dt)):
            self.step()
            y_vals.append(self._state[1])

        return float(max(y_vals) - min(y_vals))

    def bifurcation_sweep(
        self,
        param_name: str = "v",
        param_range: tuple[float, float] = (0.1, 1.0),
        n_params: int = 30,
        n_steps: int = 20000,
    ) -> dict[str, np.ndarray]:
        """Sweep a parameter and record oscillation amplitude.

        Args:
            param_name: Parameter to sweep ('v', 'sigma', 'k_s').
            param_range: (min, max) for the sweep.
            n_params: Number of parameter values to test.
            n_steps: Total steps per parameter value.

        Returns:
            Dictionary with param_values, y_max, y_min, amplitude arrays.
        """
        param_values = np.linspace(param_range[0], param_range[1], n_params)
        y_max_arr = np.zeros(n_params)
        y_min_arr = np.zeros(n_params)

        transient_steps = n_steps // 2
        measure_steps = n_steps - transient_steps

        for i, pval in enumerate(param_values):
            params = dict(self.config.parameters)
            params[param_name] = pval
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=self.config.n_steps,
                parameters=params,
            )
            sim = GoldbeterGlycolysisSimulation(config)
            sim.reset()

            # Transient
            for _ in range(transient_steps):
                sim.step()

            # Measure
            y_vals: list[float] = []
            for _ in range(measure_steps):
                sim.step()
                y_vals.append(sim.observe()[1])

            y_max_arr[i] = max(y_vals)
            y_min_arr[i] = min(y_vals)

        return {
            "param_name": param_name,
            "param_values": param_values,
            "y_max": y_max_arr,
            "y_min": y_min_arr,
            "amplitude": y_max_arr - y_min_arr,
        }
