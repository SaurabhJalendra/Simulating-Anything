"""Oregonator model of the Belousov-Zhabotinsky reaction.

The Oregonator (Field & Noyes, 1974) is the standard model for BZ reaction
oscillations. We use the Tyson (1985) dimensionless three-variable form:

    du/dt = (u*(1 - u) - f*v*(u - q)/(u + q)) / eps
    dv/dt = u - v
    dw/dt = kw * (u - w)

where u ~ [HBrO2], v ~ [Br-], w ~ [Ce(IV)] in dimensionless form.

Parameters:
    eps = timescale separation (default 0.04, u evolves 25x faster)
    f   = stoichiometric factor (default 1.0, controls Hopf bifurcation)
    q   = excitability parameter (default 0.002)
    kw  = catalyst relaxation rate (default 0.5)

Target rediscoveries:
- ODE recovery via SINDy
- Hopf bifurcation as f varies (critical f_c)
- Oscillation period and amplitude vs parameters
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class Oregonator(SimulationEnvironment):
    """Oregonator model for the Belousov-Zhabotinsky reaction.

    State vector: [u, v, w] representing dimensionless concentrations
    of HBrO2, Br-, and Ce(IV) respectively. Shape: (3,).

    The system exhibits relaxation oscillations for typical parameters.
    The fast variable u evolves on timescale eps, requiring dt < eps/10
    for numerical stability.

    Parameters:
        eps: timescale ratio (default 0.04, fast u dynamics)
        f: stoichiometric factor (default 1.0, bifurcation control)
        q: excitability parameter (default 0.002)
        kw: catalyst relaxation rate (default 0.5)
        u_0: initial u (default 0.5)
        v_0: initial v (default 0.5)
        w_0: initial w (default 0.5)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.eps = p.get("eps", 0.04)
        self.f = p.get("f", 1.0)
        self.q = p.get("q", 0.002)
        self.kw = p.get("kw", 0.5)
        self.u_0 = p.get("u_0", 0.5)
        self.v_0 = p.get("v_0", 0.5)
        self.w_0 = p.get("w_0", 0.5)

    @property
    def fixed_point(self) -> tuple[float, float, float]:
        """Compute the positive fixed point (u*, v*, w*).

        At steady state: v* = u*, w* = u* (from dv/dt = 0 and dw/dt = 0).
        From du/dt = 0:
            u*(1 - u) - f*u*(u - q)/(u + q) = 0
        For u != 0:
            (1 - u) = f*(u - q)/(u + q)
            (1 - u)*(u + q) = f*(u - q)
            u + q - u^2 - u*q = f*u - f*q
            -u^2 + u*(1 - q - f) + q*(1 + f) = 0
            u^2 - u*(1 - q - f) - q*(1 + f) = 0
        """
        f = self.f
        q = self.q
        a_coeff = 1.0
        b_coeff = -(1.0 - q - f)
        c_coeff = -q * (1.0 + f)
        disc = b_coeff**2 - 4.0 * a_coeff * c_coeff
        if disc < 0:
            return (float("nan"), float("nan"), float("nan"))
        u_star = (-b_coeff + np.sqrt(disc)) / (2.0 * a_coeff)
        if u_star <= 0:
            u_star = (-b_coeff - np.sqrt(disc)) / (2.0 * a_coeff)
        v_star = u_star
        w_star = u_star
        return (float(u_star), float(v_star), float(w_star))

    @property
    def total_concentration(self) -> float:
        """Sum of current concentrations u + v + w."""
        return float(np.sum(self._state))

    @property
    def is_oscillating(self) -> bool:
        """Detect oscillation by running a trajectory and checking sign changes.

        Looks for sign changes in (u - u*) to determine sustained oscillations.
        """
        u_star = self.fixed_point[0]
        if not np.isfinite(u_star):
            return False

        # Save current state
        saved_state = self._state.copy() if self._state is not None else None
        saved_count = self._step_count

        self.reset()
        dt = self.config.dt
        # Skip transient (200 time units)
        transient_steps = int(200.0 / dt)
        for _ in range(transient_steps):
            self._rk4_step()

        # Count sign changes over 200 time units
        sign_changes = 0
        prev_sign = np.sign(self._state[0] - u_star)
        check_steps = int(200.0 / dt)
        for _ in range(check_steps):
            self._rk4_step()
            cur_sign = np.sign(self._state[0] - u_star)
            if cur_sign != prev_sign and cur_sign != 0:
                sign_changes += 1
            prev_sign = cur_sign

        # Restore state
        if saved_state is not None:
            self._state = saved_state
        self._step_count = saved_count

        # At least 4 sign changes means 2+ full oscillations
        return sign_changes >= 4

    @property
    def period(self) -> float:
        """Measure oscillation period via upward zero crossings of u - u*."""
        return self.measure_period()

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize concentrations [u, v, w]."""
        self._state = np.array([self.u_0, self.v_0, self.w_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [u, v, w]."""
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
        # Clamp to non-negative (concentrations cannot be negative)
        self._state = np.maximum(self._state, 0.0)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute the Oregonator right-hand side.

        du/dt = (u*(1 - u) - f*v*(u - q)/(u + q)) / eps
        dv/dt = u - v
        dw/dt = kw * (u - w)
        """
        u, v, w = y
        u_safe = max(u, 0.0)
        du = (u_safe * (1.0 - u_safe) - self.f * v * (u_safe - self.q)
              / (u_safe + self.q)) / self.eps
        dv = u_safe - v
        dw = self.kw * (u_safe - w)
        return np.array([du, dv, dw])

    def measure_period(self, n_periods: int = 5) -> float:
        """Measure the oscillation period via upward zero crossings of u - u*.

        Args:
            n_periods: Number of periods to average over.

        Returns:
            Average period in time units, or inf if no oscillation detected.
        """
        u_star = self.fixed_point[0]
        if not np.isfinite(u_star):
            return float("inf")

        # Save and reset
        saved_state = self._state.copy() if self._state is not None else None
        saved_count = self._step_count
        self.reset()
        dt = self.config.dt

        # Skip transient (200 time units)
        transient_steps = int(200.0 / dt)
        for _ in range(transient_steps):
            self._rk4_step()

        # Detect upward crossings of u = u_star
        crossings = []
        prev_u = self._state[0]
        step_count = 0
        max_steps = int(n_periods * 50.0 / dt)
        for _ in range(max_steps):
            self._rk4_step()
            step_count += 1
            u = self._state[0]
            if prev_u < u_star and u >= u_star:
                frac = (u_star - prev_u) / (u - prev_u) if u != prev_u else 0.5
                t_cross = (step_count - 1 + frac) * dt
                crossings.append(t_cross)
                if len(crossings) >= n_periods + 1:
                    break
            prev_u = u

        # Restore state
        if saved_state is not None:
            self._state = saved_state
        self._step_count = saved_count

        if len(crossings) < 2:
            return float("inf")

        return float(np.mean(np.diff(crossings)))

    def measure_amplitude(self, n_periods: int = 3) -> tuple[float, float, float]:
        """Measure peak-to-peak amplitude of u, v, w after transient.

        Returns:
            Tuple of (u_amplitude, v_amplitude, w_amplitude).
        """
        saved_state = self._state.copy() if self._state is not None else None
        saved_count = self._step_count
        self.reset()
        dt = self.config.dt

        # Skip transient (200 time units)
        transient_steps = int(200.0 / dt)
        for _ in range(transient_steps):
            self._rk4_step()

        # Record extremes over n_periods * 50 time units
        u_min, v_min, w_min = np.inf, np.inf, np.inf
        u_max, v_max, w_max = -np.inf, -np.inf, -np.inf
        measure_steps = int(n_periods * 50.0 / dt)
        for _ in range(measure_steps):
            self._rk4_step()
            u, v, w = self._state
            u_min, u_max = min(u_min, u), max(u_max, u)
            v_min, v_max = min(v_min, v), max(v_max, v)
            w_min, w_max = min(w_min, w), max(w_max, w)

        # Restore
        if saved_state is not None:
            self._state = saved_state
        self._step_count = saved_count

        return (u_max - u_min, v_max - v_min, w_max - w_min)
