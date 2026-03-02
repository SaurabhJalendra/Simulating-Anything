"""Autocatalator 3-species chemical oscillator simulation.

The autocatalator models a driven autocatalytic chemical system with three
species (a, b, c) evolving on multiple timescales controlled by sigma and delta.

ODEs:
    da/dt = mu*(kappa + c) - a*b^2 - a
    db/dt = (a*b^2 + a - b) / sigma
    dc/dt = (b - c) / delta

The autocatalytic reaction A + 2B -> 3B (rate a*b^2) produces mixed-mode
oscillations, relaxation oscillations, and Hopf bifurcation as mu varies.

Target rediscoveries:
- SINDy recovery of 3-species ODEs
- Hopf bifurcation as mu varies (onset of oscillations)
- Fixed point: (a*, b*, c*) satisfying mu*(kappa + c) = a*b^2 + a, etc.
- Mixed-mode oscillation pattern detection
- Divergence of the flow: div = -(2*a*b + 1) - 1/sigma - 1/delta < 0
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class AutocatalatorSimulation(SimulationEnvironment):
    """Autocatalator: 3-species chemical oscillator with multiple timescales.

    State vector: [a, b, c] where a, b, c are chemical concentrations.

    ODEs:
        da/dt = mu*(kappa + c) - a*b^2 - a
        db/dt = (a*b^2 + a - b) / sigma
        dc/dt = (b - c) / delta

    The autocatalytic step A + 2B -> 3B (rate a*b^2) drives oscillations.
    The system is fed by a constant inflow proportional to mu, so conservation
    is not exact. Sigma << 1 and delta ~ O(1) produce multiple timescales.

    Parameters:
        mu: feed rate (default 0.002)
        kappa: background feed constant (default 65.0)
        sigma: timescale separation for b (default 0.005)
        delta: timescale separation for c (default 0.2)
        a_0: initial concentration of species a
        b_0: initial concentration of species b
        c_0: initial concentration of species c
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.mu = p.get("mu", 0.002)
        self.kappa = p.get("kappa", 65.0)
        self.sigma = p.get("sigma", 0.005)
        self.delta = p.get("delta", 0.2)
        self.a_0 = p.get("a_0", 0.5)
        self.b_0 = p.get("b_0", 1.0)
        self.c_0 = p.get("c_0", 0.5)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize concentrations [a, b, c]."""
        self._state = np.array(
            [self.a_0, self.b_0, self.c_0], dtype=np.float64
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [a, b, c]."""
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
        """Autocatalator ODEs.

        da/dt = mu*(kappa + c) - a*b^2 - a
        db/dt = (a*b^2 + a - b) / sigma
        dc/dt = (b - c) / delta
        """
        a, b, c = state
        ab2 = a * b**2
        da = self.mu * (self.kappa + c) - ab2 - a
        db = (ab2 + a - b) / self.sigma
        dc = (b - c) / self.delta
        return np.array([da, db, dc])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the fixed point(s) of the autocatalator.

        Setting all derivatives to zero:
            dc/dt = 0  =>  c* = b*
            db/dt = 0  =>  a*b*^2 + a* - b* = 0  =>  a*(b*^2 + 1) = b*
                       =>  a* = b* / (b*^2 + 1)
            da/dt = 0  =>  mu*(kappa + c*) = a*b*^2 + a* = b*
                       =>  mu*(kappa + b*) = b*
                       =>  b* = mu*kappa / (1 - mu)

        For mu << 1, b* is approximately mu*kappa.
        """
        if self.mu >= 1.0:
            # No physical fixed point when mu >= 1
            return []

        b_star = self.mu * self.kappa / (1.0 - self.mu)
        c_star = b_star
        a_star = b_star / (b_star**2 + 1.0)

        return [np.array([a_star, b_star, c_star], dtype=np.float64)]

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix at a given state.

        J = [[-(b^2 + 1),    -2*a*b,          mu      ],
             [(b^2 + 1)/sigma, (2*a*b - 1)/sigma, 0    ],
             [0,               1/delta,        -1/delta ]]
        """
        a, b, _c = state
        return np.array([
            [-(b**2 + 1.0), -2.0 * a * b, self.mu],
            [(b**2 + 1.0) / self.sigma,
             (2.0 * a * b - 1.0) / self.sigma, 0.0],
            [0.0, 1.0 / self.delta, -1.0 / self.delta],
        ], dtype=np.float64)

    def compute_divergence(self, state: np.ndarray) -> float:
        """Compute the divergence (trace of Jacobian) at a given state.

        div = d(da/dt)/da + d(db/dt)/db + d(dc/dt)/dc
            = -(b^2 + 1) + (2*a*b - 1)/sigma + (-1/delta)

        The divergence depends on the state (specifically a and b),
        unlike constant-divergence systems. For typical parameter values,
        divergence is strongly negative (dissipative system).
        """
        a, b, _c = state
        return -(b**2 + 1.0) + (2.0 * a * b - 1.0) / self.sigma - 1.0 / self.delta

    def measure_period(self, n_periods: int = 5) -> float:
        """Measure the oscillation period via zero crossings of b - b*.

        Returns the average period, or inf if no oscillations detected.
        """
        fps = self.fixed_points
        if not fps:
            return float("inf")

        dt = self.config.dt
        b_star = fps[0][1]

        # Transient: let the system settle
        transient_steps = int(500 / dt)
        for _ in range(transient_steps):
            self.step()

        # Detect upward crossings of b = b*
        crossings: list[float] = []
        prev_b = self._state[1]
        for _ in range(int(n_periods * 200 / dt)):
            self.step()
            curr_b = self._state[1]
            if prev_b < b_star and curr_b >= b_star:
                frac = (
                    (b_star - prev_b) / (curr_b - prev_b)
                    if curr_b != prev_b
                    else 0.5
                )
                t_cross = (self._step_count - 1 + frac) * dt
                crossings.append(t_cross)
            prev_b = curr_b

        if len(crossings) < 2:
            return float("inf")

        return float(np.mean(np.diff(crossings)))

    def measure_amplitude(self, transient_time: float = 500.0) -> float:
        """Measure peak-to-peak amplitude of b after transient."""
        dt = self.config.dt

        # Skip transient
        for _ in range(int(transient_time / dt)):
            self.step()

        # Collect b values
        b_vals: list[float] = []
        for _ in range(int(500 / dt)):
            self.step()
            b_vals.append(self._state[1])

        return float(max(b_vals) - min(b_vals))

    def estimate_lyapunov(
        self, n_steps: int = 50000, dt: float | None = None
    ) -> float:
        """Estimate the largest Lyapunov exponent via trajectory divergence.

        Uses the method of Wolf et al. (1985): track two nearby trajectories,
        renormalize when they diverge too far.
        """
        if dt is None:
            dt = self.config.dt

        eps = 1e-8
        state1 = self._state.copy()
        state2 = state1 + np.array([eps, 0.0, 0.0])

        lyap_sum = 0.0
        n_renorm = 0

        for _ in range(n_steps):
            # Advance both states with RK4
            k1 = self._derivatives(state1)
            k2 = self._derivatives(state1 + 0.5 * dt * k1)
            k3 = self._derivatives(state1 + 0.5 * dt * k2)
            k4 = self._derivatives(state1 + dt * k3)
            state1 = state1 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            k1 = self._derivatives(state2)
            k2 = self._derivatives(state2 + 0.5 * dt * k1)
            k3 = self._derivatives(state2 + 0.5 * dt * k2)
            k4 = self._derivatives(state2 + dt * k3)
            state2 = state2 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Bail out if trajectory diverges
            if not (
                np.all(np.isfinite(state1))
                and np.all(np.isfinite(state2))
            ):
                break

            # Compute distance
            dist = np.linalg.norm(state2 - state1)
            if dist > 0:
                lyap_sum += np.log(dist / eps)
                n_renorm += 1
                # Renormalize
                state2 = state1 + eps * (state2 - state1) / dist

        if n_renorm == 0:
            return 0.0
        return lyap_sum / (n_renorm * dt)
