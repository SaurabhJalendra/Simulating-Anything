"""Mackey-Glass delay differential equation simulation.

The Mackey-Glass equation is a canonical delay differential equation (DDE):

    dx/dt = beta * x(t - tau) / (1 + x(t - tau)^n) - gamma * x(t)

Key dynamical regimes:
- tau < 4: monotone convergence to equilibrium
- 4 < tau < 13: damped oscillations
- tau > 17: chaos (Mackey-Glass attractor)

Equilibrium: x* = (beta/gamma - 1)^(1/n)

Target rediscoveries:
- Equilibrium x* = (beta/gamma - 1)^(1/n)
- Chaos onset as function of tau
- Lyapunov exponent vs tau
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class MackeyGlassSimulation(SimulationEnvironment):
    """Mackey-Glass delay differential equation.

    State: current x value (scalar, stored as 1D array).
    Internal: ring buffer of past values for the delayed term x(t - tau).

    Parameters:
        beta: production rate (default 0.2)
        gamma: decay rate (default 0.1)
        tau: delay time (default 17.0)
        n: Hill exponent (default 10.0)
        x_0: initial value for all t in [-tau, 0] (default 0.9)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.beta = p.get("beta", 0.2)
        self.gamma = p.get("gamma", 0.1)
        self.tau = p.get("tau", 17.0)
        self.n = p.get("n", 10.0)
        self.x_0 = p.get("x_0", 0.9)

        # Ring buffer size: number of past values needed to look up x(t - tau)
        self._buf_size = int(self.tau / config.dt) + 1
        self._buffer: np.ndarray = np.empty(0)
        self._buf_idx = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize x(t) = x_0 for all t in [-tau, 0] and fill history buffer."""
        self._buffer = np.full(self._buf_size, self.x_0, dtype=np.float64)
        self._buf_idx = 0
        self._state = np.array([self.x_0], dtype=np.float64)
        self._step_count = 0
        return self._state.copy()

    def step(self) -> np.ndarray:
        """Advance one timestep using Euler integration with delayed term.

        The delayed value x(t - tau) is read from the ring buffer.
        Euler is adequate for DDEs where the delay makes higher-order
        methods less critical for qualitative accuracy.
        """
        dt = self.config.dt
        x = self._state[0]

        # Delayed value from ring buffer (oldest entry)
        x_delayed = self._buffer[self._buf_idx]

        # Mackey-Glass right-hand side
        dxdt = (
            self.beta * x_delayed / (1.0 + x_delayed ** self.n)
            - self.gamma * x
        )
        x_new = x + dt * dxdt

        # Store new value in ring buffer (overwrite oldest)
        self._buffer[self._buf_idx] = x_new
        self._buf_idx = (self._buf_idx + 1) % self._buf_size

        self._state = np.array([x_new], dtype=np.float64)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x] (1-element array)."""
        return self._state

    def get_history(self) -> np.ndarray:
        """Return the full history buffer in chronological order.

        The ring buffer is reordered so index 0 is the oldest value
        and index -1 is the most recent.
        """
        return np.roll(self._buffer, -self._buf_idx)

    @property
    def equilibrium(self) -> float:
        """Compute the nontrivial equilibrium x* = (beta/gamma - 1)^(1/n).

        Returns 0.0 if beta <= gamma (no positive equilibrium).
        """
        ratio = self.beta / self.gamma
        if ratio <= 1.0:
            return 0.0
        return (ratio - 1.0) ** (1.0 / self.n)

    def _production(self, x: float) -> float:
        """Mackey-Glass production function: beta * x / (1 + x^n)."""
        return self.beta * x / (1.0 + x ** self.n)

    def estimate_lyapunov(
        self,
        n_steps: int = 100000,
        dt: float | None = None,
    ) -> float:
        """Estimate the largest Lyapunov exponent via trajectory divergence.

        Uses two nearby trajectories with independent history buffers,
        renormalizing the perturbation periodically.
        """
        if dt is None:
            dt = self.config.dt

        eps = 1e-8
        buf_size = int(self.tau / dt) + 1

        # Primary trajectory: copy current state and buffer
        x1 = self._state[0]
        buf1 = self._buffer.copy()
        idx1 = self._buf_idx

        # Perturbed trajectory
        x2 = x1 + eps
        buf2 = buf1.copy()
        idx2 = idx1

        lyap_sum = 0.0
        n_renorm = 0

        for _ in range(n_steps):
            # Advance trajectory 1
            x1_delayed = buf1[idx1]
            dx1 = self.beta * x1_delayed / (1.0 + x1_delayed ** self.n) - self.gamma * x1
            x1_new = x1 + dt * dx1
            buf1[idx1] = x1_new
            idx1 = (idx1 + 1) % buf_size
            x1 = x1_new

            # Advance trajectory 2
            x2_delayed = buf2[idx2]
            dx2 = self.beta * x2_delayed / (1.0 + x2_delayed ** self.n) - self.gamma * x2
            x2_new = x2 + dt * dx2
            buf2[idx2] = x2_new
            idx2 = (idx2 + 1) % buf_size
            x2 = x2_new

            # Measure divergence and renormalize
            dist = abs(x2 - x1)
            if dist > 0:
                lyap_sum += np.log(dist / eps)
                n_renorm += 1
                # Renormalize perturbation (both state and buffer)
                scale = eps / dist
                x2 = x1 + (x2 - x1) * scale
                buf2 = buf1 + (buf2 - buf1) * scale

        if n_renorm == 0:
            return 0.0
        return lyap_sum / (n_renorm * dt)

    def tau_sweep(
        self,
        tau_values: np.ndarray,
        n_steps: int = 50000,
        n_transient: int = 10000,
    ) -> dict[str, np.ndarray]:
        """Sweep tau values and measure amplitude range and Lyapunov exponent.

        For each tau, runs a trajectory, discards transient, then measures
        the max-min amplitude (proxy for oscillation/chaos) and estimates
        the Lyapunov exponent.

        Args:
            tau_values: Array of delay values to sweep.
            n_steps: Number of steps after transient for measurement.
            n_transient: Steps to discard as transient.

        Returns:
            Dict with tau, amplitude, lyapunov arrays.
        """
        amplitudes = []
        lyapunovs = []

        for tau_val in tau_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    "beta": self.beta,
                    "gamma": self.gamma,
                    "tau": tau_val,
                    "n": self.n,
                    "x_0": self.x_0,
                },
            )
            sim = MackeyGlassSimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient):
                sim.step()

            # Collect values for amplitude measurement
            x_vals = []
            for _ in range(n_steps):
                state = sim.step()
                x_vals.append(state[0])

            x_arr = np.array(x_vals)
            amplitudes.append(float(np.max(x_arr) - np.min(x_arr)))

            # Lyapunov estimate
            lam = sim.estimate_lyapunov(n_steps=min(n_steps, 50000))
            lyapunovs.append(lam)

        return {
            "tau": tau_values,
            "amplitude": np.array(amplitudes),
            "lyapunov": np.array(lyapunovs),
        }
