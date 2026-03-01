"""1D Oregonator reaction-diffusion simulation (BZ traveling waves).

The Oregonator in 1D models the Belousov-Zhabotinsky reaction with diffusion,
producing traveling chemical wave pulses in an excitable medium:

    du/dt = D_u * d^2u/dx^2 + (1/eps) * (u - u^2 - f*v*(u - q)/(u + q))
    dv/dt = D_v * d^2v/dx^2 + u - v

Parameters:
    eps = 0.1   timescale separation (u evolves faster)
    f   = 1.0   stoichiometric factor
    q   = 0.002 excitability parameter
    D_u = 1.0   diffusion coefficient for u (activator)
    D_v = 0.6   diffusion coefficient for v (inhibitor)
    N   = 200   grid points
    L   = 100.0 domain length

State: (2, N) -- u and v fields on a periodic 1D grid.

Target rediscoveries:
- Traveling pulse solutions in excitable medium
- Pulse speed scaling: c ~ sqrt(D_u / eps)
- Pulse width and shape dependence on parameters
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


def _laplacian_1d(field: np.ndarray, dx: float) -> np.ndarray:
    """Compute 1D Laplacian with periodic boundary conditions via np.roll."""
    return (
        np.roll(field, 1) + np.roll(field, -1) - 2.0 * field
    ) / (dx * dx)


class Oregonator1DSimulation(SimulationEnvironment):
    """1D Oregonator reaction-diffusion model for BZ traveling waves.

    State is a (2, N) array of u and v concentrations on a 1D periodic grid.
    Traveling pulse solutions emerge when a localized stimulus excites the
    medium. Pulse speed scales as sqrt(D_u / eps) for the sharp-front limit.

    Parameters:
        eps: timescale separation (default 0.1)
        f: stoichiometric factor (default 1.0)
        q: excitability parameter (default 0.002)
        D_u: diffusion coefficient for u (default 1.0)
        D_v: diffusion coefficient for v (default 0.6)
        N: number of grid points (default 200)
        L: domain length (default 100.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.eps = p.get("eps", 0.1)
        self.f = p.get("f", 1.0)
        self.q = p.get("q", 0.002)
        self.D_u = p.get("D_u", 1.0)
        self.D_v = p.get("D_v", 0.6)
        self.N = int(p.get("N", 200))
        self.L = p.get("L", 100.0)
        self.dx = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)

        # CFL stability: dt < dx^2 / (4 * max(D_u, D_v))
        D_max = max(self.D_u, self.D_v)
        self.cfl_limit = self.dx**2 / (4.0 * D_max) if D_max > 0 else np.inf

        self._u = np.zeros(self.N, dtype=np.float64)
        self._v = np.zeros(self.N, dtype=np.float64)
        self._state = np.stack([self._u, self._v], axis=0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize quiescent state with a localized stimulus.

        The medium rests at u ~ q (small), v ~ 0. A narrow Gaussian bump
        in u near the left edge seeds a traveling pulse.
        """
        rng = np.random.default_rng(seed or self.config.seed)

        # Quiescent rest state: u near q, v near 0
        self._u = np.full(self.N, self.q, dtype=np.float64)
        self._v = np.zeros(self.N, dtype=np.float64)

        # Localized stimulus near left edge to trigger a rightward pulse
        stim_center = self.N // 10
        stim_width = max(self.N // 40, 3)
        sigma = stim_width / 2.0
        x_idx = np.arange(self.N)
        gaussian = np.exp(-((x_idx - stim_center) ** 2) / (2 * sigma**2))
        self._u += 0.8 * gaussian

        # Small noise for realism
        self._u += 0.001 * rng.standard_normal(self.N)
        self._v += 0.001 * np.abs(rng.standard_normal(self.N))

        # Clamp to non-negative
        self._u = np.maximum(self._u, 0.0)
        self._v = np.maximum(self._v, 0.0)

        self._step_count = 0
        self._state = np.stack([self._u, self._v], axis=0)
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using explicit Euler with reaction-diffusion.

        Reaction terms:
            R_u = (1/eps) * (u - u^2 - f*v*(u - q)/(u + q))
            R_v = u - v
        """
        dt = self.config.dt
        dx = self.dx

        # Diffusion
        lap_u = _laplacian_1d(self._u, dx)
        lap_v = _laplacian_1d(self._v, dx)

        # Reaction (protect against u + q = 0 division)
        u_safe = np.maximum(self._u, 0.0)
        denom = u_safe + self.q
        # Avoid division by zero
        denom = np.where(denom < 1e-15, 1e-15, denom)

        reaction_u = (
            u_safe - u_safe**2 - self.f * self._v * (u_safe - self.q) / denom
        ) / self.eps
        reaction_v = u_safe - self._v

        # Update
        self._u = self._u + dt * (self.D_u * lap_u + reaction_u)
        self._v = self._v + dt * (self.D_v * lap_v + reaction_v)

        # Clamp to non-negative (concentrations cannot be negative)
        self._u = np.maximum(self._u, 0.0)
        self._v = np.maximum(self._v, 0.0)

        self._step_count += 1
        self._state = np.stack([self._u, self._v], axis=0)
        return self._state

    def observe(self) -> np.ndarray:
        """Return current (u, v) field as shape (2, N)."""
        return self._state

    def count_pulses(self, threshold: float | None = None) -> int:
        """Count the number of pulses (peaks) in the u field.

        A pulse is detected as a local maximum in u that exceeds a threshold.
        We use u rather than v because u is the fast activator variable
        that forms the sharp pulse front.

        Args:
            threshold: Minimum u value to count as a pulse. If None, uses
                half the maximum u value, with a floor of 0.05.

        Returns:
            Number of detected pulses.
        """
        u = self._u
        if threshold is None:
            threshold = max(0.5 * np.max(u), 0.05)

        # Find local maxima with periodic boundary conditions
        left = np.roll(u, 1)
        right = np.roll(u, -1)
        is_peak = (u > left) & (u > right) & (u > threshold)
        return int(np.sum(is_peak))

    def measure_pulse_speed(self, n_steps: int = 200) -> float:
        """Measure pulse propagation speed by tracking the peak of u.

        Runs n_steps forward and measures peak displacement. Only meaningful
        when a single pulse exists.

        Args:
            n_steps: Number of steps to track.

        Returns:
            Estimated pulse speed in spatial units per time unit.
        """
        u_before = self._u.copy()
        peak_before = np.argmax(u_before)

        for _ in range(n_steps):
            self.step()

        u_after = self._u.copy()
        peak_after = np.argmax(u_after)

        # Handle periodic wrap-around
        displacement = (peak_after - peak_before) * self.dx
        if abs(displacement) > self.L / 2:
            if displacement > 0:
                displacement -= self.L
            else:
                displacement += self.L

        total_time = n_steps * self.config.dt
        speed = abs(displacement) / total_time if total_time > 0 else 0.0
        return speed

    @property
    def total_u(self) -> float:
        """Total u concentration (integral of u over domain)."""
        return float(np.sum(self._u) * self.dx)

    @property
    def total_v(self) -> float:
        """Total v concentration (integral of v over domain)."""
        return float(np.sum(self._v) * self.dx)

    @property
    def max_u(self) -> float:
        """Maximum u concentration."""
        return float(np.max(self._u))

    @property
    def max_v(self) -> float:
        """Maximum v concentration."""
        return float(np.max(self._v))

    @property
    def pulse_width(self) -> float:
        """Estimate pulse width as the spatial extent where u > threshold.

        Returns the width of the widest connected region where u exceeds
        half its maximum value.
        """
        u = self._u
        threshold = 0.5 * np.max(u)
        if threshold < 0.05:
            return 0.0

        above = u > threshold
        # Find longest connected run of True values (circular)
        # Duplicate for wrap-around detection
        extended = np.concatenate([above, above])
        max_run = 0
        current_run = 0
        for val in extended:
            if val:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        # Cap at N to avoid double-counting full wrap
        max_run = min(max_run, self.N)
        return float(max_run * self.dx)
