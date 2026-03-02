"""FitzHugh-Nagumo on a 2D square lattice (excitable medium).

Discrete-space FHN with nearest-neighbor diffusive coupling on an NxN grid
with periodic boundary conditions:

    dv_{ij}/dt = v_{ij} - v_{ij}^3/3 - w_{ij} + I + D * Lap(v)_{ij}
    dw_{ij}/dt = eps * (v_{ij} + a - b * w_{ij})

where Lap(v) is the 5-point discrete Laplacian:
    Lap(v)_{ij} = v_{i+1,j} + v_{i-1,j} + v_{i,j+1} + v_{i,j-1} - 4*v_{ij}

Only v diffuses; w is purely local (standard excitable medium convention).
Solver: forward Euler with small dt for stability.

Target rediscoveries:
- Spiral wave formation from localized perturbations
- Synchronization transition at critical D
- Pattern classification: uniform, spiral, turbulent
- Traveling wave speed measurement
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class FHNLattice(SimulationEnvironment):
    """FitzHugh-Nagumo on a 2D square lattice with periodic BCs.

    State: flattened array [v_{0,0}..v_{N-1,N-1}, w_{0,0}..w_{N-1,N-1}]
    of shape (2*N*N,).

    Parameters:
        a: recovery shape parameter (default 0.7)
        b: recovery rate parameter (default 0.8)
        eps: timescale separation (default 0.08)
        I: external current (default 0.5)
        D: diffusion coefficient for v (default 1.0)
        N: lattice side length (default 32)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Reaction parameters
        self.a = p.get("a", 0.7)
        self.b_param = p.get("b", 0.8)
        self.eps = p.get("eps", 0.08)
        self.I_ext = p.get("I", 0.5)

        # Diffusion and lattice
        self.D = p.get("D", 1.0)
        self.N = int(p.get("N", 32))

        # CFL stability: for forward Euler with 5-point stencil on unit-spacing
        # lattice, stability requires dt < 1/(4*D). Warn if violated.
        if self.D > 0:
            dt_cfl = 1.0 / (4.0 * self.D)
            if config.dt > dt_cfl:
                raise ValueError(
                    f"dt={config.dt} exceeds CFL limit {dt_cfl:.6f} "
                    f"for D={self.D} on unit-spacing lattice. "
                    f"Reduce dt or decrease D."
                )

        # Internal 2D fields
        self._v: np.ndarray | None = None
        self._w: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize fields near the resting state with a localized perturbation.

        Sets v and w to the uniform rest state, then applies a Gaussian
        pulse in the lower-left quadrant of the lattice to trigger wave
        propagation.
        """
        rng = np.random.default_rng(seed if seed is not None else self.config.seed)

        v_rest = self._find_rest_v()
        w_rest = (v_rest + self.a) / self.b_param

        self._v = np.full((self.N, self.N), v_rest, dtype=np.float64)
        self._w = np.full((self.N, self.N), w_rest, dtype=np.float64)

        # Add small random noise to break symmetry
        self._v += 0.01 * rng.standard_normal((self.N, self.N))

        # Localized pulse in lower-left quadrant to trigger a wave
        cx, cy = self.N // 4, self.N // 4
        for i in range(self.N):
            for j in range(self.N):
                dx = min(abs(i - cx), self.N - abs(i - cx))
                dy = min(abs(j - cy), self.N - abs(j - cy))
                r2 = dx * dx + dy * dy
                self._v[i, j] += 2.0 * np.exp(-r2 / (2.0 * (self.N * 0.05) ** 2))

        self._step_count = 0
        self._state = self._pack_state()
        return self._state

    def _find_rest_v(self) -> float:
        """Find the resting voltage by solving the nullcline intersection.

        Solves v - v^3/3 - w + I = 0 and w = (v + a) / b simultaneously:
            v - v^3/3 - (v + a)/b + I = 0
        via Newton iterations.
        """
        v = -1.0  # initial guess near the stable branch
        for _ in range(50):
            f = v - v ** 3 / 3 - (v + self.a) / self.b_param + self.I_ext
            fp = 1 - v ** 2 - 1.0 / self.b_param
            if abs(fp) < 1e-15:
                break
            v_new = v - f / fp
            if abs(v_new - v) < 1e-12:
                break
            v = v_new
        return v

    def _pack_state(self) -> np.ndarray:
        """Flatten 2D v and w fields into a 1D state vector."""
        return np.concatenate([self._v.ravel(), self._w.ravel()])

    def _laplacian_periodic(self, field: np.ndarray) -> np.ndarray:
        """Compute the discrete 5-point Laplacian with periodic BCs.

        Lap(v)_{ij} = v_{i+1,j} + v_{i-1,j} + v_{i,j+1} + v_{i,j-1} - 4*v_{ij}
        """
        return (
            np.roll(field, 1, axis=0)
            + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1)
            + np.roll(field, -1, axis=1)
            - 4.0 * field
        )

    def step(self) -> np.ndarray:
        """Advance one timestep using forward Euler."""
        dt = self.config.dt
        v = self._v
        w = self._w

        # Discrete Laplacian (5-point stencil, unit spacing)
        lap_v = self._laplacian_periodic(v)

        # FHN reaction terms
        dv = v - v ** 3 / 3 - w + self.I_ext + self.D * lap_v
        dw = self.eps * (v + self.a - self.b_param * w)

        self._v = v + dt * dv
        self._w = w + dt * dw

        self._step_count += 1
        self._state = self._pack_state()
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [v_flat, w_flat] of shape (2*N*N,)."""
        return self._state

    # ----- Properties -----

    @property
    def v_field(self) -> np.ndarray:
        """Voltage field v as 2D array (N, N)."""
        return self._v.copy()

    @property
    def w_field(self) -> np.ndarray:
        """Recovery field w as 2D array (N, N)."""
        return self._w.copy()

    @property
    def mean_v(self) -> float:
        """Spatial mean of the voltage field."""
        return float(np.mean(self._v))

    @property
    def std_v(self) -> float:
        """Spatial standard deviation of the voltage field."""
        return float(np.std(self._v))

    @property
    def synchronization_order_parameter(self) -> float:
        """Measure of spatial synchronization: 1 - CV(v).

        Returns a value in [0, 1] where 1 means perfectly synchronized
        (all lattice sites have the same v) and 0 means fully desynchronized.
        """
        mean_v = np.mean(self._v)
        if abs(mean_v) < 1e-15:
            # Avoid division by zero; use range-based measure instead
            v_range = np.max(self._v) - np.min(self._v)
            return 1.0 if v_range < 1e-10 else 0.0
        cv = np.std(self._v) / abs(mean_v)
        return float(max(0.0, 1.0 - cv))

    def classify_pattern(self) -> str:
        """Classify the current spatial pattern.

        Returns one of: "uniform", "wave", "spiral", "turbulent".
        Based on spatial heterogeneity and gradient statistics.
        """
        v = self._v
        std_v = np.std(v)

        if std_v < 0.05:
            return "uniform"

        # Compute gradient magnitudes
        grad_x = np.roll(v, -1, axis=0) - v
        grad_y = np.roll(v, -1, axis=1) - v
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        mean_grad = np.mean(grad_mag)
        std_grad = np.std(grad_mag)

        # Turbulent: highly irregular gradients
        if std_grad > 0.5 * mean_grad and mean_grad > 0.3:
            return "turbulent"

        # Spiral/wave: organized but spatially varying
        if std_v > 0.3:
            return "spiral"

        return "wave"

    def measure_wave_speed(self, n_measure: int = 200) -> float:
        """Measure wave speed by tracking the v peak along a row.

        Records the position of the maximum v along the central row
        at two well-separated times and computes the speed.

        Args:
            n_measure: Number of steps to measure over.

        Returns:
            Estimated wave speed in lattice units per time unit.
        """
        row = self.N // 2

        t1_step = n_measure // 3
        t2_step = 2 * n_measure // 3
        pos1 = None
        pos2 = None

        for step_i in range(1, n_measure + 1):
            self.step()
            if step_i == t1_step:
                pos1 = float(np.argmax(self._v[row, :]))
            elif step_i == t2_step:
                pos2 = float(np.argmax(self._v[row, :]))

        if pos1 is None or pos2 is None:
            return 0.0

        dt_elapsed = (t2_step - t1_step) * self.config.dt
        if dt_elapsed <= 0:
            return 0.0

        # Handle periodic wrapping
        delta = pos2 - pos1
        if delta > self.N / 2:
            delta -= self.N
        elif delta < -self.N / 2:
            delta += self.N

        return abs(delta) / dt_elapsed
