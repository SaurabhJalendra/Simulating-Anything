"""1D FitzHugh-Nagumo traveling pulse/wave propagation simulation.

Solves the 1D FHN reaction-diffusion PDE on a periodic domain:

    dv/dt = D_v * v_xx + v - v^3/3 - w + I_stim
    dw/dt = D_w * w_xx + eps * (v + a - b*w)

where v_xx denotes the 1D Laplacian (second spatial derivative).

Only v typically diffuses in the classic excitable-medium setting (D_w=0),
but a small D_w is supported. The model admits traveling pulse solutions
that propagate at speed c ~ sqrt(D_v).

Solver: explicit finite-difference Laplacian + RK4 reaction, periodic BCs.
CFL constraint: dt < dx^2 / (2 * D_max).

Target rediscoveries:
- Pulse speed c ~ sqrt(D_v)
- Pulse speed decreases with increasing eps
- Pulse width characterization
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


def _laplacian_1d_periodic(field: np.ndarray, dx: float) -> np.ndarray:
    """Compute 1D Laplacian with periodic boundary conditions via finite differences."""
    return (np.roll(field, 1) + np.roll(field, -1) - 2.0 * field) / (dx * dx)


class FHNPulseSimulation(SimulationEnvironment):
    """1D FitzHugh-Nagumo traveling pulse on a periodic domain.

    State vector: [v_1..v_N, w_1..w_N] shape (2*N,).

    Parameters:
        D_v: activator diffusion coefficient (default 1.0)
        D_w: inhibitor diffusion coefficient (default 0.0)
        a: recovery parameter (default 0.7)
        b: recovery parameter (default 0.8)
        eps: timescale separation (default 0.08)
        I_stim: external stimulus current (default 0.0)
        L: domain length (default 100.0)
        N: number of grid points (default 256)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Reaction parameters
        self.a = p.get("a", 0.7)
        self.b_param = p.get("b", 0.8)
        self.eps = p.get("eps", 0.08)
        self.I_stim = p.get("I_stim", 0.0)

        # Diffusion coefficients
        self.D_v = p.get("D_v", 1.0)
        self.D_w = p.get("D_w", 0.0)

        # Spatial grid
        self.N = int(p.get("N", 256))
        self.L = p.get("L", 100.0)
        self.dx = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)

        # CFL stability check: dt < dx^2 / (2 * D_max) for explicit diffusion
        D_max = max(self.D_v, self.D_w)
        if D_max > 0:
            self.cfl_limit = self.dx ** 2 / (2.0 * D_max)
            if config.dt > self.cfl_limit:
                raise ValueError(
                    f"dt={config.dt} exceeds CFL limit {self.cfl_limit:.6f} "
                    f"for dx={self.dx:.4f}, D_max={D_max}. "
                    f"Reduce dt or increase N."
                )
        else:
            self.cfl_limit = np.inf

        # Fields
        self._v_field: np.ndarray = np.zeros(self.N, dtype=np.float64)
        self._w_field: np.ndarray = np.zeros(self.N, dtype=np.float64)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize with a localized Gaussian pulse in v at the left end.

        The rest state is computed from the nullcline intersection. A Gaussian
        bump is applied near the left boundary to trigger a rightward-traveling
        pulse.
        """
        v_rest = self._find_rest_v()
        w_rest = (v_rest + self.a) / self.b_param

        self._v_field = np.full(self.N, v_rest, dtype=np.float64)
        self._w_field = np.full(self.N, w_rest, dtype=np.float64)

        # Localized Gaussian stimulus near the left end
        pulse_center = self.L * 0.1
        pulse_width = self.L * 0.03
        pulse = 2.0 * np.exp(
            -((self.x - pulse_center) ** 2) / (2.0 * pulse_width ** 2)
        )
        self._v_field = self._v_field + pulse

        self._step_count = 0
        self._state = np.concatenate([self._v_field, self._w_field])
        return self._state

    def _find_rest_v(self) -> float:
        """Find the resting voltage by solving the nullcline intersection.

        Solves v - v^3/3 - (v + a)/b + I_stim = 0 via Newton's method.
        """
        v = -1.0  # Initial guess near stable branch
        for _ in range(50):
            f = v - v ** 3 / 3.0 - (v + self.a) / self.b_param + self.I_stim
            fp = 1.0 - v ** 2 - 1.0 / self.b_param
            if abs(fp) < 1e-15:
                break
            v_new = v - f / fp
            if abs(v_new - v) < 1e-12:
                break
            v = v_new
        return v

    def step(self) -> np.ndarray:
        """Advance one timestep: explicit FD diffusion + RK4 reaction."""
        dt = self.config.dt
        v = self._v_field.copy()
        w = self._w_field.copy()

        # Compute diffusion terms (explicit finite differences)
        lap_v = _laplacian_1d_periodic(v, self.dx) if self.D_v > 0 else np.zeros(self.N)
        lap_w = _laplacian_1d_periodic(w, self.dx) if self.D_w > 0 else np.zeros(self.N)

        # RK4 integration of reaction + diffusion
        v_new, w_new = self._rk4_step(v, w, lap_v, lap_w, dt)

        self._v_field = v_new
        self._w_field = w_new
        self._step_count += 1
        self._state = np.concatenate([self._v_field, self._w_field])
        return self._state

    def _rk4_step(
        self,
        v: np.ndarray,
        w: np.ndarray,
        lap_v: np.ndarray,
        lap_w: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """RK4 integration of the full PDE right-hand side.

        The Laplacian is computed once at the beginning of the step (operator
        splitting), which is first-order in time for diffusion but adequate
        when dt is small relative to the CFL limit.
        """
        def derivs(
            vi: np.ndarray, wi: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            dv = self.D_v * lap_v + vi - vi ** 3 / 3.0 - wi + self.I_stim
            dw = self.D_w * lap_w + self.eps * (vi + self.a - self.b_param * wi)
            return dv, dw

        kv1, kw1 = derivs(v, w)
        kv2, kw2 = derivs(v + 0.5 * dt * kv1, w + 0.5 * dt * kw1)
        kv3, kw3 = derivs(v + 0.5 * dt * kv2, w + 0.5 * dt * kw2)
        kv4, kw4 = derivs(v + dt * kv3, w + dt * kw3)

        v_new = v + (dt / 6.0) * (kv1 + 2.0 * kv2 + 2.0 * kv3 + kv4)
        w_new = w + (dt / 6.0) * (kw1 + 2.0 * kw2 + 2.0 * kw3 + kw4)
        return v_new, w_new

    def observe(self) -> np.ndarray:
        """Return current state [v_1..v_N, w_1..w_N]."""
        return self._state

    # ----- Properties -----

    @property
    def v_field(self) -> np.ndarray:
        """Voltage (activator) field v(x)."""
        return self._v_field.copy()

    @property
    def w_field(self) -> np.ndarray:
        """Recovery (inhibitor) field w(x)."""
        return self._w_field.copy()

    @property
    def mean_v(self) -> float:
        """Spatial mean of v."""
        return float(np.mean(self._v_field))

    @property
    def max_v(self) -> float:
        """Maximum v across the domain."""
        return float(np.max(self._v_field))

    @property
    def v_max_position(self) -> float:
        """Spatial position of the v maximum."""
        return float(np.argmax(self._v_field)) * self.dx

    @property
    def pulse_width(self) -> float:
        """Full-width at half-maximum (FWHM) of the v pulse above rest.

        Measures the spatial width of the region where v exceeds the midpoint
        between rest and peak. Returns 0.0 if no clear pulse.
        """
        v_rest = self._find_rest_v()
        v_max = np.max(self._v_field)
        if v_max - v_rest < 0.1:
            return 0.0
        half_max = (v_rest + v_max) / 2.0
        above = self._v_field > half_max
        return float(np.sum(above)) * self.dx

    @property
    def pulse_count(self) -> int:
        """Count distinct pulses (connected regions above threshold)."""
        v_rest = self._find_rest_v()
        v_max = np.max(self._v_field)
        if v_max - v_rest < 0.1:
            return 0

        threshold = (v_rest + v_max) / 2.0
        above = self._v_field > threshold
        if not np.any(above):
            return 0
        if np.all(above):
            return 1

        # Count rising transitions (below -> above)
        transitions = np.diff(above.astype(int))
        n_rising = int(np.sum(transitions == 1))
        # Handle wrap-around
        if above[0] and not above[-1]:
            n_rising += 1
        return max(n_rising, 1 if np.any(above) else 0)

    def measure_wave_speed(self, n_steps: int = 500) -> float:
        """Measure traveling pulse speed by tracking peak position.

        Records peak position at 1/3 and 2/3 of the run, then computes
        the average speed. Handles periodic wrap-around.

        Args:
            n_steps: Number of steps to run for measurement.

        Returns:
            Estimated wave speed in spatial units per time unit.
        """
        t1_step = n_steps // 3
        t2_step = 2 * n_steps // 3
        pos1 = None
        pos2 = None

        for step_i in range(1, n_steps + 1):
            self.step()
            if step_i == t1_step:
                pos1 = float(np.argmax(self._v_field)) * self.dx
            elif step_i == t2_step:
                pos2 = float(np.argmax(self._v_field)) * self.dx

        if pos1 is None or pos2 is None:
            return 0.0

        dt_elapsed = (t2_step - t1_step) * self.config.dt
        if dt_elapsed <= 0:
            return 0.0

        delta_x = pos2 - pos1
        # Handle periodic wrapping
        if delta_x > self.L / 2:
            delta_x -= self.L
        elif delta_x < -self.L / 2:
            delta_x += self.L

        return abs(delta_x) / dt_elapsed

    @property
    def total_excitation(self) -> float:
        """Integral of v over the domain (spatial sum * dx)."""
        return float(np.sum(self._v_field) * self.dx)
