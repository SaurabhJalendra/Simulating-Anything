"""1D spatial FitzHugh-Nagumo PDE simulation (excitable medium waves).

Extends the FitzHugh-Nagumo neuron model to a 1D spatial domain with
diffusion of the voltage variable:

    dv/dt = v - v^3/3 - w + D_v * d^2v/dx^2
    dw/dt = eps * (v + a - b*w)

Only v diffuses (excitable medium convention); w is purely local.
Solver: semi-implicit -- exact spectral (FFT) diffusion, RK4 reaction.

Target rediscoveries:
- Traveling pulse speed c ~ f(D_v, eps)
- Pulse shape and width
- For D_v=0: reduces to local FHN at each grid point
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class FHNSpatial(SimulationEnvironment):
    """1D spatial FitzHugh-Nagumo on a periodic domain.

    State: [v_1..v_N, w_1..w_N] shape (2*N,).

    Parameters:
        a: recovery parameter (default 0.7)
        b: recovery parameter (default 0.8)
        eps: timescale separation (default 0.08)
        D_v: voltage diffusion coefficient (default 1.0)
        L: domain length (default 50.0)
        N: number of grid points (default 128)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Reaction parameters
        self.a = p.get("a", 0.7)
        self.b_param = p.get("b", 0.8)
        self.eps = p.get("eps", 0.08)

        # Diffusion
        self.D_v = p.get("D_v", 1.0)

        # Spatial grid
        self.N = int(p.get("N", 128))
        self.L = p.get("L", 50.0)
        self.dx = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)

        # Wavenumbers for spectral diffusion (periodic domain)
        self.k = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        self.k_sq = self.k ** 2

        # CFL-like check: for the explicit reaction part, dt should be
        # reasonable. The diffusion is handled exactly in spectral space,
        # but if D_v > 0 we still warn for extremely large dt.
        if self.D_v > 0:
            dt_cfl = self.dx ** 2 / (4.0 * self.D_v)
            if config.dt > 10 * dt_cfl:
                raise ValueError(
                    f"dt={config.dt} is much larger than CFL estimate {dt_cfl:.6f} "
                    f"for dx={self.dx:.4f}, D_v={self.D_v}. "
                    f"Reduce dt or increase N."
                )

        # Fields
        self._v_field: np.ndarray | None = None
        self._w_field: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize with a localized pulse perturbation.

        Sets v to the stable rest state everywhere, then applies a narrow
        Gaussian pulse in the left portion of the domain to trigger a
        traveling wave.
        """
        # Rest state: the intersection of nullclines for I=0.
        # v - v^3/3 - w = 0 and v + a - b*w = 0
        # => w = (v + a) / b and v - v^3/3 = (v + a) / b
        # For a=0.7, b=0.8 the rest is approximately v_rest ~ -1.2, w_rest ~ -0.625
        # A quick Newton iteration or we just use the known stable fixed point.
        v_rest = self._find_rest_v()
        w_rest = (v_rest + self.a) / self.b_param

        self._v_field = np.full(self.N, v_rest, dtype=np.float64)
        self._w_field = np.full(self.N, w_rest, dtype=np.float64)

        # Localized pulse: raise v above threshold in a narrow region
        pulse_center = self.L * 0.15
        pulse_width = self.L * 0.04
        pulse = 2.0 * np.exp(-((self.x - pulse_center) ** 2) / (2 * pulse_width ** 2))
        self._v_field = self._v_field + pulse

        self._step_count = 0
        self._state = np.concatenate([self._v_field, self._w_field])
        return self._state

    def _find_rest_v(self) -> float:
        """Find the resting voltage by solving the nullcline intersection.

        Solves v - v^3/3 = (v + a) / b via a few Newton iterations.
        """
        # f(v) = v - v^3/3 - (v + a)/b = 0
        v = -1.0  # initial guess near the stable branch
        for _ in range(50):
            f = v - v ** 3 / 3 - (v + self.a) / self.b_param
            fp = 1 - v ** 2 - 1.0 / self.b_param
            if abs(fp) < 1e-15:
                break
            v_new = v - f / fp
            if abs(v_new - v) < 1e-12:
                break
            v = v_new
        return v

    def step(self) -> np.ndarray:
        """Advance one timestep: exact spectral diffusion + RK4 reaction."""
        dt = self.config.dt
        v = self._v_field
        w = self._w_field

        # --- Diffusion step (exact in Fourier space) ---
        if self.D_v > 0:
            v_hat = np.fft.fft(v)
            v_hat *= np.exp(-self.D_v * self.k_sq * dt)
            v = np.real(np.fft.ifft(v_hat))

        # --- Reaction step (RK4 on the local dynamics) ---
        v, w = self._rk4_reaction(v, w, dt)

        self._v_field = v
        self._w_field = w
        self._step_count += 1
        self._state = np.concatenate([self._v_field, self._w_field])
        return self._state

    def _rk4_reaction(
        self, v: np.ndarray, w: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """RK4 integration of the local reaction terms only."""
        def derivs(v_: np.ndarray, w_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            dv = v_ - v_ ** 3 / 3 - w_
            dw = self.eps * (v_ + self.a - self.b_param * w_)
            return dv, dw

        kv1, kw1 = derivs(v, w)
        kv2, kw2 = derivs(v + 0.5 * dt * kv1, w + 0.5 * dt * kw1)
        kv3, kw3 = derivs(v + 0.5 * dt * kv2, w + 0.5 * dt * kw2)
        kv4, kw4 = derivs(v + dt * kv3, w + dt * kw3)

        v_new = v + (dt / 6.0) * (kv1 + 2 * kv2 + 2 * kv3 + kv4)
        w_new = w + (dt / 6.0) * (kw1 + 2 * kw2 + 2 * kw3 + kw4)
        return v_new, w_new

    def observe(self) -> np.ndarray:
        """Return current state [v_1..v_N, w_1..w_N]."""
        return self._state

    # ----- Properties -----

    @property
    def v_field(self) -> np.ndarray:
        """Voltage field v(x)."""
        return self._v_field.copy()

    @property
    def w_field(self) -> np.ndarray:
        """Recovery field w(x)."""
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
    def pulse_count(self) -> int:
        """Count the number of distinct pulses (connected regions above threshold).

        A pulse is defined as a contiguous region where v exceeds the midpoint
        between the rest state and the peak.
        """
        v_rest = self._find_rest_v()
        v_max = np.max(self._v_field)
        threshold = (v_rest + v_max) / 2.0

        above = self._v_field > threshold
        if not np.any(above):
            return 0

        # Count transitions from below to above (handling periodic wrapping)
        transitions = np.diff(above.astype(int))
        n_rising = np.sum(transitions == 1)
        # Account for wrap-around: if first element is above and last is not
        if above[0] and not above[-1]:
            n_rising += 1
        # If entire domain is above, that is one pulse
        if np.all(above):
            return 1
        return max(int(n_rising), 1 if np.any(above) else 0)

    @property
    def wave_speed(self) -> float:
        """Estimate wave speed from the position of the v peak.

        Tracks the peak position over the last two steps. Returns 0.0
        if no clear pulse is present or if called before stepping.
        """
        if not hasattr(self, "_prev_peak_pos") or self._prev_peak_pos is None:
            # Store current peak position for next call
            self._prev_peak_pos = float(np.argmax(self._v_field)) * self.dx
            self._prev_peak_time = self._step_count * self.config.dt
            return 0.0

        current_pos = float(np.argmax(self._v_field)) * self.dx
        current_time = self._step_count * self.config.dt
        dt_elapsed = current_time - self._prev_peak_time

        if dt_elapsed <= 0:
            return 0.0

        # Handle periodic wrapping
        delta_x = current_pos - self._prev_peak_pos
        if delta_x > self.L / 2:
            delta_x -= self.L
        elif delta_x < -self.L / 2:
            delta_x += self.L

        speed = abs(delta_x) / dt_elapsed

        self._prev_peak_pos = current_pos
        self._prev_peak_time = current_time

        return speed

    def measure_wave_speed(self, n_steps: int = 500) -> float:
        """Measure wave speed by tracking peak position over many steps.

        More robust than the single-step wave_speed property: records peak
        positions at two well-separated times and computes the average speed.
        """
        # Record positions at 1/3 and 2/3 of the run
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
        delta_x = pos2 - pos1

        # Handle periodic wrapping
        if delta_x > self.L / 2:
            delta_x -= self.L
        elif delta_x < -self.L / 2:
            delta_x += self.L

        if dt_elapsed <= 0:
            return 0.0

        return abs(delta_x) / dt_elapsed
