"""FitzHugh-Nagumo ring network simulation.

Ring of N coupled FHN neurons with nearest-neighbor diffusive coupling
and periodic boundary conditions:

    dv_i/dt = v_i - v_i^3/3 - w_i + I + D*(v_{i-1} - 2*v_i + v_{i+1})
    dw_i/dt = eps*(v_i + a - b*w_i)

Target rediscoveries:
- Synchronization transition: order parameter r vs coupling D
- Traveling wave speed: wave propagation around the ring
- Critical coupling for synchronization onset
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class FHNRingSimulation(SimulationEnvironment):
    """Ring of N coupled FitzHugh-Nagumo neurons.

    State: [v_1..v_N, w_1..w_N] (2N-dimensional).

    The coupling is a discrete Laplacian on a ring (periodic BCs) applied
    only to the voltage variable v, implemented via np.roll.

    Parameters:
        a: recovery parameter (default 0.7)
        b: recovery parameter (default 0.8)
        eps: timescale separation (default 0.08)
        I: external current (default 0.5)
        D: nearest-neighbor coupling strength (default 0.5)
        N: number of neurons in the ring (default 20)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.7)
        self.b_param = p.get("b", 0.8)
        self.eps = p.get("eps", 0.08)
        self.I = p.get("I", 0.5)
        self.D = p.get("D", 0.5)
        self.N = int(p.get("N", 20))

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize neurons with heterogeneous perturbation.

        Voltages are set to the approximate rest state with small random
        perturbations seeded by the seed parameter. Recovery variables
        start at the corresponding nullcline value.
        """
        rng = np.random.default_rng(seed if seed is not None else self.config.seed)

        # Rest state for single FHN: v - v^3/3 = (v + a)/b
        v_rest = self._find_rest_v()
        w_rest = (v_rest + self.a) / self.b_param

        v = np.full(self.N, v_rest, dtype=np.float64)
        w = np.full(self.N, w_rest, dtype=np.float64)

        # Small random perturbations to break symmetry
        v += 0.1 * rng.standard_normal(self.N)
        w += 0.01 * rng.standard_normal(self.N)

        self._v = v
        self._w = w
        self._state = np.concatenate([self._v, self._w])
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        self._state = np.concatenate([self._v, self._w])
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [v_1..v_N, w_1..w_N]."""
        return self._state

    def _rk4_step(self) -> None:
        """RK4 integration of the coupled ring dynamics."""
        dt = self.config.dt
        v, w = self._v, self._w

        kv1, kw1 = self._derivatives(v, w)
        kv2, kw2 = self._derivatives(v + 0.5 * dt * kv1, w + 0.5 * dt * kw1)
        kv3, kw3 = self._derivatives(v + 0.5 * dt * kv2, w + 0.5 * dt * kw2)
        kv4, kw4 = self._derivatives(v + dt * kv3, w + dt * kw3)

        self._v = v + (dt / 6.0) * (kv1 + 2 * kv2 + 2 * kv3 + kv4)
        self._w = w + (dt / 6.0) * (kw1 + 2 * kw2 + 2 * kw3 + kw4)

    def _derivatives(
        self, v: np.ndarray, w: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute dv/dt, dw/dt for all neurons.

        The discrete Laplacian on a ring uses np.roll for periodic BCs:
            Lap(v)_i = v_{i-1} - 2*v_i + v_{i+1}
        """
        # Ring Laplacian via np.roll (periodic boundary conditions)
        laplacian = np.roll(v, 1) - 2.0 * v + np.roll(v, -1)

        dv = v - v**3 / 3.0 - w + self.I + self.D * laplacian
        dw = self.eps * (v + self.a - self.b_param * w)
        return dv, dw

    def _find_rest_v(self) -> float:
        """Find the resting voltage of a single uncoupled FHN neuron.

        Solves v - v^3/3 - (v + a)/b + I = 0 via Newton iterations.
        """
        v = -1.0
        for _ in range(50):
            f = v - v**3 / 3.0 - (v + self.a) / self.b_param + self.I
            fp = 1.0 - v**2 - 1.0 / self.b_param
            if abs(fp) < 1e-15:
                break
            v_new = v - f / fp
            if abs(v_new - v) < 1e-12:
                break
            v = v_new
        return v

    # ----- Analysis Methods -----

    def compute_synchronization(self) -> float:
        """Compute mean-field synchronization order parameter.

        Uses a Kuramoto-style order parameter based on the voltage phases.
        Each neuron's phase is estimated from its voltage by mapping v_i
        into a complex exponential using the Hilbert-transform-inspired
        approach: phi_i = 2*pi * (v_i - v_min) / (v_max - v_min).

        Returns:
            r: order parameter in [0, 1], where 1 = fully synchronized.
        """
        v = self._v
        v_min = np.min(v)
        v_max = np.max(v)

        if v_max - v_min < 1e-10:
            # All neurons at the same voltage -- perfectly synchronized
            return 1.0

        # Map voltages to phases on [0, 2*pi]
        phases = 2.0 * np.pi * (v - v_min) / (v_max - v_min)
        z = np.mean(np.exp(1j * phases))
        return float(np.abs(z))

    def compute_voltage_variance(self) -> float:
        """Compute variance of voltages across neurons.

        A small variance indicates synchronization; large variance
        indicates desynchronized or wave-like states.
        """
        return float(np.var(self._v))

    def detect_traveling_wave(self) -> float:
        """Detect traveling wave and estimate wave speed.

        Measures the spatial phase gradient of v to determine if a
        coherent traveling wave is present. Returns the estimated
        wave speed (sites per unit time). Zero if no clear wave.

        The method finds the position of the peak voltage at two
        successive time points separated by n_gap steps and computes
        the angular velocity around the ring.
        """
        # Record current peak position
        peak_idx_before = int(np.argmax(self._v))

        # Advance a few steps
        n_gap = max(10, int(1.0 / self.config.dt))
        for _ in range(n_gap):
            self._rk4_step()
            self._step_count += 1
        self._state = np.concatenate([self._v, self._w])

        peak_idx_after = int(np.argmax(self._v))

        # Compute displacement on the ring (handle periodic wrapping)
        delta = peak_idx_after - peak_idx_before
        if delta > self.N / 2:
            delta -= self.N
        elif delta < -self.N / 2:
            delta += self.N

        elapsed_time = n_gap * self.config.dt
        if elapsed_time <= 0:
            return 0.0

        # Speed in sites per unit time
        return abs(delta) / elapsed_time

    def measure_mean_voltage(self) -> float:
        """Return mean voltage across all neurons."""
        return float(np.mean(self._v))

    def measure_phase_coherence(self) -> float:
        """Measure phase coherence using pairwise voltage differences.

        Returns a value in [0, 1] where 1 means all neurons have identical
        voltages (perfect coherence) and 0 means maximum spread.
        """
        v = self._v
        std = np.std(v)
        v_range = np.max(v) - np.min(v)
        if v_range < 1e-10:
            return 1.0
        # Normalize std by half the typical FHN amplitude (~2)
        coherence = max(0.0, 1.0 - std / 1.0)
        return float(np.clip(coherence, 0.0, 1.0))

    @property
    def v_neurons(self) -> np.ndarray:
        """Voltage array for all N neurons."""
        return self._v.copy()

    @property
    def w_neurons(self) -> np.ndarray:
        """Recovery variable array for all N neurons."""
        return self._w.copy()
