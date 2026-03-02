"""Stochastic FitzHugh-Nagumo neuron model simulation.

Excitable FHN neuron driven by additive Gaussian white noise, exhibiting
noise-induced spiking (coherence resonance). Integrated via Euler-Maruyama.

SDE system:
    dv = (v - v^3/3 - w + I) dt + sigma * dW_v
    dw = eps * (v + a - b*w) dt

Target rediscoveries:
- Coherence resonance: CV of ISI has minimum at optimal noise sigma
- Noise-induced spiking from subthreshold regime
- ISI statistics (mean, CV) as function of noise intensity
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class FHNStochasticSimulation(SimulationEnvironment):
    """Stochastic FitzHugh-Nagumo neuron with additive noise.

    dv/dt = v - v^3/3 - w + I + sigma * xi(t)
    dw/dt = eps * (v + a - b*w)

    where xi(t) is Gaussian white noise. Integrated with Euler-Maruyama.

    Parameters:
        a: recovery parameter (default 0.7)
        b: recovery parameter (default 0.8)
        eps: timescale separation (default 0.08)
        I: external current (default 0.3, subthreshold)
        sigma: noise intensity (default 0.3)
        v_0: initial voltage (default -1.0)
        w_0: initial recovery (default -0.5)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.7)
        self.b_param = p.get("b", 0.8)
        self.eps = p.get("eps", 0.08)
        self.I = p.get("I", 0.3)
        self.sigma = p.get("sigma", 0.3)
        self.v_0 = p.get("v_0", -1.0)
        self.w_0 = p.get("w_0", -0.5)
        self._rng: np.random.Generator = np.random.default_rng(config.seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize voltage and recovery variables."""
        self._rng = np.random.default_rng(seed)
        self._state = np.array([self.v_0, self.w_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using Euler-Maruyama for the SDE."""
        self._euler_maruyama_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [v, w]."""
        return self._state

    def _euler_maruyama_step(self) -> None:
        """Single Euler-Maruyama step for the stochastic FHN system."""
        dt = self.config.dt
        v, w = self._state

        # Deterministic drift
        dv_det = v - v**3 / 3 - w + self.I
        dw_det = self.eps * (v + self.a - self.b_param * w)

        # Stochastic diffusion (noise only on voltage equation)
        dW = self._rng.normal(0.0, np.sqrt(dt))

        # Euler-Maruyama update
        v_new = v + dv_det * dt + self.sigma * dW
        w_new = w + dw_det * dt

        self._state = np.array([v_new, w_new], dtype=np.float64)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Deterministic part of the FHN dynamics (no noise)."""
        v, w = y
        dv = v - v**3 / 3 - w + self.I
        dw = self.eps * (v + self.a - self.b_param * w)
        return np.array([dv, dw])

    def count_spikes(
        self,
        v_trace: np.ndarray | None = None,
        threshold: float = 0.0,
    ) -> int:
        """Count upward threshold crossings in a voltage trace.

        Args:
            v_trace: 1D array of voltage values. If None, runs a default trajectory.
            threshold: voltage threshold for spike detection.

        Returns:
            Number of spikes (positive threshold crossings).
        """
        if v_trace is None:
            traj = self.run()
            v_trace = traj.states[:, 0]

        crossings = 0
        for i in range(1, len(v_trace)):
            if v_trace[i - 1] < threshold and v_trace[i] >= threshold:
                crossings += 1
        return crossings

    def interspike_intervals(
        self,
        v_trace: np.ndarray | None = None,
        threshold: float = 0.0,
    ) -> np.ndarray:
        """Compute interspike intervals from a voltage trace.

        Args:
            v_trace: 1D array of voltage values.
            threshold: voltage threshold for spike detection.

        Returns:
            Array of interspike intervals (in timestep units).
        """
        if v_trace is None:
            traj = self.run()
            v_trace = traj.states[:, 0]

        spike_indices = []
        for i in range(1, len(v_trace)):
            if v_trace[i - 1] < threshold and v_trace[i] >= threshold:
                spike_indices.append(i)

        if len(spike_indices) < 2:
            return np.array([])

        return np.diff(spike_indices).astype(np.float64) * self.config.dt

    def mean_interspike_interval(
        self,
        v_trace: np.ndarray | None = None,
        threshold: float = 0.0,
    ) -> float:
        """Compute mean interspike interval.

        Returns:
            Mean ISI in time units, or 0.0 if fewer than 2 spikes.
        """
        isis = self.interspike_intervals(v_trace, threshold)
        if len(isis) == 0:
            return 0.0
        return float(np.mean(isis))

    def coefficient_of_variation(
        self,
        v_trace: np.ndarray | None = None,
        threshold: float = 0.0,
    ) -> float:
        """Compute coefficient of variation (CV) of interspike intervals.

        CV = std(ISI) / mean(ISI). A low CV indicates regular spiking
        (coherence resonance). CV ~ 1 for Poisson-like random spiking.

        Returns:
            CV of ISI, or float('inf') if fewer than 2 spikes.
        """
        isis = self.interspike_intervals(v_trace, threshold)
        if len(isis) < 2:
            return float("inf")
        mean_isi = np.mean(isis)
        if mean_isi == 0:
            return float("inf")
        return float(np.std(isis) / mean_isi)

    @property
    def nullcline_v(self):
        """V-nullcline: w = v - v^3/3 + I."""
        return lambda v: v - v**3 / 3 + self.I

    @property
    def nullcline_w(self):
        """W-nullcline: w = (v + a) / b."""
        return lambda v: (v + self.a) / self.b_param
