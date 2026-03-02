"""Rulkov map simulation -- 2D discrete spiking neuron model.

The Rulkov map is a phenomenological neuron model that reproduces spiking
and bursting activity using a fast-slow 2D map:

    x_{n+1} = alpha / (1 + x_n^2) + y_n      (fast variable: membrane voltage)
    y_{n+1} = y_n - mu * (x_n - sigma)        (slow variable: recovery current)

The fast variable x exhibits spikes when alpha is large enough, while y
provides slow modulation that can produce bursting patterns.

Target rediscoveries:
- Regime classification: silent, spiking, bursting as function of alpha
- Spiking onset threshold: alpha_c ~ 4.0 (transition from silence to activity)
- Firing rate as function of alpha
- Spike detection via threshold crossing
- Bifurcation diagram in alpha
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class RulkovMapSimulation(SimulationEnvironment):
    """Rulkov map: 2D discrete spiking neuron.

    x_{n+1} = alpha / (1 + x_n^2) + y_n
    y_{n+1} = y_n - mu * (x_n - sigma)

    State vector: [x, y] (shape (2,)).

    Parameters:
        alpha: controls spiking behavior (default 4.1)
        mu: slow timescale separation (default 0.001)
        sigma: bias current (default -1.0)
        x_0: initial fast variable (default -1.0)
        y_0: initial slow variable (default -3.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.alpha = p.get("alpha", 4.1)
        self.mu = p.get("mu", 0.001)
        self.sigma = p.get("sigma", -1.0)
        self.x_0 = p.get("x_0", -1.0)
        self.y_0 = p.get("y_0", -3.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state to [x_0, y_0]."""
        self._state = np.array([self.x_0, self.y_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Apply one iteration of the Rulkov map."""
        x, y = self._state
        x_new = self.alpha / (1.0 + x**2) + y
        y_new = y - self.mu * (x - self.sigma)
        self._state = np.array([x_new, y_new], dtype=np.float64)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, y]."""
        return self._state

    def detect_spikes(
        self,
        n_steps: int = 5000,
        n_transient: int = 1000,
        threshold: float = 0.0,
    ) -> list[int]:
        """Detect spike times via upward threshold crossings on x.

        A spike is counted when x crosses the threshold from below.

        Args:
            n_steps: number of iterations to record after transient.
            n_transient: transient iterations to discard.
            threshold: x value that defines a spike crossing.

        Returns:
            List of iteration indices (relative to recording start) where
            spikes occur.
        """
        x, y = self.x_0, self.y_0

        # Transient
        for _ in range(n_transient):
            x_new = self.alpha / (1.0 + x**2) + y
            y_new = y - self.mu * (x - self.sigma)
            x, y = x_new, y_new

        # Record and detect threshold crossings
        spike_times: list[int] = []
        x_prev = x
        for i in range(n_steps):
            x_new = self.alpha / (1.0 + x**2) + y
            y_new = y - self.mu * (x - self.sigma)
            x, y = x_new, y_new
            if x_prev < threshold <= x:
                spike_times.append(i)
            x_prev = x

        return spike_times

    def compute_firing_rate(
        self,
        n_steps: int = 10000,
        n_transient: int = 2000,
        threshold: float = 0.0,
    ) -> float:
        """Compute firing rate as spikes per iteration.

        Args:
            n_steps: number of iterations to record after transient.
            n_transient: transient iterations to discard.
            threshold: x value that defines a spike crossing.

        Returns:
            Firing rate (spikes per iteration).
        """
        spikes = self.detect_spikes(n_steps, n_transient, threshold)
        if n_steps == 0:
            return 0.0
        return len(spikes) / n_steps

    def classify_regime(
        self,
        n_steps: int = 10000,
        n_transient: int = 2000,
        spike_threshold: float = 0.0,
        burst_gap: int = 50,
    ) -> str:
        """Classify the dynamical regime as 'silent', 'spiking', or 'bursting'.

        Silent: no spikes detected.
        Spiking: spikes occur with roughly uniform inter-spike intervals.
        Bursting: spikes occur in clusters separated by quiescent periods.

        Args:
            n_steps: iterations to analyze after transient.
            n_transient: transient iterations to discard.
            spike_threshold: x value that defines a spike crossing.
            burst_gap: minimum inter-spike interval (in iterations) to
                separate bursts from tonic spiking.

        Returns:
            One of 'silent', 'spiking', or 'bursting'.
        """
        spikes = self.detect_spikes(n_steps, n_transient, spike_threshold)

        if len(spikes) < 2:
            return "silent"

        # Compute inter-spike intervals
        isis = np.diff(spikes)

        if len(isis) == 0:
            return "silent"

        # If any ISI exceeds burst_gap, there are quiescent periods -> bursting
        if np.any(isis > burst_gap):
            return "bursting"

        return "spiking"

    def alpha_sweep(
        self,
        alpha_values: np.ndarray,
        n_steps: int = 10000,
        n_transient: int = 2000,
        spike_threshold: float = 0.0,
    ) -> dict[str, np.ndarray | list[str]]:
        """Sweep alpha to characterize spiking onset and regimes.

        Args:
            alpha_values: array of alpha values to test.
            n_steps: iterations per alpha value.
            n_transient: transient per alpha value.
            spike_threshold: threshold for spike detection.

        Returns:
            Dict with 'alpha', 'firing_rate', and 'regime' arrays.
        """
        rates = []
        regimes = []

        for alpha in alpha_values:
            # Temporarily set alpha
            old_alpha = self.alpha
            self.alpha = alpha
            rate = self.compute_firing_rate(n_steps, n_transient, spike_threshold)
            regime = self.classify_regime(n_steps, n_transient, spike_threshold)
            rates.append(rate)
            regimes.append(regime)
            self.alpha = old_alpha

        return {
            "alpha": np.array(alpha_values),
            "firing_rate": np.array(rates),
            "regime": regimes,
        }

    def bifurcation_diagram(
        self,
        alpha_values: np.ndarray,
        n_transient: int = 2000,
        n_plot: int = 200,
    ) -> dict[str, np.ndarray]:
        """Generate bifurcation diagram data (x vs alpha).

        For each alpha value, run the map through a transient, then record
        n_plot steady-state x values.

        Args:
            alpha_values: array of alpha values to sweep.
            n_transient: transient iterations to discard.
            n_plot: number of steady-state iterates to record.

        Returns:
            Dict with 'alpha' and 'x' arrays for plotting.
        """
        all_alpha = []
        all_x = []

        for alpha in alpha_values:
            x, y = self.x_0, self.y_0

            # Transient
            for _ in range(n_transient):
                x_new = alpha / (1.0 + x**2) + y
                y_new = y - self.mu * (x - self.sigma)
                x, y = x_new, y_new

            # Collect steady-state x values
            for _ in range(n_plot):
                x_new = alpha / (1.0 + x**2) + y
                y_new = y - self.mu * (x - self.sigma)
                x, y = x_new, y_new
                all_alpha.append(alpha)
                all_x.append(x)

        return {
            "alpha": np.array(all_alpha),
            "x": np.array(all_x),
        }

    def compute_lyapunov(
        self,
        n_iterations: int = 10000,
        n_transient: int = 1000,
    ) -> float:
        """Estimate the largest Lyapunov exponent via QR method.

        The Jacobian of the Rulkov map at (x, y) is:
            J = [[-2*alpha*x / (1 + x^2)^2,  1],
                 [-mu,                         1]]

        Returns:
            Largest Lyapunov exponent (natural log per iteration).
        """
        x, y = self.x_0, self.y_0

        # Transient to reach the attractor
        for _ in range(n_transient):
            x_new = self.alpha / (1.0 + x**2) + y
            y_new = y - self.mu * (x - self.sigma)
            x, y = x_new, y_new

        # QR-based Lyapunov computation
        q = np.eye(2)
        lyap_sum = np.zeros(2)

        for _ in range(n_iterations):
            denom = (1.0 + x**2) ** 2
            jac = np.array([
                [-2.0 * self.alpha * x / denom, 1.0],
                [-self.mu, 1.0],
            ])

            m = jac @ q
            q, r = np.linalg.qr(m)
            lyap_sum += np.log(np.abs(np.diag(r)))

            x_new = self.alpha / (1.0 + x**2) + y
            y_new = y - self.mu * (x - self.sigma)
            x, y = x_new, y_new

        return float(lyap_sum[0] / n_iterations)
