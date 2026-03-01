"""Bak-Sneppen model of self-organized criticality.

Target rediscoveries:
- SOC threshold: f_c ~ 2/3 = 0.667 for the 1D ring model
- Avalanche size distribution follows a power law: P(s) ~ s^{-tau}
  with tau ~ 1.07 (mean-field) or tau ~ 1.4 (1D)
- The fitness gap (minimum fitness) evolves toward f_c from below
- All fitnesses self-organize above f_c in the steady state
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class BakSneppen(SimulationEnvironment):
    """Bak-Sneppen model: N species on a 1D ring with extremal dynamics.

    At each step the species with the lowest fitness is replaced (along with
    its two nearest neighbors) by new uniform random fitness values in [0, 1].
    The system self-organizes to a critical state where most fitnesses exceed
    a threshold f_c ~ 2/3.

    State vector: [f_0, f_1, ..., f_{N-1}] -- fitness values in [0, 1].

    Parameters:
        N: number of species on the ring (default 50)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.N = int(p.get("N", 50))

        self._rng: np.random.Generator = np.random.default_rng(config.seed)

        # Running threshold estimate (exponential moving average of min fitness)
        self._threshold_ema: float = 0.0
        self._ema_alpha: float = 0.01

        # Avalanche tracking: count consecutive mutations where the minimum
        # fitness stays below the current threshold estimate
        self._avalanche_size: int = 0
        self._prev_min: float = 0.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize all fitnesses uniformly on [0, 1]."""
        self._rng = np.random.default_rng(seed)
        self._state = self._rng.uniform(0.0, 1.0, self.N).astype(np.float64)
        self._step_count = 0
        self._threshold_ema = 0.0
        self._avalanche_size = 0
        self._prev_min = 0.0
        return self._state

    def step(self) -> np.ndarray:
        """Perform one Bak-Sneppen update.

        Find the species with the minimum fitness, replace it and its two
        nearest neighbors (on the ring) with new random values from U[0, 1].
        """
        # Find species with minimum fitness
        idx_min = int(np.argmin(self._state))
        current_min = self._state[idx_min]

        # Update threshold EMA
        self._threshold_ema = (
            (1.0 - self._ema_alpha) * self._threshold_ema
            + self._ema_alpha * current_min
        )

        # Track avalanche: consecutive steps where min is below threshold
        if current_min < self._threshold_ema:
            self._avalanche_size += 1
        else:
            self._avalanche_size = 0

        self._prev_min = current_min

        # Replace the minimum-fitness species and its two neighbors
        left = (idx_min - 1) % self.N
        right = (idx_min + 1) % self.N
        self._state[idx_min] = self._rng.uniform(0.0, 1.0)
        self._state[left] = self._rng.uniform(0.0, 1.0)
        self._state[right] = self._rng.uniform(0.0, 1.0)

        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return the current fitness values."""
        return self._state

    @property
    def min_fitness(self) -> float:
        """Current minimum fitness across all species."""
        return float(np.min(self._state))

    @property
    def mean_fitness(self) -> float:
        """Current mean fitness across all species."""
        return float(np.mean(self._state))

    @property
    def fitness_threshold(self) -> float:
        """Running EMA estimate of the SOC threshold."""
        return float(self._threshold_ema)

    @property
    def avalanche_size(self) -> int:
        """Current avalanche size (consecutive sub-threshold mutations)."""
        return self._avalanche_size

    def measure_soc_threshold(
        self,
        n_transient: int = 5000,
        n_measure: int = 5000,
        seed: int | None = None,
        percentile: float = 5.0,
    ) -> float:
        """Measure the SOC threshold after transient.

        In the Bak-Sneppen model, f_c is the value below which fitness values
        are rarely found in the self-organized critical state. We estimate it
        by collecting all fitness values across many snapshots in steady state
        and computing a low percentile of that distribution. The sharp lower
        cutoff of the fitness distribution approximates f_c ~ 2/3.

        Args:
            n_transient: Steps to reach steady state.
            n_measure: Steps over which to collect fitness snapshots.
            seed: Random seed.
            percentile: Percentile of collected fitnesses to use as threshold.
        """
        self.reset(seed=seed)

        for _ in range(n_transient):
            self.step()

        # Collect all fitness values from periodic snapshots
        sample_interval = max(1, self.N // 3)
        all_fitnesses = []
        for i in range(n_measure):
            self.step()
            if i % sample_interval == 0:
                all_fitnesses.append(self._state.copy())

        all_fitnesses = np.concatenate(all_fitnesses)
        return float(np.percentile(all_fitnesses, percentile))

    def measure_avalanche_distribution(
        self,
        threshold: float | None = None,
        n_transient: int = 5000,
        n_avalanches: int = 1000,
        seed: int | None = None,
    ) -> list[int]:
        """Measure avalanche size distribution.

        An avalanche is a sequence of consecutive steps where the minimum
        fitness is below the specified threshold. Returns a list of avalanche
        sizes.

        Args:
            threshold: Fitness threshold for avalanche definition. If None,
                uses the measured SOC threshold from transient.
            n_transient: Number of transient steps before measuring.
            n_avalanches: Target number of avalanches to collect.
            seed: Random seed.
        """
        self.reset(seed=seed)

        for _ in range(n_transient):
            self.step()

        if threshold is None:
            # Estimate threshold from a short measurement window
            min_vals = []
            for _ in range(1000):
                self.step()
                min_vals.append(self.min_fitness)
            threshold = float(np.mean(min_vals))

        avalanche_sizes = []
        current_size = 0
        max_steps = n_avalanches * 100  # Safety limit

        for _ in range(max_steps):
            self.step()
            if self.min_fitness < threshold:
                current_size += 1
            else:
                if current_size > 0:
                    avalanche_sizes.append(current_size)
                    current_size = 0
                    if len(avalanche_sizes) >= n_avalanches:
                        break

        # Capture any trailing avalanche
        if current_size > 0:
            avalanche_sizes.append(current_size)

        return avalanche_sizes

    def measure_gap_evolution(
        self,
        n_steps: int = 10000,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Track how the minimum fitness (gap) evolves over time.

        Returns arrays of step indices, minimum fitnesses, and mean fitnesses.
        """
        self.reset(seed=seed)

        steps = np.arange(n_steps)
        min_fitness = np.empty(n_steps, dtype=np.float64)
        mean_fitness = np.empty(n_steps, dtype=np.float64)

        for i in range(n_steps):
            self.step()
            min_fitness[i] = self.min_fitness
            mean_fitness[i] = self.mean_fitness

        return {
            "steps": steps,
            "min_fitness": min_fitness,
            "mean_fitness": mean_fitness,
        }
