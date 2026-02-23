"""Abstract base class for simulation environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from simulating_anything.types.simulation import SimulationConfig
from simulating_anything.types.trajectory import TrajectoryData, TrajectoryMetadata


class SimulationEnvironment(ABC):
    """Base class for all simulation backends.

    Subclasses implement domain-specific physics via reset/step/observe.
    The base class provides trajectory collection and parameter management.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self._step_count = 0
        self._state: Any = None
        self._trajectory_states: list[np.ndarray] = []
        self._trajectory_timestamps: list[float] = []

    @abstractmethod
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset simulation to initial conditions.

        Returns the initial state as a numpy array.
        """

    @abstractmethod
    def step(self) -> np.ndarray:
        """Advance simulation by one timestep.

        Returns the new state as a numpy array.
        """

    @abstractmethod
    def observe(self) -> np.ndarray:
        """Return the current observable state as a numpy array."""

    def run(self, n_steps: int | None = None) -> TrajectoryData:
        """Run simulation for n_steps and collect a trajectory."""
        if n_steps is None:
            n_steps = self.config.n_steps

        state = self.reset(seed=self.config.seed)
        self._trajectory_states = [state.copy()]
        self._trajectory_timestamps = [0.0]

        for i in range(1, n_steps + 1):
            state = self.step()
            self._trajectory_states.append(state.copy())
            self._trajectory_timestamps.append(i * self.config.dt)

        return self.get_trajectory()

    def get_trajectory(self) -> TrajectoryData:
        """Package collected states into a TrajectoryData object."""
        traj = TrajectoryData(
            parameters={k: float(v) for k, v in self.config.parameters.items()},
            metadata=TrajectoryMetadata(),
        )
        traj.states = np.array(self._trajectory_states)
        traj.timestamps = np.array(self._trajectory_timestamps)
        return traj
