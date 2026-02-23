"""Abstract base class for exploration strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from simulating_anything.types.discovery import Discovery
from simulating_anything.types.trajectory import TrajectoryData


class Explorer(ABC):
    """Base class for parameter space exploration strategies.

    An Explorer proposes simulation parameters to evaluate,
    collects trajectories, and identifies interesting findings.
    """

    @abstractmethod
    def propose_parameters(self) -> dict[str, float]:
        """Propose next set of simulation parameters to evaluate."""

    @abstractmethod
    def update(self, trajectory: TrajectoryData) -> None:
        """Incorporate a new trajectory into the exploration state."""

    @abstractmethod
    def get_interesting_trajectories(self) -> list[TrajectoryData]:
        """Return trajectories deemed interesting for further analysis."""

    @abstractmethod
    def get_exploration_progress(self) -> dict[str, Any]:
        """Return metrics about exploration coverage and progress."""
